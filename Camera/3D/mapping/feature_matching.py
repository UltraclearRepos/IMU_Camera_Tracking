import cv2
import numpy as np
import torch
from lightglue import DISK, SIFT, LightGlue
from lightglue.utils import rbd

from mapping.aruco_reference import create_aruco_detector

DEVICE = "cuda"
MAX_FEATURES = 512
MASK_ARUCO_FEATURES = True
ARUCO_MASK_MARGIN_PX = 20
FEATURE_TYPES = ("disk", "sift")
SKIN_SEED_WIDTH_FRACTION = 0.2
SKIN_SEED_HEIGHT_FRACTION = 0.70
SKIN_BOUND_STANDARD_DEVIATIONS = 1.0
SKIN_MIN_STANDARD_DEVIATION = 8.0
SKIN_MODEL_UPDATE_RATE = 0.03


def frame_to_tensor(frame, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return image.to(device)


class LightGlueFeatureMatching:
    def __init__(
        self,
        feature_roi_bottom_fraction,
        feature_type="disk",
        mask_aruco_features=MASK_ARUCO_FEATURES,
        use_adaptive_skin_mask=True,
    ):
        feature_type = feature_type.lower()
        if feature_type not in FEATURE_TYPES:
            raise ValueError(
                f"feature_type must be one of {FEATURE_TYPES}, got "
                f"{feature_type!r}"
            )

        self.device = DEVICE
        self.feature_type = feature_type
        self.requires_scale_orientation = feature_type == "sift"
        self.feature_roi_bottom_fraction = feature_roi_bottom_fraction
        self.mask_aruco_features = mask_aruco_features
        self.use_adaptive_skin_mask = use_adaptive_skin_mask
        self.skin_mean_ab = None
        self.skin_std_ab = None
        self.skin_lower_bound = None
        self.skin_upper_bound = None
        self.previous_skin_mask = None
        self.last_skin_mask = None
        self.initial_skin_frame = None
        self.initial_skin_seed = None
        self.initial_skin_valid = None
        self.initial_skin_mask = None
        self.last_aruco_ids = np.empty(0, dtype=np.int32)
        self.last_aruco_count = 0
        if feature_type == "disk":
            self.extractor = DISK(
                max_num_keypoints=MAX_FEATURES
            ).eval().to(self.device)
        else:
            # OpenCV exposes response, scale and orientation for every SIFT
            # point. LightGlue matching still runs on DEVICE.
            self.extractor = SIFT(
                max_num_keypoints=MAX_FEATURES,
                backend="opencv",
            ).eval().to(self.device)
        self.matcher = LightGlue(features=feature_type).eval().to(
            self.device
        )
        self.aruco_detector = create_aruco_detector()

    def extract(self, frame):
        height, width = frame.shape[:2]
        roi_top = round(
            height * (1.0 - self.feature_roi_bottom_fraction)
        )
        skin_mask = self.adaptive_skin_mask(frame, roi_top)
        self.last_skin_mask = skin_mask
        roi = frame[roi_top:].copy()
        if skin_mask is not None:
            roi[~skin_mask[roi_top:]] = 0

        with torch.inference_mode():
            extracted = self.extractor.extract(
                frame_to_tensor(roi, self.device)
            )

        features = rbd(extracted)
        keypoints = features["keypoints"].detach().cpu().numpy().copy()
        descriptors = features["descriptors"].detach().cpu().numpy().copy()
        scores = features["keypoint_scores"].detach().cpu().numpy().copy()
        keypoints[:, 1] += roi_top
        scales = None
        orientations = None
        if self.requires_scale_orientation:
            scales = features["scales"].detach().cpu().numpy().copy()
            orientations = features["oris"].detach().cpu().numpy().copy()

        if self.mask_aruco_features:
            keep = self.points_outside_aruco(frame, keypoints)
            keypoints = keypoints[keep]
            descriptors = descriptors[keep]
            scores = scores[keep]
            if self.requires_scale_orientation:
                scales = scales[keep]
                orientations = orientations[keep]

        if skin_mask is not None:
            pixels = np.rint(keypoints).astype(int)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
            keep = skin_mask[pixels[:, 1], pixels[:, 0]]
            keypoints = keypoints[keep]
            descriptors = descriptors[keep]
            scores = scores[keep]
            if self.requires_scale_orientation:
                scales = scales[keep]
                orientations = orientations[keep]

        result = {
            "keypoints": keypoints.astype(np.float32),
            "descriptors": descriptors.astype(np.float32),
            "scores": scores.astype(np.float32),
            "image_size": np.array([width, height], dtype=np.float32),
            "roi_top": roi_top,
        }
        if skin_mask is not None:
            result["selection_mask"] = skin_mask
        if self.requires_scale_orientation:
            result["scales"] = scales.astype(np.float32)
            result["oris"] = orientations.astype(np.float32)
        return result

    def adaptive_skin_mask(self, frame, roi_top):
        """Estimate a moving skin mask from a central seed in the first frame."""
        if not self.use_adaptive_skin_mask:
            return None

        height, width = frame.shape[:2]
        valid = self.aruco_exclusion_mask(frame).astype(bool)
        valid[:roi_top] = False
        # Lightness is intentionally ignored: skin can be shaded unevenly.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        ab = lab[:, :, 1:3]

        initialized_from_seed = self.skin_mean_ab is None
        if initialized_from_seed:
            seed = np.zeros((height, width), dtype=bool)
            seed_half_width = round(width * SKIN_SEED_WIDTH_FRACTION / 2.0)
            seed_half_height = round(
                (height - roi_top) * SKIN_SEED_HEIGHT_FRACTION / 2.0
            )
            center_x = width // 2
            center_y = roi_top + (height - roi_top) // 2
            seed[
                max(roi_top, center_y - seed_half_height):min(
                    height, center_y + seed_half_height
                ),
                max(0, center_x - seed_half_width):min(
                    width, center_x + seed_half_width
                ),
            ] = True
            seed &= valid
            samples = ab[seed]
            if len(samples) < 100:
                return None
            detected_ids = (
                ", ".join(str(marker_id) for marker_id in self.last_aruco_ids)
                if self.last_aruco_count
                else "none"
            )
            print(
                "Skin-mask initialization: "
                f"ArUco markers detected: {self.last_aruco_count} "
                f"(IDs: {detected_ids})"
            )
            median = np.median(samples, axis=0)
            distances = np.linalg.norm(samples - median, axis=1)
            samples = samples[distances <= np.percentile(distances, 80.0)]
            self._set_skin_model(samples)

        candidate = cv2.inRange(
            ab,
            self.skin_lower_bound,
            self.skin_upper_bound,
        ).astype(bool)
        candidate &= valid
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((9, 9), dtype=np.uint8),
        ).astype(bool)
        selected = self._select_skin_component(candidate, roi_top)
        if selected is None:
            return None

        if initialized_from_seed:
            self.initial_skin_frame = frame.copy()
            self.initial_skin_seed = seed.copy()
            self.initial_skin_valid = valid.copy()
            self.initial_skin_mask = selected.copy()
        self._update_skin_model(ab[selected])
        self.previous_skin_mask = selected
        return selected

    def _select_skin_component(self, candidate, roi_top):
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8),
            connectivity=8,
        )
        if component_count <= 1:
            return None

        reference = self.previous_skin_mask
        if reference is None:
            height, width = candidate.shape
            reference = np.zeros_like(candidate)
            reference[
                roi_top + (height - roi_top) // 4:
                roi_top + 3 * (height - roi_top) // 4,
                width // 3:2 * width // 3,
            ] = True

        best_label = 0
        best_score = -np.inf
        for label in range(1, component_count):
            component = labels == label
            overlap = np.count_nonzero(component & reference)
            area = stats[label, cv2.CC_STAT_AREA]
            score = 1000.0 * overlap + area
            if score > best_score:
                best_label = label
                best_score = score

        if best_label == 0:
            return None
        return labels == best_label

    def _set_skin_model(self, samples):
        self.skin_mean_ab = np.mean(samples, axis=0, dtype=np.float32)
        self.skin_std_ab = np.maximum(
            np.std(samples, axis=0, dtype=np.float32),
            SKIN_MIN_STANDARD_DEVIATION,
        )
        self._update_skin_bounds()

    def _update_skin_model(self, samples):
        if len(samples) < 100:
            return
        samples = samples[::max(1, len(samples) // 10000)]
        mean = np.mean(samples, axis=0, dtype=np.float32)
        standard_deviation = np.maximum(
            np.std(samples, axis=0, dtype=np.float32),
            SKIN_MIN_STANDARD_DEVIATION,
        )
        rate = SKIN_MODEL_UPDATE_RATE
        self.skin_mean_ab = (1.0 - rate) * self.skin_mean_ab + rate * mean
        self.skin_std_ab = (
            (1.0 - rate) * self.skin_std_ab + rate * standard_deviation
        )
        self._update_skin_bounds()

    def _update_skin_bounds(self):
        half_range = SKIN_BOUND_STANDARD_DEVIATIONS * self.skin_std_ab
        self.skin_lower_bound = np.clip(
            self.skin_mean_ab - half_range,
            0,
            255,
        ).astype(np.uint8)
        self.skin_upper_bound = np.clip(
            self.skin_mean_ab + half_range,
            0,
            255,
        ).astype(np.uint8)

    def aruco_exclusion_mask(self, frame):
        mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
        if not self.mask_aruco_features:
            return mask

        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        if not corners:
            enlarged = cv2.resize(
                frame,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            )
            corners, ids, _ = self.aruco_detector.detectMarkers(enlarged)
            corners = [corner / 2.0 for corner in corners]
        self.last_aruco_ids = (
            ids.reshape(-1).astype(np.int32)
            if ids is not None
            else np.empty(0, dtype=np.int32)
        )
        self.last_aruco_count = len(corners)
        for marker_corners in corners:
            polygon = np.rint(marker_corners.reshape(4, 2)).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, 0)

        if ARUCO_MASK_MARGIN_PX > 0:
            size = 2 * ARUCO_MASK_MARGIN_PX + 1
            mask = cv2.erode(mask, np.ones((size, size), dtype=np.uint8))
        return mask

    def points_outside_aruco(self, frame, keypoints):
        mask = self.aruco_exclusion_mask(frame)
        pixels = np.rint(keypoints).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, frame.shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, frame.shape[0] - 1)
        return mask[pixels[:, 1], pixels[:, 0]] > 0

    def as_lightglue_features(self, features):
        result = {
            "keypoints": torch.as_tensor(
                features["keypoints"],
                device=self.device,
                dtype=torch.float32,
            )[None],
            "descriptors": torch.as_tensor(
                features["descriptors"],
                device=self.device,
                dtype=torch.float32,
            )[None],
            "keypoint_scores": torch.as_tensor(
                features["scores"],
                device=self.device,
                dtype=torch.float32,
            )[None],
            "image_size": torch.as_tensor(
                features["image_size"],
                device=self.device,
                dtype=torch.float32,
            )[None],
        }
        if self.requires_scale_orientation:
            result["scales"] = torch.as_tensor(
                features["scales"],
                device=self.device,
                dtype=torch.float32,
            )[None]
            result["oris"] = torch.as_tensor(
                features["oris"],
                device=self.device,
                dtype=torch.float32,
            )[None]
        return result

    def match(self, features0, features1):
        lightglue_features0 = self.as_lightglue_features(features0)
        lightglue_features1 = self.as_lightglue_features(features1)

        with torch.inference_mode():
            output = self.matcher(
                {
                    "image0": lightglue_features0,
                    "image1": lightglue_features1,
                }
            )

        return (
            rbd(output)["matches"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint32)
        )
