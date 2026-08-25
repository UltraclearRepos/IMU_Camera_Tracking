import cv2
import numpy as np
import torch
from lightglue import DISK, SIFT, LightGlue
from lightglue.utils import rbd

from mapping.adaptive_skin_mask import AdaptiveSkinMask
from mapping.aruco_mask import ArucoMask
from mapping.mapping_data import FeatureSet

DEVICE = "cuda"
MAX_FEATURES = 512
MASK_ARUCO_FEATURES = True
ARUCO_MASK_MARGIN_PX = 20
FEATURE_TYPES = ("disk", "sift")


def frame_to_tensor(frame, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return image.to(device)


class LightGlueFeatureMatching:
    def __init__(
        self,
        feature_roi_bottom_fraction,
        feature_type="disk",
        max_features=MAX_FEATURES,
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
        self.max_features = max_features
        self.requires_scale_orientation = feature_type == "sift"
        self.feature_roi_bottom_fraction = feature_roi_bottom_fraction
        self.aruco_mask = ArucoMask(margin_px=ARUCO_MASK_MARGIN_PX) if mask_aruco_features else None
        self.adaptive_skin_mask = AdaptiveSkinMask() if use_adaptive_skin_mask else None
        if feature_type == "disk":
            self.extractor = DISK(
                max_num_keypoints=self.max_features
            ).eval().to(self.device)
        else:
            self.extractor = SIFT(
                max_num_keypoints=self.max_features,
                backend="opencv",
            ).eval().to(self.device)
        self.matcher = LightGlue(features=feature_type).eval().to(
            self.device
        )

    def extract(self, frame):
        height, width = frame.shape[:2]
        roi_top = round(
            height * (1.0 - self.feature_roi_bottom_fraction)
        )
        aruco_exclusion_mask = (
            np.full(frame.shape[:2], 255, dtype=np.uint8)
            if self.aruco_mask is None
            else self.aruco_mask.compute(frame)
        )
        skin_mask_result = (
            None
            if self.adaptive_skin_mask is None
            else self.adaptive_skin_mask.compute(frame, roi_top, aruco_exclusion_mask)
        )
        selection_bounds = (
            np.array([0, roi_top, width, height], dtype=np.int32)
            if skin_mask_result is None
            else skin_mask_result.bounds
        )
        crop_left, crop_top, crop_right, crop_bottom = selection_bounds
        crop = frame[crop_top:crop_bottom, crop_left:crop_right].copy()

        with torch.inference_mode():
            extracted = self.extractor.extract(
                frame_to_tensor(crop, self.device)
            )

        extracted_features = rbd(extracted)
        keypoints = extracted_features["keypoints"].detach().cpu().numpy().copy()
        descriptors = extracted_features["descriptors"].detach().cpu().numpy().copy()
        scores = extracted_features["keypoint_scores"].detach().cpu().numpy().copy()

        keypoints[:, 0] += crop_left
        keypoints[:, 1] += crop_top

        scales = None
        orientations = None
        if self.requires_scale_orientation:
            scales = (
                extracted_features["scales"].detach().cpu().numpy().copy()
            )
            orientations = (
                extracted_features["oris"].detach().cpu().numpy().copy()
            )

        feature_mask = aruco_exclusion_mask.astype(bool)
        if skin_mask_result is not None:
            feature_mask &= skin_mask_result.mask

        keep = self._points_in_mask(keypoints, feature_mask)
        keypoints = keypoints[keep]
        descriptors = descriptors[keep]
        scores = scores[keep]
        if scales is not None and orientations is not None:
            scales = scales[keep]
            orientations = orientations[keep]

        return FeatureSet(
            keypoints=keypoints.astype(np.float32),
            descriptors=descriptors.astype(np.float32),
            scores=scores.astype(np.float32),
            image_size=np.array([width, height], dtype=np.float32),
            roi_top=roi_top,
            scales=(
                scales.astype(np.float32)
                if scales is not None
                else None
            ),
            orientations=(
                orientations.astype(np.float32)
                if orientations is not None
                else None
            ),
            selection_mask=(
                None
                if skin_mask_result is None
                else skin_mask_result.mask
            ),
            selection_bounds=selection_bounds.copy(),
        )

    @staticmethod
    def _points_in_mask(keypoints, mask):
        pixels = np.rint(keypoints).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, mask.shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, mask.shape[0] - 1)
        return mask[pixels[:, 1], pixels[:, 0]] > 0

    def as_lightglue_features(self, features):
        result = {
            "keypoints": torch.as_tensor(
                features.keypoints,
                device=self.device,
                dtype=torch.float32,
            )[None],
            "descriptors": torch.as_tensor(
                features.descriptors,
                device=self.device,
                dtype=torch.float32,
            )[None],
            "keypoint_scores": torch.as_tensor(
                features.scores,
                device=self.device,
                dtype=torch.float32,
            )[None],
            "image_size": torch.as_tensor(
                features.image_size,
                device=self.device,
                dtype=torch.float32,
            )[None],
        }
        if self.requires_scale_orientation:
            if features.scales is None or features.orientations is None:
                raise ValueError(
                    "SIFT LightGlue features require scales and orientations"
                )
            result["scales"] = torch.as_tensor(
                features.scales,
                device=self.device,
                dtype=torch.float32,
            )[None]
            result["oris"] = torch.as_tensor(
                features.orientations,
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
