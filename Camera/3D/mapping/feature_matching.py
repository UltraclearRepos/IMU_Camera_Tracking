import cv2
import numpy as np
import torch
from lightglue import DISK, SIFT, LightGlue
from lightglue.utils import rbd


DEVICE = "cuda"
MAX_FEATURES = 512
MASK_ARUCO_FEATURES = True
ARUCO_MASK_MARGIN_PX = 6
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
        mask_aruco_features=MASK_ARUCO_FEATURES,
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

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary)

    def extract(self, frame):
        height, width = frame.shape[:2]
        roi_top = round(
            height * (1.0 - self.feature_roi_bottom_fraction)
        )
        roi = frame[roi_top:]

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

        result = {
            "keypoints": keypoints.astype(np.float32),
            "descriptors": descriptors.astype(np.float32),
            "scores": scores.astype(np.float32),
            "image_size": np.array([width, height], dtype=np.float32),
            "roi_top": roi_top,
        }
        if self.requires_scale_orientation:
            result["scales"] = scales.astype(np.float32)
            result["oris"] = orientations.astype(np.float32)
        return result

    def points_outside_aruco(self, frame, keypoints):
        corners, _, _ = self.aruco_detector.detectMarkers(frame)
        if not corners:
            return np.ones(len(keypoints), dtype=bool)

        mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
        for marker_corners in corners:
            polygon = np.rint(marker_corners.reshape(4, 2)).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, 0)

        if ARUCO_MASK_MARGIN_PX > 0:
            size = 2 * ARUCO_MASK_MARGIN_PX + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            mask = cv2.erode(mask, kernel)

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
