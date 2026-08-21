import time

import cv2
import numpy as np
import torch

from mapping.feature_matching import DEVICE


MIN_MATCHES = 20
MIN_MATCH_RATIO = 0.15
MIN_INLIERS = 20
MIN_INLIER_RATIO = 0.125
MAX_REPROJECTION_ERROR_PX = 3.0
GLOBAL_MAP_VISIBILITY_MARGIN_PX = 80

MAP_COVERAGE_GRID_ROWS = 4
MAP_COVERAGE_GRID_COLUMNS = 4
TARGET_LANDMARKS_PER_CELL = 15


def start_timer():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def elapsed_ms(started):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started)


def required_pose_counts(matchable_landmarks):
    required_matches = max(
        MIN_MATCHES,
        int(np.ceil(MIN_MATCH_RATIO * matchable_landmarks)),
    )
    required_inliers = max(
        MIN_INLIERS,
        int(np.ceil(MIN_INLIER_RATIO * matchable_landmarks)),
    )
    return required_matches, required_inliers


class SkinMapTracker:
    def __init__(
        self,
        camera_matrix,
        distortion,
        feature_roi_bottom_fraction,
        global_map,
        feature_matching,
    ):
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.feature_roi_bottom_fraction = feature_roi_bottom_fraction
        self.feature_matching = feature_matching

        if global_map.coordinate_frame != "last_registered_mapping_camera":
            raise ValueError(
                "The frozen map must use the last registered mapping camera "
                "coordinate frame"
            )

        self.map_points = global_map.positions.copy()
        self.map_descriptors = global_map.descriptors
        self.map_scores = global_map.scores
        self.map_scales = global_map.scales
        self.map_orientations = global_map.orientations
        if self.feature_matching.requires_scale_orientation and (
            self.map_scales is None or self.map_orientations is None
        ):
            raise ValueError(
                "SIFT global map must contain landmark scales and "
                "orientations"
            )
        # The map frame is the last mapping-camera frame, so I, 0 is the
        # initial pose hypothesis for the first tracking frame.
        self.R_map_to_camera = np.eye(3, dtype=np.float64)
        self.t_map_to_camera = np.zeros(3, dtype=np.float64)
        self.initialized = False

        self.landmarks = {
            landmark_id: {
                "position": self.map_points[landmark_id],
                "descriptor": self.map_descriptors[landmark_id],
            }
            for landmark_id in range(len(self.map_points))
        }
        self.keyframes = []
        self.initialization_frame_count = 0
        self.map_coverage_grid_rows = MAP_COVERAGE_GRID_ROWS
        self.map_coverage_grid_columns = MAP_COVERAGE_GRID_COLUMNS
        self.map_coverage_ratio = 0.0
        self.last_diagnostics = {}

    def all_map_points(self):
        return self.map_points

    def visible_global_map(self, current_features):
        rvec = cv2.Rodrigues(self.R_map_to_camera)[0]
        projected_points, _ = cv2.projectPoints(
            self.map_points,
            rvec,
            self.t_map_to_camera,
            self.camera_matrix,
            self.distortion,
        )
        projected_points = projected_points.reshape(-1, 2)
        camera_points = (
            self.R_map_to_camera @ self.map_points.T
        ).T + self.t_map_to_camera

        width, height = current_features["image_size"]
        roi_top = height * (1.0 - self.feature_roi_bottom_fraction)
        expected_visible = camera_points[:, 2] > 0.0
        expected_visible &= projected_points[:, 0] >= 0.0
        expected_visible &= projected_points[:, 0] < width
        expected_visible &= projected_points[:, 1] >= roi_top
        expected_visible &= projected_points[:, 1] < height

        margin = GLOBAL_MAP_VISIBILITY_MARGIN_PX
        matching_area = camera_points[:, 2] > 0.0
        matching_area &= projected_points[:, 0] >= -margin
        matching_area &= projected_points[:, 0] < width + margin
        matching_area &= projected_points[:, 1] >= roi_top - margin
        matching_area &= projected_points[:, 1] < height + margin

        landmark_ids = np.flatnonzero(matching_area)
        if not len(landmark_ids):
            return None

        global_features = {
            "keypoints": projected_points[matching_area].astype(np.float32),
            "descriptors": self.map_descriptors[matching_area],
            "scores": self.map_scores[matching_area],
            "image_size": current_features["image_size"],
        }
        if self.feature_matching.requires_scale_orientation:
            global_features["scales"] = self.map_scales[matching_area]
            global_features["oris"] = self.map_orientations[matching_area]
        return (
            landmark_ids,
            self.map_points[matching_area],
            global_features,
            int(np.sum(expected_visible)),
        )

    def map_coverage(self, projected_points, image_size):
        width, height = image_size
        roi_top = height * (1.0 - self.feature_roi_bottom_fraction)
        columns = np.floor(
            projected_points[:, 0] / width * MAP_COVERAGE_GRID_COLUMNS
        ).astype(int)
        rows = np.floor(
            (projected_points[:, 1] - roi_top)
            / (height - roi_top)
            * MAP_COVERAGE_GRID_ROWS
        ).astype(int)
        inside = (
            (columns >= 0)
            & (columns < MAP_COVERAGE_GRID_COLUMNS)
            & (rows >= 0)
            & (rows < MAP_COVERAGE_GRID_ROWS)
        )
        counts = np.zeros(
            (MAP_COVERAGE_GRID_ROWS, MAP_COVERAGE_GRID_COLUMNS),
            dtype=int,
        )
        np.add.at(counts, (rows[inside], columns[inside]), 1)
        coverage = np.clip(counts / TARGET_LANDMARKS_PER_CELL, 0.0, 1.0)
        return int(np.sum(inside)), float(np.mean(coverage))

    def reset_diagnostics(self, feature_count):
        self.last_diagnostics = {
            "tracking_method": "lightglue",
            "matches": 0,
            "flow_tracks": 0,
            "inliers": 0,
            "required_matches": np.nan,
            "required_inliers": np.nan,
            "pnp_inlier_ratio": np.nan,
            "new_features": feature_count,
            "keyframe_added": 0,
            "nearby_associations": 0,
            "new_landmarks": 0,
            "removed_landmarks": 0,
            "visible_landmarks": 0,
            "map_coverage_ratio": np.nan,
            "map_expansion_coverage_threshold": np.nan,
            "initialization_frames": 0,
            "initialization_candidates": 0,
            "initialization_confirmed": 0,
            "initialization_matches": 0,
            "initialization_points": np.empty((0, 2)),
            "feature_extraction_ms": np.nan,
            "global_map_projection_ms": np.nan,
            "lightglue_ms": np.nan,
            "optical_flow_ms": np.nan,
            "pnp_ransac_ms": np.nan,
            "pnp_refine_ms": np.nan,
            "map_coverage_ms": np.nan,
            "map_update_ms": np.nan,
            "new_landmark_points": np.empty((0, 2)),
        }

    def track(self, frame):
        initializing = not self.initialized
        extraction_started = start_timer()
        current_features = self.feature_matching.extract(frame)
        feature_extraction_ms = elapsed_ms(extraction_started)
        self.reset_diagnostics(len(current_features["keypoints"]))
        self.last_diagnostics["feature_extraction_ms"] = (
            feature_extraction_ms
        )
        if initializing:
            self.initialization_frame_count += 1
            self.last_diagnostics["initialization_frames"] = (
                self.initialization_frame_count
            )

        projection_started = start_timer()
        visible_map = self.visible_global_map(current_features)
        self.last_diagnostics["global_map_projection_ms"] = elapsed_ms(
            projection_started
        )
        if visible_map is None:
            return None

        (
            visible_landmark_ids,
            visible_map_points,
            global_features,
            expected_visible_count,
        ) = visible_map
        self.last_diagnostics["visible_landmarks"] = expected_visible_count
        matchable_landmarks = min(
            expected_visible_count,
            len(current_features["keypoints"]),
        )
        required_matches, required_inliers = required_pose_counts(
            matchable_landmarks
        )
        self.last_diagnostics["required_matches"] = required_matches
        self.last_diagnostics["required_inliers"] = required_inliers

        matching_started = start_timer()
        matches = self.feature_matching.match(
            global_features,
            current_features,
        )
        self.last_diagnostics["lightglue_ms"] = elapsed_ms(matching_started)
        self.last_diagnostics["matches"] = len(matches)
        self.last_diagnostics["new_features"] = (
            len(current_features["keypoints"])
            - len(np.unique(matches[:, 1]))
        )
        if len(matches) < required_matches:
            return None

        landmark_ids = visible_landmark_ids[matches[:, 0]]
        map_points = np.ascontiguousarray(
            visible_map_points[matches[:, 0]],
            dtype=np.float64,
        )
        image_points = np.ascontiguousarray(
            current_features["keypoints"][matches[:, 1]],
            dtype=np.float64,
        )

        pnp_guess = {}
        if self.initialized:
            pnp_guess = {
                "rvec": cv2.Rodrigues(self.R_map_to_camera)[0],
                "tvec": self.t_map_to_camera.reshape(3, 1).copy(),
                "useExtrinsicGuess": True,
            }
        pnp_started = start_timer()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            map_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            iterationsCount=200,
            reprojectionError=MAX_REPROJECTION_ERROR_PX,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
            **pnp_guess,
        )
        self.last_diagnostics["pnp_ransac_ms"] = elapsed_ms(pnp_started)

        inlier_count = 0 if inliers is None else len(inliers)
        self.last_diagnostics["inliers"] = inlier_count
        self.last_diagnostics["pnp_inlier_ratio"] = (
            inlier_count / len(matches)
        )
        if not success or inlier_count < required_inliers:
            return None

        inlier_indices = inliers.ravel()
        refinement_started = start_timer()
        rvec, tvec = cv2.solvePnPRefineLM(
            map_points[inlier_indices],
            image_points[inlier_indices],
            self.camera_matrix,
            self.distortion,
            rvec,
            tvec,
        )
        self.last_diagnostics["pnp_refine_ms"] = elapsed_ms(
            refinement_started
        )

        self.R_map_to_camera = cv2.Rodrigues(rvec)[0]
        self.t_map_to_camera = tvec.reshape(3)
        if initializing:
            self.initialized = True
            self.keyframes = ["tracking_start"]
            self.last_diagnostics["initialization_confirmed"] = 1
            self.last_diagnostics["initialization_matches"] = len(matches)

        projected_visible, _ = cv2.projectPoints(
            visible_map_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.distortion,
        )
        coverage_started = start_timer()
        visible_count, coverage_ratio = self.map_coverage(
            projected_visible.reshape(-1, 2),
            current_features["image_size"],
        )
        self.last_diagnostics["map_coverage_ms"] = elapsed_ms(
            coverage_started
        )
        self.last_diagnostics["visible_landmarks"] = visible_count
        self.last_diagnostics["map_coverage_ratio"] = coverage_ratio
        self.map_coverage_ratio = coverage_ratio

        inlier_mask = np.zeros(len(matches), dtype=bool)
        inlier_mask[inlier_indices] = True
        result_R = self.R_map_to_camera
        result_t = self.t_map_to_camera
        result_map_points = map_points[inlier_indices]
        return {
            "R": result_R,
            "t": result_t,
            "inliers": inlier_count,
            "inlier_map_points": result_map_points,
            "inlier_image_points": image_points[inlier_indices],
            "inlier_landmark_ids": landmark_ids[inlier_indices],
            "outlier_points": image_points[~inlier_mask],
            "nearby_associations": 0,
            "new_landmark_image_points": np.empty((0, 2)),
            "new_landmark_ids": np.empty(0, dtype=np.int64),
            "visible_landmarks": visible_count,
            "map_coverage_ratio": coverage_ratio,
        }
