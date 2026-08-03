import cv2
import numpy as np
import torch
import time
from lightglue import DISK, LightGlue
from lightglue.utils import rbd
from scipy.spatial import cKDTree


# Model and feature extraction
DEVICE = "cuda"  # Device used by DISK and LightGlue.
MAX_FEATURES = 512  # Maximum number of DISK features in the current frame.

# Global map matching and pose estimation
MIN_MATCHES = 20  # Absolute minimum LightGlue matches required to run PnP.
MIN_MATCH_RATIO = 0.15  # Required matches relative to potentially matchable visible landmarks.
MIN_INLIERS = 20  # Absolute minimum PnP inliers required to accept the pose.
MIN_INLIER_RATIO = 0.125  # Required inliers relative to potentially matchable visible landmarks.
MAX_REPROJECTION_ERROR_PX = 3.0  # Maximum point reprojection error in pixels.
GLOBAL_MAP_VISIBILITY_MARGIN_PX = 80  # Projection margin outside the image.

# Map expansion
MAP_COVERAGE_GRID_ROWS = 4  # Number of horizontal grid bands in the feature ROI.
MAP_COVERAGE_GRID_COLUMNS = 4  # Number of vertical grid bands in the feature ROI.
TARGET_LANDMARKS_PER_CELL = 15  # Desired visible global landmarks in one grid cell.
MAP_EXPANSION_MIN_COVERAGE_RATIO = 0.70  # Add a keyframe below this mean grid coverage.

# Landmark creation and association
LANDMARK_MIN_DISTANCE_MM = 1.0  # Minimum spacing between global map points.
LANDMARK_DESCRIPTOR_SIMILARITY = 0.80  # Minimum descriptor similarity for one landmark.
MAX_NEW_LANDMARKS_PER_KEYFRAME = 100  # Maximum new global map points added by one keyframe.
UNCOVERED_FEATURE_MIN_SCORE = 50.0  # Minimum raw DISK score in uncovered regions.
COVERED_FEATURE_MIN_SCORE = 75.0  # Minimum raw DISK score in covered regions.
MAX_GLOBAL_LANDMARKS = 1024  # Hard maximum number of global map points.
LANDMARK_PROTECTION_VISIBLE_COUNT = 10  # Visibility opportunities protecting a new landmark.
LANDMARK_INLIER_QUALITY_WEIGHT = 0.70  # PnP quality weight versus matching frequency.
LANDMARK_PRUNING_DENSITY_RADIUS_MM = 5.0  # Radius used to count nearby map points during pruning.
LANDMARK_PRUNING_DENSITY_WEIGHT = 0.35  # Importance of spatial density versus landmark quality.

# ArUco
MASK_ARUCO_FEATURES = False  # Exclude features located inside the ArUco marker.
ARUCO_ID = 7  # Marker identifier used to determine the initial pose.
ARUCO_SIZE_MM = 20.0  # Physical marker side length.

# Initial map creation
INITIALIZATION_FRAMES = 5  # Number of frames used to initialize the map.
INITIALIZATION_MIN_OBSERVATIONS = 3  # Minimum observations during initialization.
INITIALIZATION_MIN_LANDMARKS = 60  # Minimum stable landmarks in the initial map.
INITIALIZATION_POINT_MAX_DISTANCE_MM = 1.5  # Maximum point spread during initialization.
INITIAL_MAP_MAX_LANDMARKS = 200  # Maximum landmarks created in the first keyframe.


def frame_to_tensor(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return image.to(DEVICE)


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
        map_expansion_min_coverage_ratio=MAP_EXPANSION_MIN_COVERAGE_RATIO,
    ):
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.feature_roi_bottom_fraction = feature_roi_bottom_fraction
        self.map_expansion_min_coverage_ratio = (
            map_expansion_min_coverage_ratio
        )

        self.extractor = DISK(max_num_keypoints=MAX_FEATURES).eval().to(DEVICE)
        self.matcher = LightGlue(features="disk").eval().to(DEVICE)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary)

        self.keyframes = []
        self.landmarks = {}
        self.next_landmark_id = 0
        self.last_diagnostics = {}
        self.initialization = None
        self.R_map_to_camera = None
        self.t_map_to_camera = None
        self.map_coverage_ratio = 0.0

    def extract_features(self, frame):
        height, width = frame.shape[:2]
        roi_top = round(
            height * (1.0 - self.feature_roi_bottom_fraction)
        )
        roi = frame[roi_top:]

        with torch.inference_mode():
            features = self.extractor.extract(frame_to_tensor(roi))

        keypoints = features["keypoints"].clone()
        keypoints[0, :, 1] += roi_top
        features["keypoints"] = keypoints
        features["image_size"] = features["image_size"].new_tensor(
            [[width, height]]
        )
        keep = torch.ones(
            features["keypoints"].shape[1],
            dtype=torch.bool,
            device=features["keypoints"].device,
        )

        if MASK_ARUCO_FEATURES:
            corners, _, _ = self.aruco_detector.detectMarkers(frame)
            keypoints = features["keypoints"][0].detach().cpu().numpy()
            for marker_corners in corners:
                polygon = marker_corners.reshape(4, 2).astype(np.float32)
                inside_marker = np.array(
                    [
                        cv2.pointPolygonTest(
                            polygon,
                            tuple(map(float, point)),
                            False,
                        )
                        >= 0
                        for point in keypoints
                    ]
                )
                keep &= torch.from_numpy(~inside_marker).to(keep.device)

        for name in ("keypoints", "keypoint_scores", "descriptors"):
            features[name] = features[name][:, keep]
        return features

    def find_initial_pose(self, frame):
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        if ids is None or ARUCO_ID not in ids.flatten():
            return None

        marker_index = np.where(ids.flatten() == ARUCO_ID)[0][0]
        image_points = corners[marker_index].reshape(4, 2).astype(np.float64)

        half = ARUCO_SIZE_MM / 2.0
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None

        R_map_to_camera = cv2.Rodrigues(rvec)[0]
        return R_map_to_camera, tvec.reshape(3)

    def pixels_to_skin_plane(self, keypoints, R_map_to_camera, t_map_to_camera):
        normalized = cv2.undistortPoints(
            keypoints.reshape(-1, 1, 2),
            self.camera_matrix,
            self.distortion,
        ).reshape(-1, 2)

        rays_camera = np.column_stack((normalized, np.ones(len(normalized))))
        camera_origin_map = -R_map_to_camera.T @ t_map_to_camera.reshape(3)
        rays_map = (R_map_to_camera.T @ rays_camera.T).T

        scale = -camera_origin_map[2] / rays_map[:, 2]
        map_points = camera_origin_map + scale[:, None] * rays_map
        map_points[scale <= 0.0] = np.nan
        map_points[:, 2] = 0.0
        return map_points

    def start_initialization(
        self,
        features,
        R_map_to_camera,
        t_map_to_camera,
    ):
        keypoints = rbd(features)["keypoints"].detach().cpu().numpy()
        map_points = self.pixels_to_skin_plane(
            keypoints,
            R_map_to_camera,
            t_map_to_camera,
        )
        valid = np.isfinite(map_points).all(axis=1)
        observations = np.zeros(len(keypoints), dtype=np.int32)
        observations[valid] = 1

        self.initialization = {
            "features": features,
            "R": R_map_to_camera,
            "t": t_map_to_camera,
            "map_points": map_points,
            "observations": observations,
            "frames": 1,
        }
        self.last_diagnostics["initialization_frames"] = 1
        self.last_diagnostics["initialization_candidates"] = int(valid.sum())
        self.last_diagnostics["initialization_confirmed"] = 0
        self.last_diagnostics["initialization_matches"] = 0
        self.last_diagnostics["initialization_points"] = keypoints[valid]

    def update_initialization(
        self,
        features,
        R_map_to_camera,
        t_map_to_camera,
    ):
        initialization = self.initialization
        current_keypoints = (
            rbd(features)["keypoints"].detach().cpu().numpy()
        )
        current_map_points = self.pixels_to_skin_plane(
            current_keypoints,
            R_map_to_camera,
            t_map_to_camera,
        )

        with torch.inference_mode():
            matching_started = start_timer()
            output = self.matcher(
                {
                    "image0": initialization["features"],
                    "image1": features,
                }
            )
            self.last_diagnostics["lightglue_ms"] = elapsed_ms(
                matching_started
            )
        matches = rbd(output)["matches"].detach().cpu().numpy()

        consistent_points = []
        for anchor_index, current_index in matches:
            anchor_point = initialization["map_points"][anchor_index]
            current_point = current_map_points[current_index]
            if not np.isfinite(anchor_point).all():
                continue
            if not np.isfinite(current_point).all():
                continue
            if (
                np.linalg.norm(anchor_point[:2] - current_point[:2])
                > INITIALIZATION_POINT_MAX_DISTANCE_MM
            ):
                continue

            initialization["observations"][anchor_index] += 1
            consistent_points.append(current_keypoints[current_index])

        initialization["frames"] += 1
        confirmed_indices = np.flatnonzero(
            initialization["observations"]
            >= INITIALIZATION_MIN_OBSERVATIONS
        )

        self.last_diagnostics["initialization_frames"] = initialization[
            "frames"
        ]
        self.last_diagnostics["initialization_candidates"] = int(
            np.isfinite(initialization["map_points"]).all(axis=1).sum()
        )
        self.last_diagnostics["initialization_confirmed"] = len(
            confirmed_indices
        )
        self.last_diagnostics["initialization_matches"] = len(
            consistent_points
        )
        self.last_diagnostics["initialization_points"] = np.array(
            consistent_points
        ).reshape(-1, 2)

        initialization_ready = (
            initialization["frames"] >= INITIALIZATION_FRAMES
            and len(confirmed_indices) >= INITIALIZATION_MIN_LANDMARKS
        )
        if not initialization_ready:
            return None

        no_known_features = np.empty(0, dtype=np.int64)
        map_update_started = start_timer()
        map_update = self.add_keyframe(
            initialization["features"],
            initialization["R"],
            initialization["t"],
            no_known_features,
            no_known_features,
            confirmed_indices,
        )
        self.last_diagnostics["map_update_ms"] = elapsed_ms(
            map_update_started
        )
        self.initialization = None
        self.last_diagnostics.update(map_update)

        return {
            "R": R_map_to_camera,
            "t": t_map_to_camera,
            "inliers": 0,
            "inlier_map_points": np.empty((0, 3)),
            "outlier_points": np.empty((0, 2)),
            "nearby_associations": 0,
        }

    def all_map_points(self):
        return np.array(
            [landmark["position"] for landmark in self.landmarks.values()],
            dtype=float,
        ).reshape(-1, 3)

    def create_landmark(
        self,
        position,
        descriptor,
        keyframe_id,
        feature_index,
    ):
        landmark_id = self.next_landmark_id
        self.next_landmark_id += 1
        descriptor = descriptor / np.linalg.norm(descriptor)
        self.landmarks[landmark_id] = {
            "position": position.copy(),
            "observations": {keyframe_id: feature_index},
            "descriptor_sum": descriptor.copy(),
            "descriptor": descriptor.copy(),
            "visible_count": 0,
            "match_count": 0,
            "inlier_count": 0,
        }
        return landmark_id

    def landmark_quality(self, landmark):
        match_rate = (
            landmark["match_count"] / landmark["visible_count"]
        )
        inlier_rate = (
            (landmark["inlier_count"] + 1.0)
            / (landmark["match_count"] + 2.0)
        )
        return (
            LANDMARK_INLIER_QUALITY_WEIGHT * inlier_rate
            + (1.0 - LANDMARK_INLIER_QUALITY_WEIGHT) * match_rate
        )

    def update_landmark_quality(
        self,
        candidate_landmark_ids,
        candidate_map_points,
        matches,
        inlier_indices,
        R_map_to_camera,
        t_map_to_camera,
        image_size,
    ):
        rvec = cv2.Rodrigues(R_map_to_camera)[0]
        projected_points, _ = cv2.projectPoints(
            candidate_map_points,
            rvec,
            t_map_to_camera,
            self.camera_matrix,
            self.distortion,
        )
        projected_points = projected_points.reshape(-1, 2)
        camera_points = (
            R_map_to_camera @ candidate_map_points.T
        ).T + t_map_to_camera.reshape(3)

        width, height = image_size[0].detach().cpu().numpy()
        roi_top = height * (1.0 - self.feature_roi_bottom_fraction)
        visible = camera_points[:, 2] > 0.0
        visible &= projected_points[:, 0] >= 0.0
        visible &= projected_points[:, 0] < width
        visible &= projected_points[:, 1] >= roi_top
        visible &= projected_points[:, 1] < height

        for landmark_id in candidate_landmark_ids[visible]:
            self.landmarks[int(landmark_id)]["visible_count"] += 1

        matched_visible = visible[matches[:, 0]]
        matched_landmark_ids = candidate_landmark_ids[matches[:, 0]]
        for landmark_id in matched_landmark_ids[matched_visible]:
            self.landmarks[int(landmark_id)]["match_count"] += 1

        for inlier_index in inlier_indices:
            if matched_visible[inlier_index]:
                landmark_id = int(matched_landmark_ids[inlier_index])
                self.landmarks[landmark_id]["inlier_count"] += 1

    def remove_landmark(self, landmark_id):
        landmark = self.landmarks[landmark_id]
        for keyframe_id, feature_index in landmark["observations"].items():
            keyframe = self.keyframes[keyframe_id]
            if keyframe["landmark_ids"][feature_index] == landmark_id:
                keyframe["landmark_ids"][feature_index] = -1
        del self.landmarks[landmark_id]

    def prune_global_map(
        self,
        protected_landmark_ids,
        points_to_remove,
    ):
        if points_to_remove <= 0:
            return 0

        candidates = [
            landmark_id
            for landmark_id, landmark in self.landmarks.items()
            if landmark["visible_count"]
            >= LANDMARK_PROTECTION_VISIBLE_COUNT
            and landmark_id not in protected_landmark_ids
        ]

        landmark_ids = np.array(list(self.landmarks), dtype=np.int64)
        landmark_points = np.array(
            [
                self.landmarks[landmark_id]["position"][:2]
                for landmark_id in landmark_ids
            ]
        )
        landmark_tree = cKDTree(landmark_points)
        neighbor_counts = np.array(
            [
                len(neighbors) - 1
                for neighbors in landmark_tree.query_ball_point(
                    landmark_points,
                    LANDMARK_PRUNING_DENSITY_RADIUS_MM,
                )
            ],
            dtype=float,
        )
        maximum_neighbor_count = np.max(neighbor_counts)
        if maximum_neighbor_count > 0.0:
            neighbor_counts /= maximum_neighbor_count
        density_by_landmark_id = dict(
            zip(map(int, landmark_ids), neighbor_counts)
        )

        def retention_score(landmark_id):
            quality = self.landmark_quality(
                self.landmarks[landmark_id]
            )
            sparse_area_score = (
                1.0 - density_by_landmark_id[landmark_id]
            )
            return (
                (1.0 - LANDMARK_PRUNING_DENSITY_WEIGHT)
                * quality
                + LANDMARK_PRUNING_DENSITY_WEIGHT
                * sparse_area_score
            )

        candidates.sort(
            key=lambda landmark_id: (
                retention_score(landmark_id),
                self.landmarks[landmark_id]["inlier_count"],
                len(self.landmarks[landmark_id]["observations"]),
            )
        )

        removed_landmark_ids = candidates[:points_to_remove]
        for landmark_id in removed_landmark_ids:
            self.remove_landmark(landmark_id)
        return len(removed_landmark_ids)

    def add_observation(
        self,
        landmark_id,
        descriptor,
        keyframe_id,
        feature_index,
    ):
        landmark = self.landmarks[landmark_id]
        landmark["observations"][keyframe_id] = feature_index
        landmark["descriptor_sum"] += descriptor / np.linalg.norm(descriptor)
        descriptor_sum = landmark["descriptor_sum"]
        landmark["descriptor"] = descriptor_sum / np.linalg.norm(descriptor_sum)

    def feature_scores(
        self,
        feature_data,
        feature_indices,
    ):
        return (
            feature_data["keypoint_scores"]
            .detach()
            .cpu()
            .numpy()[feature_indices]
        )

    def select_new_landmarks(
        self,
        feature_data,
        feature_indices,
        map_points,
    ):
        if not len(feature_indices):
            return feature_indices, map_points

        scores = self.feature_scores(
            feature_data,
            feature_indices,
        )

        if not self.landmarks:
            selected = np.argsort(scores)[::-1][
                :INITIAL_MAP_MAX_LANDMARKS
            ]
            return feature_indices[selected], map_points[selected]

        keypoints = (
            feature_data["keypoints"]
            .detach()
            .cpu()
            .numpy()[feature_indices]
        )
        _, cell_counts = self.map_coverage_grid(
            feature_data["image_size"]
        )
        width, height = (
            feature_data["image_size"]
            .detach()
            .cpu()
            .numpy()
        )
        roi_top = height * (
            1.0 - self.feature_roi_bottom_fraction
        )
        columns = np.floor(
            keypoints[:, 0]
            / width
            * MAP_COVERAGE_GRID_COLUMNS
        ).astype(int)
        rows = np.floor(
            (keypoints[:, 1] - roi_top)
            / (height - roi_top)
            * MAP_COVERAGE_GRID_ROWS
        ).astype(int)
        columns = np.clip(columns, 0, MAP_COVERAGE_GRID_COLUMNS - 1)
        rows = np.clip(rows, 0, MAP_COVERAGE_GRID_ROWS - 1)

        selected_for_coverage_by_cell = []
        selected_for_coverage_mask = np.zeros(
            len(feature_indices),
            dtype=bool,
        )
        for row in range(MAP_COVERAGE_GRID_ROWS):
            for column in range(MAP_COVERAGE_GRID_COLUMNS):
                missing_landmarks = max(
                    TARGET_LANDMARKS_PER_CELL
                    - cell_counts[row, column],
                    0,
                )
                candidates = np.flatnonzero(
                    (rows == row)
                    & (columns == column)
                    & (scores >= UNCOVERED_FEATURE_MIN_SCORE)
                )
                candidates = candidates[
                    np.argsort(scores[candidates])[::-1]
                ][:missing_landmarks]
                selected_for_coverage_by_cell.append(candidates)
                selected_for_coverage_mask[candidates] = True

        selected_for_coverage = []
        maximum_cell_selection = max(
            map(len, selected_for_coverage_by_cell),
            default=0,
        )
        for candidate_rank in range(maximum_cell_selection):
            for candidates in selected_for_coverage_by_cell:
                if candidate_rank < len(candidates):
                    selected_for_coverage.append(
                        candidates[candidate_rank]
                    )

        exceptional_candidates = np.flatnonzero(
            ~selected_for_coverage_mask
            & (scores >= COVERED_FEATURE_MIN_SCORE)
        )
        exceptional_candidates = exceptional_candidates[
            np.argsort(scores[exceptional_candidates])[::-1]
        ]
        selected = np.concatenate(
            (
                np.asarray(selected_for_coverage, dtype=int),
                exceptional_candidates,
            )
        )[:MAX_NEW_LANDMARKS_PER_KEYFRAME]
        return feature_indices[selected], map_points[selected]

    def associate_nearby_landmarks(
        self,
        feature_indices,
        map_points,
        keypoints,
        descriptors,
        R_map_to_camera,
        t_map_to_camera,
        occupied_landmark_ids,
    ):
        if not self.landmarks or not len(feature_indices):
            return {}

        landmark_ids = np.array(list(self.landmarks), dtype=np.int64)
        landmark_points = np.array(
            [self.landmarks[landmark_id]["position"] for landmark_id in landmark_ids]
        )
        landmark_descriptors = np.array(
            [self.landmarks[landmark_id]["descriptor"] for landmark_id in landmark_ids]
        )
        landmark_tree = cKDTree(landmark_points[:, :2])

        rvec = cv2.Rodrigues(R_map_to_camera)[0]
        projected_points, _ = cv2.projectPoints(
            landmark_points,
            rvec,
            t_map_to_camera,
            self.camera_matrix,
            self.distortion,
        )
        projected_points = projected_points.reshape(-1, 2)
        camera_points = (
            R_map_to_camera @ landmark_points.T
        ).T + t_map_to_camera.reshape(3)

        candidates = []
        for feature_index, map_point in zip(feature_indices, map_points):
            nearby_indices = landmark_tree.query_ball_point(
                map_point[:2],
                LANDMARK_MIN_DISTANCE_MM,
            )
            descriptor = descriptors[feature_index]
            descriptor = descriptor / np.linalg.norm(descriptor)

            for landmark_index in nearby_indices:
                landmark_id = int(landmark_ids[landmark_index])
                if landmark_id in occupied_landmark_ids:
                    continue
                if camera_points[landmark_index, 2] <= 0.0:
                    continue

                reprojection_error = np.linalg.norm(
                    projected_points[landmark_index] - keypoints[feature_index]
                )
                if (
                    reprojection_error
                    > MAX_REPROJECTION_ERROR_PX
                ):
                    continue

                similarity = float(
                    descriptor @ landmark_descriptors[landmark_index]
                )
                if similarity < LANDMARK_DESCRIPTOR_SIMILARITY:
                    continue

                spatial_distance = np.linalg.norm(
                    map_point[:2] - landmark_points[landmark_index, :2]
                )
                candidates.append(
                    (
                        similarity,
                        -reprojection_error,
                        -spatial_distance,
                        int(feature_index),
                        landmark_id,
                    )
                )

        candidates.sort(reverse=True)
        associations = {}
        used_landmark_ids = set(occupied_landmark_ids)
        for _, _, _, feature_index, landmark_id in candidates:
            if feature_index in associations:
                continue
            if landmark_id in used_landmark_ids:
                continue
            associations[feature_index] = landmark_id
            used_landmark_ids.add(landmark_id)

        return associations

    def add_keyframe(
        self,
        features,
        R_map_to_camera,
        t_map_to_camera,
        known_feature_indices,
        known_landmark_ids,
        new_feature_indices,
    ):
        feature_data = rbd(features)
        keypoints = feature_data["keypoints"].detach().cpu().numpy()
        descriptors = feature_data["descriptors"].detach().cpu().numpy()
        landmark_ids = np.full(len(keypoints), -1, dtype=np.int64)
        keyframe_id = len(self.keyframes)

        landmark_ids[known_feature_indices] = known_landmark_ids

        new_points = self.pixels_to_skin_plane(
            keypoints[new_feature_indices],
            R_map_to_camera,
            t_map_to_camera,
        )
        valid = np.isfinite(new_points).all(axis=1)
        new_feature_indices = new_feature_indices[valid]
        new_points = new_points[valid]

        occupied_landmark_ids = set(map(int, known_landmark_ids))

        nearby_associations = self.associate_nearby_landmarks(
            new_feature_indices,
            new_points,
            keypoints,
            descriptors,
            R_map_to_camera,
            t_map_to_camera,
            occupied_landmark_ids,
        )
        for feature_index, landmark_id in nearby_associations.items():
            landmark_ids[feature_index] = landmark_id

        remains_new = np.array(
            [
                feature_index not in nearby_associations
                for feature_index in new_feature_indices
            ],
            dtype=bool,
        )
        new_feature_indices = new_feature_indices[remains_new]
        new_points = new_points[remains_new]

        grid_cells = np.rint(
            new_points[:, :2] / LANDMARK_MIN_DISTANCE_MM
        ).astype(np.int32)
        _, unique_indices = np.unique(grid_cells, axis=0, return_index=True)
        new_feature_indices = new_feature_indices[unique_indices]
        new_points = new_points[unique_indices]

        new_feature_indices, new_points = self.select_new_landmarks(
            feature_data,
            new_feature_indices,
            new_points,
        )

        if not len(new_points):
            return {
                "keyframe_added": 0,
                "nearby_associations": len(nearby_associations),
                "new_landmarks": 0,
                "removed_landmarks": 0,
            }

        protected_landmark_ids = set(
            map(int, landmark_ids[landmark_ids >= 0])
        )
        points_to_remove = max(
            0,
            len(self.landmarks)
            + len(new_points)
            - MAX_GLOBAL_LANDMARKS,
        )
        removed_landmarks = self.prune_global_map(
            protected_landmark_ids,
            points_to_remove,
        )
        available_space = MAX_GLOBAL_LANDMARKS - len(self.landmarks)
        new_feature_indices = new_feature_indices[:available_space]
        new_points = new_points[:available_space]

        if not len(new_points):
            return {
                "keyframe_added": 0,
                "nearby_associations": len(nearby_associations),
                "new_landmarks": 0,
                "removed_landmarks": removed_landmarks,
            }

        camera_rotation = R_map_to_camera.T
        camera_position = -camera_rotation @ t_map_to_camera.reshape(3)
        keyframe = {
            "landmark_ids": landmark_ids,
            "camera_position": camera_position,
            "camera_rotation": camera_rotation,
        }
        self.keyframes.append(keyframe)

        observed_feature_indices = np.flatnonzero(landmark_ids >= 0)
        for feature_index in observed_feature_indices:
            self.add_observation(
                int(landmark_ids[feature_index]),
                descriptors[feature_index],
                keyframe_id,
                int(feature_index),
            )

        for feature_index, point in zip(new_feature_indices, new_points):
            landmark_id = self.create_landmark(
                point,
                descriptors[feature_index],
                keyframe_id,
                int(feature_index),
            )
            keyframe["landmark_ids"][feature_index] = landmark_id

        return {
            "keyframe_added": 1,
            "nearby_associations": len(nearby_associations),
            "new_landmarks": len(new_points),
            "removed_landmarks": removed_landmarks,
        }

    def should_add_keyframe(self, result):
        return (
            result["map_coverage_ratio"]
            < self.map_expansion_min_coverage_ratio
            and len(result["new_feature_indices"]) > 0
        )

    def map_coverage_grid(self, image_size):
        width, height = (
            image_size.detach()
            .cpu()
            .numpy()
            .reshape(-1, 2)[0]
        )
        width = int(width)
        height = int(height)
        roi_top = round(
            height * (1.0 - self.feature_roi_bottom_fraction)
        )
        cell_counts = np.zeros(
            (MAP_COVERAGE_GRID_ROWS, MAP_COVERAGE_GRID_COLUMNS),
            dtype=int,
        )
        if not self.landmarks:
            return 0, cell_counts

        map_points = np.array(
            [landmark["position"] for landmark in self.landmarks.values()]
        )
        projected_points, _ = cv2.projectPoints(
            map_points,
            cv2.Rodrigues(self.R_map_to_camera)[0],
            self.t_map_to_camera,
            self.camera_matrix,
            self.distortion,
        )
        projected_points = projected_points.reshape(-1, 2)
        camera_points = (
            self.R_map_to_camera @ map_points.T
        ).T + self.t_map_to_camera.reshape(3)

        visible = camera_points[:, 2] > 0.0
        visible &= projected_points[:, 0] >= 0.0
        visible &= projected_points[:, 0] < width
        visible &= projected_points[:, 1] >= roi_top
        visible &= projected_points[:, 1] < height

        visible_points = projected_points[visible]
        if len(visible_points):
            columns = np.floor(
                visible_points[:, 0]
                / width
                * MAP_COVERAGE_GRID_COLUMNS
            ).astype(int)
            rows = np.floor(
                (visible_points[:, 1] - roi_top)
                / (height - roi_top)
                * MAP_COVERAGE_GRID_ROWS
            ).astype(int)
            columns = np.clip(
                columns,
                0,
                MAP_COVERAGE_GRID_COLUMNS - 1,
            )
            rows = np.clip(rows, 0, MAP_COVERAGE_GRID_ROWS - 1)
            np.add.at(cell_counts, (rows, columns), 1)

        return int(visible.sum()), cell_counts

    def measure_map_coverage(self, image_size):
        visible_landmarks, cell_counts = self.map_coverage_grid(image_size)
        cell_coverage = np.clip(
            cell_counts / TARGET_LANDMARKS_PER_CELL,
            0.0,
            1.0,
        )
        coverage_ratio = float(np.mean(cell_coverage))
        return visible_landmarks, coverage_ratio

    def visible_global_map(self, current_features):
        if not self.landmarks or self.R_map_to_camera is None:
            return None

        landmark_ids = np.array(list(self.landmarks), dtype=np.int64)
        map_points = np.array(
            [self.landmarks[landmark_id]["position"] for landmark_id in landmark_ids]
        )
        descriptors = np.array(
            [self.landmarks[landmark_id]["descriptor"] for landmark_id in landmark_ids]
        )

        rvec = cv2.Rodrigues(self.R_map_to_camera)[0]
        projected_points, _ = cv2.projectPoints(
            map_points,
            rvec,
            self.t_map_to_camera,
            self.camera_matrix,
            self.distortion,
        )
        projected_points = projected_points.reshape(-1, 2)
        camera_points = (
            self.R_map_to_camera @ map_points.T
        ).T + self.t_map_to_camera.reshape(3)

        width, height = (
            current_features["image_size"][0].detach().cpu().numpy()
        )
        roi_top = height * (1.0 - self.feature_roi_bottom_fraction)
        expected_visible = camera_points[:, 2] > 0.0
        expected_visible &= projected_points[:, 0] >= 0.0
        expected_visible &= projected_points[:, 0] < width
        expected_visible &= projected_points[:, 1] >= roi_top
        expected_visible &= projected_points[:, 1] < height
        expected_visible_count = int(expected_visible.sum())

        margin = GLOBAL_MAP_VISIBILITY_MARGIN_PX
        matching_area = camera_points[:, 2] > 0.0
        matching_area &= projected_points[:, 0] >= -margin
        matching_area &= projected_points[:, 0] < width + margin
        matching_area &= projected_points[:, 1] >= roi_top - margin
        matching_area &= projected_points[:, 1] < height + margin

        landmark_ids = landmark_ids[matching_area]
        map_points = map_points[matching_area]
        projected_points = projected_points[matching_area]
        descriptors = descriptors[matching_area]
        if not len(landmark_ids):
            return None

        descriptor_tensor = current_features["descriptors"]
        global_features = {
            "keypoints": torch.as_tensor(
                projected_points,
                device=descriptor_tensor.device,
                dtype=descriptor_tensor.dtype,
            )[None],
            "descriptors": torch.as_tensor(
                descriptors,
                device=descriptor_tensor.device,
                dtype=descriptor_tensor.dtype,
            )[None],
            "image_size": current_features["image_size"],
        }
        return (
            landmark_ids,
            map_points,
            global_features,
            expected_visible_count,
        )

    def match_global_map(self, current_features):
        current_keypoints = (
            rbd(current_features)["keypoints"].detach().cpu().numpy()
        )
        map_projection_started = start_timer()
        visible_map = self.visible_global_map(current_features)
        self.last_diagnostics["global_map_projection_ms"] = elapsed_ms(
            map_projection_started
        )
        if visible_map is None:
            return None

        (
            visible_landmark_ids,
            visible_map_points,
            global_features,
            expected_visible_count,
        ) = visible_map
        matchable_landmarks = min(
            expected_visible_count,
            len(current_keypoints),
        )
        required_matches, required_inliers = required_pose_counts(
            matchable_landmarks
        )
        self.last_diagnostics["required_matches"] = required_matches
        self.last_diagnostics["required_inliers"] = required_inliers

        matching_started = start_timer()
        with torch.inference_mode():
            output = self.matcher(
                {
                    "image0": global_features,
                    "image1": current_features,
                }
            )
        self.last_diagnostics["lightglue_ms"] = elapsed_ms(
            matching_started
        )
        matches = rbd(output)["matches"].detach().cpu().numpy()
        matched_landmark_ids = visible_landmark_ids[matches[:, 0]]
        matched_current_indices = set(map(int, matches[:, 1]))
        self.last_diagnostics["matches"] = len(matches)
        self.last_diagnostics["new_features"] = (
            len(current_keypoints) - len(matched_current_indices)
        )
        if len(matches) < required_matches:
            return None

        landmark_ids = matched_landmark_ids
        current_feature_indices = matches[:, 1]
        map_points = np.ascontiguousarray(
            visible_map_points[matches[:, 0]],
            dtype=np.float64,
        )
        image_points = np.ascontiguousarray(
            current_keypoints[current_feature_indices],
            dtype=np.float64,
        )

        rvec = cv2.Rodrigues(self.R_map_to_camera)[0]
        tvec = self.t_map_to_camera.reshape(3, 1).copy()
        pnp_started = start_timer()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            map_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            iterationsCount=200,
            reprojectionError=MAX_REPROJECTION_ERROR_PX,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        self.last_diagnostics["pnp_ransac_ms"] = elapsed_ms(pnp_started)

        inlier_count = 0 if inliers is None else len(inliers)
        self.last_diagnostics["inliers"] = inlier_count
        self.last_diagnostics["pnp_inlier_ratio"] = (
            inlier_count / len(map_points)
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

        R_map_to_camera = cv2.Rodrigues(rvec)[0]
        self.update_landmark_quality(
            visible_landmark_ids,
            visible_map_points,
            matches,
            inlier_indices,
            R_map_to_camera,
            tvec.reshape(3),
            current_features["image_size"],
        )
        inlier_mask = np.zeros(len(image_points), dtype=bool)
        inlier_mask[inlier_indices] = True
        unmatched_mask = np.ones(len(current_keypoints), dtype=bool)
        unmatched_mask[list(matched_current_indices)] = False
        return {
            "R": R_map_to_camera,
            "t": tvec.reshape(3),
            "inliers": len(inlier_indices),
            "inlier_map_points": map_points[inlier_indices],
            "inlier_image_points": image_points[inlier_indices],
            "inlier_landmark_ids": landmark_ids[inlier_indices],
            "inlier_current_indices": current_feature_indices[inlier_indices],
            "new_feature_indices": np.flatnonzero(unmatched_mask),
            "outlier_points": image_points[~inlier_mask],
        }

    def reset_diagnostics(self, feature_count, tracking_method):
        self.last_diagnostics = {
            "tracking_method": tracking_method,
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
            "map_expansion_coverage_threshold": (
                self.map_expansion_min_coverage_ratio
            ),
            "initialization_frames": 0,
            "initialization_candidates": 0,
            "initialization_confirmed": 0,
            "initialization_matches": 0,
            "initialization_points": np.empty((0, 2)),
            "initialization_aruco_detected": 0,
            "feature_extraction_ms": np.nan,
            "aruco_pose_ms": np.nan,
            "global_map_projection_ms": np.nan,
            "lightglue_ms": np.nan,
            "optical_flow_ms": np.nan,
            "pnp_ransac_ms": np.nan,
            "pnp_refine_ms": np.nan,
            "map_coverage_ms": np.nan,
            "map_update_ms": np.nan,
        }

    def track(self, frame):
        feature_extraction_started = start_timer()
        features = self.extract_features(frame)
        feature_extraction_ms = elapsed_ms(feature_extraction_started)
        feature_count = features["keypoints"].shape[1]
        self.reset_diagnostics(feature_count, "lightglue")
        self.last_diagnostics["feature_extraction_ms"] = (
            feature_extraction_ms
        )

        if not self.keyframes:
            aruco_started = start_timer()
            initial_pose = self.find_initial_pose(frame)
            self.last_diagnostics["aruco_pose_ms"] = elapsed_ms(
                aruco_started
            )
            if initial_pose is None:
                if self.initialization is not None:
                    self.last_diagnostics["initialization_frames"] = (
                        self.initialization["frames"]
                    )
                    self.last_diagnostics["initialization_candidates"] = int(
                        np.isfinite(
                            self.initialization["map_points"]
                        ).all(axis=1).sum()
                    )
                    self.last_diagnostics["initialization_confirmed"] = int(
                        (
                            self.initialization["observations"]
                            >= INITIALIZATION_MIN_OBSERVATIONS
                        ).sum()
                    )
                return None

            R_map_to_camera, t_map_to_camera = initial_pose
            self.last_diagnostics["initialization_aruco_detected"] = 1
            if self.initialization is None:
                self.start_initialization(
                    features,
                    R_map_to_camera,
                    t_map_to_camera,
                )
                return None

            result = self.update_initialization(
                features,
                R_map_to_camera,
                t_map_to_camera,
            )
            if result is not None:
                self.R_map_to_camera = result["R"]
                self.t_map_to_camera = result["t"]
            return result

        result = self.match_global_map(features)

        if result is None:
            return None

        self.R_map_to_camera = result["R"]
        self.t_map_to_camera = result["t"]
        coverage_started = start_timer()
        (
            result["visible_landmarks"],
            result["map_coverage_ratio"],
        ) = self.measure_map_coverage(
            features["image_size"]
        )
        self.last_diagnostics["map_coverage_ms"] = elapsed_ms(
            coverage_started
        )
        self.map_coverage_ratio = result["map_coverage_ratio"]
        self.last_diagnostics["visible_landmarks"] = result[
            "visible_landmarks"
        ]
        self.last_diagnostics["map_coverage_ratio"] = result[
            "map_coverage_ratio"
        ]
        result["nearby_associations"] = 0

        if self.should_add_keyframe(result):
            map_update_started = start_timer()
            map_update = self.add_keyframe(
                features,
                result["R"],
                result["t"],
                result["inlier_current_indices"],
                result["inlier_landmark_ids"],
                result["new_feature_indices"],
            )
            self.last_diagnostics["map_update_ms"] = elapsed_ms(
                map_update_started
            )
            result["nearby_associations"] = map_update[
                "nearby_associations"
            ]
            self.last_diagnostics.update(map_update)

        return result
