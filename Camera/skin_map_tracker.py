import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from scipy.spatial import cKDTree


DEVICE = "cuda"
MAX_KEYPOINTS = 2048
FEATURE_ROI_BOTTOM_FRACTION = 0.70
MIN_MATCHES = 30
MIN_INLIERS = 20
PNP_REPROJECTION_ERROR_PX = 3.0
LOCAL_KEYFRAMES = 5
NEW_MAP_POINT_MIN_DISTANCE_MM = 1.0
LANDMARK_ASSOCIATION_DISTANCE_MM = 1.0
LANDMARK_ASSOCIATION_REPROJECTION_ERROR_PX = 3.0
LANDMARK_DESCRIPTOR_SIMILARITY = 0.80
ARUCO_ID = 7
ARUCO_SIZE_MM = 20.0
INITIALIZATION_FRAMES = 5
INITIALIZATION_MIN_OBSERVATIONS = 3
INITIALIZATION_MIN_LANDMARKS = 60
INITIALIZATION_POINT_MAX_DISTANCE_MM = 1.5


def frame_to_tensor(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return image.to(DEVICE)


class SkinMapTracker:
    def __init__(
        self,
        camera_matrix,
        distortion,
        keyframe_interval,
        mask_aruco_features,
    ):
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.keyframe_interval = keyframe_interval
        self.mask_aruco_features = mask_aruco_features

        self.extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        self.matcher = LightGlue(features="superpoint").eval().to(DEVICE)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary)

        self.keyframes = []
        self.landmarks = {}
        self.next_landmark_id = 0
        self.last_diagnostics = {}
        self.initialization = None
        self.frame_index = -1
        self.last_keyframe_frame = -1

    def extract_features(self, frame):
        with torch.inference_mode():
            features = self.extractor.extract(frame_to_tensor(frame))

        roi_top = frame.shape[0] * (1.0 - FEATURE_ROI_BOTTOM_FRACTION)
        keep = features["keypoints"][0, :, 1] >= roi_top

        if self.mask_aruco_features:
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
            output = self.matcher(
                {
                    "image0": initialization["features"],
                    "image1": features,
                }
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
        map_update = self.add_keyframe(
            initialization["features"],
            initialization["R"],
            initialization["t"],
            no_known_features,
            no_known_features,
            confirmed_indices,
        )
        self.initialization = None
        self.last_diagnostics["keyframe_added"] = 1
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
        }
        return landmark_id

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
                LANDMARK_ASSOCIATION_DISTANCE_MM,
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
                    > LANDMARK_ASSOCIATION_REPROJECTION_ERROR_PX
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

    def update_covisibility(self, keyframe_id):
        keyframe = self.keyframes[keyframe_id]
        shared_landmarks = {}

        for landmark_id in np.unique(keyframe["landmark_ids"]):
            if landmark_id < 0:
                continue
            for other_keyframe_id in self.landmarks[landmark_id]["observations"]:
                if other_keyframe_id == keyframe_id:
                    continue
                shared_landmarks[other_keyframe_id] = (
                    shared_landmarks.get(other_keyframe_id, 0) + 1
                )

        keyframe["covisibility"] = shared_landmarks
        for other_keyframe_id, count in shared_landmarks.items():
            self.keyframes[other_keyframe_id]["covisibility"][keyframe_id] = count

    def local_keyframes(self):
        reference_id = len(self.keyframes) - 1
        reference = self.keyframes[reference_id]
        keyframe_ids = [reference_id]

        covisible = sorted(
            reference["covisibility"],
            key=reference["covisibility"].get,
            reverse=True,
        )
        for keyframe_id in covisible:
            if len(keyframe_ids) == LOCAL_KEYFRAMES:
                break
            keyframe_ids.append(keyframe_id)

        for keyframe_id in range(reference_id - 1, -1, -1):
            if len(keyframe_ids) == LOCAL_KEYFRAMES:
                break
            if keyframe_id not in keyframe_ids:
                keyframe_ids.append(keyframe_id)

        return [self.keyframes[keyframe_id] for keyframe_id in keyframe_ids]

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
            new_points[:, :2] / NEW_MAP_POINT_MIN_DISTANCE_MM
        ).astype(np.int32)
        _, unique_indices = np.unique(grid_cells, axis=0, return_index=True)
        new_feature_indices = new_feature_indices[unique_indices]
        new_points = new_points[unique_indices]

        camera_rotation = R_map_to_camera.T
        camera_position = -camera_rotation @ t_map_to_camera.reshape(3)
        keyframe = {
            "features": features,
            "landmark_ids": landmark_ids,
            "camera_position": camera_position,
            "camera_rotation": camera_rotation,
            "covisibility": {},
        }
        self.keyframes.append(keyframe)
        self.last_keyframe_frame = self.frame_index

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

        self.update_covisibility(keyframe_id)
        return {
            "nearby_associations": len(nearby_associations),
            "new_landmarks": len(new_points),
        }

    def should_add_keyframe(self):
        return (
            self.frame_index - self.last_keyframe_frame
            >= self.keyframe_interval
        )

    def match_local_map(self, current_features):
        current_keypoints = rbd(current_features)["keypoints"].detach().cpu().numpy()
        candidates = []
        matched_current_indices = set()

        for keyframe in self.local_keyframes():
            with torch.inference_mode():
                output = self.matcher(
                    {
                        "image0": keyframe["features"],
                        "image1": current_features,
                    }
                )

            output = rbd(output)
            matches = output["matches"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()

            for match, score in zip(matches, scores):
                keyframe_feature_index = match[0]
                current_feature_index = match[1]
                landmark_id = keyframe["landmark_ids"][keyframe_feature_index]
                if landmark_id < 0:
                    continue
                candidates.append(
                    (float(score), int(landmark_id), int(current_feature_index))
                )
                matched_current_indices.add(int(current_feature_index))

        candidates.sort(reverse=True)
        correspondences = []
        used_landmarks = set()
        used_current_features = set()
        for score, landmark_id, current_feature_index in candidates:
            if landmark_id in used_landmarks:
                continue
            if current_feature_index in used_current_features:
                continue
            correspondences.append((landmark_id, current_feature_index))
            used_landmarks.add(landmark_id)
            used_current_features.add(current_feature_index)

        new_features = len(current_keypoints) - len(matched_current_indices)
        self.last_diagnostics["matches"] = len(correspondences)
        self.last_diagnostics["new_features"] = new_features

        if len(correspondences) < MIN_MATCHES:
            return None

        landmark_ids = np.array(
            [correspondence[0] for correspondence in correspondences],
            dtype=np.int64,
        )
        current_feature_indices = np.array(
            [correspondence[1] for correspondence in correspondences],
            dtype=np.int64,
        )
        map_points = np.ascontiguousarray(
            [self.landmarks[landmark_id]["position"] for landmark_id in landmark_ids],
            dtype=np.float64,
        )
        image_points = np.ascontiguousarray(
            current_keypoints[current_feature_indices],
            dtype=np.float64,
        )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            map_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            iterationsCount=200,
            reprojectionError=PNP_REPROJECTION_ERROR_PX,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        inlier_count = 0 if inliers is None else len(inliers)
        self.last_diagnostics["inliers"] = inlier_count
        self.last_diagnostics["pnp_inlier_ratio"] = (
            inlier_count / len(map_points)
        )

        if not success or inlier_count < MIN_INLIERS:
            return None

        inlier_indices = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(
            map_points[inlier_indices],
            image_points[inlier_indices],
            self.camera_matrix,
            self.distortion,
            rvec,
            tvec,
        )

        R_map_to_camera = cv2.Rodrigues(rvec)[0]
        inlier_mask = np.zeros(len(image_points), dtype=bool)
        inlier_mask[inlier_indices] = True
        unmatched_mask = np.ones(len(current_keypoints), dtype=bool)
        unmatched_mask[list(matched_current_indices)] = False
        return {
            "R": R_map_to_camera,
            "t": tvec.reshape(3),
            "inliers": len(inlier_indices),
            "inlier_map_points": map_points[inlier_indices],
            "inlier_landmark_ids": landmark_ids[inlier_indices],
            "inlier_current_indices": current_feature_indices[inlier_indices],
            "new_feature_indices": np.flatnonzero(unmatched_mask),
            "outlier_points": image_points[~inlier_mask],
        }

    def track(self, frame):
        self.frame_index += 1
        features = self.extract_features(frame)
        feature_count = features["keypoints"].shape[1]
        self.last_diagnostics = {
            "matches": 0,
            "inliers": 0,
            "pnp_inlier_ratio": np.nan,
            "new_features": feature_count,
            "keyframe_added": 0,
            "nearby_associations": 0,
            "new_landmarks": 0,
            "initialization_frames": 0,
            "initialization_candidates": 0,
            "initialization_confirmed": 0,
            "initialization_matches": 0,
            "initialization_points": np.empty((0, 2)),
            "initialization_aruco_detected": 0,
        }

        if not self.keyframes:
            initial_pose = self.find_initial_pose(frame)
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

            return self.update_initialization(
                features,
                R_map_to_camera,
                t_map_to_camera,
            )

        result = self.match_local_map(features)

        if result is None:
            return None

        result["nearby_associations"] = 0

        if self.should_add_keyframe():
            map_update = self.add_keyframe(
                features,
                result["R"],
                result["t"],
                result["inlier_current_indices"],
                result["inlier_landmark_ids"],
                result["new_feature_indices"],
            )
            result["nearby_associations"] = map_update[
                "nearby_associations"
            ]
            self.last_diagnostics["keyframe_added"] = 1
            self.last_diagnostics.update(map_update)

        return result
