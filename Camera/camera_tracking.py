import csv
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".venv" / "torch_cache")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RECORDING_NAME = "Speed-3_2026-07-17_14.38.43"
CAMERA_NAME = "cam2"
CAMERA_CALIBRATION = "camera_jabra_1920_1080"
CAMERA_MAP_TO_DOBOT = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

DEVICE = "cuda"
MAX_KEYPOINTS = 2048
FEATURE_ROI_BOTTOM_FRACTION = 0.70
MIN_MATCHES = 30
MIN_INLIERS = 20
PNP_REPROJECTION_ERROR_PX = 3.0
LOCAL_KEYFRAMES = 5
KEYFRAME_TRANSLATION_MM = 8.0
KEYFRAME_ROTATION_DEG = 5.0
USE_PNP_CONDITIONS_FOR_KEYFRAMES = True
KEYFRAME_LOW_PNP_INLIERS = 60
KEYFRAME_MIN_PNP_INLIERS = 40
KEYFRAME_MIN_NEW_FEATURES = 100
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

SAVE_DIAGNOSTIC_VIDEO = True
DIAGNOSTIC_VIDEO_FPS = 1.0
SHOW_PREVIEW = False


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data2"
OUTPUT_DIR = SCRIPT_DIR / "results"
VIDEO_PATH = (
    DATA_DIR
    / "videos"
    / f"{RECORDING_NAME}_{CAMERA_NAME}.mp4"
)
GROUND_TRUTH_PATH = DATA_DIR / "dobot" / f"{RECORDING_NAME}.csv"
VIDEO_TIMESTAMP_PATH = (
    DATA_DIR
    / "video_timestamps"
    / f"{RECORDING_NAME}.csv"
)

CAMERA_MATRIX_PATH = (
    SCRIPT_DIR
    / "calibrations"
    / CAMERA_CALIBRATION
    / "camera_matrix.npy"
)
DISTORTION_PATH = (
    SCRIPT_DIR
    / "calibrations"
    / CAMERA_CALIBRATION
    / "dist_coeffs.npy"
)


def frame_to_tensor(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return image.to(DEVICE)


def camera_pose(R_map_to_camera, t_map_to_camera):
    R_camera_to_map = R_map_to_camera.T
    position = -R_camera_to_map @ t_map_to_camera.reshape(3)
    euler = Rotation.from_matrix(R_camera_to_map).as_euler("xyz", degrees=True)
    return position, euler


class SkinMapTracker:
    def __init__(self, camera_matrix, distortion):
        self.camera_matrix = camera_matrix
        self.distortion = distortion

        self.extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        self.matcher = LightGlue(features="superpoint").eval().to(DEVICE)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary)

        self.keyframes = []
        self.landmarks = {}
        self.next_landmark_id = 0
        self.last_diagnostics = {}
        self.R_map_to_camera = None
        self.t_map_to_camera = None
        self.initialization = None

    def extract_features(self, frame):
        with torch.inference_mode():
            features = self.extractor.extract(frame_to_tensor(frame))

        roi_top = frame.shape[0] * (1.0 - FEATURE_ROI_BOTTOM_FRACTION)
        keep = features["keypoints"][0, :, 1] >= roi_top
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
        frame_index,
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
            "frame": frame_index,
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

        map_update = self.add_keyframe(
            initialization["frame"],
            initialization["features"],
            initialization["R"],
            initialization["t"],
            new_feature_indices=confirmed_indices,
        )
        initial_frame = initialization["frame"]
        self.initialization = None
        self.R_map_to_camera = R_map_to_camera
        self.t_map_to_camera = t_map_to_camera
        self.last_diagnostics["keyframe_added"] = 1
        self.last_diagnostics.update(map_update)

        return {
            "R": R_map_to_camera,
            "t": t_map_to_camera,
            "matches": 0,
            "inliers": 0,
            "inlier_map_points": np.empty((0, 3)),
            "outlier_points": np.empty((0, 2)),
            "keyframe_frames": [initial_frame],
            "nearby_associations": 0,
        }

    def all_map_points(self):
        if not self.landmarks:
            return np.empty((0, 3))

        return np.array(
            [landmark["position"] for landmark in self.landmarks.values()]
        )

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
        frame_index,
        features,
        R_map_to_camera,
        t_map_to_camera,
        known_feature_indices=None,
        known_landmark_ids=None,
        new_feature_indices=None,
    ):
        feature_data = rbd(features)
        keypoints = feature_data["keypoints"].detach().cpu().numpy()
        descriptors = feature_data["descriptors"].detach().cpu().numpy()
        landmark_ids = np.full(len(keypoints), -1, dtype=np.int64)
        keyframe_id = len(self.keyframes)

        if known_feature_indices is not None:
            landmark_ids[known_feature_indices] = known_landmark_ids

        if new_feature_indices is None:
            new_feature_indices = np.arange(len(keypoints))

        new_points = self.pixels_to_skin_plane(
            keypoints[new_feature_indices],
            R_map_to_camera,
            t_map_to_camera,
        )
        valid = np.isfinite(new_points).all(axis=1)
        new_feature_indices = new_feature_indices[valid]
        new_points = new_points[valid]

        occupied_landmark_ids = set()
        if known_landmark_ids is not None:
            occupied_landmark_ids.update(
                int(landmark_id) for landmark_id in known_landmark_ids
            )

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

        if nearby_associations:
            remains_new = np.array(
                [
                    feature_index not in nearby_associations
                    for feature_index in new_feature_indices
                ]
            )
            new_feature_indices = new_feature_indices[remains_new]
            new_points = new_points[remains_new]

        if len(new_points):
            grid_cells = np.rint(
                new_points[:, :2] / NEW_MAP_POINT_MIN_DISTANCE_MM
            ).astype(np.int32)
            _, unique_indices = np.unique(
                grid_cells,
                axis=0,
                return_index=True,
            )
            new_feature_indices = new_feature_indices[unique_indices]
            new_points = new_points[unique_indices]

        camera_rotation = R_map_to_camera.T
        camera_position = -camera_rotation @ t_map_to_camera.reshape(3)
        keyframe = {
            "id": keyframe_id,
            "frame": frame_index,
            "features": features,
            "landmark_ids": landmark_ids,
            "camera_position": camera_position,
            "camera_rotation": camera_rotation,
            "covisibility": {},
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

        self.update_covisibility(keyframe_id)
        return {
            "nearby_associations": len(nearby_associations),
            "new_landmarks": len(new_points),
        }

    def should_add_keyframe(self, result):
        last_keyframe = self.keyframes[-1]
        camera_rotation = result["R"].T
        camera_position = -camera_rotation @ result["t"]

        translation = np.linalg.norm(
            camera_position - last_keyframe["camera_position"]
        )
        relative_rotation = (
            last_keyframe["camera_rotation"].T @ camera_rotation
        )
        rotation = np.degrees(
            Rotation.from_matrix(relative_rotation).magnitude()
        )
        pnp_inliers = result["inliers"]

        enough_new_features = (
            len(result["new_feature_indices"]) >= KEYFRAME_MIN_NEW_FEATURES
        )
        viewpoint_changed = (
            translation >= KEYFRAME_TRANSLATION_MM
            or rotation >= KEYFRAME_ROTATION_DEG
        )

        if USE_PNP_CONDITIONS_FOR_KEYFRAMES:
            pose_is_reliable = pnp_inliers >= KEYFRAME_MIN_PNP_INLIERS
            viewpoint_changed = (
                viewpoint_changed
                or pnp_inliers <= KEYFRAME_LOW_PNP_INLIERS
            )
        else:
            pose_is_reliable = True

        return pose_is_reliable and enough_new_features and viewpoint_changed

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

        if not success or inliers is None or len(inliers) < MIN_INLIERS:
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
            "matches": len(map_points),
            "inliers": len(inlier_indices),
            "inlier_map_points": map_points[inlier_indices],
            "inlier_landmark_ids": landmark_ids[inlier_indices],
            "inlier_current_indices": current_feature_indices[inlier_indices],
            "new_feature_indices": np.flatnonzero(unmatched_mask),
            "outlier_points": image_points[~inlier_mask],
            "keyframe_frames": [
                keyframe["frame"] for keyframe in self.local_keyframes()
            ],
        }

    def track(self, frame_index, frame):
        features = self.extract_features(frame)
        feature_count = len(
            rbd(features)["keypoints"].detach().cpu().numpy()
        )
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
                    frame_index,
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

        self.R_map_to_camera = result["R"]
        self.t_map_to_camera = result["t"]
        result["nearby_associations"] = 0

        if self.should_add_keyframe(result):
            map_update = self.add_keyframe(
                frame_index,
                features,
                self.R_map_to_camera,
                self.t_map_to_camera,
                known_feature_indices=result["inlier_current_indices"],
                known_landmark_ids=result["inlier_landmark_ids"],
                new_feature_indices=result["new_feature_indices"],
            )
            result["nearby_associations"] = map_update[
                "nearby_associations"
            ]
            self.last_diagnostics["keyframe_added"] = 1
            self.last_diagnostics.update(map_update)

        return result


def project_map_points(map_points, result, tracker, frame_shape):
    if len(map_points) == 0:
        return np.empty((0, 2))

    rvec = cv2.Rodrigues(result["R"])[0]
    projected, _ = cv2.projectPoints(
        map_points,
        rvec,
        result["t"],
        tracker.camera_matrix,
        tracker.distortion,
    )
    projected = projected.reshape(-1, 2)

    camera_points = (result["R"] @ map_points.T).T + result["t"]
    height, width = frame_shape[:2]
    visible = camera_points[:, 2] > 0.0
    visible &= projected[:, 0] >= 0.0
    visible &= projected[:, 0] < width
    visible &= projected[:, 1] >= height * (
        1.0 - FEATURE_ROI_BOTTOM_FRACTION
    )
    visible &= projected[:, 1] < height
    return projected[visible]


def map_points_for_display(tracker):
    return tracker.all_map_points()


def diagnostic_frame(frame, tracker, result, relative_positions):
    output = frame.copy()
    roi_top = round(frame.shape[0] * (1.0 - FEATURE_ROI_BOTTOM_FRACTION))
    output[:roi_top] = 0
    cv2.line(
        output,
        (0, roi_top),
        (frame.shape[1] - 1, roi_top),
        (255, 180, 0),
        2,
    )
    cv2.putText(
        output,
        "Feature ROI below this line",
        (12, roi_top + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        2,
    )
    tracked = result is not None
    initializing = not tracker.keyframes and tracker.initialization is not None
    if tracked:
        color = (40, 200, 40)
        label = "TRACKING"
    elif initializing:
        color = (0, 180, 255)
        label = "INITIALIZING"
    else:
        color = (0, 180, 255)
        label = "WAITING FOR ARUCO"

    if tracked:
        projected_map_points = project_map_points(
            map_points_for_display(tracker),
            result,
            tracker,
            frame.shape,
        )
        projected_inliers = project_map_points(
            result["inlier_map_points"],
            result,
            tracker,
            frame.shape,
        )

        for point in projected_map_points:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                1,
                (0, 255, 255),
                -1,
            )

        for point in projected_inliers:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 255, 0),
                -1,
            )

        for point in result["outlier_points"]:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 0, 255),
                -1,
            )

        position = relative_positions[-1]
        cv2.putText(
            output,
            f"XYZ: {position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f} mm",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.rectangle(output, (8, 84), (300, 181), (0, 0, 0), -1)
        legend = [
            (
                f"Map not detected: {len(projected_map_points) - len(projected_inliers)}",
                (0, 255, 255),
            ),
            (f"PnP inliers: {result['inliers']}", (0, 255, 0)),
            (f"PnP outliers: {len(result['outlier_points'])}", (0, 0, 255)),
            (
                f"Nearby associations: {result['nearby_associations']}",
                (255, 180, 0),
            ),
        ]
        for index, (text, text_color) in enumerate(legend):
            cv2.putText(
                output,
                text,
                (15, 105 + 23 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2,
            )

    elif initializing:
        diagnostics = tracker.last_diagnostics
        for point in diagnostics["initialization_points"]:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 255, 0),
                -1,
            )

        cv2.putText(
            output,
            (
                "ArUco: "
                + (
                    "detected"
                    if diagnostics["initialization_aruco_detected"]
                    else "not detected"
                )
            ),
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            output,
            (
                f"Frames: {diagnostics['initialization_frames']}"
                f"/{INITIALIZATION_FRAMES} | confirmed: "
                f"{diagnostics['initialization_confirmed']}"
                f"/{INITIALIZATION_MIN_LANDMARKS}"
            ),
            (12, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            output,
            (
                f"Candidates: {diagnostics['initialization_candidates']}"
                f" | consistent now: "
                f"{diagnostics['initialization_matches']}"
            ),
            (12, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    cv2.putText(output, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        output,
        f"keyframes: {len(tracker.keyframes)} | landmarks: {len(tracker.landmarks)}",
        (12, output.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return output


def save_results_csv(rows, path):
    fields = [
        "frame",
        "time_s",
        "timestamp",
        "x_mm",
        "y_mm",
        "z_mm",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "matches",
        "inliers",
        "pnp_inlier_ratio",
        "new_features",
        "nearby_associations",
        "new_landmarks",
        "initialization_frames",
        "initialization_candidates",
        "initialization_confirmed",
        "initialization_matches",
        "landmarks",
        "keyframe_added",
        "keyframes",
        "tracked",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_ground_truth(path):
    timestamps = []
    positions = []
    orientations = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            timestamps.append(float(row["timestamp"]))
            positions.append(
                [float(row[axis]) for axis in ("x", "y", "z")]
            )
            orientations.append(
                [float(row[axis]) for axis in ("roll", "pitch", "yaw")]
            )

    positions = np.array(positions)
    positions = positions - positions[0]
    return np.array(timestamps), positions, np.array(orientations)


def load_video_start_timestamp(path):
    with path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    return float(row["start_timestamp"])


def save_comparison_figure(
    frames,
    ground_truth,
    estimate,
    component_errors,
    overall_errors,
    component_names,
    unit,
    overall_name,
    output_path,
    title,
):
    figure = plt.figure(figsize=(14, 14))
    grid = figure.add_gridspec(4, 2)

    for component in range(3):
        comparison_axis = figure.add_subplot(grid[component, 0])
        comparison_axis.plot(
            frames,
            ground_truth[:, component],
            "k--",
            label="GT",
        )
        comparison_axis.plot(
            frames,
            estimate[:, component],
            color="tab:blue",
            label="Camera",
        )
        comparison_axis.set_ylabel(
            f"{component_names[component]} [{unit}]"
        )
        comparison_axis.grid(True)
        comparison_axis.legend()

        component_rmse = np.sqrt(
            np.nanmean(component_errors[:, component] ** 2)
        )
        error_axis = figure.add_subplot(grid[component, 1])
        error_axis.plot(
            frames,
            np.abs(component_errors[:, component]),
            color="red",
        )
        error_axis.set_title(
            f"{component_names[component]} error | "
            f"RMSE: {component_rmse:.2f} {unit}"
        )
        error_axis.set_ylabel(f"Absolute error [{unit}]")
        error_axis.grid(True)

    overall_mae = np.nanmean(overall_errors)
    overall_rmse = np.sqrt(np.nanmean(overall_errors**2))
    overall_axis = figure.add_subplot(grid[3, :])
    overall_axis.plot(frames, overall_errors, color="red")
    overall_axis.set_title(
        f"{overall_name} | MAE: {overall_mae:.2f} {unit} | "
        f"RMSE: {overall_rmse:.2f} {unit}"
    )
    overall_axis.set_ylabel(f"Error [{unit}]")
    overall_axis.set_xlabel("Corrected camera time [s]")
    overall_axis.grid(True)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return overall_rmse


def create_comparison_plots(
    rows,
    gt_path,
    position_output_path,
    orientation_output_path,
):
    camera_time = np.array([row["timestamp"] for row in rows])
    estimate = np.array(
        [[row["x_mm"], row["y_mm"], row["z_mm"]] for row in rows],
        dtype=float,
    )
    estimate_euler = np.array(
        [[row["roll_deg"], row["pitch_deg"], row["yaw_deg"]] for row in rows],
        dtype=float,
    )
    gt_time, gt_positions, gt_euler = load_ground_truth(gt_path)

    gt = np.column_stack(
        [
            np.interp(camera_time, gt_time, gt_positions[:, axis])
            for axis in range(3)
        ]
    )
    unwrapped_gt_euler = np.degrees(
        np.unwrap(np.radians(gt_euler), axis=0)
    )
    gt_euler = np.column_stack(
        [
            np.interp(camera_time, gt_time, unwrapped_gt_euler[:, axis])
            for axis in range(3)
        ]
    )

    within_ground_truth = camera_time >= gt_time[0]
    within_ground_truth &= camera_time <= gt_time[-1]
    camera_time = camera_time[within_ground_truth]
    estimate = estimate[within_ground_truth]
    estimate_euler = estimate_euler[within_ground_truth]
    gt = gt[within_ground_truth]
    gt_euler = gt_euler[within_ground_truth]

    valid = np.isfinite(estimate).all(axis=1)
    valid &= np.isfinite(estimate_euler).all(axis=1)
    plot_time = camera_time - min(camera_time[0], gt_time[0])

    position_component_errors = np.full_like(estimate, np.nan)
    position_component_errors[valid] = estimate[valid] - gt[valid]
    position_errors = np.full(len(estimate), np.nan)
    position_errors[valid] = np.linalg.norm(
        position_component_errors[valid],
        axis=1,
    )

    estimate_rotations = Rotation.from_euler(
        "xyz", estimate_euler[valid], degrees=True
    ).as_matrix()
    gt_rotations = Rotation.from_euler(
        "xyz", gt_euler[valid], degrees=True
    ).as_matrix()
    valid_orientation_errors = (
        estimate_euler[valid] - gt_euler[valid] + 180.0
    ) % 360.0 - 180.0
    orientation_component_errors = np.full_like(estimate_euler, np.nan)
    orientation_component_errors[valid] = valid_orientation_errors
    relative_rotations = np.transpose(gt_rotations, (0, 2, 1)) @ estimate_rotations
    valid_angular_errors = np.degrees(
        Rotation.from_matrix(relative_rotations).magnitude()
    )
    angular_errors = np.full(len(estimate), np.nan)
    angular_errors[valid] = valid_angular_errors
    tracking_coverage = 100.0 * np.mean(valid)

    position_rmse = save_comparison_figure(
        plot_time,
        gt,
        estimate,
        position_component_errors,
        position_errors,
        ["X", "Y", "Z"],
        "mm",
        "Euclidean distance on tracked frames",
        position_output_path,
        f"{RECORDING_NAME}: camera position vs GT\n"
        "Data2 timestamps synchronized to Dobot | "
        f"tracked: {tracking_coverage:.1f}%",
    )

    orientation_rmse = save_comparison_figure(
        plot_time,
        gt_euler,
        estimate_euler,
        orientation_component_errors,
        angular_errors,
        ["Roll", "Pitch", "Yaw"],
        "deg",
        "Angular distance on tracked frames",
        orientation_output_path,
        f"{RECORDING_NAME}: camera orientation vs GT\n"
        "Data2 timestamps synchronized to Dobot | "
        f"tracked: {tracking_coverage:.1f}%",
    )
    return position_rmse, orientation_rmse


def save_mapping_diagnostics(rows, output_path):
    frames = np.array([row["frame"] for row in rows])
    matches = np.array([row["matches"] for row in rows])
    inliers = np.array([row["inliers"] for row in rows])
    new_features = np.array([row["new_features"] for row in rows])
    nearby_associations = np.array(
        [row["nearby_associations"] for row in rows]
    )
    new_landmarks = np.array([row["new_landmarks"] for row in rows])
    landmarks = np.array([row["landmarks"] for row in rows])
    keyframe_added = np.array([row["keyframe_added"] for row in rows]) == 1

    figure, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)

    axes[0].plot(frames, matches, label="PnP correspondences")
    axes[0].plot(frames, inliers, label="PnP inliers")
    axes[0].set_ylabel("Points")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(frames, inliers, color="tab:green")
    if USE_PNP_CONDITIONS_FOR_KEYFRAMES:
        axes[1].axhline(
            KEYFRAME_MIN_PNP_INLIERS,
            color="red",
            linestyle="--",
            label="Reliable pose threshold",
        )
        axes[1].axhline(
            KEYFRAME_LOW_PNP_INLIERS,
            color="orange",
            linestyle="--",
            label="Map expansion threshold",
        )
    axes[1].set_ylabel("PnP inliers")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(frames, new_features, label="New features")
    axes[2].plot(
        frames,
        nearby_associations,
        label="Associated with nearby landmarks",
    )
    axes[2].axhline(
        KEYFRAME_MIN_NEW_FEATURES,
        color="orange",
        linestyle="--",
        label="New feature threshold",
    )
    axes[2].scatter(
        frames[keyframe_added],
        new_features[keyframe_added],
        color="red",
        s=18,
        label="Keyframe added",
        zorder=3,
    )
    axes[2].set_ylabel("Features")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].bar(
        frames,
        new_landmarks,
        width=1.0,
        color="tab:blue",
        label="New landmarks",
    )
    axes[3].set_ylabel("New landmarks")
    axes[3].set_xlabel("Frame")
    axes[3].grid(True)
    axes[3].legend(loc="upper left")

    landmarks_axis = axes[3].twinx()
    landmarks_axis.plot(
        frames,
        landmarks,
        color="black",
        label="Total landmarks",
    )
    landmarks_axis.set_ylabel("Total landmarks")
    landmarks_axis.legend(loc="upper right")

    figure.suptitle(f"{RECORDING_NAME}: map expansion diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    if not torch.cuda.is_available() and DEVICE == "cuda":
        raise RuntimeError("CUDA is not available in the project .venv")

    OUTPUT_DIR.mkdir(exist_ok=True)
    video_start_timestamp = load_video_start_timestamp(
        VIDEO_TIMESTAMP_PATH
    )

    camera_matrix = np.load(CAMERA_MATRIX_PATH)
    distortion = np.load(DISTORTION_PATH)
    tracker = SkinMapTracker(camera_matrix, distortion)

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(VIDEO_PATH)

    input_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = None
    if SAVE_DIAGNOSTIC_VIDEO:
        video_path_output = OUTPUT_DIR / f"{RECORDING_NAME}_tracking.mp4"
        video_writer = cv2.VideoWriter(
            str(video_path_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            DIAGNOSTIC_VIDEO_FPS,
            (width, height),
        )

    rows = []
    positions = []
    initial_position = None
    initial_rotation = None
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        result = tracker.track(frame_index, frame)
        diagnostics = tracker.last_diagnostics
        if result is None:
            position = np.full(3, np.nan)
            euler = np.full(3, np.nan)
        else:
            absolute_position, _ = camera_pose(result["R"], result["t"])
            camera_rotation = result["R"].T

            if initial_position is None:
                initial_position = absolute_position.copy()
                initial_rotation = camera_rotation.copy()

            position = CAMERA_MAP_TO_DOBOT @ (
                absolute_position - initial_position
            )
            relative_rotation = initial_rotation.T @ camera_rotation
            euler = Rotation.from_matrix(relative_rotation).as_euler(
                "xyz", degrees=True
            )

        matches = diagnostics["matches"]
        inliers = diagnostics["inliers"]

        positions.append(position)
        time_s = frame_index / input_fps
        rows.append(
            {
                "frame": frame_index,
                "time_s": time_s,
                "timestamp": video_start_timestamp + time_s,
                "x_mm": position[0],
                "y_mm": position[1],
                "z_mm": position[2],
                "roll_deg": euler[0],
                "pitch_deg": euler[1],
                "yaw_deg": euler[2],
                "matches": matches,
                "inliers": inliers,
                "pnp_inlier_ratio": diagnostics["pnp_inlier_ratio"],
                "new_features": diagnostics["new_features"],
                "nearby_associations": diagnostics["nearby_associations"],
                "new_landmarks": diagnostics["new_landmarks"],
                "initialization_frames": diagnostics[
                    "initialization_frames"
                ],
                "initialization_candidates": diagnostics[
                    "initialization_candidates"
                ],
                "initialization_confirmed": diagnostics[
                    "initialization_confirmed"
                ],
                "initialization_matches": diagnostics[
                    "initialization_matches"
                ],
                "landmarks": len(tracker.landmarks),
                "keyframe_added": diagnostics["keyframe_added"],
                "keyframes": len(tracker.keyframes),
                "tracked": int(result is not None),
            }
        )

        if SAVE_DIAGNOSTIC_VIDEO or SHOW_PREVIEW:
            preview = diagnostic_frame(
                frame,
                tracker,
                result,
                positions,
            )
            if video_writer is not None:
                video_writer.write(preview)
            if SHOW_PREVIEW:
                cv2.imshow("Camera skin tracking", preview)
                if cv2.waitKey(1) == 27:
                    break

        if frame_index % 100 == 0:
            print(f"Frame {frame_index}/{frame_count}, keyframes: {len(tracker.keyframes)}")
        frame_index += 1

    capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    csv_path = OUTPUT_DIR / f"{RECORDING_NAME}_camera.csv"
    position_plot_path = (
        OUTPUT_DIR / f"{RECORDING_NAME}_camera_vs_gt_position.png"
    )
    orientation_plot_path = (
        OUTPUT_DIR / f"{RECORDING_NAME}_camera_vs_gt_orientation.png"
    )
    diagnostics_plot_path = (
        OUTPUT_DIR / f"{RECORDING_NAME}_mapping_diagnostics.png"
    )
    save_results_csv(rows, csv_path)
    save_mapping_diagnostics(rows, diagnostics_plot_path)
    position_rmse, orientation_rmse = create_comparison_plots(
        rows,
        GROUND_TRUTH_PATH,
        position_plot_path,
        orientation_plot_path,
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {position_plot_path}")
    print(f"Saved: {orientation_plot_path}")
    print(f"Saved: {diagnostics_plot_path}")
    if SAVE_DIAGNOSTIC_VIDEO:
        print(f"Saved: {video_path_output}")
    print(f"Position RMSE: {position_rmse:.2f} mm")
    print(f"Orientation RMSE: {orientation_rmse:.2f} deg")


if __name__ == "__main__":
    main()
