import cv2
import numpy as np
from scipy.spatial import cKDTree


FEATURE_ROI_BOTTOM_FRACTION = 0.70
MAX_CORNERS = 576
CORNER_QUALITY = 0.01
CORNER_MIN_DISTANCE_PX = 4
CORNER_BLOCK_SIZE = 7

LK_WINDOW_SIZE = (21, 21)
LK_PYRAMID_LEVELS = 3
LK_FORWARD_BACKWARD_ERROR_PX = 1.0

MIN_PNP_INLIERS = 30
PNP_REPROJECTION_ERROR_PX = 3.0

KEYFRAME_SEARCH_TRIGGER_INLIERS = 50
KEYFRAME_SEARCH_TRIGGER_FRAMES = 2
KEYFRAME_CANDIDATE_MIN_INLIERS = 70

LANDMARK_ASSOCIATION_DISTANCE_MM = 1.0
ARUCO_ID = 7
ARUCO_SIZE_MM = 20.0
ARUCO_MASK_MARGIN_MM = 10.0


class OpticalFlowMapTracker:
    def __init__(self, camera_matrix, distortion):
        self.camera_matrix = camera_matrix
        self.distortion = distortion

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary)

        self.landmarks = {}
        self.keyframes = []
        self.next_landmark_id = 0
        self.last_diagnostics = {}
        self.low_inlier_streak = 0
        self.keyframe_search_active = False

    def grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        return cv2.Rodrigues(rvec)[0], tvec.reshape(3)

    def detect_features(self, gray, excluded_points):
        mask = np.zeros_like(gray)
        roi_top = round(gray.shape[0] * (1.0 - FEATURE_ROI_BOTTOM_FRACTION))
        mask[roi_top:] = 255

        aruco_corners, _, _ = self.aruco_detector.detectMarkers(gray)
        mask_scale = (
            ARUCO_SIZE_MM + 2.0 * ARUCO_MASK_MARGIN_MM
        ) / ARUCO_SIZE_MM
        for corners in aruco_corners:
            corners = corners.reshape(4, 2)
            center = corners.mean(axis=0)
            expanded_corners = center + mask_scale * (corners - center)
            cv2.fillConvexPoly(
                mask,
                np.rint(expanded_corners).astype(np.int32),
                0,
            )

        for point in excluded_points:
            cv2.circle(
                mask,
                tuple(np.rint(point).astype(int)),
                CORNER_MIN_DISTANCE_PX,
                0,
                -1,
            )

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=CORNER_QUALITY,
            minDistance=CORNER_MIN_DISTANCE_PX,
            mask=mask,
            blockSize=CORNER_BLOCK_SIZE,
        )
        if points is None:
            return np.empty((0, 2), dtype=np.float32)
        return points.reshape(-1, 2)

    def pixels_to_skin_plane(self, keypoints, R_map_to_camera, t_map_to_camera):
        if len(keypoints) == 0:
            return np.empty((0, 3))

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

    def all_map_points(self):
        return np.array(list(self.landmarks.values()), dtype=float).reshape(-1, 3)

    def create_landmark(self, position):
        landmark_id = self.next_landmark_id
        self.next_landmark_id += 1
        self.landmarks[landmark_id] = position.copy()
        return landmark_id

    def create_keyframe(
        self,
        gray,
        R_map_to_camera,
        t_map_to_camera,
        known_points,
        known_landmark_ids,
    ):
        detected_points = self.detect_features(gray, known_points)
        detected_map_points = self.pixels_to_skin_plane(
            detected_points,
            R_map_to_camera,
            t_map_to_camera,
        )
        valid = np.isfinite(detected_map_points).all(axis=1)
        detected_points = detected_points[valid]
        detected_map_points = detected_map_points[valid]

        keyframe_points = list(known_points)
        keyframe_landmark_ids = list(map(int, known_landmark_ids))
        occupied_landmarks = set(keyframe_landmark_ids)
        nearby_associations = 0
        new_landmarks = 0
        new_landmark_points = []

        landmark_ids = np.array(list(self.landmarks), dtype=np.int64)
        landmark_tree = None
        if len(landmark_ids):
            landmark_positions = np.array(
                [self.landmarks[landmark_id] for landmark_id in landmark_ids]
            )
            landmark_tree = cKDTree(landmark_positions[:, :2])

        new_cells = set()
        for image_point, map_point in zip(detected_points, detected_map_points):
            landmark_id = None
            if landmark_tree is not None:
                distance, landmark_index = landmark_tree.query(map_point[:2])
                nearby_id = int(landmark_ids[landmark_index])
                if distance <= LANDMARK_ASSOCIATION_DISTANCE_MM:
                    if nearby_id not in occupied_landmarks:
                        landmark_id = nearby_id
                        nearby_associations += 1
                    else:
                        continue

            if landmark_id is None:
                cell = tuple(
                    np.rint(
                        map_point[:2] / LANDMARK_ASSOCIATION_DISTANCE_MM
                    ).astype(int)
                )
                if cell in new_cells:
                    continue
                new_cells.add(cell)
                landmark_id = self.create_landmark(map_point)
                new_landmarks += 1
                new_landmark_points.append(image_point)

            keyframe_points.append(image_point)
            keyframe_landmark_ids.append(landmark_id)
            occupied_landmarks.add(landmark_id)

        camera_rotation = R_map_to_camera.T
        camera_position = -camera_rotation @ t_map_to_camera.reshape(3)
        self.keyframes.append(
            {
                "gray": gray.copy(),
                "points": np.asarray(keyframe_points, dtype=np.float32),
                "landmark_ids": np.asarray(
                    keyframe_landmark_ids,
                    dtype=np.int64,
                ),
                "camera_position": camera_position,
                "camera_rotation": camera_rotation,
            }
        )
        return {
            "new_features": len(detected_points),
            "nearby_associations": nearby_associations,
            "new_landmarks": new_landmarks,
            "new_landmark_points": np.array(
                new_landmark_points,
                dtype=np.float32,
            ).reshape(-1, 2),
        }

    def track_keyframe_points(self, current_gray):
        keyframe = self.keyframes[-1]
        source_points = keyframe["points"].reshape(-1, 1, 2)
        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            keyframe["gray"],
            current_gray,
            source_points,
            None,
            winSize=LK_WINDOW_SIZE,
            maxLevel=LK_PYRAMID_LEVELS,
        )
        if current_points is None:
            return None

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            keyframe["gray"],
            current_points,
            None,
            winSize=LK_WINDOW_SIZE,
            maxLevel=LK_PYRAMID_LEVELS,
        )
        if backward_points is None:
            return None

        source_points = source_points.reshape(-1, 2)
        current_points = current_points.reshape(-1, 2)
        backward_points = backward_points.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(
            source_points - backward_points,
            axis=1,
        )

        height, width = current_gray.shape
        roi_top = height * (1.0 - FEATURE_ROI_BOTTOM_FRACTION)
        valid = forward_status.ravel() == 1
        valid &= backward_status.ravel() == 1
        valid &= forward_backward_error <= LK_FORWARD_BACKWARD_ERROR_PX
        valid &= current_points[:, 0] >= 0.0
        valid &= current_points[:, 0] < width
        valid &= current_points[:, 1] >= roi_top
        valid &= current_points[:, 1] < height

        return (
            current_points[valid],
            keyframe["landmark_ids"][valid],
        )

    def estimate_pose(self, image_points, landmark_ids):
        statistics = {
            "flow_tracks": len(image_points),
            "inliers": 0,
            "pnp_inlier_ratio": np.nan,
        }
        if len(image_points) < MIN_PNP_INLIERS:
            return None, statistics

        map_points = np.ascontiguousarray(
            [self.landmarks[int(landmark_id)] for landmark_id in landmark_ids],
            dtype=np.float64,
        )
        image_points = np.ascontiguousarray(image_points, dtype=np.float64)
        keyframe = self.keyframes[-1]
        rvec = cv2.Rodrigues(keyframe["camera_rotation"].T)[0]
        tvec = (
            -keyframe["camera_rotation"].T @ keyframe["camera_position"]
        ).reshape(3, 1)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            map_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=PNP_REPROJECTION_ERROR_PX,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inlier_count = 0 if inliers is None else len(inliers)
        statistics["inliers"] = inlier_count
        statistics["pnp_inlier_ratio"] = (
            inlier_count / len(image_points)
        )
        if not success or inlier_count < MIN_PNP_INLIERS:
            return None, statistics

        inlier_indices = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(
            map_points[inlier_indices],
            image_points[inlier_indices],
            self.camera_matrix,
            self.distortion,
            rvec,
            tvec,
        )
        inlier_mask = np.zeros(len(image_points), dtype=bool)
        inlier_mask[inlier_indices] = True
        result = {
            "R": cv2.Rodrigues(rvec)[0],
            "t": tvec.reshape(3),
            "inliers": inlier_count,
            "inlier_map_points": map_points[inlier_indices],
            "inlier_image_points": image_points[inlier_indices],
            "inlier_landmark_ids": landmark_ids[inlier_indices],
            "outlier_points": image_points[~inlier_mask],
        }
        return result, statistics

    def update_keyframe_search(self, inliers):
        if inliers < KEYFRAME_SEARCH_TRIGGER_INLIERS:
            self.low_inlier_streak += 1
            if self.low_inlier_streak >= KEYFRAME_SEARCH_TRIGGER_FRAMES:
                self.keyframe_search_active = True
        else:
            self.low_inlier_streak = 0

    def track(self, frame):
        gray = self.grayscale(frame)
        self.last_diagnostics = {
            "flow_tracks": 0,
            "inliers": 0,
            "pnp_inlier_ratio": np.nan,
            "new_features": 0,
            "nearby_associations": 0,
            "new_landmarks": 0,
            "new_landmark_points": np.empty((0, 2)),
            "keyframe_added": 0,
        }

        if not self.keyframes:
            initial_pose = self.find_initial_pose(frame)
            if initial_pose is None:
                return None

            R_map_to_camera, t_map_to_camera = initial_pose
            no_points = np.empty((0, 2), dtype=np.float32)
            no_landmarks = np.empty(0, dtype=np.int64)
            map_update = self.create_keyframe(
                gray,
                R_map_to_camera,
                t_map_to_camera,
                no_points,
                no_landmarks,
            )
            if map_update is None:
                return None

            keyframe = self.keyframes[-1]
            self.last_diagnostics["keyframe_added"] = 1
            self.last_diagnostics.update(map_update)
            self.low_inlier_streak = 0
            self.keyframe_search_active = False
            return {
                "R": R_map_to_camera,
                "t": t_map_to_camera,
                "inliers": 0,
                "inlier_map_points": np.empty((0, 3)),
                "inlier_image_points": np.empty((0, 2)),
                "outlier_points": np.empty((0, 2)),
                "nearby_associations": 0,
            }

        tracked_points = self.track_keyframe_points(gray)
        if tracked_points is None:
            return None

        image_points, landmark_ids = tracked_points
        result, statistics = self.estimate_pose(image_points, landmark_ids)
        self.last_diagnostics.update(statistics)
        if result is None:
            self.update_keyframe_search(statistics["inliers"])
            return None

        result["nearby_associations"] = 0
        if (
            self.keyframe_search_active
            and result["inliers"] >= KEYFRAME_CANDIDATE_MIN_INLIERS
        ):
            map_update = self.create_keyframe(
                gray,
                result["R"],
                result["t"],
                result["inlier_image_points"],
                result["inlier_landmark_ids"],
            )
            if map_update is not None:
                result["nearby_associations"] = map_update[
                    "nearby_associations"
                ]
                self.last_diagnostics["keyframe_added"] = 1
                self.last_diagnostics.update(map_update)
                self.low_inlier_streak = 0
                self.keyframe_search_active = False

        self.update_keyframe_search(result["inliers"])

        return result
