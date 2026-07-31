import cv2
import numpy as np

from skin_map_tracker import (
    MAX_REPROJECTION_ERROR_PX,
    MIN_INLIERS,
    MIN_MATCHES,
    SkinMapTracker,
)


LK_WINDOW_SIZE = (21, 21)  # Optical-flow search window size.
LK_PYRAMID_LEVELS = 3  # Pyramid levels used for larger point motion.
LK_FORWARD_BACKWARD_ERROR_PX = 1.0  # Maximum round-trip tracking error.


class HybridSkinMapTracker(SkinMapTracker):
    def __init__(
        self,
        camera_matrix,
        distortion,
        max_optical_flow_frames,
        min_optical_flow_track_ratio,
        feature_roi_bottom_fraction,
    ):
        super().__init__(
            camera_matrix,
            distortion,
            feature_roi_bottom_fraction,
        )
        self.max_optical_flow_frames = max_optical_flow_frames
        self.min_optical_flow_track_ratio = min_optical_flow_track_ratio
        self.frame_index = -1
        self.last_lightglue_frame = None
        self.force_lightglue = True
        self.previous_gray = None
        self.active_points = np.empty((0, 2), dtype=np.float32)
        self.active_landmark_ids = np.empty(0, dtype=np.int64)
        self.lightglue_track_count = 0

    def store_active_tracks(self, gray, image_points, landmark_ids):
        existing = np.array(
            [
                int(landmark_id) in self.landmarks
                for landmark_id in landmark_ids
            ],
            dtype=bool,
        )
        self.previous_gray = gray.copy()
        self.active_points = np.asarray(
            image_points[existing],
            dtype=np.float32,
        ).reshape(-1, 2)
        self.active_landmark_ids = np.asarray(
            landmark_ids[existing],
            dtype=np.int64,
        )

    def track_with_lightglue(self, frame, gray):
        result = super().track(frame)
        self.last_diagnostics["tracking_method"] = "lightglue"
        self.last_diagnostics["flow_tracks"] = 0

        if result is None or "inlier_image_points" not in result:
            self.force_lightglue = True
            return result

        self.store_active_tracks(
            gray,
            result["inlier_image_points"],
            result["inlier_landmark_ids"],
        )
        self.lightglue_track_count = len(self.active_points)
        self.last_lightglue_frame = self.frame_index
        self.force_lightglue = False
        return result

    def track_points(self, current_gray):
        source_points = self.active_points.reshape(-1, 1, 2)
        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
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
            self.previous_gray,
            current_points,
            None,
            winSize=LK_WINDOW_SIZE,
            maxLevel=LK_PYRAMID_LEVELS,
        )
        if backward_points is None:
            return None

        current_points = current_points.reshape(-1, 2)
        backward_points = backward_points.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(
            self.active_points - backward_points,
            axis=1,
        )

        height, width = current_gray.shape
        roi_top = height * (
            1.0 - self.feature_roi_bottom_fraction
        )
        valid = forward_status.ravel() == 1
        valid &= backward_status.ravel() == 1
        valid &= forward_backward_error <= LK_FORWARD_BACKWARD_ERROR_PX
        valid &= current_points[:, 0] >= 0.0
        valid &= current_points[:, 0] < width
        valid &= current_points[:, 1] >= roi_top
        valid &= current_points[:, 1] < height

        return current_points[valid], self.active_landmark_ids[valid]

    def estimate_flow_pose(self, image_points, landmark_ids):
        self.last_diagnostics["matches"] = len(image_points)
        self.last_diagnostics["flow_tracks"] = len(image_points)
        if len(image_points) < MIN_MATCHES:
            return None

        map_points = np.ascontiguousarray(
            [
                self.landmarks[int(landmark_id)]["position"]
                for landmark_id in landmark_ids
            ],
            dtype=np.float64,
        )
        image_points = np.ascontiguousarray(
            image_points,
            dtype=np.float64,
        )
        rvec = cv2.Rodrigues(self.R_map_to_camera)[0]
        tvec = self.t_map_to_camera.reshape(3, 1).copy()
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
        return {
            "R": R_map_to_camera,
            "t": tvec.reshape(3),
            "inliers": inlier_count,
            "inlier_map_points": map_points[inlier_indices],
            "inlier_image_points": image_points[inlier_indices],
            "inlier_landmark_ids": landmark_ids[inlier_indices],
            "outlier_points": image_points[~inlier_mask],
            "nearby_associations": 0,
        }

    def track_with_optical_flow(self, frame, gray):
        self.reset_diagnostics(0, "optical_flow")
        tracked_points = self.track_points(gray)
        if tracked_points is None:
            self.force_lightglue = True
            return None

        image_points, landmark_ids = tracked_points
        track_ratio = len(image_points) / self.lightglue_track_count
        if track_ratio <= self.min_optical_flow_track_ratio:
            return self.track_with_lightglue(frame, gray)

        result = self.estimate_flow_pose(image_points, landmark_ids)
        if result is None:
            self.force_lightglue = True
            return None

        self.R_map_to_camera = result["R"]
        self.t_map_to_camera = result["t"]
        self.store_active_tracks(
            gray,
            result["inlier_image_points"],
            result["inlier_landmark_ids"],
        )
        return result

    def track(self, frame):
        self.frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        maximum_flow_frames_reached = (
            self.last_lightglue_frame is None
            or self.frame_index - self.last_lightglue_frame
            > self.max_optical_flow_frames
        )
        insufficient_tracks = len(self.active_points) < MIN_MATCHES

        if (
            not self.keyframes
            or self.force_lightglue
            or maximum_flow_frames_reached
            or insufficient_tracks
        ):
            return self.track_with_lightglue(frame, gray)

        return self.track_with_optical_flow(frame, gray)
