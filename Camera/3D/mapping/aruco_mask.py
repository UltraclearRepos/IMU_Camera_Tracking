import cv2
import numpy as np

from mapping.aruco_reference import aruco_object_points, create_aruco_detector


class ArucoMask:
    """Detect ArUco markers and mask their image regions."""

    def __init__(self, margin_mm=10.0):
        if margin_mm < 0:
            raise ValueError("margin_mm must be non-negative")
        self.margin_mm = float(margin_mm)
        self.detector = create_aruco_detector()
        self._last_detected_ids = np.empty(0, dtype=np.int32)
        self._last_detected_count = 0

    @property
    def last_detected_ids(self):
        return self._last_detected_ids

    @property
    def last_detected_count(self):
        return self._last_detected_count

    def compute(self, frame):
        mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
        corners, ids, _ = self.detector.detectMarkers(frame)
        if not corners:
            enlarged = cv2.resize(
                frame,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            )
            corners, ids, _ = self.detector.detectMarkers(enlarged)
            corners = [corner / 2.0 for corner in corners]

        self._last_detected_ids = (
            ids.reshape(-1).astype(np.int32)
            if ids is not None
            else np.empty(0, dtype=np.int32)
        )
        self._last_detected_count = len(corners)
        marker_points_mm = aruco_object_points()[:, :2].astype(np.float32)
        mask_points_mm = (
            marker_points_mm
            + np.sign(marker_points_mm) * self.margin_mm
        )
        for marker_corners in corners:
            homography = cv2.getPerspectiveTransform(
                marker_points_mm,
                marker_corners.reshape(4, 2).astype(np.float32),
            )
            mask_polygon = cv2.perspectiveTransform(
                mask_points_mm.reshape(1, 4, 2),
                homography,
            ).reshape(4, 2)
            cv2.fillConvexPoly(
                mask,
                np.rint(mask_polygon).astype(np.int32),
                0,
            )

        return mask
