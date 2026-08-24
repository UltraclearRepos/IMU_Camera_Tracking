import cv2
import numpy as np

from mapping.aruco_reference import create_aruco_detector


class ArucoMask:
    """Detect ArUco markers and mask their image regions."""

    def __init__(self, margin_px=20):
        if margin_px < 0:
            raise ValueError("margin_px must be non-negative")
        self.margin_px = int(margin_px)
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
        for marker_corners in corners:
            polygon = np.rint(marker_corners.reshape(4, 2)).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, 0)

        if self.margin_px > 0:
            size = 2 * self.margin_px + 1
            mask = cv2.erode(mask, np.ones((size, size), dtype=np.uint8))
        return mask
