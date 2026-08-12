import cv2
import numpy as np


ARUCO_ID = 7
ARUCO_SIZE_MM = 20.0


def create_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )
    return cv2.aruco.ArucoDetector(dictionary)


def aruco_object_points():
    half = ARUCO_SIZE_MM / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def detect_aruco_pose(
    frame,
    camera_matrix,
    distortion,
    detector,
):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or ARUCO_ID not in ids.flatten():
        return None

    marker_index = np.where(ids.flatten() == ARUCO_ID)[0][0]
    image_points = corners[marker_index].reshape(4, 2).astype(np.float64)
    success, rvec, tvec = cv2.solvePnP(
        aruco_object_points(),
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None

    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)
