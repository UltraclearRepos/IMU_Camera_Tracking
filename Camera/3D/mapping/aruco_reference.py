import cv2
import numpy as np


ARUCO_ID = 7
ARUCO_SIZE_MM = 20.0


def create_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(dictionary, parameters)


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

    projected_points, _ = cv2.projectPoints(
        aruco_object_points(),
        rvec,
        tvec,
        camera_matrix,
        distortion,
    )
    corner_errors_px = np.linalg.norm(
        projected_points.reshape(4, 2) - image_points,
        axis=1,
    )
    marker_side_lengths_px = np.linalg.norm(
        image_points - np.roll(image_points, -1, axis=0),
        axis=1,
    )

    return (
        cv2.Rodrigues(rvec)[0],
        tvec.reshape(3),
        {
            "reprojection_rms_px": float(
                np.sqrt(np.mean(corner_errors_px**2))
            ),
            "reprojection_max_px": float(np.max(corner_errors_px)),
            "corner_errors_px": corner_errors_px,
            "min_side_length_px": float(np.min(marker_side_lengths_px)),
        },
    )
