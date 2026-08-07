from pathlib import Path

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

VIDEO_PATH = (
    Path(__file__).resolve().parent
    / "calibration checkboard_2026-08-04_15.01.05_cam1.webm"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "camera_jabra_640_360"

SQUARES_X = 10
SQUARES_Y = 7
SQUARE_LENGTH_MM = 20.0

FRAME_STEP = 4
OUTLIER_ERROR_MULTIPLIER = 2.0
SHOW_DETECTIONS = False
DETECTION_PREVIEW_DELAY_MS = 1

INNER_CORNERS = (SQUARES_X - 1, SQUARES_Y - 1)


def create_object_points():
    points = np.zeros(
        (INNER_CORNERS[0] * INNER_CORNERS[1], 3),
        dtype=np.float32,
    )
    points[:, :2] = np.mgrid[
        0:INNER_CORNERS[0],
        0:INNER_CORNERS[1],
    ].T.reshape(-1, 2)
    points[:, :2] *= SQUARE_LENGTH_MM
    return points


def detect_chessboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    return cv2.findChessboardCornersSB(gray, INNER_CORNERS, flags)


def show_detection(frame, found, corners):
    preview = frame.copy()
    color = (0, 255, 0) if found else (0, 0, 255)

    if found:
        grid = corners.reshape(
            INNER_CORNERS[1],
            INNER_CORNERS[0],
            2,
        )
        for row in grid:
            cv2.polylines(
                preview,
                [np.rint(row).astype(np.int32)],
                False,
                color,
                1,
                cv2.LINE_AA,
            )
        for column in grid.transpose(1, 0, 2):
            cv2.polylines(
                preview,
                [np.rint(column).astype(np.int32)],
                False,
                color,
                1,
                cv2.LINE_AA,
            )
        for x, y in corners.reshape(-1, 2):
            cv2.circle(
                preview,
                (round(float(x)), round(float(y))),
                4,
                color,
                -1,
                cv2.LINE_AA,
            )

    status = "full chessboard detected" if found else "chessboard not detected"
    cv2.putText(
        preview,
        status,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Chessboard calibration detections", preview)
    cv2.waitKey(DETECTION_PREVIEW_DELAY_MS)


def collect_calibration_views(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open calibration video: {video_path}")

    image_size = (
        round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    chessboard_points = create_object_points()
    object_points = []
    image_points = []
    frame_index = 0
    sampled_frames = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % FRAME_STEP == 0:
            sampled_frames += 1
            found, corners = detect_chessboard(frame)

            if SHOW_DETECTIONS:
                show_detection(frame, found, corners)

            if found:
                object_points.append(chessboard_points.copy())
                image_points.append(corners.astype(np.float32))

        frame_index += 1

    capture.release()
    if SHOW_DETECTIONS:
        cv2.destroyAllWindows()

    return (
        object_points,
        image_points,
        image_size,
        frame_index,
        sampled_frames,
    )


def calibrate(object_points, image_points, image_size):
    return cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )


def view_reprojection_errors(
    object_points,
    image_points,
    camera_matrix,
    distortion,
    rotations,
    translations,
):
    errors = []
    for points_3d, points_2d, rotation, translation in zip(
        object_points,
        image_points,
        rotations,
        translations,
    ):
        projected, _ = cv2.projectPoints(
            points_3d,
            rotation,
            translation,
            camera_matrix,
            distortion,
        )
        residuals = projected.reshape(-1, 2) - points_2d.reshape(-1, 2)
        errors.append(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return np.asarray(errors)


def remove_outlier_views(object_points, image_points, errors):
    maximum_error = np.median(errors) * OUTLIER_ERROR_MULTIPLIER
    keep = errors <= maximum_error
    filtered_object_points = [
        points for points, accepted in zip(object_points, keep) if accepted
    ]
    filtered_image_points = [
        points for points, accepted in zip(image_points, keep) if accepted
    ]
    return filtered_object_points, filtered_image_points, maximum_error


def main():
    (
        object_points,
        image_points,
        image_size,
        frame_count,
        sampled_frames,
    ) = collect_calibration_views(VIDEO_PATH)

    if len(object_points) < 10:
        raise RuntimeError(
            f"Only {len(object_points)} complete chessboard views were found"
        )

    detected_view_count = len(object_points)
    initial = calibrate(object_points, image_points, image_size)
    initial_errors = view_reprojection_errors(
        object_points,
        image_points,
        initial[1],
        initial[2],
        initial[3],
        initial[4],
    )
    object_points, image_points, maximum_error = remove_outlier_views(
        object_points,
        image_points,
        initial_errors,
    )

    rms, camera_matrix, distortion, rotations, translations = calibrate(
        object_points,
        image_points,
        image_size,
    )
    final_errors = view_reprojection_errors(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        rotations,
        translations,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "camera_matrix.npy", camera_matrix)
    np.save(OUTPUT_DIR / "dist_coeffs.npy", distortion)

    print(f"Video frames: {frame_count}")
    print(f"Sampled frames: {sampled_frames}")
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print(
        f"Accepted chessboard views: "
        f"{len(object_points)} / {detected_view_count} detected"
    )
    print(f"Outlier threshold: {maximum_error:.4f} px")
    print(f"Calibration RMS: {rms:.4f} px")
    print(f"Mean view reprojection RMSE: {np.mean(final_errors):.4f} px")
    print(f"Maximum view reprojection RMSE: {np.max(final_errors):.4f} px")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(distortion)
    print(f"Saved: {OUTPUT_DIR / 'camera_matrix.npy'}")
    print(f"Saved: {OUTPUT_DIR / 'dist_coeffs.npy'}")


if __name__ == "__main__":
    main()
