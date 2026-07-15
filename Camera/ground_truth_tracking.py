import csv
import math
from pathlib import Path

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

REFERENCE_VIDEO_PATH = (
    SCRIPT_DIR
    / "Data"
    / "horizontal_10x_5sp__x-005"
    / "horizontal_10x_5sp__x"
    / "ground_truth"
    / "video.mp4"
)
OUTPUT_DIRECTORY = SCRIPT_DIR / "results" / "horizontal_line_1"

CALIBRATION_DIRECTORY = SCRIPT_DIR / "calibrations" / "camera_jabra_1920_1080"
CAMERA_MATRIX_PATH = CALIBRATION_DIRECTORY / "camera_matrix.npy"
DISTORTION_PATH = CALIBRATION_DIRECTORY / "dist_coeffs.npy"

ARUCO_ID = 0
ARUCO_SIZE_MM = 42.5

ALIGN_TO_INITIAL_MARKER_FRAME = False
FLIP_X = True
MARKER_TO_TRACKED_POINT_MM = np.array([0.0, 0.0, 0.0])

SHOW_PREVIEW = False


def rotation_matrix_to_euler(rotation_matrix):
    sy = math.sqrt(
        rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
    )
    singular = sy < 1e-6

    if singular:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0
    else:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return np.degrees([roll, pitch, yaw])


def estimate_marker_pose(corners, camera_matrix, distortion):
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
        corners.reshape(4, 2),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None
    return rvec.reshape(3), tvec.reshape(3)


class GroundTruthTracker:
    def __init__(self, camera_matrix, distortion):
        self.camera_matrix = camera_matrix
        self.distortion = distortion

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        self.initial_rvec = None
        self.initial_tvec = None

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or ARUCO_ID not in ids.flatten():
            return None, None, corners, ids

        marker_index = np.where(ids.flatten() == ARUCO_ID)[0][0]
        pose = estimate_marker_pose(
            corners[marker_index],
            self.camera_matrix,
            self.distortion,
        )
        if pose is None:
            return None, None, corners, ids

        rvec, tvec = pose
        if self.initial_tvec is None:
            self.initial_rvec = rvec.copy()
            self.initial_tvec = tvec.copy()

        initial_rotation = cv2.Rodrigues(self.initial_rvec)[0]
        current_rotation = cv2.Rodrigues(rvec)[0]
        relative_rotation = initial_rotation.T @ current_rotation
        relative_position = tvec - self.initial_tvec

        if ALIGN_TO_INITIAL_MARKER_FRAME:
            relative_position = initial_rotation.T @ relative_position
            relative_position += (
                relative_rotation - np.eye(3)
            ) @ MARKER_TO_TRACKED_POINT_MM
        else:
            relative_position += (
                current_rotation - initial_rotation
            ) @ MARKER_TO_TRACKED_POINT_MM

        euler = rotation_matrix_to_euler(relative_rotation)
        if ALIGN_TO_INITIAL_MARKER_FRAME:
            euler[0] *= -1.0
        if FLIP_X:
            relative_position[0] *= -1.0

        return relative_position, euler, corners, ids


def draw_preview(frame, position, euler, corners, ids):
    output = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(output, corners, ids)

    if position is None:
        cv2.putText(
            output,
            f"ArUco ID {ARUCO_ID} not detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return output

    cv2.rectangle(output, (10, 10), (500, 145), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"XYZ [mm]: {position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}",
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        output,
        f"RPY [deg]: {euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return output


def main():
    camera_matrix = np.load(CAMERA_MATRIX_PATH)
    distortion = np.load(DISTORTION_PATH)
    tracker = GroundTruthTracker(camera_matrix, distortion)

    capture = cv2.VideoCapture(str(REFERENCE_VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(REFERENCE_VIDEO_PATH)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "ground_truth.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "X", "Y", "Z", "Roll", "Pitch", "Yaw"])

        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            position, euler, corners, ids = tracker.process(frame)
            if position is None:
                writer.writerow([frame_index, "", "", "", "", "", ""])
            else:
                writer.writerow(
                    [
                        frame_index,
                        f"{position[0]:.4f}",
                        f"{position[1]:.4f}",
                        f"{position[2]:.4f}",
                        f"{euler[0]:.4f}",
                        f"{euler[1]:.4f}",
                        f"{euler[2]:.4f}",
                    ]
                )

            if SHOW_PREVIEW:
                preview = draw_preview(frame, position, euler, corners, ids)
                preview = cv2.resize(preview, (1280, 720))
                cv2.imshow("Ground truth tracking", preview)
                if cv2.waitKey(1) == 27:
                    break

            if frame_index % 100 == 0:
                print(f"Frame {frame_index}")
            frame_index += 1

    capture.release()
    cv2.destroyAllWindows()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
