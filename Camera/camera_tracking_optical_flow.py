import csv
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from optical_flow_tracker import (
    FEATURE_ROI_BOTTOM_FRACTION,
    KEYFRAME_CANDIDATE_MIN_INLIERS,
    KEYFRAME_SEARCH_TRIGGER_INLIERS,
    MIN_PNP_INLIERS,
    OpticalFlowMapTracker,
)
from recording_axes import CAMERA_MAP_TO_DOBOT_BY_RECORDING
from tracking_visualization import (
    create_comparison_plots,
    optical_flow_diagnostic_frame,
    save_optical_flow_diagnostics,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RECORDING_NAME = "horizontal_line_1"
CAMERA_NAME = "cam1"
CAMERA_CALIBRATION = "camera_jabra_640_360"
CAMERA_MAP_TO_DOBOT = CAMERA_MAP_TO_DOBOT_BY_RECORDING[RECORDING_NAME]

SAVE_DIAGNOSTIC_VIDEO = True
DIAGNOSTIC_VIDEO_FPS = 1.0
SHOW_PREVIEW = False
MAX_FRAMES = 1000000


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "Data2"
OUTPUT_DIR = SCRIPT_DIR / "results" / RECORDING_NAME / "optical_flow"
VIDEO_PATHS = list(
    (DATA_DIR / "videos").glob(
        f"{RECORDING_NAME}_{CAMERA_NAME}.*"
    )
)
if len(VIDEO_PATHS) != 1:
    raise RuntimeError(
        f"Expected one video for {RECORDING_NAME} {CAMERA_NAME}, "
        f"found {len(VIDEO_PATHS)}"
    )
VIDEO_PATH = VIDEO_PATHS[0]
GROUND_TRUTH_PATH = DATA_DIR / "dobot" / f"{RECORDING_NAME}.csv"
VIDEO_TIMESTAMP_PATH = (
    DATA_DIR / "video_timestamps" / f"{RECORDING_NAME}.csv"
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


def camera_position(R_map_to_camera, t_map_to_camera):
    R_camera_to_map = R_map_to_camera.T
    return -R_camera_to_map @ t_map_to_camera.reshape(3)


def load_video_start_timestamp(path):
    with path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    return float(row["start_timestamp"])


def save_results(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_start_timestamp = load_video_start_timestamp(VIDEO_TIMESTAMP_PATH)
    tracker = OpticalFlowMapTracker(
        np.load(CAMERA_MATRIX_PATH),
        np.load(DISTORTION_PATH),
    )

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(VIDEO_PATH)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    diagnostic_video_path = OUTPUT_DIR / f"{RECORDING_NAME}_optical_flow.mp4"
    video_writer = None
    if SAVE_DIAGNOSTIC_VIDEO:
        video_writer = cv2.VideoWriter(
            str(diagnostic_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            DIAGNOSTIC_VIDEO_FPS,
            frame_size,
        )

    rows = []
    positions = []
    tracking_times_ms = []
    initial_position = None
    initial_rotation = None
    frame_index = 0

    while frame_index < MAX_FRAMES:
        success, frame = capture.read()
        if not success:
            break

        tracking_started = time.perf_counter()
        result = tracker.track(frame)
        tracking_time_ms = 1000.0 * (time.perf_counter() - tracking_started)
        tracking_times_ms.append(tracking_time_ms)
        diagnostics = tracker.last_diagnostics

        if result is None:
            position = np.full(3, np.nan)
            euler = np.full(3, np.nan)
        else:
            absolute_position = camera_position(result["R"], result["t"])
            camera_rotation = result["R"].T
            if initial_position is None:
                initial_position = absolute_position.copy()
                initial_rotation = camera_rotation.copy()

            position = CAMERA_MAP_TO_DOBOT @ (
                absolute_position - initial_position
            )
            relative_rotation = initial_rotation.T @ camera_rotation
            euler = Rotation.from_matrix(relative_rotation).as_euler(
                "xyz",
                degrees=True,
            )

        positions.append(position)
        time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rows.append(
            {
                "frame": frame_index,
                "time_s": time_s,
                "timestamp": video_start_timestamp + time_s,
                "tracking_time_ms": tracking_time_ms,
                "x_mm": position[0],
                "y_mm": position[1],
                "z_mm": position[2],
                "roll_deg": euler[0],
                "pitch_deg": euler[1],
                "yaw_deg": euler[2],
                "flow_tracks": diagnostics["flow_tracks"],
                "pnp_inliers": diagnostics["inliers"],
                "pnp_inlier_ratio": diagnostics["pnp_inlier_ratio"],
                "new_features": diagnostics["new_features"],
                "nearby_associations": diagnostics[
                    "nearby_associations"
                ],
                "new_landmarks": diagnostics["new_landmarks"],
                "landmarks": len(tracker.landmarks),
                "keyframe_added": diagnostics["keyframe_added"],
                "keyframes": len(tracker.keyframes),
                "tracked": int(result is not None),
            }
        )

        if SAVE_DIAGNOSTIC_VIDEO or SHOW_PREVIEW:
            preview = optical_flow_diagnostic_frame(
                frame,
                tracker,
                result,
                position,
                FEATURE_ROI_BOTTOM_FRACTION,
                tracking_time_ms,
            )
            if video_writer is not None:
                video_writer.write(preview)
            if SHOW_PREVIEW:
                cv2.imshow("Optical flow skin tracking", preview)
                if cv2.waitKey(1) == 27:
                    break

        if frame_index % 100 == 0:
            print(
                f"Frame {frame_index}/{frame_count}, "
                f"keyframes: {len(tracker.keyframes)}"
            )
        frame_index += 1

    capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    csv_path = OUTPUT_DIR / f"{RECORDING_NAME}_optical_flow.csv"
    position_plot_path = OUTPUT_DIR / (
        f"{RECORDING_NAME}_optical_flow_vs_gt_position.png"
    )
    orientation_plot_path = OUTPUT_DIR / (
        f"{RECORDING_NAME}_optical_flow_vs_gt_orientation.png"
    )
    diagnostics_plot_path = OUTPUT_DIR / (
        f"{RECORDING_NAME}_optical_flow_diagnostics.png"
    )
    save_results(rows, csv_path)
    save_optical_flow_diagnostics(
        rows,
        diagnostics_plot_path,
        RECORDING_NAME,
        MIN_PNP_INLIERS,
        KEYFRAME_SEARCH_TRIGGER_INLIERS,
        KEYFRAME_CANDIDATE_MIN_INLIERS,
    )
    position_rmse, orientation_rmse = create_comparison_plots(
        rows,
        GROUND_TRUTH_PATH,
        position_plot_path,
        orientation_plot_path,
        f"{RECORDING_NAME} optical flow",
    )

    mean_time = np.mean(tracking_times_ms)
    median_time = np.median(tracking_times_ms)
    p95_time = np.percentile(tracking_times_ms, 95)
    print(f"Saved: {csv_path}")
    print(f"Saved: {position_plot_path}")
    print(f"Saved: {orientation_plot_path}")
    print(f"Saved: {diagnostics_plot_path}")
    if SAVE_DIAGNOSTIC_VIDEO:
        print(f"Saved: {diagnostic_video_path}")
    print(f"Position RMSE: {position_rmse:.2f} mm")
    print(f"Orientation RMSE: {orientation_rmse:.2f} deg")
    print(
        f"Tracking time: mean {mean_time:.1f} ms/frame | "
        f"median {median_time:.1f} ms | p95 {p95_time:.1f} ms | "
        f"{1000.0 / mean_time:.1f} FPS"
    )


if __name__ == "__main__":
    main()
