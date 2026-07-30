import csv
import os
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".venv" / "torch_cache")

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from recording_axes import CAMERA_MAP_TO_DOBOT_BY_RECORDING
from skin_map_tracker import (
    DEVICE,
    FEATURE_ROI_BOTTOM_FRACTION,
    INITIALIZATION_FRAMES,
    INITIALIZATION_MIN_LANDMARKS,
    KEYFRAME_ROTATION_DEG,
    KEYFRAME_TRANSLATION_MM,
    SkinMapTracker,
)
from tracking_visualization import (
    create_top_view_state,
    create_comparison_plots,
    diagnostic_frame,
    save_mapping_diagnostics,
    save_top_view_video,
)

# 1. Moze premiowac punkty w kierunku ktorych idziemy w konetkscie dodania do mapy globalnej. Np jesli idziemy po osi X w gore czyli jakby obraz przesuwa sie w kamerze w dol to dodawac punkty bardziej na gorze obrazu
# 2. Moze nie usuwac najgorszych punktow, tylko dbac zeby gestosc tez sie zgadzala, zeby nie wywalic np wszysytkich punktow znalezionych na poczatku w pierwszej fazie ruchu, bo jak potem wrocimy do tego miejsca to punktow
#    nie bedzie i trzeba bedzie dodawac jako nowe, a tak mielibysmy juz dobrze sprawdzone punkty tam

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RECORDING_NAME = "initialpos-white-withlight_Speed-3_2026-07-29_17.46.25"
CAMERA_NAME = "cam1"
CAMERA_CALIBRATION = "camera_jabra_640_360"
CAMERA_MAP_TO_DOBOT = CAMERA_MAP_TO_DOBOT_BY_RECORDING[RECORDING_NAME]
MAX_FRAMES = 100000


SAVE_DIAGNOSTIC_VIDEO = True
DIAGNOSTIC_VIDEO_FPS = 1.0
SAVE_TOP_VIEW_VIDEO = True
TOP_VIEW_VIDEO_FPS = 1.0
TOP_VIEW_VIDEO_SIZE_PX = 800
TOP_VIEW_PADDING_MM = 20.0
SHOW_PREVIEW = False


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data3"
OUTPUT_DIR = SCRIPT_DIR / "results" / RECORDING_NAME
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


def camera_position(R_map_to_camera, t_map_to_camera):
    R_camera_to_map = R_map_to_camera.T
    return -R_camera_to_map @ t_map_to_camera.reshape(3)


def save_results_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_video_start_timestamp(path):
    with path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    return float(row["start_timestamp"])


def main():
    if not torch.cuda.is_available() and DEVICE == "cuda":
        raise RuntimeError("CUDA is not available in the project .venv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_start_timestamp = load_video_start_timestamp(
        VIDEO_TIMESTAMP_PATH
    )

    camera_matrix = np.load(CAMERA_MATRIX_PATH)
    distortion = np.load(DISTORTION_PATH)
    tracker = SkinMapTracker(camera_matrix, distortion)

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(VIDEO_PATH)

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
    top_view_states = []
    tracking_times_ms = []
    initial_position = None
    initial_rotation = None
    frame_index = 0

    while frame_index < MAX_FRAMES:
        success, frame = capture.read()
        if not success:
            break

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        tracking_started = time.perf_counter()
        result = tracker.track(frame)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        tracking_time_ms = 1000.0 * (
            time.perf_counter() - tracking_started
        )
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
                "xyz", degrees=True
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
                "tracking_method": diagnostics["tracking_method"],
                "matches": diagnostics["matches"],
                "flow_tracks": diagnostics["flow_tracks"],
                "inliers": diagnostics["inliers"],
                "pnp_inlier_ratio": diagnostics["pnp_inlier_ratio"],
                "new_features": diagnostics["new_features"],
                "nearby_associations": diagnostics["nearby_associations"],
                "new_landmarks": diagnostics["new_landmarks"],
                "removed_landmarks": diagnostics["removed_landmarks"],
                "keyframe_inlier_threshold": diagnostics[
                    "keyframe_inlier_threshold"
                ],
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

        if SAVE_TOP_VIEW_VIDEO:
            top_view_states.append(
                create_top_view_state(
                    tracker,
                    result,
                    frame.shape,
                    frame_index,
                )
            )

        if SAVE_DIAGNOSTIC_VIDEO or SHOW_PREVIEW:
            preview = diagnostic_frame(
                frame,
                tracker,
                result,
                positions,
                FEATURE_ROI_BOTTOM_FRACTION,
                INITIALIZATION_FRAMES,
                INITIALIZATION_MIN_LANDMARKS,
                tracking_time_ms,
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

    if SAVE_TOP_VIEW_VIDEO:
        save_top_view_video(
            top_view_states,
            OUTPUT_DIR / f"{RECORDING_NAME}_map_top_view.mp4",
            TOP_VIEW_VIDEO_FPS,
            TOP_VIEW_VIDEO_SIZE_PX,
            TOP_VIEW_PADDING_MM,
        )

    average_tracking_time_ms = np.mean(tracking_times_ms)
    median_tracking_time_ms = np.median(tracking_times_ms)
    p95_tracking_time_ms = np.percentile(tracking_times_ms, 95)
    tracking_fps = 1000.0 / average_tracking_time_ms

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
    save_mapping_diagnostics(
        rows,
        diagnostics_plot_path,
        RECORDING_NAME,
        KEYFRAME_TRANSLATION_MM,
        KEYFRAME_ROTATION_DEG,
    )
    position_rmse, orientation_rmse = create_comparison_plots(
        rows,
        GROUND_TRUTH_PATH,
        position_plot_path,
        orientation_plot_path,
        RECORDING_NAME,
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {position_plot_path}")
    print(f"Saved: {orientation_plot_path}")
    print(f"Saved: {diagnostics_plot_path}")
    if SAVE_DIAGNOSTIC_VIDEO:
        print(f"Saved: {video_path_output}")
    print(f"Position RMSE: {position_rmse:.2f} mm")
    print(f"Orientation RMSE: {orientation_rmse:.2f} deg")
    print(
        f"Tracking time: mean {average_tracking_time_ms:.1f} ms/frame | "
        f"median {median_tracking_time_ms:.1f} ms | "
        f"p95 {p95_tracking_time_ms:.1f} ms | "
        f"{tracking_fps:.1f} FPS"
    )


if __name__ == "__main__":
    main()
