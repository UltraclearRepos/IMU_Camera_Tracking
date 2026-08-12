import csv
import json
import os
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".venv" / "torch_cache")

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from skin_map_tracker import (
    DEVICE,
    INITIALIZATION_FRAMES,
    INITIALIZATION_MIN_LANDMARKS,
    SkinMapTracker,
)
from tracking_visualization import (
    create_top_view_state,
    create_comparison_plots,
    diagnostic_frame,
    save_mapping_diagnostics,
    save_timing_diagnostics,
    save_top_view_video,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# 1. Dodac landmarki tylko tam gdzie pusto / ewentualnie tam gdzie juz jest pokrycie dodac tylko niewiarygodnie dobre, czyli wtedy np moze byc tak ze dodamy np tylko 20, nie trzeba za kazdym razem dopychac do limitu,
#    limit powinien byc tylko gorna granica

RECORDING_NAME = "arc2cm-far-white-withlight_Speed-3_2026-07-29_17.00.49"
DATA_FOLDER = "LineArc-1-2cm"
CAMERA_NAME = "cam1"
CAMERA_CALIBRATION = "camera_jabra_640_360"
MAX_FRAMES = 100000
MAP_EXPANSION_MIN_COVERAGE_RATIO = 0.7
FEATURE_ROI_BOTTOM_FRACTION = 0.7


SAVE_DIAGNOSTIC_VIDEO = True
DIAGNOSTIC_VIDEO_FPS = 1.0
SAVE_TOP_VIEW_VIDEO = True
TOP_VIEW_VIDEO_FPS = 1.0
TOP_VIEW_VIDEO_SIZE_PX = 800
TOP_VIEW_PADDING_MM = 20.0
TOP_VIEW_MAX_VIEW_DISTANCE_MM = 250.0
SHOW_PREVIEW = False


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / DATA_FOLDER
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


CAMERA_TO_OUTPUT_AXES = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ]
)
CAMERA_EULER_SIGNS = np.array([1.0, -1.0, 1.0])


def save_results_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_video_start_timestamp(path):
    with path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    return float(row["start_timestamp"])


def run_tracking(
    recording_name,
    output_dir,
    data_dir,
    map_expansion_min_coverage_ratio,
    feature_roi_bottom_fraction,
):
    if not torch.cuda.is_available() and DEVICE == "cuda":
        raise RuntimeError("CUDA is not available in the project .venv")

    data_dir = Path(data_dir)
    video_path = next(
        (data_dir / "videos").glob(f"{recording_name}_{CAMERA_NAME}.*")
    )
    ground_truth_path = (
        data_dir / "dobot" / f"{recording_name}.csv"
    )
    video_timestamp_path = (
        data_dir
        / "video_timestamps"
        / f"{recording_name}.csv"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_start_timestamp = load_video_start_timestamp(
        video_timestamp_path
    )

    camera_matrix = np.load(CAMERA_MATRIX_PATH)
    distortion = np.load(DISTORTION_PATH)
    tracker = SkinMapTracker(
        camera_matrix,
        distortion,
        feature_roi_bottom_fraction=feature_roi_bottom_fraction,
        map_expansion_min_coverage_ratio=map_expansion_min_coverage_ratio,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = None
    if SAVE_DIAGNOSTIC_VIDEO:
        video_path_output = output_dir / "tracking.mp4"
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

    print("Initializing map...")
    initialized = False
    while not initialized and frame_index < MAX_FRAMES:
        success, frame = capture.read()
        if not success:
            break

        initialized = tracker.initialize(frame)
        frame_index += 1

    print("Map initialized. Starting tracking...")

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

        position = np.full(3, np.nan)
        euler = np.full(3, np.nan)

        if result is not None:
            absolute_position = camera_position(result["R"], result["t"])
            camera_rotation = result["R"].T

            if initial_position is None:
                initial_position = absolute_position.copy()
                initial_rotation = camera_rotation.copy()

            position = CAMERA_TO_OUTPUT_AXES @ (
                absolute_position - initial_position
            )
            relative_rotation = initial_rotation.T @ camera_rotation
            euler = Rotation.from_matrix(relative_rotation).as_euler(
                "xyz", degrees=True
            )
            euler *= CAMERA_EULER_SIGNS

        positions.append(position)
        time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rows.append(
            {
                "frame": frame_index,
                "time_s": time_s,
                "timestamp": video_start_timestamp + time_s,
                "tracking_time_ms": tracking_time_ms,
                "feature_extraction_ms": diagnostics[
                    "feature_extraction_ms"
                ],
                "aruco_pose_ms": diagnostics["aruco_pose_ms"],
                "global_map_projection_ms": diagnostics[
                    "global_map_projection_ms"
                ],
                "lightglue_ms": diagnostics["lightglue_ms"],
                "optical_flow_ms": diagnostics["optical_flow_ms"],
                "pnp_ransac_ms": diagnostics["pnp_ransac_ms"],
                "pnp_refine_ms": diagnostics["pnp_refine_ms"],
                "map_coverage_ms": diagnostics["map_coverage_ms"],
                "map_update_ms": diagnostics["map_update_ms"],
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
                "required_matches": diagnostics["required_matches"],
                "required_inliers": diagnostics["required_inliers"],
                "pnp_inlier_ratio": diagnostics["pnp_inlier_ratio"],
                "new_features": diagnostics["new_features"],
                "nearby_associations": diagnostics["nearby_associations"],
                "new_landmarks": diagnostics["new_landmarks"],
                "removed_landmarks": diagnostics["removed_landmarks"],
                "visible_landmarks": diagnostics[
                    "visible_landmarks"
                ],
                "map_coverage_ratio": diagnostics[
                    "map_coverage_ratio"
                ],
                "map_expansion_coverage_threshold": diagnostics[
                    "map_expansion_coverage_threshold"
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
                    TOP_VIEW_MAX_VIEW_DISTANCE_MM,
                )
            )

        if SAVE_DIAGNOSTIC_VIDEO or SHOW_PREVIEW:
            preview = diagnostic_frame(
                frame,
                tracker,
                result,
                positions,
                feature_roi_bottom_fraction,
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
            output_dir / "map_top_view.mp4",
            TOP_VIEW_VIDEO_FPS,
            TOP_VIEW_VIDEO_SIZE_PX,
            TOP_VIEW_PADDING_MM,
        )

    average_tracking_time_ms = np.mean(tracking_times_ms)
    median_tracking_time_ms = np.median(tracking_times_ms)
    p95_tracking_time_ms = np.percentile(tracking_times_ms, 95)
    tracking_fps = 1000.0 / average_tracking_time_ms

    csv_path = output_dir / "camera.csv"
    position_plot_path = output_dir / "position.png"
    orientation_plot_path = output_dir / "orientation.png"
    diagnostics_plot_path = output_dir / "mapping_diagnostics.png"
    timing_plot_path = output_dir / "timing_diagnostics.png"
    save_results_csv(rows, csv_path)
    save_mapping_diagnostics(
        rows,
        diagnostics_plot_path,
        recording_name,
    )
    save_timing_diagnostics(
        rows,
        timing_plot_path,
        recording_name,
    )
    position_rmse, orientation_rmse = create_comparison_plots(
        rows,
        ground_truth_path,
        position_plot_path,
        orientation_plot_path,
        recording_name,
    )
    tracked_frames = sum(row["tracked"] for row in rows)
    tracked_percent = 100.0 * tracked_frames / len(rows)
    metrics = {
        "recording": recording_name,
        "position_rmse_mm": float(position_rmse),
        "orientation_rmse_deg": float(orientation_rmse),
        "tracked_percent": tracked_percent,
        "mean_tracking_time_ms": float(average_tracking_time_ms),
        "median_tracking_time_ms": float(median_tracking_time_ms),
        "p95_tracking_time_ms": float(p95_tracking_time_ms),
        "tracking_fps": float(tracking_fps),
        "feature_roi_bottom_fraction": (
            feature_roi_bottom_fraction
        ),
        "map_expansion_min_coverage_ratio": (
            map_expansion_min_coverage_ratio
        ),
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {position_plot_path}")
    print(f"Saved: {orientation_plot_path}")
    print(f"Saved: {diagnostics_plot_path}")
    print(f"Saved: {timing_plot_path}")
    print(f"Saved: {metrics_path}")
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
    return metrics


def main():
    run_tracking(
        RECORDING_NAME,
        RESULTS_DIR / RECORDING_NAME,
        PROJECT_DIR / "Data" / DATA_FOLDER,
        MAP_EXPANSION_MIN_COVERAGE_RATIO,
        FEATURE_ROI_BOTTOM_FRACTION,
    )


if __name__ == "__main__":
    main()
