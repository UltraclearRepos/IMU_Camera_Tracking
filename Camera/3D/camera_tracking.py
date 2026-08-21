import csv
import json
import os
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CAMERA_DIR = SCRIPT_DIR.parent
PROJECT_DIR = CAMERA_DIR.parent
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".venv" / "torch_cache")

import cv2
import numpy as np
import torch
from evaluation.mapping_evaluation import evaluate_final_mapping_poses
from mapping.feature_matching import DEVICE, LightGlueFeatureMatching
from mapping.skin_map_builder import SkinMapBuilder
from scipy.spatial.transform import Rotation
from tracking.imu_gravity import ImuGravityProvider
from tracking.skin_map_tracker import SkinMapTracker
from visualization.top_view_visualization import (
    create_tracking_top_view_state,
    save_map_build_top_view,
    save_tracking_top_view,
)
from visualization.tracking_visualization import (
    create_comparison_plots,
    diagnostic_frame,
    save_mapping_feature_video,
    save_skin_mask_initialization_diagnostics,
    save_3d_tracking_diagnostics,
    save_timing_diagnostics,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RECORDING_NAME = "initial_50mm_Arc180-Speed-3_2026-08-18_17.47.36"
DATA_FOLDER = "Cylinder"
CAMERA_NAME = "cam1"
CAMERA_CALIBRATION = "camera_jabra_640_360"
MAX_FRAMES = 100000
FEATURE_ROI_BOTTOM_FRACTION = 0.7
FEATURE_TYPE = "sift"  # "disk" or "sift".

MAPPING_START_FRAME = 1  # First frame used to build the frozen 3D map.
MAPPING_END_FRAME = 988  # Last frame used to build the frozen 3D map.
TRACKING_START_FRAME = 989  # First frame processed by frozen-map tracking.
RECONSTRUCTION_METHOD = "global"  # "global" (GLOMAP) or "incremental" (COLMAP).
MAPPING_FRAME_STEP = 1  # Use every Nth frame during map construction.
# Adaptive example: recent=2, motion=(10.0, 20.0, 40.0).
# Legacy example: recent=10, motion=() matches only the 10 previous frames.
MAPPING_RECENT_PAIR_COUNT = 10
MAPPING_MOTION_TARGETS_PX = ()
MAPPING_MAX_FEATURES = 256  # Spatially distributed features passed to LightGlue.
MAPPING_FEATURE_GRID_ROWS = 4  # Image grid rows used to distribute map features.
MAPPING_FEATURE_GRID_COLUMNS = 4  # Image grid columns used to distribute map features.
GLOBAL_MAP_MAX_LANDMARKS = 1024  # Maximum landmarks in the frozen global map.
GLOBAL_MAP_GRID_ROWS = 8  # Surface grid rows used for uniform landmark selection.
GLOBAL_MAP_GRID_COLUMNS = 8  # Surface grid columns used for uniform landmark selection.
GLOBAL_MAP_REPROJECTION_ERROR_WEIGHT = 0.70  # Geometry error weight versus track length.
MASK_ARUCO_FEATURES = True  # Prevent the removable marker becoming a landmark.
USE_ADAPTIVE_SKIN_MASK = True
USE_IMU_GRAVITY_PRIOR = False
IMU_GRAVITY_HISTORY_SECONDS = 0.05
IMU_ACCELERATION_MAGNITUDE_TOLERANCE_M_S2 = 1.5
IMU_MAXIMUM_GYROSCOPE_RAD_S = np.radians(50.0)


SAVE_DIAGNOSTIC_VIDEO = True
DIAGNOSTIC_VIDEO_FPS = 1.0
SAVE_MAPPING_FEATURE_VIDEO = True
MAPPING_FEATURE_VIDEO_FPS = 10.0
SAVE_MAP_BUILD_TOP_VIEW = True
SAVE_TRACKING_TOP_VIEW = True
TOP_VIEW_VIDEO_FPS = 1.0
TOP_VIEW_VIDEO_SIZE_PX = 800
TOP_VIEW_PADDING_MM = 20.0
SHOW_PREVIEW = False

USE_IMU_SUBFOLDER = "IMU" if USE_IMU_GRAVITY_PRIOR else "noIMU"
RESULTS_DIR = SCRIPT_DIR / "results" / DATA_FOLDER / FEATURE_TYPE / RECONSTRUCTION_METHOD / USE_IMU_SUBFOLDER
CAMERA_MATRIX_PATH = (
    CAMERA_DIR
    / "calibrations"
    / CAMERA_CALIBRATION
    / "camera_matrix.npy"
)
DISTORTION_PATH = (
    CAMERA_DIR
    / "calibrations"
    / CAMERA_CALIBRATION
    / "dist_coeffs.npy"
)


def camera_position(R_map_to_camera, t_map_to_camera):
    R_camera_to_map = R_map_to_camera.T
    return -R_camera_to_map @ t_map_to_camera.reshape(3)


def pose_geometry_diagnostics(result, camera_matrix, distortion):
    """Measure the metric 3D support behind an accepted PnP pose."""
    values = {
        "pnp_reprojection_rmse_px": np.nan,
        "pnp_reprojection_p95_px": np.nan,
        "inlier_3d_primary_std_mm": np.nan,
        "inlier_3d_secondary_std_mm": np.nan,
        "inlier_3d_minor_std_mm": np.nan,
        "inlier_depth_median_mm": np.nan,
        "inlier_depth_std_mm": np.nan,
    }
    if result is None or not len(result["inlier_map_points"]):
        return values

    map_points = np.asarray(result["inlier_map_points"], dtype=np.float64)
    image_points = np.asarray(result["inlier_image_points"], dtype=np.float64)
    rvec = cv2.Rodrigues(result["R"])[0]
    projected, _ = cv2.projectPoints(
        map_points, rvec, result["t"], camera_matrix, distortion
    )
    reprojection_errors = np.linalg.norm(
        projected.reshape(-1, 2) - image_points, axis=1
    )
    values["pnp_reprojection_rmse_px"] = float(
        np.sqrt(np.mean(reprojection_errors**2))
    )
    values["pnp_reprojection_p95_px"] = float(
        np.percentile(reprojection_errors, 95)
    )

    centered = map_points - np.mean(map_points, axis=0)
    spreads = np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(map_points))
    values["inlier_3d_primary_std_mm"] = float(spreads[0])
    values["inlier_3d_secondary_std_mm"] = float(spreads[1])
    values["inlier_3d_minor_std_mm"] = float(spreads[2])
    camera_points = (result["R"] @ map_points.T).T + result["t"]
    values["inlier_depth_median_mm"] = float(np.median(camera_points[:, 2]))
    values["inlier_depth_std_mm"] = float(np.std(camera_points[:, 2]))
    return values


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
    feature_roi_bottom_fraction,
    *,
    reconstruction_method,
    mapping_start_frame,
    mapping_end_frame,
    tracking_start_frame,
    feature_type,
    mapping_frame_step,
    mapping_recent_pair_count,
    mapping_motion_targets_px,
    use_imu,
):
    feature_type = feature_type.lower()
    if not torch.cuda.is_available() and DEVICE == "cuda":
        raise RuntimeError("CUDA is not available in the project .venv")

    data_dir = Path(data_dir)
    video_path = next(
        (data_dir / "videos").glob(f"{recording_name}_{CAMERA_NAME}.*")
    )
    ground_truth_path = (
        data_dir / "dobot" / f"{recording_name}.csv"
    )
    imu_path = data_dir / "imu" / f"{recording_name}.csv"
    imu_calibration_path = (
        PROJECT_DIR / "IMU" / "calibration" / "imu_calibration.json"
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
    imu_gravity_provider = None
    if use_imu:
        if reconstruction_method != "global":
            print(
                "IMU gravity prior is disabled: it is supported only by "
                "global mapping."
            )
        else:
            imu_gravity_provider = ImuGravityProvider(
                imu_path,
                imu_calibration_path,
                video_start_timestamp,
                history_seconds=IMU_GRAVITY_HISTORY_SECONDS,
                acceleration_magnitude_tolerance_m_s2=(
                    IMU_ACCELERATION_MAGNITUDE_TOLERANCE_M_S2
                ),
                maximum_gyroscope_rad_s=IMU_MAXIMUM_GYROSCOPE_RAD_S,
            )
            print(
                f"Using calibrated IMU2 gravity prior: {imu_path} | "
                f"calibration: {imu_calibration_path}"
            )

    camera_matrix = np.load(CAMERA_MATRIX_PATH)
    distortion = np.load(DISTORTION_PATH)
    feature_matching = LightGlueFeatureMatching(
        feature_roi_bottom_fraction,
        feature_type=feature_type,
        mask_aruco_features=MASK_ARUCO_FEATURES,
        use_adaptive_skin_mask=USE_ADAPTIVE_SKIN_MASK,
    )
    map_builder = SkinMapBuilder(
        camera_matrix,
        distortion,
        feature_matching,
        mapping_start_frame,
        mapping_end_frame,
        reconstruction_method,
        mapping_frame_step,
        mapping_recent_pair_count,
        mapping_motion_targets_px,
        MAPPING_MAX_FEATURES,
        MAPPING_FEATURE_GRID_ROWS,
        MAPPING_FEATURE_GRID_COLUMNS,
        GLOBAL_MAP_MAX_LANDMARKS,
        GLOBAL_MAP_GRID_ROWS,
        GLOBAL_MAP_GRID_COLUMNS,
        GLOBAL_MAP_REPROJECTION_ERROR_WEIGHT,
        imu_gravity_provider=imu_gravity_provider,
    )
    map_build_started = time.perf_counter()
    global_map = map_builder.build(
        video_path,
        output_dir / "map",
        diagnostics_output_dir=output_dir,
    )
    map_build_wall_time_s = time.perf_counter() - map_build_started
    (
        mapping_position_rmse,
        mapping_orientation_rmse,
        mapping_registered_percent,
    ) = evaluate_final_mapping_poses(
        global_map,
        video_start_timestamp,
        ground_truth_path,
        output_dir,
        recording_name,
    )
    if SAVE_MAPPING_FEATURE_VIDEO:
        save_mapping_feature_video(
            video_path,
            global_map,
            output_dir / "mapping_features.mp4",
            MAPPING_FEATURE_VIDEO_FPS,
            feature_roi_bottom_fraction,
        )
    save_skin_mask_initialization_diagnostics(feature_matching, output_dir)
    if SAVE_MAP_BUILD_TOP_VIEW:
        map_build_top_view_path = output_dir / "map_build_top_view.mp4"
        save_map_build_top_view(
            global_map,
            map_build_top_view_path,
            TOP_VIEW_VIDEO_FPS,
            TOP_VIEW_VIDEO_SIZE_PX,
            TOP_VIEW_PADDING_MM,
        )
    tracker = SkinMapTracker(
        camera_matrix,
        distortion,
        feature_roi_bottom_fraction=feature_roi_bottom_fraction,
        global_map=global_map,
        feature_matching=feature_matching,
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
    tracking_top_view_states = []
    tracking_times_ms = []
    initial_position = None
    initial_rotation = None
    for _ in range(tracking_start_frame):
        success, _ = capture.read()
        if not success:
            break

    frame_index = tracking_start_frame
    print(
        f"Frozen 3D map ready. Starting tracking at frame {frame_index}..."
    )

    tracking_wall_started = time.perf_counter()
    while frame_index < MAX_FRAMES:
        success, frame = capture.read()
        if not success:
            break

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        tracking_started = time.perf_counter()
        was_initializing = not tracker.initialized
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

            position = absolute_position - initial_position
            relative_rotation = initial_rotation.T @ camera_rotation
            euler = Rotation.from_matrix(relative_rotation).as_euler(
                "xyz", degrees=True
            )
        geometry = pose_geometry_diagnostics(
            result, tracker.camera_matrix, tracker.distortion
        )

        positions.append(position)
        time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if (
            was_initializing
            and diagnostics["initialization_aruco_detected"]
        ):
            aruco_decision = (
                "accepted"
                if diagnostics["initialization_aruco_accepted"]
                else "rejected"
            )
            print(
                f"Frame {frame_index} ({time_s:.3f} s): ArUco "
                f"RMS={diagnostics['initialization_aruco_reprojection_rms_px']:.2f} px, "
                f"max={diagnostics['initialization_aruco_reprojection_max_px']:.2f} px, "
                f"min side={diagnostics['initialization_aruco_min_side_length_px']:.1f} px "
                f"-> {aruco_decision}"
            )
            if diagnostics["initialization_aruco_accepted"]:
                print(
                    f"Tracking initialized from ArUco at frame "
                    f"{frame_index} ({time_s:.3f} s)."
                )
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
                **geometry,
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
                "initialization_aruco_accepted": diagnostics[
                    "initialization_aruco_accepted"
                ],
                "initialization_aruco_reprojection_rms_px": diagnostics[
                    "initialization_aruco_reprojection_rms_px"
                ],
                "initialization_aruco_reprojection_max_px": diagnostics[
                    "initialization_aruco_reprojection_max_px"
                ],
                "initialization_aruco_min_side_length_px": diagnostics[
                    "initialization_aruco_min_side_length_px"
                ],
                "landmarks": len(tracker.landmarks),
                "keyframe_added": diagnostics["keyframe_added"],
                "keyframes": len(tracker.keyframes),
                "tracked": int(result is not None),
            }
        )

        if SAVE_TRACKING_TOP_VIEW:
            tracking_top_view_states.append(
                create_tracking_top_view_state(
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
                feature_roi_bottom_fraction,
                0,
                0,
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
    tracking_wall_time_s = time.perf_counter() - tracking_wall_started

    capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    if SAVE_TRACKING_TOP_VIEW:
        tracking_top_view_path = output_dir / "tracking_top_view.mp4"
        save_tracking_top_view(
            global_map,
            tracking_top_view_states,
            tracking_top_view_path,
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
    timing_plot_path = output_dir / "timing_diagnostics.png"
    tracking_quality_plot_path = (
        output_dir / "tracking_quality_diagnostics.png"
    )
    save_results_csv(rows, csv_path)
    save_timing_diagnostics(
        rows,
        timing_plot_path,
        recording_name,
    )
    save_3d_tracking_diagnostics(
        rows,
        ground_truth_path,
        tracking_quality_plot_path,
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
        "map_build_wall_time_s": float(map_build_wall_time_s),
        "tracking_wall_time_s": float(tracking_wall_time_s),
        "feature_roi_bottom_fraction": (
            feature_roi_bottom_fraction
        ),
        "mapping_start_frame": mapping_start_frame,
        "mapping_end_frame": mapping_end_frame,
        "tracking_start_frame": tracking_start_frame,
        "reconstruction_method": reconstruction_method,
        "feature_type": feature_type,
        "mapping_frame_step": mapping_frame_step,
        "mapping_recent_pair_count": mapping_recent_pair_count,
        "mapping_motion_targets_px": list(mapping_motion_targets_px),
        "map_landmarks": len(tracker.landmarks),
        "map_candidate_landmarks": len(global_map.candidate_positions),
        "map_occupied_grid_cells": global_map.occupied_grid_cell_count,
        "mapping_position_rmse_mm": float(mapping_position_rmse),
        "mapping_orientation_rmse_deg": float(mapping_orientation_rmse),
        "mapping_registered_percent": float(mapping_registered_percent),
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {position_plot_path}")
    print(f"Saved: {orientation_plot_path}")
    print(f"Saved: {timing_plot_path}")
    print(f"Saved: {tracking_quality_plot_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {output_dir / 'mapping_camera_vs_gt.csv'}")
    print(f"Saved: {output_dir / 'mapping_position_vs_gt.png'}")
    print(f"Saved: {output_dir / 'mapping_orientation_vs_gt.png'}")
    if SAVE_DIAGNOSTIC_VIDEO:
        print(f"Saved: {video_path_output}")
    if SAVE_MAP_BUILD_TOP_VIEW:
        print(f"Saved: {map_build_top_view_path}")
    if SAVE_TRACKING_TOP_VIEW:
        print(f"Saved: {tracking_top_view_path}")
    print(f"Position RMSE: {position_rmse:.2f} mm")
    print(f"Orientation RMSE: {orientation_rmse:.2f} deg")
    print(
        "Final mapping BA RMSE: "
        f"{mapping_position_rmse:.2f} mm | "
        f"{mapping_orientation_rmse:.2f} deg"
    )
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
        FEATURE_ROI_BOTTOM_FRACTION,
        reconstruction_method=RECONSTRUCTION_METHOD,
        mapping_start_frame=MAPPING_START_FRAME,
        mapping_end_frame=MAPPING_END_FRAME,
        tracking_start_frame=TRACKING_START_FRAME,
        feature_type=FEATURE_TYPE,
        mapping_frame_step=MAPPING_FRAME_STEP,
        mapping_recent_pair_count=MAPPING_RECENT_PAIR_COUNT,
        mapping_motion_targets_px=MAPPING_MOTION_TARGETS_PX,
        use_imu=USE_IMU_GRAVITY_PRIOR,
    )


if __name__ == "__main__":
    main()
