import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import pycolmap

from camera_tracking import (
    FEATURE_TYPE,
    MAPPING_END_FRAME,
    KEYFRAME_INTERVAL,
    MAPPING_EVERY_FRAME_FROM_FRAME,
    MAPPING_EVERY_FRAME_UNTIL_FRAME,
    MAPPING_DETECTED_MAX_FEATURES,
    MAPPING_MAX_FEATURES,
    MAPPING_MOTION_TARGETS_PX,
    MAPPING_RECENT_PAIR_COUNT,
    MAPPING_START_FRAME,
    SEED_HEIGHT_FRACTION,
    SEED_WIDTH_FRACTION,
    TRACKING_MAX_FEATURES,
    TRACKING_START_FRAME,
    USE_IMU_GRAVITY_PRIOR,
    run_tracking,
)


FEATURE_ROI_BY_DISTANCE = {
    "close": 1.00,
    "initial": 0.85,
    "far": 0.70,
}


def feature_roi_bottom_fraction(recording_name):
    recording_name = recording_name.lower()
    for distance, fraction in FEATURE_ROI_BY_DISTANCE.items():
        if distance in recording_name:
            return fraction

    raise ValueError(
        "Recording name must contain close, initial or far so the feature "
        "ROI can be selected."
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--reconstruction-method",
        choices=("global", "incremental"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aruco-size-mm", type=float, required=True)
    parser.add_argument(
        "--cylinder-orientation",
        choices=("horizontal", "vertical"),
        required=True,
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyCOLMAP {pycolmap.__version__} | CUDA: {pycolmap.has_cuda}")
    if not pycolmap.has_cuda:
        raise RuntimeError(
            "CUDA-enabled PyCOLMAP is required. Install pycolmap-cuda12."
        )

    started = time.perf_counter()
    metrics = run_tracking(
        arguments.recording,
        arguments.output_dir,
        arguments.data_dir,
        feature_roi_bottom_fraction(arguments.recording),
        reconstruction_method=arguments.reconstruction_method,
        mapping_start_frame=MAPPING_START_FRAME,
        mapping_end_frame=MAPPING_END_FRAME,
        tracking_start_frame=TRACKING_START_FRAME,
        feature_type=FEATURE_TYPE,
        mapping_detected_max_features=MAPPING_DETECTED_MAX_FEATURES,
        mapping_max_features=MAPPING_MAX_FEATURES,
        tracking_max_features=TRACKING_MAX_FEATURES,
        keyframe_interval=KEYFRAME_INTERVAL,
        every_frame_until_frame=MAPPING_EVERY_FRAME_UNTIL_FRAME,
        every_frame_from_frame=MAPPING_EVERY_FRAME_FROM_FRAME,
        mapping_recent_pair_count=MAPPING_RECENT_PAIR_COUNT,
        mapping_motion_targets_px=MAPPING_MOTION_TARGETS_PX,
        use_imu=USE_IMU_GRAVITY_PRIOR,
        aruco_size_mm=arguments.aruco_size_mm,
        seed_width_fraction=SEED_WIDTH_FRACTION,
        seed_height_fraction=SEED_HEIGHT_FRACTION,
        cylinder_orientation=arguments.cylinder_orientation,
    )
    metrics["total_pipeline_seconds"] = time.perf_counter() - started
    metrics["pycolmap_version"] = str(pycolmap.__version__)
    metrics["pycolmap_cuda"] = pycolmap.has_cuda

    summary_path = arguments.output_dir / "jenkins_metrics.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Tracking FPS: {metrics['tracking_fps']:.2f}")
    print(f"Total pipeline time: {metrics['total_pipeline_seconds']:.2f} s")
    print(f"Saved Jenkins metrics: {summary_path}")


if __name__ == "__main__":
    main()
