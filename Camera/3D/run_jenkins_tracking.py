import argparse
import json
import time
from pathlib import Path

import pycolmap

from camera_tracking import (
    FEATURE_TYPE,
    MAPPING_ENABLE_RETRIEVAL,
    MAPPING_END_FRAME,
    MAPPING_FRAME_STEP,
    MAPPING_RETRIEVAL_MAX_SEQUENCE_GAP,
    MAPPING_RETRIEVAL_MIN_SEQUENCE_FRAMES,
    MAPPING_RETRIEVAL_TOP_FRAMES,
    MAPPING_SEQUENTIAL_MATCH_OVERLAP,
    MAPPING_START_FRAME,
    TRACKING_START_FRAME,
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
        mapping_frame_step=MAPPING_FRAME_STEP,
        mapping_sequential_match_overlap=(
            MAPPING_SEQUENTIAL_MATCH_OVERLAP
        ),
        mapping_enable_retrieval=MAPPING_ENABLE_RETRIEVAL,
        mapping_retrieval_top_frames=MAPPING_RETRIEVAL_TOP_FRAMES,
        mapping_retrieval_min_sequence_frames=(
            MAPPING_RETRIEVAL_MIN_SEQUENCE_FRAMES
        ),
        mapping_retrieval_max_sequence_gap=(
            MAPPING_RETRIEVAL_MAX_SEQUENCE_GAP
        ),
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
