import argparse
import csv
from pathlib import Path

from camera_tracking import run_tracking as run_lightglue_tracking
from camera_tracking_hybrid import run_tracking as run_hybrid_tracking


CAMERA_NAME = "cam1"
MAP_EXPANSION_MIN_COVERAGE_RATIO = 0.70
FEATURE_ROI_BOTTOM_FRACTION_BY_DISTANCE = {
    "close": 1.00,
    "initial": 0.85,
    "far": 0.70,
}
MAX_OPTICAL_FLOW_FRAMES = 6
MIN_OPTICAL_FLOW_TRACK_RATIO = 0.9


def find_recordings(data_dir):
    suffix = f"_{CAMERA_NAME}"
    return sorted(
        video_path.stem[: -len(suffix)]
        for video_path in (data_dir / "videos").iterdir()
        if video_path.is_file()
        and video_path.stem.endswith(suffix)
    )


def feature_roi_bottom_fraction(recording_name):
    for distance, fraction in (
        FEATURE_ROI_BOTTOM_FRACTION_BY_DISTANCE.items()
    ):
        if distance in recording_name.lower():
            return fraction

    raise ValueError(
        "Recording name does not contain close, initial or far: "
        f"{recording_name}"
    )


def run_recording(algorithm, data_dir, output_dir, recording_name):
    recording_output_dir = output_dir / recording_name
    roi_bottom_fraction = feature_roi_bottom_fraction(recording_name)

    if algorithm == "lightglue":
        return run_lightglue_tracking(
            recording_name,
            recording_output_dir,
            data_dir,
            MAP_EXPANSION_MIN_COVERAGE_RATIO,
            roi_bottom_fraction,
        )

    return run_hybrid_tracking(
        recording_name,
        recording_output_dir,
        data_dir,
        MAP_EXPANSION_MIN_COVERAGE_RATIO,
        roi_bottom_fraction,
        MAX_OPTICAL_FLOW_FRAMES,
        MIN_OPTICAL_FLOW_TRACK_RATIO,
    )


def save_summary(metrics, output_dir):
    summary_path = output_dir / "rmse_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    return summary_path


def main():
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    recording_names = find_recordings(arguments.data_dir)
    metrics = []

    for index, recording_name in enumerate(recording_names, start=1):
        print(
            f"[{index}/{len(recording_names)}] "
            f"Running {recording_name} with {arguments.algorithm}"
        )
        metrics.append(
            run_recording(
                arguments.algorithm,
                arguments.data_dir,
                arguments.output_dir,
                recording_name,
            )
        )

    summary_path = save_summary(metrics, arguments.output_dir)

    print("\nTracking performance:")
    for recording_metrics in metrics:
        print(
            f"{recording_metrics['recording']}: "
            f"{recording_metrics['tracking_fps']:.2f} FPS | "
            f"mean {recording_metrics['mean_tracking_time_ms']:.2f} ms | "
            f"p95 {recording_metrics['p95_tracking_time_ms']:.2f} ms"
        )

    print(f"Saved: {summary_path}")



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=("lightglue", "hybrid"),
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    main()
