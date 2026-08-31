import argparse
import csv
import json
import math
import shutil
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MODULE_DIR.parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import matplotlib.pyplot as plt
from camera_tracking import run_tracking


DEFAULT_CONFIG_PATH = MODULE_DIR / "batch_config.json"


def load_experiment_config(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def resolve_data_dir(config):
    return PROJECT_DIR / "Data" / config["data_folder"]


def resolve_results_dir(config, override):
    if override is not None:
        return override
    return (
        MODULE_DIR
        / "results_batch"
        / config["data_folder"]
        / config["experiment_name"]
        / config["feature_type"]
        / ("IMU" if config["use_imu"] else "noIMU")
    )


FRAME_RANGE_KEYS = (
    "mapping_start_frame",
    "mapping_end_frame",
    "tracking_start_frame",
)


def resolve_frame_range(config, recording_parameters):
    """Read the explicitly configured frame range for one recording."""
    missing = [key for key in FRAME_RANGE_KEYS if key not in recording_parameters]
    if missing:
        raise ValueError(
            "Recording configuration is missing frame range fields: "
            f"{', '.join(missing)}"
        )
    frame_range = {key: recording_parameters[key] for key in FRAME_RANGE_KEYS}
    for key, value in frame_range.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if frame_range["mapping_start_frame"] > frame_range["mapping_end_frame"]:
        raise ValueError("mapping_start_frame must not exceed mapping_end_frame")
    if frame_range["tracking_start_frame"] <= frame_range["mapping_end_frame"]:
        raise ValueError(
            "tracking_start_frame must be after mapping_end_frame"
        )
    return frame_range


def run_recording(
    recording_name,
    recording_parameters,
    config,
    data_dir,
    results_dir,
):
    recording_output_dir = (
        results_dir / config["feature_type"] / recording_name
    )
    frame_range = resolve_frame_range(config, recording_parameters)
    return run_tracking(
        recording_name,
        recording_output_dir,
        data_dir,
        recording_parameters["feature_roi_bottom_fraction"],
        reconstruction_method=config["reconstruction_method"],
        **frame_range,
        feature_type=config["feature_type"],
        mapping_detected_max_features=config["mapping_detected_max_features"],
        mapping_max_features=config["mapping_max_features"],
        tracking_max_features=config["tracking_max_features"],
        keyframe_interval=config["keyframe_interval"],
        every_frame_until_frame=config["every_frame_until_frame"],
        every_frame_from_frame=config["every_frame_from_frame"],
        mapping_recent_pair_count=config["mapping_recent_pair_count"],
        mapping_motion_targets_px=config["mapping_motion_targets_px"],
        use_imu=config["use_imu"],
        aruco_size_mm=recording_parameters["aruco_size_mm"],
        seed_width_fraction=recording_parameters["seed_width_fraction"],
        seed_height_fraction=recording_parameters["seed_height_fraction"],
        mapping_recent_pair_interval=config["mapping_recent_pair_interval"]
    )


def save_summary_csv(metrics, results_dir):
    output_path = results_dir / "rmse_summary.csv"
    fieldnames = list(
        dict.fromkeys(
            key for metric in metrics for key in metric
        )
    )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    return output_path


def format_duration(seconds):
    minutes, seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"


def print_execution_times(metrics):
    print("\nMap and tracking execution times:")
    for item in metrics:
        print(
            f"{item['recording']}: map "
            f"{format_duration(item['map_build_wall_time_s'])} | "
            f"tracking "
            f"{format_duration(item['tracking_wall_time_s'])} | "
            f"tracking FPS {item['tracking_fps']:.2f}"
        )

    total_map_time_s = sum(
        item["map_build_wall_time_s"] for item in metrics
    )
    total_tracking_time_s = sum(
        item["tracking_wall_time_s"] for item in metrics
    )
    print(
        f"TOTAL: map {format_duration(total_map_time_s)} | "
        f"tracking {format_duration(total_tracking_time_s)} | "
        "combined "
        f"{format_duration(total_map_time_s + total_tracking_time_s)}"
    )


def add_value_labels(axis, bars, unit):
    for bar in bars:
        value = bar.get_width()
        if not math.isfinite(value):
            continue
        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.2f} {unit}",
            va="center",
        )


def save_rmse_summary_plot(metrics, experiment_name, results_dir):
    recording_names = [
        f"{item['recording']} [{item['feature_type']}]" for item in metrics
    ]
    position_rmse = [item["position_rmse_mm"] for item in metrics]
    orientation_rmse = [
        item["orientation_rmse_deg"] for item in metrics
    ]
    tracked_percent = [item["tracked_percent"] for item in metrics]
    y_positions = range(len(metrics))

    figure_height = max(6.0, 0.55 * len(metrics))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(24, figure_height),
        constrained_layout=True,
    )

    position_bars = axes[0].barh(
        y_positions,
        position_rmse,
        color="tab:blue",
    )
    axes[0].set_title("Position RMSE")
    axes[0].set_xlabel("RMSE [mm]")
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    add_value_labels(axes[0], position_bars, "mm")

    orientation_bars = axes[1].barh(
        y_positions,
        orientation_rmse,
        color="tab:orange",
    )
    axes[1].set_title("Orientation RMSE")
    axes[1].set_xlabel("RMSE [deg]")
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].invert_yaxis()
    add_value_labels(axes[1], orientation_bars, "deg")

    tracked_bars = axes[2].barh(
        y_positions,
        tracked_percent,
        color="tab:green",
    )
    axes[2].set_title("Tracked frames")
    axes[2].set_xlabel("Tracked [%]")
    axes[2].set_xlim(0, 100)
    axes[2].grid(axis="x", alpha=0.3)
    axes[2].invert_yaxis()

    for axis in axes:
        axis.set_yticks(y_positions)
        axis.set_yticklabels(recording_names)

    for bar, tracked in zip(tracked_bars, tracked_percent):
        axes[2].text(
            tracked,
            bar.get_y() + bar.get_height() / 2,
            f" {tracked:.1f}%",
            ha="left",
            va="center",
        )

    figure.suptitle(experiment_name)
    output_path = results_dir / "rmse_summary.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run 3D camera-tracking experiments from a JSON file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Experiment JSON (default: {DEFAULT_CONFIG_PATH.name}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory from the configuration.",
    )
    return parser.parse_args()

def main():
    arguments = parse_arguments()
    config = load_experiment_config(arguments.config)
    data_dir = resolve_data_dir(config)
    results_dir = resolve_results_dir(config, arguments.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    saved_config_path = results_dir / "config.json"
    shutil.copy2(arguments.config, saved_config_path)

    metrics = []
    recordings = config["recordings"]
    for index, (recording_name, recording_parameters) in enumerate(
        recordings.items(),
        start=1,
    ):
        try:
            frame_range = resolve_frame_range(config, recording_parameters)
            print(
                f"\n[{index}/{len(recordings)}] Running {recording_name} | "
                f"features={config['feature_type']} | mapping="
                f"{frame_range['mapping_start_frame']}.."
                f"{frame_range['mapping_end_frame']} | tracking="
                f"{frame_range['tracking_start_frame']} | "
                f"IMU={config['use_imu']}"
            )
            recording_metrics = run_recording(
                recording_name,
                recording_parameters,
                config,
                data_dir,
                results_dir,
            )
            recording_metrics["status"] = "success"
            metrics.append(recording_metrics)
        except Exception as error:
            print(
                f"\n[{index}/{len(recordings)}] FAILED {recording_name}: "
                f"{type(error).__name__}: {error}"
            )
            traceback.print_exc()
            metrics.append(
                {
                    "recording": recording_name,
                    "feature_type": config["feature_type"],
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    csv_path = save_summary_csv(metrics, results_dir)
    successful_metrics = [
        metric for metric in metrics if metric["status"] == "success"
    ]
    plot_path = None
    if successful_metrics:
        plot_path = save_rmse_summary_plot(
            successful_metrics,
            config["experiment_name"],
            results_dir,
        )

    print(f"\nSaved: {saved_config_path}")
    print(f"Saved: {csv_path}")
    if plot_path is not None:
        print(f"Saved: {plot_path}")
    if successful_metrics:
        print_execution_times(successful_metrics)
    failed_recordings = [
        metric["recording"] for metric in metrics if metric["status"] == "failed"
    ]
    if failed_recordings:
        print(
            "\nFailed recordings (batch continued): "
            + ", ".join(failed_recordings)
        )


if __name__ == "__main__":
    main()
