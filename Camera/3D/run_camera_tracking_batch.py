import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from camera_tracking import run_tracking


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "batch_config.json"


def load_experiment_config(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def resolve_data_dir(config):
    return PROJECT_DIR / "Data" / config["data_folder"]


def resolve_results_dir(config, override):
    if override is not None:
        return override
    return (
        SCRIPT_DIR
        / "results_batch"
        / config["data_folder"]
        / config["experiment_name"]
    )


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
    return run_tracking(
        recording_name,
        recording_output_dir,
        data_dir,
        recording_parameters["feature_roi_bottom_fraction"],
        reconstruction_method=config["reconstruction_method"],
        mapping_start_frame=config["mapping_start_frame"],
        mapping_end_frame=config["mapping_end_frame"],
        feature_type=config["feature_type"],
        mapping_frame_step=config["mapping_frame_step"],
        mapping_sequential_match_overlap=config[
            "mapping_sequential_match_overlap"
        ],
        mapping_enable_retrieval=config[
            "mapping_enable_retrieval"
        ],
        mapping_retrieval_top_frames=config[
            "mapping_retrieval_top_frames"
        ],
        mapping_retrieval_min_sequence_frames=config[
            "mapping_retrieval_min_sequence_frames"
        ],
        mapping_retrieval_max_sequence_gap=config[
            "mapping_retrieval_max_sequence_gap"
        ],
    )


def save_summary_csv(metrics, results_dir):
    output_path = results_dir / "rmse_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    return output_path


def add_value_labels(axis, bars, unit):
    for bar in bars:
        value = bar.get_width()
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

    figure_height = max(6.0, 0.55 * len(metrics))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(24, figure_height),
        constrained_layout=True,
    )

    position_bars = axes[0].barh(
        recording_names,
        position_rmse,
        color="tab:blue",
    )
    axes[0].set_title("Position RMSE")
    axes[0].set_xlabel("RMSE [mm]")
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    add_value_labels(axes[0], position_bars, "mm")

    orientation_bars = axes[1].barh(
        recording_names,
        orientation_rmse,
        color="tab:orange",
    )
    axes[1].set_title("Orientation RMSE")
    axes[1].set_xlabel("RMSE [deg]")
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].invert_yaxis()
    add_value_labels(axes[1], orientation_bars, "deg")

    tracked_bars = axes[2].barh(
        recording_names,
        tracked_percent,
        color="tab:green",
    )
    axes[2].set_title("Tracked frames")
    axes[2].set_xlabel("Tracked [%]")
    axes[2].set_xlim(0, 100)
    axes[2].grid(axis="x", alpha=0.3)
    axes[2].invert_yaxis()

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
        print(
            f"\n[{index}/{len(recordings)}] Running {recording_name} | "
            f"features={config['feature_type']} | mapping="
            f"{config['mapping_start_frame']}.."
            f"{config['mapping_end_frame']}"
        )
        metrics.append(
            run_recording(
                recording_name,
                recording_parameters,
                config,
                data_dir,
                results_dir,
            )
        )

    csv_path = save_summary_csv(metrics, results_dir)
    plot_path = save_rmse_summary_plot(
        metrics,
        config["experiment_name"],
        results_dir,
    )

    print(f"\nSaved: {saved_config_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
