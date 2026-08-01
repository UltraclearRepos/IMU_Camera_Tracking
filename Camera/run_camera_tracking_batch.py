import csv
from pathlib import Path

import matplotlib.pyplot as plt
from camera_tracking import run_tracking


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------

EXPERIMENT_NAME = "visible_map_expansion"

RECORDINGS = {
    "initialpos-white-withlight_Speed-3_2026-07-29_17.46.25": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "initialpos-white-nolight_Speed-3_2026-07-29_17.47.53": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "initialpos-dark-nolight_Speed-3_2026-07-28_16.55.02": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "initialpos-dark-withlight_Speed-3_2026-07-28_16.57.56": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "far-white-nolight_Speed-3_2026-07-28_17.06.45": {
        "feature_roi_bottom_fraction": 0.7,
    },
    "far-white-withlight_Speed-3_2026-07-28_17.08.22": {
        "feature_roi_bottom_fraction": 0.7,
    },
    "far-dark-withlight_Speed-3_2026-07-28_17.02.52": {
        "feature_roi_bottom_fraction": 0.7,
    },
    "far-dark-nolight_Speed-3_2026-07-28_17.04.19": {
        "feature_roi_bottom_fraction": 0.7,
    },
    "close-white-withlight_Speed-3_2026-07-28_17.12.37": {
        "feature_roi_bottom_fraction": 1.0,
    },
    "close-white-nolight_Speed-3_2026-07-28_17.14.02": {
        "feature_roi_bottom_fraction": 1.0,
    },
    "close-dark-nolight_Speed-3_2026-07-28_17.16.20": {
        "feature_roi_bottom_fraction": 1.0,
    },
    "close-dark-withlight_Speed-3_2026-07-28_17.17.50": {
        "feature_roi_bottom_fraction": 1.0,
    },
}

MAP_EXPANSION_THRESHOLD_MULTIPLIER = 0.5


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_DISK_batch" / EXPERIMENT_NAME


def run_recording(recording_name, parameters):
    recording_output_dir = RESULTS_DIR / recording_name
    return run_tracking(
        recording_name,
        recording_output_dir,
        MAP_EXPANSION_THRESHOLD_MULTIPLIER,
        parameters["feature_roi_bottom_fraction"],
    )


def save_summary_csv(metrics):
    output_path = RESULTS_DIR / "rmse_summary.csv"
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


def save_rmse_summary_plot(metrics):
    recording_names = [item["recording"] for item in metrics]
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

    figure.suptitle(
        f"{EXPERIMENT_NAME}\n"
        "MAP_EXPANSION_THRESHOLD_MULTIPLIER = "
        f"{MAP_EXPANSION_THRESHOLD_MULTIPLIER}"
    )

    output_path = RESULTS_DIR / "rmse_summary.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = []
    for index, (recording_name, parameters) in enumerate(
        RECORDINGS.items(),
        start=1,
    ):
        print(
            f"\n[{index}/{len(RECORDINGS)}] "
            f"Running {recording_name}"
        )
        metrics.append(run_recording(recording_name, parameters))

    csv_path = save_summary_csv(metrics)
    plot_path = save_rmse_summary_plot(metrics)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
