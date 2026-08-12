import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from camera_tracking import run_tracking


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------

EXPERIMENT_NAME = "realistic_screen_visible_0.7_expansion_threshold_newExpanding"
DATA_FOLDER = "LineArc-1-2cm"
FEATURE_TYPE = "sift"  # "disk" or "sift".

RECORDINGS = {
    "arc1cm-close-dark-nolight_Speed-3_2026-07-29_16.12.21": {
        "feature_roi_bottom_fraction": 1.00,
    },
    "arc1cm-close-dark-withlight_Speed-3_2026-07-29_16.08.30": {
        "feature_roi_bottom_fraction": 1.00,
    },
    "arc1cm-close-white-nolight_Speed-3_2026-07-29_16.29.21": {
        "feature_roi_bottom_fraction": 1.00,
    },
    "arc1cm-close-white-withlight_Speed-3_2026-07-29_16.32.22": {
        "feature_roi_bottom_fraction": 1.00,
    },
    "arc1cm-far-dark-nolight_Speed-3_2026-07-29_16.52.01": {
        "feature_roi_bottom_fraction": 0.70,
    },
    "arc1cm-far-dark-withlight_Speed-3_2026-07-29_16.53.23": {
        "feature_roi_bottom_fraction": 0.70,
    },
    "arc1cm-far-white-nolight_Speed-3_2026-07-29_17.03.56": {
        "feature_roi_bottom_fraction": 0.70,
    },
    "arc1cm-far-white-withlight_Speed-3_2026-07-29_17.02.26": {
        "feature_roi_bottom_fraction": 0.70,
    },
    "arc1cm-initial-dark-nolight_Speed-3_2026-07-29_16.48.44": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "arc1cm-initial-dark-withlight_Speed-3_2026-07-29_16.47.01": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "arc1cm-initial-white-nolight_Speed-3_2026-07-29_16.35.50": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "arc1cm-initial-white-withlight_Speed-3_2026-07-29_16.34.29": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "arc2cm-close-dark-nolight_Speed-3_2026-07-29_16.14.42": {
        "feature_roi_bottom_fraction": 1.00,
    },
    "arc2cm-close-dark-withlight_Speed-3_2026-07-29_16.15.59": {
        "feature_roi_bottom_fraction": 1.00,
    },
    "arc2cm-close-white-nolight_Speed-3_2026-07-29_16.27.38": {
        "feature_roi_bottom_fraction": 1.00,
    },
    "arc2cm-close-white-withlight_Speed-3_2026-07-29_16.25.40": {
        "feature_roi_bottom_fraction": 1.00,
    },
    "arc2cm-far-dark-nolight_Speed-3_2026-07-29_16.56.25": {
        "feature_roi_bottom_fraction": 0.70,
    },
    "arc2cm-far-dark-withlight_Speed-3_2026-07-29_16.55.05": {
        "feature_roi_bottom_fraction": 0.70,
    },
    "arc2cm-far-white-nolight_Speed-3_2026-07-29_16.59.01": {
        "feature_roi_bottom_fraction": 0.70,
    },
    "arc2cm-far-white-withlight_Speed-3_2026-07-29_17.00.49": {
        "feature_roi_bottom_fraction": 0.70,
    },
    "arc2cm-initial-dark-nolight_Speed-3_2026-07-29_16.44.19": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "arc2cm-initial-dark-withlight_Speed-3_2026-07-29_16.45.45": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "arc2cm-initial-white-nolight_Speed-3_2026-07-29_16.37.23": {
        "feature_roi_bottom_fraction": 0.85,
    },
    "arc2cm-initial-white-withlight_Speed-3_2026-07-29_16.38.41": {
        "feature_roi_bottom_fraction": 0.85,
    },
}

# RECORDINGS = {
#     "initialpos-white-withlight_Speed-3_2026-07-29_17.46.25": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "initialpos-white-nolight_Speed-3_2026-07-29_17.47.53": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "initialpos-dark-nolight_Speed-3_2026-07-28_16.55.02": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "initialpos-dark-withlight_Speed-3_2026-07-28_16.57.56": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "far-white-nolight_Speed-3_2026-07-28_17.06.45": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "far-white-withlight_Speed-3_2026-07-28_17.08.22": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "far-dark-withlight_Speed-3_2026-07-28_17.02.52": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "far-dark-nolight_Speed-3_2026-07-28_17.04.19": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "close-white-withlight_Speed-3_2026-07-28_17.12.37": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
#     "close-white-nolight_Speed-3_2026-07-28_17.14.02": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
#     "close-dark-nolight_Speed-3_2026-07-28_17.16.20": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
#     "close-dark-withlight_Speed-3_2026-07-28_17.17.50": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
# }

# RECORDINGS = {
#     "initial-white-withlight-25deg_Speed-3_2026-07-30_13.27.38": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "initial-white-nolight-25deg_Speed-3_2026-07-30_13.28.57": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "initial-black-withlight-25deg_Speed-3_2026-07-30_13.46.33": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "initial-black-nolight-25deg_Speed-3_2026-07-30_13.50.30": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "far-white-withlight-25deg_Speed-3_2026-07-30_13.21.53": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "far-white-nolight-25deg_Speed-3_2026-07-30_13.20.24": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "far-black-withlight-25deg_Speed-3_2026-07-30_13.44.45": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "far-black-nolight-25deg_Speed-3_2026-07-30_13.41.36": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "close-white-withlight-25deg_Speed-3_2026-07-30_13.33.14": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
#     "close-white-nolight-25deg_Speed-3_2026-07-30_13.31.23": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
#     "close-black-withlight-25deg_Speed-3_2026-07-30_13.37.05": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
#     "close-black-nolight-25deg_Speed-3_2026-07-30_13.38.42": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
# }


# RECORDINGS = {
#     "initial-white-withlight-25deg_Speed-3_2026-07-30_13.06.03": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "initial-white-nolight-25deg_Speed-3_2026-07-30_13.07.33": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "initial-black-withlight-25deg_Speed-3_2026-07-30_13.56.04": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "initial-black-nolight-25deg_Speed-3_2026-07-30_13.55.04": {
#         "feature_roi_bottom_fraction": 0.85,
#     },
#     "far-white-withlight-25deg_Speed-3_2026-07-30_13.13.13": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "far-white-nolight-25deg_Speed-3_2026-07-30_13.14.23": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "far-black-withlight-25deg_Speed-3_2026-07-30_13.58.22": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "far-black-nolight-25deg_Speed-3_2026-07-30_14.00.25": {
#         "feature_roi_bottom_fraction": 0.70,
#     },
#     "close-white-withlight-25deg_Speed-3_2026-07-30_13.11.32": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
#     "close-white-nolight-25deg_Speed-3_2026-07-30_13.10.31": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
#     "close-black-withlight-25deg_Speed-3_2026-07-30_14.09.41": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
#     "close-black-nolight-25deg_Speed-3_2026-07-30_14.08.24": {
#         "feature_roi_bottom_fraction": 1.00,
#     },
# }

MAP_EXPANSION_MIN_COVERAGE_RATIO = 0.70


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = (
    SCRIPT_DIR
    / f"results_{FEATURE_TYPE.upper()}_batch"
    / DATA_FOLDER
    / EXPERIMENT_NAME
)


def run_recording(recording_name, parameters):
    recording_output_dir = RESULTS_DIR / recording_name
    return run_tracking(
        recording_name,
        recording_output_dir,
        SCRIPT_DIR.parent / "Data" / DATA_FOLDER,
        MAP_EXPANSION_MIN_COVERAGE_RATIO,
        parameters["feature_roi_bottom_fraction"],
        FEATURE_TYPE,
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
        if not math.isfinite(value):
            continue
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

    figure.suptitle(
        f"{EXPERIMENT_NAME}\n"
        "MAP_EXPANSION_MIN_COVERAGE_RATIO = "
        f"{MAP_EXPANSION_MIN_COVERAGE_RATIO}"
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
