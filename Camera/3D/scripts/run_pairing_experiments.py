"""Compare keyframe-pairing configurations on multiple recording folders.

The experiment configuration contains a list of pairing variants and a list
of datasets. Every variant is evaluated on every listed recording. Results are
saved as a flat CSV and as grouped plots, so a configuration can be selected
across recordings instead of from one run only.
"""

import argparse
import csv
import json
import math
import shutil
import sys
import traceback
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
THREE_D_DIRECTORY = SCRIPT_DIRECTORY.parent
CAMERA_DIRECTORY = THREE_D_DIRECTORY.parent
PROJECT_DIRECTORY = CAMERA_DIRECTORY.parent
if str(THREE_D_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(THREE_D_DIRECTORY))

import matplotlib.pyplot as plt
import numpy as np

from camera_tracking import run_tracking


DEFAULT_CONFIG_PATH = (
    THREE_D_DIRECTORY
    / "batch_configs_from_timestamps"
    / "pairing_experiment_config.example.json"
)
FRAME_RANGE_KEYS = (
    "mapping_start_frame",
    "mapping_end_frame",
    "tracking_start_frame",
)
RECORDING_DISTANCE_NAMES = ("close", "initial", "far")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple keyframe-pairing configurations on multiple "
            "recording folders."
        )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing successful metrics.json for each run.",
    )
    return parser.parse_args()


def load_config(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def require_fields(mapping, fields, context):
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context} is missing: {', '.join(missing)}")


def validate_frame_range(recording, context):
    require_fields(recording, FRAME_RANGE_KEYS, context)
    frame_range = {
        key: recording[key]
        for key in FRAME_RANGE_KEYS
    }
    for key, value in frame_range.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{context}.{key} must be a non-negative integer")
    if frame_range["mapping_start_frame"] > frame_range["mapping_end_frame"]:
        raise ValueError(f"{context} has an invalid mapping frame range")
    if frame_range["tracking_start_frame"] <= frame_range["mapping_end_frame"]:
        raise ValueError(
            f"{context}.tracking_start_frame must be after map construction"
        )
    return frame_range


def validate_pairing_variant(variant, context):
    require_fields(
        variant,
        ("name", "mapping_recent_pair_count", "mapping_motion_targets_px"),
        context,
    )
    recent_pair_count = variant["mapping_recent_pair_count"]
    if (
        not isinstance(recent_pair_count, int)
        or isinstance(recent_pair_count, bool)
        or recent_pair_count < 0
    ):
        raise ValueError(
            f"{context}.mapping_recent_pair_count must be a non-negative integer"
        )

    targets = tuple(float(target) for target in variant["mapping_motion_targets_px"])
    if any(not math.isfinite(target) or target <= 0.0 for target in targets):
        raise ValueError(
            f"{context}.mapping_motion_targets_px must contain positive numbers"
        )
    if len(set(targets)) != len(targets):
        raise ValueError(
            f"{context}.mapping_motion_targets_px must not contain duplicates"
        )
    return {
        "name": str(variant["name"]),
        "mapping_recent_pair_count": recent_pair_count,
        "mapping_motion_targets_px": targets,
    }


def validate_config(config):
    require_fields(
        config,
        (
            "experiment_name",
            "feature_type",
            "use_imu",
            "mapping_frame_step",
            "matching_variants",
            "datasets",
        ),
        "experiment configuration",
    )
    if not config["matching_variants"]:
        raise ValueError("matching_variants must not be empty")
    if not config["datasets"]:
        raise ValueError("datasets must not be empty")

    distance_names = config.get("recording_distances")
    if distance_names is not None:
        if not isinstance(distance_names, list) or not distance_names:
            raise ValueError(
                "recording_distances must be a non-empty list when provided"
            )
        distance_names = tuple(str(name).lower() for name in distance_names)
        unknown_names = set(distance_names) - set(RECORDING_DISTANCE_NAMES)
        if unknown_names:
            raise ValueError(
                "recording_distances supports only: "
                + ", ".join(RECORDING_DISTANCE_NAMES)
            )
        if len(set(distance_names)) != len(distance_names):
            raise ValueError("recording_distances must not contain duplicates")

    variants = [
        validate_pairing_variant(variant, f"matching_variants[{index}]")
        for index, variant in enumerate(config["matching_variants"])
    ]
    variant_names = [variant["name"] for variant in variants]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError("matching variant names must be unique")

    datasets = []
    for dataset_index, dataset in enumerate(config["datasets"]):
        context = f"datasets[{dataset_index}]"
        require_fields(dataset, ("name", "data_folder", "recordings"), context)
        if not dataset["recordings"]:
            raise ValueError(f"{context}.recordings must not be empty")
        recordings = []
        for recording_name, recording in dataset["recordings"].items():
            recording_context = f"{context}.recordings.{recording_name}"
            require_fields(
                recording,
                ("feature_roi_bottom_fraction",),
                recording_context,
            )
            recordings.append(
                {
                    "name": recording_name,
                    "feature_roi_bottom_fraction": recording[
                        "feature_roi_bottom_fraction"
                    ],
                    **validate_frame_range(recording, recording_context),
                }
            )
        datasets.append(
            {
                "name": str(dataset["name"]),
                "data_folder": str(dataset["data_folder"]),
                "recordings": recordings,
            }
        )

    return {
        "experiment_name": str(config["experiment_name"]),
        "feature_type": str(config["feature_type"]),
        "use_imu": bool(config["use_imu"]),
        "mapping_frame_step": config["mapping_frame_step"],
        "reconstruction_method": config.get("reconstruction_method", "global"),
        "matching_variants": variants,
        "datasets": datasets,
        "recording_distances": distance_names,
    }


def select_recordings_by_distance(datasets, distance_names):
    """Keep recordings whose file name contains one requested distance name."""
    if distance_names is None:
        return datasets

    selected_datasets = []
    for dataset in datasets:
        recordings = [
            recording
            for recording in dataset["recordings"]
            if any(
                distance_name in recording["name"].lower()
                for distance_name in distance_names
            )
        ]
        if recordings:
            selected_datasets.append({**dataset, "recordings": recordings})
    return selected_datasets


def resolve_results_directory(config, override):
    if override is not None:
        return override
    return (
        THREE_D_DIRECTORY
        / "results_pairing_experiments"
        / config["experiment_name"]
    )


def run_directory(results_directory, variant, dataset, recording):
    return (
        results_directory
        / variant["name"]
        / dataset["data_folder"]
        / recording["name"]
    )


def load_completed_metrics(output_directory):
    metrics_path = output_directory / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as file:
        return json.load(file)


def execute_run(variant, dataset, recording, config, output_directory, resume):
    output_directory.mkdir(parents=True, exist_ok=True)
    if resume:
        metrics = load_completed_metrics(output_directory)
        if metrics is not None:
            print(f"Reusing completed result: {output_directory}")
            return metrics

    data_directory = PROJECT_DIRECTORY / "Data" / dataset["data_folder"]
    return run_tracking(
        recording["name"],
        output_directory,
        data_directory,
        recording["feature_roi_bottom_fraction"],
        reconstruction_method=config["reconstruction_method"],
        mapping_start_frame=recording["mapping_start_frame"],
        mapping_end_frame=recording["mapping_end_frame"],
        tracking_start_frame=recording["tracking_start_frame"],
        feature_type=config["feature_type"],
        mapping_frame_step=config["mapping_frame_step"],
        mapping_recent_pair_count=variant["mapping_recent_pair_count"],
        mapping_motion_targets_px=variant["mapping_motion_targets_px"],
        use_imu=config["use_imu"],
    )


def read_pairing_statistics(output_directory):
    diagnostics_path = output_directory / "mapping_pipeline_diagnostics.csv"
    if not diagnostics_path.exists():
        return {}

    with diagnostics_path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        attempted_column = (
            "attempted_pairs"
            if "attempted_pairs" in rows.fieldnames
            else "pairs_attempted"
        )
        verified_column = (
            "verified_pairs"
            if "verified_pairs" in rows.fieldnames
            else "pairs_verified"
        )
        attempted_pairs = 0
        verified_pairs = 0
        for row in rows:
            attempted_pairs += int(row[attempted_column])
            verified_pairs += int(row[verified_column])
    return {
        "attempted_image_pairs": attempted_pairs,
        "verified_image_pairs": verified_pairs,
    }


def serialize_row(
    variant,
    dataset,
    recording,
    metrics,
    pairing_statistics,
    status,
    error=None,
):
    row = {
        "variant": variant["name"],
        "recent_pair_count": variant["mapping_recent_pair_count"],
        "motion_targets_px": json.dumps(variant["mapping_motion_targets_px"]),
        "dataset": dataset["name"],
        "data_folder": dataset["data_folder"],
        "recording": recording["name"],
        "status": status,
    }
    if metrics is not None:
        row.update(metrics)
    row.update(pairing_statistics)
    if error is not None:
        row["error"] = error
    return row


def save_summary_csv(rows, results_directory):
    output_path = results_directory / "pairing_summary.csv"
    fieldnames = list(
        dict.fromkeys(field for row in rows for field in row)
    )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def successful_rows(rows):
    required_metrics = (
        "mapping_position_rmse_mm",
        "mapping_orientation_rmse_deg",
        "map_build_wall_time_s",
        "attempted_image_pairs",
    )
    return [
        row
        for row in rows
        if row["status"] == "success"
        and all(metric in row and math.isfinite(float(row[metric])) for metric in required_metrics)
    ]


def ordered_recording_labels(rows):
    seen = set()
    labels = []
    for row in rows:
        label = f"{row['dataset']}\n{row['recording']}"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def add_bar_labels(axis, bars, unit):
    for bar in bars:
        value = bar.get_height()
        if not math.isfinite(value):
            continue
        axis.annotate(
            f"{value:.2f}{unit}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )


def plot_grouped_metric(axis, rows, variants, recording_labels, metric, title, unit):
    label_index = {label: index for index, label in enumerate(recording_labels)}
    variant_count = len(variants)
    width = 0.82 / variant_count
    centers = np.arange(len(recording_labels))
    palette = plt.get_cmap("tab10")

    for variant_index, variant in enumerate(variants):
        values = np.full(len(recording_labels), np.nan)
        for row in rows:
            if row["variant"] != variant["name"]:
                continue
            label = f"{row['dataset']}\n{row['recording']}"
            values[label_index[label]] = float(row[metric])
        offset = (variant_index - (variant_count - 1) / 2) * width
        bars = axis.bar(
            centers + offset,
            values,
            width=width,
            label=variant["name"],
            color=palette(variant_index % 10),
        )
        add_bar_labels(axis, bars, unit)

    axis.set_title(title)
    axis.set_ylabel(unit.strip("[]"))
    axis.set_xticks(centers)
    axis.set_xticklabels(recording_labels, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)


def save_comparison_plot(rows, variants, results_directory):
    rows = successful_rows(rows)
    if not rows:
        return None

    recording_labels = ordered_recording_labels(rows)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(max(16, 3.4 * len(recording_labels)), 13),
        constrained_layout=True,
    )
    plot_grouped_metric(
        axes[0, 0],
        rows,
        variants,
        recording_labels,
        "mapping_position_rmse_mm",
        "Mapping position RMSE",
        "mm",
    )
    plot_grouped_metric(
        axes[0, 1],
        rows,
        variants,
        recording_labels,
        "mapping_orientation_rmse_deg",
        "Mapping orientation RMSE",
        "deg",
    )
    plot_grouped_metric(
        axes[1, 0],
        rows,
        variants,
        recording_labels,
        "attempted_image_pairs",
        "Image pairs matched",
        "count",
    )
    plot_grouped_metric(
        axes[1, 1],
        rows,
        variants,
        recording_labels,
        "map_build_wall_time_s",
        "Map-build wall time",
        "s",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(variants))
    figure.suptitle("Keyframe-pairing comparison", y=1.02)

    output_path = results_directory / "pairing_comparison.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_variant_summary_plot(rows, variants, results_directory):
    rows = successful_rows(rows)
    if not rows:
        return None

    figure, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    metrics = (
        ("mapping_position_rmse_mm", "Median position RMSE", "mm"),
        ("mapping_orientation_rmse_deg", "Median orientation RMSE", "deg"),
        ("map_build_wall_time_s", "Median map-build time", "s"),
    )

    for axis, (metric, title, unit) in zip(axes, metrics):
        values = []
        for variant in variants:
            matching = [
                float(row[metric])
                for row in rows
                if row["variant"] == variant["name"]
            ]
            values.append(float(np.median(matching)) if matching else np.nan)
        bars = axis.bar(
            [variant["name"] for variant in variants],
            values,
            color=[palette(index % 10) for index in range(len(variants))],
        )
        add_bar_labels(axis, bars, unit)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=35)

    output_path = results_directory / "pairing_variant_medians.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    arguments = parse_arguments()
    raw_config = load_config(arguments.config)
    config = validate_config(raw_config)
    datasets = select_recordings_by_distance(
        config["datasets"],
        config["recording_distances"],
    )
    if not datasets:
        raise ValueError(
            "recording_distances did not select any recordings from datasets"
        )
    results_directory = resolve_results_directory(config, arguments.output_dir)
    results_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(arguments.config, results_directory / "config.json")

    total_runs = sum(
        len(dataset["recordings"])
        for dataset in datasets
    ) * len(config["matching_variants"])
    rows = []
    run_index = 0

    for variant in config["matching_variants"]:
        for dataset in datasets:
            for recording in dataset["recordings"]:
                run_index += 1
                print(
                    f"\n[{run_index}/{total_runs}] {variant['name']} | "
                    f"{dataset['name']} | {recording['name']} | "
                    f"recent={variant['mapping_recent_pair_count']} | "
                    f"motion={variant['mapping_motion_targets_px']}"
                )
                output_directory = run_directory(
                    results_directory,
                    variant,
                    dataset,
                    recording,
                )
                try:
                    metrics = execute_run(
                        variant,
                        dataset,
                        recording,
                        config,
                        output_directory,
                        arguments.resume,
                    )
                    rows.append(
                        serialize_row(
                            variant,
                            dataset,
                            recording,
                            metrics,
                            read_pairing_statistics(output_directory),
                            status="success",
                        )
                    )
                except Exception as error:
                    traceback.print_exc()
                    rows.append(
                        serialize_row(
                            variant,
                            dataset,
                            recording,
                            metrics=None,
                            pairing_statistics={},
                            status="failed",
                            error=f"{type(error).__name__}: {error}",
                        )
                    )

                save_summary_csv(rows, results_directory)

    csv_path = save_summary_csv(rows, results_directory)
    comparison_plot_path = save_comparison_plot(
        rows,
        config["matching_variants"],
        results_directory,
    )
    median_plot_path = save_variant_summary_plot(
        rows,
        config["matching_variants"],
        results_directory,
    )

    print(f"\nSaved: {csv_path}")
    if comparison_plot_path is not None:
        print(f"Saved: {comparison_plot_path}")
    if median_plot_path is not None:
        print(f"Saved: {median_plot_path}")


if __name__ == "__main__":
    main()
