"""Compare IMU acceleration filters using final position error against GT.

For every selected recording the script:
1. loads and preprocesses IMU and Dobot ground-truth data,
2. synchronizes both streams using acceleration magnitude,
3. applies multiple acceleration filters and parameter variants,
4. integrates acceleration twice to obtain position,
5. evaluates position against GT and saves a chart plus a CSV table.

Edit the configuration variables near the top of this file and run:
    python compare_imu_filter_positions.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from funkcje_GT import (
    calculate_derivatives,
    load_ground_truth,
    normalize_ground_truth,
    resample_ground_truth,
    trim_ground_truth,
)
from funkcje_IMU import (
    calculate_integrals,
    compute_orientation_and_global_acc,
    load_IMU_data,
    remove_average_trend,
    resample_IMU_data,
    trim_IMU_data,
)
from funkcje_IMU_GT import synchronize_by_cross_correlation
from IMUFilter import IMUFilter


DATASETS = {
    "horizontal_line_1": {
        "imu": "IMU/dataLog00075.TXT",
        "gt": "dobot/horizontal_line_1774951923.csv",
    },
    "horizontal_line_2": {
        "imu": "IMU/dataLog00077.TXT",
        "gt": "dobot/horizontal_line_1774952687.csv",
    },
    "vertical_line_1": {
        "imu": "IMU/dataLog00079.TXT",
        "gt": "dobot/vertical_line_1774953045.csv",
    },
    "vertical_line_2": {
        "imu": "IMU/dataLog00081.TXT",
        "gt": "dobot/vertical_line_1774953360.csv",
    },
    "square_1": {
        "imu": "IMU/dataLog00083.TXT",
        "gt": "dobot/square_1774953674.csv",
    },
    "square_2": {
        "imu": "IMU/dataLog00085.TXT",
        "gt": "dobot/square_1774953882.csv",
    },
    "triangle_1": {
        "imu": "IMU/dataLog00087.TXT",
        "gt": "dobot/triangle_1774954203.csv",
    },
    "triangle_2": {
        "imu": "IMU/dataLog00089.TXT",
        "gt": "dobot/triangle_1774954436.csv",
    },
    "cross_1": {
        "imu": "IMU/dataLog00091.TXT",
        "gt": "dobot/cross_1774954750.csv",
    },
    "cross_2": {
        "imu": "IMU/dataLog00093.TXT",
        "gt": "dobot/cross_1774954990.csv",
    },
}

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Recordings to process. Available names are the keys in DATASETS above.
SELECTED_DATASETS = [
    "horizontal_line_1",
    "vertical_line_1",
    "square_1",
    "triangle_1",
    "cross_1",
]

# Set to True to ignore SELECTED_DATASETS and process every configured dataset.
PROCESS_ALL_DATASETS = True

DATA_ROOT = Path("Data")
OUTPUT_DIR = Path("filter_position_results")
SAMPLE_RATE = 100.0

IMU_TRIM_START = 1000
IMU_TRIM_END = 500
GT_TRIM_START = 200
GT_TRIM_END = 200

# False saves plots as PNG. True additionally opens each plot in a window.
SHOW_PLOTS = False


@dataclass(frozen=True)
class FilterVariant:
    family: str
    label: str
    apply: Callable[[np.ndarray], np.ndarray]


def build_filter_variants(filter_instance: IMUFilter) -> list[FilterVariant]:
    """Return a compact but diverse parameter grid for every available filter."""

    variants = [
        FilterVariant("raw", "raw", lambda values: values.copy()),
    ]

    for cutoff in (1.0, 2.0, 5.0, 10.0, 20.0):
        for order in (2, 4):
            variants.append(
                FilterVariant(
                    "butterworth",
                    f"butterworth cutoff={cutoff:g}Hz order={order}",
                    lambda values, c=cutoff, o=order: filter_instance.butterworth_filter(
                        values, cutoff=c, order=o
                    ),
                )
            )

    for window in (7, 11, 21, 41, 81):
        for polyorder in (2, 3):
            if polyorder < window:
                variants.append(
                    FilterVariant(
                        "savgol",
                        f"savgol window={window} poly={polyorder}",
                        lambda values, w=window, p=polyorder: filter_instance.savgol_filter(
                            values, window_length=w, polyorder=p
                        ),
                    )
                )

    for size in (3, 5, 11, 21, 41):
        variants.append(
            FilterVariant(
                "median",
                f"median size={size}",
                lambda values, s=size: filter_instance.median_filter_func(values, size=s),
            )
        )

    for q_value in (1e-6, 1e-4, 1e-2):
        for r_value in (1e-4, 1e-3, 1e-2):
            variants.append(
                FilterVariant(
                    "kalman_1d",
                    f"kalman Q={q_value:.0e} R={r_value:.0e}",
                    lambda values, q=q_value, r=r_value: filter_instance.kalman_filter(
                        values, Q=q, R=r
                    ),
                )
            )

    for window in (5, 11, 21, 41, 81):
        variants.append(
            FilterVariant(
                "moving_average",
                f"moving-average window={window}",
                lambda values, w=window: filter_instance.moving_average_filter(
                    values, window=w
                ),
            )
        )

    for window in (5, 11, 21, 41, 81):
        variants.append(
            FilterVariant(
                "wiener",
                f"wiener window={window}",
                lambda values, w=window: filter_instance.wiener_filter(values, window=w),
            )
        )

    return variants


def prepare_synchronized_data(
    data_root: Path,
    paths: dict[str, str],
    sample_rate: float,
    imu_trim_start: int,
    imu_trim_end: int,
    gt_trim_start: int,
    gt_trim_end: int,
) -> pd.DataFrame:
    """Load, preprocess and synchronize one IMU/GT recording pair."""

    df_imu = load_IMU_data(data_root / paths["imu"])
    df_imu = compute_orientation_and_global_acc(df_imu)
    df_imu = trim_IMU_data(df_imu, imu_trim_start, imu_trim_end)
    df_imu = resample_IMU_data(df_imu, target_fps=sample_rate)
    df_imu = remove_average_trend(df_imu)

    df_gt = load_ground_truth(data_root / paths["gt"])
    df_gt = trim_ground_truth(df_gt, gt_trim_start, gt_trim_end)
    df_gt = normalize_ground_truth(df_gt)
    df_gt = resample_ground_truth(df_gt, target_fps=sample_rate)
    df_gt = calculate_derivatives(df_gt, target_fps=sample_rate)

    synchronized = synchronize_by_cross_correlation(df_imu, df_gt)

    # IMU integration starts at p(0)=0. Re-zero GT at the synchronized start so
    # the metric measures trajectory error rather than an arbitrary origin shift.
    for axis in "xyz":
        gt_column = f"gt_pos_{axis}"
        synchronized[gt_column] -= synchronized[gt_column].iloc[0]

    return synchronized


def evaluate_variant(
    synchronized: pd.DataFrame,
    variant: FilterVariant,
    sample_rate: float,
) -> dict[str, float | str]:
    """Filter all axes, integrate them and calculate position error metrics."""

    method = "candidate"
    candidate = synchronized.copy()

    for axis in "xyz":
        source = candidate[f"gl_acc_{axis}"].to_numpy(dtype=float)
        candidate[f"{method}_acc_{axis}"] = variant.apply(source)

    integrated = calculate_integrals(
        candidate,
        method=method,
        target_fps=sample_rate,
        use_detrend_vel=False,
        use_detrend_pos=False,
        use_loop_closure=False,
        apply_hp_filter=False,
    )

    estimated = integrated[[f"{method}_pos_{axis}" for axis in "xyz"]].to_numpy()
    ground_truth = integrated[[f"gt_pos_{axis}" for axis in "xyz"]].to_numpy()
    error = estimated - ground_truth

    axis_rmse_mm = np.sqrt(np.mean(error**2, axis=0)) * 1000.0
    distance_error = np.linalg.norm(error, axis=1)
    rmse_3d_mm = np.sqrt(np.mean(distance_error**2)) * 1000.0
    endpoint_error_mm = distance_error[-1] * 1000.0

    return {
        "family": variant.family,
        "filter": variant.label,
        "rmse_3d_mm": float(rmse_3d_mm),
        "rmse_x_mm": float(axis_rmse_mm[0]),
        "rmse_y_mm": float(axis_rmse_mm[1]),
        "rmse_z_mm": float(axis_rmse_mm[2]),
        "endpoint_error_mm": float(endpoint_error_mm),
    }


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def plot_recording_results(
    dataset_name: str,
    results: pd.DataFrame,
    output_path: Path,
    show: bool,
) -> None:
    """Create one sorted horizontal RMSE chart for a recording."""

    ordered = results.sort_values("rmse_3d_mm", ascending=True).reset_index(drop=True)
    colors_by_family = {
        family: plt.get_cmap("tab10")(index % 10)
        for index, family in enumerate(ordered["family"].drop_duplicates())
    }
    colors = [colors_by_family[family] for family in ordered["family"]]

    height = max(8.0, 0.27 * len(ordered))
    fig, ax = plt.subplots(figsize=(13, height))
    positions = np.arange(len(ordered))
    bars = ax.barh(positions, ordered["rmse_3d_mm"], color=colors)
    ax.set_yticks(positions, ordered["filter"])
    ax.invert_yaxis()
    ax.set_xlabel("3D position RMSE [mm]")
    ax.set_title(f"{dataset_name}: position error by IMU acceleration filter")
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, ordered["rmse_3d_mm"]):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.1f}",
            va="center",
            fontsize=8,
        )

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=7, label=family)
        for family, color in colors_by_family.items()
    ]
    ax.legend(handles=legend_handles, title="Filter family", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_combined_results(
    combined: pd.DataFrame,
    output_path: Path,
    show: bool,
) -> pd.DataFrame:
    """Plot mean position RMSE and its spread across all processed recordings."""

    summary = (
        combined.groupby(["family", "filter"], as_index=False)
        .agg(
            mean_rmse_3d_mm=("rmse_3d_mm", "mean"),
            std_rmse_3d_mm=("rmse_3d_mm", "std"),
            min_rmse_3d_mm=("rmse_3d_mm", "min"),
            max_rmse_3d_mm=("rmse_3d_mm", "max"),
            recording_count=("dataset", "nunique"),
        )
        .sort_values("mean_rmse_3d_mm", ascending=True)
        .reset_index(drop=True)
    )
    summary["std_rmse_3d_mm"] = summary["std_rmse_3d_mm"].fillna(0.0)

    colors_by_family = {
        family: plt.get_cmap("tab10")(index % 10)
        for index, family in enumerate(summary["family"].drop_duplicates())
    }
    bar_colors = [colors_by_family[family] for family in summary["family"]]
    positions = np.arange(len(summary))

    height = max(8.0, 0.29 * len(summary))
    fig, ax = plt.subplots(figsize=(14, height))
    bars = ax.barh(
        positions,
        summary["mean_rmse_3d_mm"],
        xerr=summary["std_rmse_3d_mm"],
        color=bar_colors,
        alpha=0.82,
        error_kw={"ecolor": "0.25", "elinewidth": 1.0, "capsize": 2},
        label="Mean RMSE",
    )

    # Add every recording result as a point on the corresponding filter row.
    row_by_filter = {name: index for index, name in enumerate(summary["filter"])}
    for filter_name, group in combined.groupby("filter"):
        y = row_by_filter[filter_name]
        ax.scatter(
            group["rmse_3d_mm"],
            np.full(len(group), y),
            s=16,
            color="black",
            alpha=0.55,
            zorder=3,
        )

    ax.set_yticks(positions, summary["filter"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean 3D position RMSE across recordings [mm]")
    ax.set_title(
        "All scenarios: mean position error by IMU acceleration filter\n"
        "error bars = standard deviation, dots = individual recordings"
    )
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, summary["mean_rmse_3d_mm"]):
        ax.text(
            max(0.0, bar.get_width() - 6.0),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="right",
            va="center",
            fontsize=8,
            color="white",
        )

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=7, label=family)
        for family, color in colors_by_family.items()
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="none",
            markersize=4,
            label="individual recording",
        )
    )
    ax.legend(handles=legend_handles, title="Filter family", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return summary


def evaluate_recording(
    dataset_name: str,
    data_root: Path,
    output_dir: Path,
    sample_rate: float,
    imu_trim_start: int,
    imu_trim_end: int,
    gt_trim_start: int,
    gt_trim_end: int,
    show: bool,
) -> pd.DataFrame:
    print(f"\n=== {dataset_name} ===")
    synchronized = prepare_synchronized_data(
        data_root=data_root,
        paths=DATASETS[dataset_name],
        sample_rate=sample_rate,
        imu_trim_start=imu_trim_start,
        imu_trim_end=imu_trim_end,
        gt_trim_start=gt_trim_start,
        gt_trim_end=gt_trim_end,
    )
    print(f"Synchronized samples: {len(synchronized)}")

    filter_instance = IMUFilter(fs=sample_rate)
    variants = build_filter_variants(filter_instance)
    rows = []
    for index, variant in enumerate(variants, start=1):
        print(f"[{index:02d}/{len(variants):02d}] {variant.label}")
        rows.append(evaluate_variant(synchronized, variant, sample_rate))

    results = pd.DataFrame(rows).sort_values("rmse_3d_mm").reset_index(drop=True)
    stem = safe_filename(dataset_name)
    csv_path = output_dir / f"{stem}_filter_position_metrics.csv"
    plot_path = output_dir / f"{stem}_filter_position_rmse.png"
    results.to_csv(csv_path, index=False)
    plot_recording_results(dataset_name, results, plot_path, show=show)

    best = results.iloc[0]
    print(f"Best: {best['filter']} -> {best['rmse_3d_mm']:.2f} mm")
    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")
    return results


def validate_datasets(names: Iterable[str]) -> list[str]:
    selected = list(names)
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        choices = ", ".join(DATASETS)
        raise ValueError(f"Unknown datasets: {unknown}. Available: {choices}")
    return selected


def main() -> None:
    if PROCESS_ALL_DATASETS:
        selected = list(DATASETS)
    else:
        selected = validate_datasets(SELECTED_DATASETS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for dataset_name in selected:
        results = evaluate_recording(
            dataset_name=dataset_name,
            data_root=DATA_ROOT,
            output_dir=OUTPUT_DIR,
            sample_rate=SAMPLE_RATE,
            imu_trim_start=IMU_TRIM_START,
            imu_trim_end=IMU_TRIM_END,
            gt_trim_start=GT_TRIM_START,
            gt_trim_end=GT_TRIM_END,
            show=SHOW_PLOTS,
        )
        result_with_dataset = results.copy()
        result_with_dataset.insert(0, "dataset", dataset_name)
        all_results.append(result_with_dataset)

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = OUTPUT_DIR / "all_filter_position_metrics.csv"
    combined.to_csv(combined_path, index=False)

    combined_plot_path = OUTPUT_DIR / "all_scenarios_filter_position_rmse.png"
    summary = plot_combined_results(
        combined,
        output_path=combined_plot_path,
        show=SHOW_PLOTS,
    )
    summary_path = OUTPUT_DIR / "all_scenarios_filter_position_summary.csv"
    summary.to_csv(summary_path, index=False)

    best = summary.iloc[0]
    print(f"\nBest filter across all scenarios: {best['filter']}")
    print(f"Mean 3D RMSE: {best['mean_rmse_3d_mm']:.2f} mm")
    print(f"Combined results saved: {combined_path}")
    print(f"Combined summary saved: {summary_path}")
    print(f"Combined plot saved: {combined_plot_path}")


if __name__ == "__main__":
    main()
