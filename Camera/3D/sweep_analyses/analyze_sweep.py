"""Analiza i wykres wynikow run_camera_tracking_parameter_sweep.py.

Zmien ustawienia w sekcji USTAWIENIA ANALIZY, a nastepnie uruchom:

    python analyze_sweep.py
"""

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).absolute().parent

# ---------------------------------------------------------------------------
# USTAWIENIA ANALIZY
# ---------------------------------------------------------------------------

INPUT_CSV = SCRIPT_DIR / "all_combinations_summary.csv"
CHART_OUTPUT = SCRIPT_DIR / "analysis_chart.png"

# Przyklad: {"feature_type": "sift", "use_imu": False}
FILTERS = {}

# Jeden wiersz/slupek dla kazdej wartosci keyframe_interval.
# Wszystkie pozostale parametry i nagrania sa usredniane.
GROUP_BY = ["mapping_pair_settings"]

METRICS = [
    "tracked_percent",
    "position_rmse_mm",
    "orientation_rmse_deg",
    "map_build_wall_time_s",
]

# "mean" albo None (wszystkie pasujace wiersze bez agregacji).
AGGREGATION = "mean"
ONLY_SUCCESS = True
DECIMAL_PLACES = 4

# Mozna tu podac tylko wybrane pozycje z METRICS.
SHOW_CHART = True
CHART_KIND = "bar"  # "bar" albo "line"
CHART_METRICS = METRICS


GROUP_ALIASES = {
    "feature_limits": [
        "mapping_detected_max_features",
        "mapping_max_features",
        "tracking_max_features",
    ],
    "mapping_pair_settings": [
        "mapping_recent_pair_count",
        "mapping_recent_pair_interval",
        "mapping_motion_targets_px",
    ],
}

RAW_PARAMETER_COLUMNS = [
    "combination_index",
    "recording",
    "feature_type",
    "use_imu",
    "mapping_detected_max_features",
    "mapping_max_features",
    "tracking_max_features",
    "keyframe_interval",
    "mapping_recent_pair_count",
    "mapping_recent_pair_interval",
    "mapping_motion_targets_px",
]


def expand_groups(columns):
    expanded = []
    for column in columns:
        for actual_column in GROUP_ALIASES.get(column, [column]):
            if actual_column not in expanded:
                expanded.append(actual_column)
    return expanded


def comparable_value(value):
    """Normalizuje wartosci filtrow, w tym listy zapisane w CSV jako JSON."""
    if isinstance(value, str):
        stripped = value.strip()
        try:
            value = json.loads(
                stripped.lower()
                if stripped.lower() in {"true", "false"}
                else stripped
            )
        except (json.JSONDecodeError, TypeError):
            return stripped
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def validate_columns(data, requested):
    missing = [column for column in requested if column not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in input CSV: {', '.join(missing)}")


def read_filtered_data(path):
    data = pd.read_csv(path)
    if ONLY_SUCCESS and "status" in data.columns:
        data = data.loc[data["status"].eq("success")]

    for column, expected in FILTERS.items():
        validate_columns(data, [column])
        normalized = data[column].map(comparable_value)
        data = data.loc[normalized.eq(comparable_value(expected))]

    return data.copy()


def aggregate_data(data, group_columns):
    numeric = data.copy()
    for metric in METRICS:
        numeric[metric] = pd.to_numeric(numeric[metric], errors="raise")

    if group_columns:
        grouped = numeric.groupby(group_columns, dropna=False, sort=True)
        means = grouped[METRICS].mean()
        deviations = grouped[METRICS].std().fillna(0).add_suffix("_std")
        counts = grouped.size().rename("row_count")
        result = pd.concat([counts, means, deviations], axis=1).reset_index()
    else:
        result_values = {"row_count": len(numeric)}
        for metric in METRICS:
            result_values[metric] = numeric[metric].mean()
            result_values[f"{metric}_std"] = numeric[metric].std()
        result = pd.DataFrame([result_values]).fillna(0)

    numeric_result_columns = [
        column
        for column in result.columns
        if column == "row_count" or column in METRICS or column.endswith("_std")
    ]
    result[numeric_result_columns] = result[numeric_result_columns].round(
        DECIMAL_PLACES
    )
    return result


def raw_data(data):
    selected = [column for column in RAW_PARAMETER_COLUMNS if column in data.columns]
    selected.extend(metric for metric in METRICS if metric not in selected)
    return data.loc[:, selected].copy()


def chart_group_label(row, group_columns):
    if not group_columns:
        return "all"
    if len(group_columns) == 1:
        return str(row[group_columns[0]])
    return "\n".join(f"{column}={row[column]}" for column in group_columns)


def create_chart(results, group_columns):
    if results.empty:
        print("Chart not created because no rows match the filters.")
        return
    if CHART_KIND not in {"bar", "line"}:
        raise ValueError("CHART_KIND must be 'bar' or 'line'")

    unknown_metrics = [metric for metric in CHART_METRICS if metric not in METRICS]
    if unknown_metrics:
        raise ValueError(
            "CHART_METRICS must be included in METRICS: " + ", ".join(unknown_metrics)
        )
    if not CHART_METRICS:
        print("Chart not created because CHART_METRICS is empty.")
        return

    import matplotlib.pyplot as plt

    labels = [
        chart_group_label(row, group_columns) for _, row in results.iterrows()
    ]
    positions = list(range(len(results)))
    subplot_columns = 1 if len(CHART_METRICS) == 1 else 2
    subplot_rows = (len(CHART_METRICS) + subplot_columns - 1) // subplot_columns
    figure_width = max(9, min(20, 1.1 * len(results)))
    figure, axes = plt.subplots(
        subplot_rows,
        subplot_columns,
        figsize=(figure_width, 3.8 * subplot_rows),
        squeeze=False,
    )

    for axis, metric in zip(axes.flat, CHART_METRICS):
        values = results[metric].astype(float).to_numpy()
        if AGGREGATION == "mean":
            deviations = results[f"{metric}_std"].astype(float).to_numpy()
        else:
            deviations = [0.0] * len(results)

        if CHART_KIND == "bar":
            axis.bar(
                positions,
                values,
                yerr=deviations,
                capsize=5,
                error_kw={"elinewidth": 1.3, "capthick": 1.3},
            )
        else:
            axis.errorbar(
                positions,
                values,
                yerr=deviations,
                marker="o",
                capsize=5,
                elinewidth=1.3,
                capthick=1.3,
            )

        axis.set_title(f"{metric} (mean ± std)")
        axis.set_ylabel(metric)
        axis.set_xticks(positions, labels)
        axis.tick_params(axis="x", labelrotation=45 if len(results) > 4 else 0)
        for tick_label in axis.get_xticklabels():
            tick_label.set_horizontalalignment(
                "right" if len(results) > 4 else "center"
            )
        axis.grid(axis="y", alpha=0.3)

        lower_bounds = values - deviations
        upper_bounds = values + deviations
        lower_bound = min(lower_bounds)
        upper_bound = max(upper_bounds)
        span = upper_bound - lower_bound or abs(upper_bound) or 1
        axis.set_ylim(
            0 if lower_bound >= 0 else lower_bound - 0.08 * span,
            upper_bound + 0.18 * span,
        )

        if AGGREGATION == "mean":
            for position, value, deviation, row_count in zip(
                positions, values, deviations, results["row_count"]
            ):
                axis.annotate(
                    f"{value:.4g} ± {deviation:.3g}\nn={row_count}",
                    (position, value + deviation),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    for unused_axis in axes.flat[len(CHART_METRICS) :]:
        unused_axis.remove()

    filter_description = ", ".join(
        f"{column}={value}" for column, value in FILTERS.items()
    ) or "all successful rows"
    grouped_by = ", ".join(group_columns) or "all rows"
    figure.suptitle(
        f"Sweep analysis — grouped by {grouped_by}\nFilters: {filter_description}"
    )
    figure.tight_layout()
    CHART_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CHART_OUTPUT, dpi=160, bbox_inches="tight")
    print(f"Saved chart: {CHART_OUTPUT}")

    if SHOW_CHART:
        plt.show()
    else:
        plt.close(figure)


def main():
    data = read_filtered_data(INPUT_CSV)
    group_columns = expand_groups(GROUP_BY)
    validate_columns(data, group_columns + METRICS)

    if AGGREGATION == "mean":
        results = aggregate_data(data, group_columns)
        displayed_columns = group_columns + ["row_count"] + METRICS
    elif AGGREGATION is None:
        results = raw_data(data)
        displayed_columns = list(results.columns)
    else:
        raise ValueError("AGGREGATION must be 'mean' or None")

    print(f"Input rows after filtering: {len(data)}")
    if results.empty:
        print("No rows match the selected filters.")
    else:
        print(results.loc[:, displayed_columns].to_string(index=False))

    create_chart(results, group_columns)


if __name__ == "__main__":
    main()
