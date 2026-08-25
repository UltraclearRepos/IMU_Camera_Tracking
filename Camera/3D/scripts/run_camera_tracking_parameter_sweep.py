import argparse
import csv
import itertools
import json
import sys
import tempfile
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).absolute().parent
MODULE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MODULE_DIR.parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
DEFAULT_CONFIG_PATH = (
    MODULE_DIR / "batch_configs_from_timestamps" / "Cylinder_parameter_sweep.json"
)
RESULT_KEYS = (
    "tracked_percent",
    "position_rmse_mm",
    "orientation_rmse_deg",
    "map_build_wall_time_s",
    "mapping_online_wall_time_s",
    "mapping_offline_wall_time_s",
    "tracking_wall_time_s",
)


def load_config(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def parameter_combinations(config):
    grid = config["parameter_grid"]
    combinations = itertools.product(
        grid["feature_type"],
        grid["use_imu"],
        grid["feature_limits"],
        grid["keyframe_interval"],
        grid["mapping_pair_settings"],
    )

    for feature_type, use_imu, limits, keyframe_interval, pairs in combinations:
        yield {
            "feature_type": feature_type,
            "use_imu": use_imu,
            **limits,
            "keyframe_interval": keyframe_interval,
            **pairs,
        }


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def save_csv(rows, path):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run camera tracking for all parameter combinations."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    config = load_config(arguments.config)
    combinations = list(parameter_combinations(config))
    recordings = config["recordings"]
    total_runs = len(combinations) * len(recordings)

    print(
        f"Combinations: {len(combinations)}, recordings: {len(recordings)}, "
        f"total runs: {total_runs}"
    )
    if arguments.dry_run:
        for index, parameters in enumerate(combinations, start=1):
            print(index, parameters)
        return

    # Import the tracking stack only for a real run.
    import camera_tracking
    from run_camera_tracking_batch import run_recording

    camera_tracking.SAVE_DIAGNOSTIC_VIDEO = False
    camera_tracking.SAVE_MAPPING_FEATURE_VIDEO = False
    camera_tracking.SAVE_MAP_BUILD_TOP_VIEW = False
    camera_tracking.SAVE_TRACKING_TOP_VIEW = False
    camera_tracking.SHOW_PREVIEW = False

    data_dir = PROJECT_DIR / "Data" / config["data_folder"]
    output_dir = arguments.output_dir or (
        MODULE_DIR
        / "results_parameter_sweep"
        / config["data_folder"]
        / config["experiment_name"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    run_index = 0
    for combination_index, parameters in enumerate(combinations, start=1):
        experiment_config = {**config, **parameters}

        for recording_name, recording_parameters in recordings.items():
            run_index += 1
            print(
                f"\n[{run_index}/{total_runs}] combination "
                f"{combination_index}, recording: {recording_name}"
            )

            try:
                # Internal map and diagnostics are removed after this run.
                with tempfile.TemporaryDirectory() as temporary_dir:
                    all_metrics = run_recording(
                        recording_name,
                        recording_parameters,
                        experiment_config,
                        data_dir,
                        Path(temporary_dir),
                    )
                results = {key: all_metrics[key] for key in RESULT_KEYS}
                status = "success"
                error_message = ""
            except Exception as error:
                traceback.print_exc()
                results = {key: "" for key in RESULT_KEYS}
                status = "failed"
                error_message = f"{type(error).__name__}: {error}"

            row = {
                "combination_index": combination_index,
                "recording": recording_name,
                **{key: csv_value(value) for key, value in parameters.items()},
                **results,
                "status": status,
                "error": error_message,
            }
            rows.append(row)

    output_path = output_dir / "all_combinations_summary.csv"
    save_csv(rows, output_path)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
