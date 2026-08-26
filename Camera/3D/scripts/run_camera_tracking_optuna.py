import argparse
import json
import math
import statistics
import sys
import tempfile
import traceback
from pathlib import Path

import optuna


SCRIPT_DIR = Path(__file__).absolute().parent
MODULE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MODULE_DIR.parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


OBJECTIVE_NAMES = (
    "mean_position_rmse_mm",
    "std_position_rmse_mm",
    "mean_orientation_rmse_deg",
    "std_orientation_rmse_deg",
)
MIN_TRACKED_PERCENT = 90.0
PENALTY = 1_000_000.0


def load_config(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def suggest_parameters(trial, grid):
    limits_index = trial.suggest_categorical(
        "feature_limits_index",
        list(range(len(grid["feature_limits"]))),
    )
    pairs_index = trial.suggest_categorical(
        "mapping_pair_settings_index",
        list(range(len(grid["mapping_pair_settings"]))),
    )
    pairs = grid["mapping_pair_settings"][pairs_index]

    return {
        "feature_type": trial.suggest_categorical(
            "feature_type", grid["feature_type"]
        ),
        "use_imu": trial.suggest_categorical("use_imu", grid["use_imu"]),
        **grid["feature_limits"][limits_index],
        "keyframe_interval": trial.suggest_categorical(
            "keyframe_interval", grid["keyframe_interval"]
        ),
        **pairs,
    }


def parameter_signature(parameters):
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def selected_recording_metrics(metrics):
    keys = (
        "recording",
        "tracked_percent",
        "position_rmse_mm",
        "orientation_rmse_deg",
        "map_build_wall_time_s",
        "mapping_online_wall_time_s",
        "mapping_offline_wall_time_s",
        "tracking_wall_time_s",
    )
    return {key: metrics[key] for key in keys}


def aggregate_metrics(recording_metrics):
    position = [item["position_rmse_mm"] for item in recording_metrics]
    orientation = [item["orientation_rmse_deg"] for item in recording_metrics]
    values = (
        statistics.fmean(position),
        statistics.pstdev(position),
        statistics.fmean(orientation),
        statistics.pstdev(orientation),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite RMSE objective")

    return values, {
        "min_tracked_percent": min(
            item["tracked_percent"] for item in recording_metrics
        ),
        "mean_map_build_wall_time_s": statistics.fmean(
            item["map_build_wall_time_s"] for item in recording_metrics
        ),
        "mean_tracking_wall_time_s": statistics.fmean(
            item["tracking_wall_time_s"] for item in recording_metrics
        ),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Optimize camera-tracking parameters with Optuna."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Target total number of trials, including resumed trials.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    config = load_config(arguments.config)
    grid = config["parameter_grid"]
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    import camera_tracking
    from run_camera_tracking_batch import run_recording

    camera_tracking.SAVE_DIAGNOSTIC_VIDEO = False
    camera_tracking.SAVE_MAPPING_FEATURE_VIDEO = False
    camera_tracking.SAVE_MAP_BUILD_TOP_VIEW = False
    camera_tracking.SAVE_TRACKING_TOP_VIEW = False
    camera_tracking.SHOW_PREVIEW = False

    data_dir = PROJECT_DIR / "Data" / config["data_folder"]

    def constraints(trial):
        return (
            trial.user_attrs.get(
                "tracking_constraint", MIN_TRACKED_PERCENT
            ),
        )

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        constraints_func=constraints,
        seed=arguments.seed,
    )
    database_path = (output_dir / "optuna_study.db").resolve()
    study = optuna.create_study(
        study_name=config["experiment_name"],
        directions=["minimize"] * len(OBJECTIVE_NAMES),
        sampler=sampler,
        storage=f"sqlite:///{database_path.as_posix()}",
        load_if_exists=True,
    )
    study.set_metric_names(list(OBJECTIVE_NAMES))

    cache = {}
    for old_trial in study.trials:
        signature = old_trial.user_attrs.get("parameter_signature")
        if old_trial.values is not None and signature:
            cache[signature] = {
                "number": old_trial.number,
                "values": old_trial.values,
                "user_attrs": old_trial.user_attrs,
            }

    def objective(trial):
        parameters = suggest_parameters(trial, grid)
        signature = parameter_signature(parameters)
        trial.set_user_attr("resolved_parameters", parameters)
        trial.set_user_attr("parameter_signature", signature)

        if signature in cache:
            cached = cache[signature]
            for key in (
                "recording_metrics",
                "min_tracked_percent",
                "mean_map_build_wall_time_s",
                "mean_tracking_wall_time_s",
                "tracking_constraint",
                "error",
            ):
                if key in cached["user_attrs"]:
                    trial.set_user_attr(key, cached["user_attrs"][key])
            trial.set_user_attr("cached_from_trial", cached["number"])
            return cached["values"]

        experiment_config = {**config, **parameters}
        recording_metrics = []
        try:
            for recording_name, recording_parameters in config[
                "recordings"
            ].items():
                print(f"Trial {trial.number}, recording: {recording_name}")
                with tempfile.TemporaryDirectory() as temporary_dir:
                    metrics = run_recording(
                        recording_name,
                        recording_parameters,
                        experiment_config,
                        data_dir,
                        Path(temporary_dir),
                    )
                recording_metrics.append(selected_recording_metrics(metrics))

            values, summary = aggregate_metrics(recording_metrics)
            constraint = (
                MIN_TRACKED_PERCENT - summary["min_tracked_percent"]
            )
            trial.set_user_attr("recording_metrics", recording_metrics)
            trial.set_user_attr("tracking_constraint", constraint)
            for key, value in summary.items():
                trial.set_user_attr(key, value)
            cache[signature] = {
                "number": trial.number,
                "values": values,
                "user_attrs": {
                    "recording_metrics": recording_metrics,
                    "tracking_constraint": constraint,
                    **summary,
                },
            }
            return values
        except Exception as error:
            traceback.print_exc()
            trial.set_user_attr("error", f"{type(error).__name__}: {error}")
            trial.set_user_attr(
                "tracking_constraint", MIN_TRACKED_PERCENT
            )
            values = (PENALTY,) * len(OBJECTIVE_NAMES)
            cache[signature] = {
                "number": trial.number,
                "values": values,
                "user_attrs": {
                    "tracking_constraint": MIN_TRACKED_PERCENT,
                    "error": f"{type(error).__name__}: {error}",
                },
            }
            return values

    target_trials = max(arguments.trials, 0)
    remaining_trials = max(target_trials - len(study.trials), 0)
    print(
        f"Study trials: {len(study.trials)}/{target_trials}; "
        f"running {remaining_trials} new trials"
    )

    study.optimize(objective, n_trials=remaining_trials)

    print(f"Saved Optuna study: {database_path}")


if __name__ == "__main__":
    main()
