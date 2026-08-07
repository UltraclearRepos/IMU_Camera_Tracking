"""Permanently synchronize IMU and video timestamps to Dobot."""

import csv
import json
from pathlib import Path

import numpy as np

from estimate_delay import estimate_delay, motion_signals, read_dobot, read_imu


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DRY_RUN = False
CAMERA_DELAY_TO_DOBOT_SECONDS = 0.084

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = "firstSkinData"
DATA_DIR = PROJECT_DIR / "Data" / DATA_FOLDER
IMU_DIR = DATA_DIR / "imu"
DOBOT_DIR = DATA_DIR / "dobot"
VIDEO_TIMESTAMP_DIR = DATA_DIR / "video_timestamps"
MANIFEST_PATH = DATA_DIR / "timestamp_synchronization.json"


def load_manifest():
    if not MANIFEST_PATH.exists():
        return {"recordings": {}}
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def recording_names():
    names = []
    for imu_path in sorted(IMU_DIR.glob("*.csv")):
        name = imu_path.stem
        if (DOBOT_DIR / f"{name}.csv").exists():
            names.append(name)
    return names


def estimate_imu_delays(names):
    results = {}
    for name in names:
        imu_time, acceleration = read_imu(IMU_DIR / f"{name}.csv")
        dobot_time, position = read_dobot(DOBOT_DIR / f"{name}.csv")
        _, imu_motion, dobot_motion = motion_signals(
            imu_time,
            acceleration,
            dobot_time,
            position,
        )
        delay, tested_delays, correlations = estimate_delay(
            imu_motion,
            dobot_motion,
        )
        best_correlation = float(np.max(correlations))
        at_search_boundary = bool(
            np.isclose(abs(delay), np.max(abs(tested_delays)))
        )
        results[name] = {
            "imu_delay_to_dobot_seconds": float(delay),
            "correlation": best_correlation,
            "at_search_boundary": at_search_boundary,
        }
        print(
            f"{name}: IMU {delay * 1000.0:+.0f} ms | "
            f"correlation {best_correlation:.3f} | "
            f"boundary {at_search_boundary}"
        )
    return results


def shift_imu_timestamps(path, delay_seconds):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames

    for row in rows:
        row["timestamp"] = f"{float(row['timestamp']) - delay_seconds:.9f}"

    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def shift_video_start_timestamp(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames

    for row in rows:
        row["start_timestamp"] = (
            f"{float(row['start_timestamp']) - CAMERA_DELAY_TO_DOBOT_SECONDS:.9f}"
        )

    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def main():
    manifest = load_manifest()
    synchronized = manifest["recordings"]
    names = [name for name in recording_names() if name not in synchronized]

    if not names:
        print(
            f"All matching {DATA_FOLDER} recordings are already synchronized."
        )
        return

    results = estimate_imu_delays(names)

    if DRY_RUN:
        print("Dry run only. No timestamps were changed.")
        return

    boundary_results = [
        name for name, result in results.items()
        if result["at_search_boundary"]
    ]
    if boundary_results:
        raise RuntimeError(
            "Delay reached search boundary for: "
            + ", ".join(boundary_results)
        )

    for name, result in results.items():
        shift_imu_timestamps(
            IMU_DIR / f"{name}.csv",
            result["imu_delay_to_dobot_seconds"],
        )

        video_timestamp_path = VIDEO_TIMESTAMP_DIR / f"{name}.csv"
        if video_timestamp_path.exists():
            shift_video_start_timestamp(video_timestamp_path)

        synchronized[name] = {
            **result,
            "camera_delay_to_dobot_seconds": CAMERA_DELAY_TO_DOBOT_SECONDS,
        }

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Synchronized {len(results)} recordings.")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
