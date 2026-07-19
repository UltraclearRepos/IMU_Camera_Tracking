"""Estimate IMU delay relative to Dobot from recorded motion."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RECORDING_NAME = "tracking_aruco_Speed-3_2026-07-16_18.21.12"

MAX_IMU_DELAY_SECONDS = 1.0
RESAMPLE_HZ = 100.0
ACCEL_BASELINE_SECONDS = 0.40
DOBOT_SMOOTHING_SECONDS = 0.10

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data2"
IMU_PATH = DATA_DIR / "imu" / f"{RECORDING_NAME}.csv"
DOBOT_PATH = DATA_DIR / "dobot" / f"{RECORDING_NAME}.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / f"{RECORDING_NAME}_delay.png"

GRAVITY = 9.80665


def read_imu(path):
    timestamps = []
    acceleration = []

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            timestamps.append(float(row["timestamp"]))
            acceleration.append(
                [
                    float(row["imu2_ax_mg"]),
                    float(row["imu2_ay_mg"]),
                    float(row["imu2_az_mg"]),
                ]
            )

    timestamps = np.array(timestamps)
    acceleration = np.array(acceleration) * GRAVITY / 1000.0
    return timestamps, acceleration


def read_dobot(path):
    timestamps = []
    positions = []

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            timestamps.append(float(row["timestamp"]))
            positions.append(
                [float(row[axis]) for axis in ("x", "y", "z")]
            )

    timestamps = np.array(timestamps)
    positions = np.array(positions) / 1000.0
    return timestamps, positions


def interpolate_columns(target_time, source_time, values):
    return np.column_stack(
        [
            np.interp(target_time, source_time, values[:, axis])
            for axis in range(values.shape[1])
        ]
    )


def motion_signals(imu_time, acceleration, dobot_time, position):
    dt = 1.0 / RESAMPLE_HZ
    start_time = max(imu_time[0], dobot_time[0])
    end_time = min(imu_time[-1], dobot_time[-1])
    time = np.arange(start_time, end_time, dt)

    acceleration = interpolate_columns(time, imu_time, acceleration)
    position = interpolate_columns(time, dobot_time, position)

    acceleration_baseline = gaussian_filter1d(
        acceleration,
        ACCEL_BASELINE_SECONDS * RESAMPLE_HZ,
        axis=0,
    )
    imu_motion = np.linalg.norm(
        acceleration - acceleration_baseline,
        axis=1,
    )

    position = gaussian_filter1d(
        position,
        DOBOT_SMOOTHING_SECONDS * RESAMPLE_HZ,
        axis=0,
    )
    velocity = np.gradient(position, dt, axis=0)
    dobot_acceleration = np.gradient(velocity, dt, axis=0)
    dobot_motion = np.linalg.norm(dobot_acceleration, axis=1)

    imu_motion = (imu_motion - np.mean(imu_motion)) / np.std(imu_motion)
    dobot_motion = (dobot_motion - np.mean(dobot_motion)) / np.std(dobot_motion)
    return time, imu_motion, dobot_motion


def correlation_for_lag(imu_motion, dobot_motion, lag_samples):
    if lag_samples > 0:
        imu_part = imu_motion[lag_samples:]
        dobot_part = dobot_motion[:-lag_samples]
    elif lag_samples < 0:
        imu_part = imu_motion[:lag_samples]
        dobot_part = dobot_motion[-lag_samples:]
    else:
        imu_part = imu_motion
        dobot_part = dobot_motion

    imu_part = imu_part - np.mean(imu_part)
    dobot_part = dobot_part - np.mean(dobot_part)
    return np.dot(imu_part, dobot_part) / (
        np.linalg.norm(imu_part) * np.linalg.norm(dobot_part)
    )


def estimate_delay(imu_motion, dobot_motion):
    maximum_lag = round(MAX_IMU_DELAY_SECONDS * RESAMPLE_HZ)
    lag_samples = np.arange(-maximum_lag, maximum_lag + 1)
    correlations = np.array(
        [
            correlation_for_lag(imu_motion, dobot_motion, lag)
            for lag in lag_samples
        ]
    )
    best_index = np.argmax(correlations)
    delay_seconds = lag_samples[best_index] / RESAMPLE_HZ
    return delay_seconds, lag_samples / RESAMPLE_HZ, correlations


def aligned_signals(time, imu_motion, dobot_motion, delay_seconds):
    lag_samples = round(delay_seconds * RESAMPLE_HZ)
    if lag_samples > 0:
        return (
            time[:-lag_samples],
            imu_motion[lag_samples:],
            dobot_motion[:-lag_samples],
        )
    if lag_samples < 0:
        return (
            time[-lag_samples:],
            imu_motion[:lag_samples],
            dobot_motion[-lag_samples:],
        )
    return time, imu_motion, dobot_motion


def save_plot(
    time,
    imu_motion,
    dobot_motion,
    delay_seconds,
    tested_delays,
    correlations,
):
    aligned_time, aligned_imu, aligned_dobot = aligned_signals(
        time,
        imu_motion,
        dobot_motion,
        delay_seconds,
    )
    aligned_time = aligned_time - aligned_time[0]

    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(tested_delays * 1000.0, correlations)
    axes[0].axvline(delay_seconds * 1000.0, color="red", linestyle="--")
    axes[0].set_xlabel("Assumed IMU delay relative to Dobot [ms]")
    axes[0].set_ylabel("Correlation")
    axes[0].grid(True)

    axes[1].plot(aligned_time, aligned_dobot, label="Dobot motion")
    axes[1].plot(aligned_time, aligned_imu, label="Corrected IMU motion")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Normalized motion")
    axes[1].grid(True)
    axes[1].legend()

    figure.suptitle(
        f"IMU delay relative to Dobot: {delay_seconds * 1000.0:+.0f} ms"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160)
    plt.close(figure)


def main():
    imu_time, acceleration = read_imu(IMU_PATH)
    dobot_time, position = read_dobot(DOBOT_PATH)
    time, imu_motion, dobot_motion = motion_signals(
        imu_time,
        acceleration,
        dobot_time,
        position,
    )
    imu_delay, tested_delays, correlations = estimate_delay(
        imu_motion,
        dobot_motion,
    )
    save_plot(
        time,
        imu_motion,
        dobot_motion,
        imu_delay,
        tested_delays,
        correlations,
    )

    print(f"IMU relative to Dobot: {imu_delay * 1000.0:+.0f} ms")
    print("Camera relative to Dobot: +0 ms (stored timestamps are synchronized)")
    print(f"IMU relative to camera: {imu_delay * 1000.0:+.0f} ms")
    print("Positive means that the signal appears later.")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
