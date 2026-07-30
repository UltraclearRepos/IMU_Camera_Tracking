"""Calibrate IMU2 accelerometer and gyroscope from six static poses."""

import csv
import json
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

GRAVITY = 9.80665
TRIM_SECONDS = 0.05
MAX_ACCEL_STD_M_S2 = 0.05
MAX_GYRO_STD_RAD_S = np.radians(0.5)

SCRIPT_DIR = Path(__file__).absolute().parent
CALIBRATION_DIR = SCRIPT_DIR / "calibration"
OUTPUT_PATH = CALIBRATION_DIR / "imu_calibration.json"

RECORDINGS = {
    "+X": "g_+X_2026-07-16_16.16.31.csv",
    "-X": "g_-X_2026-07-16_16.16.53.csv",
    "+Y": "g_+Y_2026-07-16_16.15.08.csv",
    "-Y": "g_-Y_2026-07-16_16.15.28.csv",
    "+Z": "g_+Z_2026-07-16_16.13.15.csv",
    "-Z": "g_-Z_2026-07-16_16.12.26.csv",
}

EXPECTED_ACCELERATION = {
    "+X": np.array([GRAVITY, 0.0, 0.0]),
    "-X": np.array([-GRAVITY, 0.0, 0.0]),
    "+Y": np.array([0.0, GRAVITY, 0.0]),
    "-Y": np.array([0.0, -GRAVITY, 0.0]),
    "+Z": np.array([0.0, 0.0, GRAVITY]),
    "-Z": np.array([0.0, 0.0, -GRAVITY]),
}


def read_recording(path):
    timestamps = []
    acceleration = []
    angular_rate = []
    sample_rates = []

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
            angular_rate.append(
                [
                    float(row["imu2_gx_mdps"]),
                    float(row["imu2_gy_mdps"]),
                    float(row["imu2_gz_mdps"]),
                ]
            )
            sample_rates.append(float(row["output_hz"]))

    timestamps = np.array(timestamps)
    acceleration = np.array(acceleration) * GRAVITY / 1000.0
    angular_rate = np.radians(np.array(angular_rate) / 1000.0)
    sample_rates = np.array(sample_rates)

    stationary = timestamps >= timestamps[0] + TRIM_SECONDS
    stationary &= timestamps <= timestamps[-1] - TRIM_SECONDS

    return {
        "acceleration": acceleration[stationary],
        "angular_rate": angular_rate[stationary],
        "sample_rate_hz": sample_rates[stationary],
    }


def load_recordings():
    recordings = {}
    for orientation, filename in RECORDINGS.items():
        recordings[orientation] = read_recording(
            CALIBRATION_DIR / filename
        )
    return recordings

def calculate_accelerometer_calibration(recordings):
    orientations = list(recordings)
    measured_means = np.array(
        [
            np.mean(recordings[name]["acceleration"], axis=0)
            for name in orientations
        ]
    )
    expected_means = np.array(
        [EXPECTED_ACCELERATION[name] for name in orientations]
    )

    # Find expected = matrix @ measured + offset.
    design_matrix = np.column_stack(
        [measured_means, np.ones(len(measured_means))]
    )
    parameters = np.linalg.lstsq(
        design_matrix,
        expected_means,
        rcond=None,
    )[0]
    matrix = parameters[:3].T
    offset = parameters[3]
    bias = -np.linalg.solve(matrix, offset)

    noise_residuals = []
    for recording in recordings.values():
        corrected = apply_accelerometer_calibration(
            recording["acceleration"],
            matrix,
            bias,
        )
        noise_residuals.append(corrected - np.mean(corrected, axis=0))

    noise = np.std(np.vstack(noise_residuals), axis=0, ddof=1)
    corrected_means = apply_accelerometer_calibration(
        measured_means,
        matrix,
        bias,
    )
    errors = corrected_means - expected_means
    calibration_rms = float(np.sqrt(np.mean(errors**2)))
    return matrix, bias, noise, calibration_rms


def apply_accelerometer_calibration(acceleration, matrix, bias):
    return (matrix @ (acceleration - bias).T).T


def calculate_gyroscope_calibration(recordings):
    recording_means = np.array(
        [
            np.mean(recording["angular_rate"], axis=0)
            for recording in recordings.values()
        ]
    )
    bias = np.mean(recording_means, axis=0)

    noise_residuals = []
    for recording in recordings.values():
        angular_rate = recording["angular_rate"]
        noise_residuals.append(
            angular_rate - np.mean(angular_rate, axis=0)
        )
    noise = np.std(np.vstack(noise_residuals), axis=0, ddof=1)
    return bias, noise


def save_calibration(calibration):
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(calibration, file, indent=2)
        file.write("\n")


def print_axis_values(
    name,
    values,
    unit,
    source_values,
    source_unit,
):
    print(name)
    print(
        f"  X: {values[0]:.9f} {unit}"
        f" | {source_values[0]:.6f} {source_unit}"
    )
    print(
        f"  Y: {values[1]:.9f} {unit}"
        f" | {source_values[1]:.6f} {source_unit}"
    )
    print(
        f"  Z: {values[2]:.9f} {unit}"
        f" | {source_values[2]:.6f} {source_unit}"
    )


def main():
    recordings = load_recordings()

    (
        accelerometer_matrix,
        accelerometer_bias,
        accelerometer_noise,
        accelerometer_rms,
    ) = calculate_accelerometer_calibration(recordings)
    gyroscope_bias, gyroscope_noise = (
        calculate_gyroscope_calibration(recordings)
    )
    sample_rate_hz = np.mean(
        np.concatenate(
            [
                recording["sample_rate_hz"]
                for recording in recordings.values()
            ]
        )
    )

    calibration = {
        "calibration_version": 1,
        "imu_sensor": "imu2",
        "gravity_m_s2": GRAVITY,
        "sample_rate_hz": float(sample_rate_hz),
        "accel_model": "matrix_times_raw_minus_bias",
        "accel_matrix": accelerometer_matrix.tolist(),
        "accel_bias_m_s2": accelerometer_bias.tolist(),
        "gyro_bias_rad_s": gyroscope_bias.tolist(),
        "accel_noise_std_m_s2": accelerometer_noise.tolist(),
        "gyro_noise_std_rad_s": gyroscope_noise.tolist(),
        "accel_calibration_rms_m_s2": accelerometer_rms,
        "mag_calibration": "not_calibrated",
        "recordings": RECORDINGS,
    }

    save_calibration(calibration)
    print(f"Saved: {OUTPUT_PATH}")
    print("Accelerometer correction matrix")
    print(accelerometer_matrix)
    print_axis_values(
        "Accelerometer bias",
        accelerometer_bias,
        "m/s^2",
        accelerometer_bias * 1000.0 / GRAVITY,
        "mg",
    )
    print_axis_values(
        "Accelerometer sample noise",
        accelerometer_noise,
        "m/s^2",
        accelerometer_noise * 1000.0 / GRAVITY,
        "mg",
    )
    print_axis_values(
        "Gyroscope bias",
        np.degrees(gyroscope_bias),
        "deg/s",
        np.degrees(gyroscope_bias) * 1000.0,
        "mdps",
    )
    print_axis_values(
        "Gyroscope sample noise",
        np.degrees(gyroscope_noise),
        "deg/s",
        np.degrees(gyroscope_noise) * 1000.0,
        "mdps",
    )
    print(
        "Accelerometer calibration RMS: "
        f"{accelerometer_rms:.9f} m/s^2"
    )
    print(
        "Magnetometer: not calibrated; a separate full-rotation "
        "recording is required."
    )


if __name__ == "__main__":
    main()
