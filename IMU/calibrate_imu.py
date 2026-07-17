"""Create an IMU calibration file from six stationary orientations."""

import csv
import json
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

GRAVITY = 9.80665
TRIM_SECONDS = 1.0

SCRIPT_DIR = Path(__file__).resolve().parent
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

TARGET_ACCELERATION = {
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
    magnetic_field = []
    sample_rates = []

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            timestamps.append(float(row["timestamp"]))
            acceleration.append(
                [float(row[axis]) for axis in (
                    "imu2_ax_mg",
                    "imu2_ay_mg",
                    "imu2_az_mg",
                )]
            )
            angular_rate.append(
                [float(row[axis]) for axis in (
                    "imu2_gx_mdps",
                    "imu2_gy_mdps",
                    "imu2_gz_mdps",
                )]
            )
            magnetic_field.append(
                [float(row[axis]) for axis in (
                    "mag_x_uT",
                    "mag_y_uT",
                    "mag_z_uT",
                )]
            )
            sample_rates.append(float(row["output_hz"]))

    timestamps = np.array(timestamps)
    acceleration = np.array(acceleration) * GRAVITY / 1000.0
    angular_rate = np.radians(np.array(angular_rate) / 1000.0)
    magnetic_field = np.array(magnetic_field)
    sample_rates = np.array(sample_rates)

    stationary = timestamps >= timestamps[0] + TRIM_SECONDS
    stationary &= timestamps <= timestamps[-1] - TRIM_SECONDS
    return (
        acceleration[stationary],
        angular_rate[stationary],
        magnetic_field[stationary],
        sample_rates[stationary],
    )


def accelerometer_calibration(measured_means, target_means):
    design_matrix = np.column_stack(
        (measured_means, np.ones(len(measured_means)))
    )
    affine_parameters = np.linalg.lstsq(
        design_matrix,
        target_means,
        rcond=None,
    )[0]

    accel_matrix = affine_parameters[:3].T
    affine_offset = affine_parameters[3]
    accel_bias = -np.linalg.solve(accel_matrix, affine_offset)
    return accel_matrix, accel_bias


def magnetometer_calibration(magnetic_field):
    minimum = np.min(magnetic_field, axis=0)
    maximum = np.max(magnetic_field, axis=0)
    mag_bias = 0.5 * (maximum + minimum)
    half_ranges = 0.5 * (maximum - minimum)
    mean_half_range = np.mean(half_ranges)
    mag_matrix = np.diag(mean_half_range / half_ranges)
    return mag_matrix, mag_bias


def main():
    recordings = {}
    for orientation, filename in RECORDINGS.items():
        recordings[orientation] = read_recording(
            CALIBRATION_DIR / filename
        )

    orientations = list(RECORDINGS)
    measured_means = np.array(
        [np.mean(recordings[name][0], axis=0) for name in orientations]
    )
    target_means = np.array(
        [TARGET_ACCELERATION[name] for name in orientations]
    )

    accel_matrix, accel_bias = accelerometer_calibration(
        measured_means,
        target_means,
    )
    corrected_means = (
        accel_matrix @ (measured_means - accel_bias).T
    ).T

    acceleration_residuals = []
    angular_rate_residuals = []
    magnetic_field_residuals = []
    all_angular_rates = []
    all_magnetic_fields = []
    all_sample_rates = []
    for (
        acceleration,
        angular_rate,
        magnetic_field,
        sample_rate,
    ) in recordings.values():
        corrected_acceleration = (
            accel_matrix @ (acceleration - accel_bias).T
        ).T
        acceleration_residuals.append(
            corrected_acceleration - np.mean(corrected_acceleration, axis=0)
        )
        angular_rate_residuals.append(
            angular_rate - np.mean(angular_rate, axis=0)
        )
        magnetic_field_residuals.append(
            magnetic_field - np.mean(magnetic_field, axis=0)
        )
        all_angular_rates.append(angular_rate)
        all_magnetic_fields.append(magnetic_field)
        all_sample_rates.append(sample_rate)

    acceleration_residuals = np.vstack(acceleration_residuals)
    angular_rate_residuals = np.vstack(angular_rate_residuals)
    magnetic_field_residuals = np.vstack(magnetic_field_residuals)
    all_angular_rates = np.vstack(all_angular_rates)
    all_magnetic_fields = np.vstack(all_magnetic_fields)
    all_sample_rates = np.concatenate(all_sample_rates)

    gyro_bias = np.mean(all_angular_rates, axis=0)
    mag_matrix, mag_bias = magnetometer_calibration(all_magnetic_fields)
    corrected_magnetic_field = (
        mag_matrix @ (all_magnetic_fields - mag_bias).T
    ).T
    accel_noise_std = np.std(acceleration_residuals, axis=0, ddof=1)
    gyro_noise_std = np.std(angular_rate_residuals, axis=0, ddof=1)
    mag_noise_std = np.std(magnetic_field_residuals, axis=0, ddof=1)
    calibration_error = corrected_means - target_means

    result = {
        "imu_sensor": "imu2",
        "accel_columns": ["imu2_ax_mg", "imu2_ay_mg", "imu2_az_mg"],
        "gyro_columns": [
            "imu2_gx_mdps",
            "imu2_gy_mdps",
            "imu2_gz_mdps",
        ],
        "mag_columns": ["mag_x_uT", "mag_y_uT", "mag_z_uT"],
        "gravity_m_s2": GRAVITY,
        "accel_matrix": accel_matrix.tolist(),
        "accel_bias_m_s2": accel_bias.tolist(),
        "gyro_bias_rad_s": gyro_bias.tolist(),
        "gyro_bias_deg_s": np.degrees(gyro_bias).tolist(),
        "accel_noise_std_m_s2": accel_noise_std.tolist(),
        "gyro_noise_std_rad_s": gyro_noise_std.tolist(),
        "gyro_noise_std_deg_s": np.degrees(gyro_noise_std).tolist(),
        "mag_matrix": mag_matrix.tolist(),
        "mag_bias_uT": mag_bias.tolist(),
        "mag_noise_std_uT": mag_noise_std.tolist(),
        "mag_field_strength_uT": float(
            np.mean(np.linalg.norm(corrected_magnetic_field, axis=1))
        ),
        "mag_calibration": "approximate_six_position_min_max",
        "sample_rate_hz": float(np.mean(all_sample_rates)),
        "calibration_error_rms_m_s2": float(
            np.sqrt(np.mean(calibration_error**2))
        ),
        "recordings": RECORDINGS,
        "orientation_means": {
            orientation: {
                "raw_m_s2": measured_means[index].tolist(),
                "corrected_m_s2": corrected_means[index].tolist(),
                "target_m_s2": target_means[index].tolist(),
            }
            for index, orientation in enumerate(orientations)
        },
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Accelerometer bias [m/s^2]: {accel_bias}")
    print(f"Gyroscope bias [deg/s]: {np.degrees(gyro_bias)}")
    print(f"Magnetometer bias [uT]: {mag_bias}")
    print(f"Accelerometer noise [m/s^2]: {accel_noise_std}")
    print(f"Gyroscope noise [deg/s]: {np.degrees(gyro_noise_std)}")
    print(f"Magnetometer noise [uT]: {mag_noise_std}")
    print(
        "Calibration RMS error [m/s^2]: "
        f"{result['calibration_error_rms_m_s2']:.6f}"
    )


if __name__ == "__main__":
    main()
