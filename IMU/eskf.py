"""IMU-only error-state Kalman filter and comparison with ground truth."""

import csv
import json
import math
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RECORDING_NAME = "skin_low_res_Speed-3_2026-07-21_13.15.30"
INCLUDE_CAMERA_RESULTS = True
USE_CAMERA_POSE_CORRECTION = True
CAMERA_POSITION_STD_M = 0.002
CAMERA_ORIENTATION_STD_DEG = 1.0
IMU_ACCEL_NOISE_MULTIPLIER = 10.0
IMU_GYRO_NOISE_MULTIPLIER = 10.0

SCRIPT_DIR = Path(__file__).absolute().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "Data2"
IMU_LOG_PATH = DATA_DIR / "imu" / f"{RECORDING_NAME}.csv"
GROUND_TRUTH_PATH = DATA_DIR / "dobot" / f"{RECORDING_NAME}.csv"
CAMERA_RESULTS_PATH = (
    PROJECT_DIR
    / "Camera"
    / "results"
    / RECORDING_NAME
    / f"{RECORDING_NAME}_camera.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_CSV_PATH = OUTPUT_DIR / f"{RECORDING_NAME}_eskf.csv"
OUTPUT_POSITION_PLOT_PATH = (
    OUTPUT_DIR / f"{RECORDING_NAME}_eskf_vs_gt_position.png"
)
OUTPUT_ORIENTATION_PLOT_PATH = (
    OUTPUT_DIR / f"{RECORDING_NAME}_eskf_vs_gt_orientation.png"
)
CALIBRATION_PATH = (
    SCRIPT_DIR
    / "calibration"
    / "imu_calibration.json"
)

INITIAL_CALIBRATION_SECONDS = 0.3
AUTO_ZUPT = False

GRAVITY = 9.80665


def skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quaternion_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def quaternion_conjugate(quaternion):
    return quaternion * np.array([1.0, -1.0, -1.0, -1.0])


def quaternion_from_rotation_vector(rotation_vector):
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-12:
        return np.array(
            [
                1.0,
                0.5 * rotation_vector[0],
                0.5 * rotation_vector[1],
                0.5 * rotation_vector[2],
            ]
        )

    half_angle = 0.5 * angle
    return np.concatenate(
        ([math.cos(half_angle)], math.sin(half_angle) * rotation_vector / angle)
    )


def quaternion_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quaternion_to_rotation_vector(quaternion):
    if quaternion[0] < 0.0:
        quaternion = -quaternion

    vector_norm = np.linalg.norm(quaternion[1:])
    if vector_norm < 1e-12:
        return 2.0 * quaternion[1:]

    angle = 2.0 * math.atan2(vector_norm, quaternion[0])
    return angle * quaternion[1:] / vector_norm


def rotation_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    )


def euler_degrees(quaternion):
    w, x, y, z = quaternion
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return np.degrees([roll, pitch, yaw])


def initial_orientation(mean_acceleration):
    ax, ay, az = mean_acceleration / np.linalg.norm(mean_acceleration)
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    return quaternion_from_euler(roll, pitch, 0.0)


class EskfParameters:
    def __init__(self, calibration):
        self.accel_noise_std = np.array(
            calibration["accel_noise_std_m_s2"]
        ) * IMU_ACCEL_NOISE_MULTIPLIER
        self.gyro_noise_std = np.array(
            calibration["gyro_noise_std_rad_s"]
        ) * IMU_GYRO_NOISE_MULTIPLIER
        self.accel_bias_random_walk = 5e-4
        self.gyro_bias_random_walk = math.radians(0.01)
        self.zero_velocity_std = 0.01
        self.stationary_accel_std = 0.03
        self.stationary_gyro_std = math.radians(0.2)


class ErrorStateKalmanFilter:
    def __init__(self, calibration):
        self.parameters = EskfParameters(calibration)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.covariance = np.eye(15)
        self.gravity_world = np.array([0.0, 0.0, -GRAVITY])

    def initialize(self, acceleration, angular_rate):
        mean_acceleration = np.mean(acceleration, axis=0)
        self.quaternion = initial_orientation(mean_acceleration)
        self.gyro_bias = np.mean(angular_rate, axis=0)

        gravity_in_body = (
            rotation_matrix(self.quaternion).T @ -self.gravity_world
        )
        self.accel_bias = mean_acceleration - gravity_in_body

        initial_std = np.concatenate(
            [
                np.full(3, 1e-3),
                np.full(3, 1e-2),
                np.radians([1.0, 1.0, 10.0]),
                np.full(3, 0.05),
                np.full(3, math.radians(0.5)),
            ]
        )
        self.covariance = np.diag(initial_std**2)

    def predict(self, acceleration, angular_rate, dt):
        corrected_acceleration = acceleration - self.accel_bias
        corrected_angular_rate = angular_rate - self.gyro_bias
        body_to_world = rotation_matrix(self.quaternion)
        acceleration_world = (
            body_to_world @ corrected_acceleration + self.gravity_world
        )

        self.position += self.velocity * dt + 0.5 * acceleration_world * dt * dt
        self.velocity += acceleration_world * dt

        delta_quaternion = quaternion_from_rotation_vector(
            corrected_angular_rate * dt
        )
        self.quaternion = quaternion_multiply(
            self.quaternion,
            delta_quaternion,
        )
        self.quaternion /= np.linalg.norm(self.quaternion)

        transition_rate = np.zeros((15, 15))
        transition_rate[0:3, 3:6] = np.eye(3)
        transition_rate[3:6, 6:9] = (
            -body_to_world @ skew(corrected_acceleration)
        )
        transition_rate[3:6, 9:12] = -body_to_world
        transition_rate[6:9, 6:9] = -skew(corrected_angular_rate)
        transition_rate[6:9, 12:15] = -np.eye(3)
        transition = np.eye(15) + transition_rate * dt

        noise_mapping = np.zeros((15, 12))
        noise_mapping[0:3, 0:3] = -0.5 * body_to_world * dt * dt
        noise_mapping[3:6, 0:3] = -body_to_world * dt
        noise_mapping[6:9, 3:6] = -np.eye(3) * dt
        noise_mapping[9:12, 6:9] = np.eye(3) * math.sqrt(dt)
        noise_mapping[12:15, 9:12] = np.eye(3) * math.sqrt(dt)

        parameters = self.parameters
        noise_variance = np.concatenate(
            [
                parameters.accel_noise_std**2,
                parameters.gyro_noise_std**2,
                np.full(3, parameters.accel_bias_random_walk**2),
                np.full(3, parameters.gyro_bias_random_walk**2),
            ]
        )
        process_noise = (
            noise_mapping @ np.diag(noise_variance) @ noise_mapping.T
        )
        self.covariance = (
            transition @ self.covariance @ transition.T + process_noise
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update_stationary(self, acceleration, angular_rate):
        body_to_world = rotation_matrix(self.quaternion)
        gravity_in_body = body_to_world.T @ -self.gravity_world

        residual = np.concatenate(
            [
                -self.velocity,
                angular_rate - self.gyro_bias,
                acceleration - (gravity_in_body + self.accel_bias),
            ]
        )

        measurement_matrix = np.zeros((9, 15))
        measurement_matrix[0:3, 3:6] = np.eye(3)
        measurement_matrix[3:6, 12:15] = np.eye(3)
        measurement_matrix[6:9, 6:9] = skew(gravity_in_body)
        measurement_matrix[6:9, 9:12] = np.eye(3)

        parameters = self.parameters
        measurement_variance = np.concatenate(
            [
                np.full(3, parameters.zero_velocity_std**2),
                np.full(3, parameters.stationary_gyro_std**2),
                np.full(3, parameters.stationary_accel_std**2),
            ]
        )
        self.kalman_update(
            residual,
            measurement_matrix,
            np.diag(measurement_variance),
        )

    def update_camera_pose(
        self,
        measured_position,
        measured_quaternion,
    ):
        orientation_residual = quaternion_to_rotation_vector(
            quaternion_multiply(
                quaternion_conjugate(self.quaternion),
                measured_quaternion,
            )
        )
        residual = np.concatenate(
            [
                measured_position - self.position,
                orientation_residual,
            ]
        )

        measurement_matrix = np.zeros((6, 15))
        measurement_matrix[0:3, 0:3] = np.eye(3)
        measurement_matrix[3:6, 6:9] = np.eye(3)

        measurement_variance = np.concatenate(
            [
                np.full(3, CAMERA_POSITION_STD_M**2),
                np.full(
                    3,
                    math.radians(CAMERA_ORIENTATION_STD_DEG) ** 2,
                ),
            ]
        )
        self.kalman_update(
            residual,
            measurement_matrix,
            np.diag(measurement_variance),
        )

    def kalman_update(self, residual, measurement_matrix, measurement_noise):
        innovation_covariance = (
            measurement_matrix
            @ self.covariance
            @ measurement_matrix.T
            + measurement_noise
        )
        gain = np.linalg.solve(
            innovation_covariance,
            measurement_matrix @ self.covariance,
        ).T
        error_state = gain @ residual

        identity = np.eye(15)
        correction = identity - gain @ measurement_matrix
        self.covariance = (
            correction @ self.covariance @ correction.T
            + gain @ measurement_noise @ gain.T
        )

        self.position += error_state[0:3]
        self.velocity += error_state[3:6]
        orientation_error = error_state[6:9]
        self.quaternion = quaternion_multiply(
            self.quaternion,
            quaternion_from_rotation_vector(orientation_error),
        )
        self.quaternion /= np.linalg.norm(self.quaternion)
        self.accel_bias += error_state[9:12]
        self.gyro_bias += error_state[12:15]

        reset_jacobian = np.eye(15)
        reset_jacobian[6:9, 6:9] -= 0.5 * skew(orientation_error)
        self.covariance = (
            reset_jacobian @ self.covariance @ reset_jacobian.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)


class StationaryDetector:
    def __init__(self, sample_rate_hz):
        self.window_size = round(sample_rate_hz * 0.25)
        self.acceleration = deque(maxlen=self.window_size)
        self.angular_rate = deque(maxlen=self.window_size)

    def update(self, acceleration, angular_rate, gyro_bias):
        self.acceleration.append(acceleration.copy())
        self.angular_rate.append((angular_rate - gyro_bias).copy())

        if len(self.acceleration) < self.window_size:
            return False

        acceleration_window = np.asarray(self.acceleration)
        angular_rate_window = np.asarray(self.angular_rate)
        mean_acceleration = np.mean(acceleration_window, axis=0)

        gravity_error = abs(np.linalg.norm(mean_acceleration) - GRAVITY)
        acceleration_rms = np.sqrt(
            np.mean(
                np.sum(
                    (acceleration_window - mean_acceleration) ** 2,
                    axis=1,
                )
            )
        )
        angular_rate_rms = np.sqrt(
            np.mean(np.sum(angular_rate_window**2, axis=1))
        )

        return (
            gravity_error < 0.12
            and acceleration_rms < 0.04
            and angular_rate_rms < math.radians(1.5)
        )


def load_calibration(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_imu_log(path, calibration):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    timestamps = np.array([float(row["timestamp"]) for row in rows])
    acceleration_mg = np.array(
        [
            [
                float(row["imu2_ax_mg"]),
                float(row["imu2_ay_mg"]),
                float(row["imu2_az_mg"]),
            ]
            for row in rows
        ]
    )
    angular_rate_mdps = np.array(
        [
            [
                float(row["imu2_gx_mdps"]),
                float(row["imu2_gy_mdps"]),
                float(row["imu2_gz_mdps"]),
            ]
            for row in rows
        ]
    )
    sample_rate_hz = np.mean(
        [float(row["output_hz"]) for row in rows]
    )

    acceleration = acceleration_mg * GRAVITY / 1000.0
    angular_rate = np.radians(angular_rate_mdps / 1000.0)

    accel_matrix = np.array(calibration["accel_matrix"])
    accel_bias = np.array(calibration["accel_bias_m_s2"])
    gyro_bias = np.array(calibration["gyro_bias_rad_s"])
    acceleration = (
        accel_matrix @ (acceleration - accel_bias).T
    ).T
    angular_rate = angular_rate - gyro_bias
    return timestamps, acceleration, angular_rate, sample_rate_hz


def read_ground_truth(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    time = np.array([float(row["timestamp"]) for row in rows])

    x = np.array([float(row["x"]) for row in rows])
    y = np.array([float(row["y"]) for row in rows])
    z = np.array([float(row["z"]) for row in rows])
    roll = np.array([float(row["roll"]) for row in rows])
    pitch = np.array([float(row["pitch"]) for row in rows])
    yaw = np.array([float(row["yaw"]) for row in rows])

    values = {
        "X": x - x[0],
        "Y": y - y[0],
        "Z": z - z[0],
        "Roll": roll,
        "Pitch": pitch,
        "Yaw": yaw,
    }
    return time, values


def read_camera_results(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    time = np.array([float(row["timestamp"]) for row in rows])
    values = {
        "X": np.array([float(row["x_mm"]) for row in rows]),
        "Y": np.array([float(row["y_mm"]) for row in rows]),
        "Z": np.array([float(row["z_mm"]) for row in rows]),
        "Roll": np.array([float(row["roll_deg"]) for row in rows]),
        "Pitch": np.array([float(row["pitch_deg"]) for row in rows]),
        "Yaw": np.array([float(row["yaw_deg"]) for row in rows]),
    }
    return time, values


def run_filter():
    calibration = load_calibration(CALIBRATION_PATH)
    timestamps, acceleration, angular_rate, sample_rate_hz = read_imu_log(
        IMU_LOG_PATH,
        calibration,
    )
    calibration_samples = np.searchsorted(
        timestamps,
        timestamps[0] + INITIAL_CALIBRATION_SECONDS,
    )

    eskf = ErrorStateKalmanFilter(calibration)
    eskf.initialize(
        acceleration[:calibration_samples],
        angular_rate[:calibration_samples],
    )
    initial_quaternion = eskf.quaternion.copy()
    detector = StationaryDetector(sample_rate_hz)

    if USE_CAMERA_POSE_CORRECTION:
        camera_time, camera_values = read_camera_results(
            CAMERA_RESULTS_PATH
        )
        camera_position = np.column_stack(
            [camera_values[axis] for axis in ("X", "Y", "Z")]
        ) / 1000.0
        camera_euler = np.column_stack(
            [
                camera_values[axis]
                for axis in ("Roll", "Pitch", "Yaw")
            ]
        )
        valid_camera = np.isfinite(camera_position).all(axis=1)
        valid_camera &= np.isfinite(camera_euler).all(axis=1)
        camera_time = camera_time[valid_camera]
        camera_position = camera_position[valid_camera]
        camera_euler = camera_euler[valid_camera]
        camera_index = np.searchsorted(
            camera_time,
            timestamps[calibration_samples],
        )

    output = []
    for index in range(len(acceleration)):
        stationary_update = False
        camera_update = False

        if index >= calibration_samples:
            dt = timestamps[index] - timestamps[index - 1]
            eskf.predict(
                acceleration[index],
                angular_rate[index],
                dt,
            )
            if not USE_CAMERA_POSE_CORRECTION:
                stationary_update = AUTO_ZUPT and detector.update(
                    acceleration[index],
                    angular_rate[index],
                    eskf.gyro_bias,
                )
                if stationary_update:
                    eskf.update_stationary(
                        acceleration[index],
                        angular_rate[index],
                    )

        if USE_CAMERA_POSE_CORRECTION:
            while (
                camera_index < len(camera_time)
                and camera_time[camera_index] <= timestamps[index]
            ):
                relative_camera_quaternion = quaternion_from_euler(
                    *np.radians(camera_euler[camera_index])
                )
                measured_quaternion = quaternion_multiply(
                    initial_quaternion,
                    relative_camera_quaternion,
                )
                eskf.update_camera_pose(
                    camera_position[camera_index],
                    measured_quaternion,
                )
                camera_index += 1
                camera_update = True

        relative_quaternion = quaternion_multiply(
            quaternion_conjugate(initial_quaternion),
            eskf.quaternion,
        )
        roll, pitch, yaw = euler_degrees(relative_quaternion)
        position_sigma = np.sqrt(np.diag(eskf.covariance)[0:3])

        output.append(
            {
                "timestamp": timestamps[index],
                "time_s": timestamps[index] - timestamps[0],
                "x_mm": 1000.0 * eskf.position[0],
                "y_mm": 1000.0 * eskf.position[1],
                "z_mm": 1000.0 * eskf.position[2],
                "vx_m_s": eskf.velocity[0],
                "vy_m_s": eskf.velocity[1],
                "vz_m_s": eskf.velocity[2],
                "qw": relative_quaternion[0],
                "qx": relative_quaternion[1],
                "qy": relative_quaternion[2],
                "qz": relative_quaternion[3],
                "roll_deg": roll,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
                "bax_m_s2": eskf.accel_bias[0],
                "bay_m_s2": eskf.accel_bias[1],
                "baz_m_s2": eskf.accel_bias[2],
                "bgx_deg_s": math.degrees(eskf.gyro_bias[0]),
                "bgy_deg_s": math.degrees(eskf.gyro_bias[1]),
                "bgz_deg_s": math.degrees(eskf.gyro_bias[2]),
                "sigma_x_mm": 1000.0 * position_sigma[0],
                "sigma_y_mm": 1000.0 * position_sigma[1],
                "sigma_z_mm": 1000.0 * position_sigma[2],
                "stationary_update": int(stationary_update),
                "camera_update": int(camera_update),
            }
        )

    return output


def write_output(rows):
    with OUTPUT_CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_figure(
    time,
    ground_truth,
    estimate,
    component_errors,
    overall_errors,
    component_names,
    unit,
    overall_name,
    output_path,
    title,
    estimate_label,
    camera_time,
    camera_values,
    camera_component_errors,
    camera_overall_errors,
):
    figure = plt.figure(figsize=(14, 14))
    grid = figure.add_gridspec(4, 2)

    for component in range(3):
        comparison_axis = figure.add_subplot(grid[component, 0])
        comparison_axis.plot(
            time,
            ground_truth[:, component],
            "k--",
            label="GT",
        )
        comparison_axis.plot(
            time,
            estimate[:, component],
            color="tab:blue",
            label=estimate_label,
        )
        if INCLUDE_CAMERA_RESULTS:
            comparison_axis.plot(
                camera_time,
                camera_values[:, component],
                color="tab:green",
                label="Camera",
            )
        comparison_axis.set_ylabel(
            f"{component_names[component]} [{unit}]"
        )
        comparison_axis.grid(True)
        comparison_axis.legend()

        component_rmse = np.sqrt(
            np.mean(component_errors[:, component] ** 2)
        )
        error_axis = figure.add_subplot(grid[component, 1])
        error_axis.plot(
            time,
            np.abs(component_errors[:, component]),
            color="tab:red",
            label=estimate_label,
        )
        error_title = (
            f"{component_names[component]} error | "
            f"{estimate_label} RMSE: {component_rmse:.2f} {unit}"
        )
        if INCLUDE_CAMERA_RESULTS:
            camera_component_rmse = np.sqrt(
                np.nanmean(
                    camera_component_errors[:, component] ** 2
                )
            )
            error_axis.plot(
                camera_time,
                np.abs(camera_component_errors[:, component]),
                color="tab:green",
                label="Camera",
            )
            error_title += (
                f" | Camera RMSE: {camera_component_rmse:.2f} {unit}"
            )
        error_axis.set_title(error_title)
        error_axis.set_ylabel(f"Absolute error [{unit}]")
        error_axis.grid(True)
        error_axis.legend()

    overall_mae = np.mean(overall_errors)
    overall_rmse = np.sqrt(np.mean(overall_errors**2))
    overall_axis = figure.add_subplot(grid[3, :])
    overall_axis.plot(
        time,
        overall_errors,
        color="tab:red",
        label=estimate_label,
    )
    overall_title = (
        f"{overall_name} | MAE: {overall_mae:.2f} {unit} | "
        f"RMSE: {overall_rmse:.2f} {unit}"
    )
    camera_overall_rmse = None
    if INCLUDE_CAMERA_RESULTS:
        camera_overall_mae = np.nanmean(camera_overall_errors)
        camera_overall_rmse = np.sqrt(
            np.nanmean(camera_overall_errors**2)
        )
        overall_axis.plot(
            camera_time,
            camera_overall_errors,
            color="tab:green",
            label="Camera",
        )
        overall_title += (
            f"\nCamera | MAE: {camera_overall_mae:.2f} {unit} | "
            f"RMSE: {camera_overall_rmse:.2f} {unit}"
        )
    overall_axis.set_title(overall_title)
    overall_axis.set_ylabel(f"Error [{unit}]")
    overall_axis.set_xlabel("Synchronized time [s]")
    overall_axis.grid(True)
    overall_axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return overall_rmse, camera_overall_rmse


def create_comparison_plots(imu_rows):
    ground_truth_time, ground_truth = read_ground_truth(
        GROUND_TRUTH_PATH
    )
    imu_time = np.array([row["timestamp"] for row in imu_rows])
    position = np.array(
        [
            [row["x_mm"], row["y_mm"], row["z_mm"]]
            for row in imu_rows
        ]
    )
    orientation = np.array(
        [
            [row["roll_deg"], row["pitch_deg"], row["yaw_deg"]]
            for row in imu_rows
        ]
    )

    within_ground_truth = imu_time >= ground_truth_time[0]
    within_ground_truth &= imu_time <= ground_truth_time[-1]
    imu_time = imu_time[within_ground_truth]
    position = position[within_ground_truth]
    orientation = orientation[within_ground_truth]

    ground_truth_position = np.column_stack(
        [
            np.interp(
                imu_time,
                ground_truth_time,
                ground_truth[axis],
            )
            for axis in ("X", "Y", "Z")
        ]
    )
    unwrapped_ground_truth_orientation = np.column_stack(
        [
            np.degrees(
                np.unwrap(np.radians(ground_truth[axis]))
            )
            for axis in ("Roll", "Pitch", "Yaw")
        ]
    )
    ground_truth_orientation = np.column_stack(
        [
            np.interp(
                imu_time,
                ground_truth_time,
                unwrapped_ground_truth_orientation[:, axis],
            )
            for axis in range(3)
        ]
    )

    time_origin = min(imu_time[0], ground_truth_time[0])
    plot_time = imu_time - time_origin

    camera_time = None
    camera_position = None
    camera_orientation = None
    camera_position_component_errors = None
    camera_position_errors = None
    camera_orientation_component_errors = None
    camera_angular_errors = None
    if INCLUDE_CAMERA_RESULTS:
        camera_timestamp, camera = read_camera_results(
            CAMERA_RESULTS_PATH
        )
        camera_time = camera_timestamp - time_origin
        camera_position = np.column_stack(
            [camera[axis] for axis in ("X", "Y", "Z")]
        )
        camera_orientation = np.column_stack(
            [camera[axis] for axis in ("Roll", "Pitch", "Yaw")]
        )

        camera_ground_truth_position = np.column_stack(
            [
                np.interp(
                    camera_timestamp,
                    ground_truth_time,
                    ground_truth[axis],
                )
                for axis in ("X", "Y", "Z")
            ]
        )
        camera_ground_truth_orientation = np.column_stack(
            [
                np.interp(
                    camera_timestamp,
                    ground_truth_time,
                    unwrapped_ground_truth_orientation[:, axis],
                )
                for axis in range(3)
            ]
        )

        camera_position_component_errors = (
            camera_position - camera_ground_truth_position
        )
        camera_within_ground_truth = (
            camera_timestamp >= ground_truth_time[0]
        )
        camera_within_ground_truth &= (
            camera_timestamp <= ground_truth_time[-1]
        )
        camera_position_component_errors[
            ~camera_within_ground_truth
        ] = np.nan
        camera_position_errors = np.linalg.norm(
            camera_position_component_errors,
            axis=1,
        )
        camera_orientation_component_errors = (
            camera_orientation
            - camera_ground_truth_orientation
            + 180.0
        ) % 360.0 - 180.0

        valid_camera_orientation = np.isfinite(
            camera_orientation
        ).all(axis=1)
        valid_camera_orientation &= camera_within_ground_truth
        camera_angular_errors = np.full(
            len(camera_orientation),
            np.nan,
        )
        camera_rotations = Rotation.from_euler(
            "xyz",
            camera_orientation[valid_camera_orientation],
            degrees=True,
        )
        camera_ground_truth_rotations = Rotation.from_euler(
            "xyz",
            camera_ground_truth_orientation[
                valid_camera_orientation
            ],
            degrees=True,
        )
        camera_angular_errors[valid_camera_orientation] = np.degrees(
            (
                camera_ground_truth_rotations.inv()
                * camera_rotations
            ).magnitude()
        )

    position_component_errors = position - ground_truth_position
    position_errors = np.linalg.norm(
        position_component_errors,
        axis=1,
    )

    orientation_component_errors = (
        orientation - ground_truth_orientation + 180.0
    ) % 360.0 - 180.0
    estimate_rotations = Rotation.from_euler(
        "xyz",
        orientation,
        degrees=True,
    )
    ground_truth_rotations = Rotation.from_euler(
        "xyz",
        ground_truth_orientation,
        degrees=True,
    )
    angular_errors = np.degrees(
        (ground_truth_rotations.inv() * estimate_rotations).magnitude()
    )

    estimate_label = (
        "ESKF + camera"
        if USE_CAMERA_POSE_CORRECTION
        else "ESKF"
    )
    title = (
        f"{RECORDING_NAME}: {estimate_label} vs GT\n"
        "Data2 timestamps synchronized to Dobot"
    )

    position_rmse, camera_position_rmse = save_comparison_figure(
        plot_time,
        ground_truth_position,
        position,
        position_component_errors,
        position_errors,
        ["X", "Y", "Z"],
        "mm",
        "Euclidean distance",
        OUTPUT_POSITION_PLOT_PATH,
        f"{title} | position",
        estimate_label,
        camera_time,
        camera_position,
        camera_position_component_errors,
        camera_position_errors,
    )
    orientation_rmse, camera_orientation_rmse = save_comparison_figure(
        plot_time,
        ground_truth_orientation,
        orientation,
        orientation_component_errors,
        angular_errors,
        ["Roll", "Pitch", "Yaw"],
        "deg",
        "Angular distance",
        OUTPUT_ORIENTATION_PLOT_PATH,
        f"{title} | orientation",
        estimate_label,
        camera_time,
        camera_orientation,
        camera_orientation_component_errors,
        camera_angular_errors,
    )
    return (
        position_rmse,
        orientation_rmse,
        camera_position_rmse,
        camera_orientation_rmse,
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = run_filter()
    write_output(rows)
    (
        position_rmse,
        orientation_rmse,
        camera_position_rmse,
        camera_orientation_rmse,
    ) = create_comparison_plots(rows)
    print(f"Saved trajectory: {OUTPUT_CSV_PATH}")
    print(f"Saved position plot: {OUTPUT_POSITION_PLOT_PATH}")
    print(f"Saved orientation plot: {OUTPUT_ORIENTATION_PLOT_PATH}")
    print(f"Position RMSE: {position_rmse:.2f} mm")
    print(f"Orientation RMSE: {orientation_rmse:.2f} deg")
    if INCLUDE_CAMERA_RESULTS:
        print(f"Camera position RMSE: {camera_position_rmse:.2f} mm")
        print(
            "Camera orientation RMSE: "
            f"{camera_orientation_rmse:.2f} deg"
        )


if __name__ == "__main__":
    main()
