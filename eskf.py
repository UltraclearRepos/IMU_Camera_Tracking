"""IMU-only error-state Kalman filter and comparison with ground truth."""

import csv
import math
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_ROOT = Path("Data")

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

DATASET_NAME = "horizontal_line_1"
IMU_LOG_PATH = DATA_ROOT / DATASETS[DATASET_NAME]["imu"]
GROUND_TRUTH_PATH = DATA_ROOT / DATASETS[DATASET_NAME]["gt"]
OUTPUT_CSV_PATH = Path(f"eskf_{DATASET_NAME}.csv")
OUTPUT_PLOT_PATH = Path(f"eskf_vs_gt_{DATASET_NAME}.png")

INITIALIZATION_SECONDS = 2.0
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
    def __init__(self):
        self.accel_noise_std = 0.02
        self.gyro_noise_std = math.radians(0.5)
        self.accel_bias_random_walk = 5e-4
        self.gyro_bias_random_walk = math.radians(0.01)
        self.zero_velocity_std = 0.01
        self.stationary_accel_std = 0.03
        self.stationary_gyro_std = math.radians(0.2)


class ErrorStateKalmanFilter:
    def __init__(self):
        self.parameters = EskfParameters()
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

        expected_specific_force = (
            rotation_matrix(self.quaternion).T @ -self.gravity_world
        )
        self.accel_bias = mean_acceleration - expected_specific_force

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
                np.full(3, parameters.accel_noise_std**2),
                np.full(3, parameters.gyro_noise_std**2),
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


def read_imu_log(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    acceleration_mg = np.array(
        [
            [float(row[key]) for key in ("aX", "aY", "aZ")]
            for row in rows
        ]
    )
    angular_rate_mdps = np.array(
        [
            [float(row[key]) for key in ("gX", "gY", "gZ")]
            for row in rows
        ]
    )
    sample_rate_hz = np.mean(
        [float(row["output_Hz"]) for row in rows]
    )

    acceleration = acceleration_mg * GRAVITY / 1000.0
    angular_rate = np.radians(angular_rate_mdps / 1000.0)
    return acceleration, angular_rate, sample_rate_hz


def read_ground_truth(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    time = np.array([float(row["timestamp"]) for row in rows])
    time -= time[0]

    x = np.array([float(row["X"]) for row in rows])
    y = np.array([float(row["Y"]) for row in rows])
    z = np.array([float(row["Z"]) for row in rows])
    yaw = np.array([float(row["R"]) for row in rows])

    values = {
        "X": x - x[0],
        "Y": y - y[0],
        "Z": z - z[0],
        "Roll": np.zeros(len(rows)),
        "Pitch": np.zeros(len(rows)),
        "Yaw": yaw - yaw[0],
    }
    return time, values


def run_filter():
    acceleration, angular_rate, sample_rate_hz = read_imu_log(IMU_LOG_PATH)
    dt = 1.0 / sample_rate_hz
    init_samples = round(INITIALIZATION_SECONDS * sample_rate_hz)

    eskf = ErrorStateKalmanFilter()
    eskf.initialize(
        acceleration[:init_samples],
        angular_rate[:init_samples],
    )
    initial_quaternion = eskf.quaternion.copy()
    detector = StationaryDetector(sample_rate_hz)

    output = []
    for index in range(len(acceleration)):
        stationary_update = index < init_samples

        if index >= init_samples:
            eskf.predict(
                acceleration[index],
                angular_rate[index],
                dt,
            )
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

        relative_quaternion = quaternion_multiply(
            quaternion_conjugate(initial_quaternion),
            eskf.quaternion,
        )
        roll, pitch, yaw = euler_degrees(relative_quaternion)
        position_sigma = np.sqrt(np.diag(eskf.covariance)[0:3])

        output.append(
            {
                "time_s": index * dt,
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
            }
        )

    return output


def write_output(rows):
    with OUTPUT_CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def create_comparison_plot(imu_rows):
    ground_truth_time, ground_truth = read_ground_truth(GROUND_TRUTH_PATH)
    imu_time = np.array([row["time_s"] for row in imu_rows])

    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    position_columns = (
        ("x_mm", "X", "X [mm]"),
        ("y_mm", "Y", "Y [mm]"),
        ("z_mm", "Z", "Z [mm]"),
    )
    orientation_columns = (
        ("roll_deg", "Roll", "Roll [deg]"),
        ("pitch_deg", "Pitch", "Pitch [deg]"),
        ("yaw_deg", "Yaw", "Yaw [deg]"),
    )

    for row_index, columns in enumerate(position_columns):
        imu_column, ground_truth_column, label = columns
        axis = axes[row_index, 0]
        axis.plot(
            imu_time,
            [row[imu_column] for row in imu_rows],
            label="ESKF",
        )
        axis.plot(
            ground_truth_time,
            ground_truth[ground_truth_column],
            label="Ground truth",
        )
        axis.set_ylabel(label)
        axis.grid(True)
        axis.legend()

    for row_index, columns in enumerate(orientation_columns):
        imu_column, ground_truth_column, label = columns
        axis = axes[row_index, 1]
        axis.plot(
            imu_time,
            [row[imu_column] for row in imu_rows],
            label="ESKF",
        )
        axis.plot(
            ground_truth_time,
            ground_truth[ground_truth_column],
            label="Ground truth",
        )
        axis.set_ylabel(label)
        axis.grid(True)
        axis.legend()

    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    figure.suptitle(f"IMU ESKF vs ground truth: {DATASET_NAME}")
    figure.tight_layout()
    figure.savefig(OUTPUT_PLOT_PATH, dpi=160)
    plt.close(figure)


def main():
    rows = run_filter()
    write_output(rows)
    create_comparison_plot(rows)
    print(f"Saved trajectory: {OUTPUT_CSV_PATH}")
    print(f"Saved comparison plot: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
