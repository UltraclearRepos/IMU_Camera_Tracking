import csv
import json
from pathlib import Path

import numpy as np

from geometry.coordinate_frames import imu_vectors_to_camera_axes


IMU_ACCELERATION_COLUMNS = (
    "imu2_ax_mg",
    "imu2_ay_mg",
    "imu2_az_mg",
)
IMU_GYROSCOPE_COLUMNS = (
    "imu2_gx_mdps",
    "imu2_gy_mdps",
    "imu2_gz_mdps",
)


class ImuGravityProvider:
    """Produces quality-gated gravity vectors in OpenCV camera axes."""

    def __init__(
        self,
        imu_path,
        calibration_path,
        video_start_timestamp,
        *,
        history_seconds,
        acceleration_magnitude_tolerance_m_s2,
        maximum_gyroscope_rad_s,
    ):
        self.imu_path = Path(imu_path)
        self.calibration_path = Path(calibration_path)
        self.video_start_timestamp = float(video_start_timestamp)
        self.history_seconds = float(history_seconds)
        self.acceleration_magnitude_tolerance_m_s2 = float(
            acceleration_magnitude_tolerance_m_s2
        )
        self.maximum_gyroscope_rad_s = float(maximum_gyroscope_rad_s)
        self.calibration = self._load_calibration()
        self.timestamps, self.accelerations_m_s2, self.gyroscopes_rad_s = (
            self._load_samples()
        )
        self._counts = {
            "accepted": 0,
            "no_samples": 0,
            "acceleration_magnitude": 0,
            "gyroscope": 0,
        }

    def _load_calibration(self):
        if not self.calibration_path.is_file():
            raise FileNotFoundError(
                f"IMU calibration not found: {self.calibration_path}"
            )
        with self.calibration_path.open(encoding="utf-8") as file:
            calibration = json.load(file)
        if calibration.get("imu_sensor") != "imu2":
            raise ValueError(
                "The gravity provider requires an IMU2 calibration"
            )
        if calibration.get("accel_model") != "matrix_times_raw_minus_bias":
            raise ValueError("Unsupported accelerometer calibration model")
        calibration["accel_matrix"] = np.asarray(
            calibration["accel_matrix"], dtype=np.float64
        )
        calibration["accel_bias_m_s2"] = np.asarray(
            calibration["accel_bias_m_s2"], dtype=np.float64
        )
        calibration["gyro_bias_rad_s"] = np.asarray(
            calibration["gyro_bias_rad_s"], dtype=np.float64
        )
        return calibration

    def _load_samples(self):
        if not self.imu_path.is_file():
            raise FileNotFoundError(f"IMU recording not found: {self.imu_path}")

        timestamps = []
        accelerations = []
        gyroscopes = []
        with self.imu_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required_columns = {
                "sync_timestamp",
                *IMU_ACCELERATION_COLUMNS,
                *IMU_GYROSCOPE_COLUMNS,
            }
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"IMU file {self.imu_path} lacks required columns"
                )
            for row in reader:
                timestamps.append(float(row["sync_timestamp"]))
                accelerations.append(
                    [float(row[column]) for column in IMU_ACCELERATION_COLUMNS]
                )
                gyroscopes.append(
                    [float(row[column]) for column in IMU_GYROSCOPE_COLUMNS]
                )

        timestamps = np.asarray(timestamps, dtype=np.float64)
        order = np.argsort(timestamps)
        raw_accelerations_m_s2 = (
            np.asarray(accelerations, dtype=np.float64)
            * self.calibration["gravity_m_s2"]
            / 1000.0
        )
        calibrated_accelerations_m_s2 = (
            self.calibration["accel_matrix"]
            @ (
                raw_accelerations_m_s2
                - self.calibration["accel_bias_m_s2"]
            ).T
        ).T
        raw_gyroscopes_rad_s = np.radians(
            np.asarray(gyroscopes, dtype=np.float64) / 1000.0
        )
        calibrated_gyroscopes_rad_s = (
            raw_gyroscopes_rad_s - self.calibration["gyro_bias_rad_s"]
        )
        return (
            timestamps[order],
            calibrated_accelerations_m_s2[order],
            calibrated_gyroscopes_rad_s[order],
        )

    def gravity_at_video_time(self, video_time_seconds):
        timestamp = self.video_start_timestamp + float(video_time_seconds)
        first = np.searchsorted(
            self.timestamps,
            timestamp - self.history_seconds,
            side="left",
        )
        last = np.searchsorted(
            self.timestamps,
            timestamp,
            side="right",
        )
        if first == last:
            self._counts["no_samples"] += 1
            return None, {"reason": "no synchronized IMU samples"}

        acceleration_m_s2 = np.median(
            self.accelerations_m_s2[first:last], axis=0
        )
        gyroscope_rad_s = np.median(
            self.gyroscopes_rad_s[first:last], axis=0
        )
        acceleration_magnitude_m_s2 = float(np.linalg.norm(acceleration_m_s2))
        gyroscope_magnitude_rad_s = float(np.linalg.norm(gyroscope_rad_s))
        diagnostics = {
            "samples": int(last - first),
            "acceleration_magnitude_m_s2": acceleration_magnitude_m_s2,
            "gyroscope_magnitude_rad_s": gyroscope_magnitude_rad_s,
        }
        if abs(acceleration_magnitude_m_s2 - self.calibration["gravity_m_s2"]) > self.acceleration_magnitude_tolerance_m_s2:
            self._counts["acceleration_magnitude"] += 1
            diagnostics["reason"] = "acceleration magnitude outside gate"
            return None, diagnostics
        if gyroscope_magnitude_rad_s > self.maximum_gyroscope_rad_s:
            self._counts["gyroscope"] += 1
            diagnostics["reason"] = "gyroscope magnitude outside gate"
            return None, diagnostics

        # A static accelerometer measures specific force, opposite to the
        # physical gravity vector.  COLMAP expects gravity in sensor axes.
        gravity_camera = -imu_vectors_to_camera_axes(acceleration_m_s2)
        gravity_camera /= np.linalg.norm(gravity_camera)
        self._counts["accepted"] += 1
        diagnostics["reason"] = "accepted"
        diagnostics["gravity_camera"] = gravity_camera
        return gravity_camera, diagnostics

    def summary(self):
        return {
            "imu_path": str(self.imu_path),
            "calibration_path": str(self.calibration_path),
            "imu_sensor": self.calibration["imu_sensor"],
            "calibration_version": self.calibration["calibration_version"],
            "history_seconds": self.history_seconds,
            "acceleration_magnitude_tolerance_m_s2": (
                self.acceleration_magnitude_tolerance_m_s2
            ),
            "maximum_gyroscope_rad_s": self.maximum_gyroscope_rad_s,
            "counts": self._counts.copy(),
        }
