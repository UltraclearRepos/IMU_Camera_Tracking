"""Run the Wiener(81) IMU/camera comparison with the corrected Kalman filter."""

from pathlib import Path

import compare_wiener81_imu_camera as comparison
from FixedIntegratedKalmanFilter import FixedIntegratedKalmanFilter


# Keep dataset and processing configuration in compare_wiener81_imu_camera.py.
# Only the filter implementation and result location differ from the baseline.
comparison.KALMAN_CLASS = FixedIntegratedKalmanFilter
comparison.FUSION_LABEL = "IMU + native camera 30 Hz (fixed Kalman)"
comparison.OUTPUT_DIR = Path("wiener81_imu_camera_results_fix_filter")


if __name__ == "__main__":
    comparison.main()
