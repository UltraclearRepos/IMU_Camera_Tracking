# Sensor Fusion for Ultrasound Probe Tracking using Inertial Positioning

This repository contains a set of tools for tracking of an ultrasound probe. The project focuses on the implementation and comparison of various sensor fusion algorithms combining Inertial Measurement Unit (IMU) data with a vision system (camera/optical flow) for accurate positional tracking (e.g., of an ultrasound probe). The estimation results are evaluated and compared against a reference measurement system (Ground Truth), which in this project is a Dobot robotic arm.

## Data Structure

The scripts in this project (including the main notebook) by default expect the following directory structure for the measurement data (e.g., inside a `Data/` folder):

*   **`IMU/`** - contains raw logs from the inertial sensor (`.TXT` or `.csv` files). This includes accelerometer and gyroscope readings.
*   **`dobot/`** - contains `.csv` files with reference data (Ground Truth) recorded by the robotic arm's encoders (exact X, Y, Z positions).
*   **`POMIARY/`** - contains subfolders for specific experiments (e.g., `horizontal_line_10x_5sp__x/`), which house the `optical_flow.csv` files containing vision-based tracking data (camera).

## Modules and Available Functions

The codebase is divided into modules responsible for specific signal processing stages:

### 1. `funkcje_IMU.py`
Module for handling and preprocessing data from the IMU sensor.
*   `load_IMU_data()` - Loads raw logs and converts units to standard SI (m/s² for accelerometer, rad/s for gyroscope).
*   `compute_orientation_and_global_acc()` - Implements a complementary filter to determine orientation (Roll, Pitch, Yaw) and transforms local acceleration to the global coordinate frame (removing gravity).
*   `calculate_integrals()` - Integrates global acceleration into velocity and position, with optional use of High-pass filters and signal detrending.

### 2. `funkcje_camera.py`
Module for handling vision system data.
*   `load_camera_data()`, `resample_camera_data()` - Loads data, converts units, and unifies the sampling frequency (resampling).
*   `synchronize_imu_camera()` - Synchronizes the time axes of the vision and IMU data based on cross-correlation.
*   `apply_camera_confidence()` - Simulates the confidence level of the camera (e.g., adding noise or "freezing" the position during tracking loss).
*   `calculate_final_metrics()` - Calculates final drift errors (RMSE, mean error, median) for 1-second and 10-second evaluation windows.

### 3. `funkcje_GT.py`
Module for processing reference data (Ground Truth from the robot).
*   `load_ground_truth()`, `normalize_ground_truth()`, `resample_ground_truth()` - Loads, normalizes (starting from point 0,0,0), and resamples the data.
*   `calculate_derivatives()` - Calculates velocity and acceleration from the robot's position using smoothing filters (Savitzky-Golay and median filter) to eliminate differentiation noise.

### 4. `funkcje_IMU_GT.py`
Cross-operations between IMU and Ground Truth.
*   `synchronize_by_cross_correlation()` - Precisely synchronizes the IMU signal with the Ground Truth using cross-correlation of the acceleration signals.
*   Plotting functions (`plot_velocity_and_position`, `plot_synchronized_results`) to compare IMU, filtering methods, and GT.

### 5. `IMUFilter.py`
The `IMUFilter` class containing tools for filtering noisy accelerometer signals.
*   **Implemented filters:** Butterworth (low-pass), Savitzky-Golay, Median, Moving Average, Wiener Filter.
*   `test_filter_parameters()` - An automated function that optimizes parameters (e.g., window size, cutoff frequency) for each filter by comparing the results with the Ground Truth (minimizing RMSE).

### 6. `SensorFusionEngine.py`
The main engine executing the IMU and camera data fusion. The `SensorFusionEngine` class provides various research approaches:
*   `approach_weighted_average` - Simple weighted average.
*   `approach_complementary_adaptive_filter` - Adaptive complementary filter that dynamically changes weights based on camera measurement confidence.
*   `approach_error_drift_compensation` - Drift compensation based on smoothing the difference (error) between the IMU and camera positions.
*   `approach_kalman_adaptive_filter` - Fusion using a 9-state Kalman filter.
*   `approach_wiener_filter` - Fusion using a Wiener filter.
*   `approach_loop_closure` - Optional forced error correction at the end of the trajectory (assuming a return to the starting point).

### 7. `IntegratedKalmanFilter.py`
A dedicated implementation of a 9-state Integrated Kalman Filter (position, velocity, acceleration bias) for combining accelerometer data with vision system position. It includes logic for updating the measurement noise covariance matrix `R` based on the camera's confidence vector and a `Q` & `R` parameter optimizer.

---

## Example Pipeline and Main Notebook

All the modules mentioned above are tied together in the main project file – the Jupyter Notebook **`IMU_3.0.ipynb`**. This is where the core code execution and the logic for running experiments take place.

**A typical data pipeline implemented in the notebook looks like this:**

1. **Initialization and Data Loading:** Loading files for a specific test (e.g., `horizontal_line_1`) for IMU, Camera, and GT.
2. **IMU Preprocessing:** Unit conversion, orientation calculation, and transformation to global acceleration without gravity (`compute_orientation_and_global_acc`).
3. **GT Preprocessing:** Smoothing the robot's position and differentiating it twice to obtain reference acceleration (`calculate_derivatives`).
4. **Synchronization:** Time-aligning the IMU and GT logs using cross-correlation of acceleration waveforms (`synchronize_by_cross_correlation`).
5. **IMU Filtering:** Applying tested filters (e.g., Butterworth) to the acceleration to minimize noise (`IMUFilter`).
6. **Integration & Camera Sync:** Double integration of the filtered acceleration into position (`calculate_integrals`) and syncing it with the camera data.
7. **Sensor Fusion:** Running the `SensorFusionEngine`, which calculates the new `fused_pos` for the synchronized timelines based on selected algorithms (e.g., Kalman, Complementary Adaptive).
8. **Evaluation & Analysis:** Calculating RMSE errors for 1s and 10s windows (`calculate_final_metrics`) and generating plots comparing trajectories and drift values.

*(An automated loop iterating through all datasets can be found in the `IMU_3.0` notebook within the `whole_process_pipeline` and `test_filters_on_all_data` functions).*

## Requirements
The project requires Python 3.8+ and the following libraries:
*   `numpy`
*   `pandas`
*   `scipy`
*   `matplotlib`

To run the notebook, you also need to have the `jupyter` environment installed.
