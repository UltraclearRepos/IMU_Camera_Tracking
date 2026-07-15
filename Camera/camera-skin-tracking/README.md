# Ultrasound Probe Tracking 6DoF

This repository contains a set of tools for 6DoF (6 Degrees of Freedom) tracking of an ultrasound probe using optical flow, camera calibration, and trajectory validation against an external Ground Truth reference system (ArUco marker tracking).

## File Overview

Here is a summary of the purpose of each script in the repository:

1. **`tracking_probe.py`**: The core tracking application. Performs real-time 6DoF probe tracking on a video file or live camera stream using Lucas-Kanade optical flow and homography decomposition.
2. **`evaluate_tracking.py`**: Evaluation tool that validates the optical flow tracking trajectory against ground truth trajectories obtained from an external reference camera tracking an ArUco marker. Generates comparison plots and accuracy reports.
3. **`auto_evaluate.py`**: Batch evaluation script that automatically scans a directory of measurements, runs the evaluation tool on all matching recordings, and aggregates MAE and RMSE metrics in a summary table.
4. **`calibrate.py`**: Interactive live camera calibration using a printed chessboard pattern.
5. **`calibrate_video.py`**: Chessboard calibration from a recorded calibration video file.
6. **`check_camera.py`**: Helper utility to query and print the resolution and frame rate capabilities of connected cameras.
7. **`get_aruco_id.py`**: Helper utility to detect ArUco markers from a live camera feed and print their IDs.

---

## Script Details & Usage

### 1. `tracking_probe.py`
Tracks the ultrasound probe's 6DoF pose. It initializes when it detects the ArUco marker **ID=7** (2 cm size), which establishes the initial depth scaling. Once calibrated, it tracks features on the skin using optical flow.

#### Usage
```bash
python tracking_probe.py [options]
```

#### Parameters
* `--video`: Path to an input `.mp4` video file. If not provided, the script opens a live camera feed. (Default: `None`)
* `--camera`: Index of the camera to open for live tracking. (Default: `1`)
* `--save_video`: Path to save the processed output video containing tracking visuals. (Default: `None`)
* `--save_csv`: Path to save the tracked coordinates trajectory as a `.csv` file. (Default: `None`)

---

### 2. `evaluate_tracking.py`
Compares the optical flow tracking trajectory against Ground Truth ArUco tracking (**ID=0**, 42.5 mm size) from an external camera. It aligns coordinate systems and generates comparison plots and text-based RMSE/MAE reports in the `scores/` folder.

#### Usage
```bash
python evaluate_tracking.py --video_probe <path_to_video> [options]
```

#### Parameters
* `--video_probe`: **(Required)** Path to the video recorded by the probe camera (runs optical flow tracking).
* `--video_ext`: Path to the external reference video file or a camera index (e.g. `0`, `1`) for live testing. (Default: `"1"`)
* `--calib_ext_dir`: Path to the calibration folder containing camera matrix and coefficients for the external camera. (Default: `"calibrations/camera_jabra_1920_1080"`)
* `--test_ext_only`: Run only the external ArUco reference detection/preview, skipping the probe optical flow tracking. (Flag)
* `--headless`: Run in background mode without displaying visual GUI windows (useful for batch scripts). (Flag)
* `--output_prefix`: Filename prefix for the saved accuracy reports and charts. (Default: `"tracking"`)
* `--homography_method`: OpenCV Homography estimation algorithm (e.g., `8` for RANSAC, `16` for RHO). (Default: `8` / RANSAC)
* `--ransac_threshold`: Reprojection error threshold for RANSAC. (Default: `3.0`)
* `--save_gt`: Path to save the aligned reference trajectory coordinates. (Default: `None`)
* `--save_of`: Path to save the aligned probe optical flow trajectory coordinates. (Default: `None`)
* `--load_gt`: Path to a previously saved reference CSV to load directly, skipping external video processing. (Default: `None`)
* `--axes`: Commas-separated list of active axes to evaluate (e.g. `"x,y,yaw"`). Inactive axes will be zeroed out. (Default: `"x,y,z,roll,pitch,yaw"`)
* `--aruco_camera_offset`: Lever arm offset vector between the ArUco marker and the probe camera in millimeters (e.g., `"0,30,136"`). (Default: `"0,0,0"`)
* `--align_frames`: Invert matrices and align the marker's coordinate system with the camera frame. Required for manual measurements where the external camera is set up at an arbitrary angle. (Flag)

---

### 3. `auto_evaluate.py`
Runs batch evaluation across multiple measurement directories containing `video_cam1` (probe camera) and `video_cam2` (external reference camera).

#### Usage
```bash
python auto_evaluate.py [options]
```

#### Parameters
* `--measurements_dir`: Path to the main directory containing measurement subfolders. (Default: `../measurements_with_imu`)
* `--axes`: Axes to evaluate for error reporting. (Default: `"x,y,z,roll,pitch,yaw"`)
* `--aruco_camera_offset`: Lever arm offset vector (x,y,z in mm) for hand-held measurements. (Default: `"0,0,0"`)

---

### 4. `calibrate.py`
A live camera calibration script using a printed chessboard pattern. Position the chessboard in front of the camera and press **'S'** to capture calibration frames. Press **'Q'** to finish and calculate calibration.

#### Usage
```bash
python calibrate.py
```
*Saves output matrices to `calibrations/camera_jabra_1920_1080/`.*

---

### 5. `calibrate_video.py`
Extracts chessboard corners from a recorded calibration video to calculate camera calibration.

#### Usage
```bash
python calibrate_video.py
```
* **Interactive Keys**:
  * `S`: Save current frame for calibration.
  * `SPACE`: Pause / Resume video.
  * `D`: Fast-forward 10 frames.
  * `Q`: Exit and calculate calibration.

*Note: Configuration parameters (such as `video_path`, `chessboard_size`, and `square_size`) are set directly at the top of the `calibrate_from_video()` function.*

---

### 6. `check_camera.py`
Checks if cameras at index `0` and `1` are available, printing their supported resolution and frame rate.

#### Usage
```bash
python check_camera.py
```

---

### 7. `get_aruco_id.py`
Opens a live camera feed at index `0` and prints any detected ArUco marker IDs to the console.

#### Usage
```bash
python get_aruco_id.py
```
