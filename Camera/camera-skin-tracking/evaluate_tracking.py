import cv2
import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
import os
import math

from tracking_probe import UltrasoundProbeTracker6DoF, estimate_marker_poses

def rotation_matrix_to_euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([np.degrees(x), np.degrees(y), np.degrees(z)])

def euler_to_rotation_matrix(euler_deg):
    """Inversion of Euler ZYX (yaw, pitch, roll) to rotation matrix."""
    roll, pitch, yaw = np.radians(euler_deg)
    
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
                    
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
                    
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])
                    
    R = R_z @ R_y @ R_x
    return R

class GroundTruthArUcoTracker:
    """Class for computing position from an external reference camera looking at the probe (detects ID=0)."""
    def __init__(self, camera_matrix, dist_coeffs, v_offset=None):
        self.K = camera_matrix
        self.D = dist_coeffs
        self.ARUCO_SIZE_MM = 42.5 
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.aruco_params,
        )
        
        self.initial_tvec = None
        self.initial_rvec = None
        self.v_offset = v_offset if v_offset is not None else np.array([0.0, 0.0, 0.0])

    def process(self, frame, align_frames=False):
        """Returns relative X, Y, Z with respect to (0,0,0) of the first video frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if ids is not None:
            idx_list = np.where(ids == 0)[0]
            if len(idx_list) > 0:
                marker_idx = idx_list[0]
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs = estimate_marker_poses(
                    corners,
                    self.ARUCO_SIZE_MM,
                    self.K,
                    self.D,
                )
                
                tvec = tvecs[marker_idx][0]
                rvec = rvecs[marker_idx][0]
                
                if self.initial_tvec is None:
                    self.initial_tvec = tvec.copy()
                    self.initial_rvec = rvec.copy()
                    print(f"[GT_Tracker] Initial distance from camera to marker: {self.initial_tvec[2]:.1f}mm")
                
                diff_tvec = tvec - self.initial_tvec
                
                R_init, _ = cv2.Rodrigues(self.initial_rvec)
                R_cur, _ = cv2.Rodrigues(rvec)
                R_diff = R_init.T @ R_cur
                
                if align_frames:
                    diff_tvec_local = R_init.T @ diff_tvec
                    
                    if np.any(self.v_offset != 0):
                        offset_translation = (R_diff - np.eye(3)) @ self.v_offset
                        diff_tvec_local = diff_tvec_local + offset_translation
                        
                    diff_tvec = diff_tvec_local
                else:
                    if np.any(self.v_offset != 0):
                        offset_translation = (R_cur - R_init) @ self.v_offset
                        diff_tvec = diff_tvec + offset_translation
                    
                diff_euler = rotation_matrix_to_euler(R_diff)
                
                # Draw parameters on the screen
                cv2.rectangle(frame, (10, 10), (300, 100), (0,0,0), -1)
                cv2.putText(frame, "ARUCO REFERENCE [mm]", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (200,200,200), 1)
                cv2.putText(frame, f"X: {diff_tvec[0]:.1f}", (20, 55), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Y: {diff_tvec[1]:.1f}", (150, 55), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Z: {diff_tvec[2]:.1f}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 150, 255), 2)
                
                return diff_tvec, diff_euler, frame
 
        # If no marker (or no calibration), return None
        return None, None, frame

def main():
    parser = argparse.ArgumentParser(description="Comparison of Optical Flow tracking with external ArUco reference tracking.")
    parser.add_argument("--video_probe", type=str, required=False, help="Path to probe video (Optical Flow, ID=7)")
    parser.add_argument("--video_ext", type=str, required=False, default="1", help="Path to external reference video or camera index (e.g. 0, 1) for live test")
    parser.add_argument("--calib_ext_dir", type=str, default="Camera/camera-skin-tracking/calibrations/camera_jabra_1920_1080", help="Calibration directory for external reference camera.")
    parser.add_argument("--test_ext_only", action="store_true", help="Run only ArUco preview without probe Optical Flow evaluation.")
    parser.add_argument("--headless", action="store_true", help="Headless mode - run in background without visual windows to compute statistics.")
    parser.add_argument("--output_prefix", type=str, default="Camera\camera-skin-tracking/tracking", help="Prefix for saved results files and plots.")
    parser.add_argument("--homography_method", type=int, default=cv2.RANSAC, help="Homography algorithm (e.g., 8 for RANSAC, 16 for RHO).")
    parser.add_argument("--ransac_threshold", type=float, default=3.0, help="Reprojection error threshold for homography.")
    parser.add_argument("--save_gt", type=str, default=None, help="Path to save the ground truth ArUco trajectory CSV.")
    parser.add_argument("--save_of", type=str, default=None, help="Path to save the optical flow probe trajectory CSV.")
    parser.add_argument("--load_gt", type=str, default=None, help="Path to load Ground Truth ArUco CSV, skipping reference video processing.")
    parser.add_argument("--axes", type=str, default="x,y,z,roll,pitch,yaw", help="Axes to evaluate separated by commas, e.g., 'x,y,yaw'. Others will be zeroed out.")
    parser.add_argument("--aruco_camera_offset", type=str, default="0,0,0", help="Offset vector between ArUco and camera in millimeters (e.g., '0,30,136')")
    parser.add_argument("--align_frames", action="store_true", help="Invert matrices and align marker frame of reference with camera (required for manual measurements with arbitrary external camera angle)")
    args = parser.parse_args()

    active_axes = [a.strip().lower() for a in args.axes.split(',')]
    use_x = 'x' in active_axes
    use_y = 'y' in active_axes
    use_z = 'z' in active_axes
    use_r = 'roll' in active_axes
    use_p = 'pitch' in active_axes
    use_yaw = 'yaw' in active_axes
    
    offset_parts = [float(x.strip()) for x in args.aruco_camera_offset.split(',')]
    v_offset = np.array(offset_parts)
    if np.any(v_offset != 0):
        print(f"[INFO] Set lever arm offset ArUco -> Camera: X={v_offset[0]}mm, Y={v_offset[1]}mm, Z={v_offset[2]}mm")

    if not args.test_ext_only and not args.video_probe:
        print("Probe video path --video_probe not specified (run with --test_ext_only to use external camera only)")
        return

    # Load probe camera matrix (hardcoded option)
    if not args.test_ext_only:
        probe_dir = "Camera/camera-skin-tracking/calibrations/camera_jabra_640_360"
        k_probe_path = os.path.join(probe_dir, "camera_matrix.npy")
        d_probe_path = os.path.join(probe_dir, "dist_coeffs.npy")
        
        if not os.path.exists(k_probe_path) or not os.path.exists(d_probe_path):
            print(f"Probe calibration files missing! Ensure they exist in: {probe_dir}")
            return
        K = np.load(k_probe_path)
        D = np.load(d_probe_path)

    # Load reference / external camera matrix (variable from argument)
    k_ext_path = os.path.join(args.calib_ext_dir, "camera_matrix.npy")
    d_ext_path = os.path.join(args.calib_ext_dir, "dist_coeffs.npy")

    if not os.path.exists(k_ext_path) or not os.path.exists(d_ext_path):
        print(f"External reference calibration files missing! Ensure they exist in: {args.calib_ext_dir}")
        return
    K_ext = np.load(k_ext_path)
    D_ext = np.load(d_ext_path)

    if not args.test_ext_only:
        cap_probe = cv2.VideoCapture(args.video_probe)
        if not cap_probe.isOpened():
            print("Failed to open probe video.")
            return

    loaded_gt_data = []
    if args.load_gt:
        if not os.path.exists(args.load_gt):
            print(f"Ground Truth file to load does not exist: {args.load_gt}")
            return
        with open(args.load_gt, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 4 and row[1]:
                    loaded_gt_data.append(np.array([
                        float(row[1]), float(row[2]), float(row[3]),
                        float(row[4]) if len(row) > 4 else 0.0,
                        float(row[5]) if len(row) > 5 else 0.0,
                        float(row[6]) if len(row) > 6 else 0.0
                    ]))
                else:
                    loaded_gt_data.append(None)
        cap_ext = None
    else:
        # Detect if video path (.mp4) or camera index was passed for live test
        ext_source = int(args.video_ext) if args.video_ext.isdigit() else args.video_ext
        cap_ext = cv2.VideoCapture(ext_source)
        
        # If opening camera by index, ensure 1080p resolution (required for calibration matrix)
        if isinstance(ext_source, int):
            cap_ext.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap_ext.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap_ext.isOpened():
            print("Failed to open external video file or stream.")
            return

    # Initialize trackers
    if not args.test_ext_only:
        tracker_of = UltrasoundProbeTracker6DoF(K, D, homography_method=args.homography_method, ransac_threshold=args.ransac_threshold) 
    if not args.load_gt:
        tracker_gt = GroundTruthArUcoTracker(K_ext, D_ext, v_offset=v_offset)

    gt_file_writer = None
    if args.save_gt:
        # Open CSV file for writing
        gt_path = os.path.join(args.output_prefix, args.save_gt)
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        gt_file = open(gt_path, 'w', encoding='utf-8', newline='')
        gt_file_writer = csv.writer(gt_file)
        gt_file_writer.writerow(['Frame', 'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'])

    of_file_writer = None
    if args.save_of:
        # Open CSV file for writing
        of_path = os.path.join(args.output_prefix, args.save_of)
        os.makedirs(os.path.dirname(of_path), exist_ok=True)
        of_file = open(of_path, 'w', encoding='utf-8', newline='')
        of_file_writer = csv.writer(of_file)
        of_file_writer.writerow(['Frame', 'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'])

    # Store trajectory history
    history_of_x, history_of_y, history_of_z = [], [], []
    history_gt_x, history_gt_y, history_gt_z = [], [], []
    history_of_e, history_gt_e = [], []

    offset_of_t = None
    offset_of_R = None
    
    frame_idx = 0
    while True:
        if not args.test_ext_only:
            ret1, frame_probe = cap_probe.read()
        else:
            ret1, frame_probe = False, None
            
        frame_ext = None
        if args.load_gt:
            if frame_idx < len(loaded_gt_data):
                gt_data = loaded_gt_data[frame_idx]
                ret2 = True
                out_ext = np.zeros((1080, 1920, 3), dtype=np.uint8)
                cv2.putText(out_ext, "Loading Ground Truth from file...", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
                if gt_data is not None:
                    gt_tvec = gt_data[:3]
                    gt_euler = gt_data[3:]
                    cv2.putText(out_ext, f"X: {gt_tvec[0]:.1f}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    cv2.putText(out_ext, f"Y: {gt_tvec[1]:.1f}", (50, 140), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    cv2.putText(out_ext, f"Z: {gt_tvec[2]:.1f}", (50, 180), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 150, 255), 2)
                else:
                    gt_tvec, gt_euler = None, None
            else:
                ret2 = False
                gt_tvec, gt_euler = None, None
                out_ext = np.zeros((1080, 1920, 3), dtype=np.uint8)
        else:
            ret2, frame_ext = cap_ext.read()
            if ret2:
                gt_tvec, gt_euler, out_ext = tracker_gt.process(frame_ext, align_frames=args.align_frames)
        
        if not ret2 or (not args.test_ext_only and not ret1):
            break

        if gt_tvec is not None and gt_euler is not None:
            if args.align_frames:
                gt_euler[0] = -gt_euler[0]
                
            gt_tvec[0] = -gt_tvec[0]  # Flip the X axis for GroundTruth
            if not use_x: gt_tvec[0] = 0.0
            if not use_y: gt_tvec[1] = 0.0
            if not use_z: gt_tvec[2] = 0.0
            if not use_r: gt_euler[0] = 0.0
            if not use_p: gt_euler[1] = 0.0
            if not use_yaw: gt_euler[2] = 0.0

        if gt_file_writer:
            if gt_tvec is not None and gt_euler is not None:
                gt_file_writer.writerow([frame_idx, f"{gt_tvec[0]:.4f}", f"{gt_tvec[1]:.4f}", f"{gt_tvec[2]:.4f}", f"{gt_euler[0]:.4f}", f"{gt_euler[1]:.4f}", f"{gt_euler[2]:.4f}"])
            else:
                gt_file_writer.writerow([frame_idx, '', '', '', '', '', ''])

        # ARUCO ONLY PREVIEW TEST:
        if args.test_ext_only:
            if not args.headless:
                out_ext = cv2.resize(out_ext, (1280, 720))
                cv2.imshow("External Evaluation Test (ArUco Only)", out_ext)
                if cv2.waitKey(1) == ord('q'):
                    break
            frame_idx += 1
            continue

        out_probe = tracker_of.process(frame_probe)
        
        is_gt_ready = True if args.load_gt else tracker_gt.initial_tvec is not None
        if tracker_of.is_calibrated and is_gt_ready:
            if offset_of_t is None:
                offset_of_t = tracker_of.cur_t.copy()
                offset_of_R = tracker_of.cur_R.copy()
            
            rel_of_x = (tracker_of.cur_t[0][0] - offset_of_t[0][0])
            rel_of_y = (tracker_of.cur_t[1][0] - offset_of_t[1][0])
            rel_of_z = (tracker_of.cur_t[2][0] - offset_of_t[2][0])

            R_diff_of = offset_of_R.T @ tracker_of.cur_R
            rel_of_euler = rotation_matrix_to_euler(R_diff_of)
            rel_of_roll, rel_of_pitch, rel_of_yaw = rel_of_euler

            if not use_x: rel_of_x = 0.0
            if not use_y: rel_of_y = 0.0
            if not use_z: rel_of_z = 0.0
            if not use_r: rel_of_roll = 0.0
            if not use_p: rel_of_pitch = 0.0
            if not use_yaw: rel_of_yaw = 0.0

            rel_of_euler = np.array([rel_of_roll, rel_of_pitch, rel_of_yaw])

            history_of_x.append(rel_of_x)
            history_of_y.append(rel_of_y)
            history_of_z.append(rel_of_z)
            history_of_e.append(rel_of_euler)

            if of_file_writer:
                of_file_writer.writerow([frame_idx, f"{rel_of_x:.4f}", f"{rel_of_y:.4f}", f"{rel_of_z:.4f}", f"{rel_of_roll:.4f}", f"{rel_of_pitch:.4f}", f"{rel_of_yaw:.4f}"])

            if gt_tvec is not None and gt_euler is not None:
                rel_gt_x, rel_gt_y, rel_gt_z = gt_tvec[0], gt_tvec[1], gt_tvec[2]
                rel_gt_euler = gt_euler.copy()
            else:
                rel_gt_x = history_gt_x[-1] if history_gt_x else 0.0
                rel_gt_y = history_gt_y[-1] if history_gt_y else 0.0
                rel_gt_z = history_gt_z[-1] if history_gt_z else 0.0
                rel_gt_euler = history_gt_e[-1] if history_gt_e else np.array([0., 0., 0.])
            if not use_x: rel_gt_x = 0.0
            if not use_y: rel_gt_y = 0.0
            if not use_z: rel_gt_z = 0.0
            if not use_r: rel_gt_euler[0] = 0.0
            if not use_p: rel_gt_euler[1] = 0.0
            if not use_yaw: rel_gt_euler[2] = 0.0

            history_gt_x.append(rel_gt_x)
            history_gt_y.append(rel_gt_y)
            history_gt_z.append(rel_gt_z)
            history_gt_e.append(rel_gt_euler)

        if not args.headless:
            out_probe = cv2.resize(out_probe, (640, 360))
            out_ext = cv2.resize(out_ext, (640, 360))
            combined_view = np.hstack((out_probe, out_ext))
            
            cv2.putText(combined_view, "OF (ID=7)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined_view, "EXT Truth (ID=0)", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Evaluation: OF vs Ext. Truth", combined_view)

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            if frame_idx % 100 == 0 and frame_idx > 0:
                print(f"[Processing in background...] frame {frame_idx}")
            
        frame_idx += 1

    if not args.test_ext_only:
        cap_probe.release()
    if cap_ext:
        cap_ext.release()
    if args.save_gt:
        gt_file.close()
    if args.save_of:
        of_file.close()

    cv2.destroyAllWindows()

    if args.test_ext_only:
        if not args.headless:
            print("[INFO] External camera preview finished. No evaluation plot to generate.")
        return

    if not args.headless:
        print(f"Analyzed {len(history_of_x)} frames.")
        
    if not history_of_x:
        if not args.headless: print("Could not log any frames from the sequence.")
        return None

    # Generate validation plots after processing
    frames = list(range(len(history_of_x)))

    plt.figure(figsize=(18, 16))
    plt.suptitle("6DoF Tracking Validation: OF vs ArUco", fontsize=18)

    # Convert Euler history from list to numpy array
    h_of_e = np.array(history_of_e)
    h_gt_e = np.array(history_gt_e)
    
    h_of_r, h_of_p, h_of_yaw = h_of_e[:,0], h_of_e[:,1], h_of_e[:,2]
    h_gt_r, h_gt_p, h_gt_yaw = h_gt_e[:,0], h_gt_e[:,1], h_gt_e[:,2]

    # --- Plotting translation values ---
    plt.subplot(4, 2, 1)
    plt.plot(frames, history_of_x, label="OF (Probe)", color='blue')
    plt.plot(frames, history_gt_x, label="GT (ArUco)", color='green', linestyle='dashed')
    plt.title("X [mm]"); plt.grid(); plt.legend()

    plt.subplot(4, 2, 3)
    plt.plot(frames, history_of_y, label="OF", color='blue')
    plt.plot(frames, history_gt_y, label="GT", color='green', linestyle='dashed')
    plt.title("Y [mm]"); plt.grid()

    plt.subplot(4, 2, 5)
    plt.plot(frames, history_of_z, label="OF", color='blue')
    plt.plot(frames, history_gt_z, label="GT", color='green', linestyle='dashed')
    plt.title("Z [mm]"); plt.grid()

    # --- Plotting rotation values ---
    plt.subplot(4, 2, 2)
    plt.plot(frames, h_of_r, label="OF", color='orange')
    plt.plot(frames, h_gt_r, label="GT", color='purple', linestyle='dashed')
    plt.title("Roll [deg]"); plt.grid(); plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(frames, h_of_p, label="OF", color='orange')
    plt.plot(frames, h_gt_p, label="GT", color='purple', linestyle='dashed')
    plt.title("Pitch [deg]"); plt.grid()

    plt.subplot(4, 2, 6)
    plt.plot(frames, h_of_yaw, label="OF", color='orange')
    plt.plot(frames, h_gt_yaw, label="GT", color='purple', linestyle='dashed')
    plt.title("Yaw [deg]"); plt.grid()

    # Computing Mean Squared Error (MSE)
    mse_errors_trans = []
    mse_errors_rot = []
    
    n_frames = len(history_of_x)
    total_sq_x = total_sq_y = total_sq_z = 0.0
    total_sq_r = total_sq_p = total_sq_yaw = 0.0

    for i in range(n_frames):
        ox, oy, oz = history_of_x[i], history_of_y[i], history_of_z[i]
        gx, gy, gz = history_gt_x[i], history_gt_y[i], history_gt_z[i]
        
        sq_x, sq_y, sq_z = (ox-gx)**2, (oy-gy)**2, (oz-gz)**2
        sq_trans = sq_x + sq_y + sq_z
        mse_errors_trans.append(sq_trans)
        
        total_sq_x += sq_x; total_sq_y += sq_y; total_sq_z += sq_z

        # Rotation history values
        oro, op, oyaw = h_of_r[i], h_of_p[i], h_of_yaw[i]
        gro, gp, gyaw = h_gt_r[i], h_gt_p[i], h_gt_yaw[i]
        
        # Handle wrap-around at 360 degrees for angle difference
        diff_r = (oro - gro + 180) % 360 - 180
        diff_p = (op - gp + 180) % 360 - 180
        diff_yaw = (oyaw - gyaw + 180) % 360 - 180

        sq_r, sq_p, sq_yaw = diff_r**2, diff_p**2, diff_yaw**2
        sq_rot = sq_r + sq_p + sq_yaw
        mse_errors_rot.append(sq_rot)

        total_sq_r += sq_r; total_sq_p += sq_p; total_sq_yaw += sq_yaw

    rmse_x = math.sqrt(total_sq_x / max(n_frames, 1))
    rmse_y = math.sqrt(total_sq_y / max(n_frames, 1))
    rmse_z = math.sqrt(total_sq_z / max(n_frames, 1))
    final_rmse_trans = math.sqrt((total_sq_x + total_sq_y + total_sq_z) / max(n_frames, 1))

    rmse_r = math.sqrt(total_sq_r / max(n_frames, 1))
    rmse_p = math.sqrt(total_sq_p / max(n_frames, 1))
    rmse_yaw = math.sqrt(total_sq_yaw / max(n_frames, 1))
    final_rmse_rot = math.sqrt((total_sq_r + total_sq_p + total_sq_yaw) / max(n_frames, 1))

    # Instantaneous Euclidean errors for plotting in mm and deg
    euclidean_errors_trans = [math.sqrt(err) for err in mse_errors_trans]
    euclidean_errors_rot = [math.sqrt(err) for err in mse_errors_rot]

    # Compute Mean Absolute Error (MAE)
    mae_trans = sum(euclidean_errors_trans) / max(n_frames, 1)
    mae_rot = sum(euclidean_errors_rot) / max(n_frames, 1)

    # Plot total translation error (Euclidean distance)
    plt.subplot(4, 2, 7)
    plt.plot(frames, euclidean_errors_trans, color='red', linewidth=1.5)
    plt.title(f"Euclidean Distance | MAE: {mae_trans:.2f} mm | RMSE: {final_rmse_trans:.2f} mm")
    plt.grid()
    
    # Plot total rotation error (Angular distance)
    plt.subplot(4, 2, 8)
    plt.plot(frames, euclidean_errors_rot, color='red', linewidth=1.5)
    plt.title(f"Angular Distance | MAE: {mae_rot:.2f} deg | RMSE: {final_rmse_rot:.2f} deg")
    plt.grid()
    
    # Save results report to a text file
    txt_path = os.path.join(args.output_prefix, f"rmse_result.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("=== 6DoF Validation Accuracy Report ===\n")
        f.write(f"Analyzed frames: {n_frames}\n\n")
        f.write("[ Translation ]\n")
        f.write(f"RMSE X: {rmse_x:.3f} mm\n")
        f.write(f"RMSE Y: {rmse_y:.3f} mm\n")
        f.write(f"RMSE Z: {rmse_z:.3f} mm\n")
        f.write(f"Final MAE (Mean Absolute Error): {mae_trans:.3f} mm\n")
        f.write(f"Final RMSE (Root Mean Squared Error): {final_rmse_trans:.3f} mm\n\n")
        f.write("[ Rotation ]\n")
        f.write(f"RMSE Roll: {rmse_r:.3f} deg\n")
        f.write(f"RMSE Pitch: {rmse_p:.3f} deg\n")
        f.write(f"RMSE Yaw: {rmse_yaw:.3f} deg\n")
        f.write(f"Final MAE (Mean Absolute Error): {mae_rot:.3f} deg\n")
        f.write(f"Final RMSE (Root Mean Squared Error): {final_rmse_rot:.3f} deg\n")

    # Save validation plot in output folder
    plot_path = os.path.join(args.output_prefix, f"validation_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if not args.headless:
        print(f"\n[INFO] Saved validation report to '{txt_path}'. MAE Trans: {mae_trans:.3f}, RMSE Trans: {final_rmse_trans:.3f}")
        print(f"[INFO] Saved validation plot to '{plot_path}'.")

    return mae_trans, final_rmse_trans

if __name__ == "__main__":
    main()
