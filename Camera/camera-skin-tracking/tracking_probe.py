import cv2
import numpy as np
import os
import math
import argparse


def estimate_marker_poses(corners, marker_size, camera_matrix, dist_coeffs):
    half = marker_size / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )

    rvecs = []
    tvecs = []
    for marker_corners in corners:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners.reshape(4, 2),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            raise RuntimeError("Could not estimate ArUco marker pose")
        rvecs.append(rvec.reshape(1, 3))
        tvecs.append(tvec.reshape(1, 3))

    return np.array(rvecs), np.array(tvecs)


class UltrasoundProbeTracker6DoF:
    def __init__(self, camera_matrix, dist_coeffs, homography_method=cv2.RANSAC, ransac_threshold=3.0):
        self.K = camera_matrix
        self.D = dist_coeffs
        self.focal_length = (self.K[0, 0] + self.K[1, 1]) / 2
        
        # Homography Estimation Options
        self.homography_method = homography_method
        self.ransac_threshold = ransac_threshold
        
        # --- CONFIGURATION ---
        self.ARUCO_SIZE_MM = 20.0 
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.aruco_params,
        )
        
        # Mask: ignore dark regions (noise) and edges (distortion)
        self.BRIGHTNESS_THRESHOLD = 40 
        self.EDGE_MARGIN = 40
        self.TRAJ_SCALE = 30.0 
        
        self.feature_params = dict(maxCorners=1500, qualityLevel=0.015, minDistance=12, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

        # --- SYSTEM STATE ---
        self.is_calibrated = False
        
        # Global camera position and rotation

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        
        # Offset from camera matrix to physical ultrasound probe tip [X, Y, Z] (in millimeters)
        # Measured using calipers or from CAD: e.g. [[0.0], [45.0], [10.0]]
        self.probe_offset = np.array([[0.0], [0.0], [0.0]])

        self.current_distance_mm = 0.0
        
        self.prev_gray = None
        self.prev_pts = None
        
        self.traj_img = np.zeros((800, 800, 3), dtype=np.uint8)
        self.reset_system()

    @property
    def probe_tip_t(self):
        """Returns the position of the physical probe tip. 
        X and Y start from 0.0, Z is the absolute distance to the skin.
        X and Y is only additional offset caused by rotation comparing to initial offset.
        """
        tip = self.cur_t.copy()
        
        rotated_offset = self.cur_R.dot(self.probe_offset)
        
        tip[0] += rotated_offset[0] - self.probe_offset[0]
        tip[1] += rotated_offset[1] - self.probe_offset[1]
        
        tip[2] -= rotated_offset[2]
        
        return tip

    def reset_system(self):
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        if hasattr(self, 'initial_depth_mm'):
            self.cur_t[2] = self.initial_depth_mm
            self.current_distance_mm = self.initial_depth_mm
        
        self.traj_img = np.zeros((800, 800, 3), dtype=np.uint8)
        cv2.line(self.traj_img, (0, 400), (800, 400), (40, 40, 40), 1)
        cv2.line(self.traj_img, (400, 0), (400, 800), (40, 40, 40), 1)
        print("[INFO] COORDINATE SYSTEM RESET (6DoF)")

    def rotation_matrix_to_euler(self, R):
        """Returns Euler angles in degrees: Roll, Pitch, Yaw."""
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
        return np.degrees(x), np.degrees(y), np.degrees(z)

    def get_usable_mask(self, gray_frame):
        h, w = gray_frame.shape
        _, mask_bright = cv2.threshold(gray_frame, self.BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask_border = np.zeros_like(gray_frame)
        cv2.rectangle(mask_border, (self.EDGE_MARGIN, self.EDGE_MARGIN), 
                      (w - self.EDGE_MARGIN, h - self.EDGE_MARGIN), 255, -1)
        return cv2.bitwise_and(mask_bright, mask_border)

    def try_initialize_with_aruco(self, frame, frame_gray):
        corners, ids, _ = self.aruco_detector.detectMarkers(frame_gray)
        
        if ids is not None:
            idx_list = np.where(ids == 7)[0]
            if len(idx_list) > 0:
                marker_idx = idx_list[0]
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs = estimate_marker_poses(
                    corners,
                    self.ARUCO_SIZE_MM,
                    self.K,
                    self.D,
                )
                z_dist_mm = tvecs[marker_idx][0][2]
                
                if z_dist_mm > 10:
                    self.current_distance_mm = z_dist_mm
                    self.initial_depth_mm = z_dist_mm
                    
                    self.cur_t[2] = z_dist_mm
                    
                    self.is_calibrated = True
                    print(f"[INFO] 6DoF CALIBRATION SUCCESSFUL! Starting height: {z_dist_mm:.1f}mm (ID=7)")
                    
                    mask = self.get_usable_mask(frame_gray)
                    self.prev_gray = frame_gray
                    self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
                    
                    return True, frame
            
        cv2.putText(frame, "SHOW ARUCO MARKER ID=7 (2cm)", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, "Mode: Homography 6DoF", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return False, frame

    def track_motion_homography(self, frame, frame_gray):
        if self.prev_pts is None or len(self.prev_pts) < 10:
            mask = self.get_usable_mask(frame_gray)
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params) # Does not make any sense. Find pts in current frame and save always
            return frame

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_pts, None, **self.lk_params)

        if curr_pts is not None:
            good_new = curr_pts[status == 1]
            good_old = self.prev_pts[status == 1]
            
            mask_h = None
            
            # We need at least 4 points for homography, but more is better for stability
            if len(good_new) > 20:
                H, mask_h = cv2.findHomography(good_old, good_new, self.homography_method, self.ransac_threshold)
                
                if H is not None:
                    num_sol, Rs, Ts, normals = cv2.decomposeHomographyMat(H, self.K)
                    
                    best_idx = None
                    min_diff_normal = 999.0
                    
                    for i in range(num_sol):
                        normal = normals[i]
                        diff = np.linalg.norm(normal - np.array([[0], [0], [1]]))
                        
                        t_norm = np.linalg.norm(Ts[i])
                        
                        if diff < min_diff_normal and t_norm < 0.5:
                            min_diff_normal = diff
                            best_idx = i
                    
                    if best_idx is not None:
                        R_rel = Rs[best_idx]
                        T_rel = Ts[best_idx]
                        
                        T_real_mm = T_rel * self.current_distance_mm
                        
                        T_update = T_real_mm.copy()
                        T_update[0] = -T_update[0]
                        
                        self.cur_t = self.cur_t - self.cur_R.dot(T_update)
                        self.cur_R = R_rel.dot(self.cur_R)
                        
                        if self.cur_t[2] < 5.0: self.cur_t[2] = 5.0
                        self.current_distance_mm = self.cur_t[2][0]

            if mask_h is not None:
                valid_ransac = (mask_h.ravel() == 1)
                good_new = good_new[valid_ransac]
                good_old = good_old[valid_ransac]

            usable_mask = self.get_usable_mask(frame_gray)
            h, w = usable_mask.shape
            
            valid_pts_indices = []
            for idx, pt in enumerate(good_new):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    if usable_mask[y, x] > 0:
                        valid_pts_indices.append(idx)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            good_new = good_new[valid_pts_indices]
            good_old = good_old[valid_pts_indices]

            # Point management (adding features in empty regions)
            if len(good_new) < 1000:
                mask = self.get_usable_mask(frame_gray)
                for pt in good_new:
                    cv2.circle(mask, (int(pt[0]), int(pt[1])), 10, 0, -1)
                new_pts = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                if new_pts is not None:
                    good_new = np.vstack((good_new, new_pts.reshape(-1, 2)))
            
            self.prev_gray = frame_gray.copy()
            self.prev_pts = good_new.reshape(-1, 1, 2)
            
            self.draw_hud(frame)
            self.update_map()

        return frame

    def update_map(self):
        cx, cy = 400, 400
        tip = self.probe_tip_t
        x_mm = tip[0, 0]
        y_mm = tip[1, 0]
        
        draw_x = cx + int(x_mm * self.TRAJ_SCALE)
        draw_y = cy + int(y_mm * self.TRAJ_SCALE)
        
        if 0 <= draw_x < 800 and 0 <= draw_y < 800:
            # Path
            cv2.circle(self.traj_img, (draw_x, draw_y), 2, (0, 0, 200), -1)
            
            # Cursor
            temp = self.traj_img.copy()
            cv2.circle(temp, (draw_x, draw_y), 6, (0, 255, 0), -1)
            
            # Orientation Visualization (YAW Arrow)
            r, p, yaw = self.rotation_matrix_to_euler(self.cur_R)
            rad = np.radians(yaw)
            ex = int(draw_x + 25 * math.cos(rad))
            ey = int(draw_y + 25 * math.sin(rad))
            cv2.line(temp, (draw_x, draw_y), (ex, ey), (255, 255, 0), 2)
            
            # Tilt Visualization (PITCH/ROLL) as ellipse
            # Larger tilt results in a narrower ellipse (simplified visualization)
            tilt_factor = max(0.2, 1.0 - (abs(r) + abs(p))/90.0)
            axes = (20, int(20 * tilt_factor))
            cv2.ellipse(temp, (draw_x, draw_y), axes, yaw, 0, 360, (0, 255, 255), 1)

            cv2.imshow('Map (Top View)', temp)

    def draw_hud(self, frame):
        # Background
        cv2.rectangle(frame, (10, 10), (300, 160), (0,0,0), -1)
        
        # Position
        tip = self.probe_tip_t
        cv2.putText(frame, "TIP POSITION [mm]", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (200,200,200), 1)
        cv2.putText(frame, f"X: {tip[0][0]:.1f}", (20, 55), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Y: {tip[1][0]:.1f}", (150, 55), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Z: {tip[2][0]:.1f}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 150, 255), 2)
        
        # Rotation
        r, p, y = self.rotation_matrix_to_euler(self.cur_R)
        cv2.putText(frame, "ROTATION [deg]", (20, 110), cv2.FONT_HERSHEY_PLAIN, 1.0, (200,200,200), 1)
        cv2.putText(frame, f"R: {r:.0f}", (20, 135), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
        cv2.putText(frame, f"P: {p:.0f}", (100, 135), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
        cv2.putText(frame, f"Y: {y:.0f}", (180, 135), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)

        # Mini mask preview
        mask = self.get_usable_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        mask_mini = cv2.cvtColor(cv2.resize(mask, (120, 90)), cv2.COLOR_GRAY2BGR)
        h, w, _ = frame.shape
        frame[h-90:h, w-120:w] = mask_mini
        cv2.rectangle(frame, (w-120, h-90), (w, h), (0,255,0), 1)

    def preprocess_frame(self, frame):
        frame_undistorted = cv2.undistort(frame, self.K, self.D)
        frame_gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        frame_gray = self.clahe.apply(frame_gray)
        return frame_undistorted, frame_gray

    def initialize_from_aruco(self, frame):
        frame_undistorted, frame_gray = self.preprocess_frame(frame)
        return self.try_initialize_with_aruco(frame_undistorted, frame_gray)[1]

    def track(self, frame):
        frame_undistorted, frame_gray = self.preprocess_frame(frame)
        return self.track_motion_homography(frame_undistorted, frame_gray)
        
    def process(self, frame):
        if not self.is_calibrated:
            return self.initialize_from_aruco(frame)
        return self.track(frame)

def main():
    parser = argparse.ArgumentParser(description="Ultrasound Probe Tracker 6DoF")
    parser.add_argument("--video", type=str, help="Path to the .mp4 video file", default=None)
    parser.add_argument("--camera", type=int, help="Camera index (default: 1)", default=1)
    parser.add_argument("--save_video", type=str, help="Path to save the output video (.mp4)", default=None)
    parser.add_argument("--save_csv", type=str, help="Path to save coordinates (.csv)", default=None)
    args = parser.parse_args()

    calib_dir = "calibrations/camera_jabra_640_360"
    k_path = os.path.join(calib_dir, "camera_matrix.npy")
    d_path = os.path.join(calib_dir, "dist_coeffs.npy")

    if not os.path.exists(k_path) or not os.path.exists(d_path):
        print(f"No calibration found in folder {calib_dir}!")
        return
    K = np.load(k_path)
    D = np.load(d_path)
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"Video file does not exist: {args.video}")
            return
        cap = cv2.VideoCapture(args.video)
        print(f"Opening video file: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera)
        print(f"Opening camera with index: {args.camera}")

    # Source settings logging
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Initializing source: {int(width)}x{int(height)} @ {fps} FPS")
    
    delay = int(1000 / fps) if args.video and fps > 0 else 1

    tracker = UltrasoundProbeTracker6DoF(K, D, homography_method=cv2.USAC_ACCURATE, ransac_threshold=3.0)

    # Initialize video writing
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps, (int(width), int(height)))
        print(f"Saving video to: {args.save_video}")

    # Initialize CSV writing
    csv_file = None
    if args.save_csv:
        csv_file = open(args.save_csv, 'w')
        csv_file.write("frame,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg\n")
        print(f"Saving coordinates to: {args.save_csv}")

    def handle_output(out, frame_idx):
        # Write video frame
        if video_writer is not None:
            video_writer.write(out)

        # Write CSV row
        if csv_file is not None and tracker.is_calibrated:
            tip = tracker.probe_tip_t
            r, p, y = tracker.rotation_matrix_to_euler(tracker.cur_R)
            csv_file.write(f"{frame_idx},{tip[0][0]:.3f},{tip[1][0]:.3f},{tip[2][0]:.3f},{r:.3f},{p:.3f},{y:.3f}\n")

        cv2.imshow('6DoF Tracking', out)
        return cv2.waitKey(delay)

    frame_idx = 0

    print("[INFO] Waiting for ArUco marker ID=7 to initialize tracking...")
    while not tracker.is_calibrated:
        ret, frame = cap.read()
        if not ret:
            break

        out = tracker.initialize_from_aruco(frame)

        k = handle_output(out, frame_idx)
        if k == ord('q'): break
        if k == ord('r'): tracker.reset_system()
        frame_idx += 1

    while tracker.is_calibrated:
        ret, frame = cap.read()
        if not ret:
            break

        out = tracker.track(frame)

        k = handle_output(out, frame_idx)
        if k == ord('q'): break
        if k == ord('r'): tracker.reset_system()
        frame_idx += 1

    if video_writer is not None:
        video_writer.release()
    if csv_file is not None:
        csv_file.close()
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
