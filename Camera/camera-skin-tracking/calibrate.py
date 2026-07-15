import cv2
import numpy as np

# --- chessboard parameters ---
chessboard_size = (9, 6)  # number of internal corners (not squares!)
square_size = 0.013       # size of square in meters (e.g. 13 mm)

objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points on image

# --- camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cv2.namedWindow("Chessboard", cv2.WINDOW_NORMAL)

real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"[WARNING] Actual camera resolution: {int(real_w)}x{int(real_h)}")

counter = 0
print("Click on the 'Chessboard' window and use keys: 's' = save frame, 'q' = exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to retrieve frame from camera")
        break

    cv2.imshow("Chessboard", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("[INFO] Looking for chessboard on this frame (this might cause a brief pause)...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret_cb:
            objpoints.append(objp)
            imgpoints.append(corners)
            counter += 1
            
            frame_with_corners = frame.copy()
            cv2.drawChessboardCorners(frame_with_corners, chessboard_size, corners, ret_cb)
            cv2.imshow("Chessboard", frame_with_corners)
            cv2.waitKey(500)
            
            print(f"[INFO] Saved image #{counter}")
        else:
            print("[ERROR] Chessboard not found! Make sure it is clearly visible.")
            
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(objpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    import os
    save_dir = "calibrations/camera_jabra_1920_1080"
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "camera_matrix.npy"), camera_matrix)
    np.save(os.path.join(save_dir, "dist_coeffs.npy"), dist_coeffs)

    print(f"[INFO] Calibration completed and saved in {save_dir}/")
    print("[INFO] Calibration completed")
    print("camera_matrix:\n", camera_matrix)
    print("dist_coeffs:\n", dist_coeffs)
else:
    print("[WARNING] No chessboard frames saved – calibration skipped.")
