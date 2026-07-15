import cv2
import numpy as np
import os

def calibrate_from_video():
    
    video_path = "..."

    # --- chessboard parameters ---
    chessboard_size = (9, 6)      # number of internal corners
    square_size = 0.013           # size of square in meters

    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video file")
        return

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"\n[INFO] Video opened successfully: {video_path}")
    print(f"[INFO] Video resolution: {int(w)}x{int(h)}")

    print("\nINSTRUCTIONS:")
    print("  S      → save frame")
    print("  SPACE  → pause / resume")
    print("  D      → skip 10 frames")
    print("  Q      → finish collection\n")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

    paused = False
    frame = None

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of recording.")
                break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret_cb, corners = cv2.findChessboardCorners(
            gray,
            chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )

        display = frame.copy()

        if ret_cb:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            cv2.drawChessboardCorners(display, chessboard_size, corners, ret_cb)

            cv2.putText(display,
                        f"Saved: {len(objpoints)} | S = save",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)
        else:
            cv2.putText(display,
                        "Searching for chessboard...",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and ret_cb:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"[INFO] Saved frame #{len(objpoints)}")

        elif key == ord(" "):
            paused = not paused
            print("⏸ Paused" if paused else "▶️ Resumed")

        elif key == ord("d"):
            skipped = 0
            for _ in range(10):
                ret, _ = cap.read()
                if not ret:
                    break
                skipped += 1
            print(f"[INFO] Skipped {skipped} frames")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        print(f"\n[ERROR] Too few frames ({len(objpoints)}). Minimum required is 10-15.")
        return

    print(f"\n[INFO] Computing calibration for {len(objpoints)} frames...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    if not ret:
        print("[ERROR] Calibration failed")
        return

    np.save("camera_matrix.npy", mtx)
    np.save("dist_coeffs.npy", dist)

    print("\n[INFO] Calibration completed")
    print("Saved:")
    print("  camera_matrix.npy")
    print("  dist_coeffs.npy")

    print("\nCamera matrix:")
    print(mtx)

    total_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )

        error = cv2.norm(
            imgpoints[i],
            imgpoints2,
            cv2.NORM_L2
        ) / len(imgpoints2)

        total_error += error

    mean_error = total_error / len(objpoints)

    print(f"\n[INFO] Mean reprojection error: {mean_error:.4f}")

    if mean_error < 0.1:
        print("⭐ Excellent calibration")
    elif mean_error < 0.3:
        print("👍 Very good calibration")
    elif mean_error < 0.5:
        print("✔ Acceptable calibration")
    else:
        print("⚠ Poor calibration – collect more frames")

if __name__ == "__main__":
    calibrate_from_video()