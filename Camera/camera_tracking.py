import csv
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".venv" / "torch_cache")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from scipy.spatial.transform import Rotation


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASET_NAME = "horizontal_line_1"
SKIP_INITIAL_FRAMES = 30

DEVICE = "cuda"
MAX_KEYPOINTS = 2048
MIN_MATCHES = 30
MIN_INLIERS = 20
PNP_REPROJECTION_ERROR_PX = 3.0
KEYFRAME_INTERVAL = 30
ARUCO_ID = 7
ARUCO_SIZE_MM = 20.0
MAP_DISPLAY_GRID_MM = 1.0

SAVE_DIAGNOSTIC_VIDEO = True
DIAGNOSTIC_VIDEO_FPS = 1.0
SHOW_PREVIEW = False


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results"

CAMERA_MATRIX_PATH = (
    SCRIPT_DIR
    / "calibrations"
    / "camera_jabra_640_360"
    / "camera_matrix.npy"
)
DISTORTION_PATH = (
    SCRIPT_DIR
    / "calibrations"
    / "camera_jabra_640_360"
    / "dist_coeffs.npy"
)

DATASETS = {
    "horizontal_line_1": {
        "video": "Data/horizontal_10x_5sp__x-005/horizontal_10x_5sp__x/probe_camera/video.mp4",
        "gt": "Data/horizontal_10x_5sp__x-005/horizontal_10x_5sp__x/ground_truth/position.csv",
    },
    "horizontal_line_2": {
        "video": "Data/horizontal_10x_8sp__x-003/horizontal_10x_8sp__x/probe_camera/video.mp4",
        "gt": "Data/horizontal_10x_8sp__x-003/horizontal_10x_8sp__x/ground_truth/position.csv",
    },
    "vertical_line_1": {
        "video": "Data/vertical_10x_5sp__y-001/vertical_10x_5sp__y/probe_camera/video.mp4",
        "gt": "Data/vertical_10x_5sp__y-001/vertical_10x_5sp__y/ground_truth/position.csv",
    },
    "vertical_line_2": {
        "video": "Data/vertical_10x_8sp__y-006/vertical_10x_8sp__y/probe_camera/video.mp4",
        "gt": "Data/vertical_10x_8sp__y-006/vertical_10x_8sp__y/ground_truth/position.csv",
    },
    "square_1": {
        "video": "Data/square_4x_5sp__x,y-002/square_4x_5sp__x,y/probe_camera/video.mp4",
        "gt": "Data/square_4x_5sp__x,y-002/square_4x_5sp__x,y/ground_truth/position.csv",
    },
    "square_2": {
        "video": "Data/square_4x_8sp__x,y/square_4x_8sp__x,y/probe_camera/video.mp4",
        "gt": "Data/square_4x_8sp__x,y/square_4x_8sp__x,y/ground_truth/position.csv",
    },
    "triangle_1": {
        "video": "Data/triangle_4x_5sp__x,y/probe_camera/video.mp4",
        "gt": "Data/triangle_4x_5sp__x,y/ground_truth/position.csv",
    },
    "triangle_2": {
        "video": "Data/triangle_4x_8sp__x,y/probe_camera/video.mp4",
        "gt": "Data/triangle_4x_8sp__x,y/ground_truth/position.csv",
    },
    "cross_1": {
        "video": "Data/cross_4x_5sp__x,y/probe_camera/video.mp4",
        "gt": "Data/cross_4x_5sp__x,y/ground_truth/position.csv",
    },
    "cross_2": {
        "video": "Data/cross_4x_8sp__x,y/probe_camera/video.mp4",
        "gt": "Data/cross_4x_8sp__x,y/ground_truth/position.csv",
    },
    "rotation_1": {
        "video": "Data/roration_4x_5sp__pitch/probe_camera/video.mp4",
        "gt": "Data/roration_4x_5sp__pitch/ground_truth/position.csv",
    },
    "rotation_2": {
        "video": "Data/roration_4x_8sp__pitch/roration_4x_8sp__pitch/probe_camera/video.mp4",
        "gt": "Data/roration_4x_8sp__pitch/roration_4x_8sp__pitch/ground_truth/position.csv",
    },
}


def frame_to_tensor(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return image.to(DEVICE)


def camera_pose(R_map_to_camera, t_map_to_camera):
    R_camera_to_map = R_map_to_camera.T
    position = -R_camera_to_map @ t_map_to_camera.reshape(3)
    euler = Rotation.from_matrix(R_camera_to_map).as_euler("xyz", degrees=True)
    return position, euler


class SkinMapTracker:
    def __init__(self, camera_matrix, distortion):
        self.camera_matrix = camera_matrix
        self.distortion = distortion

        self.extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        self.matcher = LightGlue(features="superpoint").eval().to(DEVICE)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary)

        self.keyframes = []
        self.last_keyframe_frame = -KEYFRAME_INTERVAL
        self.R_map_to_camera = None
        self.t_map_to_camera = None

    def extract_features(self, frame):
        with torch.inference_mode():
            return self.extractor.extract(frame_to_tensor(frame))

    def find_initial_pose(self, frame):
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        if ids is None or ARUCO_ID not in ids.flatten():
            return None

        marker_index = np.where(ids.flatten() == ARUCO_ID)[0][0]
        image_points = corners[marker_index].reshape(4, 2).astype(np.float64)

        half = ARUCO_SIZE_MM / 2.0
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None

        R_map_to_camera = cv2.Rodrigues(rvec)[0]
        return R_map_to_camera, tvec.reshape(3)

    def pixels_to_skin_plane(self, keypoints, R_map_to_camera, t_map_to_camera):
        normalized = cv2.undistortPoints(
            keypoints.reshape(-1, 1, 2),
            self.camera_matrix,
            self.distortion,
        ).reshape(-1, 2)

        rays_camera = np.column_stack((normalized, np.ones(len(normalized))))
        camera_origin_map = -R_map_to_camera.T @ t_map_to_camera.reshape(3)
        rays_map = (R_map_to_camera.T @ rays_camera.T).T

        scale = -camera_origin_map[2] / rays_map[:, 2]
        map_points = camera_origin_map + scale[:, None] * rays_map
        map_points[scale <= 0.0] = np.nan
        map_points[:, 2] = 0.0
        return map_points

    def add_keyframe(self, frame_index, features, R_map_to_camera, t_map_to_camera):
        keypoints = rbd(features)["keypoints"].detach().cpu().numpy()
        map_points = self.pixels_to_skin_plane(
            keypoints,
            R_map_to_camera,
            t_map_to_camera,
        )
        self.keyframes.append(
            {
                "frame": frame_index,
                "features": features,
                "map_points": map_points,
            }
        )
        self.last_keyframe_frame = frame_index

    def match_keyframe(self, keyframe, current_features):
        with torch.inference_mode():
            output = self.matcher(
                {"image0": keyframe["features"], "image1": current_features}
            )

        matches = rbd(output)["matches"].detach().cpu().numpy()
        current_keypoints = rbd(current_features)["keypoints"].detach().cpu().numpy()

        if len(matches) < MIN_MATCHES:
            return None

        map_points = keyframe["map_points"][matches[:, 0]]
        image_points = current_keypoints[matches[:, 1]]
        valid = np.isfinite(map_points).all(axis=1)
        map_points = np.ascontiguousarray(map_points[valid], dtype=np.float64)
        image_points = np.ascontiguousarray(image_points[valid], dtype=np.float64)

        if len(map_points) < MIN_MATCHES:
            return None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            map_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            iterationsCount=200,
            reprojectionError=PNP_REPROJECTION_ERROR_PX,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < MIN_INLIERS:
            return None

        inlier_indices = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(
            map_points[inlier_indices],
            image_points[inlier_indices],
            self.camera_matrix,
            self.distortion,
            rvec,
            tvec,
        )

        R_map_to_camera = cv2.Rodrigues(rvec)[0]
        inlier_mask = np.zeros(len(image_points), dtype=bool)
        inlier_mask[inlier_indices] = True
        return {
            "R": R_map_to_camera,
            "t": tvec.reshape(3),
            "matches": len(map_points),
            "inliers": len(inlier_indices),
            "inlier_map_points": map_points[inlier_indices],
            "outlier_points": image_points[~inlier_mask],
            "keyframe_frame": keyframe["frame"],
        }

    def track(self, frame_index, frame):
        features = self.extract_features(frame)

        if not self.keyframes:
            initial_pose = self.find_initial_pose(frame)
            if initial_pose is None:
                return None

            self.R_map_to_camera, self.t_map_to_camera = initial_pose
            self.add_keyframe(
                frame_index,
                features,
                self.R_map_to_camera,
                self.t_map_to_camera,
            )
            return {
                "R": self.R_map_to_camera,
                "t": self.t_map_to_camera,
                "matches": 0,
                "inliers": 0,
                "inlier_map_points": np.empty((0, 3)),
                "outlier_points": np.empty((0, 2)),
                "keyframe_frame": frame_index,
            }

        result = self.match_keyframe(self.keyframes[-1], features)

        if result is None:
            for keyframe in reversed(self.keyframes[:-1]):
                result = self.match_keyframe(keyframe, features)
                if result is not None:
                    break

        if result is None:
            return None

        self.R_map_to_camera = result["R"]
        self.t_map_to_camera = result["t"]

        if frame_index - self.last_keyframe_frame >= KEYFRAME_INTERVAL:
            self.add_keyframe(
                frame_index,
                features,
                self.R_map_to_camera,
                self.t_map_to_camera,
            )

        return result


def project_map_points(map_points, result, tracker, frame_shape):
    if len(map_points) == 0:
        return np.empty((0, 2))

    rvec = cv2.Rodrigues(result["R"])[0]
    projected, _ = cv2.projectPoints(
        map_points,
        rvec,
        result["t"],
        tracker.camera_matrix,
        tracker.distortion,
    )
    projected = projected.reshape(-1, 2)

    camera_points = (result["R"] @ map_points.T).T + result["t"]
    height, width = frame_shape[:2]
    visible = camera_points[:, 2] > 0.0
    visible &= projected[:, 0] >= 0.0
    visible &= projected[:, 0] < width
    visible &= projected[:, 1] >= 0.0
    visible &= projected[:, 1] < height
    return projected[visible]


def map_points_for_display(tracker):
    map_points = np.vstack(
        [keyframe["map_points"] for keyframe in tracker.keyframes]
    )
    map_points = map_points[np.isfinite(map_points).all(axis=1)]
    grid_cells = np.rint(
        map_points[:, :2] / MAP_DISPLAY_GRID_MM
    ).astype(np.int32)
    _, unique_indices = np.unique(grid_cells, axis=0, return_index=True)
    return map_points[unique_indices]


def diagnostic_frame(frame, tracker, result, relative_positions):
    output = frame.copy()
    tracked = result is not None
    color = (40, 200, 40) if tracked else (30, 30, 220)
    label = "TRACKING" if tracked else "LOST"

    if tracked:
        projected_map_points = project_map_points(
            map_points_for_display(tracker),
            result,
            tracker,
            frame.shape,
        )
        projected_inliers = project_map_points(
            result["inlier_map_points"],
            result,
            tracker,
            frame.shape,
        )

        for point in projected_map_points:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                1,
                (0, 255, 255),
                -1,
            )

        for point in projected_inliers:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 255, 0),
                -1,
            )

        for point in result["outlier_points"]:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 0, 255),
                -1,
            )

        position = relative_positions[-1]
        cv2.putText(
            output,
            f"XYZ: {position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f} mm",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.rectangle(output, (8, 84), (270, 158), (0, 0, 0), -1)
        legend = [
            (
                f"Map not detected: {len(projected_map_points) - len(projected_inliers)}",
                (0, 255, 255),
            ),
            (f"PnP inliers: {result['inliers']}", (0, 255, 0)),
            (f"PnP outliers: {len(result['outlier_points'])}", (0, 0, 255)),
        ]
        for index, (text, text_color) in enumerate(legend):
            cv2.putText(
                output,
                text,
                (15, 105 + 23 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2,
            )

    cv2.putText(output, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        output,
        f"keyframes: {len(tracker.keyframes)}",
        (12, output.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return output


def save_results_csv(rows, path):
    fields = [
        "frame",
        "time_s",
        "x_mm",
        "y_mm",
        "z_mm",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "matches",
        "inliers",
        "keyframes",
        "tracked",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_ground_truth(path):
    frames = []
    positions = []
    orientations = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            values = [
                row["X"],
                row["Y"],
                row["Z"],
                row["Roll"],
                row["Pitch"],
                row["Yaw"],
            ]
            if not all(values):
                continue
            frames.append(int(row["Frame"]))
            positions.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
            orientations.append(
                [float(row["Roll"]), float(row["Pitch"]), float(row["Yaw"])]
            )
    return np.array(frames), np.array(positions), np.array(orientations)


def create_comparison_plot(rows, gt_path, output_path):
    frames = np.array([row["frame"] for row in rows])
    estimate = np.array(
        [[row["x_mm"], row["y_mm"], row["z_mm"]] for row in rows],
        dtype=float,
    )
    estimate_euler = np.array(
        [[row["roll_deg"], row["pitch_deg"], row["yaw_deg"]] for row in rows],
        dtype=float,
    )
    gt_frames, gt_positions, gt_euler = load_ground_truth(gt_path)

    gt = np.column_stack(
        [np.interp(frames, gt_frames, gt_positions[:, axis]) for axis in range(3)]
    )
    gt_euler = np.degrees(np.unwrap(np.radians(gt_euler), axis=0))
    interpolated_gt_euler = np.column_stack(
        [np.interp(frames, gt_frames, gt_euler[:, axis]) for axis in range(3)]
    )
    valid = np.isfinite(estimate).all(axis=1)
    valid &= np.isfinite(estimate_euler).all(axis=1)
    valid &= frames >= gt_frames.min()
    valid &= frames <= gt_frames.max()

    first_valid = np.flatnonzero(valid)[0]
    relative_estimate = estimate - estimate[first_valid]
    relative_gt = gt - gt[first_valid]
    errors = np.linalg.norm(
        relative_estimate[valid] - relative_gt[valid],
        axis=1,
    )
    position_rmse = np.sqrt(np.mean(errors**2))

    absolute_estimate_rotations = Rotation.from_euler(
        "xyz", estimate_euler[valid], degrees=True
    ).as_matrix()
    absolute_gt_rotations = Rotation.from_euler(
        "xyz", interpolated_gt_euler[valid], degrees=True
    ).as_matrix()

    estimate_rotations = (
        absolute_estimate_rotations[0].T @ absolute_estimate_rotations
    )
    gt_rotations = absolute_gt_rotations[0].T @ absolute_gt_rotations
    relative_estimate_euler = Rotation.from_matrix(estimate_rotations).as_euler(
        "xyz", degrees=True
    )
    relative_estimate_euler = np.degrees(
        np.unwrap(np.radians(relative_estimate_euler), axis=0)
    )
    relative_gt_euler = Rotation.from_matrix(gt_rotations).as_euler(
        "xyz", degrees=True
    )
    relative_gt_euler = np.degrees(
        np.unwrap(np.radians(relative_gt_euler), axis=0)
    )
    relative_rotations = np.transpose(gt_rotations, (0, 2, 1)) @ estimate_rotations
    angular_errors = np.degrees(
        Rotation.from_matrix(relative_rotations).magnitude()
    )

    position_mae = np.mean(errors)
    orientation_mae = np.mean(angular_errors)
    orientation_rmse = np.sqrt(np.mean(angular_errors**2))

    position_columns = ["X [mm]", "Y [mm]", "Z [mm]"]
    orientation_columns = ["Roll [deg]", "Pitch [deg]", "Yaw [deg]"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
    for axis in range(3):
        position_axis = axes[axis, 0]
        position_axis.plot(
            frames[valid],
            relative_gt[valid, axis],
            "k--",
            label="GT",
        )
        position_axis.plot(
            frames[valid],
            relative_estimate[valid, axis],
            color="tab:blue",
            label="Camera",
        )
        position_axis.set_ylabel(position_columns[axis])
        position_axis.grid(True)
        position_axis.legend()

        orientation_axis = axes[axis, 1]
        orientation_axis.plot(
            frames[valid],
            relative_gt_euler[:, axis],
            "k--",
            label="GT",
        )
        orientation_axis.plot(
            frames[valid],
            relative_estimate_euler[:, axis],
            color="tab:blue",
            label="Camera",
        )
        orientation_axis.set_ylabel(orientation_columns[axis])
        orientation_axis.grid(True)
        orientation_axis.legend()

    distance_axis = axes[3, 0]
    distance_axis.plot(frames[valid], errors, color="red")
    distance_axis.set_title(
        f"Euclidean Distance | MAE: {position_mae:.2f} mm | "
        f"RMSE: {position_rmse:.2f} mm"
    )
    distance_axis.set_ylabel("Distance [mm]")
    distance_axis.set_xlabel("Frame")
    distance_axis.grid(True)

    angular_axis = axes[3, 1]
    angular_axis.plot(frames[valid], angular_errors, color="red")
    angular_axis.set_title(
        f"Angular Distance | MAE: {orientation_mae:.2f} deg | "
        f"RMSE: {orientation_rmse:.2f} deg"
    )
    angular_axis.set_ylabel("Angle [deg]")
    angular_axis.set_xlabel("Frame")
    angular_axis.grid(True)

    fig.suptitle(
        f"{DATASET_NAME}: Camera vs GT, position RMSE = {position_rmse:.2f} mm, "
        f"orientation RMSE = {orientation_rmse:.2f} deg\n"
        "Direct comparison; both trajectories start at their first measurement"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return position_rmse, orientation_rmse


def main():
    if DATASET_NAME not in DATASETS:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")
    if not torch.cuda.is_available() and DEVICE == "cuda":
        raise RuntimeError("CUDA is not available in the project .venv")

    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset = DATASETS[DATASET_NAME]
    video_path = SCRIPT_DIR / dataset["video"]
    gt_path = SCRIPT_DIR / dataset["gt"]

    camera_matrix = np.load(CAMERA_MATRIX_PATH)
    distortion = np.load(DISTORTION_PATH)
    tracker = SkinMapTracker(camera_matrix, distortion)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)

    input_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.set(cv2.CAP_PROP_POS_FRAMES, SKIP_INITIAL_FRAMES)

    video_writer = None
    if SAVE_DIAGNOSTIC_VIDEO:
        video_path_output = OUTPUT_DIR / f"{DATASET_NAME}_tracking.mp4"
        video_writer = cv2.VideoWriter(
            str(video_path_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            DIAGNOSTIC_VIDEO_FPS,
            (width, height),
        )

    rows = []
    positions = []
    initial_position = None
    initial_rotation = None
    frame_index = SKIP_INITIAL_FRAMES

    while True:
        success, frame = capture.read()
        if not success:
            break

        result = tracker.track(frame_index, frame)
        if result is None:
            position = np.full(3, np.nan)
            euler = np.full(3, np.nan)
            matches = 0
            inliers = 0
        else:
            absolute_position, _ = camera_pose(result["R"], result["t"])
            camera_rotation = result["R"].T

            if initial_position is None:
                initial_position = absolute_position.copy()
                initial_rotation = camera_rotation.copy()

            position = absolute_position - initial_position
            relative_rotation = initial_rotation.T @ camera_rotation
            euler = Rotation.from_matrix(relative_rotation).as_euler(
                "xyz", degrees=True
            )
            matches = result["matches"]
            inliers = result["inliers"]

        positions.append(position)
        rows.append(
            {
                "frame": frame_index,
                "time_s": frame_index / input_fps,
                "x_mm": position[0],
                "y_mm": position[1],
                "z_mm": position[2],
                "roll_deg": euler[0],
                "pitch_deg": euler[1],
                "yaw_deg": euler[2],
                "matches": matches,
                "inliers": inliers,
                "keyframes": len(tracker.keyframes),
                "tracked": int(result is not None),
            }
        )

        if SAVE_DIAGNOSTIC_VIDEO or SHOW_PREVIEW:
            preview = diagnostic_frame(
                frame,
                tracker,
                result,
                positions,
            )
            if video_writer is not None:
                video_writer.write(preview)
            if SHOW_PREVIEW:
                cv2.imshow("Camera skin tracking", preview)
                if cv2.waitKey(1) == 27:
                    break

        if frame_index % 100 == 0:
            print(f"Frame {frame_index}/{frame_count}, keyframes: {len(tracker.keyframes)}")
        frame_index += 1

    capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    csv_path = OUTPUT_DIR / f"{DATASET_NAME}_camera.csv"
    plot_path = OUTPUT_DIR / f"{DATASET_NAME}_camera_vs_gt.png"
    save_results_csv(rows, csv_path)
    position_rmse, orientation_rmse = create_comparison_plot(rows, gt_path, plot_path)

    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")
    if SAVE_DIAGNOSTIC_VIDEO:
        print(f"Saved: {video_path_output}")
    print(f"Position RMSE: {position_rmse:.2f} mm")
    print(f"Orientation RMSE: {orientation_rmse:.2f} deg")


if __name__ == "__main__":
    main()
