import csv
import json
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation


# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_FOLDER = "Line"
RECORDING_NAME = "far-white-withlight_Speed-3_2026-07-28_17.08.22"
CAMERA_NAME = "cam1"
CAMERA_CALIBRATION = "camera_jabra_640_360"

VIDEO_PATH = (
    PROJECT_DIR
    / "Data"
    / DATA_FOLDER
    / "videos"
    / f"{RECORDING_NAME}_{CAMERA_NAME}.webm"
)
OUTPUT_DIR = SCRIPT_DIR / "surface_models" / RECORDING_NAME

CAMERA_MATRIX_PATH = (
    SCRIPT_DIR
    / "calibrations"
    / CAMERA_CALIBRATION
    / "camera_matrix.npy"
)
DISTORTION_PATH = (
    SCRIPT_DIR
    / "calibrations"
    / CAMERA_CALIBRATION
    / "dist_coeffs.npy"
)


# -----------------------------------------------------------------------------
# Video and SfM
# -----------------------------------------------------------------------------

FRAME_STEP = 3
MAX_EXTRACTED_FRAMES = 600
SEQUENTIAL_MATCH_OVERLAP = 15


# -----------------------------------------------------------------------------
# Metric reference frame
# -----------------------------------------------------------------------------

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
ARUCO_ID = 7
ARUCO_SIZE_MM = 20.0
MIN_ARUCO_ALIGNMENT_FRAMES = 3


# -----------------------------------------------------------------------------
# Point filtering and surface model
# -----------------------------------------------------------------------------

MIN_POINT_TRACK_LENGTH = 3
MAX_POINT_REPROJECTION_ERROR_PX = 2.0
SURFACE_Z_MIN_MM = -40.0
SURFACE_Z_MAX_MM = 40.0
ARUCO_POINT_EXCLUSION_MARGIN_MM = 2.0

POLYNOMIAL_DEGREE = 3
POLYNOMIAL_OUTLIER_SIGMA = 3.0
SURFACE_GRID_SIZE = 100
SHOW_SURFACE_PLOT = False


def find_video():
    if VIDEO_PATH.exists():
        return VIDEO_PATH
    return next(
        (PROJECT_DIR / "Data" / DATA_FOLDER / "videos").glob(
            f"{RECORDING_NAME}_{CAMERA_NAME}.*"
        )
    )


def aruco_object_points():
    half = ARUCO_SIZE_MM / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def detect_aruco_pose(frame, detector, camera_matrix, distortion):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or ARUCO_ID not in ids.flatten():
        return None

    marker_index = np.where(ids.flatten() == ARUCO_ID)[0][0]
    image_points = corners[marker_index].reshape(4, 2).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        aruco_object_points(),
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None

    R_aruco_to_camera = cv2.Rodrigues(rvec)[0]
    t_aruco_to_camera = tvec.reshape(3)
    return R_aruco_to_camera, t_aruco_to_camera


def extract_video_frames(
    video_path,
    images_dir,
    camera_matrix,
    distortion,
):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary)
    capture = cv2.VideoCapture(str(video_path))

    aruco_poses = {}
    frame_index = 0
    extracted_count = 0

    while extracted_count < MAX_EXTRACTED_FRAMES:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % FRAME_STEP == 0:
            image_name = f"frame_{frame_index:06d}.png"
            cv2.imwrite(str(images_dir / image_name), frame)

            pose = detect_aruco_pose(
                frame,
                detector,
                camera_matrix,
                distortion,
            )
            if pose is not None:
                aruco_poses[image_name] = pose

            extracted_count += 1

        frame_index += 1

    capture.release()
    return extracted_count, aruco_poses


def colmap_camera_parameters(camera_matrix, distortion):
    distortion = distortion.reshape(-1)
    k1, k2, p1, p2, k3 = distortion[:5]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    return ",".join(
        map(
            str,
            [
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
                k3,
                0.0,
                0.0,
                0.0,
            ],
        )
    )


def run_sparse_sfm(images_dir, work_dir, camera_matrix, distortion):
    database_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir()

    reader_options = pycolmap.ImageReaderOptions(
        camera_model="FULL_OPENCV",
        camera_params=colmap_camera_parameters(
            camera_matrix,
            distortion,
        ),
    )

    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        device=pycolmap.Device.auto,
    )

    pairing_options = pycolmap.SequentialPairingOptions(
        overlap=SEQUENTIAL_MATCH_OVERLAP,
        quadratic_overlap=True,
    )
    pycolmap.match_sequential(
        database_path=database_path,
        pairing_options=pairing_options,
        device=pycolmap.Device.auto,
    )

    mapping_options = pycolmap.IncrementalPipelineOptions(
        multiple_models=False,
        ba_refine_focal_length=False,
        ba_refine_principal_point=False,
        ba_refine_extra_params=False,
    )
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_dir,
        output_path=sparse_dir,
        options=mapping_options,
    )
    return max(
        reconstructions.values(),
        key=lambda reconstruction: reconstruction.num_reg_images(),
    )


def aruco_camera_center(R_aruco_to_camera, t_aruco_to_camera):
    return -R_aruco_to_camera.T @ t_aruco_to_camera


def calculate_sfm_to_aruco(reconstruction, aruco_poses):
    rotations_sfm_to_aruco = []
    camera_centers_sfm = []
    camera_centers_aruco = []

    for image in reconstruction.images.values():
        if image.name not in aruco_poses:
            continue

        R_aruco_to_camera, t_aruco_to_camera = aruco_poses[
            image.name
        ]
        R_sfm_to_camera = image.cam_from_world().rotation.matrix()

        rotations_sfm_to_aruco.append(
            R_aruco_to_camera.T @ R_sfm_to_camera
        )
        camera_centers_sfm.append(image.projection_center())
        camera_centers_aruco.append(
            aruco_camera_center(
                R_aruco_to_camera,
                t_aruco_to_camera,
            )
        )

    if len(camera_centers_sfm) < MIN_ARUCO_ALIGNMENT_FRAMES:
        raise RuntimeError(
            "ArUco must be detected in at least "
            f"{MIN_ARUCO_ALIGNMENT_FRAMES} registered SfM frames"
        )

    R_sfm_to_aruco = Rotation.from_matrix(
        rotations_sfm_to_aruco
    ).mean().as_matrix()

    camera_centers_sfm = np.asarray(camera_centers_sfm)
    camera_centers_aruco = np.asarray(camera_centers_aruco)
    rotated_centers = (
        R_sfm_to_aruco @ camera_centers_sfm.T
    ).T

    rotated_mean = np.mean(rotated_centers, axis=0)
    aruco_mean = np.mean(camera_centers_aruco, axis=0)
    rotated_centered = rotated_centers - rotated_mean
    aruco_centered = camera_centers_aruco - aruco_mean

    scale = np.sum(rotated_centered * aruco_centered) / np.sum(
        rotated_centered**2
    )
    translation = aruco_mean - scale * R_sfm_to_aruco @ np.mean(
        camera_centers_sfm,
        axis=0,
    )

    aligned_centers = scale * rotated_centers + translation
    alignment_rmse = np.sqrt(
        np.mean(
            np.sum(
                (aligned_centers - camera_centers_aruco) ** 2,
                axis=1,
            )
        )
    )

    return {
        "scale": scale,
        "rotation": R_sfm_to_aruco,
        "translation": translation,
        "alignment_rmse_mm": alignment_rmse,
        "alignment_frames": len(camera_centers_sfm),
    }


def transform_points_to_aruco(points_sfm, alignment):
    return (
        alignment["scale"]
        * (alignment["rotation"] @ points_sfm.T).T
        + alignment["translation"]
    )


def collect_surface_points(reconstruction, alignment):
    points_sfm = []
    colors = []
    errors = []
    track_lengths = []

    for point in reconstruction.points3D.values():
        if point.error > MAX_POINT_REPROJECTION_ERROR_PX:
            continue
        if point.track.length() < MIN_POINT_TRACK_LENGTH:
            continue

        points_sfm.append(point.xyz)
        colors.append(point.color)
        errors.append(point.error)
        track_lengths.append(point.track.length())

    points = transform_points_to_aruco(
        np.asarray(points_sfm),
        alignment,
    )
    colors = np.asarray(colors)
    errors = np.asarray(errors)
    track_lengths = np.asarray(track_lengths)

    within_surface_depth = (
        (points[:, 2] >= SURFACE_Z_MIN_MM)
        & (points[:, 2] <= SURFACE_Z_MAX_MM)
    )

    marker_half = ARUCO_SIZE_MM / 2.0
    marker_limit = marker_half + ARUCO_POINT_EXCLUSION_MARGIN_MM
    inside_marker = (
        (np.abs(points[:, 0]) <= marker_limit)
        & (np.abs(points[:, 1]) <= marker_limit)
    )

    keep = within_surface_depth & ~inside_marker
    return (
        points[keep],
        colors[keep],
        errors[keep],
        track_lengths[keep],
    )


def polynomial_powers(degree):
    powers = []
    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):
            y_power = total_degree - x_power
            powers.append((x_power, y_power))
    return powers


def polynomial_matrix(x, y, powers):
    return np.column_stack(
        [
            x**x_power * y**y_power
            for x_power, y_power in powers
        ]
    )


def fit_surface_polynomial(points):
    x_center = np.mean(points[:, 0])
    y_center = np.mean(points[:, 1])
    x_scale = np.ptp(points[:, 0]) / 2.0
    y_scale = np.ptp(points[:, 1]) / 2.0

    x = (points[:, 0] - x_center) / x_scale
    y = (points[:, 1] - y_center) / y_scale
    z = points[:, 2]
    powers = polynomial_powers(POLYNOMIAL_DEGREE)
    matrix = polynomial_matrix(x, y, powers)

    coefficients = np.linalg.lstsq(matrix, z, rcond=None)[0]
    residuals = z - matrix @ coefficients
    residual_median = np.median(residuals)
    residual_mad = np.median(
        np.abs(residuals - residual_median)
    )
    robust_std = 1.4826 * residual_mad
    inliers = (
        np.abs(residuals - residual_median)
        <= POLYNOMIAL_OUTLIER_SIGMA * robust_std
    )

    coefficients = np.linalg.lstsq(
        matrix[inliers],
        z[inliers],
        rcond=None,
    )[0]
    final_residuals = z[inliers] - matrix[inliers] @ coefficients

    return {
        "degree": POLYNOMIAL_DEGREE,
        "x_center": x_center,
        "x_scale": x_scale,
        "y_center": y_center,
        "y_scale": y_scale,
        "powers": powers,
        "coefficients": coefficients,
        "fit_rmse_mm": np.sqrt(np.mean(final_residuals**2)),
        "inliers": inliers,
    }


def evaluate_surface(model, x, y):
    normalized_x = (x - model["x_center"]) / model["x_scale"]
    normalized_y = (y - model["y_center"]) / model["y_scale"]
    matrix = polynomial_matrix(
        normalized_x.reshape(-1),
        normalized_y.reshape(-1),
        model["powers"],
    )
    return (matrix @ model["coefficients"]).reshape(x.shape)


def surface_equation_text(model):
    terms = []
    for (x_power, y_power), coefficient in zip(
        model["powers"],
        model["coefficients"],
    ):
        factors = [f"{coefficient:+.12g}"]
        if x_power:
            factors.append(f"xn^{x_power}")
        if y_power:
            factors.append(f"yn^{y_power}")
        terms.append(" * ".join(factors))

    return (
        "z_mm = "
        + " ".join(terms)
        + "\n"
        + f"xn = (x_mm - {model['x_center']:.12g}) / "
        + f"{model['x_scale']:.12g}\n"
        + f"yn = (y_mm - {model['y_center']:.12g}) / "
        + f"{model['y_scale']:.12g}\n"
    )


def save_point_cloud(path, points, colors, errors, track_lengths):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "x_mm",
                "y_mm",
                "z_mm",
                "red",
                "green",
                "blue",
                "reprojection_error_px",
                "track_length",
            ]
        )
        for point, color, error, track_length in zip(
            points,
            colors,
            errors,
            track_lengths,
        ):
            writer.writerow(
                [*point, *color, error, track_length]
            )


def save_surface_model(path, model, alignment, points):
    terms = []
    for (x_power, y_power), coefficient in zip(
        model["powers"],
        model["coefficients"],
    ):
        terms.append(
            {
                "x_power": x_power,
                "y_power": y_power,
                "coefficient_mm": float(coefficient),
            }
        )

    data = {
        "coordinate_system": "aruco_marker",
        "units": "millimeters",
        "axes": {
            "origin": "center of the ArUco marker",
            "x": "ArUco X axis",
            "y": "ArUco Y axis",
            "z": "ArUco surface normal",
        },
        "surface_equation": (
            "z = sum(c * xn^x_power * yn^y_power), "
            "xn=(x-x_center)/x_scale, "
            "yn=(y-y_center)/y_scale"
        ),
        "expanded_equation": surface_equation_text(model).strip(),
        "degree": model["degree"],
        "x_center_mm": float(model["x_center"]),
        "x_scale_mm": float(model["x_scale"]),
        "y_center_mm": float(model["y_center"]),
        "y_scale_mm": float(model["y_scale"]),
        "terms": terms,
        "fit_rmse_mm": float(model["fit_rmse_mm"]),
        "fit_point_count": int(np.count_nonzero(model["inliers"])),
        "point_cloud_bounds_mm": {
            "minimum": np.min(points, axis=0).tolist(),
            "maximum": np.max(points, axis=0).tolist(),
        },
        "sfm_to_aruco": {
            "scale": float(alignment["scale"]),
            "rotation": alignment["rotation"].tolist(),
            "translation_mm": alignment["translation"].tolist(),
            "alignment_rmse_mm": float(
                alignment["alignment_rmse_mm"]
            ),
            "alignment_frames": alignment["alignment_frames"],
        },
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    path.with_name("surface_equation.txt").write_text(
        surface_equation_text(model),
        encoding="utf-8",
    )


def save_surface_preview(path, points, colors, model):
    inlier_points = points[model["inliers"]]
    x_values = np.linspace(
        np.min(inlier_points[:, 0]),
        np.max(inlier_points[:, 0]),
        SURFACE_GRID_SIZE,
    )
    y_values = np.linspace(
        np.min(inlier_points[:, 1]),
        np.max(inlier_points[:, 1]),
        SURFACE_GRID_SIZE,
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_grid = evaluate_surface(model, x_grid, y_grid)

    figure = plt.figure(figsize=(17, 8))
    surface_axis = figure.add_subplot(121, projection="3d")
    surface_axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors / 255.0,
        s=3,
        alpha=0.45,
        label="SfM points",
    )
    surface_axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap="viridis",
        alpha=0.65,
        linewidth=0,
    )
    coordinate_axis_length = 0.15 * max(
        np.ptp(x_values),
        np.ptp(y_values),
    )
    surface_axis.quiver(
        0.0, 0.0, 0.0,
        coordinate_axis_length, 0.0, 0.0,
        color="red", linewidth=2,
    )
    surface_axis.quiver(
        0.0, 0.0, 0.0,
        0.0, coordinate_axis_length, 0.0,
        color="green", linewidth=2,
    )
    surface_axis.quiver(
        0.0, 0.0, 0.0,
        0.0, 0.0, coordinate_axis_length,
        color="blue", linewidth=2,
    )
    surface_axis.text(
        coordinate_axis_length, 0.0, 0.0, "X", color="red"
    )
    surface_axis.text(
        0.0, coordinate_axis_length, 0.0, "Y", color="green"
    )
    surface_axis.text(
        0.0, 0.0, coordinate_axis_length, "Z", color="blue"
    )

    marker_half = ARUCO_SIZE_MM / 2.0
    marker_outline = np.array(
        [
            [-marker_half, marker_half, 0.0],
            [marker_half, marker_half, 0.0],
            [marker_half, -marker_half, 0.0],
            [-marker_half, -marker_half, 0.0],
            [-marker_half, marker_half, 0.0],
        ]
    )
    surface_axis.plot(
        marker_outline[:, 0],
        marker_outline[:, 1],
        marker_outline[:, 2],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="ArUco",
    )

    surface_axis.set_xlabel("ArUco X [mm]")
    surface_axis.set_ylabel("ArUco Y [mm]")
    surface_axis.set_zlabel("ArUco Z [mm]")
    surface_axis.set_title(
        "3D surface in the ArUco coordinate system\n"
        f"polynomial RMSE: {model['fit_rmse_mm']:.2f} mm"
    )
    surface_axis.set_box_aspect(
        (
            np.ptp(x_values),
            np.ptp(y_values),
            max(np.ptp(z_grid), 1.0),
        )
    )
    surface_axis.legend()

    top_axis = figure.add_subplot(122)
    height_map = top_axis.contourf(
        x_grid,
        y_grid,
        z_grid,
        levels=40,
        cmap="viridis",
    )
    top_axis.plot(
        marker_outline[:, 0],
        marker_outline[:, 1],
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    top_axis.scatter(0.0, 0.0, color="black", s=25)
    top_axis.axhline(0.0, color="green", linewidth=0.8, alpha=0.7)
    top_axis.axvline(0.0, color="red", linewidth=0.8, alpha=0.7)
    top_axis.set_xlabel("ArUco X [mm]")
    top_axis.set_ylabel("ArUco Y [mm]")
    top_axis.set_title("Top view — color represents ArUco Z")
    top_axis.set_aspect("equal", adjustable="box")
    top_axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    figure.colorbar(height_map, ax=top_axis, label="ArUco Z [mm]")

    figure.tight_layout()
    figure.savefig(path, dpi=180)
    if SHOW_SURFACE_PLOT:
        plt.show()
    plt.close(figure)

    np.savez(
        path.with_name("surface_grid.npz"),
        x_mm=x_values,
        y_mm=y_values,
        z_mm=z_grid,
    )


def main():
    video_path = find_video()
    camera_matrix = np.load(CAMERA_MATRIX_PATH)
    distortion = np.load(DISTORTION_PATH)

    work_dir = OUTPUT_DIR / "sfm_work"
    images_dir = work_dir / "images"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    images_dir.mkdir(parents=True)

    frame_count, aruco_poses = extract_video_frames(
        video_path,
        images_dir,
        camera_matrix,
        distortion,
    )
    print(f"Extracted frames: {frame_count}")
    print(f"Frames with ArUco: {len(aruco_poses)}")

    reconstruction = run_sparse_sfm(
        images_dir,
        work_dir,
        camera_matrix,
        distortion,
    )
    print(f"Registered SfM frames: {reconstruction.num_reg_images()}")
    print(f"Sparse SfM points: {reconstruction.num_points3D()}")

    alignment = calculate_sfm_to_aruco(
        reconstruction,
        aruco_poses,
    )
    points, colors, errors, track_lengths = collect_surface_points(
        reconstruction,
        alignment,
    )
    model = fit_surface_polynomial(points)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_point_cloud(
        OUTPUT_DIR / "surface_points_aruco.csv",
        points,
        colors,
        errors,
        track_lengths,
    )
    save_surface_model(
        OUTPUT_DIR / "surface_model.json",
        model,
        alignment,
        points,
    )
    save_surface_preview(
        OUTPUT_DIR / "surface_model.png",
        points,
        colors,
        model,
    )

    print(
        "ArUco alignment RMSE: "
        f"{alignment['alignment_rmse_mm']:.3f} mm"
    )
    print(f"Surface points after filtering: {len(points)}")
    print(f"Surface fit RMSE: {model['fit_rmse_mm']:.3f} mm")
    print(f"Saved results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
