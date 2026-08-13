import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from aruco_reference import create_aruco_detector, detect_aruco_pose
from pipeline_diagnostics import (
    save_mapping_pipeline_csv,
    save_mapping_pipeline_diagnostics,
)


MIN_ARUCO_ALIGNMENT_FRAMES = 3

MIN_PAIR_MATCHES = 20
MIN_PAIR_INLIERS = 20

MIN_POINT_TRACK_LENGTH = 3
MAX_POINT_REPROJECTION_ERROR_PX = 3.0



def select_spatially_distributed_features(
    features,
    maximum_features,
    grid_rows,
    grid_columns,
):
    keypoints = features["keypoints"]
    scores = features["scores"]
    width, height = features["image_size"]
    roi_top = features["roi_top"]

    columns = np.minimum(
        (keypoints[:, 0] * grid_columns / width).astype(int),
        grid_columns - 1,
    )
    rows = np.minimum(
        (
            (keypoints[:, 1] - roi_top)
            * grid_rows
            / (height - roi_top)
        ).astype(int),
        grid_rows - 1,
    )

    indices_by_cell = []
    for row in range(grid_rows):
        for column in range(grid_columns):
            indices = np.flatnonzero(
                (rows == row) & (columns == column)
            )
            indices = indices[np.argsort(scores[indices])[::-1]]
            indices_by_cell.append(indices)

    selected_indices = []
    rank = 0
    while len(selected_indices) < maximum_features:
        added_in_round = False
        for indices in indices_by_cell:
            if rank < len(indices):
                selected_indices.append(indices[rank])
                added_in_round = True
                if len(selected_indices) == maximum_features:
                    break
        if not added_in_round:
            break
        rank += 1

    selected_indices = np.asarray(selected_indices, dtype=int)
    selected = {
        "keypoints": features["keypoints"][selected_indices],
        "descriptors": features["descriptors"][selected_indices],
        "scores": features["scores"][selected_indices],
        "image_size": features["image_size"],
        "roi_top": roi_top,
    }
    for name in ("scales", "oris"):
        if name in features:
            selected[name] = features[name][selected_indices]
    return selected


def select_uniform_landmarks(
    positions,
    track_lengths,
    reprojection_errors,
    maximum_landmarks,
    grid_rows,
    grid_columns,
    reprojection_error_weight,
):
    minimum_xy = np.min(positions[:, :2], axis=0)
    extent_xy = np.maximum(
        np.ptp(positions[:, :2], axis=0),
        np.finfo(float).eps,
    )
    normalized_xy = (positions[:, :2] - minimum_xy) / extent_xy
    columns = np.minimum(
        (normalized_xy[:, 0] * grid_columns).astype(int),
        grid_columns - 1,
    )
    rows = np.minimum(
        (normalized_xy[:, 1] * grid_rows).astype(int),
        grid_rows - 1,
    )

    track_quality = (
        track_lengths - np.min(track_lengths)
    ) / max(np.ptp(track_lengths), 1)
    reprojection_quality = 1.0 - (
        reprojection_errors - np.min(reprojection_errors)
    ) / max(np.ptp(reprojection_errors), np.finfo(float).eps)
    landmark_quality = (
        reprojection_error_weight * reprojection_quality
        + (1.0 - reprojection_error_weight) * track_quality
    )

    indices_by_cell = []
    for row in range(grid_rows):
        for column in range(grid_columns):
            indices = np.flatnonzero(
                (rows == row) & (columns == column)
            )
            if len(indices) == 0:
                continue

            quality_order = np.argsort(landmark_quality[indices])[::-1]
            indices_by_cell.append(indices[quality_order])

    selected_indices = []
    rank = 0
    while len(selected_indices) < maximum_landmarks:
        added_in_round = False
        for indices in indices_by_cell:
            if rank < len(indices):
                selected_indices.append(indices[rank])
                added_in_round = True
                if len(selected_indices) == maximum_landmarks:
                    break
        if not added_in_round:
            break
        rank += 1

    occupied_cells = len(indices_by_cell)
    landmarks_per_cell = int(
        np.ceil(min(len(positions), maximum_landmarks) / occupied_cells)
    )
    return (
        np.asarray(selected_indices, dtype=int),
        occupied_cells,
        landmarks_per_cell,
    )


def create_colmap_camera(camera_matrix, distortion, width, height):
    distortion = distortion.reshape(-1)
    distortion = np.pad(distortion, (0, max(0, 8 - len(distortion))))
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3, k4, k5, k6 = distortion[:8]
    parameters = np.array(
        [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6],
        dtype=float,
    )
    camera = pycolmap.Camera(
        model="FULL_OPENCV",
        width=width,
        height=height,
        params=parameters,
    )
    camera.has_prior_focal_length = True
    return camera


def create_gravity_prior_rig(database, camera_id):
    """Create the one-camera rig required by COLMAP frame pose priors."""
    camera_sensor = pycolmap.sensor_t(
        pycolmap.SensorType.CAMERA,
        camera_id,
    )
    rig = pycolmap.Rig()
    rig.add_ref_sensor(camera_sensor)
    return camera_sensor, database.write_rig(rig)


def write_mapping_image(
    database,
    camera,
    image_name,
    *,
    camera_sensor=None,
    rig_id=None,
    image_id=None,
    gravity=None,
):
    """Write one mapping image and, optionally, its gravity pose prior."""
    if camera_sensor is None:
        database.write_image(
            pycolmap.Image(
                image_id=image_id,
                name=image_name,
                camera_id=camera.camera_id,
            ),
            use_image_id=True,
        )
        return image_id

    camera_data = pycolmap.data_t(camera_sensor, image_id)
    colmap_frame = pycolmap.Frame(rig_id=rig_id)
    colmap_frame.add_data_id(camera_data)
    colmap_frame.finalize_data_ids()
    frame_id = database.write_frame(colmap_frame)
    database.write_image(
        pycolmap.Image(
            image_id=image_id,
            name=image_name,
            camera_id=camera.camera_id,
            frame_id=frame_id,
        ),
        use_image_id=True,
    )
    if gravity is not None:
        pose_prior = pycolmap.PosePrior()
        pose_prior.corr_data_id = camera_data
        pose_prior.gravity = gravity
        database.write_pose_prior(pose_prior)
    return image_id


def camera_center(R_world_to_camera, t_world_to_camera):
    return -R_world_to_camera.T @ t_world_to_camera


def frame_number(image_name):
    return int(Path(image_name).stem.rsplit("_", 1)[1])


def summarize_track_lengths(track_lengths):
    track_lengths = np.asarray(track_lengths, dtype=int)
    lengths, counts = np.unique(track_lengths, return_counts=True)
    ranges = {
        "2": int(np.sum(track_lengths == 2)),
        "3-4": int(np.sum((track_lengths >= 3) & (track_lengths <= 4))),
        "5-9": int(np.sum((track_lengths >= 5) & (track_lengths <= 9))),
        "10-19": int(
            np.sum((track_lengths >= 10) & (track_lengths <= 19))
        ),
        "20-49": int(
            np.sum((track_lengths >= 20) & (track_lengths <= 49))
        ),
        "50-99": int(
            np.sum((track_lengths >= 50) & (track_lengths <= 99))
        ),
        "100+": int(np.sum(track_lengths >= 100)),
    }
    return {
        "landmarks": int(len(track_lengths)),
        "minimum": int(np.min(track_lengths)),
        "mean": float(np.mean(track_lengths)),
        "median": float(np.median(track_lengths)),
        "percentile_90": float(np.percentile(track_lengths, 90)),
        "maximum": int(np.max(track_lengths)),
        "exact_counts": {
            str(int(length)): int(count)
            for length, count in zip(lengths, counts)
        },
        "ranges": ranges,
    }


def print_track_length_statistics(statistics):
    print("Track length statistics")
    for name, values in statistics.items():
        print(
            f"  {name}: {values['landmarks']} landmarks | "
            f"mean {values['mean']:.1f} | "
            f"median {values['median']:.1f} | "
            f"p90 {values['percentile_90']:.1f} | "
            f"max {values['maximum']}"
        )
        ranges = " | ".join(
            f"{length}: {count}"
            for length, count in values["ranges"].items()
        )
        print(f"    ranges: {ranges}")


def print_map_build_timing(timing):
    online = timing["mapping_frame_processing"]
    after = timing["after_collection"]
    online_stages = [
        ("setup_s", "Setup"),
        ("frame_read_s", "Frame reading"),
        ("image_save_s", "Image saving"),
        (
            "feature_extraction_and_selection_s",
            "Feature extraction + selection",
        ),
        ("aruco_detection_s", "ArUco detection"),
        (
            "image_and_keypoints_database_write_s",
            "Image/keypoint database writes",
        ),
        ("lightglue_matching_s", "LightGlue matching"),
        ("pair_geometry_verification_s", "Pair geometry verification"),
        (
            "verified_pair_database_write_s",
            "Verified-pair database writes",
        ),
    ]
    after_stages = [
        (
            "colmap_reconstruction_and_bundle_adjustment_s",
            "COLMAP reconstruction + bundle adjustment",
        ),
        ("aruco_metric_alignment_s", "ArUco metric alignment"),
        (
            "landmark_selection_and_descriptors_s",
            "Landmark selection + descriptors",
        ),
        ("map_saving_s", "Map saving"),
    ]

    print("Map build timing")
    print("  MAPPING FRAME PROCESSING - executed while frames arrive")
    for key, label in online_stages:
        print(f"    {label}: {online[key]:.2f} s")
    print(
        f"    TOTAL: {online['wall_time_s']:.2f} s | "
        f"{online['average_wall_time_per_frame_ms']:.1f} ms/frame"
    )
    print(
        f"    Frames: {online['processed_frames']} | "
        f"pairs: {online['verified_pairs']}/"
        f"{online['attempted_pairs']} verified"
    )
    print("  AFTER COLLECTION - executed after the last mapping frame")
    for key, label in after_stages:
        print(f"    {label}: {after[key]:.2f} s")
    print(f"    TOTAL: {after['wall_time_s']:.2f} s")
    print(
        "  COMPLETE MAP BUILD: "
        f"{timing['total_map_build_wall_time_s']:.2f} s"
    )


class SkinMapBuilder:
    def __init__(
        self,
        camera_matrix,
        distortion,
        feature_matching,
        mapping_start_frame,
        mapping_end_frame,
        reconstruction_method,
        mapping_frame_step,
        sequential_match_overlap,
        mapping_maximum_features,
        mapping_feature_grid_rows,
        mapping_feature_grid_columns,
        maximum_global_landmarks,
        global_map_grid_rows,
        global_map_grid_columns,
        global_map_reprojection_error_weight,
        imu_gravity_provider=None,
    ):
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.feature_matching = feature_matching
        self.mapping_start_frame = mapping_start_frame
        self.mapping_end_frame = mapping_end_frame
        self.reconstruction_method = reconstruction_method
        self.mapping_frame_step = mapping_frame_step
        self.sequential_match_overlap = sequential_match_overlap
        self.mapping_maximum_features = mapping_maximum_features
        self.mapping_feature_grid_rows = mapping_feature_grid_rows
        self.mapping_feature_grid_columns = mapping_feature_grid_columns
        self.maximum_global_landmarks = maximum_global_landmarks
        self.global_map_grid_rows = global_map_grid_rows
        self.global_map_grid_columns = global_map_grid_columns
        self.global_map_reprojection_error_weight = (
            global_map_reprojection_error_weight
        )
        self.imu_gravity_provider = imu_gravity_provider
        if (
            self.imu_gravity_provider is not None
            and self.reconstruction_method != "global"
        ):
            raise ValueError(
                "IMU gravity priors are supported only by global mapping"
            )

        self.aruco_detector = create_aruco_detector()

    def detect_aruco_pose(self, frame):
        return detect_aruco_pose(
            frame,
            self.camera_matrix,
            self.distortion,
            self.aruco_detector,
        )

    def collect_and_match_mapping_frames(
        self,
        video_path,
        images_dir,
        database_path,
    ):
        frame_processing_started = time.perf_counter()
        setup_started = time.perf_counter()
        capture = cv2.VideoCapture(str(video_path))
        database = pycolmap.Database.open(database_path)
        camera = create_colmap_camera(
            self.camera_matrix,
            self.distortion,
            round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        camera_id = database.write_camera(camera)
        camera.camera_id = camera_id
        if self.imu_gravity_provider is not None:
            camera_sensor, rig_id = create_gravity_prior_rig(
                database,
                camera_id,
            )
        setup_seconds = time.perf_counter() - setup_started

        image_names = []
        image_ids = {}
        features_by_name = {}
        aruco_poses = {}
        frame_times_by_name = {}
        mapping_frame_rows = []

        geometry_options = pycolmap.TwoViewGeometryOptions()
        geometry_options.ransac.random_seed = 0
        pair_count = 0
        attempted_pair_count = 0
        frame_read_seconds = 0.0
        image_save_seconds = 0.0
        feature_extraction_seconds = 0.0
        aruco_seconds = 0.0
        image_database_write_seconds = 0.0
        lightglue_seconds = 0.0
        geometry_seconds = 0.0
        pair_database_write_seconds = 0.0

        for _ in range(self.mapping_start_frame):
            success, _ = capture.read()
            if not success:
                break

        for frame_index in range(
            self.mapping_start_frame,
            self.mapping_end_frame + 1,
        ):
            reading_started = time.perf_counter()
            success, frame = capture.read()
            frame_read_seconds += time.perf_counter() - reading_started
            if not success:
                break
            if frame_index % self.mapping_frame_step != 0:
                continue

            image_name = f"frame_{frame_index:06d}.png"
            saving_started = time.perf_counter()
            cv2.imwrite(str(images_dir / image_name), frame)
            image_save_seconds += time.perf_counter() - saving_started

            feature_extraction_started = time.perf_counter()
            detected_features = self.feature_matching.extract(frame)
            features = select_spatially_distributed_features(
                detected_features,
                self.mapping_maximum_features,
                self.mapping_feature_grid_rows,
                self.mapping_feature_grid_columns,
            )
            feature_extraction_seconds += (
                time.perf_counter() - feature_extraction_started
            )
            features_by_name[image_name] = features
            frame_times_by_name[image_name] = (
                capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            )
            if self.imu_gravity_provider is not None:
                gravity, imu_diagnostics = (
                    self.imu_gravity_provider.gravity_at_video_time(
                        frame_times_by_name[image_name]
                    )
                )
            else:
                gravity = None
                imu_diagnostics = None

            aruco_started = time.perf_counter()
            aruco_pose = self.detect_aruco_pose(frame)
            aruco_seconds += time.perf_counter() - aruco_started
            if aruco_pose is not None:
                aruco_poses[image_name] = aruco_pose

            database_write_started = time.perf_counter()
            mapping_image_id = len(image_names) + 1
            if self.imu_gravity_provider is not None:
                # Frame pose priors need a stable camera-data identifier.
                image_id = write_mapping_image(
                    database,
                    camera,
                    image_name,
                    camera_sensor=camera_sensor,
                    rig_id=rig_id,
                    image_id=mapping_image_id,
                    gravity=gravity,
                )
                imu_status = (
                    f" | IMU gravity: {imu_diagnostics['reason']}"
                    f" (|a|={imu_diagnostics.get('acceleration_magnitude_m_s2', np.nan):.3f} m/s^2, "
                    f"|w|={np.degrees(imu_diagnostics.get('gyroscope_magnitude_rad_s', np.nan)):.2f} deg/s)"
                )
            else:
                image_id = write_mapping_image(
                    database,
                    camera,
                    image_name,
                    image_id=mapping_image_id,
                )
                imu_status = ""
            image_ids[image_name] = image_id
            database.write_keypoints(image_id, features["keypoints"])
            image_database_write_seconds += (
                time.perf_counter() - database_write_started
            )

            sequential_names = image_names[
                -self.sequential_match_overlap :
            ]
            raw_match_count = 0
            verified_inlier_count = 0
            verified_pairs_for_image = 0
            for previous_name in sequential_names:
                attempted_pair_count += 1

                matching_started = time.perf_counter()
                matches = self.feature_matching.match(
                    features_by_name[previous_name],
                    features,
                )
                raw_match_count += len(matches)
                lightglue_seconds += time.perf_counter() - matching_started
                if len(matches) < MIN_PAIR_MATCHES:
                    continue

                geometry_started = time.perf_counter()
                geometry = pycolmap.estimate_two_view_geometry(
                    camera,
                    features_by_name[previous_name]["keypoints"],
                    camera,
                    features["keypoints"],
                    matches,
                    geometry_options,
                )
                geometry_seconds += time.perf_counter() - geometry_started
                if len(geometry.inlier_matches) < MIN_PAIR_INLIERS:
                    continue

                verified_inlier_count += len(geometry.inlier_matches)
                verified_pairs_for_image += 1

                previous_id = image_ids[previous_name]
                writing_started = time.perf_counter()
                database.write_two_view_geometry(
                    previous_id,
                    image_id,
                    geometry,
                )
                pair_database_write_seconds += (
                    time.perf_counter() - writing_started
                )
                pair_count += 1

            aruco_quality = None if aruco_pose is None else aruco_pose[2]
            mapping_frame_rows.append(
                {
                    "frame": frame_index,
                    "time_s": frame_times_by_name[image_name],
                    "image_name": image_name,
                    "feature_count": len(features["keypoints"]),
                    "pairs_attempted": len(sequential_names),
                    "raw_matches": raw_match_count,
                    "pairs_verified": verified_pairs_for_image,
                    "verified_inliers": verified_inlier_count,
                    "aruco_detected": int(aruco_pose is not None),
                    "aruco_reprojection_rms_px": (
                        np.nan
                        if aruco_quality is None
                        else aruco_quality["reprojection_rms_px"]
                    ),
                    "aruco_reprojection_max_px": (
                        np.nan
                        if aruco_quality is None
                        else aruco_quality["reprojection_max_px"]
                    ),
                }
            )

            image_names.append(image_name)
            print(
                f"Mapping frame processing: frame {frame_index}/"
                f"{self.mapping_end_frame} | "
                f"features: {len(features['keypoints'])} | "
                f"verified pairs: {pair_count}{imu_status}"
            )

        capture.release()
        database.close()
        imu_gravity_summary = (
            None
            if self.imu_gravity_provider is None
            else self.imu_gravity_provider.summary()
        )
        statistics = {
            "attempted_pairs": attempted_pair_count,
            "verified_pairs": pair_count,
            "setup_seconds": setup_seconds,
            "frame_read_seconds": frame_read_seconds,
            "image_save_seconds": image_save_seconds,
            "feature_extraction_and_selection_seconds": (
                feature_extraction_seconds
            ),
            "aruco_detection_seconds": aruco_seconds,
            "image_database_write_seconds": (
                image_database_write_seconds
            ),
            "lightglue_seconds": lightglue_seconds,
            "geometry_verification_seconds": geometry_seconds,
            "pair_database_write_seconds": pair_database_write_seconds,
            "mapping_frame_processing_wall_seconds": (
                time.perf_counter() - frame_processing_started
            ),
            "imu_gravity": imu_gravity_summary,
        }
        return (
            image_names,
            features_by_name,
            aruco_poses,
            frame_times_by_name,
            statistics,
            mapping_frame_rows,
        )

    def reconstruct(self, database_path, images_dir, sparse_dir):
        if self.reconstruction_method == "global":
            options = pycolmap.GlobalPipelineOptions(
                min_num_matches=MIN_PAIR_INLIERS,
                random_seed=0,
            )
            options.mapper.random_seed = 0
            options.mapper.rotation_averaging.random_seed = 0
            options.mapper.rotation_averaging.use_gravity = (
                self.imu_gravity_provider is not None
            )
            options.mapper.global_positioning.random_seed = 0
            options.mapper.bundle_adjustment.refine_focal_length = False
            options.mapper.bundle_adjustment.refine_principal_point = False
            options.mapper.bundle_adjustment.refine_extra_params = False
            options.mapper.global_positioning.use_gpu = True
            options.mapper.global_positioning.min_num_images_gpu_solver = 3
            options.mapper.bundle_adjustment.ceres.use_gpu = True
            mapping = pycolmap.global_mapping
        else:
            options = pycolmap.IncrementalPipelineOptions(
                multiple_models=False,
                structure_less_registration_fallback=False,
                ba_refine_focal_length=False,
                ba_refine_principal_point=False,
                ba_refine_extra_params=False,
            )
            options.ba_use_gpu = True
            options.mapper.init_min_num_inliers = MIN_PAIR_INLIERS
            options.mapper.abs_pose_min_num_inliers = MIN_PAIR_INLIERS
            options.random_seed = 0
            options.mapper.random_seed = 0
            options.triangulation.random_seed = 0
            mapping = pycolmap.incremental_mapping

        reconstructions = mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=sparse_dir,
            options=options,
        )
        if not reconstructions:
            raise RuntimeError(
                "COLMAP could not initialize the 3D map. The mapping pass "
                "needs camera translation and visible feature parallax."
            )
        return max(
            reconstructions.values(),
            key=lambda reconstruction: reconstruction.num_reg_images(),
        )

    def align_to_aruco(self, reconstruction, aruco_poses):
        registered_aruco_images = sorted(
            [
                image
                for image in reconstruction.images.values()
                if image.name in aruco_poses
            ],
            key=lambda image: image.name,
        )
        if len(registered_aruco_images) < MIN_ARUCO_ALIGNMENT_FRAMES:
            raise RuntimeError(
                "ArUco must be visible in at least "
                f"{MIN_ARUCO_ALIGNMENT_FRAMES} registered mapping frames"
            )

        reference_image = registered_aruco_images[0]

        rotation_candidates = []
        sfm_centers = []
        aruco_centers = []

        for image in registered_aruco_images:
            R_aruco_to_camera, t_aruco_to_camera, _ = aruco_poses[
                image.name
            ]
            R_sfm_to_camera = image.cam_from_world().rotation.matrix()
            rotation_candidates.append(
                R_aruco_to_camera.T @ R_sfm_to_camera
            )
            sfm_centers.append(image.projection_center())
            aruco_centers.append(
                camera_center(
                    R_aruco_to_camera,
                    t_aruco_to_camera,
                )
            )

        rotation = Rotation.from_matrix(
            rotation_candidates
        ).mean().as_matrix()
        sfm_centers = np.asarray(sfm_centers)
        aruco_centers = np.asarray(aruco_centers)
        rotated_centers = (rotation @ sfm_centers.T).T

        rotated_mean = np.mean(rotated_centers, axis=0)
        aruco_mean = np.mean(aruco_centers, axis=0)
        rotated_centered = rotated_centers - rotated_mean
        aruco_centered = aruco_centers - aruco_mean
        scale = np.sum(
            rotated_centered * aruco_centered
        ) / np.sum(rotated_centered**2)
        translation = aruco_mean - scale * rotated_mean

        aligned_centers = scale * rotated_centers + translation
        alignment_rmse = np.sqrt(
            np.mean(
                np.sum(
                    (aligned_centers - aruco_centers) ** 2,
                    axis=1,
                )
            )
        )
        return {
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
            "alignment_rmse_mm": alignment_rmse,
            "alignment_frames": len(registered_aruco_images),
            "reference_image": reference_image.name,
            "center_residuals_by_name": {
                image.name: float(residual)
                for image, residual in zip(
                    registered_aruco_images,
                    np.linalg.norm(
                        aligned_centers - aruco_centers,
                        axis=1,
                    ),
                )
            },
            "coordinate_frame": "aruco",
        }

    def transform_points(self, points, alignment):
        return (
            alignment["scale"]
            * (alignment["rotation"] @ points.T).T
            + alignment["translation"]
        )

    def transform_pose(self, image, alignment):
        sfm_pose = image.cam_from_world()
        R_sfm_to_camera = sfm_pose.rotation.matrix()
        t_sfm_to_camera = np.asarray(sfm_pose.translation)
        R_map_to_camera = R_sfm_to_camera @ alignment["rotation"].T
        t_map_to_camera = (
            alignment["scale"] * t_sfm_to_camera
            - R_map_to_camera @ alignment["translation"]
        )
        return R_map_to_camera, t_map_to_camera

    def create_global_map(
        self,
        reconstruction,
        features_by_name,
        alignment,
        frame_times_by_name,
    ):
        positions = []
        track_lengths = []
        reprojection_errors = []
        first_observation_frames = []
        candidate_available_frames = []
        candidate_points = []

        for point in reconstruction.points3D.values():
            if point.track.length() < MIN_POINT_TRACK_LENGTH:
                continue
            if point.error > MAX_POINT_REPROJECTION_ERROR_PX:
                continue

            observation_frames = sorted(
                frame_number(
                    reconstruction.images[observation.image_id].name
                )
                for observation in point.track.elements
            )
            candidate_points.append(point)
            positions.append(point.xyz)
            track_lengths.append(point.track.length())
            reprojection_errors.append(point.error)
            first_observation_frames.append(observation_frames[0])
            candidate_available_frames.append(
                observation_frames[MIN_POINT_TRACK_LENGTH - 1]
            )

        positions = self.transform_points(
            np.asarray(positions, dtype=np.float64),
            alignment,
        )
        track_lengths = np.asarray(track_lengths, dtype=np.int32)
        reprojection_errors = np.asarray(
            reprojection_errors,
            dtype=np.float32,
        )
        first_observation_frames = np.asarray(
            first_observation_frames,
            dtype=np.int32,
        )
        candidate_available_frames = np.asarray(
            candidate_available_frames,
            dtype=np.int32,
        )
        candidate_positions = positions.copy()
        candidate_track_length_statistics = summarize_track_lengths(
            track_lengths
        )
        candidate_landmarks = len(positions)
        (
            selected_indices,
            occupied_grid_cells,
            landmarks_per_occupied_cell,
        ) = select_uniform_landmarks(
            positions,
            track_lengths,
            reprojection_errors,
            self.maximum_global_landmarks,
            self.global_map_grid_rows,
            self.global_map_grid_columns,
            self.global_map_reprojection_error_weight,
        )

        positions = positions[selected_indices]
        track_lengths = track_lengths[selected_indices]
        reprojection_errors = reprojection_errors[selected_indices]
        first_observation_frames = first_observation_frames[selected_indices]
        selected_track_length_statistics = summarize_track_lengths(
            track_lengths
        )
        selected_points = [
            candidate_points[index] for index in selected_indices
        ]

        descriptors = []
        scores = []
        include_scale_orientation = (
            self.feature_matching.requires_scale_orientation
        )
        scales = []
        orientations = []
        for point in selected_points:
            point_descriptors = []
            point_scores = []
            point_scales = []
            point_orientations = []
            for observation in point.track.elements:
                image = reconstruction.images[observation.image_id]
                point_descriptors.append(
                    features_by_name[image.name]["descriptors"][
                        observation.point2D_idx
                    ]
                )
                point_scores.append(
                    features_by_name[image.name]["scores"][
                        observation.point2D_idx
                    ]
                )
                if include_scale_orientation:
                    point_scales.append(
                        features_by_name[image.name]["scales"][
                            observation.point2D_idx
                        ]
                    )
                    point_orientations.append(
                        features_by_name[image.name]["oris"][
                            observation.point2D_idx
                        ]
                    )

            descriptor = np.mean(point_descriptors, axis=0)
            descriptor /= np.linalg.norm(descriptor)
            descriptors.append(descriptor)
            scores.append(np.mean(point_scores))
            if include_scale_orientation:
                scales.append(np.mean(point_scales))
                orientations.append(
                    np.arctan2(
                        np.mean(np.sin(point_orientations)),
                        np.mean(np.cos(point_orientations)),
                    )
                )

        descriptors = np.asarray(descriptors, dtype=np.float32)

        last_image = max(
            reconstruction.images.values(),
            key=lambda image: image.name,
        )
        initial_R, initial_t = self.transform_pose(last_image, alignment)

        mapping_frames = []
        mapping_times_s = []
        mapping_camera_positions = []
        mapping_camera_rotations = []
        mapping_camera_headings = []
        for image in sorted(
            reconstruction.images.values(),
            key=lambda registered_image: registered_image.name,
        ):
            R_map_to_camera, t_map_to_camera = self.transform_pose(
                image,
                alignment,
            )
            mapping_frames.append(frame_number(image.name))
            mapping_times_s.append(frame_times_by_name[image.name])
            mapping_camera_positions.append(
                camera_center(R_map_to_camera, t_map_to_camera)
            )
            mapping_camera_rotations.append(R_map_to_camera.T)
            mapping_camera_headings.append(
                R_map_to_camera.T @ np.array([0.0, -1.0, 0.0])
            )

        global_map = {
            "positions": positions,
            "descriptors": descriptors,
            "scores": np.asarray(scores, dtype=np.float32),
            "track_lengths": track_lengths,
            "reprojection_errors": reprojection_errors,
            "first_observation_frames": first_observation_frames,
            "candidate_positions": candidate_positions,
            "candidate_available_frames": candidate_available_frames,
            "selected_candidate_indices": np.asarray(
                selected_indices,
                dtype=np.int32,
            ),
            "candidate_landmarks": candidate_landmarks,
            "occupied_grid_cells": occupied_grid_cells,
            "landmarks_per_occupied_cell": (
                landmarks_per_occupied_cell
            ),
            "candidate_track_length_statistics": (
                candidate_track_length_statistics
            ),
            "selected_track_length_statistics": (
                selected_track_length_statistics
            ),
            "mapping_frames": np.asarray(mapping_frames, dtype=np.int32),
            "mapping_times_s": np.asarray(mapping_times_s, dtype=np.float64),
            "mapping_camera_positions": np.asarray(
                mapping_camera_positions,
                dtype=np.float64,
            ),
            "mapping_camera_rotations": np.asarray(
                mapping_camera_rotations,
                dtype=np.float64,
            ),
            "mapping_camera_headings": np.asarray(
                mapping_camera_headings,
                dtype=np.float64,
            ),
            "mapping_reference_frame": frame_number(
                alignment["reference_image"]
            ),
            "coordinate_frame": alignment["coordinate_frame"],
            "initial_R": initial_R,
            "initial_t": initial_t,
            "last_mapping_image": last_image.name,
        }
        if include_scale_orientation:
            global_map["scales"] = np.asarray(scales, dtype=np.float32)
            global_map["oris"] = np.asarray(
                orientations,
                dtype=np.float32,
            )
        return global_map

    def save_map(self, global_map, output_path):
        saved_arrays = dict(
            positions=global_map["positions"],
            descriptors=global_map["descriptors"],
            scores=global_map["scores"],
            track_lengths=global_map["track_lengths"],
            reprojection_errors=global_map["reprojection_errors"],
            first_observation_frames=global_map[
                "first_observation_frames"
            ],
            candidate_positions=global_map["candidate_positions"],
            candidate_available_frames=global_map[
                "candidate_available_frames"
            ],
            selected_candidate_indices=global_map[
                "selected_candidate_indices"
            ],
            mapping_frames=global_map["mapping_frames"],
            mapping_times_s=global_map["mapping_times_s"],
            mapping_camera_positions=global_map[
                "mapping_camera_positions"
            ],
            mapping_camera_rotations=global_map[
                "mapping_camera_rotations"
            ],
            mapping_camera_headings=global_map[
                "mapping_camera_headings"
            ],
            mapping_reference_frame=global_map[
                "mapping_reference_frame"
            ],
            coordinate_frame=global_map["coordinate_frame"],
            mapping_extracted_image_count=global_map[
                "mapping_extracted_image_count"
            ],
            candidate_landmarks=global_map["candidate_landmarks"],
            occupied_grid_cells=global_map["occupied_grid_cells"],
            landmarks_per_occupied_cell=global_map[
                "landmarks_per_occupied_cell"
            ],
            initial_R=global_map["initial_R"],
            initial_t=global_map["initial_t"],
            last_mapping_image=global_map["last_mapping_image"],
        )
        for name in ("scales", "oris"):
            if name in global_map:
                saved_arrays[name] = global_map[name]
        np.savez_compressed(output_path, **saved_arrays)

    def build(self, video_path, output_dir):
        build_started = time.perf_counter()
        output_dir = Path(output_dir)
        work_dir = output_dir / "colmap_work"
        images_dir = work_dir / "images"
        sparse_dir = work_dir / "sparse"
        database_path = work_dir / "database.db"

        if work_dir.exists():
            shutil.rmtree(work_dir)
        images_dir.mkdir(parents=True)
        sparse_dir.mkdir()

        (
            image_names,
            features_by_name,
            aruco_poses,
            frame_times_by_name,
            frame_processing_statistics,
            mapping_frame_rows,
        ) = self.collect_and_match_mapping_frames(
            video_path,
            images_dir,
            database_path,
        )
        imu_gravity_summary = frame_processing_statistics["imu_gravity"]
        if (
            imu_gravity_summary is not None
            and imu_gravity_summary["counts"]["accepted"] == 0
        ):
            raise RuntimeError(
                "No mapping frame passed the IMU gravity quality gates"
            )

        reconstruction_started = time.perf_counter()
        reconstruction = self.reconstruct(
            database_path,
            images_dir,
            sparse_dir,
        )
        reconstruction_seconds = (
            time.perf_counter() - reconstruction_started
        )
        all_track_length_statistics = summarize_track_lengths(
            [
                point.track.length()
                for point in reconstruction.points3D.values()
            ]
        )

        alignment_started = time.perf_counter()
        alignment = self.align_to_aruco(
            reconstruction,
            aruco_poses,
        )
        alignment_seconds = time.perf_counter() - alignment_started

        registered_image_names = {
            image.name for image in reconstruction.images.values()
        }
        for row in mapping_frame_rows:
            row["registered"] = int(
                row["image_name"] in registered_image_names
            )
            row["aruco_alignment_residual_mm"] = alignment[
                "center_residuals_by_name"
            ].get(row["image_name"], np.nan)
        mapping_diagnostics_csv_path = (
            output_dir / "mapping_pipeline_diagnostics.csv"
        )
        mapping_diagnostics_plot_path = (
            output_dir / "mapping_pipeline_diagnostics.png"
        )
        save_mapping_pipeline_csv(
            mapping_frame_rows,
            mapping_diagnostics_csv_path,
        )
        save_mapping_pipeline_diagnostics(
            mapping_frame_rows,
            mapping_diagnostics_plot_path,
            video_path.stem,
        )

        map_finalization_started = time.perf_counter()
        global_map = self.create_global_map(
            reconstruction,
            features_by_name,
            alignment,
            frame_times_by_name,
        )
        map_finalization_seconds = (
            time.perf_counter() - map_finalization_started
        )
        global_map["mapping_extracted_image_count"] = len(image_names)
        track_length_statistics = {
            "all_colmap_points": all_track_length_statistics,
            "quality_filtered_candidates": global_map[
                "candidate_track_length_statistics"
            ],
            "selected_global_map": global_map[
                "selected_track_length_statistics"
            ],
        }

        saving_started = time.perf_counter()
        map_path = output_dir / "global_map.npz"
        self.save_map(global_map, map_path)
        model_dir = output_dir / "colmap_model"
        model_dir.mkdir(exist_ok=True)
        reconstruction.write(model_dir)
        saving_seconds = time.perf_counter() - saving_started

        after_collection_seconds = (
            reconstruction_seconds
            + alignment_seconds
            + map_finalization_seconds
            + saving_seconds
        )
        processed_frames = len(image_names)
        online_wall_seconds = frame_processing_statistics[
            "mapping_frame_processing_wall_seconds"
        ]
        timing = {
            "mapping_frame_processing": {
                "processed_frames": processed_frames,
                "attempted_pairs": frame_processing_statistics[
                    "attempted_pairs"
                ],
                "verified_pairs": frame_processing_statistics[
                    "verified_pairs"
                ],
                "setup_s": frame_processing_statistics["setup_seconds"],
                "frame_read_s": frame_processing_statistics[
                    "frame_read_seconds"
                ],
                "image_save_s": frame_processing_statistics[
                    "image_save_seconds"
                ],
                "feature_extraction_and_selection_s": (
                    frame_processing_statistics[
                        "feature_extraction_and_selection_seconds"
                    ]
                ),
                "aruco_detection_s": frame_processing_statistics[
                    "aruco_detection_seconds"
                ],
                "image_and_keypoints_database_write_s": (
                    frame_processing_statistics[
                        "image_database_write_seconds"
                    ]
                ),
                "lightglue_matching_s": frame_processing_statistics[
                    "lightglue_seconds"
                ],
                "pair_geometry_verification_s": frame_processing_statistics[
                    "geometry_verification_seconds"
                ],
                "verified_pair_database_write_s": frame_processing_statistics[
                    "pair_database_write_seconds"
                ],
                "wall_time_s": online_wall_seconds,
                "average_wall_time_per_frame_ms": (
                    1000.0 * online_wall_seconds / processed_frames
                ),
            },
            "after_collection": {
                "colmap_reconstruction_and_bundle_adjustment_s": (
                    reconstruction_seconds
                ),
                "aruco_metric_alignment_s": alignment_seconds,
                "landmark_selection_and_descriptors_s": (
                    map_finalization_seconds
                ),
                "map_saving_s": saving_seconds,
                "wall_time_s": after_collection_seconds,
            },
            "total_map_build_wall_time_s": (
                time.perf_counter() - build_started
            ),
        }

        summary = {
            "feature_type": self.feature_matching.feature_type,
            "mapping_start_frame": self.mapping_start_frame,
            "mapping_end_frame": self.mapping_end_frame,
            "reconstruction_method": self.reconstruction_method,
            "mapping_frame_step": self.mapping_frame_step,
            "map_coordinate_frame": global_map["coordinate_frame"],
            "sequential_match_overlap": self.sequential_match_overlap,
            "extracted_images": len(image_names),
            "attempted_image_pairs": frame_processing_statistics[
                "attempted_pairs"
            ],
            "verified_image_pairs": frame_processing_statistics[
                "verified_pairs"
            ],
            "registered_images": reconstruction.num_reg_images(),
            "landmarks": len(global_map["positions"]),
            "candidate_landmarks": global_map["candidate_landmarks"],
            "occupied_grid_cells": global_map["occupied_grid_cells"],
            "landmarks_per_occupied_cell": global_map[
                "landmarks_per_occupied_cell"
            ],
            "alignment_frames": alignment["alignment_frames"],
            "alignment_rmse_mm": alignment["alignment_rmse_mm"],
            "reference_image": alignment["reference_image"],
            "last_mapping_image": global_map["last_mapping_image"],
            "track_length_statistics": track_length_statistics,
            "mapping_pipeline_diagnostics_csv": (
                mapping_diagnostics_csv_path.name
            ),
            "mapping_pipeline_diagnostics_plot": (
                mapping_diagnostics_plot_path.name
            ),
            "imu_gravity": imu_gravity_summary,
            "timing": timing,
        }
        with (output_dir / "map_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)
        with (output_dir / "map_timing.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(timing, file, indent=2)
        with (output_dir / "track_length_statistics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(track_length_statistics, file, indent=2)
        print(f"Saved frozen 3D map: {map_path}")
        print(
            "Saved track statistics: "
            f"{output_dir / 'track_length_statistics.json'}"
        )
        print(
            f"Map: {summary['landmarks']}/"
            f"{summary['candidate_landmarks']} selected landmarks | "
            f"{summary['occupied_grid_cells']} occupied grid cells | "
            f"{summary['registered_images']}/{summary['extracted_images']} "
            "registered images"
        )
        print_track_length_statistics(track_length_statistics)
        print_map_build_timing(timing)
        return global_map
