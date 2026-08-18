import time

import cv2
import numpy as np
import pycolmap

from mapping.aruco_reference import create_aruco_detector, detect_aruco_pose
from mapping.mapping_data import (
    FrameCollectionTiming,
    FramePairingResult,
    MappingFrameCollection,
    MappingFrameDiagnostics,
    MappingImage,
)


class MappingFrameBuilder:
    """Extract mapping images and build the verified image-pair graph."""

    MINIMUM_PAIR_MATCHES = 20
    MINIMUM_PAIR_INLIERS = 20

    def __init__(
        self,
        camera_matrix,
        distortion,
        feature_matcher,
        start_frame,
        end_frame,
        frame_step,
        sequential_match_overlap,
        maximum_features,
        feature_grid_rows,
        feature_grid_columns,
        imu_gravity_provider=None,
    ):
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.feature_matcher = feature_matcher
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_step = frame_step
        self.sequential_match_overlap = sequential_match_overlap
        self.maximum_features = maximum_features
        self.feature_grid_rows = feature_grid_rows
        self.feature_grid_columns = feature_grid_columns
        self.imu_gravity_provider = imu_gravity_provider
        self.aruco_detector = create_aruco_detector()

    def build(self, video_path, images_directory, database_path):
        collection_started = time.perf_counter()
        setup_started = time.perf_counter()
        capture = cv2.VideoCapture(str(video_path))
        database = pycolmap.Database.open(database_path)

        try:
            camera = self._create_colmap_camera(capture)
            camera.camera_id = database.write_camera(camera)
            camera_sensor, rig_id = self._create_gravity_rig(
                database,
                camera.camera_id,
            )
            setup_seconds = time.perf_counter() - setup_started

            images = []
            frame_diagnostics = []
            geometry_options = pycolmap.TwoViewGeometryOptions()
            geometry_options.ransac.random_seed = 0

            attempted_pair_count = 0
            verified_pair_count = 0
            frame_read_seconds = 0.0
            image_save_seconds = 0.0
            feature_extraction_seconds = 0.0
            aruco_detection_seconds = 0.0
            image_database_write_seconds = 0.0
            feature_matching_seconds = 0.0
            geometry_verification_seconds = 0.0
            pair_database_write_seconds = 0.0

            self._skip_frames_before_mapping(capture)
            for frame_index in range(self.start_frame, self.end_frame + 1):
                read_started = time.perf_counter()
                success, frame = capture.read()
                frame_read_seconds += time.perf_counter() - read_started
                if not success:
                    break
                if frame_index % self.frame_step != 0:
                    continue

                image_name = f"frame_{frame_index:06d}.png"
                image_save_seconds += self._save_image(
                    frame,
                    images_directory / image_name,
                )

                extraction_started = time.perf_counter()
                features = self._extract_features(frame)
                feature_extraction_seconds += (
                    time.perf_counter() - extraction_started
                )

                timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                gravity, imu_status = self._gravity_for_timestamp(timestamp_s)

                aruco_started = time.perf_counter()
                aruco_pose = self._detect_aruco_pose(frame)
                aruco_detection_seconds += (
                    time.perf_counter() - aruco_started
                )

                database_write_started = time.perf_counter()
                database_image_id = self._write_mapping_image(
                    database,
                    camera,
                    image_name,
                    len(images) + 1,
                    camera_sensor,
                    rig_id,
                    gravity,
                )
                database.write_keypoints(
                    database_image_id,
                    features["keypoints"],
                )
                image_database_write_seconds += (
                    time.perf_counter() - database_write_started
                )

                current_image = MappingImage(
                    frame_index=frame_index,
                    name=image_name,
                    database_image_id=database_image_id,
                    timestamp_s=timestamp_s,
                    features=features,
                    aruco_pose=aruco_pose,
                )
                previous_images = images[-self.sequential_match_overlap :]
                pairing = self._match_previous_images(
                    database,
                    camera,
                    previous_images,
                    current_image,
                    geometry_options,
                )
                attempted_pair_count += pairing.attempted_pair_count
                verified_pair_count += pairing.verified_pair_count
                feature_matching_seconds += pairing.matching_seconds
                geometry_verification_seconds += (
                    pairing.geometry_verification_seconds
                )
                pair_database_write_seconds += pairing.database_write_seconds

                images.append(current_image)
                frame_diagnostics.append(
                    self._create_frame_diagnostics(current_image, pairing)
                )
                print(
                    f"Mapping frame processing: frame {frame_index}/"
                    f"{self.end_frame} | "
                    f"features: {len(features['keypoints'])} | "
                    f"verified pairs: {verified_pair_count}{imu_status}"
                )

            timing = FrameCollectionTiming(
                setup_seconds=setup_seconds,
                frame_read_seconds=frame_read_seconds,
                image_save_seconds=image_save_seconds,
                feature_extraction_seconds=feature_extraction_seconds,
                aruco_detection_seconds=aruco_detection_seconds,
                image_database_write_seconds=image_database_write_seconds,
                feature_matching_seconds=feature_matching_seconds,
                geometry_verification_seconds=geometry_verification_seconds,
                pair_database_write_seconds=pair_database_write_seconds,
                wall_seconds=time.perf_counter() - collection_started,
            )
            imu_summary = (
                None
                if self.imu_gravity_provider is None
                else self.imu_gravity_provider.summary()
            )
            return MappingFrameCollection(
                images=images,
                frame_diagnostics=frame_diagnostics,
                attempted_pair_count=attempted_pair_count,
                verified_pair_count=verified_pair_count,
                timing=timing,
                imu_gravity_summary=imu_summary,
            )
        finally:
            capture.release()
            database.close()

    def _skip_frames_before_mapping(self, capture):
        for _ in range(self.start_frame):
            success, _ = capture.read()
            if not success:
                break

    @staticmethod
    def _save_image(frame, output_path):
        started = time.perf_counter()
        cv2.imwrite(str(output_path), frame)
        return time.perf_counter() - started

    def _extract_features(self, frame):
        detected_features = self.feature_matcher.extract(frame)
        return self._select_spatially_distributed_features(detected_features)

    def _select_spatially_distributed_features(self, features):
        keypoints = features["keypoints"]
        scores = features["scores"]
        width, height = features["image_size"]
        roi_top = features["roi_top"]

        columns = np.minimum(
            (keypoints[:, 0] * self.feature_grid_columns / width).astype(int),
            self.feature_grid_columns - 1,
        )
        rows = np.minimum(
            (
                (keypoints[:, 1] - roi_top)
                * self.feature_grid_rows
                / (height - roi_top)
            ).astype(int),
            self.feature_grid_rows - 1,
        )

        indices_by_cell = []
        for row in range(self.feature_grid_rows):
            for column in range(self.feature_grid_columns):
                cell_indices = np.flatnonzero(
                    (rows == row) & (columns == column)
                )
                score_order = np.argsort(scores[cell_indices])[::-1]
                indices_by_cell.append(cell_indices[score_order])

        selected_indices = []
        cell_rank = 0
        while len(selected_indices) < self.maximum_features:
            added_feature = False
            for cell_indices in indices_by_cell:
                if cell_rank < len(cell_indices):
                    selected_indices.append(cell_indices[cell_rank])
                    added_feature = True
                    if len(selected_indices) == self.maximum_features:
                        break
            if not added_feature:
                break
            cell_rank += 1

        selected_indices = np.asarray(selected_indices, dtype=int)
        selected = {
            "keypoints": features["keypoints"][selected_indices],
            "descriptors": features["descriptors"][selected_indices],
            "scores": features["scores"][selected_indices],
            "image_size": features["image_size"],
            "roi_top": roi_top,
        }
        for field_name in ("scales", "oris"):
            if field_name in features:
                selected[field_name] = features[field_name][selected_indices]
        return selected

    def _match_previous_images(
        self,
        database,
        camera,
        previous_images,
        current_image,
        geometry_options,
    ):
        raw_match_count = 0
        verified_pair_count = 0
        verified_inlier_count = 0
        matching_seconds = 0.0
        geometry_seconds = 0.0
        database_write_seconds = 0.0

        for previous_image in previous_images:
            matching_started = time.perf_counter()
            matches = self.feature_matcher.match(
                previous_image.features,
                current_image.features,
            )
            matching_seconds += time.perf_counter() - matching_started
            raw_match_count += len(matches)
            if len(matches) < self.MINIMUM_PAIR_MATCHES:
                continue

            geometry_started = time.perf_counter()
            geometry = pycolmap.estimate_two_view_geometry(
                camera,
                previous_image.features["keypoints"],
                camera,
                current_image.features["keypoints"],
                matches,
                geometry_options,
            )
            geometry_seconds += time.perf_counter() - geometry_started
            if len(geometry.inlier_matches) < self.MINIMUM_PAIR_INLIERS:
                continue

            verified_pair_count += 1
            verified_inlier_count += len(geometry.inlier_matches)
            writing_started = time.perf_counter()
            database.write_two_view_geometry(
                previous_image.database_image_id,
                current_image.database_image_id,
                geometry,
            )
            database_write_seconds += time.perf_counter() - writing_started

        return FramePairingResult(
            attempted_pair_count=len(previous_images),
            raw_match_count=raw_match_count,
            verified_pair_count=verified_pair_count,
            verified_inlier_count=verified_inlier_count,
            matching_seconds=matching_seconds,
            geometry_verification_seconds=geometry_seconds,
            database_write_seconds=database_write_seconds,
        )

    def _create_frame_diagnostics(self, image, pairing):
        aruco_quality = (
            None if image.aruco_pose is None else image.aruco_pose[2]
        )
        return MappingFrameDiagnostics(
            frame_index=image.frame_index,
            timestamp_s=image.timestamp_s,
            image_name=image.name,
            feature_count=len(image.features["keypoints"]),
            attempted_pairs=pairing.attempted_pair_count,
            raw_matches=pairing.raw_match_count,
            verified_pairs=pairing.verified_pair_count,
            verified_inliers=pairing.verified_inlier_count,
            aruco_detected=image.aruco_pose is not None,
            aruco_reprojection_rms_px=(
                np.nan
                if aruco_quality is None
                else aruco_quality["reprojection_rms_px"]
            ),
            aruco_reprojection_max_px=(
                np.nan
                if aruco_quality is None
                else aruco_quality["reprojection_max_px"]
            ),
        )

    def _detect_aruco_pose(self, frame):
        return detect_aruco_pose(
            frame,
            self.camera_matrix,
            self.distortion,
            self.aruco_detector,
        )

    def _gravity_for_timestamp(self, timestamp_s):
        if self.imu_gravity_provider is None:
            return None, ""

        gravity, diagnostics = (
            self.imu_gravity_provider.gravity_at_video_time(timestamp_s)
        )
        status = (
            f" | IMU gravity: {diagnostics['reason']}"
            f" (|a|={diagnostics.get('acceleration_magnitude_m_s2', np.nan):.3f} m/s^2, "
            f"|w|={np.degrees(diagnostics.get('gyroscope_magnitude_rad_s', np.nan)):.2f} deg/s)"
        )
        return gravity, status

    def _create_colmap_camera(self, capture):
        distortion = self.distortion.reshape(-1)
        distortion = np.pad(
            distortion,
            (0, max(0, 8 - len(distortion))),
        )
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        k1, k2, p1, p2, k3, k4, k5, k6 = distortion[:8]
        camera = pycolmap.Camera(
            model="FULL_OPENCV",
            width=round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            params=np.array(
                [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6],
                dtype=float,
            ),
        )
        camera.has_prior_focal_length = True
        return camera

    def _create_gravity_rig(self, database, camera_id):
        if self.imu_gravity_provider is None:
            return None, None

        camera_sensor = pycolmap.sensor_t(
            pycolmap.SensorType.CAMERA,
            camera_id,
        )
        rig = pycolmap.Rig()
        rig.add_ref_sensor(camera_sensor)
        return camera_sensor, database.write_rig(rig)

    @staticmethod
    def _write_mapping_image(
        database,
        camera,
        image_name,
        image_id,
        camera_sensor,
        rig_id,
        gravity,
    ):
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
