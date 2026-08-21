import time

import cv2
import numpy as np
import pycolmap

from mapping.aruco_reference import create_aruco_detector, detect_aruco_pose
from mapping.mapping_data import (
    FrameCollectionTiming,
    MappingFrameCollection,
    MappingFrameDiagnostics,
    MappingImage,
)


class MappingFrameBuilder:
    """Build one COLMAP database incrementally from fixed-interval frames."""

    MINIMUM_PAIR_INLIERS = 20

    def __init__(
        self,
        camera_matrix,
        distortion,
        skin_mask_provider,
        start_frame,
        end_frame,
        keyframe_interval,
        maximum_features,
        sequential_overlap,
        vocabulary_tree_path,
        imu_gravity_provider=None,
    ):
        if keyframe_interval <= 0:
            raise ValueError("keyframe_interval must be positive")
        if maximum_features <= 0:
            raise ValueError("maximum_features must be positive")
        if sequential_overlap <= 0:
            raise ValueError("sequential_overlap must be positive")

        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.skin_mask_provider = skin_mask_provider
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.keyframe_interval = keyframe_interval
        self.maximum_features = maximum_features
        self.sequential_overlap = sequential_overlap
        self.vocabulary_tree_path = vocabulary_tree_path
        self.imu_gravity_provider = imu_gravity_provider
        self.aruco_detector = create_aruco_detector()

    def build(
        self,
        video_path,
        images_directory,
        masks_directory,
        database_path,
    ):
        self._validate_vocabulary_tree()
        collection_started = time.perf_counter()
        setup_started = time.perf_counter()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open mapping video: {video_path}")

        database = pycolmap.Database.open(database_path)
        try:
            camera = self._create_colmap_camera(capture)
            camera.camera_id = database.write_camera(camera)
            camera_sensor, rig_id = self._create_gravity_rig(
                database,
                camera.camera_id,
            )
        finally:
            database.close()
        setup_seconds = time.perf_counter() - setup_started

        images = []
        frame_diagnostics = []
        frame_read_seconds = 0.0
        image_save_seconds = 0.0
        mask_generation_seconds = 0.0
        feature_extraction_seconds = 0.0
        aruco_detection_seconds = 0.0
        image_database_write_seconds = 0.0
        sequential_matching_seconds = 0.0

        try:
            self._skip_frames_before_mapping(capture)
            for frame_index in range(self.start_frame, self.end_frame + 1):
                read_started = time.perf_counter()
                success, frame = capture.read()
                frame_read_seconds += time.perf_counter() - read_started
                if not success:
                    break
                if (frame_index - self.start_frame) % self.keyframe_interval:
                    continue

                image_name = f"frame_{frame_index:06d}.png"
                image_save_seconds += self._save_image(
                    frame,
                    images_directory / image_name,
                )

                mask_started = time.perf_counter()
                roi_top, mapping_mask = self._mapping_mask(frame)
                self._save_mask(mapping_mask, masks_directory / image_name)
                mask_generation_seconds += time.perf_counter() - mask_started

                timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                gravity, imu_status = self._gravity_for_timestamp(timestamp_s)

                aruco_started = time.perf_counter()
                aruco_pose = self._detect_aruco_pose(frame)
                aruco_detection_seconds += time.perf_counter() - aruco_started

                database_write_started = time.perf_counter()
                image_id = self._write_image_to_database(
                    database_path,
                    camera,
                    image_name,
                    len(images) + 1,
                    camera_sensor,
                    rig_id,
                    gravity,
                )
                image_database_write_seconds += (
                    time.perf_counter() - database_write_started
                )

                extraction_started = time.perf_counter()
                self._extract_new_image_features(
                    database_path,
                    images_directory,
                    masks_directory,
                    image_name,
                    camera.camera_id,
                )
                feature_extraction_seconds += (
                    time.perf_counter() - extraction_started
                )

                features = self._read_features(
                    database_path,
                    image_id,
                    frame.shape,
                    roi_top,
                    mapping_mask,
                )
                pairs_before = self._database_pair_statistics(database_path)
                matching_started = time.perf_counter()
                self._match_database_sequentially(database_path)
                sequential_matching_seconds += (
                    time.perf_counter() - matching_started
                )
                pairs_after = self._database_pair_statistics(database_path)
                pair_delta = {
                    key: pairs_after[key] - pairs_before[key]
                    for key in pairs_after
                }

                current_image = MappingImage(
                    frame_index=frame_index,
                    name=image_name,
                    database_image_id=image_id,
                    timestamp_s=timestamp_s,
                    features=features,
                    aruco_pose=aruco_pose,
                )
                images.append(current_image)
                frame_diagnostics.append(
                    self._create_frame_diagnostics(current_image, pair_delta)
                )
                print(
                    f"Mapping keyframe: frame {frame_index}/"
                    f"{self.end_frame} | SIFT features: "
                    f"{len(features['keypoints'])} | new sequential pairs: "
                    f"{pair_delta['matched_pairs']} | new verified pairs: "
                    f"{pair_delta['verified_pairs']}{imu_status}"
                )
        finally:
            capture.release()

        final_pair_statistics = self._database_pair_statistics(database_path)
        timing = FrameCollectionTiming(
            setup_seconds=setup_seconds,
            frame_read_seconds=frame_read_seconds,
            image_save_seconds=image_save_seconds,
            mask_generation_seconds=mask_generation_seconds,
            feature_extraction_seconds=feature_extraction_seconds,
            aruco_detection_seconds=aruco_detection_seconds,
            image_database_write_seconds=image_database_write_seconds,
            sequential_matching_seconds=sequential_matching_seconds,
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
            matched_pair_count=final_pair_statistics["matched_pairs"],
            verified_pair_count=final_pair_statistics["verified_pairs"],
            timing=timing,
            imu_gravity_summary=imu_summary,
        )

    def _validate_vocabulary_tree(self):
        if self.vocabulary_tree_path is None:
            raise RuntimeError(
                "COLMAP loop detection requires a vocabulary tree. Set "
                "COLMAP_VOCAB_TREE_PATH or pass vocabulary_tree_path."
            )
        if not self.vocabulary_tree_path.is_file():
            raise FileNotFoundError(
                "COLMAP vocabulary tree was not found: "
                f"{self.vocabulary_tree_path}. Set COLMAP_VOCAB_TREE_PATH "
                "to a downloaded COLMAP vocabulary-tree .bin file."
            )

    def _skip_frames_before_mapping(self, capture):
        for _ in range(self.start_frame):
            success, _ = capture.read()
            if not success:
                break

    @staticmethod
    def _save_image(frame, output_path):
        started = time.perf_counter()
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not save mapping image: {output_path}")
        return time.perf_counter() - started

    @staticmethod
    def _save_mask(mask, output_path):
        if not cv2.imwrite(str(output_path), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"Could not save mapping mask: {output_path}")

    def _mapping_mask(self, frame):
        height, width = frame.shape[:2]
        roi_top = round(
            height
            * (1.0 - self.skin_mask_provider.feature_roi_bottom_fraction)
        )
        roi_mask = np.zeros((height, width), dtype=bool)
        roi_mask[roi_top:] = True
        skin_mask = self.skin_mask_provider.adaptive_skin_mask(frame, roi_top)
        if skin_mask is None:
            skin_mask = np.ones((height, width), dtype=bool)
        return roi_top, roi_mask & skin_mask.astype(bool)

    def _write_image_to_database(
        self,
        database_path,
        camera,
        image_name,
        image_id,
        camera_sensor,
        rig_id,
        gravity,
    ):
        database = pycolmap.Database.open(database_path)
        try:
            return self._write_mapping_image(
                database,
                camera,
                image_name,
                image_id,
                camera_sensor,
                rig_id,
                gravity,
            )
        finally:
            database.close()

    def _extract_new_image_features(
        self,
        database_path,
        images_directory,
        masks_directory,
        image_name,
        camera_id,
    ):
        database = pycolmap.Database.open(database_path)
        try:
            image_id = database.read_image_with_name(image_name).image_id
            if database.exists_keypoints(image_id) or database.exists_descriptors(
                image_id
            ):
                raise RuntimeError(
                    f"Features already exist for new keyframe {image_name}"
                )
        finally:
            database.close()

        reader_options = pycolmap.ImageReaderOptions(
            existing_camera_id=camera_id,
            mask_path=masks_directory,
        )
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.max_image_size = -1
        extraction_options.sift.max_num_features = self.maximum_features
        extraction_options.sift.max_num_orientations = 1
        # The frozen-map tracker consumes unit-normalized SIFT descriptors.
        extraction_options.sift.normalization = pycolmap.Normalization.L2
        pycolmap.extract_features(
            database_path=database_path,
            image_path=images_directory,
            image_names=[image_name],
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=pycolmap.Device.auto,
        )

    def _match_database_sequentially(self, database_path):
        pairing_options = pycolmap.SequentialPairingOptions(
            overlap=self.sequential_overlap,
            quadratic_overlap=False,
            loop_detection=True,
            loop_detection_period=1,
            vocab_tree_path=self.vocabulary_tree_path,
        )
        matching_options = pycolmap.FeatureMatchingOptions(
            skip_geometric_verification=False,
        )
        verification_options = pycolmap.TwoViewGeometryOptions()
        verification_options.min_num_inliers = self.MINIMUM_PAIR_INLIERS
        verification_options.ransac.random_seed = 0
        pycolmap.match_sequential(
            database_path=database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            verification_options=verification_options,
            device=pycolmap.Device.auto,
        )

    @staticmethod
    def _database_pair_statistics(database_path):
        database = pycolmap.Database.open(database_path)
        try:
            return {
                "matched_pairs": database.num_matched_image_pairs(),
                "raw_matches": database.num_matches(),
                "verified_pairs": database.num_verified_image_pairs(),
                "verified_inliers": database.num_inlier_matches(),
            }
        finally:
            database.close()

    @staticmethod
    def _read_features(
        database_path,
        image_id,
        frame_shape,
        roi_top,
        mapping_mask,
    ):
        database = pycolmap.Database.open(database_path)
        try:
            colmap_keypoints = database.read_keypoints(image_id).copy()
            descriptors = (
                database.read_descriptors(image_id).to_float().data.copy()
            )
        finally:
            database.close()

        descriptor_norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors /= np.maximum(descriptor_norms, np.finfo(np.float32).eps)
        scales, orientations = MappingFrameBuilder._scales_and_orientations(
            colmap_keypoints
        )
        mask_y, mask_x = np.nonzero(mapping_mask)
        if len(mask_x):
            selection_bounds = np.array(
                [
                    mask_x.min(),
                    mask_y.min(),
                    mask_x.max() + 1,
                    mask_y.max() + 1,
                ],
                dtype=np.int32,
            )
        else:
            selection_bounds = np.array(
                [0, roi_top, frame_shape[1], frame_shape[0]],
                dtype=np.int32,
            )

        return {
            "keypoints": colmap_keypoints[:, :2].astype(np.float32),
            "descriptors": descriptors.astype(np.float32),
            "scores": np.ones(len(colmap_keypoints), dtype=np.float32),
            "scales": scales,
            "oris": orientations,
            "image_size": np.array(
                [frame_shape[1], frame_shape[0]], dtype=np.float32
            ),
            "roi_top": roi_top,
            "selection_bounds": selection_bounds,
            "selection_contour": MappingFrameBuilder._selection_contour(
                mapping_mask
            ),
        }

    @staticmethod
    def _scales_and_orientations(keypoints):
        if keypoints.shape[1] >= 6:
            affine = keypoints[:, 2:6].reshape(-1, 2, 2)
            scales = np.sqrt(np.abs(np.linalg.det(affine)))
            orientations = np.arctan2(affine[:, 1, 0], affine[:, 0, 0])
        elif keypoints.shape[1] >= 4:
            scales = keypoints[:, 2]
            orientations = keypoints[:, 3]
        else:
            scales = np.ones(len(keypoints), dtype=np.float32)
            orientations = np.zeros(len(keypoints), dtype=np.float32)
        return scales.astype(np.float32), orientations.astype(np.float32)

    @staticmethod
    def _selection_contour(selection_mask):
        contours, _ = cv2.findContours(
            selection_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return np.empty((0, 1, 2), dtype=np.int32)
        contour = max(contours, key=cv2.contourArea)
        return cv2.approxPolyDP(contour, 1.0, True)

    @staticmethod
    def _create_frame_diagnostics(image, pair_delta):
        aruco_quality = None if image.aruco_pose is None else image.aruco_pose[2]
        return MappingFrameDiagnostics(
            frame_index=image.frame_index,
            timestamp_s=image.timestamp_s,
            image_name=image.name,
            feature_count=len(image.features["keypoints"]),
            matched_pairs=pair_delta["matched_pairs"],
            raw_matches=pair_delta["raw_matches"],
            verified_pairs=pair_delta["verified_pairs"],
            verified_inliers=pair_delta["verified_inliers"],
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
        distortion = np.pad(distortion, (0, max(0, 8 - len(distortion))))
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

        camera_sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
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
