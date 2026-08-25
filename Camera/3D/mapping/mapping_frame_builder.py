import time

import cv2
import numpy as np
import pycolmap

from mapping.adaptive_keyframe_pair_selector import (
    AdaptiveKeyframePairSelector,
)
from mapping.aruco_reference import create_aruco_detector, detect_aruco_pose
from mapping.mapping_data import (
    FeatureSet,
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
        keyframe_interval,
        every_frame_until_frame,
        every_frame_from_frame,
        recent_pair_count,
        motion_targets_px,
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
        self.keyframe_interval = keyframe_interval
        self.every_frame_until_frame = every_frame_until_frame
        self.every_frame_from_frame = every_frame_from_frame
        self.recent_pair_count = recent_pair_count
        self.motion_targets_px = tuple(motion_targets_px)
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
            pair_selector = AdaptiveKeyframePairSelector(
                self.recent_pair_count,
                self.motion_targets_px,
            )
            geometry_options = pycolmap.TwoViewGeometryOptions()
            geometry_options.ransac.random_seed = 0

            attempted_pair_count = 0
            verified_pair_count = 0
            frame_read_seconds = 0.0
            image_save_seconds = 0.0
            feature_extraction_seconds = 0.0
            local_tracking_seconds = 0.0
            aruco_detection_seconds = 0.0
            image_database_write_seconds = 0.0
            pair_selection_seconds = 0.0
            feature_matching_seconds = 0.0
            geometry_verification_seconds = 0.0
            pair_database_write_seconds = 0.0

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0:
                raise RuntimeError("Mapping video does not report a valid FPS")

            while True:
                read_started = time.perf_counter()
                success, frame = capture.read()
                frame_read_seconds += time.perf_counter() - read_started
                if not success:
                    break

                timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_index = round(timestamp_s * fps)
                if frame_index < self.start_frame:
                    continue
                if frame_index > self.end_frame:
                    break
                if not self._is_keyframe(frame_index):
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
                tracking_started = time.perf_counter()
                local_tracks = pair_selector.update_tracks(
                    frame,
                    features.keypoints,
                )
                local_tracking_seconds += (
                    time.perf_counter() - tracking_started
                )

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
                    features.keypoints,
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
                    local_tracks=local_tracks,
                    aruco_pose=aruco_pose,
                )
                selection_started = time.perf_counter()
                pair_selector.register_keyframe(current_image)
                pair_selection = pair_selector.select_pairs(current_image)
                pair_selection_seconds += (
                    time.perf_counter() - selection_started
                )
                pairing = self._match_previous_images(
                    database,
                    camera,
                    pair_selection,
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
                    self._create_frame_diagnostics(
                        current_image,
                        pair_selection,
                        pairing,
                    )
                )
                print(
                    f"Mapping frame processing: frame {frame_index}/"
                    f"{self.end_frame} | "
                    f"features: {len(features.keypoints)} | "
                    f"selected pairs: {len(pair_selection.pairs)} | "
                    f"verified pairs: {verified_pair_count}{imu_status}"
                )

            timing = FrameCollectionTiming(
                setup_seconds=setup_seconds,
                frame_read_seconds=frame_read_seconds,
                image_save_seconds=image_save_seconds,
                feature_extraction_seconds=feature_extraction_seconds,
                local_tracking_seconds=local_tracking_seconds,
                aruco_detection_seconds=aruco_detection_seconds,
                image_database_write_seconds=image_database_write_seconds,
                pair_selection_seconds=pair_selection_seconds,
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

    def _is_keyframe(self, frame_index):
        if (
            self.every_frame_until_frame is not None
            and (
                frame_index
                <= self.start_frame + self.every_frame_until_frame
            )
        ):
            return True
        if (
            self.every_frame_from_frame is not None
            and (
                frame_index
                >= self.end_frame - self.every_frame_from_frame
            )
        ):
            return True
        return (frame_index - self.start_frame) % self.keyframe_interval == 0

    @staticmethod
    def _save_image(frame, output_path):
        started = time.perf_counter()
        cv2.imwrite(str(output_path), frame)
        return time.perf_counter() - started

    def _extract_features(self, frame):
        detected_features = self.feature_matcher.extract(frame)
        return self._select_spatially_distributed_features(detected_features)

    def _select_spatially_distributed_features(self, features):
        keypoints = features.keypoints
        scores = features.scores
        width, height = features.image_size
        roi_top = features.roi_top
        selection_mask = features.selection_mask
        if features.selection_bounds is None:
            selection_left = 0
            selection_top = roi_top
            selection_right = width
            selection_bottom = height
        else:
            (
                selection_left,
                selection_top,
                selection_right,
                selection_bottom,
            ) = features.selection_bounds

        columns = np.minimum(
            (
                (keypoints[:, 0] - selection_left)
                * self.feature_grid_columns
                / (selection_right - selection_left)
            ).astype(int),
            self.feature_grid_columns - 1,
        )
        rows = np.minimum(
            (
                (keypoints[:, 1] - selection_top)
                * self.feature_grid_rows
                / (selection_bottom - selection_top)
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
        return FeatureSet(
            keypoints=features.keypoints[selected_indices],
            descriptors=features.descriptors[selected_indices],
            scores=features.scores[selected_indices],
            image_size=features.image_size,
            roi_top=roi_top,
            scales=(
                None
                if features.scales is None
                else features.scales[selected_indices]
            ),
            orientations=(
                None
                if features.orientations is None
                else features.orientations[selected_indices]
            ),
            selection_mask=selection_mask,
            selection_bounds=np.array(
                [
                    selection_left,
                    selection_top,
                    selection_right,
                    selection_bottom,
                ],
                dtype=np.int32,
            ),
            selection_contour=self._selection_contour(selection_mask),
        )

    @staticmethod
    def _selection_contour(selection_mask):
        if selection_mask is None:
            return np.empty((0, 1, 2), dtype=np.int32)
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
    def _empty_selected_features(features):
        return FeatureSet(
            keypoints=features.keypoints[:0],
            descriptors=features.descriptors[:0],
            scores=features.scores[:0],
            image_size=features.image_size,
            roi_top=features.roi_top,
            scales=(
                None if features.scales is None else features.scales[:0]
            ),
            orientations=(
                None
                if features.orientations is None
                else features.orientations[:0]
            ),
            selection_mask=features.selection_mask,
            selection_bounds=(
                np.array(
                    [
                        0,
                        features.roi_top,
                        features.image_size[0],
                        features.image_size[1],
                    ],
                    dtype=np.int32,
                )
                if features.selection_bounds is None
                else features.selection_bounds.copy()
            ),
            selection_contour=np.empty((0, 1, 2), dtype=np.int32),
        )

    def _match_previous_images(
        self,
        database,
        camera,
        pair_selection,
        current_image,
        geometry_options,
    ):
        raw_match_count = 0
        verified_pair_count = 0
        verified_inlier_count = 0
        matching_seconds = 0.0
        geometry_seconds = 0.0
        database_write_seconds = 0.0

        for selected_pair in pair_selection.pairs:
            previous_image = selected_pair.image
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
                previous_image.features.keypoints,
                camera,
                current_image.features.keypoints,
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
            attempted_pair_count=len(pair_selection.pairs),
            raw_match_count=raw_match_count,
            verified_pair_count=verified_pair_count,
            verified_inlier_count=verified_inlier_count,
            matching_seconds=matching_seconds,
            geometry_verification_seconds=geometry_seconds,
            database_write_seconds=database_write_seconds,
        )

    def _create_frame_diagnostics(
        self,
        image,
        pair_selection,
        pairing,
    ):
        aruco_pose = image.aruco_pose
        selected_motions = [
            pair.median_displacement_px
            for pair in pair_selection.pairs
            if np.isfinite(pair.median_displacement_px)
        ]
        selected_overlaps = [
            pair.overlap
            for pair in pair_selection.pairs
        ]
        return MappingFrameDiagnostics(
            frame_index=image.frame_index,
            timestamp_s=image.timestamp_s,
            image_name=image.name,
            feature_count=len(image.features.keypoints),
            continued_track_count=(
                image.local_tracks.continued_track_count
            ),
            new_track_count=image.local_tracks.new_track_count,
            active_pair_candidates=(
                pair_selection.active_candidate_count
            ),
            recent_pair_count=pair_selection.recent_pair_count,
            motion_pair_count=pair_selection.motion_pair_count,
            maximum_selected_motion_px=(
                max(selected_motions) if selected_motions else np.nan
            ),
            minimum_selected_overlap=(
                min(selected_overlaps) if selected_overlaps else np.nan
            ),
            attempted_pairs=pairing.attempted_pair_count,
            raw_matches=pairing.raw_match_count,
            verified_pairs=pairing.verified_pair_count,
            verified_inliers=pairing.verified_inlier_count,
            aruco_detected=aruco_pose is not None,
            aruco_reprojection_rms_px=(
                np.nan
                if aruco_pose is None
                else aruco_pose.reprojection_rms_px
            ),
            aruco_reprojection_max_px=(
                np.nan
                if aruco_pose is None
                else aruco_pose.reprojection_max_px
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
