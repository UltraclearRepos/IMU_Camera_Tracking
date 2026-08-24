from pathlib import Path

import numpy as np

from mapping.mapping_data import (
    ArucoAlignment,
    FrozenMap,
    LandmarkAppearance,
    LandmarkCandidates,
    LandmarkSelection,
    MapFinalizationResult,
    MappingFrameCollection,
    MappingTrajectory,
)


class GlobalMapBuilder:
    """Build a metric map in the last registered mapping-camera frame."""

    MINIMUM_TRACK_LENGTH = 3
    MAXIMUM_REPROJECTION_ERROR_PX = 3.0

    def __init__(
        self,
        feature_matcher,
        maximum_landmarks,
        grid_rows,
        grid_columns,
        reprojection_error_weight,
    ):
        self.feature_matcher = feature_matcher
        self.maximum_landmarks = maximum_landmarks
        self.grid_rows = grid_rows
        self.grid_columns = grid_columns
        self.reprojection_error_weight = reprojection_error_weight

    def build(
        self,
        reconstruction,
        frame_collection: MappingFrameCollection,
        alignment: ArucoAlignment,
    ):
        # ArUco supplies the metric scale only.  The final map axes and
        # origin come from the last camera registered by the reconstruction.
        last_registered_image = self._last_registered_image(reconstruction)
        millimeters_per_colmap_unit = alignment.scale
        candidates = self._collect_candidates(
            reconstruction,
            millimeters_per_colmap_unit,
            last_registered_image,
        )
        selection = self._select_landmarks(candidates)
        appearance = self._build_landmark_appearance(
            selection.colmap_points,
            reconstruction,
            frame_collection,
        )
        trajectory = self._build_mapping_trajectory(
            reconstruction,
            frame_collection,
            millimeters_per_colmap_unit,
            last_registered_image,
        )
        features_by_frame = {}
        feature_bounds_by_frame = {}
        feature_contours_by_frame = {}
        for image in frame_collection.images:
            features = image.features
            if (
                features.selection_bounds is None
                or features.selection_contour is None
            ):
                raise RuntimeError(
                    "Mapping features require selection bounds and contour"
                )
            features_by_frame[image.frame_index] = features.keypoints.copy()
            feature_bounds_by_frame[image.frame_index] = (
                features.selection_bounds.copy()
            )
            feature_contours_by_frame[image.frame_index] = (
                features.selection_contour.copy()
            )

        frozen_map = FrozenMap(
            positions=selection.positions,
            descriptors=appearance.descriptors,
            scores=appearance.scores,
            candidate_positions=candidates.positions,
            candidate_available_frames=candidates.available_frames,
            selected_candidate_indices=selection.candidate_indices.astype(
                np.int32
            ),
            mapping_frames=trajectory.frames,
            mapping_times_s=trajectory.timestamps_s,
            mapping_feature_keypoints=tuple(
                features_by_frame[int(frame)]
                for frame in trajectory.frames
            ),
            mapping_feature_bounds=np.asarray(
                [
                    feature_bounds_by_frame[int(frame)]
                    for frame in trajectory.frames
                ],
                dtype=np.int32,
            ),
            mapping_feature_contours=tuple(
                feature_contours_by_frame[int(frame)]
                for frame in trajectory.frames
            ),
            mapping_camera_positions=trajectory.camera_positions,
            mapping_camera_rotations=trajectory.camera_rotations,
            mapping_camera_headings=trajectory.camera_headings,
            mapping_reference_frame=self._frame_number(
                last_registered_image.name
            ),
            coordinate_frame="last_registered_mapping_camera",
            mapping_extracted_image_count=frame_collection.image_count,
            occupied_grid_cell_count=selection.occupied_grid_cell_count,
            scales=appearance.scales,
            orientations=appearance.orientations,
        )
        return MapFinalizationResult(
            frozen_map=frozen_map,
            candidate_track_lengths=candidates.track_lengths,
            selected_track_lengths=selection.track_lengths,
        )

    def save(self, frozen_map: FrozenMap, output_path):
        arrays = {
            "positions": frozen_map.positions,
            "descriptors": frozen_map.descriptors,
            "scores": frozen_map.scores,
            "candidate_positions": frozen_map.candidate_positions,
            "candidate_available_frames": (
                frozen_map.candidate_available_frames
            ),
            "selected_candidate_indices": (
                frozen_map.selected_candidate_indices
            ),
            "mapping_frames": frozen_map.mapping_frames,
            "mapping_times_s": frozen_map.mapping_times_s,
            "mapping_feature_bounds": frozen_map.mapping_feature_bounds,
            "mapping_camera_positions": frozen_map.mapping_camera_positions,
            "mapping_camera_rotations": frozen_map.mapping_camera_rotations,
            "mapping_camera_headings": frozen_map.mapping_camera_headings,
            "mapping_reference_frame": frozen_map.mapping_reference_frame,
            "coordinate_frame": frozen_map.coordinate_frame,
            "mapping_extracted_image_count": (
                frozen_map.mapping_extracted_image_count
            ),
            "occupied_grid_cell_count": (
                frozen_map.occupied_grid_cell_count
            ),
        }
        if frozen_map.scales is not None:
            arrays["scales"] = frozen_map.scales
        if frozen_map.orientations is not None:
            arrays["orientations"] = frozen_map.orientations
        np.savez_compressed(output_path, **arrays)

    def _collect_candidates(
        self,
        reconstruction,
        millimeters_per_colmap_unit,
        last_registered_image,
    ):
        colmap_points = []
        colmap_landmark_positions = []
        track_lengths = []
        reprojection_errors = []
        available_frames = []

        for point in reconstruction.points3D.values():
            if point.track.length() < self.MINIMUM_TRACK_LENGTH:
                continue
            if point.error > self.MAXIMUM_REPROJECTION_ERROR_PX:
                continue

            observation_frames = sorted(
                self._frame_number(reconstruction.images[observation.image_id].name)
                for observation in point.track.elements
            )
            colmap_points.append(point)
            colmap_landmark_positions.append(point.xyz)
            track_lengths.append(point.track.length())
            reprojection_errors.append(point.error)
            available_frames.append(observation_frames[self.MINIMUM_TRACK_LENGTH - 1])

        if not colmap_landmark_positions:
            raise RuntimeError(
                "No reconstructed landmark passed the track-length and "
                "reprojection-error quality gates"
            )

        map_landmark_positions = (
            self._transform_colmap_landmarks_to_last_camera_frame(
                np.asarray(colmap_landmark_positions, dtype=np.float64),
                millimeters_per_colmap_unit,
                last_registered_image,
            )
        )
        return LandmarkCandidates(
            colmap_points=colmap_points,
            positions=map_landmark_positions,
            track_lengths=np.asarray(track_lengths, dtype=np.int32),
            reprojection_errors=np.asarray(
                reprojection_errors,
                dtype=np.float32,
            ),
            available_frames=np.asarray(available_frames, dtype=np.int32),
        )

    def _select_landmarks(self, candidates: LandmarkCandidates):
        positions = candidates.positions
        minimum_xy = np.min(positions[:, :2], axis=0)
        extent_xy = np.maximum(
            np.ptp(positions[:, :2], axis=0),
            np.finfo(float).eps,
        )
        normalized_xy = (positions[:, :2] - minimum_xy) / extent_xy
        columns = np.minimum(
            (normalized_xy[:, 0] * self.grid_columns).astype(int),
            self.grid_columns - 1,
        )
        rows = np.minimum(
            (normalized_xy[:, 1] * self.grid_rows).astype(int),
            self.grid_rows - 1,
        )
        landmark_quality = self._landmark_quality(candidates)

        ranked_indices_by_cell = []
        for row in range(self.grid_rows):
            for column in range(self.grid_columns):
                cell_indices = np.flatnonzero(
                    (rows == row) & (columns == column)
                )
                if len(cell_indices) == 0:
                    continue
                quality_order = np.argsort(
                    landmark_quality[cell_indices]
                )[::-1]
                ranked_indices_by_cell.append(cell_indices[quality_order])

        selected_indices = []
        cell_rank = 0
        while len(selected_indices) < self.maximum_landmarks:
            added_landmark = False
            for cell_indices in ranked_indices_by_cell:
                if cell_rank < len(cell_indices):
                    selected_indices.append(cell_indices[cell_rank])
                    added_landmark = True
                    if len(selected_indices) == self.maximum_landmarks:
                        break
            if not added_landmark:
                break
            cell_rank += 1

        selected_indices = np.asarray(selected_indices, dtype=int)
        occupied_cell_count = len(ranked_indices_by_cell)
        return LandmarkSelection(
            candidate_indices=selected_indices,
            colmap_points=[
                candidates.colmap_points[index] for index in selected_indices
            ],
            positions=candidates.positions[selected_indices],
            track_lengths=candidates.track_lengths[selected_indices],
            occupied_grid_cell_count=occupied_cell_count,
        )

    def _landmark_quality(self, candidates):
        track_lengths = candidates.track_lengths
        errors = candidates.reprojection_errors
        track_quality = (
            track_lengths - np.min(track_lengths)
        ) / max(np.ptp(track_lengths), 1)
        reprojection_quality = 1.0 - (
            errors - np.min(errors)
        ) / max(np.ptp(errors), np.finfo(float).eps)
        return (
            self.reprojection_error_weight * reprojection_quality
            + (1.0 - self.reprojection_error_weight) * track_quality
        )

    def _build_landmark_appearance(
        self,
        selected_points,
        reconstruction,
        frame_collection,
    ):
        mapping_images = {
            image.name: image for image in frame_collection.images
        }
        descriptors = []
        scores = []
        scales = []
        orientations = []
        include_scale_orientation = (
            self.feature_matcher.requires_scale_orientation
        )

        for point in selected_points:
            observation_descriptors = []
            observation_scores = []
            observation_scales = []
            observation_orientations = []
            for observation in point.track.elements:
                registered_image = reconstruction.images[observation.image_id]
                features = mapping_images[registered_image.name].features
                feature_index = observation.point2D_idx
                observation_descriptors.append(
                    features.descriptors[feature_index]
                )
                observation_scores.append(features.scores[feature_index])
                if include_scale_orientation:
                    if (
                        features.scales is None
                        or features.orientations is None
                    ):
                        raise RuntimeError(
                            "SIFT mapping features require scales and "
                            "orientations"
                        )
                    observation_scales.append(
                        features.scales[feature_index]
                    )
                    observation_orientations.append(
                        features.orientations[feature_index]
                    )

            descriptor = np.mean(observation_descriptors, axis=0)
            descriptor /= np.linalg.norm(descriptor)
            descriptors.append(descriptor)
            scores.append(np.mean(observation_scores))
            if include_scale_orientation:
                scales.append(np.mean(observation_scales))
                orientations.append(
                    np.arctan2(
                        np.mean(np.sin(observation_orientations)),
                        np.mean(np.cos(observation_orientations)),
                    )
                )

        return LandmarkAppearance(
            descriptors=np.asarray(descriptors, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            scales=(
                np.asarray(scales, dtype=np.float32)
                if include_scale_orientation
                else None
            ),
            orientations=(
                np.asarray(orientations, dtype=np.float32)
                if include_scale_orientation
                else None
            ),
        )

    def _build_mapping_trajectory(
        self,
        reconstruction,
        frame_collection,
        millimeters_per_colmap_unit,
        last_registered_image,
    ):
        mapping_images = {
            image.name: image for image in frame_collection.images
        }
        frames = []
        timestamps_s = []
        camera_positions = []
        camera_rotations = []
        camera_headings = []

        registered_images = sorted(
            reconstruction.images.values(),
            key=lambda image: self._frame_number(image.name),
        )
        for image in registered_images:
            map_to_camera_rotation, map_to_camera_translation = (
                self._map_to_camera_pose(
                    image,
                    millimeters_per_colmap_unit,
                    last_registered_image,
                )
            )
            camera_to_map_rotation = map_to_camera_rotation.T
            frames.append(self._frame_number(image.name))
            timestamps_s.append(mapping_images[image.name].timestamp_s)
            camera_positions.append(
                self._camera_center(
                    map_to_camera_rotation,
                    map_to_camera_translation,
                )
            )
            camera_rotations.append(camera_to_map_rotation)
            camera_headings.append(
                camera_to_map_rotation @ np.array([0.0, -1.0, 0.0])
            )

        return MappingTrajectory(
            frames=np.asarray(frames, dtype=np.int32),
            timestamps_s=np.asarray(timestamps_s, dtype=np.float64),
            camera_positions=np.asarray(camera_positions, dtype=np.float64),
            camera_rotations=np.asarray(camera_rotations, dtype=np.float64),
            camera_headings=np.asarray(camera_headings, dtype=np.float64),
        )

    @staticmethod
    def _frame_number(image_name):
        return int(Path(image_name).stem.rsplit("_", 1)[1])

    def _last_registered_image(self, reconstruction):
        if not reconstruction.images:
            raise RuntimeError(
                "The reconstruction does not contain registered images"
            )
        return max(
            reconstruction.images.values(),
            key=lambda image: self._frame_number(image.name),
        )

    @staticmethod
    def _transform_colmap_landmarks_to_last_camera_frame(
        colmap_landmark_positions,
        millimeters_per_colmap_unit,
        last_registered_image,
    ):
        last_pose = last_registered_image.cam_from_world()
        last_rotation = last_pose.rotation.matrix()
        last_translation = np.asarray(last_pose.translation).reshape(3)
        return millimeters_per_colmap_unit * (
            (last_rotation @ colmap_landmark_positions.T).T
            + last_translation
        )

    @staticmethod
    def _map_to_camera_pose(
        image,
        millimeters_per_colmap_unit,
        last_registered_image,
    ):
        image_pose = image.cam_from_world()
        last_pose = last_registered_image.cam_from_world()
        image_rotation = image_pose.rotation.matrix()
        image_translation = np.asarray(image_pose.translation).reshape(3)
        last_rotation = last_pose.rotation.matrix()
        last_translation = np.asarray(last_pose.translation).reshape(3)

        map_to_camera_rotation = image_rotation @ last_rotation.T
        map_to_camera_translation = millimeters_per_colmap_unit * (
            image_translation
            - map_to_camera_rotation @ last_translation
        )
        return map_to_camera_rotation, map_to_camera_translation

    @staticmethod
    def _camera_center(world_to_camera_rotation, world_to_camera_translation):
        return (
            -world_to_camera_rotation.T
            @ world_to_camera_translation.reshape(3)
        )
