import shutil
import time
from pathlib import Path

from mapping.adaptive_keyframe_pair_selector import (
    AdaptiveKeyframePairSelector,
)
from mapping.aruco_map_aligner import ArucoMapAligner
from mapping.global_map_builder import GlobalMapBuilder
from mapping.map_build_diagnostics import MapBuildDiagnostics
from mapping.mapping_data import (
    MapBuildConfiguration,
    MapBuildDurations,
)
from mapping.mapping_frame_builder import MappingFrameBuilder
from mapping.sfm_reconstructor import SfmReconstructor


class SkinMapBuilder:
    """Coordinate mapping stages without owning their implementation details."""

    def __init__(
        self,
        camera_matrix,
        distortion,
        feature_matcher,
        mapping_start_frame,
        mapping_end_frame,
        reconstruction_method,
        mapping_frame_step,
        recent_pair_count,
        motion_targets_px,
        mapping_maximum_features,
        mapping_feature_grid_rows,
        mapping_feature_grid_columns,
        maximum_global_landmarks,
        global_map_grid_rows,
        global_map_grid_columns,
        global_map_reprojection_error_weight,
        imu_gravity_provider=None,
    ):
        self.configuration = MapBuildConfiguration(
            feature_type=feature_matcher.feature_type,
            start_frame=mapping_start_frame,
            end_frame=mapping_end_frame,
            reconstruction_method=reconstruction_method,
            frame_step=mapping_frame_step,
            recent_pair_count=recent_pair_count,
            motion_targets_px=tuple(motion_targets_px),
            minimum_new_track_distance_px=(
                AdaptiveKeyframePairSelector.MINIMUM_NEW_TRACK_DISTANCE_PX
            ),
            maximum_active_track_count=(
                AdaptiveKeyframePairSelector.MAXIMUM_ACTIVE_TRACK_COUNT
            ),
            maximum_forward_backward_error_px=(
                AdaptiveKeyframePairSelector.MAXIMUM_FORWARD_BACKWARD_ERROR_PX
            ),
            minimum_keyframe_overlap=(
                AdaptiveKeyframePairSelector.MINIMUM_KEYFRAME_OVERLAP
            ),
            maximum_motion_anchor_px=(
                AdaptiveKeyframePairSelector.MAXIMUM_MOTION_ANCHOR_PX
            ),
        )
        self.frame_builder = MappingFrameBuilder(
            camera_matrix=camera_matrix,
            distortion=distortion,
            feature_matcher=feature_matcher,
            start_frame=mapping_start_frame,
            end_frame=mapping_end_frame,
            frame_step=mapping_frame_step,
            recent_pair_count=recent_pair_count,
            motion_targets_px=motion_targets_px,
            maximum_features=mapping_maximum_features,
            feature_grid_rows=mapping_feature_grid_rows,
            feature_grid_columns=mapping_feature_grid_columns,
            imu_gravity_provider=imu_gravity_provider,
        )
        self.reconstructor = SfmReconstructor(
            reconstruction_method=reconstruction_method,
            minimum_pair_inliers=(
                MappingFrameBuilder.MINIMUM_PAIR_INLIERS
            ),
            use_gravity_prior=imu_gravity_provider is not None,
        )
        self.aruco_aligner = ArucoMapAligner()
        self.global_map_builder = GlobalMapBuilder(
            feature_matcher=feature_matcher,
            maximum_landmarks=maximum_global_landmarks,
            grid_rows=global_map_grid_rows,
            grid_columns=global_map_grid_columns,
            reprojection_error_weight=(
                global_map_reprojection_error_weight
            ),
            aruco_aligner=self.aruco_aligner,
        )
        self.diagnostics = MapBuildDiagnostics(self.aruco_aligner)

    def build(self, video_path, output_directory, diagnostics_output_dir=None):
        build_started = time.perf_counter()
        video_path = Path(video_path)
        output_directory = Path(output_directory)
        diagnostics_directory = Path(
            diagnostics_output_dir
            if diagnostics_output_dir is not None
            else output_directory
        )
        diagnostics_directory.mkdir(parents=True, exist_ok=True)

        work_directory = self._prepare_work_directory(output_directory)
        images_directory = work_directory / "images"
        sparse_directory = work_directory / "sparse"
        database_path = work_directory / "database.db"
        frame_collection = self.frame_builder.build(
            video_path,
            images_directory,
            database_path,
        )
        self._validate_imu_collection(frame_collection)

        reconstruction_started = time.perf_counter()
        reconstruction = self.reconstructor.reconstruct(
            database_path,
            images_directory,
            sparse_directory,
        )
        reconstruction_seconds = time.perf_counter() - reconstruction_started

        alignment_started = time.perf_counter()
        alignment = self.aruco_aligner.align(
            reconstruction,
            frame_collection,
        )
        alignment_seconds = time.perf_counter() - alignment_started

        self.diagnostics.enrich_frame_metrics(
            frame_collection,
            reconstruction,
            alignment,
        )

        finalization_started = time.perf_counter()
        finalization = self.global_map_builder.build(
            reconstruction,
            frame_collection,
            alignment,
        )
        frozen_map = finalization.frozen_map
        map_finalization_seconds = time.perf_counter() - finalization_started

        saving_started = time.perf_counter()
        map_path = output_directory / "global_map.npz"
        self.global_map_builder.save(frozen_map, map_path)
        model_directory = output_directory / "colmap_model"
        model_directory.mkdir(exist_ok=True)
        reconstruction.write(model_directory)
        map_saving_seconds = time.perf_counter() - saving_started

        durations = MapBuildDurations(
            reconstruction_seconds=reconstruction_seconds,
            alignment_seconds=alignment_seconds,
            map_finalization_seconds=map_finalization_seconds,
            map_saving_seconds=map_saving_seconds,
            total_seconds=time.perf_counter() - build_started,
        )
        self.diagnostics.save_report(
            configuration=self.configuration,
            frame_collection=frame_collection,
            reconstruction=reconstruction,
            alignment=alignment,
            frozen_map=frozen_map,
            finalization=finalization,
            durations=durations,
            video_path=video_path,
            output_directory=output_directory,
            diagnostics_directory=diagnostics_directory,
            map_path=map_path,
        )
        return frozen_map

    @staticmethod
    def _prepare_work_directory(output_directory):
        output_directory = output_directory.resolve()
        work_directory = (output_directory / "colmap_work").resolve()
        if work_directory.parent != output_directory:
            raise RuntimeError(
                f"Invalid mapping work directory: {work_directory}"
            )
        if work_directory.exists():
            shutil.rmtree(work_directory)

        images_directory = work_directory / "images"
        sparse_directory = work_directory / "sparse"
        images_directory.mkdir(parents=True)
        sparse_directory.mkdir()
        return work_directory

    @staticmethod
    def _validate_imu_collection(frame_collection):
        summary = frame_collection.imu_gravity_summary
        if summary is not None and summary["counts"]["accepted"] == 0:
            raise RuntimeError(
                "No mapping frame passed the IMU gravity quality gates"
            )
