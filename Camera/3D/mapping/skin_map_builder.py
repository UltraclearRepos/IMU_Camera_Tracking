import shutil
import time
from pathlib import Path

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
        skin_mask_provider,
        mapping_start_frame,
        mapping_end_frame,
        keyframe_interval,
        maximum_features,
        sequential_overlap,
        loop_detection,
        loop_detection_period,
        vocabulary_tree_path,
        maximum_global_landmarks,
        global_map_grid_rows,
        global_map_grid_columns,
        global_map_reprojection_error_weight,
        imu_gravity_provider=None,
    ):
        vocabulary_tree_path = (
            None
            if vocabulary_tree_path is None
            else Path(vocabulary_tree_path)
        )
        self.configuration = MapBuildConfiguration(
            mapping_feature_type="sift",
            start_frame=mapping_start_frame,
            end_frame=mapping_end_frame,
            reconstruction_method="global",
            keyframe_interval=keyframe_interval,
            maximum_features=maximum_features,
            sequential_overlap=sequential_overlap,
            matcher_type="SIFT_LIGHTGLUE",
            loop_detection=bool(loop_detection),
            loop_detection_period=loop_detection_period,
            vocabulary_tree_path=(
                None
                if vocabulary_tree_path is None
                else str(vocabulary_tree_path)
            ),
        )
        self.frame_builder = MappingFrameBuilder(
            camera_matrix=camera_matrix,
            distortion=distortion,
            skin_mask_provider=skin_mask_provider,
            start_frame=mapping_start_frame,
            end_frame=mapping_end_frame,
            keyframe_interval=keyframe_interval,
            maximum_features=maximum_features,
            sequential_overlap=sequential_overlap,
            loop_detection=loop_detection,
            loop_detection_period=loop_detection_period,
            vocabulary_tree_path=vocabulary_tree_path,
            imu_gravity_provider=imu_gravity_provider,
        )
        self.reconstructor = SfmReconstructor(
            minimum_pair_inliers=(
                MappingFrameBuilder.MINIMUM_PAIR_INLIERS
            ),
            use_gravity_prior=imu_gravity_provider is not None,
        )
        self.aruco_aligner = ArucoMapAligner(camera_matrix, distortion)
        self.global_map_builder = GlobalMapBuilder(
            maximum_landmarks=maximum_global_landmarks,
            grid_rows=global_map_grid_rows,
            grid_columns=global_map_grid_columns,
            reprojection_error_weight=global_map_reprojection_error_weight,
        )
        self.diagnostics = MapBuildDiagnostics()

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
        masks_directory = work_directory / "masks"
        sparse_directory = work_directory / "sparse"
        database_path = work_directory / "database.db"
        collection_started = time.perf_counter()
        frame_collection = self.frame_builder.build(
            video_path,
            images_directory,
            masks_directory,
            database_path,
        )
        frame_collection_seconds = time.perf_counter() - collection_started
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
            images_directory,
        )
        alignment_seconds = time.perf_counter() - alignment_started

        finalization_started = time.perf_counter()
        finalization = self.global_map_builder.build(
            reconstruction,
            frame_collection,
            alignment,
            database_path,
            masks_directory,
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
            frame_collection_seconds=frame_collection_seconds,
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
        masks_directory = work_directory / "masks"
        sparse_directory = work_directory / "sparse"
        images_directory.mkdir(parents=True)
        masks_directory.mkdir()
        sparse_directory.mkdir()
        return work_directory

    @staticmethod
    def _validate_imu_collection(frame_collection):
        summary = frame_collection.imu_gravity_summary
        if summary is not None and summary["counts"]["accepted"] == 0:
            raise RuntimeError(
                "No mapping frame passed the IMU gravity quality gates"
            )
