from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MappingImage:
    frame_index: int
    name: str
    database_image_id: int
    timestamp_s: float


@dataclass
class MappingFrameCollection:
    images: list[MappingImage]
    imu_gravity_summary: dict[str, Any] | None

    @property
    def image_count(self):
        return len(self.images)


@dataclass(frozen=True)
class ArucoAlignment:
    scale: float
    rmse_mm: float
    candidate_frame_count: int
    aligned_frame_count: int
    reprojection_rms_threshold_px: float


@dataclass
class LandmarkCandidates:
    colmap_points: list[Any]
    positions: np.ndarray
    track_lengths: np.ndarray
    reprojection_errors: np.ndarray
    available_frames: np.ndarray


@dataclass
class LandmarkSelection:
    candidate_indices: np.ndarray
    colmap_points: list[Any]
    positions: np.ndarray
    track_lengths: np.ndarray
    occupied_grid_cell_count: int


@dataclass
class LandmarkAppearance:
    descriptors: np.ndarray
    scores: np.ndarray
    scales: np.ndarray | None
    orientations: np.ndarray | None


@dataclass
class MappingTrajectory:
    frames: np.ndarray
    timestamps_s: np.ndarray
    camera_positions: np.ndarray
    camera_rotations: np.ndarray
    camera_headings: np.ndarray


@dataclass
class FrozenMap:
    positions: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    candidate_positions: np.ndarray
    candidate_available_frames: np.ndarray
    selected_candidate_indices: np.ndarray
    mapping_frames: np.ndarray
    mapping_times_s: np.ndarray
    mapping_feature_keypoints: tuple[np.ndarray, ...]
    mapping_feature_bounds: np.ndarray
    mapping_feature_contours: tuple[np.ndarray, ...]
    mapping_camera_positions: np.ndarray
    mapping_camera_rotations: np.ndarray
    mapping_camera_headings: np.ndarray
    mapping_reference_frame: int
    coordinate_frame: str
    mapping_extracted_image_count: int
    occupied_grid_cell_count: int
    scales: np.ndarray | None = None
    orientations: np.ndarray | None = None


@dataclass
class MapFinalizationResult:
    frozen_map: FrozenMap
    candidate_track_lengths: np.ndarray
    selected_track_lengths: np.ndarray


@dataclass(frozen=True)
class MapBuildDurations:
    frame_collection_seconds: float
    reconstruction_seconds: float
    alignment_seconds: float
    map_finalization_seconds: float
    map_saving_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class MapBuildConfiguration:
    mapping_feature_type: str
    start_frame: int
    end_frame: int
    reconstruction_method: str
    keyframe_interval: int
    maximum_features: int
    sequential_overlap: int
    matcher_type: str
    loop_detection: bool
    loop_detection_period: int
    vocabulary_tree_path: str | None
