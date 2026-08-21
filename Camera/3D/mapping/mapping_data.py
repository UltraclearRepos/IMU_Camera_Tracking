from dataclasses import dataclass
from typing import Any

import numpy as np


FeatureData = dict[str, np.ndarray]
ArucoPose = tuple[np.ndarray, np.ndarray, dict[str, Any]]


@dataclass(frozen=True)
class LocalTrackSnapshot:
    track_ids: np.ndarray
    positions: np.ndarray
    continued_track_count: int
    new_track_count: int

    def __len__(self):
        return len(self.track_ids)


@dataclass
class MappingImage:
    frame_index: int
    name: str
    database_image_id: int
    timestamp_s: float
    features: FeatureData
    local_tracks: LocalTrackSnapshot
    aruco_pose: ArucoPose | None


@dataclass(frozen=True)
class KeyframePairCandidate:
    image: MappingImage
    shared_track_count: int
    overlap: float
    median_displacement_px: float
    reason: str
    motion_target_px: float | None


@dataclass(frozen=True)
class KeyframePairSelection:
    pairs: tuple[KeyframePairCandidate, ...]
    active_candidate_count: int

    @property
    def recent_pair_count(self):
        return sum(pair.reason == "recent" for pair in self.pairs)

    @property
    def motion_pair_count(self):
        return sum(pair.reason == "motion_target" for pair in self.pairs)

@dataclass
class MappingFrameDiagnostics:
    frame_index: int
    timestamp_s: float
    image_name: str
    feature_count: int
    continued_track_count: int
    new_track_count: int
    active_pair_candidates: int
    recent_pair_count: int
    motion_pair_count: int
    maximum_selected_motion_px: float
    minimum_selected_overlap: float
    attempted_pairs: int
    raw_matches: int
    verified_pairs: int
    verified_inliers: int
    aruco_detected: bool
    aruco_reprojection_rms_px: float
    aruco_reprojection_max_px: float
    registered: bool = False
    aruco_alignment_used: bool = False
    triangulated_observations: int = 0
    triangulated_feature_ratio: float = 0.0
    median_point_track_length: float = np.nan
    median_point_reprojection_error_px: float = np.nan
    camera_translation_step_mm: float = np.nan
    camera_rotation_step_deg: float = np.nan
    aruco_alignment_residual_mm: float = np.nan


@dataclass(frozen=True)
class FrameCollectionTiming:
    setup_seconds: float
    frame_read_seconds: float
    image_save_seconds: float
    feature_extraction_seconds: float
    local_tracking_seconds: float
    aruco_detection_seconds: float
    image_database_write_seconds: float
    pair_selection_seconds: float
    feature_matching_seconds: float
    geometry_verification_seconds: float
    pair_database_write_seconds: float
    wall_seconds: float


@dataclass(frozen=True)
class FramePairingResult:
    attempted_pair_count: int
    raw_match_count: int
    verified_pair_count: int
    verified_inlier_count: int
    matching_seconds: float
    geometry_verification_seconds: float
    database_write_seconds: float


@dataclass
class MappingFrameCollection:
    images: list[MappingImage]
    frame_diagnostics: list[MappingFrameDiagnostics]
    attempted_pair_count: int
    verified_pair_count: int
    timing: FrameCollectionTiming
    imu_gravity_summary: dict[str, Any] | None

    @property
    def image_count(self):
        return len(self.images)


@dataclass(frozen=True)
class ArucoAlignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rmse_mm: float
    candidate_frame_count: int
    aligned_frame_count: int
    reprojection_rms_threshold_px: float
    reference_image_name: str
    center_residuals_by_image: dict[str, float]


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
    reconstruction_seconds: float
    alignment_seconds: float
    map_finalization_seconds: float
    map_saving_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class MapBuildConfiguration:
    feature_type: str
    start_frame: int
    end_frame: int
    reconstruction_method: str
    frame_step: int
    recent_pair_count: int
    motion_targets_px: tuple[float, ...]
    minimum_new_track_distance_px: float
    maximum_active_track_count: int
    maximum_forward_backward_error_px: float
    minimum_keyframe_overlap: float
    maximum_motion_anchor_px: float
