import json

import numpy as np


class MapBuildDiagnostics:
    """Save compact post-reconstruction mapping summaries."""

    def save_report(
        self,
        configuration,
        frame_collection,
        reconstruction,
        alignment,
        frozen_map,
        finalization,
        durations,
        video_path,
        output_directory,
        diagnostics_directory,
        map_path,
    ):
        del diagnostics_directory
        track_statistics = self._track_statistics(
            reconstruction,
            finalization,
        )
        timing = {
            "frame_collection_s": durations.frame_collection_seconds,
            "glomap_reconstruction_s": durations.reconstruction_seconds,
            "aruco_scale_estimation_s": durations.alignment_seconds,
            "map_finalization_s": durations.map_finalization_seconds,
            "map_saving_s": durations.map_saving_seconds,
            "total_map_build_wall_time_s": durations.total_seconds,
        }
        summary = {
            "recording": video_path.stem,
            "mapping_feature_type": configuration.mapping_feature_type,
            "mapping_start_frame": configuration.start_frame,
            "mapping_end_frame": configuration.end_frame,
            "reconstruction_method": configuration.reconstruction_method,
            "colmap_mapping": {
                "keyframe_interval": configuration.keyframe_interval,
                "maximum_features": configuration.maximum_features,
                "sequential_overlap": configuration.sequential_overlap,
                "matcher_type": configuration.matcher_type,
                "loop_detection": configuration.loop_detection,
                "loop_detection_period": (
                    configuration.loop_detection_period
                ),
                "vocabulary_tree_path": configuration.vocabulary_tree_path,
            },
            "coordinate_frame": frozen_map.coordinate_frame,
            "extracted_images": frame_collection.image_count,
            "registered_images": reconstruction.num_reg_images(),
            "selected_landmarks": len(frozen_map.positions),
            "candidate_landmarks": len(frozen_map.candidate_positions),
            "occupied_grid_cells": frozen_map.occupied_grid_cell_count,
            "scale_candidate_frames": alignment.candidate_frame_count,
            "scale_frames": alignment.aligned_frame_count,
            "scale_reprojection_rms_threshold_px": (
                alignment.reprojection_rms_threshold_px
            ),
            "scale_pairwise_distance_rmse_mm": alignment.rmse_mm,
            "last_mapping_image": max(
                image.name for image in reconstruction.images.values()
            ),
            "track_length_statistics": track_statistics,
            "imu_gravity": frame_collection.imu_gravity_summary,
            "timing": timing,
        }
        self._write_json(output_directory / "map_summary.json", summary)
        self._write_json(output_directory / "map_timing.json", timing)
        self._write_json(
            output_directory / "track_length_statistics.json",
            track_statistics,
        )
        print(
            f"Map: {len(frozen_map.positions)}/"
            f"{len(frozen_map.candidate_positions)} selected landmarks | "
            f"{reconstruction.num_reg_images()}/"
            f"{frame_collection.image_count} registered images | "
            f"saved to {map_path}"
        )

    def _track_statistics(self, reconstruction, finalization):
        return {
            "all_reconstructed_points": self._summarize_track_lengths(
                [
                    point.track.length()
                    for point in reconstruction.points3D.values()
                ]
            ),
            "quality_filtered_candidates": self._summarize_track_lengths(
                finalization.candidate_track_lengths
            ),
            "selected_frozen_map": self._summarize_track_lengths(
                finalization.selected_track_lengths
            ),
        }

    @staticmethod
    def _summarize_track_lengths(track_lengths):
        track_lengths = np.asarray(track_lengths, dtype=int)
        if len(track_lengths) == 0:
            return {
                "landmarks": 0,
                "minimum": 0,
                "mean": np.nan,
                "median": np.nan,
                "percentile_90": np.nan,
                "maximum": 0,
            }
        return {
            "landmarks": int(len(track_lengths)),
            "minimum": int(np.min(track_lengths)),
            "mean": float(np.mean(track_lengths)),
            "median": float(np.median(track_lengths)),
            "percentile_90": float(np.percentile(track_lengths, 90)),
            "maximum": int(np.max(track_lengths)),
        }

    @staticmethod
    def _write_json(path, document):
        with path.open("w", encoding="utf-8") as file:
            json.dump(document, file, indent=2, allow_nan=True)
