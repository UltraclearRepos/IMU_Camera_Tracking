import csv
import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from mapping.mapping_data import (
    ArucoAlignment,
    FrozenMap,
    MapBuildConfiguration,
    MapBuildDurations,
    MapFinalizationResult,
    MappingFrameCollection,
)


class MapBuildDiagnostics:
    """Collect, save, plot, and print mapping diagnostics."""

    def enrich_frame_metrics(
        self,
        frame_collection: MappingFrameCollection,
        reconstruction,
        alignment: ArucoAlignment,
    ):
        registered_images = {
            image.name: image for image in reconstruction.images.values()
        }
        camera_steps = self._camera_steps(reconstruction, alignment.scale)

        for metrics in frame_collection.frame_diagnostics:
            image = registered_images.get(metrics.image_name)
            metrics.registered = image is not None
            if image is None:
                continue

            points = [
                reconstruction.points3D[point.point3D_id]
                for point in image.points2D
                if point.has_point3D()
            ]
            metrics.triangulated_observations = len(points)
            metrics.triangulated_feature_ratio = (
                len(points) / metrics.feature_count
                if metrics.feature_count
                else 0.0
            )
            if points:
                metrics.median_point_track_length = float(
                    np.median([point.track.length() for point in points])
                )
                metrics.median_point_reprojection_error_px = float(
                    np.median([point.error for point in points])
                )
            (
                metrics.camera_translation_step_mm,
                metrics.camera_rotation_step_deg,
            ) = camera_steps[metrics.image_name]

    def save_report(
        self,
        configuration: MapBuildConfiguration,
        frame_collection: MappingFrameCollection,
        reconstruction,
        alignment: ArucoAlignment,
        frozen_map: FrozenMap,
        finalization: MapFinalizationResult,
        durations: MapBuildDurations,
        video_path,
        output_directory,
        diagnostics_directory,
        map_path,
    ):
        diagnostics_csv_path = (
            diagnostics_directory / "mapping_pipeline_diagnostics.csv"
        )
        diagnostics_plot_path = (
            diagnostics_directory / "mapping_pipeline_diagnostics.png"
        )
        pairing_plot_path = (
            diagnostics_directory / "mapping_pairing_diagnostics.png"
        )
        self._save_frame_csv(
            frame_collection,
            diagnostics_csv_path,
        )
        self._save_frame_plot(
            frame_collection,
            diagnostics_plot_path,
            video_path.stem,
        )
        self._save_pairing_plot(
            frame_collection,
            pairing_plot_path,
            video_path.stem,
        )

        track_statistics = self._track_statistics(
            reconstruction,
            finalization,
        )
        timing = self._timing_document(frame_collection, durations)
        summary = self._summary_document(
            configuration,
            frame_collection,
            reconstruction,
            alignment,
            frozen_map,
            track_statistics,
            timing,
            diagnostics_csv_path,
            diagnostics_plot_path,
            pairing_plot_path,
        )

        self._write_json(output_directory / "map_summary.json", summary)
        self._write_json(output_directory / "map_timing.json", timing)
        self._write_json(
            output_directory / "track_length_statistics.json",
            track_statistics,
        )
        self._print_summary(
            frozen_map,
            reconstruction,
            frame_collection,
            track_statistics,
            timing,
            map_path,
            output_directory,
        )

    def _camera_steps(self, reconstruction, metric_scale):
        camera_steps = {}
        previous_position = None
        previous_rotation = None
        registered_images = sorted(
            reconstruction.images.values(),
            key=lambda image: image.name,
        )
        for image in registered_images:
            position = metric_scale * image.projection_center()
            camera_rotation = image.cam_from_world().rotation.matrix().T
            if previous_position is None:
                translation_step = np.nan
                rotation_step = np.nan
            else:
                translation_step = float(
                    np.linalg.norm(position - previous_position)
                )
                rotation_step = float(
                    np.degrees(
                        Rotation.from_matrix(
                            previous_rotation.T @ camera_rotation
                        ).magnitude()
                    )
                )
            camera_steps[image.name] = (translation_step, rotation_step)
            previous_position = position
            previous_rotation = camera_rotation
        return camera_steps

    @staticmethod
    def _save_frame_csv(frame_collection, output_path):
        rows = [
            asdict(metrics) for metrics in frame_collection.frame_diagnostics
        ]
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _save_frame_plot(self, frame_collection, output_path, recording_name):
        metrics = frame_collection.frame_diagnostics
        frames = self._field(metrics, "frame_index")
        feature_count = self._field(metrics, "feature_count")
        raw_matches = self._field(metrics, "raw_matches")
        verified_inliers = self._field(metrics, "verified_inliers")
        matched_pairs = self._field(metrics, "matched_pairs")
        verified_pairs = self._field(metrics, "verified_pairs")
        registered = self._field(metrics, "registered").astype(bool)
        triangulated = self._field(metrics, "triangulated_observations")
        triangulated_ratio = self._field(
            metrics,
            "triangulated_feature_ratio",
        )
        track_length = self._field(metrics, "median_point_track_length")
        point_reprojection = self._field(
            metrics,
            "median_point_reprojection_error_px",
        )
        camera_translation_step = self._field(
            metrics,
            "camera_translation_step_mm",
        )
        camera_rotation_step = self._field(
            metrics,
            "camera_rotation_step_deg",
        )
        aruco_detected = self._field(metrics, "aruco_detected").astype(bool)
        aruco_rms = self._field(metrics, "aruco_reprojection_rms_px")

        matches_per_pair = raw_matches / np.maximum(matched_pairs, 1)
        inliers_per_pair = verified_inliers / np.maximum(verified_pairs, 1)
        pair_acceptance = verified_pairs / np.maximum(matched_pairs, 1)

        figure, axes = plt.subplots(6, 1, figsize=(16, 19), sharex=True)
        axes[0].plot(frames, feature_count, color="tab:blue")
        axes[0].set_ylabel("Selected features")
        axes[0].set_title("1. Spatial feature selection")
        axes[0].grid(True)

        axes[1].plot(frames, matches_per_pair, label="Matches / matched pair")
        axes[1].plot(frames, inliers_per_pair, label="Inliers / accepted pair")
        axes[1].set_ylabel("Correspondences")
        axes[1].set_title("2. Matching and two-view verification")
        axes[1].grid(True)
        axes[1].legend(loc="upper left")
        acceptance_axis = axes[1].twinx()
        acceptance_axis.plot(
            frames,
            100.0 * pair_acceptance,
            color="tab:green",
            alpha=0.7,
            label="Accepted pair ratio",
        )
        acceptance_axis.set_ylabel("Accepted pairs [%]")
        acceptance_axis.set_ylim(0.0, 105.0)
        acceptance_axis.legend(loc="upper right")

        axes[2].plot(frames, triangulated, label="Triangulated observations")
        axes[2].scatter(
            frames[~registered],
            np.zeros(np.sum(~registered)),
            color="tab:red",
            s=24,
            label="Image not registered",
            zorder=3,
        )
        axes[2].set_ylabel("3D observations")
        axes[2].set_title("3. Registration and triangulation")
        axes[2].grid(True)
        axes[2].legend(loc="upper left")
        triangulation_axis = axes[2].twinx()
        triangulation_axis.plot(
            frames,
            100.0 * triangulated_ratio,
            color="tab:orange",
            label="Features assigned to 3D",
        )
        triangulation_axis.set_ylabel("Triangulated features [%]")
        triangulation_axis.set_ylim(0.0, 105.0)
        triangulation_axis.legend(loc="upper right")

        axes[3].plot(frames, track_length, label="Median track length")
        axes[3].set_ylabel("Images / landmark")
        axes[3].set_title("4. Landmark support and reprojection quality")
        axes[3].grid(True)
        axes[3].legend(loc="upper left")
        reprojection_axis = axes[3].twinx()
        reprojection_axis.plot(
            frames,
            point_reprojection,
            color="tab:red",
            label="Median reprojection error",
        )
        reprojection_axis.set_ylabel("Reprojection error [px]")
        reprojection_axis.legend(loc="upper right")

        axes[4].plot(
            frames,
            camera_translation_step,
            label="Consecutive camera-center distance",
        )
        axes[4].set_ylabel("Translation [mm]")
        axes[4].set_title("5. Recovered camera motion")
        axes[4].grid(True)
        axes[4].legend(loc="upper left")
        rotation_axis = axes[4].twinx()
        rotation_axis.plot(
            frames,
            camera_rotation_step,
            color="tab:orange",
            label="Consecutive camera rotation",
        )
        rotation_axis.set_ylabel("Rotation [deg]")
        rotation_axis.legend(loc="upper right")

        axes[5].plot(
            frames[aruco_detected],
            aruco_rms[aruco_detected],
            marker="o",
            markersize=3,
            label="ArUco reprojection RMS",
        )
        axes[5].set_ylabel("ArUco error [px]")
        axes[5].set_xlabel("Video frame")
        axes[5].set_title("6. ArUco pose detection")
        axes[5].grid(True)
        axes[5].legend(loc="upper left")

        figure.suptitle(f"{recording_name}: mapping pipeline diagnostics")
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)

    @staticmethod
    def _field(metrics, field_name):
        return np.asarray(
            [getattr(item, field_name) for item in metrics],
            dtype=float,
        )

    def _save_pairing_plot(
        self,
        frame_collection,
        output_path,
        recording_name,
    ):
        """Save diagnostics for COLMAP sequential and loop matching."""
        metrics = frame_collection.frame_diagnostics
        frames = self._field(metrics, "frame_index")
        matched_pairs = self._field(metrics, "matched_pairs")
        verified_pairs = self._field(metrics, "verified_pairs")
        raw_matches = self._field(metrics, "raw_matches")
        verified_inliers = self._field(metrics, "verified_inliers")

        matches_per_pair = raw_matches / np.maximum(matched_pairs, 1)
        inliers_per_pair = verified_inliers / np.maximum(verified_pairs, 1)

        figure, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        axes[0].plot(frames, matched_pairs, label="New matched pairs")
        axes[0].plot(frames, verified_pairs, label="New verified pairs")
        axes[0].set_title("1. COLMAP sequential and loop pairs")
        axes[0].set_ylabel("Image pairs")
        axes[0].grid(True)
        axes[0].legend(loc="upper left")

        axes[1].plot(frames, matches_per_pair, label="Matches / new pair")
        axes[1].plot(frames, inliers_per_pair, label="Inliers / verified pair")
        axes[1].set_title("2. COLMAP matching and geometric verification")
        axes[1].set_ylabel("Correspondences")
        axes[1].set_xlabel("Video frame")
        axes[1].grid(True)
        axes[1].legend(loc="upper left")

        figure.suptitle(f"{recording_name}: COLMAP matching diagnostics")
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)

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
                "exact_counts": {},
                "ranges": {},
            }

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

    @staticmethod
    def _timing_document(frame_collection, durations):
        collection_timing = frame_collection.timing
        after_collection_seconds = (
            durations.reconstruction_seconds
            + durations.alignment_seconds
            + durations.map_finalization_seconds
            + durations.map_saving_seconds
        )
        return {
            "frame_collection": {
                "processed_frames": frame_collection.image_count,
                "matched_pairs": frame_collection.matched_pair_count,
                "verified_pairs": frame_collection.verified_pair_count,
                "setup_s": collection_timing.setup_seconds,
                "frame_read_s": collection_timing.frame_read_seconds,
                "image_save_s": collection_timing.image_save_seconds,
                "mask_generation_s": (
                    collection_timing.mask_generation_seconds
                ),
                "feature_extraction_s": (
                    collection_timing.feature_extraction_seconds
                ),
                "aruco_detection_s": (
                    collection_timing.aruco_detection_seconds
                ),
                "image_database_write_s": (
                    collection_timing.image_database_write_seconds
                ),
                "sequential_matching_and_verification_s": (
                    collection_timing.sequential_matching_seconds
                ),
                "wall_time_s": collection_timing.wall_seconds,
                "average_wall_time_per_frame_ms": (
                    1000.0
                    * collection_timing.wall_seconds
                    / max(frame_collection.image_count, 1)
                ),
            },
            "after_collection": {
                "reconstruction_s": durations.reconstruction_seconds,
                "aruco_scale_estimation_s": durations.alignment_seconds,
                "map_finalization_s": durations.map_finalization_seconds,
                "map_saving_s": durations.map_saving_seconds,
                "wall_time_s": after_collection_seconds,
            },
            "total_map_build_wall_time_s": durations.total_seconds,
        }

    @staticmethod
    def _summary_document(
        configuration,
        frame_collection,
        reconstruction,
        alignment,
        frozen_map,
        track_statistics,
        timing,
        diagnostics_csv_path,
        diagnostics_plot_path,
        pairing_plot_path,
    ):
        return {
            "mapping_feature_type": configuration.mapping_feature_type,
            "mapping_start_frame": configuration.start_frame,
            "mapping_end_frame": configuration.end_frame,
            "reconstruction_method": configuration.reconstruction_method,
            "colmap_mapping": {
                "keyframe_interval": configuration.keyframe_interval,
                "maximum_features": configuration.maximum_features,
                "sequential_overlap": configuration.sequential_overlap,
                "loop_detection": configuration.loop_detection,
                "vocabulary_tree_path": configuration.vocabulary_tree_path,
            },
            "coordinate_frame": frozen_map.coordinate_frame,
            "extracted_images": frame_collection.image_count,
            "matched_image_pairs": frame_collection.matched_pair_count,
            "verified_image_pairs": frame_collection.verified_pair_count,
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
            "mapping_pipeline_diagnostics_csv": diagnostics_csv_path.name,
            "mapping_pipeline_diagnostics_plot": diagnostics_plot_path.name,
            "mapping_pairing_diagnostics_plot": pairing_plot_path.name,
            "imu_gravity": frame_collection.imu_gravity_summary,
            "timing": timing,
        }

    @staticmethod
    def _write_json(path, contents):
        with path.open("w", encoding="utf-8") as file:
            json.dump(contents, file, indent=2)

    def _print_summary(
        self,
        frozen_map,
        reconstruction,
        frame_collection,
        track_statistics,
        timing,
        map_path,
        output_directory,
    ):
        print(f"Saved frozen 3D map: {map_path}")
        print(
            "Saved track statistics: "
            f"{output_directory / 'track_length_statistics.json'}"
        )
        print(
            f"Map: {len(frozen_map.positions)}/"
            f"{len(frozen_map.candidate_positions)} selected landmarks | "
            f"{frozen_map.occupied_grid_cell_count} occupied grid cells | "
            f"{reconstruction.num_reg_images()}/"
            f"{frame_collection.image_count} registered images"
        )
        self._print_track_statistics(track_statistics)
        self._print_timing(timing)

    @staticmethod
    def _print_track_statistics(statistics):
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

    @staticmethod
    def _print_timing(timing):
        collection = timing["frame_collection"]
        after = timing["after_collection"]
        print("Map build timing")
        print("  FRAME COLLECTION")
        print(f"    Setup: {collection['setup_s']:.2f} s")
        print(f"    Frame reading: {collection['frame_read_s']:.2f} s")
        print(f"    Image saving: {collection['image_save_s']:.2f} s")
        print(f"    Mask generation: {collection['mask_generation_s']:.2f} s")
        print(
            "    Feature extraction: "
            f"{collection['feature_extraction_s']:.2f} s"
        )
        print(
            "    Sequential matching and geometric verification: "
            f"{collection['sequential_matching_and_verification_s']:.2f} s"
        )
        print(f"    TOTAL: {collection['wall_time_s']:.2f} s")
        print("  AFTER COLLECTION")
        print(f"    Reconstruction: {after['reconstruction_s']:.2f} s")
        print(
            "    ArUco scale estimation: "
            f"{after['aruco_scale_estimation_s']:.2f} s"
        )
        print(f"    Map finalization: {after['map_finalization_s']:.2f} s")
        print(f"    Map saving: {after['map_saving_s']:.2f} s")
        print(f"    TOTAL: {after['wall_time_s']:.2f} s")
        print(
            "  COMPLETE MAP BUILD: "
            f"{timing['total_map_build_wall_time_s']:.2f} s"
        )
