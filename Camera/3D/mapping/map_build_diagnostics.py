import csv
import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

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
    ):
        registered_images = {
            image.name: image for image in reconstruction.images.values()
        }

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
            alignment,
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
        self._print_scale_statistics(
            alignment.scale,
            self._scale_pair_statistics(alignment)["statistics"],
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

    @staticmethod
    def _save_frame_csv(frame_collection, output_path):
        rows = [
            asdict(metrics) for metrics in frame_collection.frame_diagnostics
        ]
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _save_frame_plot(
        self,
        frame_collection,
        alignment,
        output_path,
        recording_name,
    ):
        metrics = frame_collection.frame_diagnostics
        frames = self._field(metrics, "frame_index")
        feature_count = self._field(metrics, "feature_count")
        raw_matches = self._field(metrics, "raw_matches")
        verified_inliers = self._field(metrics, "verified_inliers")
        attempted_pairs = self._field(metrics, "attempted_pairs")
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
        aruco_detected = self._field(metrics, "aruco_detected").astype(bool)
        aruco_rms = self._field(metrics, "aruco_reprojection_rms_px")
        aligned_image_names = set(alignment.aligned_image_names)
        used_for_scale = np.asarray(
            [item.image_name in aligned_image_names for item in metrics],
            dtype=bool,
        )
        frame_by_image_name = {
            item.image_name: item.frame_index for item in metrics
        }

        matches_per_pair = raw_matches / np.maximum(attempted_pairs, 1)
        inliers_per_pair = verified_inliers / np.maximum(verified_pairs, 1)
        pair_acceptance = verified_pairs / np.maximum(attempted_pairs, 1)

        figure, axes = plt.subplots(6, 1, figsize=(16, 19))
        for axis_index in (1, 2, 3, 5):
            axes[axis_index].sharex(axes[0])
        for axis_index in range(4):
            axes[axis_index].tick_params(labelbottom=False)
        axes[0].plot(frames, feature_count, color="tab:blue")
        axes[0].set_ylabel("Selected features")
        axes[0].set_title("1. Spatial feature selection")
        axes[0].grid(True)

        axes[1].plot(frames, matches_per_pair, label="Matches / attempted pair")
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

        pair_scale_diagnostics = self._scale_pair_statistics(alignment)
        scale_statistics = pair_scale_diagnostics["statistics"]
        ordered_pair_indices = np.asarray(
            sorted(
                range(len(alignment.aligned_image_pairs)),
                key=lambda index: (
                    frame_by_image_name[
                        alignment.aligned_image_pairs[index][0]
                    ],
                    frame_by_image_name[
                        alignment.aligned_image_pairs[index][1]
                    ],
                ),
            ),
            dtype=int,
        )
        ordered_image_pairs = [
            alignment.aligned_image_pairs[index]
            for index in ordered_pair_indices
        ]
        aruco_pair_distances = np.asarray(
            alignment.aligned_pair_distances_mm,
            dtype=float,
        )[ordered_pair_indices]
        sfm_pair_distances = np.asarray(
            alignment.aligned_pair_sfm_distances,
            dtype=float,
        )[ordered_pair_indices]
        pair_numbers = np.arange(1, len(aruco_pair_distances) + 1)
        aruco_line = axes[4].plot(
            pair_numbers,
            aruco_pair_distances,
            color="tab:blue",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="ArUco displacement",
        )
        sfm_distance_axis = axes[4].twinx()
        sfm_line = sfm_distance_axis.plot(
            pair_numbers,
            sfm_pair_distances,
            color="tab:orange",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="SfM displacement",
        )
        comparison_maximum_mm = 1.08 * max(
            float(np.max(aruco_pair_distances)),
            float(alignment.scale * np.max(sfm_pair_distances)),
        )
        axes[4].set_ylim(0.0, comparison_maximum_mm)
        sfm_distance_axis.set_ylim(
            0.0,
            comparison_maximum_mm / alignment.scale,
        )

        axes[4].set_xticks(pair_numbers)
        axes[4].set_xticklabels(
            [
                (
                    f"{frame_by_image_name[first_name]}-"
                    f"{frame_by_image_name[second_name]}"
                )
                for first_name, second_name in ordered_image_pairs
            ],
            rotation=90,
            ha="center",
            fontsize=6,
        )
        axes[4].text(
            0.01,
            0.97,
            (
                f"Axes linked by LS scale: {alignment.scale:.3f} mm/unit\n"
                "overlap = pair agrees with fitted scale\n"
                "robust CV: "
                f"{scale_statistics['robust_coefficient_of_variation_percent']:.1f}%\n"
                "relative fit RMSE: "
                f"{scale_statistics['fit_relative_rmse_percent']:.1f}%"
            ),
            transform=axes[4].transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        axes[4].set_ylabel("ArUco displacement [mm]", color="tab:blue")
        sfm_distance_axis.set_ylabel(
            "SfM displacement [SfM unit]",
            color="tab:orange",
        )
        axes[4].set_xlabel("Selected frame pair")
        axes[4].set_title("5. ArUco and SfM displacement for each pair")
        axes[4].grid(True)
        axes[4].legend(
            aruco_line + sfm_line,
            [line.get_label() for line in aruco_line + sfm_line],
            loc="upper right",
        )

        axes[5].plot(
            frames[aruco_detected],
            aruco_rms[aruco_detected],
            marker="o",
            markersize=3,
            label="ArUco reprojection RMS",
        )
        axes[5].scatter(
            frames[used_for_scale],
            aruco_rms[used_for_scale],
            color="tab:red",
            marker="x",
            s=40,
            linewidths=1.5,
            label="Used for scale estimation",
            zorder=3,
        )
        pair_distance_axis = axes[5].twinx()
        for pair_index, (image_pair, distance_mm) in enumerate(
            zip(
                alignment.aligned_image_pairs,
                alignment.aligned_pair_distances_mm,
            )
        ):
            first_name, second_name = image_pair
            pair_distance_axis.plot(
                [
                    frame_by_image_name[first_name],
                    frame_by_image_name[second_name],
                ],
                [distance_mm, distance_mm],
                color="tab:purple",
                marker="o",
                markersize=4,
                linewidth=1.5,
                alpha=0.7,
                label=(
                    "Selected ArUco pair distance"
                    if pair_index == 0
                    else None
                ),
                zorder=2,
            )
        pair_distance_axis.set_ylabel("Selected pair distance [mm]")
        pair_distance_axis.legend(loc="upper right")
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
        """Save diagnostics specific to local tracking and pair selection."""
        metrics = frame_collection.frame_diagnostics
        frames = self._field(metrics, "frame_index")
        continued_tracks = self._field(metrics, "continued_track_count")
        new_tracks = self._field(metrics, "new_track_count")
        active_candidates = self._field(metrics, "active_pair_candidates")
        recent_pairs = self._field(metrics, "recent_pair_count")
        motion_pairs = self._field(metrics, "motion_pair_count")
        attempted_pairs = self._field(metrics, "attempted_pairs")
        verified_pairs = self._field(metrics, "verified_pairs")
        maximum_motion = self._field(
            metrics,
            "maximum_selected_motion_px",
        )
        minimum_overlap = self._field(metrics, "minimum_selected_overlap")
        raw_matches = self._field(metrics, "raw_matches")
        verified_inliers = self._field(metrics, "verified_inliers")

        total_tracks = continued_tracks + new_tracks
        matches_per_pair = raw_matches / np.maximum(attempted_pairs, 1)
        inliers_per_pair = verified_inliers / np.maximum(verified_pairs, 1)

        figure, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

        axes[0].plot(frames, continued_tracks, label="Continued LK tracks")
        axes[0].plot(frames, new_tracks, label="New SIFT tracks")
        axes[0].plot(
            frames,
            total_tracks,
            color="black",
            alpha=0.65,
            label="Active local tracks",
        )
        axes[0].set_title("1. Local tracks used to choose keyframe pairs")
        axes[0].set_ylabel("Tracks")
        axes[0].grid(True)
        axes[0].legend(loc="upper left")

        axes[1].plot(
            frames,
            active_candidates,
            color="tab:gray",
            label="Older candidates with sufficient overlap",
        )
        axes[1].plot(frames, recent_pairs, label="Selected recent pairs")
        axes[1].plot(frames, motion_pairs, label="Selected motion pairs")
        axes[1].plot(
            frames,
            attempted_pairs,
            color="black",
            alpha=0.65,
            label="Pairs sent to matching",
        )
        axes[1].set_title("2. Pair selection")
        axes[1].set_ylabel("Image pairs")
        axes[1].grid(True)
        axes[1].legend(loc="upper left")
        verified_axis = axes[1].twinx()
        verified_axis.plot(
            frames,
            verified_pairs,
            color="tab:green",
            alpha=0.7,
            label="Verified pairs",
        )
        verified_axis.set_ylabel("Verified pairs")
        verified_axis.legend(loc="upper right")

        axes[2].plot(
            frames,
            maximum_motion,
            color="tab:purple",
            label="Maximum selected displacement",
        )
        axes[2].set_title("3. Selected-pair geometry")
        axes[2].set_ylabel("Displacement [px]")
        axes[2].grid(True)
        axes[2].legend(loc="upper left")
        overlap_axis = axes[2].twinx()
        overlap_axis.plot(
            frames,
            100.0 * minimum_overlap,
            color="tab:orange",
            label="Minimum selected overlap",
        )
        overlap_axis.set_ylabel("Overlap [%]")
        overlap_axis.set_ylim(0.0, 105.0)
        overlap_axis.legend(loc="upper right")

        axes[3].plot(
            frames,
            matches_per_pair,
            label="Raw matches / selected pair",
        )
        axes[3].plot(
            frames,
            inliers_per_pair,
            label="Inliers / verified pair",
        )
        axes[3].set_title("4. Matching result for selected pairs")
        axes[3].set_ylabel("Correspondences")
        axes[3].set_xlabel("Video frame")
        axes[3].grid(True)
        axes[3].legend(loc="upper left")

        figure.suptitle(f"{recording_name}: keyframe-pairing diagnostics")
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
                "attempted_pairs": frame_collection.attempted_pair_count,
                "verified_pairs": frame_collection.verified_pair_count,
                "setup_s": collection_timing.setup_seconds,
                "frame_read_s": collection_timing.frame_read_seconds,
                "image_save_s": collection_timing.image_save_seconds,
                "feature_extraction_s": (
                    collection_timing.feature_extraction_seconds
                ),
                "skin_masking_s": collection_timing.skin_masking_seconds,
                "local_tracking_s": (
                    collection_timing.local_tracking_seconds
                ),
                "aruco_detection_s": (
                    collection_timing.aruco_detection_seconds
                ),
                "image_database_write_s": (
                    collection_timing.image_database_write_seconds
                ),
                "pair_selection_s": (
                    collection_timing.pair_selection_seconds
                ),
                "feature_matching_s": (
                    collection_timing.feature_matching_seconds
                ),
                "geometry_verification_s": (
                    collection_timing.geometry_verification_seconds
                ),
                "pair_database_write_s": (
                    collection_timing.pair_database_write_seconds
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
            "feature_type": configuration.feature_type,
            "mapping_start_frame": configuration.start_frame,
            "mapping_end_frame": configuration.end_frame,
            "reconstruction_method": configuration.reconstruction_method,
            "keyframe_interval": configuration.keyframe_interval,
            "adaptive_pair_selection": {
                "recent_pair_count": configuration.recent_pair_count,
                "motion_targets_px": list(configuration.motion_targets_px),
                "minimum_new_track_distance_px": (
                    configuration.minimum_new_track_distance_px
                ),
                "maximum_active_track_count": (
                    configuration.maximum_active_track_count
                ),
                "maximum_forward_backward_error_px": (
                    configuration.maximum_forward_backward_error_px
                ),
                "minimum_keyframe_overlap": (
                    configuration.minimum_keyframe_overlap
                ),
                "maximum_motion_anchor_px": (
                    configuration.maximum_motion_anchor_px
                ),
            },
            "coordinate_frame": frozen_map.coordinate_frame,
            "extracted_images": frame_collection.image_count,
            "attempted_image_pairs": frame_collection.attempted_pair_count,
            "verified_image_pairs": frame_collection.verified_pair_count,
            "registered_images": reconstruction.num_reg_images(),
            "selected_landmarks": len(frozen_map.positions),
            "candidate_landmarks": len(frozen_map.candidate_positions),
            "scale_candidate_frames": alignment.candidate_frame_count,
            "scale_frames": alignment.aligned_frame_count,
            "scale_reprojection_rms_threshold_px": (
                alignment.reprojection_rms_threshold_px
            ),
            "scale_image_names": list(alignment.aligned_image_names),
            "scale_image_pairs": [
                list(pair) for pair in alignment.aligned_image_pairs
            ],
            "scale_pair_distances_mm": list(
                alignment.aligned_pair_distances_mm
            ),
            "scale_mm_per_sfm_unit": alignment.scale,
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
    def _scale_pair_statistics(alignment):
        aruco_distances = np.asarray(
            alignment.aligned_pair_distances_mm,
            dtype=float,
        )
        sfm_distances = np.asarray(
            alignment.aligned_pair_sfm_distances,
            dtype=float,
        )
        pair_scales = aruco_distances / sfm_distances

        median = float(np.median(pair_scales))
        mad = float(np.median(np.abs(pair_scales - median)))
        robust_standard_deviation = 1.4826 * mad
        percentile_10, percentile_90 = np.percentile(
            pair_scales,
            [10.0, 90.0],
        )
        rms_aruco_distance = float(
            np.sqrt(np.mean(aruco_distances**2))
        )

        return {
            "pair_estimates_mm_per_sfm_unit": pair_scales.tolist(),
            "statistics": {
                "pair_count": len(pair_scales),
                "mean_mm_per_sfm_unit": float(np.mean(pair_scales)),
                "median_mm_per_sfm_unit": median,
                "standard_deviation_mm_per_sfm_unit": float(
                    np.std(pair_scales)
                ),
                "median_absolute_deviation_mm_per_sfm_unit": mad,
                "robust_standard_deviation_mm_per_sfm_unit": (
                    robust_standard_deviation
                ),
                "robust_coefficient_of_variation_percent": float(
                    100.0 * robust_standard_deviation / abs(median)
                ),
                "percentile_10_mm_per_sfm_unit": float(percentile_10),
                "percentile_90_mm_per_sfm_unit": float(percentile_90),
                "central_80_percent_relative_span_percent": float(
                    100.0
                    * (percentile_90 - percentile_10)
                    / abs(median)
                ),
                "fit_relative_rmse_percent": float(
                    100.0 * alignment.rmse_mm / rms_aruco_distance
                ),
            },
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
    def _print_scale_statistics(scale, statistics):
        print("ArUco scale diagnostics")
        print(f"  Least-squares scale: {scale:.6f} mm/SfM unit")
        print(
            "  Pair scale median: "
            f"{statistics['median_mm_per_sfm_unit']:.6f} mm/SfM unit"
        )
        print(
            "  Robust coefficient of variation: "
            f"{statistics['robust_coefficient_of_variation_percent']:.2f}%"
        )
        print(
            "  Central 80% relative span: "
            f"{statistics['central_80_percent_relative_span_percent']:.2f}%"
        )
        print(
            "  Relative distance-fit RMSE: "
            f"{statistics['fit_relative_rmse_percent']:.2f}%"
        )

    @staticmethod
    def _print_timing(timing):
        collection = timing["frame_collection"]
        after = timing["after_collection"]
        print("Map build timing")
        print("  FRAME COLLECTION")
        print(f"    Setup: {collection['setup_s']:.2f} s")
        print(f"    Frame reading: {collection['frame_read_s']:.2f} s")
        print(f"    Image saving: {collection['image_save_s']:.2f} s")
        print(
            "    Feature extraction: "
            f"{collection['feature_extraction_s']:.2f} s"
        )
        print(f"    Skin masking: {collection['skin_masking_s']:.2f} s")
        print(
            f"    Matching: {collection['feature_matching_s']:.2f} s"
        )
        print(
            "    Geometry verification: "
            f"{collection['geometry_verification_s']:.2f} s"
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
