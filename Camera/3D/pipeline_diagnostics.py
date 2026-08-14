import csv

import matplotlib.pyplot as plt
import numpy as np


def save_mapping_pipeline_diagnostics(rows, output_path, recording_name):
    """Plot feature, SfM, BA, and metric-alignment stages of mapping."""
    frames = np.asarray([row["frame"] for row in rows])

    def field(name):
        return np.asarray([row[name] for row in rows], dtype=float)

    feature_count = field("feature_count")
    raw_matches = field("raw_matches")
    verified_inliers = field("verified_inliers")
    pairs_attempted = field("pairs_attempted")
    pairs_verified = field("pairs_verified")
    registered = field("registered").astype(bool)
    triangulated = field("triangulated_observations")
    triangulated_ratio = field("triangulated_feature_ratio")
    track_length = field("median_point_track_length")
    point_reprojection = field("median_point_reprojection_error_px")
    camera_translation_step = field("camera_translation_step_mm")
    camera_rotation_step = field("camera_rotation_step_deg")
    aruco_detected = field("aruco_detected").astype(bool)
    aruco_alignment_used = field("aruco_alignment_used").astype(bool)
    aruco_rms = field("aruco_reprojection_rms_px")
    alignment_residual = field("aruco_alignment_residual_mm")

    matches_per_pair = raw_matches / np.maximum(pairs_attempted, 1)
    inliers_per_verified_pair = verified_inliers / np.maximum(
        pairs_verified, 1
    )
    pair_acceptance = pairs_verified / np.maximum(pairs_attempted, 1)

    figure, axes = plt.subplots(6, 1, figsize=(16, 19), sharex=True)

    axes[0].plot(frames, feature_count, color="tab:blue")
    axes[0].set_ylabel("Selected features")
    axes[0].set_title("1. Spatial feature selection passed to pair matching")
    axes[0].grid(True)

    axes[1].plot(
        frames,
        matches_per_pair,
        label="Raw matches / attempted pair",
    )
    axes[1].plot(
        frames,
        inliers_per_verified_pair,
        label="Geometric inliers / accepted pair",
    )
    axes[1].set_ylabel("Correspondences")
    axes[1].set_title("2. Descriptor matching and two-view geometry verification")
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

    axes[2].plot(
        frames,
        triangulated,
        label="Triangulated observations",
    )
    axes[2].scatter(
        frames[~registered],
        np.zeros(np.sum(~registered)),
        color="tab:red",
        s=24,
        label="Image not registered",
        zorder=3,
    )
    axes[2].set_ylabel("3D observations")
    axes[2].set_title("3. SfM image registration and landmark triangulation")
    axes[2].grid(True)
    axes[2].legend(loc="upper left")
    triangulation_axis = axes[2].twinx()
    triangulation_axis.plot(
        frames,
        100.0 * triangulated_ratio,
        color="tab:orange",
        label="Features assigned to a 3D point",
    )
    triangulation_axis.set_ylabel("Triangulated features [%]")
    triangulation_axis.set_ylim(0.0, 105.0)
    triangulation_axis.legend(loc="upper right")

    axes[3].plot(
        frames,
        track_length,
        label="Median landmark track length",
    )
    axes[3].set_ylabel("Images / landmark")
    axes[3].set_title("4. Landmark support and final BA reprojection quality")
    axes[3].grid(True)
    axes[3].legend(loc="upper left")
    reprojection_axis = axes[3].twinx()
    reprojection_axis.plot(
        frames,
        point_reprojection,
        color="tab:red",
        label="Median landmark reprojection error",
    )
    reprojection_axis.set_ylabel("Reprojection error [px]")
    reprojection_axis.legend(loc="upper right")

    axes[4].plot(
        frames,
        camera_translation_step,
        label="Consecutive BA camera-center distance",
    )
    axes[4].set_ylabel("Translation [mm]")
    axes[4].set_title("5. Recovered camera motion after BA and metric alignment")
    axes[4].grid(True)
    axes[4].legend(loc="upper left")
    rotation_axis = axes[4].twinx()
    rotation_axis.plot(
        frames,
        camera_rotation_step,
        color="tab:orange",
        label="Consecutive BA camera rotation",
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
    axes[5].scatter(
        frames[aruco_alignment_used],
        aruco_rms[aruco_alignment_used],
        color="tab:green",
        s=22,
        label="Used for metric alignment",
        zorder=3,
    )
    axes[5].set_ylabel("ArUco error [px]")
    axes[5].set_xlabel("Video frame")
    axes[5].set_title("6. ArUco pose quality and SfM-to-metric alignment")
    axes[5].grid(True)
    axes[5].legend(loc="upper left")
    alignment_used = np.isfinite(alignment_residual)
    alignment_axis = axes[5].twinx()
    alignment_axis.plot(
        frames[alignment_used],
        alignment_residual[alignment_used],
        marker="o",
        markersize=3,
        color="tab:purple",
        label="SfM-to-ArUco center residual",
    )
    if np.any(alignment_used):
        rmse = np.sqrt(np.mean(alignment_residual[alignment_used] ** 2))
        alignment_axis.axhline(
            rmse,
            color="black",
            linestyle="--",
            label=f"Alignment RMSE: {rmse:.2f} mm",
        )
    alignment_axis.set_ylabel("Alignment residual [mm]")
    alignment_axis.legend(loc="upper right")

    figure.suptitle(
        f"{recording_name}: mapping pipeline diagnostics\n"
        "features -> pairs -> geometry -> triangulation -> BA -> ArUco alignment"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_mapping_pipeline_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
