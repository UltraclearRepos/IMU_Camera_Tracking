import csv

import matplotlib.pyplot as plt
import numpy as np


def save_mapping_pipeline_diagnostics(rows, output_path, recording_name):
    """Plot quality gates from mapping inputs through metric alignment."""
    frames = np.asarray([row["frame"] for row in rows])
    feature_count = np.asarray([row["feature_count"] for row in rows])
    raw_matches = np.asarray([row["raw_matches"] for row in rows])
    verified_inliers = np.asarray(
        [row["verified_inliers"] for row in rows]
    )
    pairs_attempted = np.asarray(
        [row["pairs_attempted"] for row in rows]
    )
    pairs_verified = np.asarray(
        [row["pairs_verified"] for row in rows]
    )
    registered = np.asarray([row["registered"] for row in rows], dtype=bool)
    aruco_detected = np.asarray(
        [row["aruco_detected"] for row in rows], dtype=bool
    )
    aruco_rms = np.asarray(
        [row["aruco_reprojection_rms_px"] for row in rows]
    )
    aruco_max = np.asarray(
        [row["aruco_reprojection_max_px"] for row in rows]
    )
    alignment_residual = np.asarray(
        [row["aruco_alignment_residual_mm"] for row in rows]
    )

    figure, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)

    axes[0].plot(frames, feature_count, label="Selected features")
    axes[0].plot(frames, raw_matches, label="Raw matches to prior frames")
    axes[0].plot(frames, verified_inliers, label="Verified pair inliers")
    axes[0].set_ylabel("Features / matches")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(frames, pairs_attempted, label="Pairs attempted")
    axes[1].plot(frames, pairs_verified, label="Pairs accepted")
    axes[1].scatter(
        frames[registered],
        pairs_verified[registered],
        color="tab:green",
        s=20,
        label="Registered by SfM",
        zorder=3,
    )
    axes[1].scatter(
        frames[~registered],
        pairs_verified[~registered],
        color="tab:red",
        s=20,
        label="Not registered",
        zorder=3,
    )
    axes[1].set_ylabel("Image pairs")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(
        frames[aruco_detected],
        aruco_rms[aruco_detected],
        marker="o",
        markersize=3,
        label="ArUco reprojection RMS",
    )
    axes[2].plot(
        frames[aruco_detected],
        aruco_max[aruco_detected],
        marker="o",
        markersize=3,
        label="ArUco max corner error",
    )
    axes[2].scatter(
        frames[aruco_detected & registered],
        np.zeros(np.sum(aruco_detected & registered)),
        color="tab:green",
        marker="|",
        s=80,
        label="Used by ArUco alignment",
        zorder=3,
    )
    axes[2].set_ylabel("Reprojection error [px]")
    axes[2].grid(True)
    axes[2].legend()

    alignment_used = np.isfinite(alignment_residual)
    axes[3].plot(
        frames[alignment_used],
        alignment_residual[alignment_used],
        marker="o",
        markersize=3,
        color="tab:purple",
        label="SfM-to-ArUco center residual",
    )
    if np.any(alignment_used):
        rmse = np.sqrt(np.mean(alignment_residual[alignment_used] ** 2))
        axes[3].axhline(
            rmse,
            color="black",
            linestyle="--",
            label=f"Alignment RMSE: {rmse:.2f} mm",
        )
    axes[3].set_ylabel("Alignment residual [mm]")
    axes[3].set_xlabel("Video frame")
    axes[3].grid(True)
    axes[3].legend()

    figure.suptitle(
        f"{recording_name}: mapping pipeline diagnostics\n"
        "features → verified pairs → SfM registration → ArUco metric alignment"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_mapping_pipeline_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
