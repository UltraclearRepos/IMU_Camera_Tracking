import csv
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).absolute().parent
THREE_D_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(THREE_D_DIR))

from visualization.tracking_visualization import save_comparison_figure


# Change this path to process a different tracking result.
RESULT_FOLDER = (
    THREE_D_DIR
    / "jenkins_results"
    / "Cylinder"
    / "Cylinder_keyframe_interval=1_recent=10_sift_512_256_512"
    / "sift"
    / "initial_50mm_Arc180-Speed-3_2026-08-20_14.39.08"
)
CAMERA_GT_PATH = RESULT_FOLDER / "camera_gt.csv"
OUTPUT_PATH = RESULT_FOLDER / "position_scaled.png"


def load_camera_gt(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"No data rows in {path}")

    times_s = np.asarray([float(row["time_s"]) for row in rows])
    estimate = np.asarray(
        [
            [float(row[column]) for column in ("x_mm", "y_mm", "z_mm")]
            for row in rows
        ]
    )
    ground_truth = np.asarray(
        [
            [
                float(row[column])
                for column in ("gt_x_mm", "gt_y_mm", "gt_z_mm")
            ]
            for row in rows
        ]
    )
    return times_s, estimate, ground_truth


def fit_global_scale(estimate, ground_truth, valid):
    estimate_valid = estimate[valid]
    ground_truth_valid = ground_truth[valid]
    denominator = np.sum(estimate_valid**2)
    if denominator <= np.finfo(float).eps:
        raise RuntimeError("Camera trajectory has no usable translation")

    scale = float(
        np.sum(estimate_valid * ground_truth_valid) / denominator
    )
    if scale <= 0.0:
        raise RuntimeError(
            "Best fitted scale is not positive; check coordinate-frame alignment"
        )
    return scale


def main():
    times_s, estimate, ground_truth = load_camera_gt(CAMERA_GT_PATH)
    valid = np.isfinite(estimate).all(axis=1)
    valid &= np.isfinite(ground_truth).all(axis=1)
    if not np.any(valid):
        raise RuntimeError("No finite camera/GT pose pairs available")

    scale = fit_global_scale(estimate, ground_truth, valid)
    scaled_estimate = estimate * scale

    component_errors = np.full_like(scaled_estimate, np.nan)
    component_errors[valid] = (
        scaled_estimate[valid] - ground_truth[valid]
    )
    position_errors = np.full(len(scaled_estimate), np.nan)
    position_errors[valid] = np.linalg.norm(
        component_errors[valid],
        axis=1,
    )

    original_errors = np.linalg.norm(
        estimate[valid] - ground_truth[valid],
        axis=1,
    )
    original_rmse = float(np.sqrt(np.mean(original_errors**2)))
    scaled_rmse = float(np.sqrt(np.mean(position_errors[valid] ** 2)))
    plot_times = times_s - times_s[0]

    save_comparison_figure(
        plot_times,
        ground_truth,
        scaled_estimate,
        component_errors,
        position_errors,
        ["X", "Y", "Z"],
        "mm",
        "Euclidean distance on tracked frames",
        OUTPUT_PATH,
        f"{RESULT_FOLDER.name}: scale-corrected camera position vs GT\n"
        f"Optimal global scale: {scale:.6f} | "
        f"RMSE: {original_rmse:.2f} -> {scaled_rmse:.2f} mm",
        estimate_label="Camera x fitted scale",
        x_label="Tracking time [s]",
    )

    print(f"Input: {CAMERA_GT_PATH}")
    print(f"Optimal global scale: {scale:.12f}")
    print(f"Position RMSE: {original_rmse:.6f} -> {scaled_rmse:.6f} mm")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
