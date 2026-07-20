import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def project_map_points(
    map_points,
    result,
    tracker,
    frame_shape,
    feature_roi_bottom_fraction,
):
    if len(map_points) == 0:
        return np.empty((0, 2))

    rvec = cv2.Rodrigues(result["R"])[0]
    projected, _ = cv2.projectPoints(
        map_points,
        rvec,
        result["t"],
        tracker.camera_matrix,
        tracker.distortion,
    )
    projected = projected.reshape(-1, 2)

    camera_points = (result["R"] @ map_points.T).T + result["t"]
    height, width = frame_shape[:2]
    visible = camera_points[:, 2] > 0.0
    visible &= projected[:, 0] >= 0.0
    visible &= projected[:, 0] < width
    visible &= projected[:, 1] >= height * (
        1.0 - feature_roi_bottom_fraction
    )
    visible &= projected[:, 1] < height
    return projected[visible]


def diagnostic_frame(
    frame,
    tracker,
    result,
    relative_positions,
    feature_roi_bottom_fraction,
    initialization_frames,
    initialization_min_landmarks,
    tracking_time_ms,
):
    output = frame.copy()
    roi_top = round(frame.shape[0] * (1.0 - feature_roi_bottom_fraction))
    output[:roi_top] = 0
    cv2.line(
        output,
        (0, roi_top),
        (frame.shape[1] - 1, roi_top),
        (255, 180, 0),
        2,
    )
    cv2.putText(
        output,
        "Feature ROI below this line",
        (12, roi_top + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        2,
    )
    cv2.putText(
        output,
        (
            f"Tracking: {tracking_time_ms:.1f} ms | "
            f"{1000.0 / tracking_time_ms:.1f} FPS"
        ),
        (max(12, frame.shape[1] - 360), 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    tracked = result is not None
    initializing = not tracker.keyframes and tracker.initialization is not None
    if tracked:
        color = (40, 200, 40)
        method = tracker.last_diagnostics.get(
            "tracking_method",
            "lightglue",
        )
        label = f"TRACKING - {method.replace('_', ' ').upper()}"
    elif initializing:
        color = (0, 180, 255)
        label = "INITIALIZING"
    elif not tracker.keyframes:
        color = (0, 180, 255)
        label = "WAITING FOR ARUCO"
    else:
        color = (30, 30, 220)
        label = "LOST"

    if tracked:
        projected_map_points = project_map_points(
            tracker.all_map_points(),
            result,
            tracker,
            frame.shape,
            feature_roi_bottom_fraction,
        )
        projected_inliers = project_map_points(
            result["inlier_map_points"],
            result,
            tracker,
            frame.shape,
            feature_roi_bottom_fraction,
        )

        for point in projected_map_points:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                1,
                (0, 255, 255),
                -1,
            )

        for point in projected_inliers:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 255, 0),
                -1,
            )

        for point in result["outlier_points"]:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 0, 255),
                -1,
            )

        position = relative_positions[-1]
        cv2.putText(
            output,
            f"XYZ: {position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f} mm",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.rectangle(output, (8, 84), (300, 181), (0, 0, 0), -1)
        legend = [
            (
                f"Map not detected: {len(projected_map_points) - len(projected_inliers)}",
                (0, 255, 255),
            ),
            (f"PnP inliers: {result['inliers']}", (0, 255, 0)),
            (f"PnP outliers: {len(result['outlier_points'])}", (0, 0, 255)),
            (
                f"Nearby associations: {result['nearby_associations']}",
                (255, 180, 0),
            ),
        ]
        for index, (text, text_color) in enumerate(legend):
            cv2.putText(
                output,
                text,
                (15, 105 + 23 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2,
            )

    elif initializing:
        diagnostics = tracker.last_diagnostics
        for point in diagnostics["initialization_points"]:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 255, 0),
                -1,
            )

        aruco_status = (
            "detected"
            if diagnostics["initialization_aruco_detected"]
            else "not detected"
        )
        cv2.putText(
            output,
            f"ArUco: {aruco_status}",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            output,
            (
                f"Frames: {diagnostics['initialization_frames']}"
                f"/{initialization_frames} | confirmed: "
                f"{diagnostics['initialization_confirmed']}"
                f"/{initialization_min_landmarks}"
            ),
            (12, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            output,
            (
                f"Candidates: {diagnostics['initialization_candidates']}"
                f" | consistent now: "
                f"{diagnostics['initialization_matches']}"
            ),
            (12, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    cv2.putText(output, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        output,
        f"keyframes: {len(tracker.keyframes)} | landmarks: {len(tracker.landmarks)}",
        (12, output.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return output


def optical_flow_diagnostic_frame(
    frame,
    tracker,
    result,
    position,
    feature_roi_bottom_fraction,
    tracking_time_ms,
):
    output = frame.copy()
    roi_top = round(frame.shape[0] * (1.0 - feature_roi_bottom_fraction))
    output[:roi_top] = 0
    cv2.line(
        output,
        (0, roi_top),
        (frame.shape[1] - 1, roi_top),
        (255, 180, 0),
        2,
    )
    cv2.putText(
        output,
        (
            f"Optical flow: {tracking_time_ms:.1f} ms | "
            f"{1000.0 / tracking_time_ms:.1f} FPS"
        ),
        (max(12, frame.shape[1] - 380), 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    if result is None:
        label = "WAITING FOR ARUCO" if not tracker.keyframes else "LOST"
        color = (0, 180, 255) if not tracker.keyframes else (30, 30, 220)
    else:
        label = "TRACKING"
        color = (40, 200, 40)
        projected_map_points = project_map_points(
            tracker.all_map_points(),
            result,
            tracker,
            frame.shape,
            feature_roi_bottom_fraction,
        )
        for point in projected_map_points:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                1,
                (0, 255, 255),
                -1,
            )
        for point in result["outlier_points"]:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 0, 255),
                -1,
            )
        for point in result["inlier_image_points"]:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                2,
                (0, 255, 0),
                -1,
            )

        if tracker.last_diagnostics["keyframe_added"]:
            for point in tracker.last_diagnostics["new_landmark_points"]:
                cv2.circle(
                    output,
                    tuple(np.rint(point).astype(int)),
                    3,
                    (255, 0, 255),
                    -1,
                )

        cv2.putText(
            output,
            f"XYZ: {position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f} mm",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    diagnostics = tracker.last_diagnostics
    cv2.rectangle(output, (8, 76), (280, 151), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"Flow tracks: {diagnostics['flow_tracks']}",
        (15, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output,
        f"PnP inliers: {diagnostics['inliers']}",
        (15, 121),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        output,
        f"Keyframes: {len(tracker.keyframes)}",
        (15, 144),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(output, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        output,
        f"landmarks: {len(tracker.landmarks)}",
        (12, output.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return output


def load_ground_truth(path):
    timestamps = []
    positions = []
    orientations = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            timestamps.append(float(row["timestamp"]))
            positions.append(
                [float(row[axis]) for axis in ("x", "y", "z")]
            )
            orientations.append(
                [float(row[axis]) for axis in ("roll", "pitch", "yaw")]
            )

    positions = np.array(positions)
    positions = positions - positions[0]
    return np.array(timestamps), positions, np.array(orientations)


def save_comparison_figure(
    frames,
    ground_truth,
    estimate,
    component_errors,
    overall_errors,
    component_names,
    unit,
    overall_name,
    output_path,
    title,
):
    figure = plt.figure(figsize=(14, 14))
    grid = figure.add_gridspec(4, 2)

    for component in range(3):
        comparison_axis = figure.add_subplot(grid[component, 0])
        comparison_axis.plot(
            frames,
            ground_truth[:, component],
            "k--",
            label="GT",
        )
        comparison_axis.plot(
            frames,
            estimate[:, component],
            color="tab:blue",
            label="Camera",
        )
        comparison_axis.set_ylabel(
            f"{component_names[component]} [{unit}]"
        )
        comparison_axis.grid(True)
        comparison_axis.legend()

        component_rmse = np.sqrt(
            np.nanmean(component_errors[:, component] ** 2)
        )
        error_axis = figure.add_subplot(grid[component, 1])
        error_axis.plot(
            frames,
            np.abs(component_errors[:, component]),
            color="red",
        )
        error_axis.set_title(
            f"{component_names[component]} error | "
            f"RMSE: {component_rmse:.2f} {unit}"
        )
        error_axis.set_ylabel(f"Absolute error [{unit}]")
        error_axis.grid(True)

    overall_mae = np.nanmean(overall_errors)
    overall_rmse = np.sqrt(np.nanmean(overall_errors**2))
    overall_axis = figure.add_subplot(grid[3, :])
    overall_axis.plot(frames, overall_errors, color="red")
    overall_axis.set_title(
        f"{overall_name} | MAE: {overall_mae:.2f} {unit} | "
        f"RMSE: {overall_rmse:.2f} {unit}"
    )
    overall_axis.set_ylabel(f"Error [{unit}]")
    overall_axis.set_xlabel("Corrected camera time [s]")
    overall_axis.grid(True)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return overall_rmse


def create_comparison_plots(
    rows,
    gt_path,
    position_output_path,
    orientation_output_path,
    recording_name,
):
    camera_time = np.array([row["timestamp"] for row in rows])
    estimate = np.array(
        [[row["x_mm"], row["y_mm"], row["z_mm"]] for row in rows],
        dtype=float,
    )
    estimate_euler = np.array(
        [[row["roll_deg"], row["pitch_deg"], row["yaw_deg"]] for row in rows],
        dtype=float,
    )
    gt_time, gt_positions, gt_euler = load_ground_truth(gt_path)

    gt = np.column_stack(
        [
            np.interp(camera_time, gt_time, gt_positions[:, axis])
            for axis in range(3)
        ]
    )
    unwrapped_gt_euler = np.degrees(
        np.unwrap(np.radians(gt_euler), axis=0)
    )
    gt_euler = np.column_stack(
        [
            np.interp(camera_time, gt_time, unwrapped_gt_euler[:, axis])
            for axis in range(3)
        ]
    )

    within_ground_truth = camera_time >= gt_time[0]
    within_ground_truth &= camera_time <= gt_time[-1]
    camera_time = camera_time[within_ground_truth]
    estimate = estimate[within_ground_truth]
    estimate_euler = estimate_euler[within_ground_truth]
    gt = gt[within_ground_truth]
    gt_euler = gt_euler[within_ground_truth]

    valid = np.isfinite(estimate).all(axis=1)
    valid &= np.isfinite(estimate_euler).all(axis=1)
    plot_time = camera_time - min(camera_time[0], gt_time[0])

    position_component_errors = np.full_like(estimate, np.nan)
    position_component_errors[valid] = estimate[valid] - gt[valid]
    position_errors = np.full(len(estimate), np.nan)
    position_errors[valid] = np.linalg.norm(
        position_component_errors[valid],
        axis=1,
    )

    estimate_rotations = Rotation.from_euler(
        "xyz", estimate_euler[valid], degrees=True
    ).as_matrix()
    gt_rotations = Rotation.from_euler(
        "xyz", gt_euler[valid], degrees=True
    ).as_matrix()
    valid_orientation_errors = (
        estimate_euler[valid] - gt_euler[valid] + 180.0
    ) % 360.0 - 180.0
    orientation_component_errors = np.full_like(estimate_euler, np.nan)
    orientation_component_errors[valid] = valid_orientation_errors
    relative_rotations = np.transpose(gt_rotations, (0, 2, 1)) @ estimate_rotations
    valid_angular_errors = np.degrees(
        Rotation.from_matrix(relative_rotations).magnitude()
    )
    angular_errors = np.full(len(estimate), np.nan)
    angular_errors[valid] = valid_angular_errors
    tracking_coverage = 100.0 * np.mean(valid)

    position_rmse = save_comparison_figure(
        plot_time,
        gt,
        estimate,
        position_component_errors,
        position_errors,
        ["X", "Y", "Z"],
        "mm",
        "Euclidean distance on tracked frames",
        position_output_path,
        f"{recording_name}: camera position vs GT\n"
        "Data2 timestamps synchronized to Dobot | "
        f"tracked: {tracking_coverage:.1f}%",
    )

    orientation_rmse = save_comparison_figure(
        plot_time,
        gt_euler,
        estimate_euler,
        orientation_component_errors,
        angular_errors,
        ["Roll", "Pitch", "Yaw"],
        "deg",
        "Angular distance on tracked frames",
        orientation_output_path,
        f"{recording_name}: camera orientation vs GT\n"
        "Data2 timestamps synchronized to Dobot | "
        f"tracked: {tracking_coverage:.1f}%",
    )
    return position_rmse, orientation_rmse


def save_mapping_diagnostics(
    rows,
    output_path,
    recording_name,
    keyframe_translation_mm,
    keyframe_rotation_deg,
):
    frames = np.array([row["frame"] for row in rows])
    matches = np.array([row["matches"] for row in rows])
    inliers = np.array([row["inliers"] for row in rows])
    keyframe_inlier_threshold = np.array(
        [row["keyframe_inlier_threshold"] for row in rows]
    )
    new_features = np.array([row["new_features"] for row in rows])
    nearby_associations = np.array(
        [row["nearby_associations"] for row in rows]
    )
    new_landmarks = np.array([row["new_landmarks"] for row in rows])
    removed_landmarks = np.array(
        [row["removed_landmarks"] for row in rows]
    )
    landmarks = np.array([row["landmarks"] for row in rows])
    keyframe_added = np.array([row["keyframe_added"] for row in rows]) == 1

    figure, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)

    axes[0].plot(frames, matches, label="PnP correspondences")
    axes[0].plot(frames, inliers, label="PnP inliers")
    axes[0].set_ylabel("Points")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        frames,
        inliers,
        color="tab:green",
        label="PnP inliers",
    )
    axes[1].scatter(
        frames[keyframe_added],
        inliers[keyframe_added],
        color="red",
        s=18,
        label="Keyframe added",
        zorder=3,
    )
    axes[1].plot(
        frames,
        keyframe_inlier_threshold,
        color="orange",
        linestyle="--",
        label="Dynamic keyframe inlier threshold",
    )
    axes[1].set_ylabel("PnP inliers")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(frames, new_features, label="New features")
    axes[2].plot(
        frames,
        nearby_associations,
        label="Associated with nearby landmarks",
    )
    axes[2].scatter(
        frames[keyframe_added],
        new_features[keyframe_added],
        color="red",
        s=18,
        label="Keyframe added",
        zorder=3,
    )
    axes[2].set_ylabel("Features")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].bar(
        frames,
        new_landmarks,
        width=1.0,
        color="tab:blue",
        label="New landmarks",
    )
    axes[3].bar(
        frames,
        -removed_landmarks,
        width=1.0,
        color="tab:orange",
        label="Removed landmarks",
    )
    axes[3].set_ylabel("Landmark change")
    axes[3].set_xlabel("Frame")
    axes[3].grid(True)
    axes[3].legend(loc="upper left")

    landmarks_axis = axes[3].twinx()
    landmarks_axis.plot(
        frames,
        landmarks,
        color="black",
        label="Total landmarks",
    )
    landmarks_axis.set_ylabel("Total landmarks")
    landmarks_axis.legend(loc="upper right")

    figure.suptitle(
        f"{recording_name}: map expansion diagnostics | "
        f"keyframe after {keyframe_translation_mm:g} mm or "
        f"{keyframe_rotation_deg:g} deg"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_optical_flow_diagnostics(
    rows,
    output_path,
    recording_name,
    min_pnp_inliers,
    keyframe_search_trigger_inliers,
    keyframe_candidate_min_inliers,
):
    frames = np.array([row["frame"] for row in rows])
    flow_tracks = np.array([row["flow_tracks"] for row in rows])
    pnp_inliers = np.array([row["pnp_inliers"] for row in rows])
    inlier_ratio = np.array([row["pnp_inlier_ratio"] for row in rows])
    new_features = np.array([row["new_features"] for row in rows])
    nearby_associations = np.array(
        [row["nearby_associations"] for row in rows]
    )
    new_landmarks = np.array([row["new_landmarks"] for row in rows])
    landmarks = np.array([row["landmarks"] for row in rows])
    tracking_time = np.array([row["tracking_time_ms"] for row in rows])
    keyframe_added = np.array([row["keyframe_added"] for row in rows]) == 1

    figure, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)

    axes[0].plot(frames, flow_tracks, label="KLT tracks")
    axes[0].plot(frames, pnp_inliers, label="PnP inliers")
    axes[0].axhline(
        min_pnp_inliers,
        color="red",
        linestyle="--",
        label="Minimum valid pose",
    )
    axes[0].axhline(
        keyframe_search_trigger_inliers,
        color="orange",
        linestyle="--",
        label="Keyframe search trigger",
    )
    axes[0].axhline(
        keyframe_candidate_min_inliers,
        color="green",
        linestyle="--",
        label="Keyframe candidate minimum",
    )
    axes[0].set_ylabel("Points")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(frames, inlier_ratio, color="tab:green")
    axes[1].scatter(
        frames[keyframe_added],
        inlier_ratio[keyframe_added],
        color="red",
        s=22,
        label="Keyframe added",
        zorder=3,
    )
    axes[1].set_ylabel("PnP inlier ratio")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(frames, new_features, label="Detected new features")
    axes[2].plot(
        frames,
        nearby_associations,
        label="Associated with existing landmarks",
    )
    axes[2].bar(
        frames,
        new_landmarks,
        width=1.0,
        alpha=0.45,
        label="New landmarks",
    )
    axes[2].set_ylabel("Map update")
    axes[2].grid(True)
    axes[2].legend(loc="upper left")

    landmarks_axis = axes[2].twinx()
    landmarks_axis.plot(
        frames,
        landmarks,
        color="black",
        label="Total landmarks",
    )
    landmarks_axis.set_ylabel("Total landmarks")
    landmarks_axis.legend(loc="upper right")

    axes[3].plot(frames, tracking_time, color="tab:blue")
    axes[3].axhline(
        1000.0 / 30.0,
        color="red",
        linestyle="--",
        label="30 FPS limit: 33.3 ms",
    )
    axes[3].set_ylabel("Tracking time [ms]")
    axes[3].set_xlabel("Frame")
    axes[3].grid(True)
    axes[3].legend()

    figure.suptitle(f"{recording_name}: optical flow diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
