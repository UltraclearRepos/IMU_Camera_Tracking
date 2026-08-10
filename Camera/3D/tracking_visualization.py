import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def save_timing_diagnostics(rows, output_path, recording_name):
    stages = {
        "DISK feature extraction": "feature_extraction_ms",
        "ArUco initial pose": "aruco_pose_ms",
        "Global map preparation": "global_map_projection_ms",
        "LightGlue matching": "lightglue_ms",
        "Optical flow": "optical_flow_ms",
        "PnP RANSAC": "pnp_ransac_ms",
        "PnP refinement": "pnp_refine_ms",
        "Map coverage": "map_coverage_ms",
        "Map update": "map_update_ms",
    }
    labels = []
    means = []
    medians = []
    p95_values = []
    for label, field in stages.items():
        values = np.array([row[field] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        labels.append(label)
        means.append(np.mean(values))
        medians.append(np.median(values))
        p95_values.append(np.percentile(values, 95))

    positions = np.arange(len(labels))
    bar_height = 0.25
    figure, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].barh(
        positions - bar_height,
        means,
        height=bar_height,
        label="Mean",
    )
    axes[0].barh(
        positions,
        medians,
        height=bar_height,
        label="Median",
    )
    axes[0].barh(
        positions + bar_height,
        p95_values,
        height=bar_height,
        label="95th percentile",
    )
    axes[0].set_yticks(positions, labels)
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_title("Time by algorithm stage")
    axes[0].grid(axis="x")
    axes[0].legend()
    axes[0].invert_yaxis()

    frames = np.array([row["frame"] for row in rows])
    total_time = np.array(
        [row["tracking_time_ms"] for row in rows],
        dtype=float,
    )
    axes[1].plot(frames, total_time, color="black", label="Total tracking")
    for label, field in stages.items():
        values = np.array([row[field] for row in rows], dtype=float)
        if np.isfinite(values).any():
            axes[1].plot(frames, values, label=label, alpha=0.75)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Time [ms]")
    axes[1].set_title("Processing time per frame")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    figure.suptitle(f"{recording_name}: tracking performance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


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


def image_points_on_skin_plane(
    image_points,
    result,
    camera_matrix,
    distortion,
    maximum_view_distance_mm,
):
    normalized_points = cv2.undistortPoints(
        np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix,
        distortion,
    ).reshape(-1, 2)
    camera_rays = np.column_stack(
        [normalized_points, np.ones(len(normalized_points))]
    )

    camera_to_map = result["R"].T
    camera_position = (
        -camera_to_map @ result["t"].reshape(3)
    )
    map_rays = (camera_to_map @ camera_rays.T).T
    distances = -camera_position[2] / map_rays[:, 2]
    intersections = camera_position + distances[:, None] * map_rays

    planar_directions = map_rays[:, :2]
    planar_directions /= np.linalg.norm(
        planar_directions,
        axis=1,
        keepdims=True,
    )
    planar_offsets = intersections[:, :2] - camera_position[:2]
    planar_distances = np.linalg.norm(planar_offsets, axis=1)
    clipped = (distances <= 0.0) | (
        planar_distances > maximum_view_distance_mm
    )
    intersections[clipped, :2] = (
        camera_position[:2]
        + maximum_view_distance_mm * planar_directions[clipped]
    )
    intersections[clipped, 2] = 0.0
    return intersections


def camera_view_on_skin_plane(
    result,
    camera_matrix,
    distortion,
    frame_shape,
    maximum_view_distance_mm,
):
    height, width = frame_shape[:2]
    image_corners = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ]
    )
    return image_points_on_skin_plane(
        image_corners,
        result,
        camera_matrix,
        distortion,
        maximum_view_distance_mm,
    )


def coverage_grid_on_skin_plane(
    tracker,
    result,
    frame_shape,
    maximum_view_distance_mm,
):
    height, width = frame_shape[:2]
    roi_top = height * (1.0 - tracker.feature_roi_bottom_fraction)
    grid_lines = []

    for column in range(tracker.map_coverage_grid_columns + 1):
        x = column * (width - 1.0) / tracker.map_coverage_grid_columns
        grid_lines.append(np.array([[x, roi_top], [x, height - 1.0]]))

    for row in range(tracker.map_coverage_grid_rows + 1):
        y = roi_top + row * (
            height - 1.0 - roi_top
        ) / tracker.map_coverage_grid_rows
        grid_lines.append(np.array([[0.0, y], [width - 1.0, y]]))

    return [
        image_points_on_skin_plane(
            line,
            result,
            tracker.camera_matrix,
            tracker.distortion,
            maximum_view_distance_mm,
        )
        for line in grid_lines
    ]


def create_top_view_state(
    tracker,
    result,
    frame_shape,
    frame_index,
    maximum_view_distance_mm,
):
    state = {
        "frame_index": frame_index,
        "map_points": tracker.all_map_points().copy(),
        "inlier_map_points": np.empty((0, 3)),
        "camera_position": None,
        "view_polygon": np.empty((0, 3)),
        "coverage_grid_lines": [],
        "landmarks": len(tracker.landmarks),
        "inliers": 0,
        "status": (
            "INITIALIZING"
            if not tracker.keyframes
            else "LOST"
        ),
    }
    if result is None:
        return state

    state["inlier_map_points"] = result["inlier_map_points"].copy()
    state["camera_position"] = (
        -result["R"].T @ result["t"].reshape(3)
    )
    state["view_polygon"] = camera_view_on_skin_plane(
        result,
        tracker.camera_matrix,
        tracker.distortion,
        frame_shape,
        maximum_view_distance_mm,
    )
    state["coverage_grid_lines"] = coverage_grid_on_skin_plane(
        tracker,
        result,
        frame_shape,
        maximum_view_distance_mm,
    )
    state["inliers"] = result["inliers"]
    state["status"] = "TRACKING"
    return state


def top_view_bounds(states, padding_mm):
    viewed_areas = [
        state["view_polygon"][:, :2]
        for state in states
        if len(state["view_polygon"]) > 0
    ]
    if viewed_areas:
        points = np.vstack(viewed_areas)
    else:
        mapped_areas = [
            state["map_points"][:, :2]
            for state in states
            if len(state["map_points"]) > 0
        ]
        points = (
            np.vstack(mapped_areas)
            if mapped_areas
            else np.zeros((1, 2))
        )

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    view_size_mm = np.max(maximum - minimum) + 2.0 * padding_mm
    half_size = view_size_mm / 2.0
    return np.array(
        [
            center[0] - half_size,
            center[0] + half_size,
            center[1] - half_size,
            center[1] + half_size,
        ]
    )


def top_view_frame(state, bounds, view_size_pixels):
    output = np.full(
        (view_size_pixels, view_size_pixels, 3),
        24,
        dtype=np.uint8,
    )
    margin = 55
    drawing_size = view_size_pixels - 2 * margin
    x_min, x_max, y_min, y_max = bounds
    view_size_mm = x_max - x_min
    scale = drawing_size / view_size_mm

    def to_pixels(points):
        points = np.asarray(points).reshape(-1, 2)
        pixels = np.empty_like(points)
        pixels[:, 0] = margin + (points[:, 0] - x_min) * scale
        pixels[:, 1] = margin + (y_max - points[:, 1]) * scale
        return np.rint(pixels).astype(np.int32)

    def draw_dashed_line(start, end):
        delta = end.astype(float) - start.astype(float)
        length = np.linalg.norm(delta)
        direction = delta / length
        for offset in np.arange(0.0, length, 8.0):
            segment_start = start + direction * offset
            segment_end = start + direction * min(offset + 4.0, length)
            cv2.line(
                output,
                tuple(np.rint(segment_start).astype(int)),
                tuple(np.rint(segment_end).astype(int)),
                (85, 85, 85),
                1,
            )

    grid_step_mm = 50.0
    grid_x = np.arange(
        np.ceil(x_min / grid_step_mm) * grid_step_mm,
        x_max,
        grid_step_mm,
    )
    grid_y = np.arange(
        np.ceil(y_min / grid_step_mm) * grid_step_mm,
        y_max,
        grid_step_mm,
    )
    for value in grid_x:
        vertical = to_pixels([[value, y_min], [value, y_max]])
        cv2.line(output, vertical[0], vertical[1], (55, 55, 55), 1)
    for value in grid_y:
        horizontal = to_pixels([[x_min, value], [x_max, value]])
        cv2.line(output, horizontal[0], horizontal[1], (55, 55, 55), 1)

    if y_min <= 0.0 <= y_max:
        axis_x = to_pixels([[x_min, 0.0], [x_max, 0.0]])
        cv2.line(output, axis_x[0], axis_x[1], (40, 80, 230), 2)
    if x_min <= 0.0 <= x_max:
        axis_y = to_pixels([[0.0, y_min], [0.0, y_max]])
        cv2.line(output, axis_y[0], axis_y[1], (40, 210, 80), 2)

    for grid_line in state["coverage_grid_lines"]:
        line_pixels = to_pixels(grid_line[:, :2])
        draw_dashed_line(line_pixels[0], line_pixels[1])

    map_points = state["map_points"]
    if len(map_points) > 0:
        for point in to_pixels(map_points[:, :2]):
            cv2.circle(output, tuple(point), 2, (0, 220, 220), -1)

    camera_position = state["camera_position"]
    if camera_position is not None:
        polygon_pixels = to_pixels(state["view_polygon"][:, :2])

        overlay = output.copy()
        cv2.fillPoly(
            overlay,
            [polygon_pixels],
            (120, 75, 20),
        )
        output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0.0)
        cv2.polylines(
            output,
            [polygon_pixels],
            True,
            (255, 180, 40),
            2,
        )

        for point in to_pixels(state["inlier_map_points"][:, :2]):
            cv2.circle(output, tuple(point), 3, (0, 255, 0), -1)

        camera_pixel = to_pixels([camera_position[:2]])[0]
        cv2.circle(output, tuple(camera_pixel), 7, (255, 100, 30), -1)
        cv2.putText(
            output,
            f"Camera Z: {camera_position[2]:.1f} mm",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    status = state["status"]
    if status == "TRACKING":
        status_color = (0, 220, 0)
    elif status == "INITIALIZING":
        status_color = (0, 180, 255)
    else:
        status_color = (30, 30, 220)

    cv2.putText(
        output,
        f"Frame: {state['frame_index']} | {status}",
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )
    cv2.putText(
        output,
        (
            f"Landmarks: {state['landmarks']} | "
            f"PnP inliers: {state['inliers']}"
        ),
        (15, view_size_pixels - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output,
        f"View: {view_size_mm:.0f} x {view_size_mm:.0f} mm",
        (view_size_pixels - 275, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        2,
    )
    return output


def save_top_view_video(
    states,
    output_path,
    fps,
    view_size_pixels,
    padding_mm,
):
    bounds = top_view_bounds(states, padding_mm)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (view_size_pixels, view_size_pixels),
    )
    for state in states:
        writer.write(
            top_view_frame(state, bounds, view_size_pixels)
        )
    writer.release()


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
            timestamps.append(float(row["sync_timestamp"]))
            positions.append(
                [float(row[axis]) for axis in ("x", "y", "z")]
            )
            orientations.append(
                [float(row[axis]) for axis in ("roll", "pitch", "yaw")]
            )

    positions = np.array(positions)
    positions = positions - positions[0]
    orientation_matrices = Rotation.from_euler(
        "xyz",
        np.array(orientations),
        degrees=True,
    ).as_matrix()
    relative_orientation_matrices = (
        orientation_matrices[0].T @ orientation_matrices
    )
    relative_orientations = Rotation.from_matrix(
        relative_orientation_matrices
    ).as_euler("xyz", degrees=True)
    return np.array(timestamps), positions, relative_orientations


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
    estimate_label="Camera",
    x_label="Corrected camera time [s]",
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
            label=estimate_label,
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
    overall_axis.set_xlabel(x_label)
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
        "Timestamps synchronized to Dobot | "
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
        "Timestamps synchronized to Dobot | "
        f"tracked: {tracking_coverage:.1f}%",
    )
    return position_rmse, orientation_rmse


def save_mapping_diagnostics(
    rows,
    output_path,
    recording_name,
):
    frames = np.array([row["frame"] for row in rows])
    matches = np.array([row["matches"] for row in rows])
    inliers = np.array([row["inliers"] for row in rows])
    required_matches = np.array(
        [row["required_matches"] for row in rows]
    )
    required_inliers = np.array(
        [row["required_inliers"] for row in rows]
    )
    visible_landmarks = np.array(
        [row["visible_landmarks"] for row in rows]
    )
    map_coverage_ratio = np.array(
        [row["map_coverage_ratio"] for row in rows]
    )
    map_expansion_coverage_threshold = np.array(
        [row["map_expansion_coverage_threshold"] for row in rows]
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

    figure, axes = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

    axes[0].plot(frames, matches, label="PnP correspondences")
    axes[0].plot(frames, inliers, label="PnP inliers")
    axes[0].plot(
        frames,
        required_matches,
        linestyle="--",
        label="Required correspondences",
    )
    axes[0].plot(
        frames,
        required_inliers,
        linestyle=":",
        color="black",
        label="Required PnP inliers",
    )
    axes[0].scatter(
        frames[keyframe_added],
        inliers[keyframe_added],
        color="red",
        s=24,
        label="Map expanded",
        zorder=3,
    )
    axes[0].set_ylabel("Points")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        frames,
        visible_landmarks,
        color="tab:green",
        label="Visible global landmarks",
    )
    axes[1].plot(
        frames,
        inliers,
        color="tab:blue",
        label="PnP inliers",
    )
    axes[1].plot(
        frames,
        required_inliers,
        color="black",
        linestyle=":",
        label="Required PnP inliers",
    )
    axes[1].set_ylabel("Points")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(
        frames,
        100.0 * map_coverage_ratio,
        color="tab:green",
        label="Mean grid coverage",
    )
    axes[2].plot(
        frames,
        100.0 * map_expansion_coverage_threshold,
        color="orange",
        linestyle="--",
        label="Map expansion threshold",
    )
    axes[2].scatter(
        frames[keyframe_added],
        100.0 * map_coverage_ratio[keyframe_added],
        color="red",
        s=18,
        label="Keyframe added",
        zorder=3,
    )
    axes[2].set_ylabel("ROI coverage [%]")
    axes[2].set_ylim(0.0, 105.0)
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(frames, new_features, label="New features")
    axes[3].plot(
        frames,
        nearby_associations,
        label="Associated with nearby landmarks",
    )
    axes[3].scatter(
        frames[keyframe_added],
        new_features[keyframe_added],
        color="red",
        s=18,
        label="Keyframe added",
        zorder=3,
    )
    axes[3].set_ylabel("Features")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].bar(
        frames,
        new_landmarks,
        width=1.0,
        color="tab:blue",
        label="New landmarks",
    )
    axes[4].bar(
        frames,
        -removed_landmarks,
        width=1.0,
        color="tab:orange",
        label="Removed landmarks",
    )
    axes[4].set_ylabel("Landmark change")
    axes[4].set_xlabel("Frame")
    axes[4].grid(True)
    axes[4].legend(loc="upper left")

    landmarks_axis = axes[4].twinx()
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
        "expand when visible map coverage falls below the threshold"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_hybrid_method_diagnostics(
    rows,
    output_path,
    recording_name,
    window_frames=30,
):
    window_starts = range(0, len(rows), window_frames)
    window_labels = []
    optical_flow_counts = []
    lightglue_counts = []
    lost_counts = []

    for window_start in window_starts:
        window_rows = rows[window_start : window_start + window_frames]
        window_labels.append(
            f"{window_rows[0]['frame']}-{window_rows[-1]['frame']}"
        )
        optical_flow_counts.append(
            sum(
                row["tracking_method"] == "optical_flow"
                for row in window_rows
            )
        )
        lightglue_counts.append(
            sum(
                row["tracking_method"] == "lightglue"
                for row in window_rows
            )
        )
        lost_counts.append(
            sum(not row["tracked"] for row in window_rows)
        )

    x = np.arange(len(window_labels))
    figure_width = max(12.0, 0.55 * len(window_labels))
    figure, axis = plt.subplots(figsize=(figure_width, 5.5))

    axis.bar(
        x,
        optical_flow_counts,
        color="tab:green",
        label="Optical flow",
    )
    axis.bar(
        x,
        lightglue_counts,
        bottom=optical_flow_counts,
        color="tab:blue",
        label="LightGlue",
    )
    axis.plot(
        x,
        lost_counts,
        color="tab:red",
        marker="o",
        linestyle="--",
        linewidth=1.5,
        label="Lost frames",
    )

    for bar_index, optical_flow_count in enumerate(optical_flow_counts):
        if optical_flow_count > 0:
            axis.text(
                bar_index,
                optical_flow_count / 2,
                str(optical_flow_count),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    axis.set_xticks(x, window_labels, rotation=45, ha="right")
    axis.set_xlabel(f"Frame range ({window_frames}-frame windows)")
    axis.set_ylabel("Frames")
    axis.set_ylim(0, window_frames + 2)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    axis.set_title(
        f"{recording_name}: LightGlue and optical-flow usage over time"
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
