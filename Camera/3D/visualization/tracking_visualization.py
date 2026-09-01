import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from geometry.coordinate_frames import (
    tcp_displacements_to_camera_axes,
    tcp_rotations_to_camera_axes,
)


def continuous_rotation_vectors_degrees(rotations):
    quaternions = Rotation.from_matrix(rotations).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0

    vectors = quaternions[:, :3]
    norms = np.linalg.norm(vectors, axis=1)
    angles = 2.0 * np.arctan2(norms, quaternions[:, 3])
    nonzero = norms > np.finfo(float).eps
    vectors[nonzero] *= (angles[nonzero] / norms[nonzero])[:, None]
    return np.degrees(vectors)


def save_skin_mask_initialization_diagnostics(feature_matching, output_dir):
    """Save the central seed and the first adaptive skin-mask result."""
    adaptive_skin_mask = feature_matching.adaptive_skin_mask
    if adaptive_skin_mask is None:
        return False
    frame = adaptive_skin_mask.initial_frame
    seed = adaptive_skin_mask.initial_seed
    valid = adaptive_skin_mask.initial_valid
    skin_mask_result = adaptive_skin_mask.initial_result
    if frame is None or seed is None or valid is None or skin_mask_result is None:
        return False
    skin_mask = skin_mask_result.mask

    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    roi_top = round(
        height * (1.0 - feature_matching.feature_roi_bottom_fraction)
    )
    roi_color = (255, 180, 0)

    seed_image = frame.copy()
    seed_image[:roi_top] = 0
    seed_image[~valid] = 0
    seed_overlay = seed_image.copy()
    seed_overlay[seed] = (0, 255, 255)
    seed_image = cv2.addWeighted(seed_overlay, 0.45, seed_image, 0.55, 0)
    cv2.line(seed_image, (0, roi_top), (width - 1, roi_top), roi_color, 2)
    cv2.putText(
        seed_image,
        "Initial skin seed (yellow): pixels used to learn skin colour",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(output_dir / "skin_mask_initial_seed.png"), seed_image)

    result_image = np.zeros_like(frame)
    left, top, right, bottom = map(int, skin_mask_result.bounds)
    result_image[top:bottom, left:right] = frame[top:bottom, left:right]
    result_image[~valid] = 0
    cv2.line(result_image, (0, roi_top), (width - 1, roi_top), roi_color, 2)
    cv2.rectangle(
        result_image,
        (left, top),
        (right - 1, bottom - 1),
        roi_color,
        2,
    )
    contours, _ = cv2.findContours(
        skin_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(result_image, contours, -1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(
        result_image,
        "Initial skin mask | cyan: bounds | magenta: mask boundary",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(output_dir / "skin_mask_initial_result.png"), result_image)
    return True


def save_timing_diagnostics(rows, output_path, recording_name):
    stages = {
        "DISK feature extraction": "feature_extraction_ms",
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


def save_mapping_feature_video(
    video_path,
    global_map,
    output_path,
    fps,
    feature_roi_bottom_fraction,
):
    """Render all detected skin features on their mapping frames.

    These are the same 2D features selected by the map builder before SfM,
    so the overlay does not depend on a reconstructed 3D pose.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0.0:
        capture.release()
        raise RuntimeError("Mapping video does not report a valid FPS")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video writer: {output_path}")

    roi_top = round(height * (1.0 - feature_roi_bottom_fraction))
    decoded_frame = None
    decoded_frame_index = None
    for frame_index, keypoints, selection_bounds, selection_contour in zip(
        global_map.mapping_frames,
        global_map.mapping_feature_keypoints,
        global_map.mapping_feature_bounds,
        global_map.mapping_feature_contours,
    ):
        target_frame = int(frame_index)
        while (
            decoded_frame_index is None
            or decoded_frame_index < target_frame
        ):
            success, decoded_frame = capture.read()
            if not success:
                decoded_frame = None
                break
            timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            decoded_frame_index = round(timestamp_s * source_fps)
        if (
            decoded_frame is None
            or decoded_frame_index != target_frame
        ):
            continue
        frame = decoded_frame

        keypoints = np.rint(keypoints).astype(np.int32)

        output = np.zeros_like(frame)
        left, top, right, bottom = selection_bounds.astype(int)
        output[top:bottom, left:right] = frame[top:bottom, left:right]
        cv2.line(
            output,
            (0, roi_top),
            (width - 1, roi_top),
            (255, 180, 0),
            2,
        )
        cv2.rectangle(
            output,
            (left, top),
            (right - 1, bottom - 1),
            (255, 180, 0),
            2,
        )
        if len(selection_contour):
            cv2.drawContours(
                output,
                [selection_contour.astype(np.int32)],
                -1,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        for x, y in keypoints:
            cv2.circle(output, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            output,
            f"MAPPING | frame {int(frame_index)} | detected features: "
            f"{len(keypoints)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            output,
            "Green: features selected for mapping",
            (12, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            output,
            "Cyan: adaptive skin ROI | Magenta: skin mask",
            (12, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )
        writer.write(output)

    capture.release()
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
    height, width = frame.shape[:2]
    roi_top = round(height * (1.0 - feature_roi_bottom_fraction))
    adaptive_skin_mask = tracker.feature_matching.adaptive_skin_mask
    skin_mask_result = (
        None
        if adaptive_skin_mask is None
        else adaptive_skin_mask.last_result
    )
    skin_mask = None if skin_mask_result is None else skin_mask_result.mask
    selection_left = 0
    selection_top = roi_top
    selection_right = width
    selection_bottom = height
    selection_contour = None
    if skin_mask is not None and skin_mask.shape == frame.shape[:2]:
        (
            selection_left,
            selection_top,
            selection_right,
            selection_bottom,
        ) = map(int, skin_mask_result.bounds)
        contours, _ = cv2.findContours(
            skin_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            selection_contour = max(contours, key=cv2.contourArea)

    output = np.zeros_like(frame)
    output[
        selection_top:selection_bottom,
        selection_left:selection_right,
    ] = frame[
        selection_top:selection_bottom,
        selection_left:selection_right,
    ]
    cv2.line(
        output,
        (0, roi_top),
        (width - 1, roi_top),
        (255, 180, 0),
        2,
    )
    cv2.rectangle(
        output,
        (selection_left, selection_top),
        (selection_right - 1, selection_bottom - 1),
        (255, 180, 0),
        2,
    )
    if selection_contour is not None:
        cv2.drawContours(
            output,
            [selection_contour],
            -1,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
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
    initializing = not tracker.initialized
    if tracked:
        color = (40, 200, 40)
        method = tracker.last_diagnostics.get(
            "tracking_method",
            "lightglue",
        )
        label = f"TRACKING - {method.replace('_', ' ').upper()}"
    elif initializing:
        color = (0, 180, 255)
        label = "WAITING FOR GLOBAL-MAP PNP"
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

        cv2.putText(
            output,
            "Waiting for global-map PnP",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            output,
            f"Frames checked: {diagnostics['initialization_frames']}",
            (12, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    cv2.putText(output, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        output,
        "Cyan: adaptive skin ROI | Magenta: skin mask",
        (12, output.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        2,
    )
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
        label = "WAITING FOR GLOBAL-MAP PNP" if not tracker.keyframes else "LOST"
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

    return (
        np.asarray(timestamps),
        np.asarray(positions),
        np.asarray(orientations),
    )


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


def save_camera_gt_csv(
    frames,
    times_s,
    timestamps,
    estimate_positions,
    ground_truth_positions,
    estimate_rotation_vectors,
    ground_truth_rotation_vectors,
    output_path,
):
    """Save tracking poses and synchronized GT in a common camera frame."""
    fieldnames = [
        "frame",
        "time_s",
        "timestamp",
        "x_mm",
        "y_mm",
        "z_mm",
        "gt_x_mm",
        "gt_y_mm",
        "gt_z_mm",
        "rotation_x_deg",
        "rotation_y_deg",
        "rotation_z_deg",
        "gt_rotation_x_deg",
        "gt_rotation_y_deg",
        "gt_rotation_z_deg",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, frame in enumerate(frames):
            writer.writerow(
                {
                    "frame": int(frame),
                    "time_s": times_s[index],
                    "timestamp": timestamps[index],
                    "x_mm": estimate_positions[index, 0],
                    "y_mm": estimate_positions[index, 1],
                    "z_mm": estimate_positions[index, 2],
                    "gt_x_mm": ground_truth_positions[index, 0],
                    "gt_y_mm": ground_truth_positions[index, 1],
                    "gt_z_mm": ground_truth_positions[index, 2],
                    "rotation_x_deg": estimate_rotation_vectors[index, 0],
                    "rotation_y_deg": estimate_rotation_vectors[index, 1],
                    "rotation_z_deg": estimate_rotation_vectors[index, 2],
                    "gt_rotation_x_deg": ground_truth_rotation_vectors[index, 0],
                    "gt_rotation_y_deg": ground_truth_rotation_vectors[index, 1],
                    "gt_rotation_z_deg": ground_truth_rotation_vectors[index, 2],
                }
            )


def create_comparison_plots(
    rows,
    gt_path,
    position_output_path,
    orientation_output_path,
    camera_gt_output_path,
    recording_name,
    cylinder_orientation,
):
    camera_frames = np.asarray([row["frame"] for row in rows])
    camera_times_s = np.asarray([row["time_s"] for row in rows], dtype=float)
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

    interpolated_gt_positions = np.column_stack(
        [
            np.interp(camera_time, gt_time, gt_positions[:, axis])
            for axis in range(3)
        ]
    )
    unwrapped_gt_euler = np.degrees(
        np.unwrap(np.radians(gt_euler), axis=0)
    )
    interpolated_gt_euler = np.column_stack(
        [
            np.interp(camera_time, gt_time, unwrapped_gt_euler[:, axis])
            for axis in range(3)
        ]
    )

    within_ground_truth = camera_time >= gt_time[0]
    within_ground_truth &= camera_time <= gt_time[-1]
    tracker_valid = np.isfinite(estimate).all(axis=1)
    tracker_valid &= np.isfinite(estimate_euler).all(axis=1)
    reference_timestamp = camera_time[
        np.flatnonzero(tracker_valid)[0]
    ] if np.any(tracker_valid) else camera_time[0]
    camera_time = camera_time[within_ground_truth]
    estimate = estimate[within_ground_truth]
    estimate_euler = estimate_euler[within_ground_truth]
    interpolated_gt_positions = interpolated_gt_positions[
        within_ground_truth
    ]
    interpolated_gt_euler = interpolated_gt_euler[within_ground_truth]

    valid = np.isfinite(estimate).all(axis=1)
    valid &= np.isfinite(estimate_euler).all(axis=1)
    reference_gt_position = np.column_stack(
        [
            np.interp(
                [reference_timestamp],
                gt_time,
                gt_positions[:, axis],
            )
            for axis in range(3)
        ]
    )[0]
    reference_gt_euler = np.column_stack(
        [
            np.interp(
                [reference_timestamp],
                gt_time,
                unwrapped_gt_euler[:, axis],
            )
            for axis in range(3)
        ]
    )[0]
    camera_frames = camera_frames[within_ground_truth]
    camera_times_s = camera_times_s[within_ground_truth]
    reference_gt_rotation = Rotation.from_euler(
        "xyz",
        reference_gt_euler,
        degrees=True,
    ).as_matrix()
    gt_rotations = Rotation.from_euler(
        "xyz",
        interpolated_gt_euler,
        degrees=True,
    ).as_matrix()
    tcp_displacements = (
        reference_gt_rotation.T
        @ (interpolated_gt_positions - reference_gt_position).T
    ).T
    gt = tcp_displacements_to_camera_axes(
        tcp_displacements,
        cylinder_orientation,
    )
    relative_gt_rotations = reference_gt_rotation.T @ gt_rotations
    gt_rotations_camera = tcp_rotations_to_camera_axes(
        relative_gt_rotations,
        cylinder_orientation,
    )
    # Do not convert the transformed ground truth back to XYZ Euler angles.
    # Around a 90-degree pitch, the same physical pose has an equivalent Euler
    # representation with roll and yaw shifted by 180 degrees.  That made a
    # pure pitch movement appear as large roll/yaw jumps in the comparison.
    gt_rotation_vectors = continuous_rotation_vectors_degrees(
        gt_rotations_camera
    )
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
    estimate_rotation_vectors = np.full_like(estimate_euler, np.nan)
    estimate_rotation_vectors[valid] = continuous_rotation_vectors_degrees(
        estimate_rotations
    )
    relative_rotations = (
        np.transpose(gt_rotations_camera[valid], (0, 2, 1))
        @ estimate_rotations
    )
    valid_orientation_errors = Rotation.from_matrix(
        relative_rotations
    ).as_rotvec(degrees=True)
    orientation_component_errors = np.full_like(
        estimate_rotation_vectors,
        np.nan,
    )
    orientation_component_errors[valid] = valid_orientation_errors
    valid_angular_errors = np.degrees(
        Rotation.from_matrix(relative_rotations).magnitude()
    )
    angular_errors = np.full(len(estimate), np.nan)
    angular_errors[valid] = valid_angular_errors
    tracking_coverage = 100.0 * np.mean(valid)

    save_camera_gt_csv(
        camera_frames,
        camera_times_s,
        camera_time,
        estimate,
        gt,
        estimate_rotation_vectors,
        gt_rotation_vectors,
        camera_gt_output_path,
    )

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
        gt_rotation_vectors,
        estimate_rotation_vectors,
        orientation_component_errors,
        angular_errors,
        [
            "Rotation about camera X",
            "Rotation about camera Y",
            "Rotation about camera Z",
        ],
        "deg",
        "Angular distance on tracked frames",
        orientation_output_path,
        f"{recording_name}: camera orientation vs GT\n"
        "Timestamps synchronized to Dobot | "
        f"tracked: {tracking_coverage:.1f}%",
    )
    return position_rmse, orientation_rmse


def save_3d_tracking_diagnostics(
    rows,
    gt_path,
    output_path,
    recording_name,
    cylinder_orientation,
):
    """Diagnose metric pose estimation and 3D observability during tracking."""
    frames = np.asarray([row["frame"] for row in rows])
    timestamps = np.asarray([row["timestamp"] for row in rows], dtype=float)
    estimate_positions = np.asarray(
        [[row["x_mm"], row["y_mm"], row["z_mm"]] for row in rows],
        dtype=float,
    )
    estimate_euler = np.asarray(
        [[row["roll_deg"], row["pitch_deg"], row["yaw_deg"]] for row in rows],
        dtype=float,
    )
    valid = np.isfinite(estimate_positions).all(axis=1)
    valid &= np.isfinite(estimate_euler).all(axis=1)

    gt_time, gt_positions, gt_euler = load_ground_truth(gt_path)
    unwrapped_gt_euler = np.degrees(np.unwrap(np.radians(gt_euler), axis=0))
    reference_time = (
        timestamps[np.flatnonzero(valid)[0]] if np.any(valid) else timestamps[0]
    )
    reference_position = np.column_stack(
        [
            np.interp([reference_time], gt_time, gt_positions[:, axis])
            for axis in range(3)
        ]
    )[0]
    reference_euler = np.column_stack(
        [
            np.interp([reference_time], gt_time, unwrapped_gt_euler[:, axis])
            for axis in range(3)
        ]
    )[0]
    reference_rotation = Rotation.from_euler(
        "xyz", reference_euler, degrees=True
    ).as_matrix()
    interpolated_positions = np.column_stack(
        [
            np.interp(timestamps, gt_time, gt_positions[:, axis])
            for axis in range(3)
        ]
    )
    interpolated_euler = np.column_stack(
        [
            np.interp(timestamps, gt_time, unwrapped_gt_euler[:, axis])
            for axis in range(3)
        ]
    )
    gt_rotations = Rotation.from_euler(
        "xyz", interpolated_euler, degrees=True
    ).as_matrix()
    gt_positions_c0 = tcp_displacements_to_camera_axes(
        (
            reference_rotation.T
            @ (interpolated_positions - reference_position).T
        ).T,
        cylinder_orientation,
    )
    gt_rotations_c0 = tcp_rotations_to_camera_axes(
        reference_rotation.T @ gt_rotations,
        cylinder_orientation,
    )

    position_error = np.full(len(rows), np.nan)
    orientation_error = np.full(len(rows), np.nan)
    estimate_rotations = np.full((len(rows), 3, 3), np.nan)
    if np.any(valid):
        estimate_rotations[valid] = Rotation.from_euler(
            "xyz", estimate_euler[valid], degrees=True
        ).as_matrix()
        position_error[valid] = np.linalg.norm(
            estimate_positions[valid] - gt_positions_c0[valid], axis=1
        )
        orientation_error[valid] = np.degrees(
            Rotation.from_matrix(
                np.transpose(gt_rotations_c0[valid], (0, 2, 1))
                @ estimate_rotations[valid]
            ).magnitude()
        )

    def pose_increments(positions, rotations, available):
        translations = np.full(len(rows), np.nan)
        rotations_deg = np.full(len(rows), np.nan)
        for index in range(1, len(rows)):
            if not (available[index - 1] and available[index]):
                continue
            translations[index] = np.linalg.norm(
                positions[index] - positions[index - 1]
            )
            rotations_deg[index] = np.degrees(
                Rotation.from_matrix(
                    rotations[index - 1].T @ rotations[index]
                ).magnitude()
            )
        return translations, rotations_deg

    gt_translation, gt_rotation = pose_increments(
        gt_positions_c0,
        gt_rotations_c0,
        np.ones(len(rows), dtype=bool),
    )
    estimated_translation, estimated_rotation = pose_increments(
        estimate_positions,
        estimate_rotations,
        valid,
    )

    def cumulative_distance(increments):
        return np.cumsum(np.nan_to_num(increments, nan=0.0))

    def field(name):
        return np.asarray([row[name] for row in rows], dtype=float)

    primary_spread = field("inlier_3d_primary_std_mm")
    secondary_spread = field("inlier_3d_secondary_std_mm")
    minor_spread = field("inlier_3d_minor_std_mm")
    median_depth = field("inlier_depth_median_mm")
    reprojection_rmse = field("pnp_reprojection_rmse_px")
    reprojection_p95 = field("pnp_reprojection_p95_px")
    inliers = field("inliers")

    figure, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True)
    axes[0].plot(frames, primary_spread, label="Primary 3D inlier spread")
    axes[0].plot(frames, secondary_spread, label="Secondary 3D inlier spread")
    axes[0].plot(frames, minor_spread, label="Minor 3D inlier spread")
    axes[0].set_ylabel("Std. dev. [mm]")
    axes[0].set_title("Metric 3D distribution of points used by PnP")
    axes[0].grid(True)
    axes[0].legend(loc="upper left")
    depth_axis = axes[0].twinx()
    depth_axis.plot(frames, median_depth, color="black", alpha=0.6, label="Median depth")
    depth_axis.set_ylabel("Camera depth [mm]")
    depth_axis.legend(loc="upper right")

    axes[1].plot(frames, reprojection_rmse, label="Inlier reprojection RMSE")
    axes[1].plot(frames, reprojection_p95, label="Inlier reprojection p95")
    axes[1].set_ylabel("Reprojection [px]")
    axes[1].set_title("Image consistency of the accepted metric 3D pose")
    axes[1].grid(True)
    axes[1].legend(loc="upper left")
    inlier_axis = axes[1].twinx()
    inlier_axis.plot(frames, inliers, color="tab:green", alpha=0.6, label="PnP inliers")
    inlier_axis.set_ylabel("3D–2D inliers")
    inlier_axis.legend(loc="upper right")

    axes[2].plot(frames, gt_translation, color="black", label="GT translation")
    axes[2].plot(frames, estimated_translation, color="tab:blue", label="Estimated translation")
    axes[2].set_ylabel("Translation [mm/frame]")
    axes[2].set_title("Frame-to-frame physical camera motion")
    axes[2].grid(True)
    axes[2].legend(loc="upper left")
    rotation_axis = axes[2].twinx()
    rotation_axis.plot(frames, gt_rotation, color="gray", linestyle="--", label="GT rotation")
    rotation_axis.plot(frames, estimated_rotation, color="tab:orange", label="Estimated rotation")
    rotation_axis.set_ylabel("Rotation [deg/frame]")
    rotation_axis.legend(loc="upper right")

    axes[3].plot(frames, cumulative_distance(gt_translation), color="black", label="GT cumulative path")
    axes[3].plot(frames, cumulative_distance(estimated_translation), color="tab:blue", label="Estimated cumulative path")
    axes[3].set_ylabel("Path length [mm]")
    axes[3].set_title("Accumulated translation; untracked gaps add no estimate")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].plot(frames, position_error, color="tab:red", label="3D position error")
    axes[4].set_ylabel("Position error [mm]")
    axes[4].set_xlabel("Frame")
    axes[4].set_title("Pose error against GT on accepted tracking frames")
    axes[4].grid(True)
    axes[4].legend(loc="upper left")
    orientation_axis = axes[4].twinx()
    orientation_axis.plot(frames, orientation_error, color="tab:purple", label="Orientation error")
    orientation_axis.set_ylabel("Orientation error [deg]")
    orientation_axis.legend(loc="upper right")

    figure.suptitle(
        f"{recording_name}: 3D pose and geometry diagnostics | "
        f"tracked: {100.0 * np.mean(valid):.1f}%"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


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
