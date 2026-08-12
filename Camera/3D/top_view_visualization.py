from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PLOT_MARGIN_PX = 70
BACKGROUND_COLOR = (18, 18, 18)
GRID_COLOR = (55, 55, 55)
TEXT_COLOR = (235, 235, 235)


def camera_position(R_map_to_camera, t_map_to_camera):
    return -R_map_to_camera.T @ t_map_to_camera.reshape(3)


def fixed_bounds(map_points, camera_positions, padding_mm):
    points = [map_points[:, :2]]
    if len(camera_positions):
        points.append(np.asarray(camera_positions)[:, :2])
    points = np.vstack(points)

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = 0.5 * (minimum + maximum)
    span = max(np.max(maximum - minimum), 1.0) + 2.0 * padding_mm
    half_span = 0.5 * span
    return np.array(
        [
            center[0] - half_span,
            center[0] + half_span,
            center[1] - half_span,
            center[1] + half_span,
        ]
    )


def world_to_pixels(points_xy, bounds, size_px):
    x_min, x_max, y_min, y_max = bounds
    plot_size = size_px - 2 * PLOT_MARGIN_PX
    pixels = np.empty_like(points_xy, dtype=float)
    pixels[:, 0] = PLOT_MARGIN_PX + (
        (points_xy[:, 0] - x_min) / (x_max - x_min) * plot_size
    )
    pixels[:, 1] = size_px - PLOT_MARGIN_PX - (
        (points_xy[:, 1] - y_min) / (y_max - y_min) * plot_size
    )
    return np.rint(pixels).astype(int)


def nice_grid_step(span):
    raw_step = span / 8.0
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1.0:
        return magnitude
    if normalized <= 2.0:
        return 2.0 * magnitude
    if normalized <= 5.0:
        return 5.0 * magnitude
    return 10.0 * magnitude


def dashed_line(image, point1, point2, color, thickness=1, dash_px=5):
    point1 = np.asarray(point1, dtype=float)
    point2 = np.asarray(point2, dtype=float)
    delta = point2 - point1
    length = np.linalg.norm(delta)
    if length == 0.0:
        return

    direction = delta / length
    for start in np.arange(0.0, length, 2.0 * dash_px):
        end = min(start + dash_px, length)
        segment_start = np.rint(point1 + start * direction).astype(int)
        segment_end = np.rint(point1 + end * direction).astype(int)
        cv2.line(
            image,
            tuple(segment_start),
            tuple(segment_end),
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_grid(image, bounds):
    x_min, x_max, y_min, y_max = bounds
    step = nice_grid_step(max(x_max - x_min, y_max - y_min))

    for x in np.arange(np.ceil(x_min / step) * step, x_max, step):
        pixels = world_to_pixels(
            np.array([[x, y_min], [x, y_max]]),
            bounds,
            image.shape[0],
        )
        dashed_line(image, pixels[0], pixels[1], GRID_COLOR)
        cv2.putText(
            image,
            f"{x:.0f}",
            (pixels[0, 0] - 12, image.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

    for y in np.arange(np.ceil(y_min / step) * step, y_max, step):
        pixels = world_to_pixels(
            np.array([[x_min, y], [x_max, y]]),
            bounds,
            image.shape[0],
        )
        dashed_line(image, pixels[0], pixels[1], GRID_COLOR)
        cv2.putText(
            image,
            f"{y:.0f}",
            (18, pixels[0, 1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

    cv2.rectangle(
        image,
        (PLOT_MARGIN_PX, PLOT_MARGIN_PX),
        (image.shape[0] - PLOT_MARGIN_PX, image.shape[0] - PLOT_MARGIN_PX),
        (100, 100, 100),
        1,
    )
    cv2.putText(
        image,
        "X [mm]",
        (image.shape[1] // 2 - 25, image.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
    )
    cv2.putText(
        image,
        "Y [mm]",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
    )


def depth_colors(points, z_min, z_max):
    if z_max <= z_min:
        normalized = np.zeros(len(points), dtype=np.uint8)
    else:
        normalized = np.clip(
            255.0 * (points[:, 2] - z_min) / (z_max - z_min),
            0.0,
            255.0,
        ).astype(np.uint8)
    return cv2.applyColorMap(normalized[:, None], cv2.COLORMAP_TURBO)[
        :, 0, :
    ]


def draw_landmarks(image, points, colors, bounds, radius=1):
    pixels = world_to_pixels(points[:, :2], bounds, image.shape[0])
    for pixel, color in zip(pixels, colors):
        cv2.circle(
            image,
            tuple(pixel),
            radius,
            tuple(map(int, color)),
            -1,
            cv2.LINE_AA,
        )


def draw_camera(image, position, direction, bounds, color):
    camera_pixel = world_to_pixels(
        position[None, :2], bounds, image.shape[0]
    )[0]
    direction_xy = direction[:2]
    direction_norm = np.linalg.norm(direction_xy)
    if direction_norm > 0.0:
        direction_xy = direction_xy / direction_norm
        arrow_end = world_to_pixels(
            (position[:2] + 12.0 * direction_xy)[None],
            bounds,
            image.shape[0],
        )[0]
        cv2.arrowedLine(
            image,
            tuple(camera_pixel),
            tuple(arrow_end),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
    cv2.circle(image, tuple(camera_pixel), 5, color, -1, cv2.LINE_AA)


def create_video_writer(path, fps, size_px):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (size_px, size_px),
    )


def save_retrieval_diagnostics(diagnostics, output_path):
    if not diagnostics:
        return

    entries_by_frame = {}
    for entry in diagnostics:
        entries_by_frame.setdefault(entry["current_frame"], []).append(
            entry
        )

    figure, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    rank_colors = ["tab:blue", "tab:orange", "tab:purple"]
    maximum_rank = max(map(len, entries_by_frame.values()))
    for rank in range(maximum_rank):
        ranked_entries = [
            entries[rank]
            for entries in entries_by_frame.values()
            if rank < len(entries)
        ]
        axes[0].plot(
            [entry["current_frame"] for entry in ranked_entries],
            [entry["retrieval_score"] for entry in ranked_entries],
            ".-",
            color=rank_colors[rank % len(rank_colors)],
            label=f"Retrieved rank {rank + 1}",
        )

    frames = [entry["current_frame"] for entry in diagnostics]
    axes[1].scatter(
        frames,
        [entry["raw_matches"] for entry in diagnostics],
        s=10,
        color="tab:blue",
        label="LightGlue matches",
    )
    axes[1].scatter(
        frames,
        [entry["inliers"] for entry in diagnostics],
        s=10,
        color="tab:green",
        label="Geometry inliers",
    )

    covered_cells = [
        min(
            entry["covered_cells_previous"],
            entry["covered_cells_current"],
        )
        for entry in diagnostics
    ]
    axes[2].scatter(
        frames,
        covered_cells,
        s=10,
        color="tab:purple",
        label="Covered cells in both images",
    )
    axes[2].axhline(
        diagnostics[0]["required_covered_cells"],
        color="tab:red",
        linestyle="--",
        label="Required covered cells",
    )

    unique_frames = sorted(entries_by_frame)
    accepted_counts = [
        sum(entry["accepted"] for entry in entries_by_frame[frame])
        for frame in unique_frames
    ]
    axes[3].bar(
        unique_frames,
        accepted_counts,
        width=1.0,
        color="tab:green",
        label="Accepted retrieved pairs",
    )

    axes[0].set_ylabel("MNN retrieval score")
    axes[1].set_ylabel("Points")
    axes[2].set_ylabel("Image-grid cells")
    axes[3].set_ylabel("Accepted pairs")
    axes[3].set_xlabel("Current mapping frame")
    axes[3].set_ylim(0, maximum_rank + 0.5)
    axes[0].set_title(
        "Old-frame retrieval: single-frame scoring, old-sequence support and geometry"
    )
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_map_build_top_view(
    global_map,
    output_path,
    fps,
    size_px,
    padding_mm,
):
    candidate_points = global_map["candidate_positions"]
    available_frames = global_map["candidate_available_frames"]
    selected_indices = global_map["selected_candidate_indices"]
    mapping_frames = global_map["mapping_frames"]
    camera_positions = global_map["mapping_camera_positions"]
    camera_headings = global_map["mapping_camera_headings"]
    retrieval_diagnostics = global_map["retrieval_diagnostics"]
    retrieval_by_current_frame = {}
    for entry in retrieval_diagnostics:
        retrieval_by_current_frame.setdefault(
            entry["current_frame"],
            [],
        ).append(entry)
    camera_position_by_frame = dict(
        zip(mapping_frames, camera_positions)
    )
    bounds = fixed_bounds(candidate_points, camera_positions, padding_mm)
    z_min = np.min(candidate_points[:, 2])
    z_max = np.max(candidate_points[:, 2])
    colors = depth_colors(candidate_points, z_min, z_max)
    writer = create_video_writer(output_path, fps, size_px)

    for camera_index, frame_index in enumerate(mapping_frames):
        image = np.full((size_px, size_px, 3), BACKGROUND_COLOR, np.uint8)
        draw_grid(image, bounds)

        present = available_frames <= frame_index
        added_now = available_frames == frame_index
        draw_landmarks(
            image,
            candidate_points[present],
            colors[present],
            bounds,
        )
        if np.any(added_now):
            new_pixels = world_to_pixels(
                candidate_points[added_now, :2], bounds, size_px
            )
            for pixel in new_pixels:
                cv2.circle(
                    image,
                    tuple(pixel),
                    3,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        trajectory = world_to_pixels(
            camera_positions[: camera_index + 1, :2], bounds, size_px
        )
        if len(trajectory) > 1:
            cv2.polylines(
                image,
                [trajectory],
                False,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )
        draw_camera(
            image,
            camera_positions[camera_index],
            camera_headings[camera_index],
            bounds,
            (0, 180, 255),
        )

        current_retrieval = retrieval_by_current_frame.get(
            int(frame_index),
            [],
        )
        current_camera_position = camera_position_by_frame.get(frame_index)
        for entry in current_retrieval:
            previous_camera_position = camera_position_by_frame.get(
                entry["previous_frame"]
            )
            if (
                current_camera_position is None
                or previous_camera_position is None
            ):
                continue
            connection_pixels = world_to_pixels(
                np.array(
                    [
                        previous_camera_position[:2],
                        current_camera_position[:2],
                    ]
                ),
                bounds,
                size_px,
            )
            color = (
                (80, 230, 80)
                if entry["accepted"]
                else (80, 80, 220)
            )
            dashed_line(
                image,
                connection_pixels[0],
                connection_pixels[1],
                color,
                thickness=2,
                dash_px=4,
            )

        cv2.putText(
            image,
            f"MAP BUILD | frame {frame_index}",
            (PLOT_MARGIN_PX, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 230, 80),
            2,
        )
        cv2.putText(
            image,
            "Candidates available after 3 observations: "
            f"{np.sum(present)}/{len(candidate_points)} | "
            f"new: {np.sum(added_now)}",
            (PLOT_MARGIN_PX, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            TEXT_COLOR,
            1,
        )
        for row, entry in enumerate(current_retrieval):
            status = "OK" if entry["accepted"] else "rejected"
            cv2.putText(
                image,
                f"retrieval <- {entry['previous_frame']} | "
                f"MNN {entry['votes']} | "
                f"score {entry['retrieval_score']:.1f} | "
                f"similarity {entry['mean_similarity']:.2f} | "
                f"matches {entry['raw_matches']} | "
                f"inliers {entry['inliers']} | {status}",
                (PLOT_MARGIN_PX, 76 + 20 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (
                    (80, 230, 80)
                    if entry["accepted"]
                    else (100, 100, 230)
                ),
                1,
            )
        cv2.putText(
            image,
            f"Landmark color: Z = {z_min:.1f} .. {z_max:.1f} mm",
            (size_px - 330, size_px - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            TEXT_COLOR,
            1,
        )
        writer.write(image)

    selected = np.zeros(len(candidate_points), dtype=bool)
    selected[selected_indices] = True
    hold_frames = max(1, round(2.0 * fps))

    def final_stage_frame(title, subtitle):
        image = np.full((size_px, size_px, 3), BACKGROUND_COLOR, np.uint8)
        draw_grid(image, bounds)
        trajectory = world_to_pixels(
            camera_positions[:, :2], bounds, size_px
        )
        if len(trajectory) > 1:
            cv2.polylines(
                image,
                [trajectory],
                False,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )
        draw_camera(
            image,
            camera_positions[-1],
            camera_headings[-1],
            bounds,
            (0, 180, 255),
        )
        cv2.putText(
            image,
            title,
            (PLOT_MARGIN_PX, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 230, 80),
            2,
        )
        cv2.putText(
            image,
            subtitle,
            (PLOT_MARGIN_PX, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
        )
        return image

    image = final_stage_frame(
        "RECONSTRUCTION COMPLETE",
        f"Quality-filtered candidates: {len(candidate_points)}",
    )
    draw_landmarks(image, candidate_points, colors, bounds)
    for _ in range(hold_frames):
        writer.write(image)

    image = final_stage_frame(
        "GLOBAL MAP REDUCTION",
        f"kept: {np.sum(selected)} | removed: {np.sum(~selected)}",
    )
    removed_colors = np.tile((60, 60, 220), (np.sum(~selected), 1))
    selected_colors = np.tile((60, 220, 60), (np.sum(selected), 1))
    draw_landmarks(
        image,
        candidate_points[~selected],
        removed_colors,
        bounds,
    )
    draw_landmarks(
        image,
        candidate_points[selected],
        selected_colors,
        bounds,
        radius=2,
    )
    for _ in range(hold_frames):
        writer.write(image)

    image = final_stage_frame(
        "FINAL FROZEN MAP",
        f"Selected landmarks: {np.sum(selected)}/{len(candidate_points)}",
    )
    draw_landmarks(
        image,
        candidate_points[selected],
        colors[selected],
        bounds,
        radius=2,
    )
    for _ in range(hold_frames):
        writer.write(image)

    writer.release()


def camera_view_polygon(
    tracker,
    R_map_to_camera,
    t_map_to_camera,
    frame_shape,
    visible_map_points,
):
    if not len(visible_map_points):
        return np.empty((0, 3))

    camera_points = (
        R_map_to_camera @ visible_map_points.T
    ).T + t_map_to_camera
    depth = np.median(camera_points[:, 2])
    height, width = frame_shape[:2]
    roi_top = height * (1.0 - tracker.feature_roi_bottom_fraction)
    corners = np.array(
        [
            [0.0, roi_top],
            [width - 1.0, roi_top],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )
    normalized = cv2.undistortPoints(
        corners.reshape(-1, 1, 2),
        tracker.camera_matrix,
        tracker.distortion,
    ).reshape(-1, 2)
    points_camera = depth * np.column_stack(
        [normalized, np.ones(len(normalized))]
    )
    return (
        tracker.R_map_to_camera.T
        @ (points_camera - tracker.t_map_to_camera).T
    ).T


def create_tracking_top_view_state(tracker, result, frame_shape, frame_index):
    R_map_to_camera = tracker.R_map_to_camera
    t_map_to_camera = tracker.t_map_to_camera
    camera_points = (
        R_map_to_camera @ tracker.map_points.T
    ).T + t_map_to_camera
    projected, _ = cv2.projectPoints(
        tracker.map_points,
        cv2.Rodrigues(R_map_to_camera)[0],
        t_map_to_camera,
        tracker.camera_matrix,
        tracker.distortion,
    )
    projected = projected.reshape(-1, 2)
    height, width = frame_shape[:2]
    roi_top = height * (1.0 - tracker.feature_roi_bottom_fraction)
    visible = camera_points[:, 2] > 0.0
    visible &= projected[:, 0] >= 0.0
    visible &= projected[:, 0] < width
    visible &= projected[:, 1] >= roi_top
    visible &= projected[:, 1] < height
    visible_ids = np.flatnonzero(visible)

    inlier_ids = np.empty(0, dtype=np.int64)
    if result is not None:
        inlier_ids = result["inlier_landmark_ids"]

    return {
        "frame": frame_index,
        "tracked": result is not None,
        "camera_position": camera_position(
            R_map_to_camera,
            t_map_to_camera,
        ),
        "camera_heading": R_map_to_camera.T
        @ np.array([0.0, -1.0, 0.0]),
        "visible_ids": visible_ids,
        "inlier_ids": inlier_ids,
        "view_polygon": camera_view_polygon(
            tracker,
            R_map_to_camera,
            t_map_to_camera,
            frame_shape,
            tracker.map_points[visible],
        ),
    }


def save_tracking_top_view(
    global_map,
    states,
    output_path,
    fps,
    size_px,
    padding_mm,
):
    map_points = global_map["positions"]
    camera_positions = np.array(
        [state["camera_position"] for state in states]
    )
    bounds = fixed_bounds(map_points, camera_positions, padding_mm)
    z_min = np.min(map_points[:, 2])
    z_max = np.max(map_points[:, 2])
    map_colors = depth_colors(map_points, z_min, z_max)
    writer = create_video_writer(output_path, fps, size_px)

    for state_index, state in enumerate(states):
        image = np.full((size_px, size_px, 3), BACKGROUND_COLOR, np.uint8)
        draw_grid(image, bounds)
        draw_landmarks(image, map_points, map_colors, bounds)

        visible_points = map_points[state["visible_ids"]]
        visible_pixels = world_to_pixels(
            visible_points[:, :2], bounds, size_px
        )
        for pixel in visible_pixels:
            cv2.circle(image, tuple(pixel), 3, (0, 200, 0), 1, cv2.LINE_AA)

        inlier_points = map_points[state["inlier_ids"]]
        inlier_pixels = world_to_pixels(
            inlier_points[:, :2], bounds, size_px
        )
        for pixel in inlier_pixels:
            cv2.circle(image, tuple(pixel), 2, (255, 255, 255), -1)

        polygon = state["view_polygon"]
        if len(polygon):
            polygon_pixels = world_to_pixels(
                polygon[:, :2], bounds, size_px
            )
            for index in range(len(polygon_pixels)):
                dashed_line(
                    image,
                    polygon_pixels[index],
                    polygon_pixels[(index + 1) % len(polygon_pixels)],
                    (255, 180, 0),
                )

        trajectory = world_to_pixels(
            camera_positions[: state_index + 1, :2], bounds, size_px
        )
        if len(trajectory) > 1:
            cv2.polylines(
                image,
                [trajectory],
                False,
                (255, 120, 0),
                1,
                cv2.LINE_AA,
            )
        draw_camera(
            image,
            state["camera_position"],
            state["camera_heading"],
            bounds,
            (0, 0, 255) if not state["tracked"] else (0, 220, 255),
        )

        status = "TRACKING" if state["tracked"] else "LOST"
        status_color = (80, 230, 80) if state["tracked"] else (50, 50, 240)
        cv2.putText(
            image,
            f"Frame {state['frame']} | {status}",
            (PLOT_MARGIN_PX, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
        )
        cv2.putText(
            image,
            f"Visible: {len(state['visible_ids'])} | PnP inliers: {len(state['inlier_ids'])}",
            (PLOT_MARGIN_PX, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            TEXT_COLOR,
            1,
        )
        cv2.putText(
            image,
            f"Camera XYZ: {state['camera_position'][0]:.1f}, "
            f"{state['camera_position'][1]:.1f}, "
            f"{state['camera_position'][2]:.1f} mm",
            (PLOT_MARGIN_PX, size_px - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            TEXT_COLOR,
            1,
        )
        writer.write(image)

    writer.release()
