import csv

import numpy as np
from scipy.spatial.transform import Rotation

from tracking_visualization import save_comparison_figure


def load_ground_truth(path):
    timestamps = []
    positions = []
    euler_angles = []

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            timestamps.append(float(row["sync_timestamp"]))
            positions.append([float(row[axis]) for axis in ("x", "y", "z")])
            euler_angles.append(
                [float(row[axis]) for axis in ("roll", "pitch", "yaw")]
            )

    return (
        np.asarray(timestamps),
        np.asarray(positions),
        np.asarray(euler_angles),
    )


def interpolate_columns(sample_times, source_times, values):
    return np.column_stack(
        [
            np.interp(sample_times, source_times, values[:, axis])
            for axis in range(values.shape[1])
        ]
    )


def relative_euler(rotation_matrices, reference_rotation):
    relative_rotations = reference_rotation.T @ rotation_matrices
    return Rotation.from_matrix(relative_rotations).as_euler(
        "xyz",
        degrees=True,
    )


def save_mapping_csv(
    frames,
    times_s,
    timestamps,
    estimate_positions,
    ground_truth_positions,
    estimate_euler,
    ground_truth_euler,
    output_path,
):
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
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "gt_roll_deg",
        "gt_pitch_deg",
        "gt_yaw_deg",
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
                    "roll_deg": estimate_euler[index, 0],
                    "pitch_deg": estimate_euler[index, 1],
                    "yaw_deg": estimate_euler[index, 2],
                    "gt_roll_deg": ground_truth_euler[index, 0],
                    "gt_pitch_deg": ground_truth_euler[index, 1],
                    "gt_yaw_deg": ground_truth_euler[index, 2],
                }
            )


def evaluate_final_mapping_poses(
    global_map,
    video_start_timestamp,
    ground_truth_path,
    output_dir,
    recording_name,
    camera_to_output_axes,
    camera_euler_signs,
):
    frames = global_map["mapping_frames"]
    times_s = global_map["mapping_times_s"]
    timestamps = video_start_timestamp + times_s
    camera_positions = global_map["mapping_camera_positions"]
    camera_rotations = global_map["mapping_camera_rotations"]

    reference_frame = global_map["mapping_reference_frame"]
    reference_index = int(np.flatnonzero(frames == reference_frame)[0])
    reference_position = camera_positions[reference_index]
    reference_rotation = camera_rotations[reference_index]

    estimate_positions = (
        camera_to_output_axes
        @ reference_rotation.T
        @ (camera_positions - reference_position).T
    ).T
    estimate_euler = relative_euler(
        camera_rotations,
        reference_rotation,
    )
    estimate_euler *= camera_euler_signs

    gt_times, gt_positions, gt_euler = load_ground_truth(ground_truth_path)
    gt_euler = np.degrees(np.unwrap(np.radians(gt_euler), axis=0))

    inside_gt = timestamps >= gt_times[0]
    inside_gt &= timestamps <= gt_times[-1]
    frames = frames[inside_gt]
    times_s = times_s[inside_gt]
    timestamps = timestamps[inside_gt]
    estimate_positions = estimate_positions[inside_gt]
    estimate_euler = estimate_euler[inside_gt]

    interpolated_gt_positions = interpolate_columns(
        timestamps,
        gt_times,
        gt_positions,
    )
    interpolated_gt_euler = interpolate_columns(
        timestamps,
        gt_times,
        gt_euler,
    )
    reference_timestamp = video_start_timestamp + global_map[
        "mapping_times_s"
    ][reference_index]
    reference_gt_position = interpolate_columns(
        np.array([reference_timestamp]),
        gt_times,
        gt_positions,
    )[0]
    reference_gt_euler = interpolate_columns(
        np.array([reference_timestamp]),
        gt_times,
        gt_euler,
    )[0]
    ground_truth_positions = (
        interpolated_gt_positions - reference_gt_position
    )
    ground_truth_rotations = Rotation.from_euler(
        "xyz",
        interpolated_gt_euler,
        degrees=True,
    ).as_matrix()
    reference_gt_rotation = Rotation.from_euler(
        "xyz",
        reference_gt_euler,
        degrees=True,
    ).as_matrix()
    ground_truth_euler = relative_euler(
        ground_truth_rotations,
        reference_gt_rotation,
    )

    position_component_errors = estimate_positions - ground_truth_positions
    position_errors = np.linalg.norm(position_component_errors, axis=1)

    orientation_component_errors = (
        estimate_euler - ground_truth_euler + 180.0
    ) % 360.0 - 180.0
    estimate_rotations = Rotation.from_euler(
        "xyz",
        estimate_euler,
        degrees=True,
    ).as_matrix()
    gt_rotations = Rotation.from_euler(
        "xyz",
        ground_truth_euler,
        degrees=True,
    ).as_matrix()
    orientation_errors = np.degrees(
        Rotation.from_matrix(
            np.transpose(gt_rotations, (0, 2, 1)) @ estimate_rotations
        ).magnitude()
    )

    registered_percent = (
        100.0
        * len(global_map["mapping_frames"])
        / global_map["mapping_extracted_image_count"]
    )
    plot_times = times_s - times_s[0]
    title_suffix = (
        "final camera poses after global bundle adjustment | "
        f"registered: {registered_percent:.1f}%"
    )
    position_rmse = save_comparison_figure(
        plot_times,
        ground_truth_positions,
        estimate_positions,
        position_component_errors,
        position_errors,
        ["X", "Y", "Z"],
        "mm",
        "Euclidean distance on registered mapping frames",
        output_dir / "mapping_position_vs_gt.png",
        f"{recording_name}: mapping position vs GT\n{title_suffix}",
        estimate_label="Final BA pose",
        x_label="Mapping time [s]",
    )
    orientation_rmse = save_comparison_figure(
        plot_times,
        ground_truth_euler,
        estimate_euler,
        orientation_component_errors,
        orientation_errors,
        ["Roll", "Pitch", "Yaw"],
        "deg",
        "Angular distance on registered mapping frames",
        output_dir / "mapping_orientation_vs_gt.png",
        f"{recording_name}: mapping orientation vs GT\n{title_suffix}",
        estimate_label="Final BA pose",
        x_label="Mapping time [s]",
    )
    save_mapping_csv(
        frames,
        times_s,
        timestamps,
        estimate_positions,
        ground_truth_positions,
        estimate_euler,
        ground_truth_euler,
        output_dir / "mapping_camera_vs_gt.csv",
    )
    return position_rmse, orientation_rmse, registered_percent
