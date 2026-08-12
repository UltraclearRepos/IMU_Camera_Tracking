import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from scipy.spatial.transform import Rotation


ARUCO_ID = 7
ARUCO_SIZE_MM = 20.0
MIN_ARUCO_ALIGNMENT_FRAMES = 3

MIN_PAIR_MATCHES = 20
MIN_PAIR_INLIERS = 20

MIN_POINT_TRACK_LENGTH = 3
MAX_POINT_REPROJECTION_ERROR_PX = 3.0
RETRIEVAL_ROBUST_Z_THRESHOLD = 2.0
RETRIEVAL_RELATIVE_SCORE_THRESHOLD = 0.55
RETRIEVAL_SEQUENCE_AMBIGUITY_RATIO = 1.10


def select_spatially_distributed_features(
    features,
    maximum_features,
    grid_rows,
    grid_columns,
):
    keypoints = features["keypoints"]
    scores = features["scores"]
    width, height = features["image_size"]
    roi_top = features["roi_top"]

    columns = np.minimum(
        (keypoints[:, 0] * grid_columns / width).astype(int),
        grid_columns - 1,
    )
    rows = np.minimum(
        (
            (keypoints[:, 1] - roi_top)
            * grid_rows
            / (height - roi_top)
        ).astype(int),
        grid_rows - 1,
    )

    indices_by_cell = []
    for row in range(grid_rows):
        for column in range(grid_columns):
            indices = np.flatnonzero(
                (rows == row) & (columns == column)
            )
            indices = indices[np.argsort(scores[indices])[::-1]]
            indices_by_cell.append(indices)

    selected_indices = []
    rank = 0
    while len(selected_indices) < maximum_features:
        added_in_round = False
        for indices in indices_by_cell:
            if rank < len(indices):
                selected_indices.append(indices[rank])
                added_in_round = True
                if len(selected_indices) == maximum_features:
                    break
        if not added_in_round:
            break
        rank += 1

    selected_indices = np.asarray(selected_indices, dtype=int)
    return {
        "keypoints": features["keypoints"][selected_indices],
        "descriptors": features["descriptors"][selected_indices],
        "scores": features["scores"][selected_indices],
        "image_size": features["image_size"],
        "roi_top": roi_top,
    }


def select_retrieval_features(features, maximum_descriptors):
    # Mapping features are already ordered round-robin across image-grid
    # cells, so retaining that order gives retrieval broad spatial support.
    indices = np.arange(
        min(maximum_descriptors, len(features["descriptors"]))
    )
    descriptors = features["descriptors"][indices]
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return {
        "descriptors": descriptors
        / np.maximum(norms, np.finfo(np.float32).eps),
        "keypoints": features["keypoints"][indices],
        "image_size": features["image_size"],
        "roi_top": features["roi_top"],
    }


def score_similar_frames(
    current_features,
    eligible_names,
    retrieval_features_by_name,
    device,
    grid_rows,
    grid_columns,
    minimum_covered_cells,
):
    current_descriptors = current_features["descriptors"]
    if not eligible_names or not len(current_descriptors):
        return []

    valid_names = [
        name
        for name in eligible_names
        if len(retrieval_features_by_name[name]["descriptors"])
    ]
    if not valid_names:
        return []

    descriptor_counts = np.asarray(
        [
            len(retrieval_features_by_name[name]["descriptors"])
            for name in valid_names
        ],
        dtype=np.int64,
    )
    maximum_previous_descriptors = int(np.max(descriptor_counts))
    descriptor_dimension = current_descriptors.shape[1]
    previous_descriptors = np.zeros(
        (
            len(valid_names),
            maximum_previous_descriptors,
            descriptor_dimension,
        ),
        dtype=np.float32,
    )
    for frame_index, previous_name in enumerate(valid_names):
        descriptors = retrieval_features_by_name[previous_name][
            "descriptors"
        ]
        previous_descriptors[frame_index, : len(descriptors)] = (
            descriptors
        )

    query = torch.as_tensor(
        current_descriptors,
        dtype=torch.float32,
        device=device,
    )
    previous = torch.as_tensor(
        previous_descriptors,
        dtype=torch.float32,
        device=device,
    )
    counts = torch.as_tensor(descriptor_counts, device=device)
    similarities = torch.einsum("qd,fnd->fqn", query, previous)
    valid_previous = (
        torch.arange(maximum_previous_descriptors, device=device)[None]
        < counts[:, None]
    )
    similarities = similarities.masked_fill(
        ~valid_previous[:, None, :],
        -torch.inf,
    )

    query_values, query_best_previous = similarities.max(dim=2)
    previous_best_query = similarities.argmax(dim=1)
    best_previous_query = torch.gather(
        previous_best_query,
        1,
        query_best_previous,
    )
    query_indices = torch.arange(
        similarities.shape[1],
        device=device,
    )[None]
    mutual = best_previous_query == query_indices

    if maximum_previous_descriptors > 1:
        query_second = torch.topk(
            similarities,
            k=2,
            dim=2,
        ).values[:, :, 1]
        query_second = torch.where(
            counts[:, None] > 1,
            query_second,
            torch.zeros_like(query_second),
        )
    else:
        query_second = torch.zeros_like(query_values)
    if similarities.shape[1] > 1:
        previous_second = torch.topk(
            similarities,
            k=2,
            dim=1,
        ).values[:, 1, :]
    else:
        previous_second = torch.zeros(
            (
                similarities.shape[0],
                similarities.shape[2],
            ),
            dtype=similarities.dtype,
            device=device,
        )
    matched_previous_second = torch.gather(
        previous_second,
        1,
        query_best_previous,
    )
    query_margins = torch.clamp(
        query_values - query_second,
        min=0.0,
    )
    previous_margins = torch.clamp(
        query_values - matched_previous_second,
        min=0.0,
    )
    distinctiveness = torch.sqrt(query_margins * previous_margins)
    weights = (
        torch.clamp(query_values, min=0.0)
        * distinctiveness
        * mutual
    )

    mutual_np = mutual.detach().cpu().numpy()
    best_previous_np = query_best_previous.detach().cpu().numpy()
    similarities_np = query_values.detach().cpu().numpy()
    retrieval_scores = weights.sum(dim=1).detach().cpu().numpy()

    candidates = []
    for frame_index, previous_name in enumerate(valid_names):
        previous_features = retrieval_features_by_name[previous_name]
        query_indices_np = np.flatnonzero(mutual_np[frame_index])
        if not len(query_indices_np):
            continue
        previous_indices_np = best_previous_np[
            frame_index,
            query_indices_np,
        ]
        mutual_similarities = similarities_np[
            frame_index,
            query_indices_np,
        ]
        covered_cells_current = occupied_image_grid_cells(
            current_features["keypoints"][query_indices_np],
            current_features,
            grid_rows,
            grid_columns,
        )
        covered_cells_previous = occupied_image_grid_cells(
            previous_features["keypoints"][previous_indices_np],
            previous_features,
            grid_rows,
            grid_columns,
        )
        if min(
            covered_cells_current,
            covered_cells_previous,
        ) < minimum_covered_cells:
            continue

        candidates.append(
            {
                "image_name": previous_name,
                "previous_frame": frame_number(previous_name),
                "votes": int(len(query_indices_np)),
                "mean_similarity": float(np.mean(mutual_similarities)),
                "retrieval_score": float(
                    retrieval_scores[frame_index]
                ),
                "covered_cells_current": covered_cells_current,
                "covered_cells_previous": covered_cells_previous,
            }
        )
    candidates.sort(
        key=lambda candidate: candidate["retrieval_score"],
        reverse=True,
    )
    return candidates


def select_old_frame_sequence(
    candidates,
    maximum_frames,
    minimum_sequence_frames,
    maximum_sequence_gap,
):
    if not candidates:
        return [], [], None

    scores = np.asarray(
        [candidate["retrieval_score"] for candidate in candidates],
        dtype=np.float64,
    )
    median_score = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median_score)))
    robust_sigma = 1.4826 * mad
    robust_threshold = median_score + (
        RETRIEVAL_ROBUST_Z_THRESHOLD * robust_sigma
    )
    relative_threshold = (
        RETRIEVAL_RELATIVE_SCORE_THRESHOLD * float(np.max(scores))
    )
    score_threshold = max(robust_threshold, relative_threshold)
    strong_candidates = [
        candidate
        for candidate in candidates
        if candidate["retrieval_score"] >= score_threshold
    ]
    strong_candidates.sort(key=lambda item: item["previous_frame"])

    sequences = []
    for candidate in strong_candidates:
        if (
            not sequences
            or candidate["previous_frame"]
            - sequences[-1][-1]["previous_frame"]
            > maximum_sequence_gap
        ):
            sequences.append([candidate])
        else:
            sequences[-1].append(candidate)

    sequences = [
        sequence
        for sequence in sequences
        if len(sequence) >= minimum_sequence_frames
    ]
    if not sequences:
        return [], candidates[:maximum_frames], {
            "score_threshold": score_threshold,
            "support_frames": len(strong_candidates),
            "sequence_frames": 0,
            "ambiguous": False,
        }

    def sequence_score(sequence):
        top_scores = sorted(
            (item["retrieval_score"] for item in sequence),
            reverse=True,
        )[:maximum_frames]
        return float(np.mean(top_scores))

    ranked_sequences = sorted(
        sequences,
        key=sequence_score,
        reverse=True,
    )
    best_sequence = ranked_sequences[0]
    best_sequence_score = sequence_score(best_sequence)
    second_sequence_score = (
        sequence_score(ranked_sequences[1])
        if len(ranked_sequences) > 1
        else 0.0
    )
    ambiguous = (
        second_sequence_score > 0.0
        and best_sequence_score
        < RETRIEVAL_SEQUENCE_AMBIGUITY_RATIO * second_sequence_score
    )
    sequence_info = {
        "score_threshold": score_threshold,
        "support_frames": len(strong_candidates),
        "sequence_frames": len(best_sequence),
        "sequence_start": best_sequence[0]["previous_frame"],
        "sequence_end": best_sequence[-1]["previous_frame"],
        "sequence_score": best_sequence_score,
        "second_sequence_score": second_sequence_score,
        "ambiguous": ambiguous,
    }
    if ambiguous:
        return [], candidates[:maximum_frames], sequence_info

    selected = []
    for candidate in sorted(
        best_sequence,
        key=lambda item: item["retrieval_score"],
        reverse=True,
    ):
        if all(
            abs(
                candidate["previous_frame"]
                - selected_candidate["previous_frame"]
            ) >= 2
            for selected_candidate in selected
        ):
            selected.append(candidate)
        if len(selected) == maximum_frames:
            break
    if len(selected) < maximum_frames:
        for candidate in sorted(
            best_sequence,
            key=lambda item: item["retrieval_score"],
            reverse=True,
        ):
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == maximum_frames:
                break

    for candidate in selected:
        candidate["sequence_frames"] = len(best_sequence)
        candidate["sequence_start"] = best_sequence[0]["previous_frame"]
        candidate["sequence_end"] = best_sequence[-1]["previous_frame"]
    return selected, selected, sequence_info


def occupied_image_grid_cells(keypoints, features, rows, columns):
    if not len(keypoints):
        return 0

    width, height = features["image_size"]
    roi_top = features["roi_top"]
    column_indices = np.floor(
        keypoints[:, 0] * columns / width
    ).astype(int)
    row_indices = np.floor(
        (keypoints[:, 1] - roi_top)
        * rows
        / (height - roi_top)
    ).astype(int)
    column_indices = np.clip(column_indices, 0, columns - 1)
    row_indices = np.clip(row_indices, 0, rows - 1)
    return len(
        np.unique(row_indices * columns + column_indices)
    )


def select_uniform_landmarks(
    positions,
    track_lengths,
    reprojection_errors,
    maximum_landmarks,
    grid_rows,
    grid_columns,
    reprojection_error_weight,
):
    minimum_xy = np.min(positions[:, :2], axis=0)
    extent_xy = np.maximum(
        np.ptp(positions[:, :2], axis=0),
        np.finfo(float).eps,
    )
    normalized_xy = (positions[:, :2] - minimum_xy) / extent_xy
    columns = np.minimum(
        (normalized_xy[:, 0] * grid_columns).astype(int),
        grid_columns - 1,
    )
    rows = np.minimum(
        (normalized_xy[:, 1] * grid_rows).astype(int),
        grid_rows - 1,
    )

    track_quality = (
        track_lengths - np.min(track_lengths)
    ) / max(np.ptp(track_lengths), 1)
    reprojection_quality = 1.0 - (
        reprojection_errors - np.min(reprojection_errors)
    ) / max(np.ptp(reprojection_errors), np.finfo(float).eps)
    landmark_quality = (
        reprojection_error_weight * reprojection_quality
        + (1.0 - reprojection_error_weight) * track_quality
    )

    indices_by_cell = []
    for row in range(grid_rows):
        for column in range(grid_columns):
            indices = np.flatnonzero(
                (rows == row) & (columns == column)
            )
            if len(indices) == 0:
                continue

            quality_order = np.argsort(landmark_quality[indices])[::-1]
            indices_by_cell.append(indices[quality_order])

    selected_indices = []
    rank = 0
    while len(selected_indices) < maximum_landmarks:
        added_in_round = False
        for indices in indices_by_cell:
            if rank < len(indices):
                selected_indices.append(indices[rank])
                added_in_round = True
                if len(selected_indices) == maximum_landmarks:
                    break
        if not added_in_round:
            break
        rank += 1

    occupied_cells = len(indices_by_cell)
    landmarks_per_cell = int(
        np.ceil(min(len(positions), maximum_landmarks) / occupied_cells)
    )
    return (
        np.asarray(selected_indices, dtype=int),
        occupied_cells,
        landmarks_per_cell,
    )


def aruco_object_points():
    half = ARUCO_SIZE_MM / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def create_colmap_camera(camera_matrix, distortion, width, height):
    distortion = distortion.reshape(-1)
    distortion = np.pad(distortion, (0, max(0, 8 - len(distortion))))
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3, k4, k5, k6 = distortion[:8]
    parameters = np.array(
        [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6],
        dtype=float,
    )
    camera = pycolmap.Camera(
        model="FULL_OPENCV",
        width=width,
        height=height,
        params=parameters,
    )
    camera.has_prior_focal_length = True
    return camera


def camera_center(R_world_to_camera, t_world_to_camera):
    return -R_world_to_camera.T @ t_world_to_camera


def frame_number(image_name):
    return int(Path(image_name).stem.rsplit("_", 1)[1])


def summarize_track_lengths(track_lengths):
    track_lengths = np.asarray(track_lengths, dtype=int)
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


def print_track_length_statistics(statistics):
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


def print_map_build_timing(timing):
    online = timing["mapping_frame_processing"]
    after = timing["after_collection"]
    online_stages = [
        ("setup_s", "Setup"),
        ("frame_read_s", "Frame reading"),
        ("image_save_s", "Image saving"),
        ("disk_and_feature_selection_s", "DISK + feature selection"),
        ("retrieval_voting_s", "Old-frame descriptor voting"),
        ("aruco_detection_s", "ArUco detection"),
        (
            "image_and_keypoints_database_write_s",
            "Image/keypoint database writes",
        ),
        ("lightglue_matching_s", "LightGlue matching"),
        ("pair_geometry_verification_s", "Pair geometry verification"),
        (
            "verified_pair_database_write_s",
            "Verified-pair database writes",
        ),
    ]
    after_stages = [
        (
            "colmap_reconstruction_and_bundle_adjustment_s",
            "COLMAP reconstruction + bundle adjustment",
        ),
        ("aruco_metric_alignment_s", "ArUco metric alignment"),
        (
            "landmark_selection_and_descriptors_s",
            "Landmark selection + descriptors",
        ),
        ("map_saving_s", "Map saving"),
    ]

    print("Map build timing")
    print("  MAPPING FRAME PROCESSING - executed while frames arrive")
    for key, label in online_stages:
        print(f"    {label}: {online[key]:.2f} s")
    print(
        f"    TOTAL: {online['wall_time_s']:.2f} s | "
        f"{online['average_wall_time_per_frame_ms']:.1f} ms/frame"
    )
    print(
        f"    Frames: {online['processed_frames']} | "
        f"pairs: {online['verified_pairs']}/"
        f"{online['attempted_pairs']} verified"
    )
    print(
        "    Sequential pairs: "
        f"{online['sequential_verified_pairs']}/"
        f"{online['sequential_attempted_pairs']} | "
        "retrieved pairs: "
        f"{online['retrieval_verified_pairs']}/"
        f"{online['retrieval_attempted_pairs']}"
    )

    print("  AFTER COLLECTION - executed after the last mapping frame")
    for key, label in after_stages:
        print(f"    {label}: {after[key]:.2f} s")
    print(f"    TOTAL: {after['wall_time_s']:.2f} s")
    print(
        "  COMPLETE MAP BUILD: "
        f"{timing['total_map_build_wall_time_s']:.2f} s"
    )


class SkinMapBuilder:
    def __init__(
        self,
        camera_matrix,
        distortion,
        feature_matching,
        mapping_start_frame,
        mapping_end_frame,
        reconstruction_method,
        mapping_frame_step,
        sequential_match_overlap,
        enable_retrieval,
        retrieval_top_frames,
        retrieval_min_frame_gap,
        retrieval_descriptors_per_frame,
        retrieval_min_sequence_frames,
        retrieval_max_sequence_gap,
        retrieval_min_covered_cells,
        mapping_maximum_features,
        mapping_feature_grid_rows,
        mapping_feature_grid_columns,
        maximum_global_landmarks,
        global_map_grid_rows,
        global_map_grid_columns,
        global_map_reprojection_error_weight,
    ):
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.feature_matching = feature_matching
        self.mapping_start_frame = mapping_start_frame
        self.mapping_end_frame = mapping_end_frame
        self.reconstruction_method = reconstruction_method
        self.mapping_frame_step = mapping_frame_step
        self.sequential_match_overlap = sequential_match_overlap
        self.enable_retrieval = enable_retrieval
        self.retrieval_top_frames = retrieval_top_frames
        self.retrieval_min_frame_gap = retrieval_min_frame_gap
        self.retrieval_descriptors_per_frame = (
            retrieval_descriptors_per_frame
        )
        self.retrieval_min_sequence_frames = (
            retrieval_min_sequence_frames
        )
        self.retrieval_max_sequence_gap = (
            retrieval_max_sequence_gap
        )
        self.retrieval_min_covered_cells = retrieval_min_covered_cells
        self.mapping_maximum_features = mapping_maximum_features
        self.mapping_feature_grid_rows = mapping_feature_grid_rows
        self.mapping_feature_grid_columns = mapping_feature_grid_columns
        self.maximum_global_landmarks = maximum_global_landmarks
        self.global_map_grid_rows = global_map_grid_rows
        self.global_map_grid_columns = global_map_grid_columns
        self.global_map_reprojection_error_weight = (
            global_map_reprojection_error_weight
        )

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary)

    def detect_aruco_pose(self, frame):
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        if ids is None or ARUCO_ID not in ids.flatten():
            return None

        marker_index = np.where(ids.flatten() == ARUCO_ID)[0][0]
        image_points = corners[marker_index].reshape(4, 2).astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(
            aruco_object_points(),
            image_points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None

        return cv2.Rodrigues(rvec)[0], tvec.reshape(3)

    def collect_and_match_mapping_frames(
        self,
        video_path,
        images_dir,
        database_path,
    ):
        frame_processing_started = time.perf_counter()
        setup_started = time.perf_counter()
        capture = cv2.VideoCapture(str(video_path))
        database = pycolmap.Database.open(database_path)
        camera = create_colmap_camera(
            self.camera_matrix,
            self.distortion,
            round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        camera_id = database.write_camera(camera)
        camera.camera_id = camera_id
        setup_seconds = time.perf_counter() - setup_started

        image_names = []
        image_ids = {}
        features_by_name = {}
        retrieval_features_by_name = {}
        aruco_poses = {}
        frame_times_by_name = {}
        retrieval_diagnostics = []

        geometry_options = pycolmap.TwoViewGeometryOptions()
        geometry_options.ransac.random_seed = 0
        pair_count = 0
        attempted_pair_count = 0
        sequential_attempted_pair_count = 0
        sequential_verified_pair_count = 0
        retrieval_attempted_pair_count = 0
        retrieval_verified_pair_count = 0
        frame_read_seconds = 0.0
        image_save_seconds = 0.0
        disk_seconds = 0.0
        aruco_seconds = 0.0
        image_database_write_seconds = 0.0
        lightglue_seconds = 0.0
        geometry_seconds = 0.0
        pair_database_write_seconds = 0.0
        retrieval_voting_seconds = 0.0

        for _ in range(self.mapping_start_frame):
            success, _ = capture.read()
            if not success:
                break

        for frame_index in range(
            self.mapping_start_frame,
            self.mapping_end_frame + 1,
        ):
            reading_started = time.perf_counter()
            success, frame = capture.read()
            frame_read_seconds += time.perf_counter() - reading_started
            if not success:
                break
            if frame_index % self.mapping_frame_step != 0:
                continue

            image_name = f"frame_{frame_index:06d}.png"
            saving_started = time.perf_counter()
            cv2.imwrite(str(images_dir / image_name), frame)
            image_save_seconds += time.perf_counter() - saving_started

            disk_started = time.perf_counter()
            detected_features = self.feature_matching.extract(frame)
            features = select_spatially_distributed_features(
                detected_features,
                self.mapping_maximum_features,
                self.mapping_feature_grid_rows,
                self.mapping_feature_grid_columns,
            )
            disk_seconds += time.perf_counter() - disk_started
            features_by_name[image_name] = features
            current_retrieval_features = None
            if self.enable_retrieval:
                current_retrieval_features = select_retrieval_features(
                    features,
                    self.retrieval_descriptors_per_frame,
                )
                retrieval_features_by_name[image_name] = (
                    current_retrieval_features
                )
            frame_times_by_name[image_name] = (
                capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            )

            aruco_started = time.perf_counter()
            aruco_pose = self.detect_aruco_pose(frame)
            aruco_seconds += time.perf_counter() - aruco_started
            if aruco_pose is not None:
                aruco_poses[image_name] = aruco_pose

            database_write_started = time.perf_counter()
            image = pycolmap.Image(
                name=image_name,
                camera_id=camera.camera_id,
            )
            image_id = database.write_image(image)
            image_ids[image_name] = image_id
            database.write_keypoints(image_id, features["keypoints"])
            image_database_write_seconds += (
                time.perf_counter() - database_write_started
            )

            sequential_names = image_names[
                -self.sequential_match_overlap :
            ]
            scored_retrieval_candidates = []
            retrieval_candidates = []
            diagnostic_candidates = []
            retrieval_sequence_info = None
            if self.enable_retrieval:
                eligible_retrieval_names = [
                    previous_name
                    for previous_name in image_names
                    if frame_index - frame_number(previous_name)
                    >= self.retrieval_min_frame_gap
                ]
                voting_started = time.perf_counter()
                scored_retrieval_candidates = score_similar_frames(
                    current_retrieval_features,
                    eligible_retrieval_names,
                    retrieval_features_by_name,
                    self.feature_matching.device,
                    self.mapping_feature_grid_rows,
                    self.mapping_feature_grid_columns,
                    self.retrieval_min_covered_cells,
                )
                (
                    retrieval_candidates,
                    diagnostic_candidates,
                    retrieval_sequence_info,
                ) = select_old_frame_sequence(
                    scored_retrieval_candidates,
                    self.retrieval_top_frames,
                    self.retrieval_min_sequence_frames,
                    self.retrieval_max_sequence_gap,
                )
                retrieval_voting_seconds += (
                    time.perf_counter() - voting_started
                )

            for candidate in scored_retrieval_candidates:
                candidate["current_frame"] = frame_index

            selected_retrieval_names = {
                candidate["image_name"]
                for candidate in retrieval_candidates
            }
            frame_retrieval_diagnostics = []
            for candidate in diagnostic_candidates:
                selected_for_geometry = (
                    candidate["image_name"] in selected_retrieval_names
                )
                diagnostic = {
                    "current_frame": frame_index,
                    "previous_frame": candidate["previous_frame"],
                    "votes": candidate["votes"],
                    "mean_similarity": candidate["mean_similarity"],
                    "retrieval_score": candidate["retrieval_score"],
                    "raw_matches": 0,
                    "inliers": 0,
                    "geometry": "not_estimated",
                    "covered_cells_previous": candidate[
                        "covered_cells_previous"
                    ],
                    "covered_cells_current": candidate[
                        "covered_cells_current"
                    ],
                    "required_covered_cells": (
                        self.retrieval_min_covered_cells
                    ),
                    "sequence_frames": (
                        retrieval_sequence_info["sequence_frames"]
                        if retrieval_sequence_info is not None
                        else 0
                    ),
                    "required_sequence_frames": (
                        self.retrieval_min_sequence_frames
                    ),
                    "maximum_sequence_gap": (
                        self.retrieval_max_sequence_gap
                    ),
                    "sequence_start": (
                        retrieval_sequence_info.get("sequence_start")
                        if retrieval_sequence_info is not None
                        else None
                    ),
                    "sequence_end": (
                        retrieval_sequence_info.get("sequence_end")
                        if retrieval_sequence_info is not None
                        else None
                    ),
                    "score_threshold": (
                        retrieval_sequence_info["score_threshold"]
                        if retrieval_sequence_info is not None
                        else None
                    ),
                    "rejection_reason": (
                        "too_few_matches"
                        if selected_for_geometry
                        else (
                            "ambiguous_old_sequence"
                            if retrieval_sequence_info is not None
                            and retrieval_sequence_info["ambiguous"]
                            else "old_sequence_not_found"
                        )
                    ),
                    "accepted": False,
                }
                candidate["diagnostic"] = diagnostic
                retrieval_diagnostics.append(diagnostic)
                frame_retrieval_diagnostics.append(diagnostic)

            retrieval_by_name = {
                candidate["image_name"]: candidate
                for candidate in retrieval_candidates
            }
            frames_to_match = sequential_names + list(retrieval_by_name)

            for previous_name in frames_to_match:
                retrieval_candidate = retrieval_by_name.get(previous_name)
                is_retrieval = retrieval_candidate is not None
                attempted_pair_count += 1
                if is_retrieval:
                    retrieval_attempted_pair_count += 1
                else:
                    sequential_attempted_pair_count += 1

                matching_started = time.perf_counter()
                matches = self.feature_matching.match(
                    features_by_name[previous_name],
                    features,
                )
                lightglue_seconds += time.perf_counter() - matching_started
                raw_match_count = len(matches)
                diagnostic = None
                if is_retrieval:
                    diagnostic = retrieval_candidate["diagnostic"]
                    diagnostic["raw_matches"] = raw_match_count
                if len(matches) < MIN_PAIR_MATCHES:
                    continue

                geometry_started = time.perf_counter()
                geometry = pycolmap.estimate_two_view_geometry(
                    camera,
                    features_by_name[previous_name]["keypoints"],
                    camera,
                    features["keypoints"],
                    matches,
                    geometry_options,
                )
                geometry_seconds += time.perf_counter() - geometry_started
                inlier_count = len(geometry.inlier_matches)
                if diagnostic is not None:
                    diagnostic["inliers"] = inlier_count
                    diagnostic["geometry"] = (
                        pycolmap.TwoViewGeometryConfiguration(
                            geometry.config
                        ).name
                    )
                    diagnostic["rejection_reason"] = "too_few_inliers"
                if len(geometry.inlier_matches) < MIN_PAIR_INLIERS:
                    continue

                previous_id = image_ids[previous_name]
                writing_started = time.perf_counter()
                database.write_two_view_geometry(
                    previous_id,
                    image_id,
                    geometry,
                )
                pair_database_write_seconds += (
                    time.perf_counter() - writing_started
                )
                pair_count += 1
                if is_retrieval:
                    diagnostic["accepted"] = True
                    diagnostic["rejection_reason"] = None
                    retrieval_verified_pair_count += 1
                else:
                    sequential_verified_pair_count += 1

            image_names.append(image_name)
            retrieval_log = ", ".join(
                f"{item['previous_frame']}: "
                f"mnn{item['votes']} r{item['retrieval_score']:.2f} "
                f"s{item['mean_similarity']:.3f} "
                f"m{item['raw_matches']} "
                f"i{item['inliers']} "
                f"g{item['geometry']} "
                f"cells{min(item['covered_cells_previous'], item['covered_cells_current'])} "
                f"{'OK' if item['accepted'] else item['rejection_reason']}"
                for item in frame_retrieval_diagnostics
            )
            print(
                f"Mapping frame processing: frame {frame_index}/"
                f"{self.mapping_end_frame} | "
                f"features: {len(features['keypoints'])} | "
                f"verified pairs: {pair_count} | "
                f"retrieval scored: {len(scored_retrieval_candidates)} | "
                f"old-sequence selected: {len(retrieval_candidates)} | "
                f"retrieval: {retrieval_log or 'not available'}"
            )

        capture.release()
        database.close()
        statistics = {
            "attempted_pairs": attempted_pair_count,
            "verified_pairs": pair_count,
            "sequential_attempted_pairs": (
                sequential_attempted_pair_count
            ),
            "sequential_verified_pairs": sequential_verified_pair_count,
            "retrieval_attempted_pairs": retrieval_attempted_pair_count,
            "retrieval_verified_pairs": retrieval_verified_pair_count,
            "retrieval_diagnostics": retrieval_diagnostics,
            "setup_seconds": setup_seconds,
            "frame_read_seconds": frame_read_seconds,
            "image_save_seconds": image_save_seconds,
            "disk_and_selection_seconds": disk_seconds,
            "aruco_detection_seconds": aruco_seconds,
            "image_database_write_seconds": (
                image_database_write_seconds
            ),
            "lightglue_seconds": lightglue_seconds,
            "geometry_verification_seconds": geometry_seconds,
            "pair_database_write_seconds": pair_database_write_seconds,
            "retrieval_voting_seconds": retrieval_voting_seconds,
            "mapping_frame_processing_wall_seconds": (
                time.perf_counter() - frame_processing_started
            ),
        }
        return (
            image_names,
            features_by_name,
            aruco_poses,
            frame_times_by_name,
            statistics,
        )

    def reconstruct(self, database_path, images_dir, sparse_dir):
        if self.reconstruction_method == "global":
            options = pycolmap.GlobalPipelineOptions(
                min_num_matches=MIN_PAIR_INLIERS,
                random_seed=0,
            )
            options.mapper.random_seed = 0
            options.mapper.rotation_averaging.random_seed = 0
            options.mapper.global_positioning.random_seed = 0
            options.mapper.bundle_adjustment.refine_focal_length = False
            options.mapper.bundle_adjustment.refine_principal_point = False
            options.mapper.bundle_adjustment.refine_extra_params = False
            options.mapper.global_positioning.use_gpu = True
            options.mapper.global_positioning.min_num_images_gpu_solver = 3
            options.mapper.bundle_adjustment.ceres.use_gpu = True
            mapping = pycolmap.global_mapping
        else:
            options = pycolmap.IncrementalPipelineOptions(
                multiple_models=False,
                structure_less_registration_fallback=False,
                ba_refine_focal_length=False,
                ba_refine_principal_point=False,
                ba_refine_extra_params=False,
            )
            options.ba_use_gpu = True
            options.mapper.init_min_num_inliers = MIN_PAIR_INLIERS
            options.mapper.abs_pose_min_num_inliers = MIN_PAIR_INLIERS
            options.random_seed = 0
            options.mapper.random_seed = 0
            options.triangulation.random_seed = 0
            mapping = pycolmap.incremental_mapping

        reconstructions = mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=sparse_dir,
            options=options,
        )
        if not reconstructions:
            raise RuntimeError(
                "COLMAP could not initialize the 3D map. The mapping pass "
                "needs camera translation and visible feature parallax."
            )
        return max(
            reconstructions.values(),
            key=lambda reconstruction: reconstruction.num_reg_images(),
        )

    def align_to_first_camera(self, reconstruction, aruco_poses):
        registered_aruco_images = sorted(
            [
                image
                for image in reconstruction.images.values()
                if image.name in aruco_poses
            ],
            key=lambda image: image.name,
        )
        if len(registered_aruco_images) < MIN_ARUCO_ALIGNMENT_FRAMES:
            raise RuntimeError(
                "ArUco must be visible in at least "
                f"{MIN_ARUCO_ALIGNMENT_FRAMES} registered mapping frames"
            )

        reference_image = registered_aruco_images[0]
        R_aruco_to_reference, t_aruco_to_reference = aruco_poses[
            reference_image.name
        ]

        rotation_candidates = []
        sfm_centers = []
        reference_centers = []

        for image in registered_aruco_images:
            R_aruco_to_camera, t_aruco_to_camera = aruco_poses[image.name]
            R_reference_to_camera = (
                R_aruco_to_camera @ R_aruco_to_reference.T
            )
            t_reference_to_camera = (
                t_aruco_to_camera
                - R_reference_to_camera @ t_aruco_to_reference
            )

            R_sfm_to_camera = image.cam_from_world().rotation.matrix()
            rotation_candidates.append(
                R_reference_to_camera.T @ R_sfm_to_camera
            )
            sfm_centers.append(image.projection_center())
            reference_centers.append(
                camera_center(
                    R_reference_to_camera,
                    t_reference_to_camera,
                )
            )

        rotation = Rotation.from_matrix(
            rotation_candidates
        ).mean().as_matrix()
        sfm_centers = np.asarray(sfm_centers)
        reference_centers = np.asarray(reference_centers)
        rotated_centers = (rotation @ sfm_centers.T).T

        rotated_mean = np.mean(rotated_centers, axis=0)
        reference_mean = np.mean(reference_centers, axis=0)
        rotated_centered = rotated_centers - rotated_mean
        reference_centered = reference_centers - reference_mean
        scale = np.sum(
            rotated_centered * reference_centered
        ) / np.sum(rotated_centered**2)
        translation = reference_mean - scale * rotated_mean

        aligned_centers = scale * rotated_centers + translation
        alignment_rmse = np.sqrt(
            np.mean(
                np.sum(
                    (aligned_centers - reference_centers) ** 2,
                    axis=1,
                )
            )
        )
        return {
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
            "alignment_rmse_mm": alignment_rmse,
            "alignment_frames": len(registered_aruco_images),
            "reference_image": reference_image.name,
        }

    def transform_points(self, points, alignment):
        return (
            alignment["scale"]
            * (alignment["rotation"] @ points.T).T
            + alignment["translation"]
        )

    def transform_pose(self, image, alignment):
        sfm_pose = image.cam_from_world()
        R_sfm_to_camera = sfm_pose.rotation.matrix()
        t_sfm_to_camera = np.asarray(sfm_pose.translation)
        R_map_to_camera = R_sfm_to_camera @ alignment["rotation"].T
        t_map_to_camera = (
            alignment["scale"] * t_sfm_to_camera
            - R_map_to_camera @ alignment["translation"]
        )
        return R_map_to_camera, t_map_to_camera

    def create_global_map(
        self,
        reconstruction,
        features_by_name,
        alignment,
        frame_times_by_name,
    ):
        positions = []
        track_lengths = []
        reprojection_errors = []
        first_observation_frames = []
        candidate_available_frames = []
        candidate_points = []

        for point in reconstruction.points3D.values():
            if point.track.length() < MIN_POINT_TRACK_LENGTH:
                continue
            if point.error > MAX_POINT_REPROJECTION_ERROR_PX:
                continue

            observation_frames = sorted(
                frame_number(
                    reconstruction.images[observation.image_id].name
                )
                for observation in point.track.elements
            )
            candidate_points.append(point)
            positions.append(point.xyz)
            track_lengths.append(point.track.length())
            reprojection_errors.append(point.error)
            first_observation_frames.append(observation_frames[0])
            candidate_available_frames.append(
                observation_frames[MIN_POINT_TRACK_LENGTH - 1]
            )

        positions = self.transform_points(
            np.asarray(positions, dtype=np.float64),
            alignment,
        )
        track_lengths = np.asarray(track_lengths, dtype=np.int32)
        reprojection_errors = np.asarray(
            reprojection_errors,
            dtype=np.float32,
        )
        first_observation_frames = np.asarray(
            first_observation_frames,
            dtype=np.int32,
        )
        candidate_available_frames = np.asarray(
            candidate_available_frames,
            dtype=np.int32,
        )
        candidate_positions = positions.copy()
        candidate_track_length_statistics = summarize_track_lengths(
            track_lengths
        )
        candidate_landmarks = len(positions)
        (
            selected_indices,
            occupied_grid_cells,
            landmarks_per_occupied_cell,
        ) = select_uniform_landmarks(
            positions,
            track_lengths,
            reprojection_errors,
            self.maximum_global_landmarks,
            self.global_map_grid_rows,
            self.global_map_grid_columns,
            self.global_map_reprojection_error_weight,
        )

        positions = positions[selected_indices]
        track_lengths = track_lengths[selected_indices]
        reprojection_errors = reprojection_errors[selected_indices]
        first_observation_frames = first_observation_frames[selected_indices]
        selected_track_length_statistics = summarize_track_lengths(
            track_lengths
        )
        selected_points = [
            candidate_points[index] for index in selected_indices
        ]

        descriptors = []
        scores = []
        for point in selected_points:
            point_descriptors = []
            point_scores = []
            for observation in point.track.elements:
                image = reconstruction.images[observation.image_id]
                point_descriptors.append(
                    features_by_name[image.name]["descriptors"][
                        observation.point2D_idx
                    ]
                )
                point_scores.append(
                    features_by_name[image.name]["scores"][
                        observation.point2D_idx
                    ]
                )

            descriptor = np.mean(point_descriptors, axis=0)
            descriptor /= np.linalg.norm(descriptor)
            descriptors.append(descriptor)
            scores.append(np.mean(point_scores))

        descriptors = np.asarray(descriptors, dtype=np.float32)

        last_image = max(
            reconstruction.images.values(),
            key=lambda image: image.name,
        )
        initial_R, initial_t = self.transform_pose(last_image, alignment)

        mapping_frames = []
        mapping_times_s = []
        mapping_camera_positions = []
        mapping_camera_rotations = []
        mapping_camera_headings = []
        for image in sorted(
            reconstruction.images.values(),
            key=lambda registered_image: registered_image.name,
        ):
            R_map_to_camera, t_map_to_camera = self.transform_pose(
                image,
                alignment,
            )
            mapping_frames.append(frame_number(image.name))
            mapping_times_s.append(frame_times_by_name[image.name])
            mapping_camera_positions.append(
                camera_center(R_map_to_camera, t_map_to_camera)
            )
            mapping_camera_rotations.append(R_map_to_camera.T)
            mapping_camera_headings.append(
                R_map_to_camera.T @ np.array([0.0, -1.0, 0.0])
            )

        return {
            "positions": positions,
            "descriptors": descriptors,
            "scores": np.asarray(scores, dtype=np.float32),
            "track_lengths": track_lengths,
            "reprojection_errors": reprojection_errors,
            "first_observation_frames": first_observation_frames,
            "candidate_positions": candidate_positions,
            "candidate_available_frames": candidate_available_frames,
            "selected_candidate_indices": np.asarray(
                selected_indices,
                dtype=np.int32,
            ),
            "candidate_landmarks": candidate_landmarks,
            "occupied_grid_cells": occupied_grid_cells,
            "landmarks_per_occupied_cell": (
                landmarks_per_occupied_cell
            ),
            "candidate_track_length_statistics": (
                candidate_track_length_statistics
            ),
            "selected_track_length_statistics": (
                selected_track_length_statistics
            ),
            "mapping_frames": np.asarray(mapping_frames, dtype=np.int32),
            "mapping_times_s": np.asarray(mapping_times_s, dtype=np.float64),
            "mapping_camera_positions": np.asarray(
                mapping_camera_positions,
                dtype=np.float64,
            ),
            "mapping_camera_rotations": np.asarray(
                mapping_camera_rotations,
                dtype=np.float64,
            ),
            "mapping_camera_headings": np.asarray(
                mapping_camera_headings,
                dtype=np.float64,
            ),
            "mapping_reference_frame": frame_number(
                alignment["reference_image"]
            ),
            "initial_R": initial_R,
            "initial_t": initial_t,
            "last_mapping_image": last_image.name,
        }

    def save_map(self, global_map, output_path):
        np.savez_compressed(
            output_path,
            positions=global_map["positions"],
            descriptors=global_map["descriptors"],
            scores=global_map["scores"],
            track_lengths=global_map["track_lengths"],
            reprojection_errors=global_map["reprojection_errors"],
            first_observation_frames=global_map[
                "first_observation_frames"
            ],
            candidate_positions=global_map["candidate_positions"],
            candidate_available_frames=global_map[
                "candidate_available_frames"
            ],
            selected_candidate_indices=global_map[
                "selected_candidate_indices"
            ],
            mapping_frames=global_map["mapping_frames"],
            mapping_times_s=global_map["mapping_times_s"],
            mapping_camera_positions=global_map[
                "mapping_camera_positions"
            ],
            mapping_camera_rotations=global_map[
                "mapping_camera_rotations"
            ],
            mapping_camera_headings=global_map[
                "mapping_camera_headings"
            ],
            mapping_reference_frame=global_map[
                "mapping_reference_frame"
            ],
            mapping_extracted_image_count=global_map[
                "mapping_extracted_image_count"
            ],
            candidate_landmarks=global_map["candidate_landmarks"],
            occupied_grid_cells=global_map["occupied_grid_cells"],
            landmarks_per_occupied_cell=global_map[
                "landmarks_per_occupied_cell"
            ],
            initial_R=global_map["initial_R"],
            initial_t=global_map["initial_t"],
            last_mapping_image=global_map["last_mapping_image"],
        )

    def build(self, video_path, output_dir):
        build_started = time.perf_counter()
        output_dir = Path(output_dir)
        work_dir = output_dir / "colmap_work"
        images_dir = work_dir / "images"
        sparse_dir = work_dir / "sparse"
        database_path = work_dir / "database.db"

        if work_dir.exists():
            shutil.rmtree(work_dir)
        images_dir.mkdir(parents=True)
        sparse_dir.mkdir()

        (
            image_names,
            features_by_name,
            aruco_poses,
            frame_times_by_name,
            frame_processing_statistics,
        ) = self.collect_and_match_mapping_frames(
            video_path,
            images_dir,
            database_path,
        )

        reconstruction_started = time.perf_counter()
        reconstruction = self.reconstruct(
            database_path,
            images_dir,
            sparse_dir,
        )
        reconstruction_seconds = (
            time.perf_counter() - reconstruction_started
        )
        all_track_length_statistics = summarize_track_lengths(
            [
                point.track.length()
                for point in reconstruction.points3D.values()
            ]
        )

        alignment_started = time.perf_counter()
        alignment = self.align_to_first_camera(
            reconstruction,
            aruco_poses,
        )
        alignment_seconds = time.perf_counter() - alignment_started

        map_finalization_started = time.perf_counter()
        global_map = self.create_global_map(
            reconstruction,
            features_by_name,
            alignment,
            frame_times_by_name,
        )
        global_map["retrieval_diagnostics"] = (
            frame_processing_statistics["retrieval_diagnostics"]
        )
        map_finalization_seconds = (
            time.perf_counter() - map_finalization_started
        )
        global_map["mapping_extracted_image_count"] = len(image_names)
        track_length_statistics = {
            "all_colmap_points": all_track_length_statistics,
            "quality_filtered_candidates": global_map[
                "candidate_track_length_statistics"
            ],
            "selected_global_map": global_map[
                "selected_track_length_statistics"
            ],
        }

        saving_started = time.perf_counter()
        map_path = output_dir / "global_map.npz"
        self.save_map(global_map, map_path)
        model_dir = output_dir / "colmap_model"
        model_dir.mkdir(exist_ok=True)
        reconstruction.write(model_dir)
        saving_seconds = time.perf_counter() - saving_started

        after_collection_seconds = (
            reconstruction_seconds
            + alignment_seconds
            + map_finalization_seconds
            + saving_seconds
        )
        processed_frames = len(image_names)
        online_wall_seconds = frame_processing_statistics[
            "mapping_frame_processing_wall_seconds"
        ]
        timing = {
            "mapping_frame_processing": {
                "processed_frames": processed_frames,
                "attempted_pairs": frame_processing_statistics[
                    "attempted_pairs"
                ],
                "verified_pairs": frame_processing_statistics[
                    "verified_pairs"
                ],
                "sequential_attempted_pairs": (
                    frame_processing_statistics[
                        "sequential_attempted_pairs"
                    ]
                ),
                "sequential_verified_pairs": (
                    frame_processing_statistics[
                        "sequential_verified_pairs"
                    ]
                ),
                "retrieval_attempted_pairs": (
                    frame_processing_statistics[
                        "retrieval_attempted_pairs"
                    ]
                ),
                "retrieval_verified_pairs": (
                    frame_processing_statistics[
                        "retrieval_verified_pairs"
                    ]
                ),
                "setup_s": frame_processing_statistics["setup_seconds"],
                "frame_read_s": frame_processing_statistics[
                    "frame_read_seconds"
                ],
                "image_save_s": frame_processing_statistics[
                    "image_save_seconds"
                ],
                "disk_and_feature_selection_s": frame_processing_statistics[
                    "disk_and_selection_seconds"
                ],
                "retrieval_voting_s": frame_processing_statistics[
                    "retrieval_voting_seconds"
                ],
                "aruco_detection_s": frame_processing_statistics[
                    "aruco_detection_seconds"
                ],
                "image_and_keypoints_database_write_s": (
                    frame_processing_statistics[
                        "image_database_write_seconds"
                    ]
                ),
                "lightglue_matching_s": frame_processing_statistics[
                    "lightglue_seconds"
                ],
                "pair_geometry_verification_s": frame_processing_statistics[
                    "geometry_verification_seconds"
                ],
                "verified_pair_database_write_s": frame_processing_statistics[
                    "pair_database_write_seconds"
                ],
                "wall_time_s": online_wall_seconds,
                "average_wall_time_per_frame_ms": (
                    1000.0 * online_wall_seconds / processed_frames
                ),
            },
            "after_collection": {
                "colmap_reconstruction_and_bundle_adjustment_s": (
                    reconstruction_seconds
                ),
                "aruco_metric_alignment_s": alignment_seconds,
                "landmark_selection_and_descriptors_s": (
                    map_finalization_seconds
                ),
                "map_saving_s": saving_seconds,
                "wall_time_s": after_collection_seconds,
            },
            "total_map_build_wall_time_s": (
                time.perf_counter() - build_started
            ),
        }

        summary = {
            "mapping_start_frame": self.mapping_start_frame,
            "mapping_end_frame": self.mapping_end_frame,
            "reconstruction_method": self.reconstruction_method,
            "mapping_frame_step": self.mapping_frame_step,
            "sequential_match_overlap": self.sequential_match_overlap,
            "retrieval_enabled": self.enable_retrieval,
            "retrieval_top_frames": self.retrieval_top_frames,
            "retrieval_min_frame_gap": self.retrieval_min_frame_gap,
            "retrieval_descriptors_per_frame": (
                self.retrieval_descriptors_per_frame
            ),
            "retrieval_min_sequence_frames": (
                self.retrieval_min_sequence_frames
            ),
            "retrieval_max_sequence_gap": (
                self.retrieval_max_sequence_gap
            ),
            "retrieval_min_covered_cells": (
                self.retrieval_min_covered_cells
            ),
            "extracted_images": len(image_names),
            "attempted_image_pairs": frame_processing_statistics[
                "attempted_pairs"
            ],
            "verified_image_pairs": frame_processing_statistics[
                "verified_pairs"
            ],
            "sequential_attempted_image_pairs": (
                frame_processing_statistics[
                    "sequential_attempted_pairs"
                ]
            ),
            "sequential_verified_image_pairs": (
                frame_processing_statistics[
                    "sequential_verified_pairs"
                ]
            ),
            "retrieval_attempted_image_pairs": (
                frame_processing_statistics[
                    "retrieval_attempted_pairs"
                ]
            ),
            "retrieval_verified_image_pairs": (
                frame_processing_statistics[
                    "retrieval_verified_pairs"
                ]
            ),
            "registered_images": reconstruction.num_reg_images(),
            "landmarks": len(global_map["positions"]),
            "candidate_landmarks": global_map["candidate_landmarks"],
            "occupied_grid_cells": global_map["occupied_grid_cells"],
            "landmarks_per_occupied_cell": global_map[
                "landmarks_per_occupied_cell"
            ],
            "alignment_frames": alignment["alignment_frames"],
            "alignment_rmse_mm": alignment["alignment_rmse_mm"],
            "reference_image": alignment["reference_image"],
            "last_mapping_image": global_map["last_mapping_image"],
            "track_length_statistics": track_length_statistics,
            "timing": timing,
        }
        with (output_dir / "map_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)
        with (output_dir / "map_timing.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(timing, file, indent=2)
        with (output_dir / "track_length_statistics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(track_length_statistics, file, indent=2)
        with (output_dir / "retrieval_diagnostics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                global_map["retrieval_diagnostics"],
                file,
                indent=2,
            )

        print(f"Saved frozen 3D map: {map_path}")
        print(
            "Saved track statistics: "
            f"{output_dir / 'track_length_statistics.json'}"
        )
        print(
            f"Map: {summary['landmarks']}/"
            f"{summary['candidate_landmarks']} selected landmarks | "
            f"{summary['occupied_grid_cells']} occupied grid cells | "
            f"{summary['registered_images']}/{summary['extracted_images']} "
            "registered images"
        )
        print_track_length_statistics(track_length_statistics)
        print_map_build_timing(timing)
        return global_map
