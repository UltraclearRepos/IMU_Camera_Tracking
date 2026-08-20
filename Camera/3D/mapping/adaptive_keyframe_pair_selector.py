from collections import Counter, deque
from dataclasses import replace

import cv2
import numpy as np

from mapping.mapping_data import (
    KeyframePairCandidate,
    KeyframePairSelection,
    LocalTrackSnapshot,
)


class AdaptiveKeyframePairSelector:
    """Maintain local LK tracks and use them to select useful image pairs."""

    MINIMUM_NEW_TRACK_DISTANCE_PX = 5.0
    MAXIMUM_ACTIVE_TRACK_COUNT = 256
    MAXIMUM_FORWARD_BACKWARD_ERROR_PX = 1.0
    MINIMUM_KEYFRAME_OVERLAP = 0.5
    MAXIMUM_MOTION_ANCHOR_PX = 40.0

    def __init__(self, recent_pair_count, motion_targets_px=()):
        if recent_pair_count < 0:
            raise ValueError("recent_pair_count must be non-negative")

        motion_targets_px = tuple(float(target) for target in motion_targets_px)
        if any(
            not np.isfinite(target) or target <= 0.0
            for target in motion_targets_px
        ):
            raise ValueError("motion targets must be positive finite values")
        if len(set(motion_targets_px)) != len(motion_targets_px):
            raise ValueError("motion targets must be unique")
        if any(
            target > self.MAXIMUM_MOTION_ANCHOR_PX
            for target in motion_targets_px
        ):
            raise ValueError(
                "motion targets must not exceed "
                f"{self.MAXIMUM_MOTION_ANCHOR_PX:g} px"
            )

        self.recent_pair_count = recent_pair_count
        self.motion_targets_px = motion_targets_px
        self._next_track_id = 0
        self._previous_gray = None
        self._active_track_positions = np.empty((0, 2), dtype=np.float32)
        self._active_track_ids = np.empty(0, dtype=np.int64)
        self._active_keyframes = {}
        self._keyframes_by_track = {}
        self._recent_keyframes = deque(maxlen=recent_pair_count + 1)

    def update_tracks(self, frame, sift_keypoints):
        current_gray = self._grayscale(frame)
        continued_positions, continued_ids = self._track_active_points(
            current_gray
        )
        new_positions = self._select_new_track_positions(
            sift_keypoints,
            continued_positions,
            self.MAXIMUM_ACTIVE_TRACK_COUNT - len(continued_positions),
        )
        new_ids = np.arange(
            self._next_track_id,
            self._next_track_id + len(new_positions),
            dtype=np.int64,
        )
        self._next_track_id += len(new_ids)

        self._active_track_positions = np.concatenate(
            (continued_positions, new_positions),
            axis=0,
        )
        self._active_track_ids = np.concatenate((continued_ids, new_ids))
        self._previous_gray = current_gray.copy()

        return LocalTrackSnapshot(
            track_ids=self._active_track_ids.copy(),
            positions=self._active_track_positions.copy(),
            continued_track_count=len(continued_ids),
            new_track_count=len(new_ids),
        )

    def register_keyframe(self, image):
        image_id = image.database_image_id
        self._active_keyframes[image_id] = image
        for track_id in image.local_tracks.track_ids:
            self._keyframes_by_track.setdefault(int(track_id), set()).add(
                image_id
            )
        self._recent_keyframes.append(image)

    def select_pairs(self, current_image):
        common_track_counts = self._common_track_counts(
            current_image.local_tracks.track_ids
        )
        recent_images = list(self._recent_keyframes)[:-1]
        recent_image_ids = {
            image.database_image_id for image in recent_images
        }

        motion_candidates = []
        active_candidate_count = 0
        expired_keyframe_ids = []
        overlap_by_image_id = {}
        for image_id, previous_image in self._active_keyframes.items():
            if image_id == current_image.database_image_id:
                continue
            shared_track_count = common_track_counts.get(image_id, 0)
            overlap = self._overlap(previous_image, shared_track_count)
            overlap_by_image_id[image_id] = overlap
            if overlap < self.MINIMUM_KEYFRAME_OVERLAP:
                if image_id not in recent_image_ids:
                    expired_keyframe_ids.append(image_id)
                continue
            active_candidate_count += 1
            if image_id in recent_image_ids:
                continue
            motion_candidates.append(
                self._candidate(
                    previous_image,
                    current_image,
                    overlap=overlap,
                    reason="overlap_candidate",
                )
            )

        recent_pairs = self._recent_pairs(
            current_image,
            recent_images,
            overlap_by_image_id,
        )
        motion_anchors = self._select_motion_anchors(motion_candidates)

        for image_id in expired_keyframe_ids:
            self._remove_active_keyframe(image_id)

        return KeyframePairSelection(
            pairs=tuple(recent_pairs + motion_anchors),
            active_candidate_count=active_candidate_count,
        )

    def _track_active_points(self, current_gray):
        if self._previous_gray is None or not len(self._active_track_positions):
            return self._empty_tracks()

        previous_points = self._active_track_positions.reshape(-1, 1, 2)
        forward_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if forward_points is None or forward_status is None:
            return self._empty_tracks()

        forward_points = forward_points.reshape(-1, 2)
        image_height, image_width = current_gray.shape[:2]
        forward_valid = (
            forward_status.reshape(-1).astype(bool)
            & np.all(np.isfinite(forward_points), axis=1)
            & (forward_points[:, 0] >= 0.0)
            & (forward_points[:, 0] < image_width)
            & (forward_points[:, 1] >= 0.0)
            & (forward_points[:, 1] < image_height)
        )
        if not np.any(forward_valid):
            return self._empty_tracks()

        forward_points = forward_points[forward_valid]
        previous_positions = self._active_track_positions[forward_valid]
        track_ids = self._active_track_ids[forward_valid]
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self._previous_gray,
            forward_points.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if backward_points is None or backward_status is None:
            return self._empty_tracks()

        backward_points = backward_points.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(
            backward_points - previous_positions,
            axis=1,
        )
        reliable = (
            backward_status.reshape(-1).astype(bool)
            & np.all(np.isfinite(backward_points), axis=1)
            & (
                forward_backward_error
                <= self.MAXIMUM_FORWARD_BACKWARD_ERROR_PX
            )
        )
        return forward_points[reliable], track_ids[reliable]

    def _select_new_track_positions(
        self,
        sift_keypoints,
        continued_positions,
        maximum_new_track_count,
    ):
        if maximum_new_track_count <= 0:
            return np.empty((0, 2), dtype=np.float32)

        sift_keypoints = np.asarray(
            sift_keypoints,
            dtype=np.float32,
        ).reshape(-1, 2)

        occupied_positions = [
            position.copy() for position in continued_positions
        ]
        new_positions = []

        for keypoint in sift_keypoints:
            if not np.all(np.isfinite(keypoint)):
                continue
            if occupied_positions:
                distances = np.linalg.norm(
                    np.asarray(occupied_positions) - keypoint,
                    axis=1,
                )
                if np.min(distances) <= self.MINIMUM_NEW_TRACK_DISTANCE_PX:
                    continue
            new_positions.append(keypoint.copy())
            occupied_positions.append(keypoint.copy())
            if len(new_positions) == maximum_new_track_count:
                break

        if not new_positions:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(new_positions, dtype=np.float32)

    def _common_track_counts(self, current_track_ids):
        counts = Counter()
        for track_id in current_track_ids:
            counts.update(self._keyframes_by_track.get(int(track_id), ()))
        return counts

    def _recent_pairs(
        self,
        current_image,
        recent_images,
        overlap_by_image_id,
    ):
        if self.recent_pair_count == 0:
            return []
        return [
            self._candidate(
                previous_image,
                current_image,
                overlap=overlap_by_image_id[previous_image.database_image_id],
                reason="recent",
            )
            for previous_image in recent_images
        ]

    def _select_motion_anchors(self, candidates):
        if not self.motion_targets_px:
            return []

        candidates = [
            candidate
            for candidate in candidates
            if (
                np.isfinite(candidate.median_displacement_px)
                and candidate.median_displacement_px
                <= self.MAXIMUM_MOTION_ANCHOR_PX
            )
        ]
        if not candidates:
            return []

        maximum_available_motion = max(
            candidate.median_displacement_px for candidate in candidates
        )

        selected_by_image_id = {}
        for target_px in self.motion_targets_px:
            if maximum_available_motion < target_px:
                continue
            anchor = min(
                candidates,
                key=lambda candidate: (
                    abs(candidate.median_displacement_px - target_px),
                    -candidate.overlap,
                    candidate.image.database_image_id,
                ),
            )
            selected_by_image_id.setdefault(
                anchor.image.database_image_id,
                replace(
                    anchor,
                    reason="motion_target",
                    motion_target_px=target_px,
                ),
            )

        return list(selected_by_image_id.values())

    @staticmethod
    def _candidate(previous_image, current_image, overlap, reason):
        shared_ids, previous_indices, current_indices = np.intersect1d(
            previous_image.local_tracks.track_ids,
            current_image.local_tracks.track_ids,
            assume_unique=True,
            return_indices=True,
        )
        shared_track_count = len(shared_ids)
        if shared_track_count:
            displacements = np.linalg.norm(
                current_image.local_tracks.positions[current_indices]
                - previous_image.local_tracks.positions[previous_indices],
                axis=1,
            )
            median_displacement_px = float(np.median(displacements))
        else:
            median_displacement_px = np.nan

        return KeyframePairCandidate(
            image=previous_image,
            shared_track_count=shared_track_count,
            overlap=overlap,
            median_displacement_px=median_displacement_px,
            reason=reason,
            motion_target_px=None,
        )

    @staticmethod
    def _overlap(previous_image, shared_track_count):
        if not len(previous_image.local_tracks):
            return 0.0
        return shared_track_count / len(previous_image.local_tracks)

    def _remove_active_keyframe(self, image_id):
        image = self._active_keyframes.pop(image_id)
        for track_id in image.local_tracks.track_ids:
            track_keyframes = self._keyframes_by_track[int(track_id)]
            track_keyframes.discard(image_id)
            if not track_keyframes:
                del self._keyframes_by_track[int(track_id)]

    @staticmethod
    def _empty_tracks():
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    @staticmethod
    def _grayscale(frame):
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
