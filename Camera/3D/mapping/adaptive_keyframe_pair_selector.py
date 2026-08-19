from collections import Counter, deque
from dataclasses import replace

import cv2
import numpy as np

from mapping.mapping_data import (
    KeyframePairCandidate,
    KeyframePairSelection,
    LocalTrackAssignment,
)


class AdaptiveKeyframePairSelector:
    """Track local features and select geometrically useful image pairs."""

    TRACK_ASSOCIATION_RADIUS_PX = 3.0
    MAXIMUM_FORWARD_BACKWARD_ERROR_PX = 1.0
    MINIMUM_KEYFRAME_OVERLAP = 0.5
    FIRST_MOTION_TARGET_PX = 10.0
    MOTION_TARGET_STEP_PX = 20.0

    def __init__(self, recent_pair_count):
        if recent_pair_count < 0:
            raise ValueError("recent_pair_count must be non-negative")

        self.recent_pair_count = recent_pair_count
        self._next_track_id = 0
        self._previous_gray = None
        self._previous_keypoints = np.empty((0, 2), dtype=np.float32)
        self._previous_track_ids = np.empty(0, dtype=np.int64)
        self._active_keyframes = {}
        self._keyframes_by_track = {}
        self._recent_keyframes = deque(maxlen=recent_pair_count or None)

    def assign_track_ids(self, frame, features):
        gray = self._grayscale(frame)
        current_keypoints = features["keypoints"]
        current_track_ids = np.full(
            len(current_keypoints),
            -1,
            dtype=np.int64,
        )

        tracked_positions, tracked_ids = self._track_previous_points(gray)
        continued_count = self._associate_tracked_points(
            tracked_positions,
            tracked_ids,
            current_keypoints,
            current_track_ids,
        )
        new_mask = current_track_ids < 0
        new_count = int(np.count_nonzero(new_mask))
        current_track_ids[new_mask] = np.arange(
            self._next_track_id,
            self._next_track_id + new_count,
            dtype=np.int64,
        )
        self._next_track_id += new_count

        self._previous_gray = gray
        self._previous_keypoints = current_keypoints.copy()
        self._previous_track_ids = current_track_ids.copy()
        return LocalTrackAssignment(
            track_ids=current_track_ids,
            continued_track_count=continued_count,
            new_track_count=new_count,
        )

    def select_and_register(self, current_image):
        common_track_counts = self._common_track_counts(
            current_image.track_ids
        )
        valid_candidates = []
        expired_keyframe_ids = []
        for image_id, previous_image in self._active_keyframes.items():
            shared_track_count = common_track_counts.get(image_id, 0)
            overlap = self._overlap(previous_image, shared_track_count)
            if overlap < self.MINIMUM_KEYFRAME_OVERLAP:
                expired_keyframe_ids.append(image_id)
                continue
            valid_candidates.append(
                self._candidate(
                    previous_image,
                    current_image,
                    reason="overlap_candidate",
                )
            )

        recent_pairs = self._recent_pairs(current_image)
        recent_image_ids = {
            pair.image.database_image_id for pair in recent_pairs
        }
        motion_candidates = [
            candidate
            for candidate in valid_candidates
            if candidate.image.database_image_id not in recent_image_ids
        ]
        motion_pairs = self._select_motion_pairs(motion_candidates)

        for image_id in expired_keyframe_ids:
            self._remove_active_keyframe(image_id)
        self._register_active_keyframe(current_image)

        return KeyframePairSelection(
            pairs=tuple(recent_pairs + motion_pairs),
            active_candidate_count=len(valid_candidates),
        )

    def _track_previous_points(self, current_gray):
        if self._previous_gray is None or not len(self._previous_keypoints):
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        previous_points = self._previous_keypoints.reshape(-1, 1, 2)
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
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self._previous_gray,
            forward_points,
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
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        forward_points = forward_points.reshape(-1, 2)
        backward_points = backward_points.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(
            backward_points - self._previous_keypoints,
            axis=1,
        )
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.all(np.isfinite(forward_points), axis=1)
            & np.all(np.isfinite(backward_points), axis=1)
            & (
                forward_backward_error
                <= self.MAXIMUM_FORWARD_BACKWARD_ERROR_PX
            )
        )
        return forward_points[valid], self._previous_track_ids[valid]

    def _associate_tracked_points(
        self,
        tracked_positions,
        tracked_ids,
        current_keypoints,
        current_track_ids,
    ):
        if not len(tracked_positions) or not len(current_keypoints):
            return 0

        distances = np.linalg.norm(
            tracked_positions[:, None, :] - current_keypoints[None, :, :],
            axis=2,
        )
        tracked_indices, current_indices = np.nonzero(
            distances <= self.TRACK_ASSOCIATION_RADIUS_PX
        )
        candidate_order = np.argsort(
            distances[tracked_indices, current_indices]
        )
        used_tracked = np.zeros(len(tracked_positions), dtype=bool)
        used_current = np.zeros(len(current_keypoints), dtype=bool)

        continued_count = 0
        for candidate_index in candidate_order:
            tracked_index = tracked_indices[candidate_index]
            current_index = current_indices[candidate_index]
            if used_tracked[tracked_index] or used_current[current_index]:
                continue
            current_track_ids[current_index] = tracked_ids[tracked_index]
            used_tracked[tracked_index] = True
            used_current[current_index] = True
            continued_count += 1
        return continued_count

    def _common_track_counts(self, current_track_ids):
        counts = Counter()
        for track_id in current_track_ids:
            counts.update(self._keyframes_by_track.get(int(track_id), ()))
        return counts

    def _recent_pairs(self, current_image):
        if self.recent_pair_count == 0:
            return []
        return [
            self._candidate(
                previous_image,
                current_image,
                reason="recent",
            )
            for previous_image in self._recent_keyframes
        ]

    def _select_motion_pairs(self, candidates):
        candidates = [
            candidate
            for candidate in candidates
            if np.isfinite(candidate.median_displacement_px)
        ]
        if not candidates:
            return []

        maximum_motion = max(
            candidate.median_displacement_px for candidate in candidates
        )
        targets = self._motion_targets(maximum_motion)
        possible_assignments = sorted(
            (
                abs(candidate.median_displacement_px - target),
                target,
                candidate.image.database_image_id,
                candidate,
            )
            for target in targets
            for candidate in candidates
        )
        selected_by_target = {}
        selected_image_ids = set()
        for _, target, _, candidate in possible_assignments:
            image_id = candidate.image.database_image_id
            if target in selected_by_target or image_id in selected_image_ids:
                continue
            selected_by_target[target] = replace(
                candidate,
                reason="motion_target",
                motion_target_px=target,
            )
            selected_image_ids.add(image_id)

        return [
            selected_by_target[target]
            for target in targets
            if target in selected_by_target
        ]

    def _motion_targets(self, maximum_motion):
        targets = []
        if maximum_motion >= self.FIRST_MOTION_TARGET_PX:
            targets.append(self.FIRST_MOTION_TARGET_PX)

        target = self.MOTION_TARGET_STEP_PX
        while target <= maximum_motion:
            targets.append(target)
            target += self.MOTION_TARGET_STEP_PX
        return targets

    @staticmethod
    def _candidate(previous_image, current_image, reason):
        shared_ids, previous_indices, current_indices = np.intersect1d(
            previous_image.track_ids,
            current_image.track_ids,
            assume_unique=True,
            return_indices=True,
        )
        shared_track_count = len(shared_ids)
        overlap = AdaptiveKeyframePairSelector._overlap(
            previous_image,
            shared_track_count,
        )
        if shared_track_count:
            displacements = np.linalg.norm(
                current_image.features["keypoints"][current_indices]
                - previous_image.features["keypoints"][previous_indices],
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
        if not len(previous_image.track_ids):
            return 0.0
        return shared_track_count / len(previous_image.track_ids)

    def _register_active_keyframe(self, image):
        image_id = image.database_image_id
        self._active_keyframes[image_id] = image
        for track_id in image.track_ids:
            self._keyframes_by_track.setdefault(int(track_id), set()).add(
                image_id
            )
        if self.recent_pair_count:
            self._recent_keyframes.append(image)

    def _remove_active_keyframe(self, image_id):
        image = self._active_keyframes.pop(image_id)
        for track_id in image.track_ids:
            track_keyframes = self._keyframes_by_track[int(track_id)]
            track_keyframes.discard(image_id)
            if not track_keyframes:
                del self._keyframes_by_track[int(track_id)]

    @staticmethod
    def _grayscale(frame):
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
