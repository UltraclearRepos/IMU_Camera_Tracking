from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np


@dataclass(frozen=True)
class SkinMaskResult:
    mask: np.ndarray
    bounds: np.ndarray


class AdaptiveSkinMask:
    """Learn skin from the first central seed and mask it in every frame."""

    def __init__(
        self,
        seed_width_fraction=0.4,
        seed_height_fraction=0.85,
        cluster_count=3,
        processing_width=160,
    ):
        self.seed_width_fraction = seed_width_fraction
        self.seed_height_fraction = seed_height_fraction
        self.cluster_count = cluster_count
        self.processing_width = processing_width

        self._centres = None
        self._scales = None
        self._distance_threshold = None
        self.last_result = None
        self.initial_frame = None
        self.initial_seed = None
        self.initial_valid = None
        self.initial_result = None

        self.compute_count = 0
        self.compute_seconds = 0.0

    @property
    def initialized(self):
        return self._centres is not None

    @property
    def average_compute_ms(self):
        if self.compute_count == 0:
            return 0.0
        return 1000.0 * self.compute_seconds / self.compute_count

    def compute(self, frame, roi_top, aruco_exclusion_mask):
        started = perf_counter()
        try:
            return self._compute(frame, roi_top, aruco_exclusion_mask)
        finally:
            self.compute_count += 1
            self.compute_seconds += perf_counter() - started

    def _compute(self, frame, roi_top, aruco_exclusion_mask):
        height, width = frame.shape[:2]
        if aruco_exclusion_mask.shape != (height, width):
            raise ValueError("aruco_exclusion_mask must match the frame size")

        scale = min(1.0, self.processing_width / width)
        if scale < 1.0:
            processing_size = (round(width * scale), round(height * scale))
            processing_frame = cv2.resize(
                frame,
                processing_size,
                interpolation=cv2.INTER_AREA,
            )
            processing_aruco_mask = cv2.resize(
                aruco_exclusion_mask,
                processing_size,
                interpolation=cv2.INTER_NEAREST,
            )
            processing_roi_top = round(roi_top * scale)
        else:
            processing_frame = frame
            processing_aruco_mask = aruco_exclusion_mask
            processing_roi_top = roi_top

        processing_height, processing_width = processing_frame.shape[:2]
        valid = processing_aruco_mask.astype(bool).copy()
        valid[:processing_roi_top] = False
        features = self._features(processing_frame, valid)
        first_frame = not self.initialized

        seed = None
        if first_frame:
            seed = self._initial_seed(
                processing_height,
                processing_width,
                processing_roi_top,
                valid,
            )
            samples = features[seed]
            if len(samples) < 100:
                self.last_result = None
                return None
            self._centres, self._scales = self._fit_model(samples)
            seed_distances = self._distance(features, seed)
            self._distance_threshold = float(
                np.clip(np.percentile(seed_distances, 99.0), 7.0, 16.0)
            )

        distance = self._distance(features)
        candidate = (distance <= self._distance_threshold) & valid
        if first_frame:
            candidate[seed] = True

        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((7, 7), dtype=np.uint8),
        )
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        ).astype(bool)
        candidate &= valid

        selected = self._select_component(candidate, seed)
        if not np.any(selected):
            self.last_result = None
            return None

        if scale < 1.0:
            selected = cv2.resize(
                selected.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        selected &= aruco_exclusion_mask.astype(bool)
        selected[:roi_top] = False

        mask_y, mask_x = np.nonzero(selected)
        result = SkinMaskResult(
            mask=selected,
            bounds=np.array(
                [mask_x.min(), mask_y.min(), mask_x.max() + 1, mask_y.max() + 1],
                dtype=np.int32,
            ),
        )

        if first_frame:
            self.initial_frame = frame.copy()
            self.initial_seed = cv2.resize(
                seed.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            self.initial_valid = aruco_exclusion_mask.astype(bool).copy()
            self.initial_valid[:roi_top] = False
            self.initial_result = SkinMaskResult(
                mask=result.mask.copy(),
                bounds=result.bounds.copy(),
            )

        self.last_result = result
        return result

    def _initial_seed(self, height, width, roi_top, valid):
        seed = np.zeros((height, width), dtype=bool)
        half_width = round(width * self.seed_width_fraction / 2.0)
        half_height = round(
            (height - roi_top) * self.seed_height_fraction / 2.0
        )
        center_x = width // 2
        center_y = roi_top + (height - roi_top) // 2
        seed[
            max(roi_top, center_y - half_height):min(height, center_y + half_height),
            max(0, center_x - half_width):min(width, center_x + half_width),
        ] = True
        seed &= valid
        return seed

    @staticmethod
    def _features(frame, valid):
        # Opponent colour ratios and frame-local standardisation reduce
        # exposure sensitivity without costly logarithms or percentiles.
        bgr = frame.astype(np.float32) + 8.0
        blue, green, red = cv2.split(bgr)
        channel_sum = blue + green + red
        features = np.empty_like(bgr)
        features[:, :, 0] = (red - green) / channel_sum
        features[:, :, 1] = (blue - green) / channel_sum
        features[:, :, 2] = 0.114 * blue + 0.587 * green + 0.299 * red

        centre, scale = cv2.meanStdDev(
            features,
            mask=valid.astype(np.uint8),
        )
        centre = centre.ravel()
        scale = np.maximum(scale.ravel(), np.array([0.01, 0.01, 8.0]))
        return ((features - centre) / scale).astype(np.float32)

    def _fit_model(self, samples):
        if len(samples) > 10000:
            samples = samples[::max(1, len(samples) // 10000)]

        # Splitting by luminance models textured dark skin without expensive
        # k-means and keeps cluster ordering stable during online updates.
        samples = samples[np.argsort(samples[:, 2])]
        groups = np.array_split(samples, self.cluster_count)
        centres = []
        scales = []
        for group in groups:
            centre = np.median(group, axis=0)
            scale = 1.4826 * np.median(np.abs(group - centre), axis=0)
            centres.append(centre)
            scales.append(np.maximum(scale, 0.12))
        return np.asarray(centres), np.asarray(scales)

    def _distance(self, features, mask=None):
        samples = features if mask is None else features[mask]
        distance = np.full(samples.shape[:-1], np.inf, dtype=np.float32)
        for centre, scale in zip(self._centres, self._scales):
            cluster_distance = np.sum(((samples - centre) / scale) ** 2, axis=-1)
            distance = np.minimum(distance, cluster_distance)
        return distance

    @staticmethod
    def _select_component(candidate, seed):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8),
            connectivity=8,
        )
        if count <= 1:
            return np.zeros_like(candidate)
        if seed is None:
            label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        else:
            overlap = np.bincount(labels[seed], minlength=count)
            overlap[0] = 0
            label = int(np.argmax(overlap))
            if overlap[label] == 0:
                return np.zeros_like(candidate)
        return labels == label
