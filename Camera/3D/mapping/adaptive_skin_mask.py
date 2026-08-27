from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SkinMaskResult:
    mask: np.ndarray
    bounds: np.ndarray


class AdaptiveSkinMask:
    """Maintain an adaptive LAB skin-colour model and produce binary masks."""

    STANDARD_DEVIATION_BY_SKIN_TONE = {
        "black": 0.5,
        "white": 5.0,
    }

    def __init__(
        self,
        skin_tone,
        seed_width_fraction=0.4,
        seed_height_fraction=0.85,
        model_update_rate=0.03,
    ):
        skin_tone = skin_tone.lower()
        if skin_tone not in self.STANDARD_DEVIATION_BY_SKIN_TONE:
            raise ValueError(
                "skin_tone must be one of "
                f"{tuple(self.STANDARD_DEVIATION_BY_SKIN_TONE)}, "
                f"got {skin_tone!r}"
            )

        self.skin_tone = skin_tone
        self.seed_width_fraction = seed_width_fraction
        self.seed_height_fraction = seed_height_fraction
        self.standard_deviation = (
            self.STANDARD_DEVIATION_BY_SKIN_TONE[skin_tone]
        )
        self.model_update_rate = model_update_rate
        self._skin_mean_ab = None
        self._skin_lower_bound = None
        self._skin_upper_bound = None
        self._previous_mask = None
        self.last_result = None
        self.initial_frame = None
        self.initial_seed = None
        self.initial_valid = None
        self.initial_result = None

    @property
    def initialized(self):
        return self._skin_mean_ab is not None

    def compute(self, frame, roi_top, aruco_exclusion_mask):
        height, width = frame.shape[:2]
        if aruco_exclusion_mask.shape != (height, width):
            raise ValueError("aruco_exclusion_mask must match the frame size")

        valid = aruco_exclusion_mask.astype(bool).copy()
        valid[:roi_top] = False
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        ab = lab[:, :, 1:3]

        initialized_from_seed = not self.initialized
        seed = None
        if initialized_from_seed:
            seed = self._initial_seed(height, width, roi_top, valid)
            samples = ab[seed]
            if len(samples) < 100:
                self.last_result = None
                return None
            median = np.median(samples, axis=0)
            distances = np.linalg.norm(samples - median, axis=1)
            samples = samples[distances <= np.percentile(distances, 80.0)]
            self._set_model(samples)

        candidate = cv2.inRange(
            ab,
            self._skin_lower_bound,
            self._skin_upper_bound,
        ).astype(bool)
        candidate &= valid
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((9, 9), dtype=np.uint8),
        ).astype(bool)
        selected = self._select_component(candidate, roi_top)
        if selected is None:
            self.last_result = None
            return None

        mask_y, mask_x = np.nonzero(selected)
        result = SkinMaskResult(
            mask=selected,
            bounds=np.array(
                [
                    mask_x.min(),
                    mask_y.min(),
                    mask_x.max() + 1,
                    mask_y.max() + 1,
                ],
                dtype=np.int32,
            ),
        )
        if initialized_from_seed:
            self.initial_frame = frame.copy()
            self.initial_seed = seed.copy()
            self.initial_valid = valid.copy()
            self.initial_result = SkinMaskResult(
                mask=result.mask.copy(),
                bounds=result.bounds.copy(),
            )
        self._update_model(ab[selected])
        self._previous_mask = selected
        self.last_result = result
        return result

    def _initial_seed(self, height, width, roi_top, valid):
        seed = np.zeros((height, width), dtype=bool)
        seed_half_width = round(width * self.seed_width_fraction / 2.0)
        seed_half_height = round(
            (height - roi_top) * self.seed_height_fraction / 2.0
        )
        center_x = width // 2
        center_y = roi_top + (height - roi_top) // 2
        seed[
            max(roi_top, center_y - seed_half_height):min(
                height, center_y + seed_half_height
            ),
            max(0, center_x - seed_half_width):min(
                width, center_x + seed_half_width
            ),
        ] = True
        seed &= valid
        return seed

    def _select_component(self, candidate, roi_top):
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8),
            connectivity=8,
        )
        if component_count <= 1:
            return None

        reference = self._previous_mask
        if reference is None:
            height, width = candidate.shape
            reference = np.zeros_like(candidate)
            reference[
                roi_top + (height - roi_top) // 4:
                roi_top + 3 * (height - roi_top) // 4,
                width // 3:2 * width // 3,
            ] = True

        best_label = 0
        best_score = -np.inf
        for label in range(1, component_count):
            component = labels == label
            overlap = np.count_nonzero(component & reference)
            area = stats[label, cv2.CC_STAT_AREA]
            score = 1000.0 * overlap + area
            if score > best_score:
                best_label = label
                best_score = score

        if best_label == 0:
            return None
        return labels == best_label

    def _set_model(self, samples):
        self._skin_mean_ab = np.mean(samples, axis=0, dtype=np.float32)
        measured_standard_deviation = np.std(
            samples,
            axis=0,
            dtype=np.float32,
        )
        print(
            f"Initial {self.skin_tone} skin colour std (LAB a/b): "
            f"measured=[{measured_standard_deviation[0]:.2f}, "
            f"{measured_standard_deviation[1]:.2f}], "
            f"used={self.standard_deviation:g}"
        )
        self._update_bounds()

    def _update_model(self, samples):
        if len(samples) < 100:
            return
        samples = samples[::max(1, len(samples) // 10000)]
        mean = np.mean(samples, axis=0, dtype=np.float32)
        rate = self.model_update_rate
        self._skin_mean_ab = (1.0 - rate) * self._skin_mean_ab + rate * mean
        self._update_bounds()

    def _update_bounds(self):
        self._skin_lower_bound = np.clip(
            self._skin_mean_ab - self.standard_deviation,
            0,
            255,
        ).astype(np.uint8)
        self._skin_upper_bound = np.clip(
            self._skin_mean_ab + self.standard_deviation,
            0,
            255,
        ).astype(np.uint8)
