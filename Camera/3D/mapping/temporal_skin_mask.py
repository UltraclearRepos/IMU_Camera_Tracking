import cv2
import numpy as np

from mapping.adaptive_skin_mask import AdaptiveSkinMask, SkinMaskResult


class TemporalSkinMask(AdaptiveSkinMask):
    """Grow skin from the eroded previous mask in a fixed search area."""

    def __init__(
        self,
        erosion_kernel_size=13,
        search_kernel_size=17,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.erosion_kernel = np.ones(
            (erosion_kernel_size, erosion_kernel_size),
            dtype=np.uint8,
        )
        self.search_kernel = np.ones(
            (search_kernel_size, search_kernel_size),
            dtype=np.uint8,
        )
        self.close_kernel = np.ones((7, 7), dtype=np.uint8)
        self.open_kernel = np.ones((3, 3), dtype=np.uint8)
        self._previous_processing_mask = None
        self.last_eroded_previous_mask = None
        self.last_search_mask = None

    def _compute(self, frame, roi_top, aruco_exclusion_mask):
        (
            processing_frame,
            valid,
            processing_roi_top,
        ) = self._prepare_frame(frame, roi_top, aruco_exclusion_mask)

        if not self.initialized:
            return self._initialize(
                frame,
                roi_top,
                aruco_exclusion_mask,
                processing_frame,
                valid,
                processing_roi_top,
            )

        eroded_previous = cv2.erode(
            self._previous_processing_mask.astype(np.uint8),
            self.erosion_kernel,
        ).astype(bool)
        search_mask = cv2.dilate(
            self._previous_processing_mask.astype(np.uint8),
            self.search_kernel,
        ).astype(bool)
        eroded_previous &= valid
        search_mask &= valid
        self.last_eroded_previous_mask = eroded_previous.copy()
        self.last_search_mask = search_mask.copy()

        if not np.any(eroded_previous):
            self.last_result = None
            return None

        search_y, search_x = np.nonzero(search_mask)
        top, bottom = search_y.min(), search_y.max() + 1
        left, right = search_x.min(), search_x.max() + 1
        region = np.s_[top:bottom, left:right]
        region_seed = eroded_previous[region]
        region_search = search_mask[region]
        region_valid = valid[region]

        features = self._normalised_features(
            processing_frame[region],
            region_seed,
        )
        distance = self._distance(features)
        candidate = (
            (distance <= self._distance_threshold)
            & region_search
            & region_valid
        )
        candidate[region_seed] = True
        selected_region = self._grow(candidate, region_search, region_seed)
        if not np.any(selected_region):
            self.last_result = None
            return None

        selected = np.zeros_like(valid)
        selected[region] = selected_region
        self._previous_processing_mask = selected
        result = self._result(
            selected,
            frame.shape[1],
            frame.shape[0],
            roi_top,
            aruco_exclusion_mask,
        )
        self.last_result = result
        return result

    def _initialize(
        self,
        frame,
        roi_top,
        aruco_exclusion_mask,
        processing_frame,
        valid,
        processing_roi_top,
    ):
        height, width = processing_frame.shape[:2]
        seed = self._initial_seed(
            height,
            width,
            processing_roi_top,
            valid,
        )
        if np.count_nonzero(seed) < 100:
            self.last_result = None
            return None

        features = self._normalised_features(processing_frame, seed)
        self._centres, self._scales = self._fit_model(features[seed])
        seed_distances = self._distance(features, seed)
        self._distance_threshold = float(
            np.clip(np.percentile(seed_distances, 99.0), 7.0, 16.0)
        )

        candidate = (self._distance(features) <= self._distance_threshold) & valid
        candidate[seed] = True
        selected = self._grow(candidate, valid, seed)
        if not np.any(selected):
            self.last_result = None
            return None

        self._previous_processing_mask = selected
        result = self._result(
            selected,
            frame.shape[1],
            frame.shape[0],
            roi_top,
            aruco_exclusion_mask,
        )
        self.initial_frame = frame.copy()
        self.initial_seed = self._resize_mask(
            seed,
            frame.shape[1],
            frame.shape[0],
        )
        self.initial_valid = aruco_exclusion_mask.astype(bool).copy()
        self.initial_valid[:roi_top] = False
        self.initial_result = SkinMaskResult(
            mask=result.mask.copy(),
            bounds=result.bounds.copy(),
        )
        self.last_eroded_previous_mask = None
        self.last_search_mask = None
        self.last_result = result
        return result

    def _prepare_frame(self, frame, roi_top, aruco_exclusion_mask):
        height, width = frame.shape[:2]
        if aruco_exclusion_mask.shape != (height, width):
            raise ValueError("aruco_exclusion_mask must match the frame size")

        scale = min(1.0, self.processing_width / width)
        if scale < 1.0:
            size = (round(width * scale), round(height * scale))
            processing_frame = cv2.resize(
                frame,
                size,
                interpolation=cv2.INTER_AREA,
            )
            processing_aruco_mask = cv2.resize(
                aruco_exclusion_mask,
                size,
                interpolation=cv2.INTER_NEAREST,
            )
            processing_roi_top = round(roi_top * scale)
        else:
            processing_frame = frame
            processing_aruco_mask = aruco_exclusion_mask
            processing_roi_top = roi_top

        valid = processing_aruco_mask.astype(bool)
        valid[:processing_roi_top] = False
        return processing_frame, valid, processing_roi_top

    @staticmethod
    def _normalised_features(frame, reference_mask):
        bgr = frame.astype(np.float32) + 8.0
        blue, green, red = cv2.split(bgr)
        channel_sum = blue + green + red
        features = np.empty_like(bgr)
        features[:, :, 0] = (red - green) / channel_sum
        features[:, :, 1] = (blue - green) / channel_sum
        features[:, :, 2] = 0.114 * blue + 0.587 * green + 0.299 * red

        centre, scale = cv2.meanStdDev(
            features,
            mask=reference_mask.astype(np.uint8),
        )
        centre = centre.ravel()
        scale = np.maximum(scale.ravel(), np.array([0.01, 0.01, 8.0]))
        return ((features - centre) / scale).astype(np.float32)

    def _grow(self, candidate, allowed, seed):
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            self.close_kernel,
        )
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            self.open_kernel,
        ).astype(bool)
        candidate &= allowed
        return self._select_component(candidate, seed)

    def _result(
        self,
        processing_mask,
        width,
        height,
        roi_top,
        aruco_exclusion_mask,
    ):
        mask = self._resize_mask(processing_mask, width, height)
        mask &= aruco_exclusion_mask.astype(bool)
        mask[:roi_top] = False
        mask_y, mask_x = np.nonzero(mask)
        return SkinMaskResult(
            mask=mask,
            bounds=np.array(
                [mask_x.min(), mask_y.min(), mask_x.max() + 1, mask_y.max() + 1],
                dtype=np.int32,
            ),
        )

    @staticmethod
    def _resize_mask(mask, width, height):
        if mask.shape == (height, width):
            return mask.copy()
        return cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
