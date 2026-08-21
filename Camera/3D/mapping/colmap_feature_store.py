import cv2
import numpy as np
import pycolmap


class ColmapFeatureStore:
    """Load registered RootSIFT observations after reconstruction."""

    def __init__(self, database_path, masks_directory):
        self.database_path = database_path
        self.masks_directory = masks_directory

    def load_registered(self, reconstruction):
        database = pycolmap.Database.open(self.database_path)
        try:
            return {
                image.name: self._load_image(database, image)
                for image in reconstruction.images.values()
            }
        finally:
            database.close()

    def _load_image(self, database, image):
        keypoints = database.read_keypoints(image.image_id).copy()
        descriptors = database.read_descriptors(image.image_id).to_float().data.copy()
        descriptors /= np.maximum(
            np.linalg.norm(descriptors, axis=1, keepdims=True),
            np.finfo(np.float32).eps,
        )
        scales, orientations = self._scales_and_orientations(keypoints)
        mask = cv2.imread(
            str(self.masks_directory / image.name),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise RuntimeError(f"Could not read mapping mask for {image.name}")
        mask = mask > 0
        return {
            "keypoints": keypoints[:, :2].astype(np.float32),
            "descriptors": descriptors.astype(np.float32),
            "scores": np.ones(len(keypoints), dtype=np.float32),
            "scales": scales,
            "oris": orientations,
            "selection_bounds": self._selection_bounds(mask),
            "selection_contour": self._selection_contour(mask),
        }

    @staticmethod
    def _scales_and_orientations(keypoints):
        if keypoints.shape[1] >= 6:
            affine = keypoints[:, 2:6].reshape(-1, 2, 2)
            scales = np.sqrt(np.abs(np.linalg.det(affine)))
            orientations = np.arctan2(affine[:, 1, 0], affine[:, 0, 0])
        elif keypoints.shape[1] >= 4:
            scales = keypoints[:, 2]
            orientations = keypoints[:, 3]
        else:
            scales = np.ones(len(keypoints), dtype=np.float32)
            orientations = np.zeros(len(keypoints), dtype=np.float32)
        return scales.astype(np.float32), orientations.astype(np.float32)

    @staticmethod
    def _selection_bounds(mask):
        mask_y, mask_x = np.nonzero(mask)
        if not len(mask_x):
            return np.array([0, 0, mask.shape[1], mask.shape[0]], dtype=np.int32)
        return np.array(
            [mask_x.min(), mask_y.min(), mask_x.max() + 1, mask_y.max() + 1],
            dtype=np.int32,
        )

    @staticmethod
    def _selection_contour(mask):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return np.empty((0, 1, 2), dtype=np.int32)
        contour = max(contours, key=cv2.contourArea)
        return cv2.approxPolyDP(contour, 1.0, True)
