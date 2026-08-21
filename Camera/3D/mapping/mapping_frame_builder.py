import cv2
import numpy as np

from mapping.colmap_database import ColmapMappingDatabase
from mapping.colmap_matching import ColmapIncrementalMatcher
from mapping.mapping_data import MappingFrameCollection, MappingImage


class MappingFrameBuilder:
    """Collect fixed-interval frames directly into one COLMAP database."""

    MINIMUM_PAIR_INLIERS = ColmapIncrementalMatcher.MINIMUM_PAIR_INLIERS

    def __init__(
        self,
        camera_matrix,
        distortion,
        skin_mask_provider,
        start_frame,
        end_frame,
        keyframe_interval,
        maximum_features,
        sequential_overlap,
        matcher_type,
        loop_detection,
        loop_detection_period,
        vocabulary_tree_path,
        imu_gravity_provider=None,
    ):
        if keyframe_interval <= 0:
            raise ValueError("keyframe_interval must be positive")
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.skin_mask_provider = skin_mask_provider
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.keyframe_interval = keyframe_interval
        self.imu_gravity_provider = imu_gravity_provider
        self.colmap = ColmapIncrementalMatcher(
            maximum_features=maximum_features,
            sequential_overlap=sequential_overlap,
            matcher_type=matcher_type,
            loop_detection=loop_detection,
            loop_detection_period=loop_detection_period,
            vocabulary_tree_path=vocabulary_tree_path,
        )

    @property
    def matcher_type(self):
        return self.colmap.matcher_type

    def build(
        self,
        video_path,
        images_directory,
        masks_directory,
        database_path,
    ):
        self.colmap.validate()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open mapping video: {video_path}")

        database = ColmapMappingDatabase(
            database_path=database_path,
            camera_matrix=self.camera_matrix,
            distortion=self.distortion,
            image_size=(
                round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
            imu_gravity_provider=self.imu_gravity_provider,
        )
        database.initialize()
        images = []

        try:
            self._skip_to_mapping_start(capture)
            for frame_index in range(self.start_frame, self.end_frame + 1):
                success, frame = capture.read()
                if not success:
                    break
                if not self._is_keyframe(frame_index):
                    continue

                image_name = f"frame_{frame_index:06d}.png"
                self._save_png(images_directory / image_name, frame)
                self._save_png(
                    masks_directory / image_name,
                    self._mapping_mask(frame).astype(np.uint8) * 255,
                )

                timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                image_id, imu_status = database.add_image(
                    image_name,
                    len(images) + 1,
                    timestamp_s,
                )
                self.colmap.extract(
                    database_path,
                    images_directory,
                    masks_directory,
                    image_name,
                    database.camera_id,
                )
                self.colmap.match(database_path)
                images.append(
                    MappingImage(
                        frame_index=frame_index,
                        name=image_name,
                        database_image_id=image_id,
                        timestamp_s=timestamp_s,
                    )
                )
                print(
                    f"Mapping keyframe: frame {frame_index}/"
                    f"{self.end_frame}{imu_status}"
                )
        finally:
            capture.release()

        return MappingFrameCollection(
            images=images,
            imu_gravity_summary=database.summary(),
        )

    def _is_keyframe(self, frame_index):
        return (frame_index - self.start_frame) % self.keyframe_interval == 0

    def _mapping_mask(self, frame):
        height, width = frame.shape[:2]
        roi_top = round(
            height
            * (1.0 - self.skin_mask_provider.feature_roi_bottom_fraction)
        )
        roi_mask = np.zeros((height, width), dtype=bool)
        roi_mask[roi_top:] = True
        skin_mask = self.skin_mask_provider.adaptive_skin_mask(frame, roi_top)
        if skin_mask is None:
            skin_mask = np.ones((height, width), dtype=bool)
        return roi_mask & skin_mask.astype(bool)

    def _skip_to_mapping_start(self, capture):
        for _ in range(self.start_frame):
            success, _ = capture.read()
            if not success:
                break

    @staticmethod
    def _save_png(path, image):
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save mapping file: {path}")
