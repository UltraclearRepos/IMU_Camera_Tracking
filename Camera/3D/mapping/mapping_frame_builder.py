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
            loop_detection=loop_detection,
            loop_detection_period=loop_detection_period,
            vocabulary_tree_path=vocabulary_tree_path,
        )

    @property
    def matcher_type(self):
        return self.colmap.MATCHER_TYPE

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
        self.colmap.bind_reader(database.camera_id, masks_directory)
        images = []
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            capture.release()
            raise RuntimeError("Mapping video does not report a valid FPS")

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_index = round(timestamp_s * fps)
                if frame_index < self.start_frame:
                    continue
                if frame_index > self.end_frame:
                    break
                if not self._is_keyframe(frame_index):
                    continue

                image_name = f"frame_{frame_index:06d}.png"
                self._save_png(images_directory / image_name, frame)
                self._save_png(
                    masks_directory / image_name,
                    self._mapping_mask(frame).astype(np.uint8) * 255,
                )

                image_id, imu_status = database.add_image(
                    image_name,
                    len(images) + 1,
                    timestamp_s,
                )
                self.colmap.extract(
                    database_path,
                    images_directory,
                    image_name,
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

    @staticmethod
    def _save_png(path, image):
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save mapping file: {path}")
