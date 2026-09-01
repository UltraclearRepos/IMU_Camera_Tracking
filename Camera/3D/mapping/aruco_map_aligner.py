from dataclasses import dataclass

import numpy as np
import pycolmap

from mapping.mapping_data import (
    ArucoAlignment,
    ArucoPoseResult,
    MappingFrameCollection,
)


@dataclass(frozen=True)
class AlignmentFrame:
    name: str
    frame_number: int
    aruco_pose: ArucoPoseResult
    sfm_image: pycolmap.Image
    aruco_center_mm: np.ndarray
    sfm_center: np.ndarray


@dataclass(frozen=True)
class AlignmentPair:
    first: AlignmentFrame
    second: AlignmentFrame
    aruco_distance_mm: float
    sfm_distance: float


class ArucoMapAligner:
    """Estimate the metric scale of a COLMAP/GLOMAP reconstruction."""

    MINIMUM_ALIGNMENT_FRAMES = 3
    MAXIMUM_ARUCO_REPROJECTION_RMS_PX = 1.0
    MINIMUM_ARUCO_PAIR_DISPLACEMENT_MM = 3.0
    MAXIMUM_ALIGNMENT_PAIR_FRAME_GAP = 200

    def align(self, reconstruction, frame_collection: MappingFrameCollection):
        registered_images_by_name = {
            image.name: image
            for image in reconstruction.images.values()
        }
        candidate_frames = []
        for image in frame_collection.images:
            if (
                image.aruco_pose is None
                or image.name not in registered_images_by_name
            ):
                continue

            sfm_image = registered_images_by_name[image.name]
            candidate_frames.append(
                AlignmentFrame(
                    name=image.name,
                    frame_number=image.frame_index,
                    aruco_pose=image.aruco_pose,
                    sfm_image=sfm_image,
                    aruco_center_mm=self.camera_center(
                        image.aruco_pose.rotation,
                        image.aruco_pose.translation,
                    ),
                    sfm_center=sfm_image.projection_center(),
                )
            )

        if len(candidate_frames) < self.MINIMUM_ALIGNMENT_FRAMES:
            raise RuntimeError(
                "ArUco must be visible in at least "
                f"{self.MINIMUM_ALIGNMENT_FRAMES} registered mapping frames"
            )

        alignment_frames, rms_threshold = self._select_alignment_frames(
            candidate_frames
        )
        alignment_pairs = self._select_alignment_pairs(alignment_frames)
        scale, rmse_mm = self._estimate_scale(alignment_pairs)
        aligned_image_names = sorted(
            {
                frame.name
                for pair in alignment_pairs
                for frame in (pair.first, pair.second)
            }
        )

        return ArucoAlignment(
            scale=float(scale),
            rmse_mm=rmse_mm,
            candidate_frame_count=len(candidate_frames),
            aligned_frame_count=len(aligned_image_names),
            reprojection_rms_threshold_px=rms_threshold,
            aligned_image_names=tuple(aligned_image_names),
            aligned_image_pairs=tuple(
                (pair.first.name, pair.second.name)
                for pair in alignment_pairs
            ),
            aligned_pair_distances_mm=tuple(
                pair.aruco_distance_mm for pair in alignment_pairs
            ),
            aligned_pair_sfm_distances=tuple(
                pair.sfm_distance for pair in alignment_pairs
            ),
        )

    @staticmethod
    def camera_center(world_to_camera_rotation, world_to_camera_translation):
        return (
            -world_to_camera_rotation.T
            @ world_to_camera_translation.reshape(3)
        )

    def _select_alignment_frames(self, candidate_frames):
        alignment_frames = [
            frame
            for frame in candidate_frames
            if frame.aruco_pose.reprojection_rms_px
            <= self.MAXIMUM_ARUCO_REPROJECTION_RMS_PX
        ]
        if len(alignment_frames) < self.MINIMUM_ALIGNMENT_FRAMES:
            raise RuntimeError(
                "ArUco reprojection RMS must not exceed "
                f"{self.MAXIMUM_ARUCO_REPROJECTION_RMS_PX:g} px in at least "
                f"{self.MINIMUM_ALIGNMENT_FRAMES} registered mapping frames"
            )

        alignment_frames.sort(key=lambda frame: frame.name)
        return alignment_frames, self.MAXIMUM_ARUCO_REPROJECTION_RMS_PX

    def _select_alignment_pairs(self, alignment_frames):
        candidate_pairs = []
        for first_index, first in enumerate(alignment_frames):
            for second in alignment_frames[first_index + 1 :]:

                if (
                    first.aruco_pose.marker_id
                    != second.aruco_pose.marker_id
                ):
                    continue

                if (abs(first.frame_number - second.frame_number) > self.MAXIMUM_ALIGNMENT_PAIR_FRAME_GAP):
                    continue

                aruco_distance_mm = float(
                    np.linalg.norm(
                        first.aruco_center_mm - second.aruco_center_mm
                    )
                )

                if (aruco_distance_mm < self.MINIMUM_ARUCO_PAIR_DISPLACEMENT_MM):
                    continue

                sfm_distance = float(
                    np.linalg.norm(first.sfm_center - second.sfm_center)
                )
                if sfm_distance <= np.finfo(float).eps:
                    continue

                candidate_pairs.append(
                    AlignmentPair(
                        first=first,
                        second=second,
                        aruco_distance_mm=aruco_distance_mm,
                        sfm_distance=sfm_distance,
                    )
                )

        candidate_pairs.sort(
            key=lambda pair: (
                -pair.aruco_distance_mm,
                pair.first.name,
                pair.second.name,
            )
        )

        used_image_names = set()
        alignment_pairs = []
        for pair in candidate_pairs:
            if (pair.first.name in used_image_names or pair.second.name in used_image_names):
                continue

            alignment_pairs.append(pair)
            used_image_names.add(pair.first.name)
            used_image_names.add(pair.second.name)

        if not alignment_pairs:
            raise RuntimeError(
                "ArUco alignment frames do not contain a camera-position "
                "pair separated by at least "
                f"{self.MINIMUM_ARUCO_PAIR_DISPLACEMENT_MM:g} mm"
            )
        return alignment_pairs

    @staticmethod
    def _estimate_scale(alignment_pairs):
        sfm_distances = np.asarray(
            [pair.sfm_distance for pair in alignment_pairs]
        )
        aruco_distances = np.asarray(
            [pair.aruco_distance_mm for pair in alignment_pairs]
        )
        scale = float(
            np.dot(sfm_distances, aruco_distances)
            / np.dot(sfm_distances, sfm_distances)
        )
        residuals = aruco_distances - scale * sfm_distances
        rmse_mm = float(np.sqrt(np.mean(residuals**2)))
        return scale, rmse_mm
