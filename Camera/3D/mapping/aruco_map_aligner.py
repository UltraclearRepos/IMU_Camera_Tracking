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
    aruco_pose: ArucoPoseResult
    sfm_image: pycolmap.Image


class ArucoMapAligner:
    """Estimate the metric scale of a COLMAP/GLOMAP reconstruction."""

    MINIMUM_ALIGNMENT_FRAMES = 3
    MINIMUM_ARUCO_PAIR_DISPLACEMENT_MM = 10.0

    def align(self, reconstruction, frame_collection: MappingFrameCollection):
        registered_images_by_name = {
            image.name: image
            for image in reconstruction.images.values()
        }
        candidate_frames = [
            AlignmentFrame(
                name=image.name,
                aruco_pose=image.aruco_pose,
                sfm_image=registered_images_by_name[image.name],
            )
            for image in frame_collection.images
            if (
                image.aruco_pose is not None
                and image.name in registered_images_by_name
            )
        ]
        if len(candidate_frames) < self.MINIMUM_ALIGNMENT_FRAMES:
            raise RuntimeError(
                "ArUco must be visible in at least "
                f"{self.MINIMUM_ALIGNMENT_FRAMES} registered mapping frames"
            )

        alignment_frames, rms_threshold = self._select_alignment_frames(
            candidate_frames
        )
        sfm_centers = np.asarray(
            [frame.sfm_image.projection_center() for frame in alignment_frames]
        )
        aruco_centers = np.asarray(
            [
                self.camera_center(
                    frame.aruco_pose.rotation,
                    frame.aruco_pose.translation,
                )
                for frame in alignment_frames
            ]
        )
        scale, rmse_mm = self._estimate_scale(sfm_centers, aruco_centers)

        return ArucoAlignment(
            scale=float(scale),
            rmse_mm=rmse_mm,
            candidate_frame_count=len(candidate_frames),
            aligned_frame_count=len(alignment_frames),
            reprojection_rms_threshold_px=rms_threshold,
            aligned_image_names=tuple(
                frame.name for frame in alignment_frames
            ),
        )

    @staticmethod
    def camera_center(world_to_camera_rotation, world_to_camera_translation):
        return (
            -world_to_camera_rotation.T
            @ world_to_camera_translation.reshape(3)
        )

    def _select_alignment_frames(self, candidate_frames):
        keep_count = max(
            self.MINIMUM_ALIGNMENT_FRAMES,
            int(np.ceil(0.5 * len(candidate_frames))),
        )

        frames_by_reprojection_quality = sorted(
            candidate_frames,
            key=lambda frame: (
                frame.aruco_pose.reprojection_rms_px,
                frame.name,
            ),
        )
        alignment_frames = frames_by_reprojection_quality[:keep_count]
        rms_threshold = max(
            frame.aruco_pose.reprojection_rms_px
            for frame in alignment_frames
        )
        alignment_frames.sort(key=lambda frame: frame.name)
        return alignment_frames, float(rms_threshold)

    @staticmethod
    def _estimate_scale(sfm_centers, aruco_centers):
        pair_indices = np.triu_indices(len(sfm_centers), k=1)
        sfm_distances = np.linalg.norm(
            sfm_centers[pair_indices[0]] - sfm_centers[pair_indices[1]],
            axis=1,
        )
        aruco_distances = np.linalg.norm(
            aruco_centers[pair_indices[0]] - aruco_centers[pair_indices[1]],
            axis=1,
        )
        valid_pairs = sfm_distances > np.finfo(float).eps
        valid_pairs &= (
            aruco_distances
            >= ArucoMapAligner.MINIMUM_ARUCO_PAIR_DISPLACEMENT_MM
        )
        if not np.any(valid_pairs):
            raise RuntimeError(
                "ArUco alignment frames do not contain a camera-position "
                "pair separated by at least "
                f"{ArucoMapAligner.MINIMUM_ARUCO_PAIR_DISPLACEMENT_MM:g} mm"
            )
        distance_ratios = (
            aruco_distances[valid_pairs] / sfm_distances[valid_pairs]
        )
        scale = float(np.median(distance_ratios))
        residuals = (
            aruco_distances[valid_pairs]
            - scale * sfm_distances[valid_pairs]
        )
        rmse_mm = float(np.sqrt(np.mean(residuals**2)))
        return scale, rmse_mm
