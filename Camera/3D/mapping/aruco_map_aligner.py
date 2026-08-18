import numpy as np
from scipy.spatial.transform import Rotation

from mapping.mapping_data import ArucoAlignment, MappingFrameCollection


class ArucoMapAligner:
    """Estimate and apply the SfM-to-ArUco metric similarity transform."""

    MINIMUM_ALIGNMENT_FRAMES = 3

    def align(self, reconstruction, frame_collection: MappingFrameCollection):
        aruco_poses = {
            image.name: image.aruco_pose
            for image in frame_collection.images
            if image.aruco_pose is not None
        }
        registered_images = sorted(
            (
                image
                for image in reconstruction.images.values()
                if image.name in aruco_poses
            ),
            key=lambda image: image.name,
        )
        if len(registered_images) < self.MINIMUM_ALIGNMENT_FRAMES:
            raise RuntimeError(
                "ArUco must be visible in at least "
                f"{self.MINIMUM_ALIGNMENT_FRAMES} registered mapping frames"
            )

        alignment_images, rms_threshold = self._select_alignment_images(
            registered_images,
            aruco_poses,
        )
        rotation, sfm_centers, aruco_centers = self._estimate_rotation(
            alignment_images,
            aruco_poses,
        )
        scale, translation, aligned_centers = self._estimate_scale_translation(
            rotation,
            sfm_centers,
            aruco_centers,
        )
        residuals = np.linalg.norm(aligned_centers - aruco_centers, axis=1)
        rmse_mm = float(np.sqrt(np.mean(residuals**2)))

        return ArucoAlignment(
            scale=float(scale),
            rotation=rotation,
            translation=translation,
            rmse_mm=rmse_mm,
            candidate_frame_count=len(registered_images),
            aligned_frame_count=len(alignment_images),
            reprojection_rms_threshold_px=rms_threshold,
            reference_image_name=alignment_images[0].name,
            center_residuals_by_image={
                image.name: float(residual)
                for image, residual in zip(alignment_images, residuals)
            },
        )

    def transform_points(self, points, alignment: ArucoAlignment):
        return (
            alignment.scale * (alignment.rotation @ points.T).T
            + alignment.translation
        )

    def transform_pose(self, image, alignment: ArucoAlignment):
        sfm_pose = image.cam_from_world()
        sfm_to_camera_rotation = sfm_pose.rotation.matrix()
        sfm_to_camera_translation = np.asarray(sfm_pose.translation)
        map_to_camera_rotation = (
            sfm_to_camera_rotation @ alignment.rotation.T
        )
        map_to_camera_translation = (
            alignment.scale * sfm_to_camera_translation
            - map_to_camera_rotation @ alignment.translation
        )
        return map_to_camera_rotation, map_to_camera_translation

    @staticmethod
    def camera_center(world_to_camera_rotation, world_to_camera_translation):
        return (
            -world_to_camera_rotation.T
            @ world_to_camera_translation.reshape(3)
        )

    def _select_alignment_images(self, registered_images, aruco_poses):
        reprojection_rms = np.asarray(
            [
                aruco_poses[image.name][2]["reprojection_rms_px"]
                for image in registered_images
            ],
            dtype=float,
        )
        keep_count = max(
            self.MINIMUM_ALIGNMENT_FRAMES,
            int(np.ceil(0.5 * len(registered_images))),
        )
        best_indices = np.argsort(reprojection_rms)[:keep_count]
        alignment_images = [registered_images[index] for index in best_indices]
        alignment_images.sort(key=lambda image: image.name)
        return alignment_images, float(np.max(reprojection_rms[best_indices]))

    def _estimate_rotation(self, alignment_images, aruco_poses):
        rotation_candidates = []
        sfm_centers = []
        aruco_centers = []
        for image in alignment_images:
            aruco_to_camera_rotation, aruco_to_camera_translation, _ = (
                aruco_poses[image.name]
            )
            sfm_to_camera_rotation = image.cam_from_world().rotation.matrix()
            rotation_candidates.append(
                aruco_to_camera_rotation.T @ sfm_to_camera_rotation
            )
            sfm_centers.append(image.projection_center())
            aruco_centers.append(
                self.camera_center(
                    aruco_to_camera_rotation,
                    aruco_to_camera_translation,
                )
            )

        rotation = Rotation.from_matrix(rotation_candidates).mean().as_matrix()
        return rotation, np.asarray(sfm_centers), np.asarray(aruco_centers)

    @staticmethod
    def _estimate_scale_translation(rotation, sfm_centers, aruco_centers):
        rotated_centers = (rotation @ sfm_centers.T).T
        rotated_mean = np.mean(rotated_centers, axis=0)
        aruco_mean = np.mean(aruco_centers, axis=0)
        rotated_centered = rotated_centers - rotated_mean
        aruco_centered = aruco_centers - aruco_mean
        scale_denominator = np.sum(rotated_centered**2)
        if scale_denominator <= np.finfo(float).eps:
            raise RuntimeError(
                "ArUco alignment frames do not contain enough camera translation"
            )
        scale = (
            np.sum(rotated_centered * aruco_centered) / scale_denominator
        )
        translation = aruco_mean - scale * rotated_mean
        aligned_centers = scale * rotated_centers + translation
        return scale, translation, aligned_centers
