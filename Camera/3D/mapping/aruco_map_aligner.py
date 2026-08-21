import numpy as np

from mapping.mapping_data import ArucoAlignment, MappingFrameCollection


class ArucoMapAligner:
    """Estimate the metric scale of a COLMAP/GLOMAP reconstruction."""

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
        sfm_centers = np.asarray(
            [image.projection_center() for image in alignment_images]
        )
        aruco_centers = np.asarray(
            [
                self.camera_center(*aruco_poses[image.name][:2])
                for image in alignment_images
            ]
        )
        scale, rmse_mm = self._estimate_scale(sfm_centers, aruco_centers)

        return ArucoAlignment(
            scale=float(scale),
            rmse_mm=rmse_mm,
            candidate_frame_count=len(registered_images),
            aligned_frame_count=len(alignment_images),
            reprojection_rms_threshold_px=rms_threshold,
        )

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
        if not np.any(valid_pairs):
            raise RuntimeError(
                "ArUco alignment frames do not contain enough camera translation"
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
