import pycolmap


class SfmReconstructor:
    """Run GLOMAP once on the completed COLMAP feature database."""

    def __init__(
        self,
        minimum_pair_inliers,
        use_gravity_prior,
    ):
        self.minimum_pair_inliers = minimum_pair_inliers
        self.use_gravity_prior = use_gravity_prior

    def reconstruct(self, database_path, images_directory, sparse_directory):
        reconstructions = pycolmap.global_mapping(
            database_path=database_path,
            image_path=images_directory,
            output_path=sparse_directory,
            options=self._global_options(),
        )
        if not reconstructions:
            raise RuntimeError(
                "COLMAP could not initialize the 3D map. The mapping pass "
                "needs camera translation and visible feature parallax."
            )
        return max(
            reconstructions.values(),
            key=lambda reconstruction: reconstruction.num_reg_images(),
        )

    def _global_options(self):
        options = pycolmap.GlobalPipelineOptions(
            min_num_matches=self.minimum_pair_inliers,
            random_seed=0,
        )
        options.mapper.random_seed = 0
        options.mapper.rotation_averaging.random_seed = 0
        options.mapper.rotation_averaging.use_gravity = self.use_gravity_prior
        options.mapper.global_positioning.random_seed = 0
        options.mapper.bundle_adjustment.refine_focal_length = False
        options.mapper.bundle_adjustment.refine_principal_point = False
        options.mapper.bundle_adjustment.refine_extra_params = False
        options.mapper.global_positioning.use_gpu = True
        options.mapper.global_positioning.min_num_images_gpu_solver = 3
        options.mapper.bundle_adjustment.ceres.use_gpu = True
        return options
