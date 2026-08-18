import pycolmap


class SfmReconstructor:
    """Run COLMAP or GLOMAP on an already prepared image-pair database."""

    def __init__(
        self,
        reconstruction_method,
        minimum_pair_inliers,
        use_gravity_prior,
    ):
        self.reconstruction_method = reconstruction_method
        self.minimum_pair_inliers = minimum_pair_inliers
        self.use_gravity_prior = use_gravity_prior
        if use_gravity_prior and reconstruction_method != "global":
            raise ValueError(
                "IMU gravity priors are supported only by global mapping"
            )

    def reconstruct(self, database_path, images_directory, sparse_directory):
        mapping_function, options = self._mapping_configuration()
        reconstructions = mapping_function(
            database_path=database_path,
            image_path=images_directory,
            output_path=sparse_directory,
            options=options,
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

    def _mapping_configuration(self):
        if self.reconstruction_method == "global":
            return pycolmap.global_mapping, self._global_options()
        return pycolmap.incremental_mapping, self._incremental_options()

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

    def _incremental_options(self):
        options = pycolmap.IncrementalPipelineOptions(
            multiple_models=False,
            structure_less_registration_fallback=False,
            ba_refine_focal_length=False,
            ba_refine_principal_point=False,
            ba_refine_extra_params=False,
        )
        options.ba_use_gpu = True
        options.mapper.init_min_num_inliers = self.minimum_pair_inliers
        options.mapper.abs_pose_min_num_inliers = self.minimum_pair_inliers
        options.random_seed = 0
        options.mapper.random_seed = 0
        options.triangulation.random_seed = 0
        return options
