import pycolmap


class ColmapIncrementalMatcher:
    """Configure incremental RootSIFT extraction and sequential matching."""

    MINIMUM_PAIR_INLIERS = 20

    def __init__(
        self,
        maximum_features,
        sequential_overlap,
        matcher_type,
        loop_detection,
        loop_detection_period,
        vocabulary_tree_path,
    ):
        if maximum_features <= 0:
            raise ValueError("maximum_features must be positive")
        if sequential_overlap <= 0:
            raise ValueError("sequential_overlap must be positive")
        if loop_detection_period <= 0:
            raise ValueError("loop_detection_period must be positive")
        self.maximum_features = maximum_features
        self.sequential_overlap = sequential_overlap
        self.matcher_type = self.resolve_matcher_type(matcher_type)
        self.loop_detection = bool(loop_detection)
        self.loop_detection_period = loop_detection_period
        self.vocabulary_tree_path = vocabulary_tree_path

    def validate(self):
        if (
            self.matcher_type == pycolmap.FeatureMatcherType.SIFT_LIGHTGLUE
            and not pycolmap.has_cuda
        ):
            raise RuntimeError(
                "COLMAP SIFT_LIGHTGLUE requires a CUDA-enabled PyCOLMAP build"
            )
        if self.loop_detection and (
            self.vocabulary_tree_path is None
            or not self.vocabulary_tree_path.is_file()
        ):
            raise FileNotFoundError(
                "COLMAP loop detection requires an existing vocabulary-tree "
                f"file, got: {self.vocabulary_tree_path}"
            )

    def extract(
        self,
        database_path,
        images_directory,
        masks_directory,
        image_name,
        camera_id,
    ):
        reader_options = pycolmap.ImageReaderOptions(
            existing_camera_id=camera_id,
            mask_path=masks_directory,
        )
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.max_image_size = -1
        extraction_options.sift.max_num_features = self.maximum_features
        extraction_options.sift.max_num_orientations = 1
        extraction_options.sift.normalization = pycolmap.Normalization.L1_ROOT
        pycolmap.extract_features(
            database_path=database_path,
            image_path=images_directory,
            image_names=[image_name],
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=pycolmap.Device.auto,
        )

    def match(self, database_path):
        pairing_options = pycolmap.SequentialPairingOptions(
            overlap=self.sequential_overlap,
            quadratic_overlap=False,
            loop_detection=self.loop_detection,
            loop_detection_period=self.loop_detection_period,
        )
        if self.loop_detection:
            pairing_options.vocab_tree_path = self.vocabulary_tree_path
        verification_options = pycolmap.TwoViewGeometryOptions(
            min_num_inliers=self.MINIMUM_PAIR_INLIERS,
        )
        verification_options.ransac.random_seed = 0
        pycolmap.match_sequential(
            database_path=database_path,
            matching_options=pycolmap.FeatureMatchingOptions(
                type=self.matcher_type,
                skip_geometric_verification=False,
            ),
            pairing_options=pairing_options,
            verification_options=verification_options,
            device=pycolmap.Device.auto,
        )

    @staticmethod
    def resolve_matcher_type(matcher_type):
        if isinstance(matcher_type, pycolmap.FeatureMatcherType):
            resolved = matcher_type
        else:
            try:
                resolved = pycolmap.FeatureMatcherType.__members__[
                    str(matcher_type).upper()
                ]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported COLMAP matcher: {matcher_type!r}"
                ) from error
        if resolved not in (
            pycolmap.FeatureMatcherType.SIFT_LIGHTGLUE,
            pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE,
        ):
            raise ValueError("Mapping requires a SIFT feature matcher")
        return resolved
