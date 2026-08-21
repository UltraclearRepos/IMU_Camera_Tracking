import pycolmap


class ColmapIncrementalMatcher:
    """Configure incremental RootSIFT extraction and sequential matching."""

    MINIMUM_PAIR_INLIERS = 20
    MATCHER_TYPE = pycolmap.FeatureMatcherType.SIFT_LIGHTGLUE

    def __init__(
        self,
        maximum_features,
        sequential_overlap,
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
        self.loop_detection = bool(loop_detection)
        self.loop_detection_period = loop_detection_period
        self.vocabulary_tree_path = vocabulary_tree_path
        self.reader_options = None

        self.extraction_options = pycolmap.FeatureExtractionOptions()
        self.extraction_options.max_image_size = -1
        self.extraction_options.sift.max_num_features = maximum_features
        self.extraction_options.sift.max_num_orientations = 1
        self.extraction_options.sift.normalization = pycolmap.Normalization.L1_ROOT

        self.pairing_options = pycolmap.SequentialPairingOptions(
            overlap=sequential_overlap,
            quadratic_overlap=False,
            loop_detection=self.loop_detection,
            loop_detection_period=loop_detection_period,
        )

        if self.loop_detection and vocabulary_tree_path is not None:
            self.pairing_options.vocab_tree_path = vocabulary_tree_path
        self.matching_options = pycolmap.FeatureMatchingOptions(
            type=self.MATCHER_TYPE,
            skip_geometric_verification=False,
        )
        self.verification_options = pycolmap.TwoViewGeometryOptions(
            min_num_inliers=self.MINIMUM_PAIR_INLIERS,
        )
        self.verification_options.ransac.random_seed = 0

    def validate(self):
        if (
            self.MATCHER_TYPE == pycolmap.FeatureMatcherType.SIFT_LIGHTGLUE
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

    def bind_reader(self, camera_id, masks_directory):
        self.reader_options = pycolmap.ImageReaderOptions(
            existing_camera_id=camera_id,
            mask_path=masks_directory,
        )

    def extract(
        self,
        database_path,
        images_directory,
        image_name,
    ):
        if self.reader_options is None:
            raise RuntimeError("COLMAP image reader has not been bound")
        pycolmap.extract_features(
            database_path=database_path,
            image_path=images_directory,
            image_names=[image_name],
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=self.reader_options,
            extraction_options=self.extraction_options,
            device=pycolmap.Device.auto,
        )

    def match(self, database_path):
        pycolmap.match_sequential(
            database_path=database_path,
            matching_options=self.matching_options,
            pairing_options=self.pairing_options,
            verification_options=self.verification_options,
            device=pycolmap.Device.auto,
        )
