# Camera 3D

The main entrypoint remains `camera_tracking.py`. Supporting modules are grouped
by responsibility:

- `mapping/` - fixed-interval keyframe collection, incremental COLMAP RootSIFT
  extraction and COLMAP SIFT-LightGlue sequential/loop matching, one-shot
  GLOMAP reconstruction,
  ArUco scale alignment, frozen-map assembly, and mapping diagnostics.
  Mapping images remain full-size; `ROI_TOP AND skin_mask` is passed to COLMAP
  as a same-size PNG mask.
- `tracking/` - frozen-map tracking and the optional IMU gravity prior.
- `geometry/` - coordinate-frame conversions.
- `evaluation/` - final mapping evaluation against ground truth.
- `visualization/` - tracking and top-view renderers.
- `scripts/` - batch, Jenkins, and configuration-generation entrypoints.

Configuration and output directories remain at this level so existing result
paths do not change.

COLMAP loop detection requires a vocabulary-tree `.bin` file. By default the
entrypoint looks for `vocab_tree_flickr100K_words32K.bin` next to
`camera_tracking.py`. Set `COLMAP_VOCAB_TREE_PATH` to use a different location.

Batch JSON files expose the mapping controls directly:

- `keyframe_interval`
- `colmap_max_num_features`
- `colmap_sequential_overlap`
- `colmap_loop_detection`
- `colmap_loop_detection_period`, measured in mapping keyframes

When loop detection is disabled, the vocabulary-tree file is not required.
The matcher is fixed to `SIFT_LIGHTGLUE`. The vocabulary-tree location is
fixed by `COLMAP_VOCAB_TREE_PATH`, with the adjacent default file as fallback.

`database.db` is the only feature source of truth. During collection,
`MappingFrameBuilder` stores only frame identity and timestamp. RootSIFT
keypoints/descriptors are loaded once, after GLOMAP, and only for registered
images while the frozen global map is assembled. ArUco detection likewise runs
after reconstruction on registered mapping images.
