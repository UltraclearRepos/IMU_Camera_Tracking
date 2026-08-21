# Camera 3D

The main entrypoint remains `camera_tracking.py`. Supporting modules are grouped
by responsibility:

- `mapping/` - fixed-interval keyframe collection, incremental COLMAP SIFT
  extraction and sequential/loop matching, one-shot GLOMAP reconstruction,
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
