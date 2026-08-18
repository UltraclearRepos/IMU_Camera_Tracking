# Camera 3D

The main entrypoint remains `camera_tracking.py`. Supporting modules are grouped
by responsibility:

- `mapping/` - feature extraction, frame-graph construction, SfM, ArUco
  alignment, frozen-map assembly, and mapping diagnostics. `SkinMapBuilder`
  only coordinates these stages; `MappingFrameBuilder` owns the frame and
  image-pair selection logic intended for further development.
- `tracking/` - frozen-map tracking and the optional IMU gravity prior.
- `geometry/` - coordinate-frame conversions.
- `evaluation/` - final mapping evaluation against ground truth.
- `visualization/` - tracking and top-view renderers.
- `scripts/` - batch, Jenkins, and configuration-generation entrypoints.

Configuration and output directories remain at this level so existing result
paths do not change.
