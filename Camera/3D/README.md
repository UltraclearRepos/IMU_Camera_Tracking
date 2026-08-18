# Camera 3D

The main entrypoint remains `camera_tracking.py`. Supporting modules are grouped
by responsibility:

- `mapping/` - feature extraction, matching, ArUco reference, and map building.
- `tracking/` - frozen-map tracking and the optional IMU gravity prior.
- `geometry/` - coordinate-frame conversions.
- `evaluation/` - mapping evaluation and pipeline diagnostics.
- `visualization/` - tracking and top-view renderers.
- `scripts/` - batch, Jenkins, and configuration-generation entrypoints.

Configuration and output directories remain at this level so existing result
paths do not change.
