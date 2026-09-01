
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

MAP_PATH = (
    SCRIPT_DIR.parent
    / "jenkins_results"
    / "Cylinder"
    / "keyframe_int=1_recent=5_interval_1_20px_30px_mapF=512_256_new"
    / "sift"
    / "initial_50mm_Arc180-Speed-3_2026-08-20_14.39.08"
    / "map"
    / "global_map.npz"
)

# "3d"  - current perspective view
# "top" - orthographic top-down view along Z axis
VIEW_MODE = "3d"
TOP_PLANE = "xz"  # "xy", "xz" albo "yz"

SHOW_CANDIDATES = False
SHOW_CAMERAS = True

POINT_SIZE = 1.0

MAXIMUM_CAMERA_FRUSTUMS = 30
CAMERA_FRUSTUM_SIZE_MM = 1.0

REMOVE_STRONG_CAMERA_OUTLIERS = True
CAMERA_OUTLIER_IQR_FACTOR = 8.0

# Set to a Path to additionally save a standalone, shareable HTML file.
SAVE_HTML_PATH = None


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_map(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name]
            for name in archive.files
        }

    if "positions" not in arrays:
        raise ValueError(
            f"{path} does not contain a 'positions' array"
        )

    positions = np.asarray(
        arrays["positions"],
        dtype=np.float64,
    )

    if (
        positions.ndim != 2
        or positions.shape[1] != 3
    ):
        raise ValueError(
            f"Expected positions with shape (N, 3), "
            f"got {positions.shape}"
        )

    if not len(positions):
        raise ValueError(
            f"{path} contains no selected map landmarks"
        )

    if not np.isfinite(positions).all():
        raise ValueError(
            f"{path} contains non-finite landmark positions"
        )

    return arrays


def scalar_text(value, default="unknown"):
    if value is None:
        return default

    scalar = (
        np.asarray(value)
        .reshape(())
        .item()
    )

    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")

    return str(scalar)


# -----------------------------------------------------------------------------
# Filtering
# -----------------------------------------------------------------------------


def strong_camera_inlier_mask(points):
    """Reject only extreme per-axis outliers from the camera trajectory."""

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if not len(points):
        return np.empty(
            0,
            dtype=bool,
        )

    finite = np.isfinite(
        points
    ).all(axis=1)

    finite_points = points[finite]

    if (
        not REMOVE_STRONG_CAMERA_OUTLIERS
        or len(finite_points) < 8
    ):
        return finite

    lower_quartile, upper_quartile = np.percentile(
        finite_points,
        (25.0, 75.0),
        axis=0,
    )

    iqr = (
        upper_quartile
        - lower_quartile
    )

    useful_axis = (
        iqr
        > np.finfo(np.float64).eps
    )

    lower_bound = (
        lower_quartile
        - CAMERA_OUTLIER_IQR_FACTOR * iqr
    )

    upper_bound = (
        upper_quartile
        + CAMERA_OUTLIER_IQR_FACTOR * iqr
    )

    inlier = finite.copy()

    if np.any(useful_axis):
        inlier[finite] = np.all(
            (
                finite_points[:, useful_axis]
                >= lower_bound[useful_axis]
            )
            & (
                finite_points[:, useful_axis]
                <= upper_bound[useful_axis]
            ),
            axis=1,
        )

    return inlier


# -----------------------------------------------------------------------------
# Landmarks
# -----------------------------------------------------------------------------


def point_trace(
    points,
    name,
    color,
    size,
    visible=True,
    opacity=1.0,
):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        visible=(
            True
            if visible
            else "legendonly"
        ),
        customdata=np.arange(
            len(points)
        ),
        marker={
            "size": size,
            "color": color,
            "opacity": opacity,
            "colorscale": "Turbo",
            "colorbar": (
                {
                    "title": {
                        "text": "Z [mm]",
                    },
                    "thickness": 16,
                }
                if not isinstance(color, str)
                else None
            ),
        },
        hovertemplate=(
            "point %{customdata}<br>"
            "X: %{x:.2f} mm<br>"
            "Y: %{y:.2f} mm<br>"
            "Z: %{z:.2f} mm"
            "<extra>%{fullData.name}</extra>"
        ),
    )


# -----------------------------------------------------------------------------
# Cameras
# -----------------------------------------------------------------------------


def camera_frustum_coordinates(
    positions,
    rotations,
    size,
):
    local_corners = np.asarray(
        [
            [-0.65, -0.45, 1.0],
            [0.65, -0.45, 1.0],
            [0.65, 0.45, 1.0],
            [-0.65, 0.45, 1.0],
        ],
        dtype=np.float64,
    ) * size

    x = []
    y = []
    z = []

    def add_segment(start, end):
        x.extend(
            (
                start[0],
                end[0],
                None,
            )
        )
        y.extend(
            (
                start[1],
                end[1],
                None,
            )
        )
        z.extend(
            (
                start[2],
                end[2],
                None,
            )
        )

    for center, rotation in zip(
        positions,
        rotations,
    ):
        corners = (
            rotation
            @ local_corners.T
        ).T + center

        for corner in corners:
            add_segment(
                center,
                corner,
            )

        for index in range(4):
            add_segment(
                corners[index],
                corners[
                    (index + 1) % 4
                ],
            )

    return x, y, z


def add_camera_traces(
    figure,
    positions,
    rotations,
):
    visibility = (
        True
        if SHOW_CAMERAS
        else "legendonly"
    )

    figure.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines+markers",
            name="mapping trajectory",
            visible=visibility,
            line={
                "color": "#ff9800",
                "width": 5,
            },
            marker={
                "color": "#ff9800",
                "size": 2,
            },
            hovertemplate=(
                "camera %{pointNumber}<br>"
                "X: %{x:.2f} mm<br>"
                "Y: %{y:.2f} mm<br>"
                "Z: %{z:.2f} mm"
                "<extra>mapping trajectory</extra>"
            ),
        )
    )

    if rotations.shape != (
        len(positions),
        3,
        3,
    ):
        return

    stride = max(
        1,
        int(
            np.ceil(
                len(positions)
                / MAXIMUM_CAMERA_FRUSTUMS
            )
        ),
    )

    x, y, z = camera_frustum_coordinates(
        positions[::stride],
        rotations[::stride],
        CAMERA_FRUSTUM_SIZE_MM,
    )

    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            name="mapping cameras",
            visible=visibility,
            line={
                "color": "#ffb74d",
                "width": 2,
            },
            hoverinfo="skip",
        )
    )


# -----------------------------------------------------------------------------
# Coordinate axes
# -----------------------------------------------------------------------------


def add_coordinate_axes(
    figure,
    size,
    coordinate_frame,
):
    axes = (
        (
            "X",
            (size, 0, 0),
            "#ef5350",
        ),
        (
            "Y",
            (0, size, 0),
            "#66bb6a",
        ),
        (
            "Z",
            (0, 0, size),
            "#42a5f5",
        ),
    )

    for label, endpoint, color in axes:
        figure.add_trace(
            go.Scatter3d(
                x=(
                    0,
                    endpoint[0],
                ),
                y=(
                    0,
                    endpoint[1],
                ),
                z=(
                    0,
                    endpoint[2],
                ),
                mode="lines+text",
                text=(
                    "",
                    label,
                ),
                textposition="top center",
                name=(
                    f"{coordinate_frame} "
                    f"{label}"
                ),
                line={
                    "color": color,
                    "width": 7,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )


# -----------------------------------------------------------------------------
# View
# -----------------------------------------------------------------------------


def get_scene_camera():
    if VIEW_MODE == "3d":
        return {
            "projection": {
                "type": "perspective",
            },
        }

    if VIEW_MODE != "top":
        raise ValueError(
            f"Unknown VIEW_MODE: {VIEW_MODE}. "
            "Use '3d' or 'top'."
        )

    if TOP_PLANE == "xy":
        # XY:
        # X -> horizontal
        # Y -> vertical
        # looking along -Z
        eye = {
            "x": 0.0,
            "y": 0.0,
            "z": 2.5,
        }
        up = {
            "x": 0.0,
            "y": 1.0,
            "z": 0.0,
        }

    elif TOP_PLANE == "xz":
        # XZ:
        # X -> horizontal
        # Z -> vertical
        # looking along +Y
        eye = {
            "x": 0.0,
            "y": -2.5,
            "z": 0.0,
        }
        up = {
            "x": 0.0,
            "y": 0.0,
            "z": 1.0,
        }

    elif TOP_PLANE == "yz":
        # YZ:
        # Y -> horizontal
        # Z -> vertical
        # looking along -X
        eye = {
            "x": 2.5,
            "y": 0.0,
            "z": 0.0,
        }
        up = {
            "x": 0.0,
            "y": 0.0,
            "z": 1.0,
        }

    else:
        raise ValueError(
            f"Unknown TOP_PLANE: {TOP_PLANE}. "
            "Use 'xy', 'xz' or 'yz'."
        )

    return {
        "eye": eye,
        "center": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        "up": up,
        "projection": {
            "type": "orthographic",
        },
    }


def get_drag_mode():
    if VIEW_MODE == "top":
        # Prevent accidental rotation of the top-down view.
        return "pan"

    return "orbit"


def get_view_description():
    if VIEW_MODE == "top":
        return (
            "Orthographic top view (XY) · "
            "drag: pan · wheel: zoom · "
            "click legend: toggle layer"
        )

    return (
        "Drag: orbit · "
        "Shift+drag: pan · "
        "wheel: zoom · "
        "click legend: toggle layer"
    )


# -----------------------------------------------------------------------------
# Viewer
# -----------------------------------------------------------------------------


def create_viewer(
    arrays,
    map_path,
):
    all_positions = np.asarray(
        arrays["positions"],
        dtype=np.float64,
    )

    all_candidates = np.asarray(
        arrays.get(
            "candidate_positions",
            np.empty((0, 3)),
        ),
        dtype=np.float64,
    )

    all_camera_positions = np.asarray(
        arrays.get(
            "mapping_camera_positions",
            np.empty((0, 3)),
        ),
        dtype=np.float64,
    )

    all_camera_rotations = np.asarray(
        arrays.get(
            "mapping_camera_rotations",
            np.empty((0, 3, 3)),
        ),
        dtype=np.float64,
    )

    coordinate_frame = scalar_text(
        arrays.get(
            "coordinate_frame"
        )
    )

    position_mask = np.isfinite(
        all_positions
    ).all(axis=1)

    candidate_mask = np.isfinite(
        all_candidates
    ).all(axis=1)

    camera_mask = strong_camera_inlier_mask(
        all_camera_positions
    )

    positions = all_positions[
        position_mask
    ]

    candidates = all_candidates[
        candidate_mask
    ]

    camera_positions = (
        all_camera_positions[
            camera_mask
        ]
    )

    if all_camera_rotations.shape == (
        len(all_camera_positions),
        3,
        3,
    ):
        camera_rotations = (
            all_camera_rotations[
                camera_mask
            ]
        )
    else:
        camera_rotations = (
            all_camera_rotations
        )

    if not len(positions):
        raise ValueError(
            "Outlier filtering rejected "
            "every selected landmark"
        )

    extent_arrays = [
        positions
    ]

    if len(candidates):
        extent_arrays.append(
            candidates
        )

    if len(camera_positions):
        extent_arrays.append(
            camera_positions
        )

    all_points = np.concatenate(
        extent_arrays,
        axis=0,
    )

    map_extent = max(
        float(
            np.max(
                np.ptp(
                    all_points,
                    axis=0,
                )
            )
        ),
        1.0,
    )

    figure = go.Figure()

    if len(candidates):
        figure.add_trace(
            point_trace(
                candidates,
                "candidate landmarks",
                "#9e9e9e",
                max(
                    1.0,
                    POINT_SIZE * 0.55,
                ),
                visible=SHOW_CANDIDATES,
                opacity=0.22,
            )
        )

    figure.add_trace(
        point_trace(
            positions,
            "selected landmarks",
            positions[:, 2],
            POINT_SIZE,
        )
    )

    if len(camera_positions):
        add_camera_traces(
            figure,
            camera_positions,
            camera_rotations,
        )

    add_coordinate_axes(
        figure,
        0.09 * map_extent,
        coordinate_frame,
    )

    figure.update_layout(
        title={
            "text": (
                f"{map_path.name} — "
                f"{coordinate_frame} frame"
                "<br>"
                f"<sup>{get_view_description()}</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_dark",
        paper_bgcolor="#111418",
        margin={
            "l": 0,
            "r": 0,
            "t": 80,
            "b": 0,
        },
        legend={
            "x": 0.01,
            "y": 0.99,
            "bgcolor": (
                "rgba(20, 24, 28, 0.72)"
            ),
            "bordercolor": "#555",
            "borderwidth": 1,
            "groupclick": "toggleitem",
        },
        scene={
            "aspectmode": "data",
            "dragmode": get_drag_mode(),
            "xaxis": {
                "title": "X [mm]",
                "showspikes": False,
            },
            "yaxis": {
                "title": "Y [mm]",
                "showspikes": False,
            },
            "zaxis": {
                "title": "Z [mm]",
                "showspikes": False,
            },
            "camera": get_scene_camera(),
        },
        hoverlabel={
            "font_size": 13,
        },
    )

    counts = (
        (
            len(positions),
            len(all_positions)
            - len(positions),
        ),
        (
            len(candidates),
            len(all_candidates)
            - len(candidates),
        ),
        (
            len(camera_positions),
            len(all_camera_positions)
            - len(camera_positions),
        ),
    )

    return (
        figure,
        coordinate_frame,
        counts,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    map_path = (
        MAP_PATH
        .expanduser()
        .resolve()
    )

    arrays = load_map(
        map_path
    )

    figure, frame, counts = create_viewer(
        arrays,
        map_path,
    )

    print(
        f"Map: {map_path}"
    )

    print(
        f"Coordinate frame: {frame}"
    )

    print(
        f"View mode: {VIEW_MODE}"
    )

    print(
        f"Selected landmarks: "
        f"{counts[0][0]} "
        f"(removed outliers: "
        f"{counts[0][1]})"
    )

    print(
        f"Candidate landmarks: "
        f"{counts[1][0]} "
        f"(removed outliers: "
        f"{counts[1][1]})"
    )

    print(
        f"Mapping cameras: "
        f"{counts[2][0]} "
        f"(removed outliers: "
        f"{counts[2][1]})"
    )

    print(
        "Opening interactive viewer "
        "in the browser..."
    )

    if SAVE_HTML_PATH is not None:
        html_path = (
            Path(SAVE_HTML_PATH)
            .expanduser()
            .resolve()
        )

        figure.write_html(
            html_path,
            include_plotlyjs=True,
        )

        print(
            f"Saved standalone viewer: "
            f"{html_path}"
        )

    pio.renderers.default = "browser"

    figure.show(
        config={
            "displaylogo": False,
            "scrollZoom": True,
        }
    )


if __name__ == "__main__":
    main()