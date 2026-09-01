import numpy as np


# Fixed orientations of the camera mounted on the Dobot TCP. They convert
# vectors expressed in TCP axes into native camera axes.
TCP_TO_CAMERA_AXES_BY_CYLINDER_ORIENTATION = {
    "horizontal": np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ]
    ),
    "vertical": np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ),
}

# Provisional fixed orientation of the IMU rigidly mounted to the camera.
# It converts a vector expressed in IMU axes into native OpenCV camera axes.
CAMERA_FROM_IMU = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


def get_tcp_to_camera_axes(cylinder_orientation):
    if not isinstance(cylinder_orientation, str):
        raise ValueError("cylinder_orientation must be 'horizontal' or 'vertical'")
    orientation = cylinder_orientation.lower()
    try:
        return TCP_TO_CAMERA_AXES_BY_CYLINDER_ORIENTATION[orientation]
    except KeyError as error:
        raise ValueError(
            "cylinder_orientation must be 'horizontal' or 'vertical', "
            f"got {cylinder_orientation!r}"
        ) from error


def tcp_displacements_to_camera_axes(
    displacements,
    cylinder_orientation,
):
    axes = get_tcp_to_camera_axes(cylinder_orientation)
    return (axes @ np.asarray(displacements).T).T


def tcp_rotations_to_camera_axes(rotations, cylinder_orientation):
    rotations = np.asarray(rotations)
    axes = get_tcp_to_camera_axes(cylinder_orientation)
    return np.einsum(
        "ij,njk,kl->nil",
        axes,
        rotations,
        axes.T,
    )


def imu_vectors_to_camera_axes(vectors):
    return (CAMERA_FROM_IMU @ np.asarray(vectors).T).T
