import numpy as np


# Provisional fixed orientation of the camera mounted on the Dobot TCP.
# It converts vectors expressed in TCP axes into native camera axes:
# Dobot +X -> camera -X, Dobot +Z -> camera -Y.
TCP_TO_CAMERA_AXES = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ]
)


def tcp_displacements_to_camera_axes(displacements):
    return (TCP_TO_CAMERA_AXES @ np.asarray(displacements).T).T


def tcp_rotations_to_camera_axes(rotations):
    rotations = np.asarray(rotations)
    return np.einsum(
        "ij,njk,kl->nil",
        TCP_TO_CAMERA_AXES,
        rotations,
        TCP_TO_CAMERA_AXES.T,
    )
