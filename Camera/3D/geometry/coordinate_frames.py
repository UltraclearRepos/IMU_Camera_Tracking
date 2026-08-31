import numpy as np


# Fixed orientation of the camera mounted on the Dobot TCP.
# Relative to the previous camera convention: X' = -Y, Y' = X, Z' = Z.
# It converts vectors expressed in TCP axes into native camera axes:
# Dobot +X -> camera +Z, Dobot +Y -> camera -X, Dobot +Z -> camera -Y.
# Cylinder Horizontal
TCP_TO_CAMERA_AXES = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)

# Cylinder Vertical
# TCP_TO_CAMERA_AXES = np.array(
#     [
#         [0.0, 0.0, -1.0],
#         [0.0, 1.0, 0.0],
#         [1.0, 0.0, 0.0],
#     ]
# )


# Provisional fixed orientation of the IMU rigidly mounted to the camera.
# It converts a vector expressed in IMU axes into native OpenCV camera axes.
CAMERA_FROM_IMU = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
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


def imu_vectors_to_camera_axes(vectors):
    return (CAMERA_FROM_IMU @ np.asarray(vectors).T).T
