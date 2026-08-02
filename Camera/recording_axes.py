import numpy as np


# Add a separate verified axis mapping for every recording.
CAMERA_MAP_TO_DOBOT_BY_RECORDING = {
    "horizontal_line_1": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "square_1": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "square_2": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "triangle_1": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "triangle_2": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "skin_low_res_Speed-3_2026-07-21_13.15.30": np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "tattoo_low_res_Speed-3_2026-07-21_13.27.00": np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "skin_high_res_Speed-3_2026-07-21_13.18.44": np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "initialpos-dark-nolight_Speed-3_2026-07-28_16.55.02": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "initialpos-dark-withlight_Speed-3_2026-07-28_16.57.56": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "far-dark-withlight_Speed-3_2026-07-28_17.02.52": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "far-dark-nolight_Speed-3_2026-07-28_17.04.19": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "far-white-nolight_Speed-3_2026-07-28_17.06.45": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "far-white-withlight_Speed-3_2026-07-28_17.08.22": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "close-white-withlight_Speed-3_2026-07-28_17.12.37": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "close-white-nolight_Speed-3_2026-07-28_17.14.02": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "close-dark-nolight_Speed-3_2026-07-28_17.16.20": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "close-dark-withlight_Speed-3_2026-07-28_17.17.50": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
    "initialpos-white-withlight_Speed-3_2026-07-29_17.46.25": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "initialpos-white-nolight_Speed-3_2026-07-29_17.47.53": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "initial-white-withlight-25deg_Speed-3_2026-07-30_13.06.03": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "initial-white-nolight-25deg_Speed-3_2026-07-30_13.07.33": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "close-white-nolight-25deg_Speed-3_2026-07-30_13.10.31": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "close-white-withlight-25deg_Speed-3_2026-07-30_13.11.32": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "far-white-withlight-25deg_Speed-3_2026-07-30_13.13.13": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "far-white-nolight-25deg_Speed-3_2026-07-30_13.14.23": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "initial-black-nolight-25deg_Speed-3_2026-07-30_13.55.04": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "initial-black-withlight-25deg_Speed-3_2026-07-30_13.56.04": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "far-black-withlight-25deg_Speed-3_2026-07-30_13.58.22": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "far-black-nolight-25deg_Speed-3_2026-07-30_14.00.25": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "close-black-nolight-25deg_Speed-3_2026-07-30_14.08.24": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    "close-black-withlight-25deg_Speed-3_2026-07-30_14.09.41": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
}


# OnlyR uses the opposite pitch direction for the camera orientation.
CAMERA_EULER_SIGNS_BY_RECORDING = {
    recording_name: np.array(
        [1.0, -1.0, 1.0]
    )
    for recording_name in CAMERA_MAP_TO_DOBOT_BY_RECORDING
}
