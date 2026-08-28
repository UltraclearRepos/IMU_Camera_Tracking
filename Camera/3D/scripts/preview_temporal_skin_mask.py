import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MODULE_DIR.parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from mapping.aruco_mask import ArucoMask
from mapping.temporal_skin_mask import TemporalSkinMask
from scripts.preview_skin_mask import (
    blend_colour,
    draw_mask_diagnostics,
    find_video,
    save_initial_growing_diagnostics,
)


RECORDING_NAME = "initial_50mm_Arc180-Speed-3_2026-08-20_15.30.28"
DATA_FOLDER = "Cylinder"
FEATURE_ROI_BOTTOM_FRACTION = 0.85
ARUCO_MASK_MARGIN_MM = 7.0
EROSION_KERNEL_SIZE = 13
SEARCH_KERNEL_SIZE = 17
MAX_FRAMES = None
OUTPUT_PATH = (
    MODULE_DIR
    / "results"
    / DATA_FOLDER
    / "temporal_skin_mask_preview"
    / f"{RECORDING_NAME}.mp4"
)


def draw_temporal_diagnostics(
    frame,
    frame_index,
    roi_top,
    skin_result,
    skin_mask,
    aruco_exclusion_mask,
    detected_aruco_ids,
):
    context = frame.copy()
    search_mask = skin_mask.last_search_mask
    eroded_previous = skin_mask.last_eroded_previous_mask

    if search_mask is not None:
        search_mask = cv2.resize(
            search_mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        blend_colour(context, search_mask, (0, 255, 255), 0.08)
        search_contours, _ = cv2.findContours(
            search_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            context,
            search_contours,
            -1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if eroded_previous is not None:
        eroded_previous = cv2.resize(
            eroded_previous.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        blend_colour(context, eroded_previous, (255, 0, 0), 0.22)
        eroded_contours, _ = cv2.findContours(
            eroded_previous.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            context,
            eroded_contours,
            -1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    output = draw_mask_diagnostics(
        context,
        frame_index,
        roi_top,
        skin_result,
        aruco_exclusion_mask,
        detected_aruco_ids,
    )
    height, width = output.shape[:2]
    cv2.rectangle(
        output,
        (0, height - 25),
        (width - 1, height - 1),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        output,
        "blue: eroded previous | yellow: search boundary | green/magenta: new mask",
        (10, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def run_preview(
    recording_name,
    data_folder,
    roi_bottom_fraction,
    aruco_mask_margin_mm,
    erosion_kernel_size,
    search_kernel_size,
    max_frames,
    output_path,
):
    video_path = find_video(data_folder, recording_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or fps <= 0.0:
        capture.release()
        raise RuntimeError("Video does not report valid dimensions and FPS")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video: {output_path}")

    roi_top = round(height * (1.0 - roi_bottom_fraction))
    aruco_mask = ArucoMask(margin_mm=aruco_mask_margin_mm)
    skin_mask = TemporalSkinMask(
        erosion_kernel_size=erosion_kernel_size,
        search_kernel_size=search_kernel_size,
    )
    processed_frames = 0
    initial_growing_path = None

    try:
        while max_frames is None or processed_frames < max_frames:
            success, frame = capture.read()
            if not success:
                break

            frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            aruco_exclusion_mask = aruco_mask.compute(frame)
            skin_result = skin_mask.compute(
                frame,
                roi_top,
                aruco_exclusion_mask,
            )
            output = draw_temporal_diagnostics(
                frame,
                frame_index,
                roi_top,
                skin_result,
                skin_mask,
                aruco_exclusion_mask,
                aruco_mask.last_detected_ids,
            )
            if processed_frames == 0:
                initial_growing_path = save_initial_growing_diagnostics(
                    output,
                    skin_mask,
                    output_path,
                )
            writer.write(output)
            processed_frames += 1
            if processed_frames == 1 or processed_frames % 100 == 0:
                print(f"Temporal skin mask preview: frame {frame_index}")
    finally:
        capture.release()
        writer.release()

    if processed_frames == 0:
        raise RuntimeError(f"No frames were read from {video_path}")

    effective_fps = (
        skin_mask.compute_count / skin_mask.compute_seconds
        if skin_mask.compute_seconds > 0.0
        else 0.0
    )
    print(
        f"Temporal skin mask timing: {skin_mask.compute_seconds:.3f} s total, "
        f"{skin_mask.average_compute_ms:.2f} ms/frame, "
        f"{effective_fps:.1f} FPS"
    )
    if initial_growing_path is not None:
        print(f"Saved initial growing image: {initial_growing_path}")
    print(f"Saved {processed_frames} frames: {output_path}")
    return output_path


def main():
    run_preview(
        recording_name=RECORDING_NAME,
        data_folder=DATA_FOLDER,
        roi_bottom_fraction=FEATURE_ROI_BOTTOM_FRACTION,
        aruco_mask_margin_mm=ARUCO_MASK_MARGIN_MM,
        erosion_kernel_size=EROSION_KERNEL_SIZE,
        search_kernel_size=SEARCH_KERNEL_SIZE,
        max_frames=MAX_FRAMES,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
