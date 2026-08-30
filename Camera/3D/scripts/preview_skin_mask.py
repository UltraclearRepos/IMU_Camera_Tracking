import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MODULE_DIR.parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from mapping.adaptive_skin_mask import AdaptiveSkinMask
from mapping.aruco_mask import ArucoMask


RECORDING_NAME = "initial_50mm_Arc180-Speed-3_2026-08-20_15.30.28"
DATA_FOLDER = "Cylinder"
CAMERA_NAME = "cam1"
KEYFRAME_INTERVAL = 10
FEATURE_ROI_BOTTOM_FRACTION = 0.85
ARUCO_MASK_MARGIN_MM = 7.0
MAX_FRAMES = None  # None processes the entire video.
REINITIALIZE_FRAME = 1081  # Source frame number, or None to disable.
OUTPUT_PATH = (
    MODULE_DIR
    / "results"
    / DATA_FOLDER
    / "skin_mask_preview"
    / f"{RECORDING_NAME}.mp4"
)


def find_video(data_folder, recording_name):
    videos_directory = PROJECT_DIR / "Data" / data_folder / "videos"
    paths = list(
        videos_directory.glob(f"{recording_name}_{CAMERA_NAME}.*")
    )
    if not paths:
        raise FileNotFoundError(
            f"Video for {recording_name!r} was not found in "
            f"{videos_directory}"
        )
    if len(paths) > 1:
        raise RuntimeError(
            f"Multiple videos found for {recording_name!r}: "
            + ", ".join(str(path) for path in paths)
        )
    return paths[0]


def blend_colour(image, selected, colour, alpha):
    if not np.any(selected):
        return
    colour = np.asarray(colour, dtype=np.float32)
    pixels = image[selected].astype(np.float32)
    image[selected] = np.rint(
        (1.0 - alpha) * pixels + alpha * colour
    ).astype(np.uint8)


def draw_mask_diagnostics(
    frame,
    frame_index,
    roi_top,
    skin_result,
    aruco_exclusion_mask,
    detected_aruco_ids,
):
    output = frame.copy()
    aruco_excluded = aruco_exclusion_mask == 0
    blend_colour(output, aruco_excluded, (0, 0, 255), 0.45)

    if skin_result is None:
        skin_mask = np.zeros(frame.shape[:2], dtype=bool)
        status = "NO SKIN COMPONENT"
        status_colour = (0, 0, 255)
    else:
        skin_mask = skin_result.mask
        blend_colour(output, skin_mask, (0, 200, 0), 0.25)
        contours, _ = cv2.findContours(
            skin_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            output,
            contours,
            -1,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        left, top, right, bottom = map(int, skin_result.bounds)
        cv2.rectangle(
            output,
            (left, top),
            (right - 1, bottom - 1),
            (255, 255, 0),
            1,
        )
        valid_roi = np.ones(frame.shape[:2], dtype=bool)
        valid_roi[:roi_top] = False
        valid_roi &= ~aruco_excluded
        valid_pixel_count = np.count_nonzero(valid_roi)
        coverage = (
            100.0 * np.count_nonzero(skin_mask) / valid_pixel_count
            if valid_pixel_count
            else 0.0
        )
        status = f"skin coverage: {coverage:.1f}%"
        status_colour = (255, 255, 255)

    aruco_contours, _ = cv2.findContours(
        aruco_excluded.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        output,
        aruco_contours,
        -1,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        output,
        (0, roi_top),
        (output.shape[1] - 1, roi_top),
        (255, 255, 0),
        1,
    )

    aruco_ids = (
        "none"
        if not len(detected_aruco_ids)
        else ",".join(str(int(value)) for value in detected_aruco_ids)
    )
    cv2.rectangle(output, (0, 0), (output.shape[1] - 1, 58), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"frame {frame_index} | ArUco: {aruco_ids}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        status,
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        status_colour,
        1,
        cv2.LINE_AA,
    )
    return output


def save_initial_growing_diagnostics(output, skin_mask, output_path):
    seed = skin_mask.initial_seed
    if seed is None:
        return None

    diagnostic = output.copy()
    blend_colour(diagnostic, seed, (0, 255, 255), 0.35)
    seed_contours, _ = cv2.findContours(
        seed.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        diagnostic,
        seed_contours,
        -1,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    height, width = diagnostic.shape[:2]
    cv2.rectangle(diagnostic, (0, height - 25), (width - 1, height - 1), (0, 0, 0), -1)
    cv2.putText(
        diagnostic,
        "yellow: initial skin seed | green/magenta: grown skin mask",
        (10, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    image_path = output_path.with_name(
        f"{output_path.stem}_initial_growing.png"
    )
    if not cv2.imwrite(str(image_path), diagnostic):
        raise RuntimeError(f"Could not save initial growing image: {image_path}")
    return image_path


def run_preview(
    recording_name,
    data_folder,
    roi_bottom_fraction,
    aruco_mask_margin_mm,
    max_frames,
    output_path,
    reinitialize_frame=None,
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
    skin_mask = AdaptiveSkinMask()
    skin_masks = [skin_mask]
    processed_frames = 0
    initial_growing_path = None
    reinitialized = False

    try:
        while max_frames is None or processed_frames < max_frames:
            success, frame = capture.read()
            if not success:
                break

            timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_index = round(timestamp_s * fps)

            if (
                frame_index % KEYFRAME_INTERVAL != 0
                and frame_index != reinitialize_frame
            ):
                continue

            if frame_index == reinitialize_frame:
                skin_mask = AdaptiveSkinMask()
                skin_masks.append(skin_mask)
                reinitialized = True
                print(f"Reinitializing skin mask at frame {frame_index}")

            aruco_exclusion_mask = aruco_mask.compute(frame)
            skin_result = skin_mask.compute(
                frame,
                roi_top,
                aruco_exclusion_mask,
            )
            output = draw_mask_diagnostics(
                frame,
                frame_index,
                roi_top,
                skin_result,
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
                print(f"Skin mask preview: frame {frame_index}")
    finally:
        capture.release()
        writer.release()

    if processed_frames == 0:
        raise RuntimeError(f"No frames were read from {video_path}")

    compute_count = sum(mask.compute_count for mask in skin_masks)
    compute_seconds = sum(mask.compute_seconds for mask in skin_masks)
    average_compute_ms = (
        1000.0 * compute_seconds / compute_count
        if compute_count > 0
        else 0.0
    )
    effective_fps = (
        compute_count / compute_seconds
        if compute_seconds > 0.0
        else 0.0
    )
    print(
        f"Skin mask timing: {compute_seconds:.3f} s total, "
        f"{average_compute_ms:.2f} ms/frame, "
        f"{effective_fps:.1f} FPS"
    )
    if reinitialize_frame is not None and not reinitialized:
        print(f"Reinitialization frame {reinitialize_frame} was not reached")
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
        max_frames=MAX_FRAMES,
        output_path=OUTPUT_PATH,
        reinitialize_frame=REINITIALIZE_FRAME,
    )


if __name__ == "__main__":
    main()
