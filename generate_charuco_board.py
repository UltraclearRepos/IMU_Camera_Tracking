from pathlib import Path

import cv2
from PIL import Image


SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH_MM = 30.0
MARKER_LENGTH_MM = 15.0
MARGIN_MM = 10.0
DPI = 600
ARUCO_DICTIONARY = cv2.aruco.DICT_5X5_100

OUTPUT_PATH = Path(__file__).resolve().parent / "charuco_board.png"


def millimeters_to_pixels(value):
    return round(value * DPI / 25.4)


def main():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_MM,
        MARKER_LENGTH_MM,
        dictionary,
    )

    board_width_mm = SQUARES_X * SQUARE_LENGTH_MM
    board_height_mm = SQUARES_Y * SQUARE_LENGTH_MM
    board_size = (
        millimeters_to_pixels(board_width_mm),
        millimeters_to_pixels(board_height_mm),
    )
    board_image = Image.fromarray(
        board.generateImage(
            board_size,
            marginSize=0,
            borderBits=1,
        )
    )

    margin_pixels = millimeters_to_pixels(MARGIN_MM)
    image = Image.new(
        "L",
        (
            board_size[0] + 2 * margin_pixels,
            board_size[1] + 2 * margin_pixels,
        ),
        255,
    )
    image.paste(board_image, (margin_pixels, margin_pixels))
    image.save(OUTPUT_PATH, dpi=(DPI, DPI))

    print(f"Saved: {OUTPUT_PATH}")
    print(
        f"Board size: {board_width_mm:.0f} x "
        f"{board_height_mm:.0f} mm"
    )
    print(
        f"Image size with margins: "
        f"{board_width_mm + 2.0 * MARGIN_MM:.0f} x "
        f"{board_height_mm + 2.0 * MARGIN_MM:.0f} mm"
    )


if __name__ == "__main__":
    main()
