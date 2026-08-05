from pathlib import Path

import numpy as np
from PIL import Image


SQUARES_X = 10
SQUARES_Y = 7
SQUARE_LENGTH_MM = 20.0
MARGIN_MM = 10.0
DPI = 600

OUTPUT_PATH = Path(__file__).resolve().parent / "chessboard.png"


def millimeters_to_pixels(value):
    return round(value * DPI / 25.4)


def create_chessboard():
    square_pixels = millimeters_to_pixels(SQUARE_LENGTH_MM)
    board = np.full(
        (SQUARES_Y * square_pixels, SQUARES_X * square_pixels),
        255,
        dtype=np.uint8,
    )

    for row in range(SQUARES_Y):
        for column in range(SQUARES_X):
            if (row + column) % 2 == 0:
                y0 = row * square_pixels
                y1 = (row + 1) * square_pixels
                x0 = column * square_pixels
                x1 = (column + 1) * square_pixels
                board[y0:y1, x0:x1] = 0

    return Image.fromarray(board)


def main():
    board = create_chessboard()
    margin_pixels = millimeters_to_pixels(MARGIN_MM)
    image = Image.new(
        "L",
        (
            board.width + 2 * margin_pixels,
            board.height + 2 * margin_pixels,
        ),
        255,
    )
    image.paste(board, (margin_pixels, margin_pixels))
    image.save(OUTPUT_PATH, dpi=(DPI, DPI))

    board_width_mm = SQUARES_X * SQUARE_LENGTH_MM
    board_height_mm = SQUARES_Y * SQUARE_LENGTH_MM
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Squares: {SQUARES_X} x {SQUARES_Y}")
    print(f"Inner corners: {SQUARES_X - 1} x {SQUARES_Y - 1}")
    print(
        f"Board size: {board_width_mm:.0f} x "
        f"{board_height_mm:.0f} mm"
    )
    print(
        f"Image size with margins: "
        f"{board_width_mm + 2.0 * MARGIN_MM:.0f} x "
        f"{board_height_mm + 2.0 * MARGIN_MM:.0f} mm"
    )
    print("Print at 100% scale and verify the square size with a ruler.")


if __name__ == "__main__":
    main()
