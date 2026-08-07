import csv
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "Camera" / "DataOld"
DATA_FOLDER = "firstSkinData"
OUTPUT_DIR = PROJECT_DIR / "Data" / DATA_FOLDER
FPS = 30.0

DATASETS = {
    "square_1": (
        SOURCE_DIR
        / "square_4x_5sp__x,y-002"
        / "square_4x_5sp__x,y",
        1774953674.0,
    ),
    "square_2": (
        SOURCE_DIR
        / "square_4x_8sp__x,y"
        / "square_4x_8sp__x,y",
        1774953882.0,
    ),
    "triangle_1": (
        SOURCE_DIR / "triangle_4x_5sp__x,y",
        1774954203.0,
    ),
    "triangle_2": (
        SOURCE_DIR / "triangle_4x_8sp__x,y",
        1774954436.0,
    ),
}


def convert_ground_truth(source_path, output_path, start_timestamp):
    with source_path.open(newline="", encoding="utf-8") as source_file:
        rows = list(csv.DictReader(source_file))

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(["timestamp", "x", "y", "z", "roll", "pitch", "yaw"])
        for row in rows:
            timestamp = start_timestamp + float(row["Frame"]) / FPS
            writer.writerow(
                [
                    f"{timestamp:.9f}",
                    row["X"],
                    row["Y"],
                    row["Z"],
                    row["Roll"],
                    row["Pitch"],
                    row["Yaw"],
                ]
            )


def write_video_timestamp(output_path, start_timestamp):
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(["source", "start_timestamp"])
        writer.writerow(["cam1", f"{start_timestamp:.9f}"])


def main():
    for directory in ("dobot", "video_timestamps", "videos"):
        (OUTPUT_DIR / directory).mkdir(parents=True, exist_ok=True)

    for name, (source, start_timestamp) in DATASETS.items():
        convert_ground_truth(
            source / "ground_truth" / "position.csv",
            OUTPUT_DIR / "dobot" / f"{name}.csv",
            start_timestamp,
        )
        write_video_timestamp(
            OUTPUT_DIR / "video_timestamps" / f"{name}.csv",
            start_timestamp,
        )
        shutil.copyfile(
            source / "probe_camera" / "video.mp4",
            OUTPUT_DIR / "videos" / f"{name}_cam1.mp4",
        )
        print(f"Prepared {name}")


if __name__ == "__main__":
    main()
