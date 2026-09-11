"""Normalize the new Cylinder recordings to the repository's CSV schema.

The logger originally stored all IMU sample fields in one ``data`` column.
Other datasets expose these fields as individual columns.  Dobot records also
need the Euler columns expected by the evaluation code; its planar ``r`` axis
is the yaw angle in the current dataset convention.
"""

import csv
from pathlib import Path


DATASET_DIRECTORIES = tuple(
    Path(__file__).resolve().parent / "Data" / name
    for name in ("CylinderVertical", "CylinderVertical3Aruco")
)

IMU_DATA_COLUMNS = (
    "rtc_date",
    "rtc_time",
    "imu1_ax_mg",
    "imu1_ay_mg",
    "imu1_az_mg",
    "imu1_gx_deg_s",
    "imu1_gy_deg_s",
    "imu1_gz_deg_s",
    "mag_x_uT",
    "mag_y_uT",
    "mag_z_uT",
    "imu_temperature_c",
    "imu2_ax_mg",
    "imu2_ay_mg",
    "imu2_az_mg",
    "imu2_gx_mdps",
    "imu2_gy_mdps",
    "imu2_gz_mdps",
    "data_ready",
    "output_hz",
)


def write_csv(path, fieldnames, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def unpack_imu_file(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        input_columns = reader.fieldnames
        rows = list(reader)

    if input_columns != ["timestamp", "sync_timestamp", "data"]:
        if input_columns == ["timestamp", "sync_timestamp", *IMU_DATA_COLUMNS]:
            return False
        raise ValueError(f"Unexpected IMU columns in {path}: {input_columns}")

    output_rows = []
    for row_number, row in enumerate(rows, start=2):
        values = row["data"].rstrip(",").split(",")
        if len(values) != len(IMU_DATA_COLUMNS):
            raise ValueError(
                f"{path}:{row_number} has {len(values)} IMU values; "
                f"expected {len(IMU_DATA_COLUMNS)}"
            )
        output_rows.append(
            {
                "timestamp": row["timestamp"],
                "sync_timestamp": row["sync_timestamp"],
                **dict(zip(IMU_DATA_COLUMNS, values, strict=True)),
            }
        )

    write_csv(path, ["timestamp", "sync_timestamp", *IMU_DATA_COLUMNS], output_rows)
    return True


def add_dobot_orientation_file(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        input_columns = reader.fieldnames
        rows = list(reader)

    target_columns = [
        "timestamp",
        "sync_timestamp",
        "x",
        "y",
        "z",
        "r",
        "roll",
        "pitch",
        "yaw",
    ]
    if input_columns == target_columns:
        return False
    if input_columns != target_columns[:6]:
        raise ValueError(f"Unexpected Dobot columns in {path}: {input_columns}")

    for row in rows:
        row["roll"] = "0"
        row["pitch"] = "0"
        row["yaw"] = row["r"]

    write_csv(path, target_columns, rows)
    return True


def main():
    changed_imu = sum(
        unpack_imu_file(path)
        for dataset_directory in DATASET_DIRECTORIES
        for path in sorted((dataset_directory / "imu").glob("*.csv"))
    )
    changed_dobot = sum(
        add_dobot_orientation_file(path)
        for dataset_directory in DATASET_DIRECTORIES
        for path in sorted((dataset_directory / "dobot").glob("*.csv"))
    )
    print(f"Normalized {changed_imu} IMU files and {changed_dobot} Dobot files.")


if __name__ == "__main__":
    main()
