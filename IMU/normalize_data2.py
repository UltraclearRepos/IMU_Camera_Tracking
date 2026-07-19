"""Normalize Data2 IMU and Dobot CSV files to explicit columns."""

import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
IMU_DIR = PROJECT_DIR / "Data2" / "imu"
DOBOT_DIR = PROJECT_DIR / "Data2" / "dobot"

IMU_COLUMNS = [
    "timestamp",
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
]


def write_rows(path, columns, rows):
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    temporary_path.replace(path)


def normalize_imu(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    if rows[0] == IMU_COLUMNS:
        return False

    normalized_rows = []
    for row in rows[1:]:
        values = row[1].split(",")
        if values[-1] == "":
            values = values[:-1]
        normalized_rows.append([row[0], *values])

    write_rows(path, IMU_COLUMNS, normalized_rows)
    return True


def add_dobot_orientation(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames

    if all(column in columns for column in ("roll", "pitch", "yaw")):
        return False

    output_columns = [*columns, "roll", "pitch", "yaw"]
    output_rows = []
    for row in rows:
        output_rows.append([row.get(column, "0.0") for column in output_columns])

    write_rows(path, output_columns, output_rows)
    return True


def main():
    for path in sorted(IMU_DIR.glob("*.csv")):
        if normalize_imu(path):
            print(f"Normalized IMU: {path.name}")

    for path in sorted(DOBOT_DIR.glob("*.csv")):
        if add_dobot_orientation(path):
            print(f"Added Dobot orientation: {path.name}")


if __name__ == "__main__":
    main()
