"""Convert calibration recordings to explicit, unique CSV columns."""

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPT_DIR / "calibration"

FILES = [
    "g_+X_2026-07-16_16.16.31.csv",
    "g_-X_2026-07-16_16.16.53.csv",
    "g_+Y_2026-07-16_16.15.08.csv",
    "g_-Y_2026-07-16_16.15.28.csv",
    "g_+Z_2026-07-16_16.13.15.csv",
    "g_-Z_2026-07-16_16.12.26.csv",
]

COLUMNS = [
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


def normalize_file(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    if rows[0] == COLUMNS:
        return False

    normalized_rows = []
    for row in rows[1:]:
        values = row[1].split(",")
        if values[-1] == "":
            values = values[:-1]
        normalized_rows.append([row[0], *values])

    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(normalized_rows)

    temporary_path.replace(path)
    return True


def main():
    for filename in FILES:
        path = CALIBRATION_DIR / filename
        if normalize_file(path):
            print(f"Normalized: {path}")
        else:
            print(f"Already normalized: {path}")


if __name__ == "__main__":
    main()
