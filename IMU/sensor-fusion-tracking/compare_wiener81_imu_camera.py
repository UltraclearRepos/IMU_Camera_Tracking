"""Compare Wiener(81) IMU positioning with the current IMU+camera Kalman.

The existing Kalman implementation is intentionally used without modification,
so this script can serve as the baseline before improving the fusion filter.
Edit the configuration variables below and run this file without CLI arguments.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags

from funkcje_GT import (
    calculate_derivatives,
    load_ground_truth,
    normalize_ground_truth,
    resample_ground_truth,
    trim_ground_truth,
)
from funkcje_IMU import (
    calculate_integrals,
    compute_orientation_and_global_acc,
    load_IMU_data,
    remove_average_trend,
    resample_IMU_data,
    trim_IMU_data,
)
from funkcje_IMU_GT import synchronize_by_cross_correlation
from funkcje_camera import load_camera_data
from IMUFilter import IMUFilter
from IntegratedKalmanFilter import IntegratedKalmanFilter


# The fixed-filter runner replaces these two values while reusing this exact
# preprocessing, synchronization, metric and plotting pipeline.
KALMAN_CLASS = IntegratedKalmanFilter
FUSION_LABEL = "IMU + native camera 30 Hz (current Kalman)"


DATASETS = {
    "horizontal_line_1": {
        "imu": "IMU/dataLog00075.TXT",
        "gt": "dobot/horizontal_line_1774951923.csv",
        "camera": "POMIARY/horizontal_10x_5sp__x/optical_flow.csv",
    },
    "horizontal_line_2": {
        "imu": "IMU/dataLog00077.TXT",
        "gt": "dobot/horizontal_line_1774952687.csv",
        "camera": "POMIARY/horizontal_10x_8sp__x/optical_flow.csv",
    },
    "vertical_line_1": {
        "imu": "IMU/dataLog00079.TXT",
        "gt": "dobot/vertical_line_1774953045.csv",
        "camera": "POMIARY/vertical_10x_5sp__y/optical_flow.csv",
    },
    "vertical_line_2": {
        "imu": "IMU/dataLog00081.TXT",
        "gt": "dobot/vertical_line_1774953360.csv",
        "camera": "POMIARY/vertical_10x_8sp__y/optical_flow.csv",
    },
    "square_1": {
        "imu": "IMU/dataLog00083.TXT",
        "gt": "dobot/square_1774953674.csv",
        "camera": "POMIARY/square_4x_5sp__x,y/optical_flow.csv",
    },
    "square_2": {
        "imu": "IMU/dataLog00085.TXT",
        "gt": "dobot/square_1774953882.csv",
        "camera": "POMIARY/square_4x_8sp__x,y/optical_flow.csv",
    },
    "triangle_1": {
        "imu": "IMU/dataLog00087.TXT",
        "gt": "dobot/triangle_1774954203.csv",
        "camera": "POMIARY/triangle_4x_5sp__x,y/optical_flow.csv",
    },
    "triangle_2": {
        "imu": "IMU/dataLog00089.TXT",
        "gt": "dobot/triangle_1774954436.csv",
        "camera": "POMIARY/triangle_4x_8sp__x,y/optical_flow.csv",
    },
    "cross_1": {
        "imu": "IMU/dataLog00091.TXT",
        "gt": "dobot/cross_1774954750.csv",
        "camera": "POMIARY/cross_4x_5sp__x,y/optical_flow.csv",
    },
    "cross_2": {
        "imu": "IMU/dataLog00093.TXT",
        "gt": "dobot/cross_1774954990.csv",
        "camera": "POMIARY/cross_4x_8sp__x,y/optical_flow.csv",
    },
}


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SELECTED_DATASET = "horizontal_line_1"
DATA_ROOT = Path("Data")
OUTPUT_DIR = Path("wiener81_imu_camera_results")

SAMPLE_RATE = 100.0
CAMERA_RATE = 30.0
WIENER_WINDOW = 81

IMU_TRIM_START = 1000
IMU_TRIM_END = 500
GT_TRIM_START = 200
GT_TRIM_END = 200

LOOP_CLOSURE = False
SHOW_PLOT = True

# Parameters currently hard-coded in SensorFusionEngine. They are copied here
# so the existing Kalman behavior can be compared without modifying the filter.
KALMAN_Q_POSITION = 1e-8
KALMAN_Q_VELOCITY = 5e-4
KALMAN_Q_BIAS = 1e-5
KALMAN_R_CAMERA = 3e-2


def estimate_camera_time_offset(df_imu, df_camera):
    """Synchronize native 30 Hz camera with IMU without interpolating camera."""

    imu_time = np.arange(len(df_imu), dtype=float) / SAMPLE_RATE
    imu_acc = df_imu[[f"wiener81_acc_{axis}" for axis in "xyz"]].to_numpy()
    imu_magnitude = np.linalg.norm(imu_acc, axis=1)

    if "Frame" not in df_camera.columns:
        raise ValueError("Native camera data must contain the 'Frame' column.")
    camera_time = df_camera["Frame"].to_numpy(dtype=float) / CAMERA_RATE
    camera_position = df_camera[["X", "Y", "Z"]].to_numpy(dtype=float)

    camera_velocity = np.gradient(camera_position, camera_time, axis=0)
    camera_acceleration = np.gradient(camera_velocity, camera_time, axis=0)
    camera_magnitude = np.linalg.norm(camera_acceleration, axis=1)

    # Correlation requires a common rate. Downsample/interpolate IMU to 30 Hz;
    # camera samples remain exactly the original Frame measurements.
    imu_30_time = np.arange(0.0, imu_time[-1], 1.0 / CAMERA_RATE)
    imu_magnitude_30 = np.interp(imu_30_time, imu_time, imu_magnitude)

    imu_centered = imu_magnitude_30 - np.mean(imu_magnitude_30)
    camera_centered = camera_magnitude - np.mean(camera_magnitude)
    cross_correlation = correlate(imu_centered, camera_centered, mode="full")
    lags = correlation_lags(len(imu_centered), len(camera_centered), mode="full")
    lag_frames = int(lags[np.argmax(cross_correlation)])
    offset_seconds = lag_frames / CAMERA_RATE

    print("=== Native camera/IMU synchronization ===")
    print(f"Offset: {lag_frames} camera frames ({offset_seconds:.4f} s)")
    return offset_seconds


def run_native_camera_fusion(df_imu, df_camera, camera_offset_seconds):
    """Predict at IMU rate and update only on original camera frames."""

    dt = 1.0 / SAMPLE_RATE
    kalman = KALMAN_CLASS(dt=dt)
    acceleration = df_imu[[f"wiener81_acc_{axis}" for axis in "xyz"]].to_numpy()
    camera_position = df_camera[["X", "Y", "Z"]].to_numpy(dtype=float)
    camera_time = (
        df_camera["Frame"].to_numpy(dtype=float) / CAMERA_RATE
        + camera_offset_seconds
    )

    # Use the first valid camera measurement as the position origin/state.
    first_valid_camera = int(np.searchsorted(camera_time, 0.0, side="left"))
    if first_valid_camera < len(camera_position):
        kalman.x[0:3, 0] = camera_position[first_valid_camera]

    fused_position = np.zeros((len(df_imu), 3))
    estimated_velocity = np.zeros((len(df_imu), 3))
    estimated_bias = np.zeros((len(df_imu), 3))
    position_std = np.zeros((len(df_imu), 3))
    velocity_std = np.zeros((len(df_imu), 3))
    bias_std = np.zeros((len(df_imu), 3))
    innovation_norm = np.full(len(df_imu), np.nan)
    normalized_innovation_squared = np.full(len(df_imu), np.nan)
    gain_position_norm = np.full(len(df_imu), np.nan)
    gain_velocity_norm = np.full(len(df_imu), np.nan)
    gain_bias_norm = np.full(len(df_imu), np.nan)
    camera_measurement_at_imu = np.full((len(df_imu), 3), np.nan)
    camera_update_used = np.zeros(len(df_imu), dtype=bool)
    camera_index = 0

    for imu_index in range(len(df_imu)):
        current_time = imu_index * dt
        kalman.predict(
            acceleration[imu_index],
            KALMAN_Q_POSITION,
            KALMAN_Q_VELOCITY,
            KALMAN_Q_BIAS,
        )

        # Usually this loop performs zero or one update. The while also handles
        # rare cases where more than one camera frame falls between IMU samples.
        while (
            camera_index < len(camera_time)
            and camera_time[camera_index] <= current_time + 0.5 * dt
        ):
            if camera_time[camera_index] >= current_time - 0.5 * dt:
                kalman.update(
                    camera_position[camera_index],
                    conf=1.0,
                    r_base=KALMAN_R_CAMERA,
                )
                camera_measurement_at_imu[imu_index] = camera_position[camera_index]
                camera_update_used[imu_index] = True

                if getattr(kalman, "last_innovation", None) is not None:
                    innovation_norm[imu_index] = np.linalg.norm(
                        kalman.last_innovation
                    )
                    normalized_innovation_squared[imu_index] = kalman.last_nis
                    gain = kalman.last_kalman_gain
                    gain_position_norm[imu_index] = np.linalg.norm(gain[0:3])
                    gain_velocity_norm[imu_index] = np.linalg.norm(gain[3:6])
                    gain_bias_norm[imu_index] = np.linalg.norm(gain[6:9])
            camera_index += 1

        fused_position[imu_index] = kalman.x[0:3, 0]
        estimated_velocity[imu_index] = kalman.x[3:6, 0]
        estimated_bias[imu_index] = kalman.x[6:9, 0]
        state_std = np.sqrt(np.maximum(np.diag(kalman.P), 0.0))
        position_std[imu_index] = state_std[0:3]
        velocity_std[imu_index] = state_std[3:6]
        bias_std[imu_index] = state_std[6:9]

    result = df_imu.copy()
    result[[f"fused_pos_{axis}" for axis in "xyz"]] = fused_position
    result[[f"estimated_velocity_{axis}" for axis in "xyz"]] = estimated_velocity
    result[[f"estimated_bias_{axis}" for axis in "xyz"]] = estimated_bias
    result[[f"position_std_{axis}" for axis in "xyz"]] = position_std
    result[[f"velocity_std_{axis}" for axis in "xyz"]] = velocity_std
    result[[f"bias_std_{axis}" for axis in "xyz"]] = bias_std
    result["innovation_norm"] = innovation_norm
    result["nis"] = normalized_innovation_squared
    result["gain_position_norm"] = gain_position_norm
    result["gain_velocity_norm"] = gain_velocity_norm
    result["gain_bias_norm"] = gain_bias_norm
    result[[f"native_camera_pos_{axis}" for axis in "xyz"]] = (
        camera_measurement_at_imu
    )
    result["camera_update"] = camera_update_used
    print(f"Native camera updates used: {camera_update_used.sum()}")
    print(
        "Final estimated acceleration bias [m/s^2]: "
        f"{estimated_bias[-1]}"
    )
    return result


def load_and_prepare_data():
    """Prepare IMU/GT and fuse original, non-interpolated camera frames."""

    if SELECTED_DATASET not in DATASETS:
        available = ", ".join(DATASETS)
        raise ValueError(
            f"Unknown dataset {SELECTED_DATASET!r}. Available datasets: {available}"
        )

    paths = DATASETS[SELECTED_DATASET]

    df_imu = load_IMU_data(DATA_ROOT / paths["imu"])
    df_imu = compute_orientation_and_global_acc(df_imu)
    df_imu = trim_IMU_data(df_imu, IMU_TRIM_START, IMU_TRIM_END)
    df_imu = resample_IMU_data(df_imu, target_fps=SAMPLE_RATE)
    df_imu = remove_average_trend(df_imu)

    df_gt = load_ground_truth(DATA_ROOT / paths["gt"])
    df_gt = trim_ground_truth(df_gt, GT_TRIM_START, GT_TRIM_END)
    df_gt = normalize_ground_truth(df_gt)
    df_gt = resample_ground_truth(df_gt, target_fps=SAMPLE_RATE)
    df_gt = calculate_derivatives(df_gt, target_fps=SAMPLE_RATE)

    df_sync = synchronize_by_cross_correlation(df_imu, df_gt)

    # Pure IMU integration starts at zero, so use the same position origin for GT.
    for axis in "xyz":
        gt_column = f"gt_pos_{axis}"
        df_sync[gt_column] -= df_sync[gt_column].iloc[0]

    imu_filter = IMUFilter(fs=SAMPLE_RATE)
    for axis in "xyz":
        df_sync[f"wiener81_acc_{axis}"] = imu_filter.wiener_filter(
            df_sync[f"gl_acc_{axis}"].to_numpy(),
            window=WIENER_WINDOW,
        )

    df_imu_position = calculate_integrals(
        df_sync,
        method="wiener81",
        target_fps=SAMPLE_RATE,
        use_detrend_vel=False,
        use_detrend_pos=False,
        use_loop_closure=LOOP_CLOSURE,
        apply_hp_filter=False,
    )

    df_camera = load_camera_data(DATA_ROOT / paths["camera"])
    camera_offset = estimate_camera_time_offset(
        df_imu_position,
        df_camera,
    )
    df_fused = run_native_camera_fusion(
        df_imu_position,
        df_camera,
        camera_offset_seconds=camera_offset,
    )
    return df_fused


def calculate_position_metrics(df):
    """Calculate per-axis and 3D position errors for both compared methods."""

    gt = df[[f"gt_pos_{axis}" for axis in "xyz"]].to_numpy()
    imu = df[[f"wiener81_pos_{axis}" for axis in "xyz"]].to_numpy()
    fused = df[[f"fused_pos_{axis}" for axis in "xyz"]].to_numpy()

    imu_error = imu - gt
    fused_error = fused - gt
    imu_distance_error = np.linalg.norm(imu_error, axis=1)
    fused_distance_error = np.linalg.norm(fused_error, axis=1)

    metrics = {
        "imu_rmse_3d_mm": float(
            np.sqrt(np.mean(imu_distance_error**2)) * 1000.0
        ),
        "fused_rmse_3d_mm": float(
            np.sqrt(np.mean(fused_distance_error**2)) * 1000.0
        ),
        "imu_axis_rmse_mm": np.sqrt(np.mean(imu_error**2, axis=0)) * 1000.0,
        "fused_axis_rmse_mm": np.sqrt(np.mean(fused_error**2, axis=0)) * 1000.0,
    }
    return gt, imu, fused, imu_distance_error, fused_distance_error, metrics


def plot_comparison(
    df,
    gt,
    imu,
    fused,
    imu_distance_error,
    fused_distance_error,
    metrics,
):
    """Plot X/Y/Z positions and 3D error against GT on a common timeline."""

    if "ts" in df.columns:
        time = df["ts"].to_numpy() - df["ts"].iloc[0]
    else:
        time = np.arange(len(df)) / SAMPLE_RATE

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    position_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

    for index, (axis_name, ax) in enumerate(zip("XYZ", position_axes)):
        ax.plot(time, gt[:, index] * 1000.0, label="GT", color="black", linewidth=2)
        ax.plot(
            time,
            imu[:, index] * 1000.0,
            label="IMU Wiener(81)",
            alpha=0.8,
        )
        ax.plot(
            time,
            fused[:, index] * 1000.0,
            label=FUSION_LABEL,
            alpha=0.85,
        )
        ax.set_title(f"Position {axis_name}")
        ax.set_ylabel("Position [mm]")
        ax.grid(alpha=0.25)
        ax.legend()

    error_ax = axes[1, 1]
    error_ax.plot(
        time,
        imu_distance_error * 1000.0,
        label=f"IMU, RMSE={metrics['imu_rmse_3d_mm']:.1f} mm",
    )
    error_ax.plot(
        time,
        fused_distance_error * 1000.0,
        label=f"IMU+camera, RMSE={metrics['fused_rmse_3d_mm']:.1f} mm",
    )
    error_ax.set_title("3D position error against GT")
    error_ax.set_ylabel("Euclidean error [mm]")
    error_ax.grid(alpha=0.25)
    error_ax.legend()

    for ax in axes[1, :]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(
        f"{SELECTED_DATASET}: Wiener(81) IMU vs native 30 Hz camera fusion",
        fontsize=14,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{SELECTED_DATASET}_position_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Plot saved: {output_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


def plot_gt_vs_fused(df, gt, fused, metrics):
    """Plot only GT and native-camera fusion for the three position axes."""

    if "ts" in df.columns:
        time = df["ts"].to_numpy() - df["ts"].iloc[0]
    else:
        time = np.arange(len(df)) / SAMPLE_RATE

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for index, (axis_name, ax) in enumerate(zip("XYZ", axes)):
        ax.plot(
            time,
            gt[:, index] * 1000.0,
            label="GT",
            color="black",
            linewidth=2.2,
        )
        ax.plot(
            time,
            fused[:, index] * 1000.0,
            label=FUSION_LABEL,
            color="tab:orange",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.set_title(f"Position {axis_name}")
        ax.set_ylabel("Position [mm]")
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(
        f"{SELECTED_DATASET}: GT vs IMU + native camera 30 Hz\n"
        f"3D RMSE = {metrics['fused_rmse_3d_mm']:.2f} mm",
        fontsize=14,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{SELECTED_DATASET}_gt_vs_imu_camera.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"GT vs IMU+camera plot saved: {output_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


def plot_filter_diagnostics(df):
    """Plot state, uncertainty, innovation and gain diagnostics."""

    if "ts" in df.columns:
        time = df["ts"].to_numpy() - df["ts"].iloc[0]
    else:
        time = np.arange(len(df)) / SAMPLE_RATE

    update_mask = df["camera_update"].to_numpy(dtype=bool)
    update_time = time[update_mask]
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)

    bias_ax = axes[0, 0]
    for axis in "xyz":
        bias_ax.plot(time, df[f"estimated_bias_{axis}"], label=axis.upper())
    bias_ax.set_title("Estimated accelerometer bias")
    bias_ax.set_ylabel("Bias [m/s²]")
    bias_ax.grid(alpha=0.25)
    bias_ax.legend()

    position_std_ax = axes[0, 1]
    for axis in "xyz":
        position_std_ax.plot(
            time,
            df[f"position_std_{axis}"] * 1000.0,
            label=axis.upper(),
        )
    position_std_ax.set_title("Estimated position uncertainty (1σ)")
    position_std_ax.set_ylabel("σ position [mm]")
    position_std_ax.grid(alpha=0.25)
    position_std_ax.legend()

    velocity_std_ax = axes[1, 0]
    for axis in "xyz":
        velocity_std_ax.plot(
            time,
            df[f"velocity_std_{axis}"] * 1000.0,
            label=axis.upper(),
        )
    velocity_std_ax.set_title("Estimated velocity uncertainty (1σ)")
    velocity_std_ax.set_ylabel("σ velocity [mm/s]")
    velocity_std_ax.grid(alpha=0.25)
    velocity_std_ax.legend()

    bias_std_ax = axes[1, 1]
    for axis in "xyz":
        bias_std_ax.plot(time, df[f"bias_std_{axis}"], label=axis.upper())
    bias_std_ax.set_title("Estimated bias uncertainty (1σ)")
    bias_std_ax.set_ylabel("σ bias [m/s²]")
    bias_std_ax.grid(alpha=0.25)
    bias_std_ax.legend()

    innovation_ax = axes[2, 0]
    innovation_ax.plot(
        update_time,
        df.loc[update_mask, "innovation_norm"] * 1000.0,
        linewidth=1.0,
        label="innovation norm",
    )
    innovation_ax.set_title("Camera innovation before correction")
    innovation_ax.set_ylabel("|camera − prediction| [mm]")
    innovation_ax.grid(alpha=0.25)
    innovation_ax.legend()

    gain_ax = axes[2, 1]
    gain_ax.plot(
        update_time,
        df.loc[update_mask, "gain_position_norm"],
        label="position block",
    )
    gain_ax.plot(
        update_time,
        df.loc[update_mask, "gain_velocity_norm"],
        label="velocity block",
    )
    gain_ax.plot(
        update_time,
        df.loc[update_mask, "gain_bias_norm"],
        label="bias block",
    )
    gain_ax.set_title("Kalman Gain block norms")
    gain_ax.set_ylabel("Frobenius norm")
    gain_ax.grid(alpha=0.25)
    gain_ax.legend()

    for ax in axes[2, :]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(f"{SELECTED_DATASET}: filter diagnostics", fontsize=14)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{SELECTED_DATASET}_filter_diagnostics.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Filter diagnostic plot saved: {output_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


def plot_camera_vs_gt(df):
    """Compare original 30 Hz camera measurements directly against GT."""

    if "ts" in df.columns:
        time = df["ts"].to_numpy() - df["ts"].iloc[0]
    else:
        time = np.arange(len(df)) / SAMPLE_RATE

    update_mask = df["camera_update"].to_numpy(dtype=bool)
    camera_time = time[update_mask]
    gt = df[[f"gt_pos_{axis}" for axis in "xyz"]].to_numpy()
    camera = df[[f"native_camera_pos_{axis}" for axis in "xyz"]].to_numpy()
    camera_error = camera[update_mask] - gt[update_mask]
    camera_distance_error = np.linalg.norm(camera_error, axis=1)
    camera_rmse_3d_mm = float(
        np.sqrt(np.mean(camera_distance_error**2)) * 1000.0
    )

    # Keep the metrics based on untouched data, but display sub-micrometre
    # floating-point/interpolation residue as exact zero on the chart.
    gt_mm = gt * 1000.0
    camera_mm = camera * 1000.0
    gt_mm[np.abs(gt_mm) < 0.001] = 0.0
    camera_mm[np.abs(camera_mm) < 0.001] = 0.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for index, (axis_name, ax) in enumerate(zip("XYZ", axes)):
        ax.plot(
            time,
            gt_mm[:, index],
            label="GT",
            color="black",
            linewidth=2.0,
        )
        ax.scatter(
            camera_time,
            camera_mm[update_mask, index],
            label="native camera 30 Hz",
            color="tab:green",
            s=7,
            alpha=0.65,
        )
        ax.set_title(f"Position {axis_name}")
        ax.set_ylabel("Position [mm]")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        visible_values = np.concatenate(
            [gt_mm[:, index], camera_mm[update_mask, index]]
        )
        if np.max(np.abs(visible_values)) < 0.1:
            ax.set_ylim(-0.1, 0.1)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(
        f"{SELECTED_DATASET}: native camera 30 Hz vs GT\n"
        f"3D RMSE = {camera_rmse_3d_mm:.2f} mm",
        fontsize=14,
    )
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{SELECTED_DATASET}_camera_vs_gt.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Camera vs GT 3D RMSE: {camera_rmse_3d_mm:.2f} mm")
    print(f"Camera vs GT plot saved: {output_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


def print_metrics(metrics):
    print(f"\nDataset: {SELECTED_DATASET}")
    print(f"IMU Wiener(81) 3D RMSE: {metrics['imu_rmse_3d_mm']:.2f} mm")
    print(f"IMU + camera 3D RMSE:   {metrics['fused_rmse_3d_mm']:.2f} mm")
    print("\nPer-axis RMSE [mm]:")
    for index, axis in enumerate("XYZ"):
        print(
            f"{axis}: IMU={metrics['imu_axis_rmse_mm'][index]:.2f}, "
            f"IMU+camera={metrics['fused_axis_rmse_mm'][index]:.2f}"
        )


def main():
    df = load_and_prepare_data()
    comparison = calculate_position_metrics(df)
    gt, imu, fused, imu_error, fused_error, metrics = comparison
    print_metrics(metrics)
    plot_comparison(df, gt, imu, fused, imu_error, fused_error, metrics)
    plot_gt_vs_fused(df, gt, fused, metrics)
    plot_filter_diagnostics(df)
    plot_camera_vs_gt(df)


if __name__ == "__main__":
    main()
