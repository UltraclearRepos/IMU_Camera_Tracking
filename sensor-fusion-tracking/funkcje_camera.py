import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple



def load_camera_data(file_path):
    """
    Wczytuje dane z kamery, konwertuje jednostki i zwraca DataFrame.
    """
    df_raw = pd.read_csv(file_path)
    
    # Konwersja jednostek mm -> m
    for col in ['X', 'Y', 'Z']:
        df_raw[col] = df_raw[col] / 1000.0
    
    # Zamiana osi Y na Z
    df_raw.rename(columns={'Y': 'Z', 'Z': 'Y'}, inplace=True)

    # Odwrócenie kierunku osi X, Z
    df_raw['X'] = -df_raw['X']
    df_raw['Z'] = -df_raw['Z']
    
    return df_raw


def resample_camera_data(df_raw, target_fps=100.0, current_fps=30.0):
    """
    Resampling do stalego FPS i interpolacja liniowa.
    Przyjmuje surowe dane z kolumna Frame albo dane juz po resamplingu z timestamp.
    """
    if 'Frame' in df_raw.columns:
        source_time = df_raw['Frame'].values / current_fps
    elif 'timestamp' in df_raw.columns:
        source_time = df_raw['timestamp'].values
    else:
        raise ValueError("Brak kolumny czasu w danych kamery: oczekiwano 'Frame' albo 'timestamp'.")

    t_start = float(source_time[0])
    t_end = float(source_time[-1])
    t_uniform = np.arange(t_start, t_end, 1.0 / target_fps)
    df_resampled = pd.DataFrame({'timestamp': t_uniform})

    for col in ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']:
        f_interp = interp1d(source_time, df_raw[col], kind='linear')
        df_resampled[col] = f_interp(t_uniform)

    return df_resampled

def trim_camera_data(df, trim_start=100, trim_end=100):
    """
    Usuwa trim_start pierwszych próbek i trim_end ostatnich próbek z df, resetuje indeksy.
    """
    if trim_start < 0 or trim_end < 0:
        raise ValueError("trim_start i trim_end muszą być nieujemne.")
    if trim_start + trim_end >= len(df):
        raise ValueError("Suma trim_start i trim_end musi być mniejsza niż liczba próbek w df.")
    
    df_trimmed = df.iloc[trim_start:len(df)-trim_end].reset_index(drop=True)
    return df_trimmed

def plot_camera_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['X'], label='Camera Pos X', alpha=0.7)
    plt.plot(df['timestamp'], df['Y'], label='Camera Pos Y', alpha=0.7)
    plt.plot(df['timestamp'], df['Z'], label='Camera Pos Z', alpha=0.7)
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja Camera [m]')
    plt.title('Pozycja Camera w 3 osiach')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['Roll'], label='Camera Roll', alpha=0.7)
    plt.plot(df['timestamp'], df['Pitch'], label='Camera Pitch', alpha=0.7)
    plt.plot(df['timestamp'], df['Yaw'], label='Camera Yaw', alpha=0.7)
    plt.xlabel('Czas [s]')
    plt.ylabel('Orientacja Camera [deg]')
    plt.title('Orientacja Camera w 3 osiach')
    plt.legend()
    plt.show()


def synchronize_imu_camera(df_imu: pd.DataFrame, df_camera: pd.DataFrame,
                           imu_prefix: str = 'gl',
                           cam_cols: List[str] = ['X', 'Y', 'Z'],
                           target_fps: float = 100.0,
                           plot_data: bool = False) -> pd.DataFrame:
    """
    Synchronizuje dane IMU i Kamery na podstawie prędkości z 3 osi (X, Y, Z).
    
    IMU: wykorzystuje kolumny {imu_prefix}_vel_x, _y, _z
    Kamera: oblicza prędkość z pozycji X, Y, Z
    """
    dt = 1.0 / target_fps
    
    # Przygotowanie prędkości dla kamery
    v_cam_axes = []
    for col in cam_cols:
        v_cam_axes.append(np.gradient(df_camera[col].values, dt))
    # Przygotowanie przyśpieszenia dla kamery
    acc_cam_axes = []
    for v in v_cam_axes:
        acc_cam_axes.append(np.gradient(v, dt))
    
    sig_cam = np.sqrt(np.sum(np.square(acc_cam_axes), axis=0))
    
    # v_imu_axes = []
    # for ax in ['x', 'y', 'z']:
    #     col_name = f"{imu_prefix}_vel_{ax}"
    #     if col_name in df_imu.columns:
    #         v_imu_axes.append(df_imu[col_name].values)
    #     else:
    #         raise ValueError(f"Brak kolumny {col_name} w df_imu")
        
    acc_imu_axes = []
    for ax in ['x', 'y', 'z']:
        col_name = f"{imu_prefix}_acc_{ax}"
        if col_name in df_imu.columns:
            acc_imu_axes.append(df_imu[col_name].values)
        else:
            raise ValueError(f"Brak kolumny {col_name} w df_imu")
            
    sig_imu = np.sqrt(np.sum(np.square(acc_imu_axes), axis=0))
    
    # Korelacja krzyżowa
    sig_imu_centered = sig_imu - np.mean(sig_imu)
    sig_cam_centered = sig_cam - np.mean(sig_cam)
    
    cross_corr = correlate(sig_imu_centered, sig_cam_centered, mode='same')
    
    # Obliczenie przesunięcia (offset)
    offset = np.argmax(cross_corr) - len(sig_cam_centered) // 2
    
    # Wykres korelacji
    print(f"=== Synchronizacja przez korelację krzyżową ===")
    print(f"Obliczone przesunięcie: {offset} próbek")


    if plot_data:
        plt.figure(figsize=(12, 6))
        plt.plot(cross_corr, label='Cross-Correlation', color='steelblue')
        
        # Prawdziwy pik korelacji
        peak_idx = np.argmax(cross_corr)
        
        plt.axvline(x=peak_idx, color='r', linestyle='--', label='Pik korelacji (Maksimum)')
        
        plt.title('Korelacja krzyżowa między IMU i Kamerą')
        plt.xlabel('Indeks tablicy')
        plt.ylabel('Wartość korelacji')
        plt.legend()
        plt.grid()
        plt.show()
    
    # Synchronizacja
    if offset > 0:
        sync_imu = df_imu.iloc[offset:].reset_index(drop=True)
        sync_cam = df_camera.reset_index(drop=True)
    else:
        sync_cam = df_camera.iloc[abs(offset):].reset_index(drop=True)
        sync_imu = df_imu.reset_index(drop=True)
    
    min_len = min(len(sync_imu), len(sync_cam))
    sync_imu = sync_imu.iloc[:min_len].copy()
    sync_cam = sync_cam.iloc[:min_len]
    
    # Dodanie zsynchronizowanych kolumn kamery do DataFrame IMU
    for col in cam_cols:
        sync_imu[f'cam_pos_{col.lower()}'] = sync_cam[col].values
        
    for rot in ['Roll', 'Pitch', 'Yaw']:
        if rot in sync_cam.columns:
            sync_imu[f'cam_{rot.lower()}'] = sync_cam[rot].values
            
    if 'timestamp' in sync_cam.columns:
        sync_imu['cam_timestamp'] = sync_cam['timestamp'].values
    
    return sync_imu


def plot_sync_IMU_camera_data(df, imu_prefix: str = 'gl', cam_prefix: str = 'cam'):
    # Os X
    plt.figure(figsize=(12, 6))
    plt.plot(df['ts'], df[f'{imu_prefix}_pos_x'], label='IMU Pos X', alpha=0.7)
    plt.plot(df['ts'], df[f'{cam_prefix}_pos_x'], label='Camera Pos X', alpha=0.7)
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja X [m]')
    plt.title('Synchronizacja IMU i Kamery - Oś X')
    plt.grid()
    plt.legend()
    plt.show()

    # Os Y
    plt.figure(figsize=(12, 6))
    plt.plot(df['ts'], df[f'{imu_prefix}_pos_y'], label='IMU Pos Y', alpha=0.7)
    plt.plot(df['ts'], df[f'{cam_prefix}_pos_y'], label='Camera Pos Y', alpha=0.7)
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja Y [m]')
    plt.title('Synchronizacja IMU i Kamery - Oś Y')
    plt.grid()
    plt.legend()
    plt.show()

    # Os Z
    plt.figure(figsize=(12, 6))
    plt.plot(df['ts'], df[f'{imu_prefix}_pos_z'], label='IMU Pos Z', alpha=0.7)
    plt.plot(df['ts'], df[f'{cam_prefix}_pos_z'], label='Camera Pos Z', alpha=0.7)
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja Z [m]')
    plt.title('Synchronizacja IMU i Kamery - Oś Z')
    plt.grid()
    plt.legend()
    plt.show()


def plot_IMU_camera_fusion_results(df, imu_prefix: str = 'gl', cam_prefix: str = 'cam', fused_prefix: str = 'fused'):
    max_val = 0.0
    for axis in ['x', 'y', 'z']:
        max_val = max(
            max_val,
            df[f'{imu_prefix}_pos_{axis}'].abs().max(),
            df[f'{cam_prefix}_pos_{axis}'].abs().max(),
            df[f'{fused_prefix}_pos_{axis}'].abs().max(),
            df[f'gt_pos_{axis}'].abs().max()
        )

    # Os X
    plt.figure(figsize=(12, 6))
    plt.plot(df['ts'], df[f'{imu_prefix}_pos_x'], label='IMU Pos X', alpha=0.7)
    plt.plot(df['ts'], df[f'{cam_prefix}_pos_x'], label='Camera Pos X', alpha=0.7)
    plt.plot(df['ts'], df[f'{fused_prefix}_pos_x'], label='Fused Pos X', alpha=0.9, linewidth=1.5)
    plt.plot(df['ts'], df['gt_pos_x'], label='GT Pos X', alpha=0.9, linewidth=1.5)
    plt.ylim(-1.2 * max_val, 1.2 * max_val)
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja X [m]')
    plt.title('Fuzja IMU i Kamery - Oś X')
    plt.grid()
    plt.legend()
    plt.show()

    # Os Y
    plt.figure(figsize=(12, 6))
    plt.plot(df['ts'], df[f'{imu_prefix}_pos_y'], label='IMU Pos Y', alpha=0.7)
    plt.plot(df['ts'], df[f'{cam_prefix}_pos_y'], label='Camera Pos Y', alpha=0.7)
    plt.plot(df['ts'], df[f'{fused_prefix}_pos_y'], label='Fused Pos Y', alpha=0.9, linewidth=1.5)
    plt.plot(df['ts'], df['gt_pos_y'], label='GT Pos Y', alpha=0.9, linewidth=1.5)
    plt.ylim(-1.2 * max_val, 1.2 * max_val)
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja Y [m]')
    plt.title('Fuzja IMU i Kamery - Oś Y')
    plt.grid()
    plt.legend()
    plt.show()

    # Os Z
    plt.figure(figsize=(12, 6))
    plt.plot(df['ts'], df[f'{imu_prefix}_pos_z'], label='IMU Pos Z', alpha=0.7)
    plt.plot(df['ts'], df[f'{cam_prefix}_pos_z'], label='Camera Pos Z', alpha=0.7)
    plt.plot(df['ts'], df[f'{fused_prefix}_pos_z'], label='Fused Pos Z', alpha=0.9, linewidth=1.5)
    plt.plot(df['ts'], df['gt_pos_z'], label='GT Pos Z', alpha=0.9, linewidth=1.5)
    plt.ylim(-1.2 * max_val, 1.2 * max_val)
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja Z [m]')
    plt.title('Fuzja IMU i Kamery - Oś Z')
    plt.grid()
    plt.legend()
    plt.show()


def apply_camera_confidence(df, cam_prefix='cam', noise_type=True, conf_intervals: List[Tuple[float, float]] = []):
    """
    Dodaje kolumnę 'conf' do danych, która wskazuje na jakość danych z kamery.
    
    Args:
        df: DataFrame z danymi (musi zawierać kolumnę 'ts' oraz kolumny pozycji kamery)
        cam_prefix: prefiks dla kolumn kamery (domyślnie 'cam')
        noise_type: True - dodaje szum, False - zeruje przesunięcie (kamera stoi w miejscu)
        conf_intervals: lista przedziałów czasowych [(start, end), ...] gdzie zaufanie jest mniejsze
    """
    df[f'{cam_prefix}_conf'] = 1.0
    
    for start, end in conf_intervals:
        mask = (df['ts'] >= start) & (df['ts'] <= end)
        df.loc[mask, f'{cam_prefix}_conf'] = 0.1
    
    if noise_type:
        # Opcja 1: Dodanie szumu do danych z kamery w przedziałach z niskim zaufaniem
        for axis in ['x', 'y', 'z']:
            col_name = f'{cam_prefix}_pos_{axis}'
            low_conf_mask = df[f'{cam_prefix}_conf'] < 1.0
            noise = np.random.normal(0, 0.03, size=low_conf_mask.sum())
            df.loc[low_conf_mask, col_name] += noise
            
    else:
        # Opcja 2: Zerowanie przesunięcia (kamera "stoi w miejscu" w przedziałach)
        for start, end in conf_intervals:
            mask = (df['ts'] >= start) & (df['ts'] <= end)
            
            for axis in ['x', 'y', 'z']:
                col_name = f'{cam_prefix}_pos_{axis}'
                before_mask = df['ts'] < start
                after_mask = df['ts'] > end
                
                if before_mask.any() and after_mask.any():
                    value_before = df.loc[before_mask, col_name].iloc[-1]
                    value_after = df.loc[after_mask, col_name].iloc[0]
                    diff_to_correct = value_after - value_before

                    df.loc[mask, col_name] = value_before
                    df.loc[after_mask, col_name] -= diff_to_correct
                    
    return df
                
           
def calculate_final_metrics(df, gt_prefix='gt', method_prefix='fused'):
    # Pobieramy dane Ground Truth
    gt_cols = [f'{gt_prefix}_pos_{ax}' for ax in ['x', 'y', 'z']]
    method_cols = [f'{method_prefix}_pos_{ax}' for ax in ['x', 'y', 'z']]
    
    if not all(col in df.columns for col in gt_cols):
        raise ValueError(f"Brak kolumn Ground Truth w DataFrame: {gt_cols}")
    if not all(col in df.columns for col in method_cols):
        raise ValueError(f"Brak kolumn metody {method_prefix} w DataFrame: {method_cols}")
        
    gt_values = df[gt_cols].values
    method_values = df[method_cols].values
    
    # Obliczamy błędy pozycji dla każdej próbki względem 1 sekundy (100 próbek)
    # czyli gt[i] - gt[i-100] i method[i] - method[i-100], a następnie błąd euklidesowy tych różnic
    drift_errors_1s = []
    for i in range(100, len(df), 100):
        gt_diff = gt_values[i] - gt_values[i-100]
        method_diff = method_values[i] - method_values[i-100]
        drift_errors_1s.append(np.abs(gt_diff - method_diff))

    drift_errors_10s = []
    for i in range(1000, len(df), 1000):
        gt_diff = gt_values[i] - gt_values[i-1000]
        method_diff = method_values[i] - method_values[i-1000]
        drift_errors_10s.append(np.abs(gt_diff - method_diff))

    # Metryki na 1 sekundę
    rmse_drift_per_1s = np.sqrt(np.mean(np.square(drift_errors_1s)))
    mean_drift_error_per_1s = np.mean(drift_errors_1s)
    median_drift_error_per_1s = np.median(drift_errors_1s)    

    # Metryki na 10 sekund
    rmse_drift_per_10s = np.sqrt(np.mean(np.square(drift_errors_10s)))
    mean_drift_error_per_10s = np.mean(drift_errors_10s)
    median_drift_error_per_10s = np.median(drift_errors_10s)
    
    return {
        'rmse_drift_per_1s': rmse_drift_per_1s * 1000,
        'mean_drift_error_per_1s': mean_drift_error_per_1s * 1000,
        'median_drift_error_per_1s': median_drift_error_per_1s * 1000,
        'rmse_drift_per_10s': rmse_drift_per_10s * 1000,
        'mean_drift_error_per_10s': mean_drift_error_per_10s * 1000,
        'median_drift_error_per_10s': median_drift_error_per_10s * 1000
    }


def print_final_metrics(metrics_dict, method_name='fused'):
    print(f"Metryki dryfu na 1s:")
    print(f"RMSE: {metrics_dict['rmse_drift_per_1s']:.2f} mm")
    print(f"Średni błąd: {metrics_dict['mean_drift_error_per_1s']:.2f} mm")
    print(f"Mediana błędu: {metrics_dict['median_drift_error_per_1s']:.2f} mm")
    print(f"\nMetryki dryfu na 10s:")
    print(f"RMSE: {metrics_dict['rmse_drift_per_10s']:.2f} mm")
    print(f"Średni błąd: {metrics_dict['mean_drift_error_per_10s']:.2f} mm")
    print(f"Mediana błędu: {metrics_dict['median_drift_error_per_10s']:.2f} mm")
