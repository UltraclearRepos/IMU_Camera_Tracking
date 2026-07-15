import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
try:
    from final_code.funkcje_IMU_GT import cross_correlation
except ModuleNotFoundError:
    from funkcje_IMU_GT import cross_correlation


def load_ground_truth(file_path):
    """
    Wczytuje logi, zamienia osie.
    """
    df_raw = pd.read_csv(file_path)

    # Konwersja jednostek mm -> m
    for col in ['X', 'Y', 'Z']:
        df_raw[col] = df_raw[col] / 1000.0

    # Zamiana osi X na Y
    # df_raw.rename(columns={'X': 'Y', 'Y': 'X'}, inplace=True)

    return df_raw

def normalize_ground_truth(df_gt):
    """
    Normalizuje dane GT, przesuwając początek pomiaru do punktu (0,0,0).
    """
    for axis in ['X', 'Y', 'Z']:
        df_gt[axis] = df_gt[axis] - df_gt[axis].iloc[0]
    
    # df_gt['X'] = -df_gt['X']
    return df_gt

def resample_ground_truth(df_raw, target_fps=100.0):
    """
    Resampling do stałego FPS i interpolacja liniowa.
    """
    # Tworzymy nową, równą oś czasu od początku do końca nagrania
    t_start = df_raw['timestamp'].iloc[0]
    t_end = df_raw['timestamp'].iloc[-1]
    t_uniform = np.arange(t_start, t_end, 1.0 / target_fps)
    
    df_resampled = pd.DataFrame({'timestamp': t_uniform})
    
    # Interpolacja liniowa pozycji na nową oś czasu
    for col in ['X', 'Y', 'Z', 'R']:
        f_interp = interp1d(df_raw['timestamp'], df_raw[col], kind='linear')
        df_resampled[col] = f_interp(t_uniform)
    
    return df_resampled

def calculate_derivatives(df_gt, savgol_window_length=21, median_size=5, target_fps=100.0):
    """
    Obliczanie prędkości i przyspieszenia z pozycji.
    """
    dt = 1.0 / target_fps
    
    for axis in ['X', 'Y', 'Z']:
        # Wygładzamy pozycję przed różniczkowaniem (usuwa szum kwantyzacji enkoderów)
        pos_smoothed = savgol_filter(df_gt[axis], window_length=savgol_window_length, polyorder=3)
        pos_smoothed = median_filter(pos_smoothed, size=median_size)
        
        # Prędkość v = ds/dt
        vel = np.gradient(pos_smoothed, dt)
        vel_smoothed = savgol_filter(vel, window_length=savgol_window_length, polyorder=3)
        vel_smoothed = median_filter(vel_smoothed, size=median_size)
        df_gt[f'gt_vel_{axis.lower()}'] = vel_smoothed
        
        # Przyspieszenie a = dv/dt
        acc = np.gradient(vel_smoothed, dt)
        # Finalne wygładzenie przyspieszenia
        acc_smoothed = savgol_filter(acc, window_length=savgol_window_length, polyorder=3)
        acc_smoothed = median_filter(acc_smoothed, size=median_size)
        df_gt[f'gt_acc_{axis.lower()}'] = acc_smoothed
        
    return df_gt

def trim_ground_truth(df_gt, trim_start=100, trim_end=100):
    """
    Usuwa trim_start pierwszych próbek i trim_end ostatnich próbek z df_gt, resetuje indeksy.
    """
    if trim_start < 0 or trim_end < 0:
        raise ValueError("trim_start i trim_end muszą być nieujemne.")
    if trim_start + trim_end >= len(df_gt):
        raise ValueError("Suma trim_start i trim_end musi być mniejsza niż liczba próbek w df_gt.")
    
    df_trimmed = df_gt.iloc[trim_start:len(df_gt)-trim_end].reset_index(drop=True)
    return df_trimmed


def plot_ground_truth(df_gt):
    # Wykres przyśpieszenia GT w 3 osiach
    plt.figure(figsize=(12, 6))
    plt.plot(df_gt['timestamp'], df_gt['gt_acc_x'], label='GT Acc X', alpha=0.7)
    plt.plot(df_gt['timestamp'], df_gt['gt_acc_y'], label='GT Acc Y', alpha=0.7)
    plt.plot(df_gt['timestamp'], df_gt['gt_acc_z'], label='GT Acc Z', alpha=0.7)
    plt.xlabel('Czas [s]')
    plt.ylabel('Przyspieszenie GT [m/s^2]')
    plt.title('Przyspieszenie GT w 3 osiach')
    plt.legend()
    plt.show()

    # Wykres prędkości GT w 3 osiach
    plt.figure(figsize=(12, 6))
    plt.plot(df_gt['timestamp'], df_gt['gt_vel_x'], label='GT Vel X', alpha=0.7)
    plt.plot(df_gt['timestamp'], df_gt['gt_vel_y'], label='GT Vel Y', alpha=0.7)
    plt.plot(df_gt['timestamp'], df_gt['gt_vel_z'], label='GT Vel Z', alpha=0.7)
    plt.xlabel('Czas [s]')
    plt.ylabel('Prędkość GT [m/s]')
    plt.title('Prędkość GT w 3 osiach')
    plt.legend()
    plt.show()

    # Wykres pozycję GT w 3 osiach
    plt.figure(figsize=(12, 6))
    plt.plot(df_gt['timestamp'], df_gt['X'], label='GT Pos X', alpha=0.7)
    plt.plot(df_gt['timestamp'], df_gt['Y'], label='GT Pos Y', alpha=0.7)
    plt.plot(df_gt['timestamp'], df_gt['Z'], label='GT Pos Z', alpha=0.7)
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja GT [m]')
    plt.title('Pozycja GT w 3 osiach')
    plt.legend()
    plt.show()


def check_best_parameters_for_derivatives(df_imu, df_gt):
    tab_results = []
    filter_params = [(w, s) for w in range(5, 46, 2) for s in range(1, 46, 2)]
    for w, s in filter_params:
        print(f"Testowanie parametrów: window={w}, size={s}")
        df_new = calculate_derivatives(df_gt, w, s, target_fps=100.0)
        corr = cross_correlation(df_imu, df_new)
        tab_results.append((w, s, corr))

    # find max correlation
    best_params = max(tab_results, key=lambda x: x[2])
    print(f"Najlepsze parametry: window={best_params[0]}, size={best_params[1]}, correlation={best_params[2]:.4f}\n")

    w_values = [res[0] for res in tab_results]
    s_values = [res[1] for res in tab_results]
    correlations = [res[2] for res in tab_results]

    plt.figure(figsize=(12, 6))
    plt.scatter(w_values[2:], s_values[2:], c=correlations[2:], cmap='viridis', s=100)
    plt.colorbar(label='Korelacja')
    plt.xlabel('Polynomial Window (w)')
    plt.ylabel('Median Size (s)')
    plt.title('Korelacja między IMU a GT dla różnych parametrów filtra')
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist([res[2] for res in tab_results], color='green', bins=100, alpha=0.7)
    plt.title('Histogram korelacji dla różnych parametrów')
    plt.xlabel('Korelacja')
    plt.ylabel('Liczba wystąpień')
    plt.grid()
    plt.show()
