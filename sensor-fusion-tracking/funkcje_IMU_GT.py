import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from typing import List, Tuple


def cross_correlation(df_imu: pd.DataFrame, df_gt: pd.DataFrame,
                      axes: List[str] = ['x', 'y', 'z'],
                      use_magnitude: bool = True,
                      max_lag: int = 500) -> float:
    """
    Funkcja obliczająca maksymalną wartość korelacji Pearsona 
    między sygnałami po ich optymalnym dopasowaniu czasowym.
    """
    
    # Przygotowanie sygnałów
    imu_data = np.array([df_imu[f'gl_acc_{axis}'].values for axis in axes])
    gt_data = np.array([df_gt[f'gt_acc_{axis}'].values for axis in axes])

    if use_magnitude:
        sig_imu = np.sqrt(np.sum(np.square(imu_data), axis=0))
        sig_gt = np.sqrt(np.sum(np.square(gt_data), axis=0))
    else:
        sig_imu = np.sum(np.abs(imu_data), axis=0)
        sig_gt = np.sum(np.abs(gt_data), axis=0)

    # Korelacja krzyżowa
    cross_corr = correlate(sig_imu, sig_gt, mode='same')
    offset = np.argmax(cross_corr) - len(sig_gt) // 2

    # Przycięcie sygnałów
    if offset > 0:
        s_imu = sig_imu[offset:]
        s_gt = sig_gt
    else:
        s_imu = sig_imu
        s_gt = sig_gt[abs(offset):]

    # Wyrównanie długości
    min_len = min(len(s_imu), len(s_gt))
    if min_len < 2:
        return 0.0
        
    s_imu = s_imu[:min_len]
    s_gt = s_gt[:min_len]

    # Obliczenie końcowej korelacji Pearsona
    return float(np.corrcoef(s_imu, s_gt)[0, 1])

def synchronize_by_cross_correlation(df_imu: pd.DataFrame, df_gt: pd.DataFrame,
                                     axes: List[str] = ['x', 'y', 'z'],
                                     use_magnitude: bool = True,
                                     plot_data: bool = False) -> pd.DataFrame:
    """
    Metoda synchronizacji wykorzystująca korelację krzyżową.
    """
    # Przygotowanie sygnałów
    imu_data = np.array([df_imu[f'gl_acc_{axis}'].values for axis in axes])
    gt_data = np.array([df_gt[f'gt_acc_{axis}'].values for axis in axes])

    if use_magnitude:        
        sig_imu = np.sqrt(np.sum(np.square(imu_data), axis=0))
        sig_gt = np.sqrt(np.sum(np.square(gt_data), axis=0))
    else:
        sig_imu = np.sum([np.abs(df_imu[f'gl_acc_{axis}'].values) for axis in axes], axis=0)
        sig_gt = np.sum([np.abs(df_gt[f'gt_acc_{axis}'].values) for axis in axes], axis=0)
    
    # Korelacja krzyżowa
    cross_corr = correlate(sig_imu, sig_gt, mode='same')
    offset = np.argmax(cross_corr) - len(sig_gt) // 2
    
    print(f"=== Synchronizacja przez korelację krzyżową ===")
    print(f"Obliczone przesunięcie: {offset} próbek")

    if plot_data:
        plt.figure(figsize=(12, 6))
        plt.plot(cross_corr, label='Cross-Correlation')
        plt.axvline(x=len(cross_corr)//2 + offset, color='r', linestyle='--', label='Detected Offset')
        plt.title('Korelacja krzyżowa między sygnałami IMU i GT')
        plt.xlabel('Lag (próbki)')
        plt.ylabel('Korelacja')
        plt.legend()
        plt.grid()
        plt.show()

    # Synchronizacja
    if offset > 0:
        sync_imu = df_imu.iloc[offset:].reset_index(drop=True)
        sync_gt = df_gt.reset_index(drop=True)
    else:
        sync_gt = df_gt.iloc[abs(offset):].reset_index(drop=True)
        sync_imu = df_imu.reset_index(drop=True)
    
    min_len = min(len(sync_imu), len(sync_gt))
    sync_imu = sync_imu.iloc[:min_len].copy()
    sync_gt = sync_gt.iloc[:min_len]
    
    # Dodanie kolumn GT
    for axis in axes:
        sync_imu[f'gt_acc_{axis}'] = sync_gt[f'gt_acc_{axis}'].values
        sync_imu[f'gt_vel_{axis}'] = sync_gt[f'gt_vel_{axis}'].values
        sync_imu[f'gt_pos_{axis}'] = sync_gt[axis.upper()].values
    
    if 'Timestamp' in df_gt.columns:
        sync_imu['gt_timestamp'] = sync_gt['Timestamp'].values
    
    return sync_imu


def plot_comparison_IMU_ground_truth_data(df_IMU, df_gt, column_name_IMU, column_name_gt):
    """
    Wizualizacja dany z IMU i Ground Truth dla porównania (np. prędkość lub pozycja).
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for i, axis in enumerate(['x', 'y', 'z']):
        col_IMU = f"{column_name_IMU}_{axis}"
        col_gt = f"{column_name_gt}_{axis}"
        if col_IMU in df_IMU.columns and col_gt in df_gt.columns:
            axes[i].plot(df_IMU['ts'], df_IMU[col_IMU], label=f'IMU {axis.upper()}', color=['red', 'green', 'blue'][i], alpha=0.7)
            axes[i].plot(df_gt['ts'], df_gt[col_gt], label=f'GT {axis.upper()}', color=['darkred', 'darkgreen', 'darkblue'][i], alpha=0.7)
            axes[i].set_ylabel(f'{column_name_IMU} {axis.upper()}')
            axes[i].set_title(f'Porównanie: IMU vs Ground Truth - {axis.upper()}')
            axes[i].set_xlabel('Czas [s]')
            axes[i].set_ylabel('Wartość')
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_velocity_and_position(df_sync, method='gl', axes=['x', 'y', 'z']):
    time_s = df_sync['ts'] - df_sync['ts'].iloc[0]
    max_val = 0.0
    for axis in axes:
        max_val = max(
            max_val,
            df_sync[f'gt_vel_{axis}'].abs().max(),
            df_sync[f'{method}_vel_{axis}'].abs().max() if f'{method}_vel_{axis}' in df_sync.columns else 0
        )

    for axis in ['x', 'y', 'z']:
        plt.figure(figsize=(14, 6))
        plt.plot(time_s, df_sync[f'gt_vel_{axis}'], label='Prędkość (gt)', color='blue')
        plt.plot(time_s, df_sync[f'{method}_vel_{axis}'], label=f'Prędkość ({method})', color='orange')
        plt.title(f'Prędkość - Oś {axis.upper()}')
        plt.ylim(-1.2 * max_val, 1.2 * max_val)
        plt.xlabel('Czas [s]')
        plt.ylabel('Wartość')
        plt.legend()
        plt.grid()
        plt.show()

    max_val = 0.0
    for axis in axes:
        max_val = max(
            max_val,
            df_sync[f'gt_pos_{axis}'].abs().max(),
            df_sync[f'{method}_pos_{axis}'].abs().max() if f'{method}_pos_{axis}' in df_sync.columns else 0
        )

    for axis in ['x', 'y', 'z']:
        plt.figure(figsize=(14, 6))
        plt.plot(time_s, df_sync[f'gt_pos_{axis}'], label='Pozycja (gt)', color='blue')
        plt.plot(time_s, df_sync[f'{method}_pos_{axis}'], label=f'Pozycja ({method})', color='orange')
        plt.title(f'Pozycja - Oś {axis.upper()}')
        plt.ylim(-1.2 * max_val, 1.2 * max_val)
        plt.xlabel('Czas [s]')
        plt.ylabel('Wartość')
        plt.legend()
        plt.grid()
        plt.show()


def plot_synchronized_results(df_sync, axes=['x', 'y', 'z'], filter_name='butterworth'):
    """
    Rysuje wykres porównawczy na podstawie zsynchronizowanego DataFrame.
    Porównuje: gl_acc_{axis} (surowe dane z IMU), gt_acc_{axis} (Ground Truth) oraz {filter_name}_{axis} (dane po filtrze).
    """

    max_val = 0.0
    for axis in axes:
        max_val = max(
            max_val,
            df_sync[f'gt_acc_{axis}'].abs().max(),
            df_sync[f'gl_acc_{axis}'].abs().max(),
            df_sync[f'{filter_name}_acc_{axis}'].abs().max() if f'{filter_name}_acc_{axis}' in df_sync.columns else 0
        )

    for axis in axes:
        plt.figure(figsize=(14, 7))
        time_s = np.arange(len(df_sync)) * 0.01 
        
        # 1. Ground Truth (z robota)
        plt.plot(time_s, df_sync[f'gt_acc_{axis}'], color='black', label='Ground Truth (Robot)', linewidth=1.2)
        
        # 2. Surowe dane z IMU
        plt.plot(time_s, df_sync[f'gl_acc_{axis}'], color='gray', label='IMU', alpha=0.4, linewidth=0.8)
        
        # 3. Wybrany filtr
        filter_col = f'{filter_name}_acc_{axis}'
        if filter_col in df_sync.columns:
            plt.plot(time_s, df_sync[filter_col], color='red', label=f'Filtered ({filter_name})', linewidth=1)
        
        plt.ylim(-1.2 * max_val, 1.2 * max_val)
        plt.title(f'Zsynchronizowane przyspieszenie - Oś {axis.upper()}')
        plt.xlabel('Czas od startu ruchu [s]')
        plt.ylabel('Przyspieszenie [m/s²]')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def calculate_rmse_vs_gt(df_sync: pd.DataFrame, 
                         axes: List[str] = ['x', 'y', 'z']) -> List[Tuple[str, float]]:
    """
    Oblicza całkowite RMSE dla wszystkich filtrów oraz danych surowych 
    względem Ground Truth (gt_acc_x, gt_acc_y, gt_acc_z).
    Zwraca listę krotek (nazwa_metody, wartość_rmse) posortowaną od najlepszego wyniku.
    """
    gt_cols = [f'gt_acc_{axis}' for axis in axes]
    if not all(col in df_sync.columns for col in gt_cols):
        raise ValueError(f"Brak kolumn Ground Truth w DataFrame: {gt_cols}")
        
    gt_values = df_sync[gt_cols].values
    
    # Identyfikujemy unikalne prefiksy metod
    method_prefixes = [
        col.replace('_acc_x', '') 
        for col in df_sync.columns 
        if col.endswith('_acc_x') and not col.startswith('gt_')
    ]
    
    results = {}

    # Obliczamy RMSE dla każdej metody
    for method in method_prefixes:
        method_cols = [f'{method}_acc_{axis}' for axis in axes]
        
        # Sprawdzamy czy metoda ma komplet osi w df
        if all(col in df_sync.columns for col in method_cols):
            method_values = df_sync[method_cols].values
            
            squared_diff = (gt_values - method_values) ** 2
            rmse = np.sqrt(np.mean(squared_diff))
            
            display_name = "Raw (gl_acc)" if method == "gl" else method
            results[display_name] = rmse

    # Sortujemy wyniki (im mniejsze RMSE, tym lepiej)
    return sorted(results.items(), key=lambda x: x[1])
    