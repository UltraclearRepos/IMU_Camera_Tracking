import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from scipy.signal import butter, filtfilt, savgol_filter, wiener
from scipy.ndimage import median_filter


class IMUFilter:
    def __init__(self, fs=100.0):
        self.fs = fs
        self.filters = {}
        
    # 1. Filtr Butterwortha
    def butterworth_filter(self, data, cutoff=5.0, order=4):
        """
        Filtr dolnoprzepustowy Butterwortha
        cutoff: częstotliwość odcięcia w Hz
        order: rząd filtra
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    
    # 2. Filtr Savitzky-Golay
    def savgol_filter(self, data, window_length=11, polyorder=3):
        """
        Filtr Savitzky-Golay
        window_length: długość okna (nieparzysta)
        polyorder: rząd wielomianu
        """
        return savgol_filter(data, window_length, polyorder)
    
    # 3. Filtr medianowy
    def median_filter_func(self, data, size=5):
        """
        Filtr medianowy
        size: rozmiar okna
        """
        return median_filter(data, size=size)
    
    # 4. Filtr Kalmana (wersja uproszczona 1D)
    def kalman_filter(self, data, Q=1e-5, R=1e-4):
        """
        Prosty filtr Kalmana 1D
        Q: szum procesu
        R: szum pomiaru (wariancja)
        """
        n = len(data)
        x_est = np.zeros(n)
        p_est = np.zeros(n)
        
        x_est[0] = data[0]
        p_est[0] = 1.0
        
        for k in range(1, n):
            x_pred = x_est[k-1]
            p_pred = p_est[k-1] + Q
            
            K = p_pred / (p_pred + R)
            x_est[k] = x_pred + K * (data[k] - x_pred)
            p_est[k] = (1 - K) * p_pred
            
        return x_est
    
    # 5. Średnia ruchoma
    def moving_average_filter(self, data, window=5):
        """
        Filtr średniej ruchomej
        window: rozmiar okna
        """
        return np.convolve(data, np.ones(window)/window, mode='same')
    
    # 6. Filtr Wienera
    def wiener_filter(self, data, window=11):
        """
        Filtr Wienera
        window: rozmiar okna (im większy, tym silniejsze wygładzanie)
        """
        return wiener(data, mysize=window)
    
    def apply_all_filters(self, data, params=None):
        """
        Zastosuj wszystkie filtry do danych
        """
        if params is None:
            params = {
                'butterworth': {'cutoff': 5.0, 'order': 4},
                'savgol': {'window_length': 11, 'polyorder': 3},
                'median': {'size': 10},
                'kalman': {'Q': 1e-4, 'R': 2e-3},
                'moving_avg': {'window': 10},
                'wiener': {'window': 11}
            }
        
        results = {
            'original': data,
            'butterworth': self.butterworth_filter(data, **params['butterworth']),
            'savgol': self.savgol_filter(data, **params['savgol']),
            'median': self.median_filter_func(data, **params['median']),
            'kalman': self.kalman_filter(data, **params['kalman']),
            'moving_avg': self.moving_average_filter(data, **params['moving_avg']),
            'wiener': self.wiener_filter(data, **params['wiener'])
        }
        return results
    
    def add_filtered_columns(self, df, params=None):
        """
        Dodaj kolumny z przefiltrowanymi danymi do DataFrame
        """
        for axis in ['x', 'y', 'z']:
            data = df[f'gl_acc_{axis}'].values
            filtered_results = self.apply_all_filters(data, params)
            
            for key, filtered_data in filtered_results.items():
                col_name = f'{key}_acc_{axis}'
                df[col_name] = filtered_data
        return df
    
    def apply_butterworth_to_all_axes(self, df, cutoff=5.0, order=4):
        """
        Stosuje filtr Butterwortha do wszystkich osi i tworzy nowy dataframe z kolumnami 'butter_acc_x', 'butter_acc_y', 'butter_acc_z'
        """
        df_filtered = pd.DataFrame()
        for axis in ['x', 'y', 'z']:
            data = df[f'gl_acc_{axis}'].values
            df_filtered[f'butter_acc_{axis}'] = self.butterworth_filter(data, cutoff=cutoff, order=order)
        return df_filtered
    
    def apply_savgol_to_all_axes(self, df, window_length=11, polyorder=3):
        """
        Stosuje filtr Savitzky-Golay do wszystkich osi i tworzy nowy dataframe z kolumnami 'savgol_acc_x', 'savgol_acc_y', 'savgol_acc_z'
        """
        df_filtered = pd.DataFrame()
        for axis in ['x', 'y', 'z']:
            data = df[f'gl_acc_{axis}'].values
            df_filtered[f'savgol_acc_{axis}'] = self.savgol_filter(data, window_length=window_length, polyorder=polyorder)
        return df_filtered
    
    def apply_median_to_all_axes(self, df, size=5):
        """
        Stosuje filtr medianowy do wszystkich osi i tworzy nowy dataframe z kolumnami 'median_acc_x', 'median_acc_y', 'median_acc_z'
        """
        df_filtered = pd.DataFrame()
        for axis in ['x', 'y', 'z']:
            data = df[f'gl_acc_{axis}'].values
            df_filtered[f'median_acc_{axis}'] = self.median_filter_func(data, size=size)
        return df_filtered
    
    def apply_kalman_to_all_axes(self, df, Q=1e-5, R=1e-4):
        """
        Stosuje filtr Kalmana do wszystkich osi i tworzy nowy dataframe z kolumnami 'kalman_acc_x', 'kalman_acc_y', 'kalman_acc_z'
        """
        df_filtered = pd.DataFrame()
        for axis in ['x', 'y', 'z']:
            data = df[f'gl_acc_{axis}'].values
            df_filtered[f'kalman_acc_{axis}'] = self.kalman_filter(data, Q=Q, R=R)
        return df_filtered
    
    def apply_moving_average_to_all_axes(self, df, window=5):
        """
        Stosuje filtr średniej ruchomej do wszystkich osi i tworzy nowy dataframe z kolumnami 'moving_avg_acc_x', 'moving_avg_acc_y', 'moving_avg_acc_z'
        """
        df_filtered = pd.DataFrame()
        for axis in ['x', 'y', 'z']:
            data = df[f'gl_acc_{axis}'].values
            df_filtered[f'moving_avg_acc_{axis}'] = self.moving_average_filter(data, window=window)
        return df_filtered
    
    def apply_wiener_to_all_axes(self, df, window=11):
        """
        Stosuje filtr Wienera do wszystkich osi i tworzy nowy dataframe z kolumnami 'wiener_acc_x', 'wiener_acc_y', 'wiener_acc_z'
        """
        df_filtered = pd.DataFrame()
        for axis in ['x', 'y', 'z']:
            data = df[f'gl_acc_{axis}'].values
            df_filtered[f'wiener_acc_{axis}'] = self.wiener_filter(data, window=window)
        return df_filtered


    def calculate_rmse_vs_gt(self, df_gt: pd.DataFrame, df_filtered: pd.DataFrame,
                            axes: List[str] = ['x', 'y', 'z']) -> float:
        """
        Oblicza RMSE między ground truth (z df_gt) a przefiltrowanymi danymi (z df_filtered).
        RMSE liczone jest zbiorczo dla wszystkich podanych osi.
        """
        gt_cols = [f'gt_acc_{axis}' for axis in axes]
        
        filtered_cols = []
        for axis in axes:
            # Szukamy kolumny która kończy się na np. "_acc_x"
            match = [c for c in df_filtered.columns if c.endswith(f'_acc_{axis}')]
            if not match:
                raise ValueError(f"Nie znaleziono kolumny dla osi {axis} w df_filtered (oczekiwano końcówki _acc_{axis})")
            filtered_cols.append(match[0])

        gt_values = df_gt[gt_cols].values
        filt_values = df_filtered[filtered_cols].values

        # Sprawdzenie czy wymiary się zgadzają
        if gt_values.shape != filt_values.shape:
            min_len = min(len(gt_values), len(filt_values))
            gt_values = gt_values[:min_len]
            filt_values = filt_values[:min_len]

        # Obliczenie RMSE
        rmse = np.sqrt(np.mean((gt_values - filt_values)**2))

        return float(rmse*1000)  # zwracamy w mm/s²

    
def plot_result_test_filter_parameters(results, scale='linear'):
    """
    Funkcja do wizualizacji wyników testowania parametrów filtrów
    results: słownik z kluczami jako nazwy filtrów i wartościami jako listy (parametry, RMSE)
    scale: 'log' lub 'linear' - skala osi X
    """
    plt.figure(figsize=(12, 8))
    
    for filter_name, data in results.items():
        params, rmse = zip(*data)
        plt.plot(params, rmse, marker='o', label=filter_name)
    plt.xscale(scale)
    plt.title('Testowanie parametrów filtrów - RMSE vs Parametry')
    plt.xlabel('Wielkość okna')
    plt.ylabel('RMSE względem GT [mm/s²]')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_filter_parameters(df):
    """
    Testuj różne parametry dla każdego filtra
    """
    IMUFilterInstance = IMUFilter(fs=100.0)
    best_params_by_filter = {}

    print("\nTESTOWANIE PARAMETROW FILTROW")
    print("="*60)
    
    # 1. Test Butterwortha - różne częstotliwości odcięcia
    print("\n1. BUTTERWORTH - test cutoff and order:")
    cutoffs = np.linspace(1, 40, 15)
    orders = [o for o in range(1, 6)]
    results = {}
    for cutoff in cutoffs:
        for order in orders:
            df_filtered = IMUFilterInstance.apply_butterworth_to_all_axes(df, cutoff=cutoff, order=order)
            rmse = IMUFilterInstance.calculate_rmse_vs_gt(df, df_filtered)
            results[(cutoff, order)] = rmse
            # print(f"   cutoff={cutoff:.1f} Hz, order={order}: RMSE={rmse:.6f}")
    print("\n   Najlepszy wynik Butterwortha:")
    best_params = min(results, key=results.get)
    best_params_by_filter['butterworth'] = {'cutoff': float(best_params[0]), 'order': int(best_params[1])}
    print(f"   cutoff={best_params[0]:.1f} Hz, order={best_params[1]}: RMSE={results[best_params]:.6f}")
    plot_result_test_filter_parameters(results={f'Butterworth (order={o})': [(c, results[(c, o)]) for c in cutoffs] for o in orders})
    
    # 2. Test Savitzky-Golay - różne okna
    print("\n2. SAVITZKY-GOLAY - test window lengths and polyorder:")
    windows = [w for w in range(3, 80, 2)]
    polyorders = [2, 3, 4, 5]
    results = {}
    for p in polyorders:
        for w in windows:
            if w <= p:
                continue  # Okno musi być większe niż rząd wielomianu
            df_filtered = IMUFilterInstance.apply_savgol_to_all_axes(df, window_length=w, polyorder=p)
            rmse = IMUFilterInstance.calculate_rmse_vs_gt(df, df_filtered)
            results[(w, p)] = rmse
            # print(f"   window={w}, polyorder={p}: RMSE={rmse:.6f}")
    print("\n   Najlepszy wynik Savitzky-Golay:")
    best_params = min(results, key=results.get)
    best_params_by_filter['savgol'] = {'window_length': int(best_params[0]), 'polyorder': int(best_params[1])}
    print(f"   window={best_params[0]}, polyorder={best_params[1]}: RMSE={results[best_params]:.6f}")
    plot_result_test_filter_parameters(results={f'Savitzky-Golay (polyorder={p})': [(w, results[(w, p)]) for w in windows if w > p] for p in polyorders})
    
    # 3. Test medianowy - różne okna
    print("\n3. MEDIAN - test window sizes:")
    sizes = [s for s in range(3, 100, 2)]
    results = {}
    for size in sizes:
        df_filtered = IMUFilterInstance.apply_median_to_all_axes(df, size=size)
        rmse = IMUFilterInstance.calculate_rmse_vs_gt(df, df_filtered)
        results[size] = rmse
        # print(f"   size={size}: RMSE={rmse:.6f}")
    print("\n   Najlepszy wynik mediany:")
    best_size = min(results, key=results.get)
    best_params_by_filter['median'] = {'size': int(best_size)}
    print(f"   size={best_size}: RMSE={results[best_size]:.6f}")
    plot_result_test_filter_parameters(results={f'Median': [(s, results[s]) for s in sizes]})

    # 4. Test Kalmana - różne Q i R
    print("\n4. KALMAN - test Q and R values:")
    Q_values = np.logspace(-8, -2, 15)
    R_values = np.logspace(-6, -1, 15)
    results = {}
    for Q in Q_values:
        for R in R_values:
            df_filtered = IMUFilterInstance.apply_kalman_to_all_axes(df, Q=Q, R=R)
            rmse = IMUFilterInstance.calculate_rmse_vs_gt(df, df_filtered)
            results[(Q, R)] = rmse
            # print(f"   Q={Q:.1e}, R={R:.1e}: RMSE={rmse:.6f}")
    print("\n   Najlepszy wynik Kalmana:")
    best_params = min(results, key=results.get)
    best_params_by_filter['kalman'] = {'Q': float(best_params[0]), 'R': float(best_params[1])}
    print(f"   Q={best_params[0]:.1e}, R={best_params[1]:.1e}: RMSE={results[best_params]:.6f}")
    plot_result_test_filter_parameters(results={f'Kalman (R={R:.1e})': [(Q, results[(Q, R)]) for Q in Q_values] for R in R_values}, scale='log')

    # 5. Test średniej ruchomej - różne okna
    print("\n5. MOVING AVERAGE - test window sizes:")
    windows = [w for w in range(3, 100, 2)]
    results = {}
    for window in windows:
        df_filtered = IMUFilterInstance.apply_moving_average_to_all_axes(df, window=window)
        rmse = IMUFilterInstance.calculate_rmse_vs_gt(df, df_filtered)
        results[window] = rmse
        # print(f"   window={window}: RMSE={rmse:.6f}")
    print("\n   Najlepszy wynik średniej ruchomej:")
    best_window = min(results, key=results.get)
    best_params_by_filter['moving_avg'] = {'window': int(best_window)}
    print(f"   window={best_window}: RMSE={results[best_window]:.6f}")
    plot_result_test_filter_parameters(results={f'Moving Average': [(w, results[w]) for w in windows]})

    # 6. Test Wienera - różne okna
    print("\n6. WIENER - test window sizes:")
    windows = [w for w in range(3, 100, 2)]
    results = {}
    for window in windows:
        df_filtered = IMUFilterInstance.apply_wiener_to_all_axes(df, window=window)
        rmse = IMUFilterInstance.calculate_rmse_vs_gt(df, df_filtered)
        results[window] = rmse
        # print(f"   window={window}: RMSE={rmse:.6f}")
    print("\n   Najlepszy wynik Wienera:")
    best_window = min(results, key=results.get)
    best_params_by_filter['wiener'] = {'window': int(best_window)}
    print(f"   window={best_window}: RMSE={results[best_window]:.6f}")
    plot_result_test_filter_parameters(results={f'Wiener': [(w, results[w]) for w in windows]})

    return best_params_by_filter
