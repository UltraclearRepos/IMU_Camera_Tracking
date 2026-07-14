import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation



GRAVITY_MAG = 9.64  # Przyspieszenie ziemskie w m/s^2

# --- KONFIGURACJA PARAMETRÓW CZUJNIKA ---
# 1000 mg = 1 g = 9.81 m/s^2
# ACC_SCALE = 9.81 / 1000.0  
ACC_SCALE = GRAVITY_MAG / 1000.0  

# 1000 mdps = 1 deg/s
# Zamieniamy na radiany na sekundę: (val / 1000) * (pi / 180)
GYRO_SCALE = (1.0 / 1000.0) * (np.pi / 180.0)


def load_IMU_data(filepath):
    """
    Wczytuje dane z CSV, parsuje czas i przelicza jednostki na SI.
    """
    df = pd.read_csv(filepath)

    # Usunięcie niepotrzebnych kolumn
    df.drop(columns=['rtcDate'], inplace=True, errors='ignore')
    df.drop(columns=['dataRdy'], inplace=True, errors='ignore')
    df.drop(columns=['Unnamed: 10'], inplace=True, errors='ignore')
    df.drop(index=0, inplace=True, errors='ignore')
    
    # Obliczenie czasu relatywnego w sekundach (od startu pomiaru) z użyciem output_Hz,
    # aby uniknąć problemów z nieregularnymi rtcTime   
    df['dt'] = 1.0 / df['output_Hz']
    df['ts'] = df['dt'].cumsum()
    df['ts'] = df['ts'].round(5)

    df['rtcTime'] = pd.to_datetime(df['rtcTime'], format='%H:%M:%S.%f')
    df['rtcTime'] = df['rtcTime'] - df['rtcTime'].iloc[0]

    # --- KONWERSJA JEDNOSTEK NA SI ---
    
    # Akcelerometr: mg -> m/s^2
    df['acc_x'] = df['aX'] * ACC_SCALE
    df['acc_y'] = df['aY'] * ACC_SCALE
    df['acc_z'] = df['aZ'] * ACC_SCALE
    
    # Żyroskop: mdps -> rad/s
    df['gyro_x'] = df['gX'] * GYRO_SCALE
    df['gyro_y'] = df['gY'] * GYRO_SCALE
    df['gyro_z'] = df['gZ'] * GYRO_SCALE
    
    return df


def save_segment(df, start_s: int, end_s: int, filename: str):
    """
    Wytnij fragment `df` między start_s a end_s (sekundy) i zapisz do pliku CSV o nazwie `filename`.
    Zwraca wycięty dataframe.
    """
    # Walidacja wejścia
    if not isinstance(start_s, (int, float)) or not isinstance(end_s, (int, float)):
        raise TypeError("start_s i end_s muszą być liczbami (int lub float).")
    if start_s >= end_s:
        raise ValueError("start_s musi być mniejsze od end_s.")

    start = float(start_s)
    end = float(end_s)

    ts_min, ts_max = df['ts'].min(), df['ts'].max()
    if end < ts_min or start > ts_max:
        raise ValueError(f"Zakres [{start}, {end}] poza zakresem danych [{ts_min}, {ts_max}].")

    # Przycięcie do zakresu danych (jeśli trzeba)
    start_clipped = max(start, ts_min)
    end_clipped = min(end, ts_max)
    if (start_clipped, end_clipped) != (start, end):
        print(f"Zakres przycięty do [{start_clipped}, {end_clipped}] aby zmieścić się w danych.")

    seg = df[(df['ts'] >= start_clipped) & (df['ts'] <= end_clipped)].copy()
    if seg.empty:
        raise ValueError("Brak próbek w zadanym przedziale czasowym.")

    out = filename if filename.lower().endswith('.csv') else f"{filename}.csv"
    seg.to_csv(out, index=False)
    print(f"Zapisano {len(seg)} wierszy do '{out}'")
    return seg


def plot_raw_IMU_data(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Akcelerometr
    ax1.plot(df['ts'], df['acc_x'], label='Acc X', alpha=0.7)
    ax1.plot(df['ts'], df['acc_y'], label='Acc Y', alpha=0.7)
    ax1.plot(df['ts'], df['acc_z'], label='Acc Z', alpha=0.7)
    ax1.set_xlabel('Czas [s]')
    ax1.set_ylabel('Przyspieszenie [$m/s^2$]')
    ax1.set_title('Surowe dane: Akcelerometr')
    ax1.legend(loc='upper right')
    
    # Żyroskop
    ax2.plot(df['ts'], df['gyro_x'], label='Gyro X', alpha=0.7)
    ax2.plot(df['ts'], df['gyro_y'], label='Gyro Y', alpha=0.7)
    ax2.plot(df['ts'], df['gyro_z'], label='Gyro Z', alpha=0.7)
    ax2.set_ylabel('Prędkość kątowa [rad/s]')
    ax2.set_xlabel('Czas [s]')
    ax2.set_title('Surowe dane: Żyroskop')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def compute_orientation_and_global_acc(df, alpha=0.98):
    """
    Oblicza orientację (Roll, Pitch) i transformuje przyspieszenie do układu świata.
    alpha: waga filtru komplementarnego (im bliżej 1, tym bardziej ufamy żyroskopowi)
    """
    acc = df[['acc_x', 'acc_y', 'acc_z']].values
    gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
    dt = df['dt'].values
    
    n = len(df)
    roll = np.zeros(n)
    pitch = np.zeros(n)
    
    # Inicjalizacja
    roll[0] = np.arctan2(acc[0, 1], acc[0, 2])
    pitch[0] = np.arctan2(-acc[0, 0], np.sqrt(acc[0, 1]**2 + acc[0, 2]**2))
    
    # Pętla filtra
    for i in range(1, n):
        # Gyro update
        roll[i] = roll[i-1] + gyro[i, 0] * dt[i]
        pitch[i] = pitch[i-1] + gyro[i, 1] * dt[i]
        
        # Acc correction
        a_roll = np.arctan2(acc[i, 1], acc[i, 2])
        a_pitch = np.arctan2(-acc[i, 0], np.sqrt(acc[i, 1]**2 + acc[i, 2]**2))
        
        # Complementary mix
        roll[i] = alpha * roll[i] + (1 - alpha) * a_roll
        pitch[i] = alpha * pitch[i] + (1 - alpha) * a_pitch

    # Yaw - całka skumulowana (wektoryzowana)
    yaw = np.cumsum(gyro[:, 2] * dt)
    yaw = detrend(yaw)
    
    # Transformacja do układu świata
    # Tworzymy obiekt rotacji dla wszystkich próbek naraz
    angles = np.stack([roll, pitch, yaw], axis=1)
    rotations = Rotation.from_euler('xyz', angles)
    
    # Obracamy wszystkie wektory przyspieszenia naraz
    acc_world = rotations.apply(acc)
    
    # Odejmujemy grawitację
    gravity = np.array([0, 0, GRAVITY_MAG])
    lin_acc = acc_world - gravity
    
    # Zapis do DataFrame
    df['roll'], df['pitch'], df['yaw'] = roll, pitch, yaw
    df[['gl_acc_x', 'gl_acc_y', 'gl_acc_z']] = lin_acc
    
    return df


def trim_IMU_data(df, trim_start=0, trim_end=0):
    """
    Usuwa trim_start pierwszych próbek i trim_end ostatnich próbek z df_gt, resetuje indeksy.
    """
    if trim_start < 0 or trim_end < 0:
        raise ValueError("trim_start i trim_end muszą być nieujemne.")
    if trim_start + trim_end >= len(df):
        raise ValueError("Suma trim_start i trim_end musi być mniejsza niż liczba próbek.")
    
    tf_trimmed = df.iloc[trim_start:len(df)-trim_end].reset_index(drop=True)
    return tf_trimmed


def resample_IMU_data(df, target_fps=100.0):
    """
    Resampling danych z IMU do stałego FPS i interpolacja liniowa.
    """
    t_start = df['ts'].iloc[0]
    t_end = df['ts'].iloc[-1]
    t_uniform = np.arange(t_start, t_end, 1.0 / target_fps)
    
    df_resampled = pd.DataFrame({'ts': t_uniform})
    
    for col in ['gl_acc_x', 'gl_acc_y', 'gl_acc_z', 'roll', 'pitch', 'yaw']:
        f_interp = interp1d(df['ts'], df[col], kind='linear')
        df_resampled[col] = f_interp(t_uniform)
    
    return df_resampled


def remove_trend(df):
    """
    Usuwa trend z danych przyspieszenia globalnego (gl_acc_x, gl_acc_y, gl_acc_z) za pomocą funkcji detrend.
    """
    for axis in ['x', 'y', 'z']:
        col_name = f'gl_acc_{axis}'
        if col_name in df.columns:
            df[col_name] = detrend(df[col_name])
    return df

def remove_moving_average_trend(df, window_size_sec=10, target_fps=100.0):
    """
    Usuwa trend z danych przyspieszenia globalnego (gl_acc_x, gl_acc_y, gl_acc_z) za pomocą średniej kroczącej.
    window_size_sec: rozmiar okna w sekundach (np. 5 sekund) - konwertowany na liczbę próbek na podstawie średniego dt.
    """
    window_size_samples = int(window_size_sec * target_fps)
    for axis in ['x', 'y', 'z']:
        col_name = f'gl_acc_{axis}'
        if col_name in df.columns:
            rolling_mean = df[col_name].rolling(window=window_size_samples, min_periods=1, center=True).mean()
            df[col_name] = df[col_name] - rolling_mean
    return df

def remove_average_trend(df):
    """
    Usuwa trend z danych przyspieszenia globalnego (gl_acc_x, gl_acc_y, gl_acc_z) przez odjęcie średniej.
    """
    for axis in ['x', 'y', 'z']:
        col_name = f'gl_acc_{axis}'
        if col_name in df.columns:
            mean_val = df[col_name].mean()
            df[col_name] = df[col_name] - mean_val
    return df

def moving_average_correction(df, window_size_sec=5, target_fps=100.0):
    """
    Usuwa dryft z danych przyspieszenia globalnego (gl_acc_x, gl_acc_y, gl_acc_z) za pomocą średniej kroczącej.
    window_size_sec: rozmiar okna w sekundach (np. 5 sekund) - konwertowany na liczbę próbek na podstawie średniego dt.
    """
    window_size_samples = int(window_size_sec * target_fps)
    for axis in ['x', 'y', 'z']:
        col_name = f'gl_acc_{axis}'
        if col_name in df.columns:
            rolling_mean = df[col_name].rolling(window=window_size_samples, min_periods=1, center=True).mean()
            df[col_name] = df[col_name] - rolling_mean
    return df

def show_acceleration_stats(df):
    """
    Wyświetla statystyki dla przyspieszenia globalnego.
    """
    stats = {}
    for axis in ['x', 'y', 'z']:
        col_name = f'gl_acc_{axis}'
        if col_name in df.columns:
            min_val = df[col_name].min()
            max_val = df[col_name].max()
            mean_val = df[col_name].mean()
            std_val = df[col_name].std()
            stats[axis] = {'mean': mean_val, 'std': std_val, 'min': min_val, 'max': max_val}
            print(f"Global Acc {axis.upper()}: Mean={mean_val:.4f} m/s², Std={std_val:.4f} m/s², Min={min_val:.4f} m/s², Max={max_val:.4f} m/s²")
    return stats


def plot_acceleration_and_orientation(df):
    """
    Wizualizacja danych po obliczeniu orientacji i transformacji do układu globalnego.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    # 1. Kąty Eulera (Roll, Pitch, Yaw) w radianach
    ax1.plot(df['ts'], df['roll'], label='Roll', color='red', alpha=0.7)
    ax1.plot(df['ts'], df['pitch'], label='Pitch', color='green', alpha=0.7)
    ax1.plot(df['ts'], df['yaw'], label='Yaw', color='blue', alpha=0.7)
    ax1.set_xlabel('Czas [s]')
    ax1.set_ylabel('Kąt [rad]')
    ax1.set_title('Orientacja: Kąty Eulera')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
   
    acc_limits = (-0.3, 0.3)  # Zakres przyspieszenia
    ax2.set_ylim(acc_limits)
    ax3.set_ylim(acc_limits)
    ax4.set_ylim(acc_limits)

    # 2. Przyspieszenie liniowe w układzie globalnym dla X (bez grawitacji)
    ax2.plot(df['ts'], df['gl_acc_x'], label='Linear Acc X', color='red', alpha=0.7)
    ax2.set_xlabel('Czas [s]')
    ax2.set_ylabel('Przyspieszenie [m/s²]')
    ax2.set_title('Przyspieszenie liniowe (bez grawitacji)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. Przyspieszenie liniowe w układzie globalnym Y (bez grawitacji)
    ax3.plot(df['ts'], df['gl_acc_y'], label='Linear Acc Y', color='green', alpha=0.7)
    ax3.set_xlabel('Czas [s]')
    ax3.set_ylabel('Przyspieszenie [m/s²]')
    ax3.set_title('Przyspieszenie liniowe (bez grawitacji)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 4. Przyspieszenie liniowe w układzie globalnym Z (bez grawitacji)
    ax4.plot(df['ts'], df['gl_acc_z'], label='Linear Acc Z', color='blue', alpha=0.7)
    ax4.set_xlabel('Czas [s]')
    ax4.set_ylabel('Przyspieszenie [m/s²]')
    ax4.set_title('Przyspieszenie liniowe (bez grawitacji)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_acceleration_distribution(df):
    """
    Wizualizacja rozkładu przyspieszenia globalnego (histogramy i KDE).
    """
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.hist(df['gl_acc_x'], bins=150, color='red', alpha=0.7)
    plt.title('Rozkład przyspieszenia globalnego X')
    plt.xlabel('Przyspieszenie [m/s²]')
    plt.ylabel('Liczba próbek')
    plt.subplot(3, 1, 2)
    plt.hist(df['gl_acc_y'], bins=150, color='green', alpha=0.7)
    plt.title('Rozkład przyspieszenia globalnego Y')
    plt.xlabel('Przyspieszenie [m/s²]')
    plt.ylabel('Liczba próbek')
    plt.subplot(3, 1, 3)
    plt.hist(df['gl_acc_z'], bins=150, color='blue', alpha=0.7)
    plt.title('Rozkład przyspieszenia globalnego Z')
    plt.xlabel('Przyspieszenie [m/s²]')
    plt.ylabel('Liczba próbek')

    max_count = max(plt.subplot(3, 1, 1).get_ylim()[1], plt.subplot(3, 1, 2).get_ylim()[1], plt.subplot(3, 1, 3).get_ylim()[1])
    plt.subplot(3, 1, 1).set_ylim(0, max_count * 1.1)
    plt.subplot(3, 1, 2).set_ylim(0, max_count * 1.1)
    plt.subplot(3, 1, 3).set_ylim(0, max_count * 1.1)

    acc_range = max(abs(df['gl_acc_x'].min()), abs(df['gl_acc_x'].max()), abs(df['gl_acc_y'].min()), abs(df['gl_acc_y'].max()), abs(df['gl_acc_z'].min()), abs(df['gl_acc_z'].max()))
    plt.subplot(3, 1, 1).set_xlim(-acc_range * 1.1, acc_range * 1.1)
    plt.subplot(3, 1, 2).set_xlim(-acc_range * 1.1, acc_range * 1.1)
    plt.subplot(3, 1, 3).set_xlim(-acc_range * 1.1, acc_range * 1.1)

    plt.tight_layout()
    plt.show()


def visualize_trajectory(df, method='gl', axis_size=0.30):
    fig = plt.figure(figsize=(16, 10))
    
    # Wykres 3D
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(df[f'{method}_pos_x'], df[f'{method}_pos_y'], df[f'{method}_pos_z'], label='Trajektoria')
    ax3d.scatter(df[f'{method}_pos_x'].iloc[0], df[f'{method}_pos_y'].iloc[0], df[f'{method}_pos_z'].iloc[0], c='green', s=100, label='Start')
    ax3d.scatter(df[f'{method}_pos_x'].iloc[-1], df[f'{method}_pos_y'].iloc[-1], df[f'{method}_pos_z'].iloc[-1], c='red', s=100, label='Koniec')
    ax3d.set_xlim(-axis_size, axis_size)
    ax3d.set_ylim(-axis_size, axis_size)
    ax3d.set_zlim(-axis_size, axis_size)
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.set_title('Trajektoria 3D')
    ax3d.legend()
    
    # Rzut XY
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(df[f'{method}_pos_x'], df[f'{method}_pos_y'])
    ax_xy.scatter(df[f'{method}_pos_x'].iloc[0], df[f'{method}_pos_y'].iloc[0], c='green')
    ax_xy.scatter(df[f'{method}_pos_x'].iloc[-1], df[f'{method}_pos_y'].iloc[-1], c='red')
    ax_xy.set_xlim(-axis_size, axis_size)
    ax_xy.set_ylim(-axis_size, axis_size)
    ax_xy.set_xlabel('X [m]')
    ax_xy.set_ylabel('Y [m]')
    ax_xy.set_title('Rzut XY (Widok z góry)')
    ax_xy.axis('equal')
    ax_xy.grid(True)

    # Rzut XZ
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.plot(df[f'{method}_pos_x'], df[f'{method}_pos_z'])
    ax_xz.scatter(df[f'{method}_pos_x'].iloc[0], df[f'{method}_pos_z'].iloc[0], c='green')
    ax_xz.scatter(df[f'{method}_pos_x'].iloc[-1], df[f'{method}_pos_z'].iloc[-1], c='red')
    ax_xz.set_xlim(-axis_size, axis_size)
    ax_xz.set_ylim(-axis_size, axis_size)
    ax_xz.set_xlabel('X [m]')
    ax_xz.set_ylabel('Z [m]')
    ax_xz.set_title('Rzut XZ')
    ax_xz.grid(True)
    
    # Rzut YZ
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.plot(df[f'{method}_pos_y'], df[f'{method}_pos_z'])
    ax_yz.scatter(df[f'{method}_pos_y'].iloc[0], df[f'{method}_pos_z'].iloc[0], c='green')
    ax_yz.scatter(df[f'{method}_pos_y'].iloc[-1], df[f'{method}_pos_z'].iloc[-1], c='red')
    ax_yz.set_xlim(-axis_size, axis_size)
    ax_yz.set_ylim(-axis_size, axis_size)
    ax_yz.set_xlabel('Y [m]')
    ax_yz.set_ylabel('Z [m]')
    ax_yz.set_title('Rzut YZ')
    ax_yz.grid(True)
    
    plt.tight_layout()
    plt.show()



def calculate_integrals(df: pd.DataFrame, 
                        method: str = "gl", 
                        target_fps: float = 100.0,
                        high_pass_cutoff: float = 0.5,
                        use_detrend_vel: bool = True,
                        use_detrend_pos: bool = True,
                        use_loop_closure: bool = False,
                        apply_hp_filter: bool = True) -> pd.DataFrame:
    """
    Całkuje przyspieszenie do prędkości i pozycji z użyciem filtracji 
    w celu minimalizacji dryftu.
    """
    dt = 1.0 / target_fps
    df_res = df.copy()
    
    # Filtra górnoprzepustowy (usuwa dryft niskoczęstotliwościowy)
    def highpass_filter(data, cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    for axis in ['x', 'y', 'z']:
        acc_col = f"{method}_acc_{axis}"
        if acc_col not in df.columns:
            continue
            
        acc = df[acc_col].values
        
        # Wstępne oczyszczenie przyspieszenia (High-pass)
        if apply_hp_filter:
            acc = highpass_filter(acc, high_pass_cutoff, target_fps)
        
        # Pierwsze całkowanie: Przyspieszenie -> Prędkość
        vel = cumulative_trapezoid(acc, dx=dt, initial=0)
        
        # Detrending prędkości
        if use_detrend_vel:
            vel = detrend(vel)
            
        # Drugie całkowanie: Prędkość -> Pozycja
        pos = cumulative_trapezoid(vel, dx=dt, initial=0)
        
        # Detrending pozycji
        if use_detrend_pos:
            pos = detrend(pos)
            
        df_res[f"{method}_vel_{axis}"] = vel
        df_res[f"{method}_pos_{axis}"] = pos
    
    if use_loop_closure:
        df_res = approach_loop_closure(df_res, method=method)

    return df_res

def approach_loop_closure(df: pd.DataFrame, method: str = "gl") -> pd.DataFrame:
    """
    Loop Closure (Zamknięcie pętli).
    Jeśli wiesz, że na końcu wracasz do punktu (0,0,0), funkcja
    siłowo usuwa błąd końcowy, rozkładając go proporcjonalnie na cały ruch.
    """
    for axis in ['x', 'y', 'z']:
        p_base = df[f'{method}_pos_{axis}'].values

        error_end = p_base[-1]
        correction = np.linspace(0, error_end, len(p_base))
        p_base = p_base - correction
            
        df[f'{method}_pos_{axis}'] = p_base
    return df
