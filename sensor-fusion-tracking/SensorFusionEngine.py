import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import wiener, savgol_filter
try:
    from final_code.IntegratedKalmanFilter import run_filter, optimize_params
except ModuleNotFoundError:
    from IntegratedKalmanFilter import run_filter, optimize_params


class SensorFusionEngine:
    def __init__(self, df_sync: pd.DataFrame, imu_prefix: str = 'gl', cam_prefix: str = 'cam'):
        """
        Inicjalizacja z już zsynchronizowanym DF, gdzie mamy 
        kolumny {imu_prefix}_pos_x i cam_pos_x itd.
        """
        self.df = df_sync.copy()
        self.method = imu_prefix
        self.imu_prefix = imu_prefix + '_pos_'
        self.cam_prefix = cam_prefix + '_pos_'
        self.axes = ['x', 'y', 'z']

    def approach_weighted_average(self, cam_trust: float = 0.9, loop_closure: bool = False):
        """
        Podejście 1: Prosta średnia ważona pozycji.
        cam_trust: 0.0 do 1.0 (ile ufasz kamerze).
        """
        df_fused = self.df.copy()
        imu_trust = 1.0 - cam_trust
        for axis in self.axes:
            p_imu = df_fused[f'{self.imu_prefix}{axis}'].values
            p_cam = df_fused[f'{self.cam_prefix}{axis}'].values
            
            df_fused[f'fused_pos_{axis}'] = (imu_trust * p_imu) + (cam_trust * p_cam)
        if loop_closure:
            self.approach_loop_closure(df_fused)
        return df_fused

    def approach_complementary_adaptive_filter(self, use_confidence: bool = False, alpha_base: float = 0.9, loop_closure: bool = False):
        df_fused = self.df.copy()
        
        for axis in self.axes:
            p_imu = df_fused[f'{self.imu_prefix}{axis}'].values
            p_cam = df_fused[f'{self.cam_prefix}{axis}'].values
            
            fused = np.zeros(len(p_imu))
            fused[0] = p_cam[0]
            
            if use_confidence:
                conf = df_fused['cam_conf'].values
                for i in range(1, len(p_imu)):
                    delta_imu = p_imu[i] - p_imu[i-1]
                    # Adaptacyjne alfa
                    alpha_dyn = 1.0 - ((1.0 - alpha_base) * conf[i])
                    # Fuzja: predykcja z IMU + korekta z kamery
                    fused[i] = alpha_dyn * (fused[i-1] + delta_imu) + (1.0 - alpha_dyn) * p_cam[i]
            else:
                for i in range(1, len(p_imu)):
                    delta_imu = p_imu[i] - p_imu[i-1]
                    # Fuzja: predykcja z IMU + korekta z kamery
                    fused[i] = alpha_base * (fused[i-1] + delta_imu) + (1 - alpha_base) * p_cam[i]
              
            df_fused[f'fused_pos_{axis}'] = fused
        if loop_closure:
            self.approach_loop_closure(df_fused)
        return df_fused

    def approach_error_drift_compensation(self, smoothing_window: int = 101, loop_closure: bool = False):
        """
        Podejście 3: Kompensacja dryftu błędu.
        Liczy różnicę między IMU a Kamerą, wygładza ją i odejmuje od IMU.
        Dobre, gdy oba czujniki mają "pływający" błąd.
        """
        df_fused = self.df.copy()
        for axis in self.axes:
            p_imu = df_fused[f'{self.imu_prefix}{axis}'].values
            p_cam = df_fused[f'{self.cam_prefix}{axis}'].values
            
            # 1. Surowy błąd między czujnikami
            error = p_imu - p_cam
            
            # 2. Wygładzenie błędu (wyciągnięcie trendu dryftu)
            if len(error) > smoothing_window:
                drift_trend = savgol_filter(error, smoothing_window, 3)
            else:
                drift_trend = error
            
            # 3. Odejmujemy dryft od pozycji IMU
            df_fused[f'fused_pos_{axis}'] = p_imu - drift_trend
        if loop_closure:
            self.approach_loop_closure(df_fused)
        return df_fused    

    def approach_kalman_adaptive_filter(self, use_confidence: bool = False, loop_closure: bool = False):
        # df_train = self.df.iloc[:1000].copy()
        # best_params = optimize_params(df_train, self.method, use_confidence)
        # q_p, q_v, q_b, r_b = best_params
        
        q_p, q_v, q_b, r_b = 1e-8, 5e-4, 1e-5, 3e-2

        # print("best_params: ")
        # print(q_p, q_v, q_b, r_b)

        df_fused = self.df.copy()
        final_pos = run_filter(df_fused, self.method, use_confidence, q_p, q_v, q_b, r_b)
        df_fused['fused_pos_x'] = final_pos[:, 0]
        df_fused['fused_pos_y'] = final_pos[:, 1]
        df_fused['fused_pos_z'] = final_pos[:, 2]
        if loop_closure:
            self.approach_loop_closure(df_fused)
        return df_fused

    
    def approach_wiener_filter(self, loop_closure: bool = False):
        """
        Podejście 5: Filtr Wienera (dla pozycji).
        Wygładza różnicę między IMU a Kamerą, zakładając, że błąd jest stacjonarnym szumem.
        Działa dobrze, gdy błąd jest losowy i nie ma silnego dryftu.
        """
        df_fused = self.df.copy()
        for axis in self.axes:
            p_imu = df_fused[f'{self.imu_prefix}{axis}'].values
            p_cam = df_fused[f'{self.cam_prefix}{axis}'].values
            
            error = p_imu - p_cam
            error_wiener = wiener(error)
            df_fused[f'fused_pos_{axis}'] = p_imu - error_wiener
        if loop_closure:
            self.approach_loop_closure(df_fused)
        return df_fused

    def approach_loop_closure(self, df_fused):
        """
        Loop Closure (Zamknięcie pętli).
        Jeśli wiesz, że na końcu wracasz do punktu (0,0,0), funkcja
        siłowo usuwa błąd końcowy, rozkładając go proporcjonalnie na cały ruch.
        """
        for axis in self.axes:
            p_base = df_fused[f'fused_pos_{axis}'].values

            error_end = p_base[-1]
            correction = np.linspace(0, error_end, len(p_base))
            p_base = p_base - correction
                
            df_fused[f'fused_pos_{axis}'] = p_base
        return df_fused
