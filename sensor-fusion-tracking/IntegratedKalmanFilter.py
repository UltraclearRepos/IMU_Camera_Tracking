import numpy as np
import pandas as pd
from scipy.optimize import minimize

class IntegratedKalmanFilter:
    def __init__(self, dt=0.01):
        self.dt = dt
        # Stan x: [px, py, pz, vx, vy, vz, bx, by, bz]
        self.x = np.zeros((9, 1))
        # Macierz kowariancji błędu P
        self.P = np.eye(9) * 0.1
        
        # Macierz przejścia stanu F
        self.F = np.eye(9)
        self.F[0:3, 3:6] = np.eye(3) * dt
        
        # Macierz wpływu wejścia B
        self.B = np.zeros((9, 3))
        self.B[0:3, 0:3] = np.eye(3) * (0.5 * dt**2)
        self.B[3:6, 0:3] = np.eye(3) * dt
        
        # Macierz pomiaru H
        self.H = np.zeros((3, 9))
        self.H[0:3, 0:3] = np.eye(3)
        
        self.I = np.eye(9)

    def predict(self, acc_raw, q_pos, q_vel, q_bias):
        # Budowa macierzy szumu procesu Q
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_bias, q_bias, q_bias])
        
        # Korekcja o bias (bias jest w x[6:9])
        acc_corrected = acc_raw.reshape(3, 1) - self.x[6:9]
        
        # Predykcja stanu i kowariancji
        self.x = self.F @ self.x + self.B @ acc_corrected
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, cam_pos, conf, r_base, conf_threshold=0.1):
        if conf < conf_threshold:
            return

        # Skalowanie macierzy szumu R na podstawie pewności (Confidence)
        R = np.eye(3) * (r_base / (conf + 1e-6))
        
        # Innowacja (różnica między kamerą a predykcją)
        z = cam_pos.reshape(3, 1)
        y = z - self.H @ self.x
        # Wzmocnienie Kalmana (Kalman Gain)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Aktualizacja stanu x i macierzy P
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

def run_filter(df, method, use_confidence, q_pos, q_vel, q_bias, r_base):
    kf = IntegratedKalmanFilter(dt=0.01)
    
    # Pre-konwersja kolumn DataFrame na tablice NumPy
    acc_data = df[[f'{method}_acc_x', f'{method}_acc_y', f'{method}_acc_z']].values
    cam_data = df[['cam_pos_x', 'cam_pos_y', 'cam_pos_z']].values
    
    if use_confidence:
        conf_data = df['cam_conf'].values
    else:
        conf_data = np.ones(len(df))
        
    n = len(df)
    positions = np.zeros((n, 3))
    
    for i in range(n):
        kf.predict(acc_data[i], q_pos, q_vel, q_bias)
        kf.update(cam_data[i], conf_data[i], r_base)
        # Zapisujemy wynik (kopiujemy wartości px, py, pz)
        positions[i, 0] = kf.x[0, 0]
        positions[i, 1] = kf.x[1, 0]
        positions[i, 2] = kf.x[2, 0]
        
    return positions

def objective_function(params, df, method, use_confidence):
    q_pos, q_vel, q_bias, r_base = params
    
    pred_pos = run_filter(df, method, use_confidence, q_pos, q_vel, q_bias, r_base)
    gt_pos = df[['gt_pos_x', 'gt_pos_y', 'gt_pos_z']].values
    rmse = np.sqrt(np.mean((pred_pos - gt_pos)**2))
    
    return rmse

def optimize_params(df_fragment, method, use_confidence):
    # [q_pos, q_vel, q_bias, r_base]
    initial_guess = [1e-4, 1e-3, 1e-5, 1e-2]
    bounds = [(1e-8, 1e-1), (1e-8, 1e-1), (1e-9, 1e-2), (1e-6, 1)]
    
    print("Optymalizacja parametrów...")
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(df_fragment, method, use_confidence), 
        bounds=bounds, 
        method='L-BFGS-B'
    )
    
    return result.x