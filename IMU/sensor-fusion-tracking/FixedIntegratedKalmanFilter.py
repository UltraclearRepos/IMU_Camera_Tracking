"""Corrected 9-state position/velocity/accelerometer-bias Kalman filter."""

import numpy as np


class FixedIntegratedKalmanFilter:
    """Kalman filter with state [position, velocity, accelerometer bias]."""

    def __init__(self, dt=0.01):
        self.dt = float(dt)
        self.x = np.zeros((9, 1))
        self.P = np.eye(9) * 0.1

        self.H = np.zeros((3, 9))
        self.H[0:3, 0:3] = np.eye(3)
        self.I = np.eye(9)

        # Values from the most recent camera update, exposed for diagnostics.
        self.last_innovation = None
        self.last_innovation_covariance = None
        self.last_kalman_gain = None
        self.last_nis = None

    def predict(self, acc_raw, q_pos, q_vel, q_bias):
        """Propagate position and velocity while estimating acceleration bias."""

        dt = self.dt
        acceleration = np.asarray(acc_raw, dtype=float).reshape(3, 1)
        bias = self.x[6:9].copy()
        acceleration_corrected = acceleration - bias

        # State propagation. Bias follows a random-walk model, so its expected
        # value stays unchanged during prediction.
        self.x[0:3] += self.x[3:6] * dt + 0.5 * acceleration_corrected * dt**2
        self.x[3:6] += acceleration_corrected * dt

        # Jacobian of the state transition. These bias blocks are what allow a
        # later camera position residual to correct velocity and bias.
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -np.eye(3) * (0.5 * dt**2)
        F[3:6, 6:9] = -np.eye(3) * dt

        Q = np.diag(
            [
                q_pos,
                q_pos,
                q_pos,
                q_vel,
                q_vel,
                q_vel,
                q_bias,
                q_bias,
                q_bias,
            ]
        )
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, cam_pos, conf, r_base, conf_threshold=0.1):
        """Correct the state with a camera position measurement."""

        if conf < conf_threshold:
            return

        R = np.eye(3) * (r_base / (conf + 1e-6))
        measurement = np.asarray(cam_pos, dtype=float).reshape(3, 1)
        innovation = measurement - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + R
        kalman_gain = (
            self.P @ self.H.T @ np.linalg.inv(innovation_covariance)
        )

        self.last_innovation = innovation.copy()
        self.last_innovation_covariance = innovation_covariance.copy()
        self.last_kalman_gain = kalman_gain.copy()
        self.last_nis = float(
            innovation.T
            @ np.linalg.inv(innovation_covariance)
            @ innovation
        )

        self.x = self.x + kalman_gain @ innovation

        # Joseph form is more numerically stable and preserves positive
        # semidefiniteness better than (I-KH)P.
        correction = self.I - kalman_gain @ self.H
        self.P = (
            correction @ self.P @ correction.T
            + kalman_gain @ R @ kalman_gain.T
        )
        self.P = 0.5 * (self.P + self.P.T)
