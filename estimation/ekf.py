import math
import time
import threading
import numpy as np


class EKF:
    """
    Extended Kalman Filter for UGV state estimation.

    State vector: [x, y, yaw, v, yaw_rate]
                   x        - position east  (m)
                   y        - position north (m)
                   yaw      - heading        (rad, ENU convention)
                   v        - linear velocity (m/s)
                   yaw_rate - angular velocity (rad/s)

    Sensors fused:
        - IMU       (100Hz) → predict step (yaw_rate)
        - Encoders  (50Hz)  → update step  (v, yaw_rate)
        - GPS       (10Hz)  → update step  (x, y)
    """

    def __init__(self):
        # --- State vector [x, y, yaw, v, yaw_rate] ---
        self.x = np.zeros(5)

        # --- State covariance ---
        self.P = np.diag([1.0, 1.0, 0.1, 0.1, 0.01])

        # --- Process noise (Q) ---
        # How much we trust our motion model
        self.Q = np.diag(
            [
                0.01,  # x process noise
                0.01,  # y process noise
                0.1,  # yaw process noise — allow faster yaw correction
                0.1,  # velocity process noise
                0.1,  # yaw_rate process noise
            ]
        )

        # --- Measurement noise (R) per sensor ---
        # GPS: noisy position only — no heading from GPS
        self.R_gps = np.diag(
            [0.5**2, 0.5**2]  # x variance (matches GPS noise std=0.5m)  # y variance
        )

        # Encoder: reliable linear velocity, less reliable yaw_rate
        self.R_enc = np.diag(
            [
                0.05**2,  # linear velocity variance
                0.1
                ** 2,  # yaw_rate variance — increased, encoder angular less reliable during spin
            ]
        )

        # Magnetometer: absolute heading, low noise but has calibration errors
        self.R_mag = np.array([[0.02**2]])  # heading variance (~1 degree noise)

        # IMU: gyro yaw_rate (used in predict)
        self.R_imu_yaw_rate = 0.01**2

        # --- Thread safety ---
        self._lock = threading.Lock()

        # --- Timing ---
        self.last_time = time.time()

        # --- Output cache ---
        self._state_cache = None

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wrap_angle(angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _compute_dt(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        dt = max(0.0001, min(dt, 0.5))  # clamp: avoid zero or huge dt
        return dt

    # ------------------------------------------------------------------ #
    #  Predict Step (runs at IMU rate ~100Hz)                             #
    # ------------------------------------------------------------------ #

    def predict(self, gyro_z, dt=None):
        """
        Predict step using IMU yaw rate.

        Args:
            gyro_z : float - yaw rate from IMU (rad/s)
            dt     : float - time delta (if None, computed internally)
        """
        with self._lock:
            if dt is None:
                dt = self._compute_dt()

            x, y, yaw, v, yaw_rate = self.x

            # Use IMU yaw_rate directly in predict
            yaw_rate = gyro_z

            # --- Motion model ---
            # x_new = x + v * cos(yaw) * dt
            # y_new = y + v * sin(yaw) * dt
            # yaw_new = yaw + yaw_rate * dt
            # v_new = v (constant velocity assumption)
            # yaw_rate_new = yaw_rate (constant)

            x_new = x + v * math.cos(yaw) * dt
            y_new = y + v * math.sin(yaw) * dt
            yaw_new = self._wrap_angle(yaw + yaw_rate * dt)
            v_new = v
            yaw_rate_new = yaw_rate

            self.x = np.array([x_new, y_new, yaw_new, v_new, yaw_rate_new])

            # --- Jacobian of motion model (F) ---
            F = np.eye(5)
            F[0, 2] = -v * math.sin(yaw) * dt  # dx/dyaw
            F[0, 3] = math.cos(yaw) * dt  # dx/dv
            F[1, 2] = v * math.cos(yaw) * dt  # dy/dyaw
            F[1, 3] = math.sin(yaw) * dt  # dy/dv
            F[2, 4] = dt  # dyaw/dyaw_rate

            # --- Propagate covariance ---
            self.P = F @ self.P @ F.T + self.Q

            self._update_cache()

    # ------------------------------------------------------------------ #
    #  Update Steps                                                        #
    # ------------------------------------------------------------------ #

    def update_encoder(self, linear_vel, angular_vel):
        """
        EKF update using encoder odometry.

        Args:
            linear_vel  : float m/s
            angular_vel : float rad/s
        """
        with self._lock:
            # Measurement: [v, yaw_rate]
            z = np.array([linear_vel, angular_vel])

            # Observation matrix H (which states we observe)
            H = np.zeros((2, 5))
            H[0, 3] = 1.0  # observe v
            H[1, 4] = 1.0  # observe yaw_rate

            self._apply_update(z, H, self.R_enc)
            self._update_cache()

    def update_magnetometer(self, heading):
        """
        EKF update using magnetometer absolute heading.
        This is the key fix for yaw drift — magnetometer gives absolute reference.

        Args:
            heading : float - absolute yaw in ENU frame (rad)
        """
        with self._lock:
            # Wrap heading to match state
            heading = self._wrap_angle(heading)

            z = np.array([heading])

            H = np.zeros((1, 5))
            H[0, 2] = 1.0  # observe yaw directly

            # Custom update to handle yaw wrapping in innovation
            innovation = z - H @ self.x
            innovation[0] = self._wrap_angle(innovation[0])  # wrap yaw innovation

            S = H @ self.P @ H.T + self.R_mag
            K = self.P @ H.T @ np.linalg.inv(S)

            self.x = self.x + K @ innovation
            self.x[2] = self._wrap_angle(self.x[2])

            I = np.eye(len(self.x))
            I_KH = I - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R_mag @ K.T

            self._update_cache()

    def update_gps(self, gps_x, gps_y):
        """
        EKF update using GPS position only.
        Real GPS gives position only — heading comes from IMU + encoders.

        Args:
            gps_x : float - ENU x position (m)
            gps_y : float - ENU y position (m)
        """
        with self._lock:
            z = np.array([gps_x, gps_y])
            H = np.zeros((2, 5))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            self._apply_update(z, H, self.R_gps)
            self._update_cache()

    def _apply_update(self, z, H, R):
        """
        Generic EKF update step.

        Args:
            z : measurement vector
            H : observation matrix
            R : measurement noise covariance
        """
        # Innovation
        y = z - H @ self.x

        # Wrap yaw in innovation if yaw is observed
        # (not needed for GPS/encoder but good habit)

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        self.x[2] = self._wrap_angle(self.x[2])  # wrap yaw

        # Covariance update (Joseph form for numerical stability)
        I = np.eye(len(self.x))
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # ------------------------------------------------------------------ #
    #  State Access                                                        #
    # ------------------------------------------------------------------ #

    def _update_cache(self):
        """Update cached state (called inside lock)."""
        self._state_cache = {
            "x": self.x[0],
            "y": self.x[1],
            "yaw": self.x[2],
            "v": self.x[3],
            "yaw_rate": self.x[4],
            "timestamp": time.time(),
        }

    def get_state(self):
        """
        Returns latest EKF state estimate (thread safe).

        Returns dict:
            x, y      : position (m)
            yaw       : heading (rad)
            v         : linear velocity (m/s)
            yaw_rate  : angular velocity (rad/s)
            timestamp : float
        """
        with self._lock:
            return dict(self._state_cache) if self._state_cache else None

    def initialize(self, x, y, yaw):
        """
        Initialize EKF state from known starting position.
        Call this before starting the EKF loop.

        Args:
            x   : initial x position (m)
            y   : initial y position (m)
            yaw : initial heading (rad)
        """
        with self._lock:
            self.x = np.array([float(x), float(y), float(yaw), 0.0, 0.0])
            self.P = np.diag([0.1, 0.1, 0.05, 0.1, 0.01])
            self.last_time = time.time()
            self._update_cache()
            print(
                f"[EKF] Initialized at x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f} deg"
            )
