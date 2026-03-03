import time
import threading

from estimation.ekf import EKF


class EKFRunner:
    """
    Manages the EKF thread and connects sensor buffers to EKF updates.

    Threading model:
        - Predict    at 50Hz  — IMU gyro
        - Update     at 50Hz  — Magnetometer (absolute yaw, fixes drift)
        - Update     at 50Hz  — Encoder (velocity)
        - Update     at 10Hz  — GPS (position only)

    Magnetometer is the key addition — fixes yaw drift during motion.
    """

    def __init__(self, imu_buffer, gps_buffer, encoder_buffer, mag_buffer):
        self.imu_buffer = imu_buffer
        self.gps_buffer = gps_buffer
        self.encoder_buffer = encoder_buffer
        self.mag_buffer = mag_buffer

        self.ekf = EKF()
        self.running = False

        self._last_gps_ts = None
        self._last_encoder_ts = None
        self._last_imu_ts = None
        self._last_mag_ts = None

        self._thread = None

    def start(self, init_from_gps=True):
        """
        Start EKF runner thread.
        Initializes position from GPS and heading from magnetometer.
        """
        self._wait_and_initialize(init_from_gps)

        self.running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="EKF-Thread", daemon=True
        )
        self._thread.start()
        print("[EKFRunner] Started")

    def stop(self):
        self.running = False

    def get_state(self):
        """Get latest EKF state estimate. Safe to call from any thread."""
        return self.ekf.get_state()

    def _wait_and_initialize(self, init_from_gps=True, timeout=5.0):
        """
        Wait for first GPS + magnetometer readings to initialize EKF.
        GPS gives initial position, magnetometer gives initial heading.
        """
        print("[EKFRunner] Waiting for GPS and magnetometer fix...")
        start = time.time()

        init_x, init_y, init_yaw = 0.0, 0.0, 0.0

        while time.time() - start < timeout:
            gps_data = self.gps_buffer.read()
            mag_data = self.mag_buffer.read()

            if gps_data is not None and mag_data is not None:
                pos, gps_ts = gps_data
                heading, mag_ts = mag_data

                init_x = float(pos[0])
                init_y = float(pos[1])
                init_yaw = float(heading)  # ensure plain float, not numpy type

                self._last_gps_ts = gps_ts
                self._last_mag_ts = mag_ts
                break

            time.sleep(0.05)
        else:
            print("[EKFRunner] Sensor timeout — initializing at origin")

        self.ekf.initialize(init_x, init_y, init_yaw)

    def _run_loop(self):
        """
        EKF main loop at 50Hz.

        Order each iteration:
        1. Predict  — IMU gyro (motion model)
        2. Update   — Magnetometer (absolute yaw — fixes drift!)
        3. Update   — Encoder (velocity)
        4. Update   — GPS (position only)
        """
        interval = 1.0 / 50  # 50Hz

        while self.running:
            loop_start = time.time()

            # --- 1. Predict using IMU gyro ---
            imu_data = self.imu_buffer.read()
            if imu_data is not None:
                gyro, accel, ts = imu_data
                if ts != self._last_imu_ts:
                    self.ekf.predict(gyro_z=gyro[2])
                    self._last_imu_ts = ts

            # --- 2. Update with magnetometer (absolute yaw) ---
            mag_data = self.mag_buffer.read()
            if mag_data is not None:
                heading, ts = mag_data
                if ts != self._last_mag_ts:
                    self.ekf.update_magnetometer(heading)
                    self._last_mag_ts = ts

            # --- 3. Update with encoder ---
            enc_data = self.encoder_buffer.read()
            if enc_data is not None:
                linear_vel, angular_vel, vl, vr, ts = enc_data
                if ts != self._last_encoder_ts:
                    self.ekf.update_encoder(linear_vel, angular_vel)
                    self._last_encoder_ts = ts

            # --- 4. Update with GPS (position only) ---
            gps_data = self.gps_buffer.read()
            if gps_data is not None:
                pos, ts = gps_data
                if ts != self._last_gps_ts:
                    self.ekf.update_gps(pos[0], pos[1])
                    self._last_gps_ts = ts

            # --- Sleep for remainder of interval ---
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
