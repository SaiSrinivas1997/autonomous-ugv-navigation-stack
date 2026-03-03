import math
import time
import threading
import pybullet as p

from sim.simulator import Simulator
from sim.world import World
from sim.lidar import Lidar2D
from sim.sensors import (
    SimulatedIMU,
    SimulatedGPS,
    SimulatedEncoders,
    SimulatedUltrasonic,
    SimulatedMagnetometer,
)

from robot.husky import Husky
from robot.skid_steer import SkidSteerController
from control.teleop_arrow import ArrowTeleop
from control.autonomous_controller import AutonomousController
from gui.control_panel import UGVControlPanel
from estimation.ekf_runner import EKFRunner
from planning.planning_runner import PlanningRunner


class SensorBuffer:
    """Thread-safe buffer for latest sensor reading."""

    def __init__(self):
        self._data = None
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            self._data = data

    def read(self):
        with self._lock:
            return self._data


class RoverSimApp:
    def __init__(self):
        self.dt = 1 / 240
        self.running = True

        # --- Core Systems ---
        self.sim = Simulator(gui=True)
        self.robot = Husky()
        self.world = World()
        self._create_world()

        self.ctrl = SkidSteerController(
            self.robot.id,
            self.robot.left_wheels,
            self.robot.right_wheels,
            wheel_radius=0.165,
            track_width=0.55,
        )

        # --- Sensors ---
        self.lidar = Lidar2D(
            robot_id=self.robot.id, range_max=8.0, fov=180, num_rays=91, height=0.25
        )

        self.imu = SimulatedIMU(self.robot.id)
        self.gps = SimulatedGPS(self.robot.id)
        self.encoders = SimulatedEncoders(
            self.robot.id,
            self.robot.left_wheels,
            self.robot.right_wheels,
            wheel_radius=0.165,
            track_width=0.55,
        )
        self.ultrasonic = SimulatedUltrasonic(self.robot.id)
        self.magnetometer = SimulatedMagnetometer(self.robot.id)

        # --- Sensor Buffers (latest reading from each sensor) ---
        self.imu_buffer = SensorBuffer()
        self.gps_buffer = SensorBuffer()
        self.encoder_buffer = SensorBuffer()
        self.ultrasonic_buffer = SensorBuffer()
        self.lidar_buffer = SensorBuffer()
        self.mag_buffer = SensorBuffer()

        # --- Teleop ---
        self.teleop = ArrowTeleop()

        # --- Start sensor threads ---
        self._start_sensor_threads()

        # --- Start EKF ---
        self.ekf_runner = EKFRunner(
            self.imu_buffer, self.gps_buffer, self.encoder_buffer, self.mag_buffer
        )
        self.ekf_runner.start(init_from_gps=True)

        # --- Start Planning ---
        self.planning_runner = PlanningRunner(
            ekf_runner=self.ekf_runner,
            lidar_buffer=self.lidar_buffer,
            grid_size_m=20.0,
            grid_resolution=0.2,
            inflation_radius=0.4,
        )
        # Set a test goal — change this to wherever you want the robot to go
        self.planning_runner.set_goal(goal_x=5.0, goal_y=3.0)
        self.planning_runner.start()

        # --- Autonomous Controller ---
        self.auto_ctrl = AutonomousController(
            ekf_runner=self.ekf_runner,
            planning_runner=self.planning_runner,
            lidar_buffer=self.lidar_buffer,
        )
        self.auto_ctrl.start()

        # --- GUI Control Panel ---
        self.gui = UGVControlPanel(self)
        self.gui.start()
        print("[GUI] Control panel started")

    def _create_world(self):
        self.world.add_box([2, 0, 0.5], [0.3, 0.3, 0.5])
        self.world.add_box([4, 1, 0.5], [0.4, 0.4, 0.5])
        self.world.add_box([4, -1, 0.5], [0.4, 0.4, 0.5])

    # ------------------------------------------------------------------ #
    #  Sensor Threads                                                      #
    # ------------------------------------------------------------------ #

    def _make_sensor_loop(self, read_fn, buffer, hz, name):
        """Generic sensor loop factory."""
        interval = 1.0 / hz

        def loop():
            while self.running:
                start = time.time()
                try:
                    data = read_fn()
                    buffer.write(data)
                except Exception as e:
                    print(f"[{name}] Error: {e}")

                elapsed = time.time() - start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        return loop

    def _lidar_loop(self):
        """LiDAR runs at 20Hz — also handles drawing."""
        interval = 1.0 / 20
        while self.running:
            start = time.time()
            try:
                _, ray_from, hit_positions = self.lidar.scan()
                self.lidar.draw(ray_from, hit_positions)
                points = self.lidar.get_obstacle_points()
                self.lidar_buffer.write(points)
            except Exception as e:
                print(f"[LiDAR] Error: {e}")

            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _start_sensor_threads(self):
        threads = [
            (
                self._make_sensor_loop(self.imu.read, self.imu_buffer, 100, "IMU"),
                "IMU-Thread",
            ),
            (
                self._make_sensor_loop(
                    self.encoders.read, self.encoder_buffer, 50, "Encoder"
                ),
                "Encoder-Thread",
            ),
            (
                self._make_sensor_loop(
                    self.ultrasonic.read, self.ultrasonic_buffer, 20, "Ultrasonic"
                ),
                "Ultrasonic-Thread",
            ),
            (
                self._make_sensor_loop(self.gps.read, self.gps_buffer, 10, "GPS"),
                "GPS-Thread",
            ),
            (
                self._make_sensor_loop(
                    self.magnetometer.read, self.mag_buffer, 50, "Magnetometer"
                ),
                "Magnetometer-Thread",
            ),
            (self._lidar_loop, "LiDAR-Thread"),
        ]

        self._threads = []
        for target, name in threads:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            print(f"[Init] Started {name}")

    # ------------------------------------------------------------------ #
    #  Main Loop                                                           #
    # ------------------------------------------------------------------ #

    def _update_camera(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot.id)
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=pos
        )

    def _debug_print_sensors(self):
        imu_data = self.imu_buffer.read()
        gps_data = self.gps_buffer.read()
        enc_data = self.encoder_buffer.read()
        us_data = self.ultrasonic_buffer.read()

        if imu_data:
            gyro, accel, _ = imu_data
            print(
                f"[IMU]       gyro_z={gyro[2]:.3f} rad/s  accel_x={accel[0]:.3f} m/s^2"
            )
        if gps_data:
            pos, _ = gps_data
            print(f"[GPS]       x={pos[0]:.2f}  y={pos[1]:.2f}")
        if enc_data:
            lin, ang, vl, vr, _ = enc_data
            print(f"[Encoder]   linear={lin:.3f} m/s  angular={ang:.3f} rad/s")
        if us_data:
            dists, _ = us_data
            print(f"[Ultrasonic] US1={dists[0]:.2f}m  US2={dists[1]:.2f}m")

        mag_data = self.mag_buffer.read()
        if mag_data:
            heading, _ = mag_data
            print(f"[Mag]       heading={math.degrees(heading):.1f}°")

        # Planning path
        path = self.planning_runner.get_path()
        if path:
            print(
                f"[Plan]      waypoints={len(path)}  next=({path[0][0]:.1f}, {path[0][1]:.1f})"
            )

        # Mode
        print(f"[Mode]      {self.auto_ctrl.get_mode()}")
        ekf_state = self.ekf_runner.get_state()
        gt_pos, gt_orn = p.getBasePositionAndOrientation(self.robot.id)
        gt_yaw = p.getEulerFromQuaternion(gt_orn)[2]
        if ekf_state:
            print(
                f"[EKF]       x={ekf_state['x']:.2f} y={ekf_state['y']:.2f} yaw={math.degrees(ekf_state['yaw']):.1f}°  v={ekf_state['v']:.2f} m/s"
            )
            print(
                f"[Truth]     x={gt_pos[0]:.2f} y={gt_pos[1]:.2f} yaw={math.degrees(gt_yaw):.1f}°"
            )

    def step(self):
        keys = p.getKeyboardEvents()

        # Press 'a' to toggle between teleop and autonomous mode
        if ord("a") in keys and (keys[ord("a")] & p.KEY_WAS_TRIGGERED):
            current = self.auto_ctrl.get_mode()
            if current == "TELEOP":
                self.auto_ctrl.set_mode("AUTONOMOUS")
            else:
                self.auto_ctrl.set_mode("TELEOP")

        # Get velocity command based on mode
        if self.auto_ctrl.get_mode() == "AUTONOMOUS":
            v, w = self.auto_ctrl.get_cmd_vel()
        else:
            v, w = self.teleop.update()

        self.ctrl.cmd_vel(v, w)
        self.sim.step()
        self._update_camera()

    def run(self, debug=False):
        debug_timer = time.time()
        try:
            while self.running:
                loop_start = time.time()

                if debug and (time.time() - debug_timer > 1.0):
                    self._debug_print_sensors()
                    debug_timer = time.time()

                self.step()

                elapsed = time.time() - loop_start
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Shutdown] Stopping all threads...")
            self.running = False
            self.ekf_runner.stop()
            self.planning_runner.stop()
            self.auto_ctrl.stop()
