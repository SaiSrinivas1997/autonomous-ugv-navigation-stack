import time
import math
import threading

from control.dwa import DWA, DWAConfig


class AutonomousController:
    """
    Manages autonomous navigation using DWA.

    Responsibilities:
    - Track current waypoint index in D* Lite path
    - Advance to next waypoint when current is reached
    - Call DWA to compute velocity commands
    - Detect goal reached
    - Provide (v, w) commands at 20Hz

    Mode switching:
        TELEOP     — keyboard controls robot directly
        AUTONOMOUS — DWA controls robot toward goal
    """

    MODE_TELEOP = "TELEOP"
    MODE_AUTONOMOUS = "AUTONOMOUS"

    def __init__(self, ekf_runner, planning_runner, lidar_buffer):
        self.ekf_runner = ekf_runner
        self.planning_runner = planning_runner
        self.lidar_buffer = lidar_buffer

        self.dwa = DWA(DWAConfig())
        self.mode = self.MODE_TELEOP

        # Waypoint tracking
        self._waypoint_index = 0
        self._current_path = []
        self._goal_reached = False

        # Current command output
        self._v = 0.0
        self._w = 0.0
        self._lock = threading.Lock()

        self.running = False
        self._thread = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_mode(self, mode):
        """Switch between TELEOP and AUTONOMOUS."""
        with self._lock:
            self.mode = mode
            if mode == self.MODE_AUTONOMOUS:
                self._waypoint_index = 0
                self._goal_reached = False
                self._current_path = []
                print("[AutoCtrl] Switched to AUTONOMOUS mode")
            else:
                self._v = 0.0
                self._w = 0.0
                print("[AutoCtrl] Switched to TELEOP mode")

    def get_cmd_vel(self):
        """Get current velocity command. Thread safe."""
        with self._lock:
            return self._v, self._w

    def is_goal_reached(self):
        with self._lock:
            return self._goal_reached

    def get_mode(self):
        with self._lock:
            return self.mode

    def start(self):
        self.running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="AutoCtrl-Thread", daemon=True
        )
        self._thread.start()
        print("[AutoCtrl] Started")

    def stop(self):
        self.running = False

    # ------------------------------------------------------------------ #
    #  Control Loop                                                        #
    # ------------------------------------------------------------------ #

    def _run_loop(self):
        """
        Autonomous control loop at 20Hz.

        Each iteration:
        1. Check mode — skip if TELEOP
        2. Get EKF state
        3. Get latest path from planning runner
        4. Get current waypoint
        5. Check if waypoint reached → advance
        6. Get obstacle points from LiDAR
        7. Call DWA → get (v, w)
        8. Store command
        """
        interval = 1.0 / 20  # 20Hz

        while self.running:
            loop_start = time.time()

            with self._lock:
                mode = self.mode

            if mode != self.MODE_AUTONOMOUS:
                time.sleep(interval)
                continue

            # --- 1. Get EKF state ---
            state = self.ekf_runner.get_state()
            if state is None:
                time.sleep(interval)
                continue

            # --- 2. Get latest path ---
            new_path = self.planning_runner.get_path()
            with self._lock:
                if new_path and new_path != self._current_path:
                    self._current_path = new_path
                    # Find closest waypoint ahead instead of always starting at 0
                    # This prevents re-visiting already passed waypoints
                    self._waypoint_index = self._find_closest_waypoint(state, new_path)

                path = list(self._current_path)
                index = self._waypoint_index

            if not path:
                # No path available — stop
                with self._lock:
                    self._v, self._w = 0.0, 0.0
                time.sleep(interval)
                continue

            # --- 3. Get current waypoint ---
            # Skip waypoints already behind us
            while index < len(path) - 1:
                wp_x, wp_y = path[index]
                if self.dwa.is_waypoint_reached(state, wp_x, wp_y):
                    index += 1
                    print(f"[AutoCtrl] Waypoint {index}/{len(path)} reached")
                else:
                    break

            with self._lock:
                self._waypoint_index = index

            wp_x, wp_y = path[index]

            # --- 4. Check if final goal reached ---
            goal = self.planning_runner.planner.get_goal()
            if goal and self.dwa.is_goal_reached(state, goal[0], goal[1]):
                print("[AutoCtrl] GOAL REACHED!")
                with self._lock:
                    self._goal_reached = True
                    self._v, self._w = 0.0, 0.0
                    self.mode = self.MODE_TELEOP
                continue

            # --- 5. Get obstacle points from LiDAR (robot local frame) ---
            lidar_points = self.lidar_buffer.read() or []

            # Convert LiDAR local frame to world frame for DWA
            world_obstacles = self._local_to_world(
                lidar_points, state["x"], state["y"], state["yaw"]
            )

            # --- 6. DWA compute ---
            v, w = self.dwa.compute(state, (wp_x, wp_y), world_obstacles)

            if abs(v) < 0.01 and abs(w) < 0.01:
                print(
                    f"[AutoCtrl] DWA returned zero cmd — wp=({wp_x:.1f},{wp_y:.1f}) robot=({state['x']:.1f},{state['y']:.1f})"
                )

            with self._lock:
                self._v = v
                self._w = w

            # --- Sleep ---
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _find_closest_waypoint(self, state, path):
        """
        Find the first waypoint that is NOT already within waypoint tolerance.
        Skips waypoints behind or too close to robot.
        """
        for i, (wx, wy) in enumerate(path):
            dist = math.sqrt((state["x"] - wx) ** 2 + (state["y"] - wy) ** 2)
            if dist > self.dwa.config.waypoint_tolerance:
                return i
        return len(path) - 1  # all waypoints close — aim for last one

    def _local_to_world(self, local_points, robot_x, robot_y, robot_yaw):
        """Convert points from robot local frame to world frame."""
        world_points = []
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        for lx, ly in local_points:
            wx = robot_x + cos_yaw * lx - sin_yaw * ly
            wy = robot_y + sin_yaw * lx + cos_yaw * ly
            world_points.append((wx, wy))
        return world_points
