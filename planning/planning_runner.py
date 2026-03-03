import time
import math
import threading

from planning.rolling_grid import RollingGrid
from planning.dstar_lite import DStarLite


class PlanningRunner:
    """
    Manages rolling grid updates and D* Lite replanning in a thread.

    Responsibilities:
    - Update rolling grid from LiDAR at 10Hz
    - Replan with D* Lite when grid changes or robot moves
    - Provide current waypath to DWA

    Threading model:
        - Grid update + replan at 10Hz (matches LiDAR rate)
    """

    def __init__(
        self,
        ekf_runner,
        lidar_buffer,
        grid_size_m=20.0,
        grid_resolution=0.2,
        inflation_radius=0.4,
        replan_distance=0.5,
    ):
        """
        Args:
            ekf_runner       : EKFRunner instance for current pose
            lidar_buffer     : SensorBuffer with LiDAR obstacle points
            grid_size_m      : rolling grid size in meters
            grid_resolution  : meters per cell
            inflation_radius : obstacle inflation for robot footprint
            replan_distance  : replan if robot moved more than this (meters)
        """
        self.ekf_runner = ekf_runner
        self.lidar_buffer = lidar_buffer
        self.replan_distance = replan_distance

        # Core planning components
        self.grid = RollingGrid(grid_size_m, grid_resolution, inflation_radius)
        self.planner = DStarLite(self.grid)

        # State
        self.running = False
        self._path = []
        self._lock = threading.Lock()
        self._thread = None

        # Track last robot position for replan trigger
        self._last_replan_x = None
        self._last_replan_y = None

    def set_goal(self, goal_x, goal_y):
        """Set navigation goal in world coordinates."""
        self.planner.set_goal(goal_x, goal_y)

    def get_path(self):
        """Get current planned path as list of (x,y) waypoints. Thread safe."""
        with self._lock:
            return list(self._path)

    def get_grid(self):
        """Get current grid for visualization."""
        return self.grid

    def start(self):
        self.running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="Planning-Thread", daemon=True
        )
        self._thread.start()
        print("[PlanningRunner] Started")

    def stop(self):
        self.running = False

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _should_replan(self, robot_x, robot_y):
        """Trigger replan if robot moved enough or no path exists."""
        if not self._path:
            return True
        if self._last_replan_x is None:
            return True
        dist = math.sqrt(
            (robot_x - self._last_replan_x) ** 2 + (robot_y - self._last_replan_y) ** 2
        )
        return dist >= self.replan_distance

    def _convert_lidar_to_world(self, lidar_points, robot_x, robot_y, robot_yaw):
        """
        Convert LiDAR obstacle points from robot local frame to world frame.

        LiDAR points come in robot frame (x forward, y left).
        Need to rotate by robot yaw and translate by robot position.
        """
        world_points = []
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        for lx, ly in lidar_points:
            wx = robot_x + cos_yaw * lx - sin_yaw * ly
            wy = robot_y + sin_yaw * lx + cos_yaw * ly
            world_points.append((wx, wy))

        return world_points

    def _run_loop(self):
        """
        Planning loop at 10Hz.

        Each iteration:
        1. Get latest EKF pose
        2. Get latest LiDAR obstacle points
        3. Update rolling grid
        4. Replan if needed
        """
        interval = 1.0 / 10  # 10Hz

        while self.running:
            loop_start = time.time()

            # --- 1. Get current robot pose from EKF ---
            state = self.ekf_runner.get_state()
            if state is None:
                time.sleep(interval)
                continue

            robot_x = state["x"]
            robot_y = state["y"]
            robot_yaw = state["yaw"]

            # --- 2. Get LiDAR obstacle points ---
            lidar_points = self.lidar_buffer.read()
            if lidar_points is None:
                lidar_points = []

            # --- 3. Convert LiDAR to world frame ---
            world_obstacles = self._convert_lidar_to_world(
                lidar_points, robot_x, robot_y, robot_yaw
            )

            # --- 4. Update rolling grid ---
            self.grid.update(robot_x, robot_y, world_obstacles)

            # --- 5. Replan if needed ---
            if self._should_replan(robot_x, robot_y):
                new_path = self.planner.replan(robot_x, robot_y)
                with self._lock:
                    self._path = new_path
                self._last_replan_x = robot_x
                self._last_replan_y = robot_y

                if new_path:
                    print(f"[PlanningRunner] Path found: {len(new_path)} waypoints")

            # --- Sleep ---
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
