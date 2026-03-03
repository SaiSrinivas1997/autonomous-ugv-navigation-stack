import math
import numpy as np
import threading


class DWAConfig:
    """
    DWA tuning parameters.
    Adjust these to change robot behaviour.
    """

    # --- Robot constraints ---
    max_speed = 1.0  # m/s   max linear speed
    min_speed = -0.3  # m/s   allow slight reverse
    max_yaw_rate = 1.5  # rad/s max rotation speed
    max_accel = 2.0  # m/s²  max linear acceleration (increased)
    max_delta_yaw_rate = 3.0  # rad/s² max angular acceleration (increased)
    v_resolution = 0.1  # m/s   linear velocity sample step (coarser = more samples)
    yaw_rate_resolution = (
        0.2  # rad/s angular velocity sample step (coarser = more samples)
    )

    # --- Simulation ---
    dt = 0.1  # s     time step for trajectory rollout
    predict_time = 2.0  # s     how far ahead to simulate trajectory

    # --- Scoring weights ---
    heading_weight = 2.0  # weight for heading toward goal (increased)
    clearance_weight = 0.5  # weight for obstacle clearance (reduced — open space)
    velocity_weight = 1.0  # weight for forward speed (increased)

    # --- Goal tolerance ---
    goal_tolerance = 0.3  # m     distance to consider goal reached
    waypoint_tolerance = 0.4  # m     distance to advance to next waypoint


class DWA:
    """
    Dynamic Window Approach local planner.

    DWA works by:
    1. Computing dynamic window — reachable (v, w) given current velocity + acceleration limits
    2. Sampling (v, w) pairs within the window
    3. Simulating each trajectory forward in time
    4. Scoring each trajectory: heading + clearance + speed
    5. Outputting best (v, w) command

    Input:  current pose (x, y, yaw, v, yaw_rate), next waypoint, obstacle points
    Output: (v, w) velocity command
    """

    def __init__(self, config: DWAConfig = None):
        self.config = config or DWAConfig()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute(self, state, waypoint, obstacle_points):
        """
        Compute best velocity command to reach waypoint while avoiding obstacles.

        Args:
            state           : dict with keys x, y, yaw, v, yaw_rate (from EKF)
            waypoint        : (goal_x, goal_y) in world frame
            obstacle_points : list of (x, y) in world frame

        Returns:
            (v, w) : linear velocity (m/s) and angular velocity (rad/s)
                     Returns (0, 0) if no safe trajectory found
        """
        with self._lock:
            cfg = self.config

            x = state["x"]
            y = state["y"]
            yaw = state["yaw"]
            v = state["v"]
            yaw_rate = state["yaw_rate"]

            goal_x, goal_y = waypoint

            # --- Dynamic window ---
            dw = self._dynamic_window(v, yaw_rate)

            # --- Sample and score trajectories ---
            best_score = -float("inf")
            best_v = 0.0
            best_w = 0.0
            best_traj = None

            v_samples = np.arange(dw[0], dw[1] + cfg.v_resolution, cfg.v_resolution)
            w_samples = np.arange(
                dw[2], dw[3] + cfg.yaw_rate_resolution, cfg.yaw_rate_resolution
            )

            n_collision = 0
            for v_cmd in v_samples:
                for w_cmd in w_samples:
                    traj = self._simulate_trajectory(x, y, yaw, v_cmd, w_cmd)
                    if self._check_collision(traj, obstacle_points):
                        n_collision += 1
                        continue
                    score = self._score(traj, v_cmd, goal_x, goal_y, obstacle_points)
                    if score > best_score:
                        best_score = score
                        best_v = v_cmd
                        best_w = w_cmd
                        best_traj = traj

            if best_traj is None:
                total = len(v_samples) * len(w_samples)
                print(
                    f"[DWA] v=[{dw[0]:.2f},{dw[1]:.2f}] w=[{dw[2]:.2f},{dw[3]:.2f}] "
                    f"n={total} collisions={n_collision} obs={len(obstacle_points)}"
                )
                return 0.0, 0.0

            return best_v, best_w

    def is_goal_reached(self, state, goal_x, goal_y):
        """Check if robot has reached the goal position."""
        dist = math.sqrt((state["x"] - goal_x) ** 2 + (state["y"] - goal_y) ** 2)
        return dist < self.config.goal_tolerance

    def is_waypoint_reached(self, state, wp_x, wp_y):
        """Check if robot has reached current waypoint — advance to next."""
        dist = math.sqrt((state["x"] - wp_x) ** 2 + (state["y"] - wp_y) ** 2)
        return dist < self.config.waypoint_tolerance

    # ------------------------------------------------------------------ #
    #  Dynamic Window                                                      #
    # ------------------------------------------------------------------ #

    def _dynamic_window(self, v, yaw_rate):
        """
        Compute dynamic window: reachable velocities given current v, w and acceleration limits.
        Always guarantees a minimum window size so robot can start from rest.
        """
        cfg = self.config
        dt = cfg.dt

        v_min = max(cfg.min_speed, v - cfg.max_accel * dt)
        v_max = min(cfg.max_speed, v + cfg.max_accel * dt)
        w_min = max(-cfg.max_yaw_rate, yaw_rate - cfg.max_delta_yaw_rate * dt)
        w_max = min(cfg.max_yaw_rate, yaw_rate + cfg.max_delta_yaw_rate * dt)

        # Guarantee minimum window — prevents empty sample set when starting from rest
        if v_max - v_min < 0.4:
            mid = (v_max + v_min) / 2.0
            v_min = max(cfg.min_speed, mid - 0.2)
            v_max = min(cfg.max_speed, mid + 0.2)
        if w_max - w_min < 0.6:
            mid = (w_max + w_min) / 2.0
            w_min = max(-cfg.max_yaw_rate, mid - 0.3)
            w_max = min(cfg.max_yaw_rate, mid + 0.3)

        return [v_min, v_max, w_min, w_max]

    # ------------------------------------------------------------------ #
    #  Trajectory Simulation                                               #
    # ------------------------------------------------------------------ #

    def _simulate_trajectory(self, x, y, yaw, v, w):
        """
        Simulate robot trajectory for predict_time seconds with constant (v, w).

        Returns list of (x, y, yaw) poses.
        """
        cfg = self.config
        traj = []
        t = 0.0

        while t <= cfg.predict_time:
            traj.append((x, y, yaw))
            x += v * math.cos(yaw) * cfg.dt
            y += v * math.sin(yaw) * cfg.dt
            yaw += w * cfg.dt
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi  # wrap
            t += cfg.dt

        return traj

    # ------------------------------------------------------------------ #
    #  Collision Check                                                     #
    # ------------------------------------------------------------------ #

    def _check_collision(self, traj, obstacle_points, robot_radius=0.3):
        """
        Check if trajectory comes within robot_radius of any obstacle.

        Args:
            traj           : list of (x, y, yaw)
            obstacle_points: list of (ox, oy) in world frame
            robot_radius   : collision radius in meters

        Returns True if collision detected.
        """
        if not obstacle_points:
            return False

        obs = np.array(obstacle_points)  # (N, 2)

        for px, py, _ in traj:
            diff = obs - np.array([px, py])
            dists = np.sqrt((diff**2).sum(axis=1))
            if np.any(dists < robot_radius):
                return True

        return False

    # ------------------------------------------------------------------ #
    #  Scoring                                                             #
    # ------------------------------------------------------------------ #

    def _score(self, traj, v, goal_x, goal_y, obstacle_points):
        """
        Score a trajectory on three criteria:

        1. Heading  — how well final pose faces the goal
        2. Clearance — minimum distance to nearest obstacle
        3. Velocity — reward higher forward speeds
        """
        cfg = self.config

        final_x, final_y, final_yaw = traj[-1]

        # --- Heading score ---
        # Angle from final position to goal
        angle_to_goal = math.atan2(goal_y - final_y, goal_x - final_x)
        heading_error = abs(self._angle_diff(angle_to_goal, final_yaw))
        heading_score = math.pi - heading_error  # max when facing goal

        # --- Clearance score ---
        if obstacle_points:
            obs = np.array(obstacle_points)
            min_dist = float("inf")
            for px, py, _ in traj:
                diff = obs - np.array([px, py])
                dists = np.sqrt((diff**2).sum(axis=1))
                min_dist = min(min_dist, dists.min())
            clearance_score = min(min_dist, 2.0)  # cap at 2m
        else:
            clearance_score = 2.0

        # --- Velocity score ---
        velocity_score = v  # reward forward motion

        # --- Weighted total ---
        total = (
            cfg.heading_weight * heading_score
            + cfg.clearance_weight * clearance_score
            + cfg.velocity_weight * velocity_score
        )

        return total

    @staticmethod
    def _angle_diff(a, b):
        """Compute smallest signed difference between two angles."""
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
