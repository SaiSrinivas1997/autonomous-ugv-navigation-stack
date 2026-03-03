import math
import threading
import pybullet as p


class Lidar2D:
    def __init__(self, robot_id, range_max=8.0, fov=180, num_rays=91, height=0.25):

        self.robot_id = robot_id
        self.range_max = range_max
        self.fov = math.radians(fov)
        self.num_rays = num_rays
        self.height = height

        # Thread-safe cached scan
        self._lock = threading.Lock()
        self._cached_distances = []
        self._cached_ray_from = []
        self._cached_hit_positions = []
        self._cached_obstacle_points = []

    def scan(self):
        """
        Performs a LiDAR scan and updates internal cache.
        Call this from a dedicated LiDAR thread at 20Hz.
        """
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        ray_from = []
        ray_to = []

        start_angle = -self.fov / 2
        angle_step = self.fov / (self.num_rays - 1)

        for i in range(self.num_rays):
            angle = yaw + start_angle + i * angle_step

            # Start rays from just outside robot body (0.6m from center)
            # prevents rays from hitting robot's own chassis
            ray_start_dist = 0.6
            from_pt = [
                pos[0] + ray_start_dist * math.cos(angle),
                pos[1] + ray_start_dist * math.sin(angle),
                pos[2] + self.height,
            ]

            to_pt = [
                pos[0] + self.range_max * math.cos(angle),
                pos[1] + self.range_max * math.sin(angle),
                pos[2] + self.height,
            ]

            ray_from.append(from_pt)
            ray_to.append(to_pt)

        results = p.rayTestBatch(ray_from, ray_to)

        distances = []
        hit_positions = []
        obstacle_points = []

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        for res in results:
            hit_fraction = res[2]
            hit_pos = res[3]

            distance = (
                hit_fraction * self.range_max if hit_fraction < 1.0 else self.range_max
            )
            distances.append(distance)
            hit_positions.append(hit_pos)

            # Filter hits:
            # 1. Must be above ground (z > 0.05) — removes floor hits
            # 2. Must be farther than robot body radius (0.6m) — removes self-hits
            if hit_pos is not None and hit_fraction < 1.0:
                hit_z = hit_pos[2]
                dx = hit_pos[0] - pos[0]
                dy = hit_pos[1] - pos[1]
                dist_2d = math.sqrt(dx * dx + dy * dy)

                if hit_z > 0.05 and dist_2d > 0.6:
                    x_local = cos_yaw * dx + sin_yaw * dy
                    y_local = -sin_yaw * dx + cos_yaw * dy
                    obstacle_points.append((x_local, y_local))

        # Update cache safely
        with self._lock:
            self._cached_distances = distances
            self._cached_ray_from = ray_from
            self._cached_hit_positions = hit_positions
            self._cached_obstacle_points = obstacle_points

        return distances, ray_from, hit_positions

    def draw(self, ray_from, hit_positions):
        """Draw LiDAR rays in PyBullet GUI."""
        if not hasattr(self, "line_ids"):
            self.line_ids = []
            for start, hit in zip(ray_from, hit_positions):
                line_id = p.addUserDebugLine(start, hit, [1, 0, 0], lineWidth=1)
                self.line_ids.append(line_id)
        else:
            for i, (start, hit) in enumerate(zip(ray_from, hit_positions)):
                p.addUserDebugLine(
                    start,
                    hit,
                    [1, 0, 0],
                    lineWidth=1,
                    replaceItemUniqueId=self.line_ids[i],
                )

    def get_cached_scan(self):
        """Returns last cached scan — safe to call from any thread."""
        with self._lock:
            return (list(self._cached_ray_from), list(self._cached_hit_positions))

    def get_obstacle_points(self):
        """
        Returns cached obstacle points in robot local frame.
        No scan is triggered — uses last cached result.
        """
        with self._lock:
            return list(self._cached_obstacle_points)
