import math
import threading
import numpy as np

# Cell states
FREE = 0
OCCUPIED = 1
UNKNOWN = 2


class RollingGrid:
    """
    A rolling occupancy grid centered on the robot.

    - Grid moves with the robot (rolling window)
    - Updated from LiDAR obstacle points
    - Used by D* Lite as cost map

    Grid convention:
        - Cell (row, col) maps to world (x, y)
        - Origin is always robot position
        - Cells outside LiDAR range = UNKNOWN
        - Cells with obstacles = OCCUPIED
        - Cells with clear LiDAR reading = FREE
    """

    def __init__(self, size_m=20.0, resolution=0.2, inflation_radius=0.4):
        """
        Args:
            size_m          : grid size in meters (square)
            resolution      : meters per cell
            inflation_radius: obstacle inflation in meters (robot footprint buffer)
        """
        self.size_m = size_m
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.inflation_cells = int(inflation_radius / resolution)

        # Grid dimensions
        self.num_cells = int(size_m / resolution)  # e.g. 100x100

        # Grid data — start all UNKNOWN
        self.grid = np.full((self.num_cells, self.num_cells), UNKNOWN, dtype=np.uint8)

        # Current robot world position (grid center)
        self.robot_x = 0.0
        self.robot_y = 0.0

        # Thread safety
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Coordinate Conversion                                               #
    # ------------------------------------------------------------------ #

    def world_to_cell(self, wx, wy):
        """Convert world coordinates to grid cell (row, col)."""
        half = self.size_m / 2.0
        col = int((wx - self.robot_x + half) / self.resolution)
        row = int((wy - self.robot_y + half) / self.resolution)
        return row, col

    def cell_to_world(self, row, col):
        """Convert grid cell (row, col) to world coordinates."""
        half = self.size_m / 2.0
        wx = col * self.resolution - half + self.robot_x
        wy = row * self.resolution - half + self.robot_y
        return wx, wy

    def is_valid_cell(self, row, col):
        return 0 <= row < self.num_cells and 0 <= col < self.num_cells

    # ------------------------------------------------------------------ #
    #  Update                                                              #
    # ------------------------------------------------------------------ #

    def update(self, robot_x, robot_y, obstacle_points_world):
        """
        Update rolling grid with new robot position and LiDAR obstacle points.
        """
        with self._lock:
            self.robot_x = robot_x
            self.robot_y = robot_y

            # Reset to UNKNOWN
            self.grid[:] = UNKNOWN

            # Mark entire LiDAR coverage area as FREE using numpy
            # Much faster than nested Python loop
            center = self.num_cells // 2
            lidar_range_cells = int(8.0 / self.resolution)

            r_min = max(0, center - lidar_range_cells)
            r_max = min(self.num_cells, center + lidar_range_cells + 1)
            c_min = max(0, center - lidar_range_cells)
            c_max = min(self.num_cells, center + lidar_range_cells + 1)

            # Create distance mask using numpy meshgrid
            rows = np.arange(r_min, r_max) - center
            cols = np.arange(c_min, c_max) - center
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            dist = np.sqrt(rr**2 + cc**2) * self.resolution
            mask = dist <= 8.0

            self.grid[r_min:r_max, c_min:c_max][mask] = FREE

            # Mark obstacle cells as OCCUPIED with inflation
            for wx, wy in obstacle_points_world:
                row, col = self.world_to_cell(wx, wy)
                if not self.is_valid_cell(row, col):
                    continue

                r0 = max(0, row - self.inflation_cells)
                r1 = min(self.num_cells, row + self.inflation_cells + 1)
                c0 = max(0, col - self.inflation_cells)
                c1 = min(self.num_cells, col + self.inflation_cells + 1)
                self.grid[r0:r1, c0:c1] = OCCUPIED

    def get_cell_state(self, row, col):
        """Get state of a cell. Thread safe."""
        with self._lock:
            if not self.is_valid_cell(row, col):
                return UNKNOWN
            return int(self.grid[row, col])

    def get_grid_copy(self):
        """Get a copy of the full grid. Thread safe."""
        with self._lock:
            return self.grid.copy(), self.robot_x, self.robot_y

    def is_occupied(self, row, col):
        return self.get_cell_state(row, col) == OCCUPIED

    def is_free(self, row, col):
        return self.get_cell_state(row, col) == FREE
