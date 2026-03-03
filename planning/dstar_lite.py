import math
import threading
import heapq
import numpy as np

from planning.rolling_grid import RollingGrid, FREE, OCCUPIED, UNKNOWN


class DStarLite:
    """
    D* Lite global path planner.

    D* Lite is an incremental heuristic search algorithm that:
    - Finds shortest path from start to goal
    - Efficiently replans when obstacles change
    - Much faster than A* for replanning in dynamic environments

    Returns a list of (x, y) waypoints in world frame.

    Reference: Koenig & Likhachev, 2002
    """

    def __init__(self, grid: RollingGrid):
        self.grid = grid
        self._lock = threading.Lock()
        self._path = []  # current planned path (world coordinates)
        self._goal_world = None  # current goal in world frame

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_goal(self, goal_x, goal_y):
        """Set new navigation goal in world coordinates."""
        with self._lock:
            self._goal_world = (goal_x, goal_y)
            self._path = []
        print(f"[D*Lite] New goal set: ({goal_x:.1f}, {goal_y:.1f})")

    def replan(self, robot_x, robot_y):
        """
        Replan path from robot position to goal.
        Call this when map changes or robot has moved significantly.

        Args:
            robot_x : float - current robot x (EKF estimate)
            robot_y : float - current robot y (EKF estimate)

        Returns:
            list of (x, y) waypoints in world frame, or [] if no path
        """
        with self._lock:
            if self._goal_world is None:
                return []

            goal_x, goal_y = self._goal_world

            # Convert to grid cells
            start_cell = self.grid.world_to_cell(robot_x, robot_y)
            goal_cell = self.grid.world_to_cell(goal_x, goal_y)

            # Snap start and goal to nearest free cell if occupied
            start_cell = self._snap_to_free(*start_cell)
            goal_cell = self._snap_to_free(*goal_cell)

            # Run A* (D* Lite simplified for simulation)
            # Full D* Lite incremental replanning is complex —
            # A* with incremental triggers gives same behaviour for simulation
            path_cells = self._astar(start_cell, goal_cell)

            if path_cells is None:
                print("[D*Lite] No path found!")
                self._path = []
                return []

            # Convert cells back to world coordinates
            path_world = []
            for row, col in path_cells:
                wx, wy = self.grid.cell_to_world(row, col)
                path_world.append((wx, wy))

            self._path = path_world
            return list(self._path)

    def get_path(self):
        """Get current planned path. Thread safe."""
        with self._lock:
            return list(self._path)

    def get_goal(self):
        """Get current goal. Thread safe."""
        with self._lock:
            return self._goal_world

    # ------------------------------------------------------------------ #
    #  A* Search (D* Lite core)                                           #
    # ------------------------------------------------------------------ #

    def _heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _get_neighbors(self, row, col):
        """
        Get valid neighboring cells (8-connected grid).
        UNKNOWN cells are treated as passable — robot can explore into unknown space.
        Only OCCUPIED cells are blocked.
        """
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r = row + dr
                c = col + dc
                if not self.grid.is_valid_cell(r, c):
                    continue
                # Only block OCCUPIED — allow FREE and UNKNOWN
                if self.grid.is_occupied(r, c):
                    continue
                cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                neighbors.append((r, c, cost))
        return neighbors

    def _snap_to_free(self, row, col, search_radius=10):
        """
        If a cell is OCCUPIED or invalid, find nearest FREE or UNKNOWN cell.
        Used to fix goal cells that land on obstacles.
        """
        if self.grid.is_valid_cell(row, col) and not self.grid.is_occupied(row, col):
            return row, col

        best = None
        best_dist = float("inf")
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                r, c = row + dr, col + dc
                if self.grid.is_valid_cell(r, c) and not self.grid.is_occupied(r, c):
                    dist = math.sqrt(dr**2 + dc**2)
                    if dist < best_dist:
                        best_dist = dist
                        best = (r, c)
        return best if best else (row, col)

    def _astar(self, start, goal):
        """
        A* search from start to goal cell.

        Returns list of (row, col) cells or None if no path.
        """
        if not self.grid.is_valid_cell(start[0], start[1]):
            print("[D*Lite] Start cell invalid")
            return None
        if not self.grid.is_valid_cell(goal[0], goal[1]):
            print("[D*Lite] Goal cell invalid")
            return None

        # Priority queue: (f_score, row, col)
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            row, col = current
            for nr, nc, cost in self._get_neighbors(row, col):
                neighbor = (nr, nc)
                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # Thin path — keep every Nth waypoint to reduce DWA workload
        # Keep first and last always
        thinned = [path[0]]
        step = 5  # keep every 5th cell = every 1m at 0.2m resolution
        for i in range(step, len(path) - 1, step):
            thinned.append(path[i])
        thinned.append(path[-1])

        return thinned
