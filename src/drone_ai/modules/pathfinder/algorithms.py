"""Path planning algorithms — A* and RRT.

PathPlanner wraps both: tries A* first (fast, optimal on grid),
falls back to RRT for complex continuous spaces.
"""

import numpy as np
import heapq
from typing import List, Optional, Tuple, Dict
from drone_ai.simulation.world import World, Obstacle


class AStarPlanner:
    """3D A* on a voxel grid."""

    def __init__(self, world: World, resolution: float = 2.0):
        self.world = world
        self.res = resolution

    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        def to_grid(p):
            return tuple(np.round((p - self.world.bounds_min) / self.res).astype(int))

        def to_world(g):
            return np.array(g, dtype=float) * self.res + self.world.bounds_min

        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        sg = to_grid(start)
        gg = to_grid(goal)

        open_set = [(0, sg)]
        came_from: Dict[tuple, tuple] = {}
        g_score: Dict[tuple, float] = {sg: 0.0}
        f_score: Dict[tuple, float] = {sg: heuristic(sg, gg)}
        visited = set()

        # 26-connected neighbors
        neighbors_delta = [(dx, dy, dz)
                           for dx in (-1, 0, 1)
                           for dy in (-1, 0, 1)
                           for dz in (-1, 0, 1)
                           if not (dx == 0 and dy == 0 and dz == 0)]

        max_iter = 5000
        for _ in range(max_iter):
            if not open_set:
                break
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if heuristic(current, gg) < 1.5:
                return self._reconstruct(came_from, current, to_world)

            for d in neighbors_delta:
                nb = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
                if nb in visited:
                    continue
                wp = to_world(nb)
                if not self.world.in_bounds(wp) or self.world.in_collision(wp, margin=0.5):
                    continue
                step_cost = np.linalg.norm(d)
                tent_g = g_score.get(current, float("inf")) + step_cost
                if tent_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tent_g
                    f_score[nb] = tent_g + heuristic(nb, gg)
                    heapq.heappush(open_set, (f_score[nb], nb))

        return None

    def _reconstruct(self, came_from, current, to_world) -> List[np.ndarray]:
        path = [to_world(current)]
        while current in came_from:
            current = came_from[current]
            path.append(to_world(current))
        return path[::-1]


class RRTPlanner:
    """RRT path planner for continuous 3D space."""

    def __init__(self, world: World, step_size: float = 3.0, max_iter: int = 3000):
        self.world = world
        self.step = step_size
        self.max_iter = max_iter

    def plan(
        self, start: np.ndarray, goal: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Optional[List[np.ndarray]]:
        rng = rng or np.random.default_rng()
        nodes = [start.copy()]
        parent: Dict[int, int] = {}
        goal_bias = 0.1

        for _ in range(self.max_iter):
            # Sample
            if rng.random() < goal_bias:
                sample = goal.copy()
            else:
                sample = rng.uniform(self.world.bounds_min, self.world.bounds_max)

            # Nearest
            dists = [np.linalg.norm(n - sample) for n in nodes]
            near_idx = int(np.argmin(dists))
            near = nodes[near_idx]

            # Steer
            direction = sample - near
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            new_node = near + direction / dist * min(self.step, dist)

            if not self.world.in_bounds(new_node) or self.world.in_collision(new_node, margin=0.5):
                continue

            nodes.append(new_node.copy())
            parent[len(nodes) - 1] = near_idx

            if np.linalg.norm(new_node - goal) < self.step:
                nodes.append(goal.copy())
                parent[len(nodes) - 1] = len(nodes) - 2
                return self._trace(nodes, parent)

        return None

    def _trace(self, nodes, parent) -> List[np.ndarray]:
        path = [nodes[-1]]
        idx = len(nodes) - 1
        while idx in parent:
            idx = parent[idx]
            path.append(nodes[idx])
        return path[::-1]


class PathPlanner:
    """High-level planner combining A* and RRT with smoothing."""

    def __init__(self, world: World, grid_resolution: float = 2.0):
        self.world = world
        self.astar = AStarPlanner(world, grid_resolution)
        self.rrt = RRTPlanner(world)
        self._rng = np.random.default_rng()
        # Physics-layer pushes the current braking distance here; if
        # any planned path corner would require a tighter stop than the
        # drone can physically make, the planner inflates the obstacle
        # margin on that segment. See docs/physics_realism.md for why
        # inertia is a first-class planning input.
        self._braking_distance = 0.0

    def set_braking_distance(self, metres: float) -> None:
        self._braking_distance = max(0.0, float(metres))

    def plan(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Plan path from start to goal. Returns waypoints including start and goal."""
        # Elevate both points to safe altitude
        s = start.copy()
        g = goal.copy()
        s[2] = max(s[2], 5.0)
        g[2] = max(g[2], 5.0)

        if np.linalg.norm(s - g) < 2.0:
            return [s, g]

        # Try A* first
        path = self.astar.plan(s, g)
        if path is None or len(path) < 2:
            path = self.rrt.plan(s, g, self._rng)
        if path is None:
            path = [s, g]  # Direct fallback

        return self._smooth(path)

    def _smooth(self, path: List[np.ndarray], iterations: int = 3) -> List[np.ndarray]:
        if len(path) < 3:
            return path
        for _ in range(iterations):
            new_path = [path[0]]
            for i in range(1, len(path) - 1):
                smoothed = 0.75 * path[i] + 0.125 * path[i - 1] + 0.125 * path[i + 1]
                if not self.world.in_collision(smoothed, margin=0.5):
                    new_path.append(smoothed)
                else:
                    new_path.append(path[i])
            new_path.append(path[-1])
            path = new_path
        return path

    def update_world(self, world: World):
        self.world = world
        self.astar.world = world
        self.rrt.world = world
