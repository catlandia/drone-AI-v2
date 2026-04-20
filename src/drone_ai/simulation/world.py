"""World environment — obstacles, bounds, and spatial queries."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Obstacle:
    position: np.ndarray       # center (x, y, z)
    size: np.ndarray           # half-extents (rx, ry, rz) for box
    obstacle_type: str = "box" # box | sphere | cylinder
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def contains(self, point: np.ndarray, margin: float = 0.3) -> bool:
        if self.obstacle_type == "sphere":
            r = self.size[0] + margin
            return float(np.linalg.norm(point - self.position)) < r
        else:
            diff = np.abs(point - self.position)
            return bool(np.all(diff < self.size + margin))

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        if self.obstacle_type == "sphere":
            d = point - self.position
            dist = np.linalg.norm(d)
            if dist < 1e-6:
                return self.position + np.array([self.size[0], 0.0, 0.0])
            return self.position + d / dist * self.size[0]
        else:
            return np.clip(point, self.position - self.size, self.position + self.size)

    def distance_to(self, point: np.ndarray) -> float:
        cp = self.closest_point(point)
        return float(np.linalg.norm(point - cp))


class World:
    """3D world with obstacles and spatial queries."""

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray] = (
            np.array([-200.0, -200.0, 0.0]),
            np.array([200.0, 200.0, 150.0]),
        ),
    ):
        self.bounds_min, self.bounds_max = bounds
        self.obstacles: List[Obstacle] = []

    def add_obstacle(self, obs: Obstacle):
        self.obstacles.append(obs)

    def set_obstacles(self, obstacles: List[Obstacle]):
        self.obstacles = list(obstacles)

    def clear(self):
        self.obstacles.clear()

    def in_collision(self, point: np.ndarray, margin: float = 0.3) -> bool:
        return any(o.contains(point, margin) for o in self.obstacles)

    def nearest_obstacle(self, point: np.ndarray) -> Tuple[Optional[Obstacle], float]:
        if not self.obstacles:
            return None, float("inf")
        dists = [o.distance_to(point) for o in self.obstacles]
        idx = int(np.argmin(dists))
        return self.obstacles[idx], dists[idx]

    def obstacles_in_radius(self, point: np.ndarray, radius: float) -> List[Obstacle]:
        return [o for o in self.obstacles if o.distance_to(point) < radius]

    def in_bounds(self, point: np.ndarray) -> bool:
        return bool(np.all(point >= self.bounds_min) and np.all(point <= self.bounds_max))

    def random_free_point(
        self,
        rng: np.random.Generator,
        z_min: float = 5.0,
        z_max: float = 30.0,
        margin: float = 2.0,
        max_tries: int = 200,
    ) -> np.ndarray:
        for _ in range(max_tries):
            p = rng.uniform(
                [self.bounds_min[0], self.bounds_min[1], z_min],
                [self.bounds_max[0], self.bounds_max[1], z_max],
            )
            if not self.in_collision(p, margin):
                return p
        return np.array([0.0, 0.0, max(z_min, 10.0)])

    def generate_random_obstacles(
        self,
        n: int,
        rng: np.random.Generator,
        min_size: float = 1.0,
        max_size: float = 8.0,
    ) -> List[Obstacle]:
        obs = []
        for _ in range(n):
            pos = rng.uniform(
                self.bounds_min + max_size,
                self.bounds_max - max_size,
            )
            pos[2] = rng.uniform(0.0, 40.0)
            sz = rng.uniform(min_size, max_size, size=3)
            sz[2] = rng.uniform(5.0, 40.0)  # tall obstacles
            obs.append(Obstacle(position=pos, size=sz))
        self.obstacles.extend(obs)
        return obs
