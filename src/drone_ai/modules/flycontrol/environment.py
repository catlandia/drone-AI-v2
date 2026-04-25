"""FlyControl Gymnasium environment.

All observations are restricted to what a FULLY OFFLINE drone could sense
on its own — no GPS, no satellites, no cloud map data. Every slot below
is annotated with the real sensor that would produce it in the field.

Observation (25D):
  [0:3]   position_vio   — displacement since takeoff (VIO: camera+IMU).
                            NOT absolute world XY. The drone does not know
                            where it is on Earth, only how far it has
                            drifted from the power-on point. Drifts slowly.
  [3:6]   velocity       — linear velocity (VIO + IMU fusion).
  [6:9]   acceleration   — linear acceleration (IMU).  Added so the policy
                            can feel "still decelerating, don't command
                            full stop yet."
  [9:12]  orientation    — roll, pitch, yaw (IMU + magnetometer).
  [12:15] angular_velocity — gyro (IMU), very accurate.
  [15:18] target_relative — (target - current_position). Target is a
                            waypoint loaded from the flight plan BEFORE
                            takeoff, expressed in the same VIO frame.
  [18]    distance_to_target — derived, |target_relative|.
  [19]    battery        — onboard battery monitor (voltage-derived SoC).
  [20]    battery_temp   — battery temperature sensor, normalized.
  [21]    barometer_alt  — altitude above takeoff (barometer, drifts
                            on its own axis vs VIO z).
  [22]    braking_dist   — current braking distance estimate from
                            velocity — physics-layer helper so the policy
                            (and upstream planners) can reason about
                            inertia.
  [23]    nearest_obstacle_dist — camera/ToF depth, clipped at 20m.
  [24]    carrying_package — internal state bit (cargo released or not).

Action (4D): motor commands [0, 1]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List

from drone_ai.simulation.physics import QuadrotorPhysics, DroneState
from drone_ai.simulation.world import World, Obstacle


class TaskType(Enum):
    HOVER = "hover"
    DELIVERY = "delivery"
    DELIVERY_ROUTE = "delivery_route"
    DEPLOYMENT = "deployment"


OBS_DIM = 25
ACT_DIM = 4


class FlyControlEnv(gym.Env):
    """FlyControl — low-level motor control environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task: TaskType = TaskType.HOVER,
        difficulty: float = 0.5,
        domain_randomization: bool = False,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        world: Optional[World] = None,
    ):
        super().__init__()
        self.task = task
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self.domain_rand = domain_randomization
        self.render_mode = render_mode

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.physics = QuadrotorPhysics()
        self.world = world or World()

        # Episode state
        self.target: np.ndarray = np.zeros(3)
        self.waypoints: List[np.ndarray] = []
        self.waypoint_idx: int = 0
        self.carrying: bool = False
        self.pickup: np.ndarray = np.zeros(3)
        self.dropzone: np.ndarray = np.zeros(3)
        self.deliveries_done: int = 0
        self.deliveries_total: int = 0
        self._step: int = 0
        self._max_steps: int = 1500
        self.position_history: List[np.ndarray] = []
        # Takeoff origin — the drone's VIO frame anchors here. The policy
        # only ever sees positions RELATIVE to this, because a real offline
        # drone with no GPS cannot know its absolute world coordinates.
        self._start_position: np.ndarray = np.zeros(3)

        self._renderer = None

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.domain_rand:
            self.physics.MASS = float(self._rng.uniform(0.4, 0.6))
            self.physics.G = float(self._rng.uniform(9.5, 10.1))
            # Realistic-ish outdoor wind + gust budget so deployment-stage
            # policies learn to hold station against it.
            wind_mean = self._rng.uniform(-4.0, 4.0, size=3).astype(np.float32)
            wind_mean[2] *= 0.25  # vertical gusts are smaller than horizontal
            gust_std = float(self._rng.uniform(0.0, 1.5))
            self.physics.set_wind(wind_mean, gust_std=gust_std)
            # Small probability of prop wear / failure at deployment.
            self.physics.set_wear(True, rate=1e-7, failure_prob=1e-7)
        else:
            self.physics.set_wind(np.zeros(3), gust_std=0.0)
            self.physics.set_wear(False)

        start = self._rng.uniform([-5, -5, 2], [5, 5, 8]).astype(np.float32)
        self.physics.reset(position=start)
        self._start_position = start.copy()
        self._step = 0
        self.carrying = False
        self.position_history = [start.copy()]

        self._setup_task()
        return self._observe(), {}

    def _setup_task(self):
        t = self.task
        rng = self._rng
        d = self.difficulty

        if t == TaskType.HOVER:
            height = float(rng.uniform(5.0, 5.0 + d * 15.0))
            self.target = np.array([
                float(rng.uniform(-d * 10, d * 10)),
                float(rng.uniform(-d * 10, d * 10)),
                height,
            ], dtype=np.float32)
            self._max_steps = 1500

        elif t == TaskType.DELIVERY:
            self.pickup = np.array([
                float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5)), 0.0
            ], dtype=np.float32)
            self.dropzone = np.array([
                float(rng.uniform(-50 * d, 50 * d)),
                float(rng.uniform(-50 * d, 50 * d)),
                0.0,
            ], dtype=np.float32)
            self.target = self.pickup.copy()
            self.target[2] = 5.0
            self._max_steps = 2500

        elif t in (TaskType.DELIVERY_ROUTE, TaskType.DEPLOYMENT):
            n_deliveries = int(2 + d * 4)
            self.deliveries_total = n_deliveries
            self.deliveries_done = 0
            self.pickup = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            n_obs = int(5 + d * 20)
            self.world.clear()
            self.world.generate_random_obstacles(n_obs, rng)
            self.waypoints = []
            for _ in range(n_deliveries):
                wp = self.world.random_free_point(rng, z_min=3.0, z_max=30.0)
                self.waypoints.append(wp)
            self.waypoint_idx = 0
            self.target = self.waypoints[0].copy()
            self._max_steps = 4000

    def step(self, action: np.ndarray):
        state = self.physics.step(action)
        self._step += 1
        self.position_history.append(state.position.copy())

        reward, terminated = self._compute_reward(state)
        truncated = self._step >= self._max_steps or state.battery <= 0

        if terminated or truncated:
            self.physics.state.crashed = state.crashed

        obs = self._observe()
        info = {
            "position": state.position.copy(),
            "distance_to_target": float(np.linalg.norm(state.position - self.target)),
            "crashed": state.crashed,
            "deliveries": self.deliveries_done,
        }
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, state: DroneState) -> Tuple[float, bool]:
        pos = state.position
        dist = float(np.linalg.norm(pos - self.target))
        reward = 0.0
        done = False

        # Crash penalty — real custom FPV drones are expensive and slow to
        # rebuild, so crashing is NOT cheap. Kept high (not the highest
        # possible) so the policy learns to avoid it, while still allowing
        # exploration through hover-biased init and lowered log_std.
        if state.crashed:
            return -50.0, True

        # Out-of-bounds soft penalty
        if not self.world.in_bounds(pos):
            reward -= 5.0

        # Collision penalty
        _, obs_dist = self.world.nearest_obstacle(pos)
        if obs_dist < 0.5:
            reward -= 10.0 * (0.5 - obs_dist)

        # --- Reward-hacking guards --------------------------------------
        # Targets for every task sit at z ≥ 5 m (hover/delivery have
        # target_z = 5..20 m, waypoints z = 3..30 m). A drone that soft-
        # lands directly under the target is still within the 20 m
        # distance band and would otherwise farm shaping forever. Gate
        # distance-shaping on the drone actually FLYING: airborne and
        # roughly upright. Pickup/dropzone triggers fire regardless so
        # legitimate touchdowns still work.
        airborne = pos[2] > 0.3
        upright = (
            abs(state.orientation[0]) < np.pi / 3
            and abs(state.orientation[1]) < np.pi / 3
        )
        can_shape = airborne and upright

        if self.task == TaskType.HOVER:
            # Two-band shaping, gated on can_shape so a grounded drone
            # earns zero and must re-launch to score. Far-band
            # coefficient reduced (0.5 → 0.3) so even in the air the
            # far band can't accumulate into an exploitable pile.
            if can_shape:
                reward += max(0.0, 0.3 * (1.0 - dist / 20.0))
                if dist < 2.0:
                    reward += max(0.0, 2.0 - dist) * 0.5
                if dist < 0.3:
                    reward += 1.0

        elif self.task == TaskType.DELIVERY:
            if not self.carrying:
                if can_shape:
                    reward += max(0.0, 3.0 - dist) * 0.3
                if dist < 1.0:
                    self.carrying = True
                    self.target = self.dropzone.copy()
                    self.target[2] = 5.0
                    reward += 5.0
            else:
                if can_shape:
                    reward += max(0.0, 3.0 - dist) * 0.3
                if dist < 1.5:
                    reward += 50.0
                    done = True

        elif self.task in (TaskType.DELIVERY_ROUTE, TaskType.DEPLOYMENT):
            if can_shape:
                reward += max(0.0, 5.0 - dist) * 0.1
            if dist < 2.0:
                reward += 30.0
                self.deliveries_done += 1
                self.waypoint_idx += 1
                if self.waypoint_idx >= len(self.waypoints):
                    reward += 100.0
                    done = True
                else:
                    self.target = self.waypoints[self.waypoint_idx].copy()

        # Idle-on-ground penalty — discourages the "soft-land and wait
        # it out" exploit. The crash model intentionally allows soft
        # touchdowns so the drone can actually land, but a grounded
        # drone that isn't completing a delivery is idling on purpose.
        # Raised to -0.10 (with the -0.02 time penalty that's -0.12/step
        # = -180 over the 1500-step hover episode). At -0.04/step the
        # cumulative penalty was only on par with a single crash, so
        # population mutations that landed-and-sat scored about the
        # same as ones that occasionally crashed mid-air — local optima
        # the BC-warmed lineage couldn't escape with shallow per-drone
        # PPO budgets. Now landing is strictly worse than crashing.
        at_delivery_touchdown = (
            self.task == TaskType.DELIVERY and self.carrying and dist < 1.5
        )
        if not airborne and not at_delivery_touchdown:
            reward -= 0.10

        # Airborne baseline bonus — being upright and inside the
        # flight envelope earns a small per-step reward regardless of
        # horizontal distance from target. This makes "stay in the
        # air, even if you can't reach the target" strictly better
        # than "land," pushing the gradient toward 3D exploration
        # rather than collapse onto the floor. Capped at z ≤ 25 m so
        # "spiral up forever" stops paying — without the cap the
        # policy can find an attractor where it just gains altitude
        # indefinitely, since +0.05/step at any altitude is positive.
        if airborne and upright and 3.0 <= pos[2] <= 25.0:
            reward += 0.05

        # Yaw-rate penalty — without this the reward function gives
        # the policy no gradient signal to stop spinning. Drones early
        # in training drift into spiral attractors (constant yaw rate
        # combined with thrust = circular ascent) because spinning
        # neither helps nor hurts the score. Quadratic in rad/s so
        # mild yaw is essentially free and only fast spinning bites.
        yaw_rate = float(state.angular_velocity[2])
        reward -= 0.005 * yaw_rate * yaw_rate

        # Time penalty. Raised from -0.01 to -0.02 so pure survival is
        # net-negative unless the policy is actually earning shaping —
        # otherwise a drone that just hovers in-bounds forever grades
        # higher than one that occasionally nails the task.
        reward -= 0.02

        return reward, done

    def _observe(self) -> np.ndarray:
        s = self.physics.state
        _, obs_dist = self.world.nearest_obstacle(s.position)
        # VIO-relative position: the drone only knows how far it has moved
        # since takeoff, not its absolute world coordinates. NO GPS.
        pos_vio = s.position - self._start_position
        rel = self.target - s.position
        baro_alt = pos_vio[2]  # treated as baro here; noise injection later
        braking = self.physics.braking_distance(s.velocity)
        # Battery temp normalized around 25 °C reference, ±30 °C swing.
        btemp_norm = np.clip((s.battery_temp - 25.0) / 30.0, -1.0, 1.0)
        obs = np.array([
            pos_vio[0] / 50.0,
            pos_vio[1] / 50.0,
            pos_vio[2] / 50.0,
            s.velocity[0] / 15.0,
            s.velocity[1] / 15.0,
            s.velocity[2] / 15.0,
            s.acceleration[0] / 20.0,
            s.acceleration[1] / 20.0,
            s.acceleration[2] / 20.0,
            np.clip(s.orientation[0] / np.pi, -1, 1),
            np.clip(s.orientation[1] / np.pi, -1, 1),
            np.clip(s.orientation[2] / np.pi, -1, 1),
            s.angular_velocity[0] / 6.0,
            s.angular_velocity[1] / 6.0,
            s.angular_velocity[2] / 6.0,
            rel[0] / 50.0,
            rel[1] / 50.0,
            rel[2] / 50.0,
            np.linalg.norm(rel) / 100.0,
            s.battery,
            btemp_norm,
            baro_alt / 50.0,
            min(braking / 30.0, 1.0),
            min(obs_dist / 20.0, 1.0),
            float(self.carrying),
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def set_waypoints(self, waypoints: List[np.ndarray]):
        self.waypoints = list(waypoints)
        self.waypoint_idx = 0
        if waypoints:
            self.target = waypoints[0].copy()

    def render(self):
        if self.render_mode != "human":
            return
        try:
            if self._renderer is None:
                from drone_ai.viz import Renderer
                self._renderer = Renderer()
            self._renderer.draw(self.physics.state, self.target, self.position_history, self.world)
        except Exception:
            pass

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None
