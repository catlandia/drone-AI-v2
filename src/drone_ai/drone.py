"""DroneAI — integrates all 4 layers into one autonomous system.

Usage:
    drone = DroneAI()
    drone.add_delivery([100, 50, 0], priority=Priority.URGENT)
    drone.run(max_steps=10000)

Layer interactions:
  Manager → selects next delivery destination
  PathFinder → plans obstacle-free path to destination
  Perception → detects obstacles in real-time, updates world model
  FlyControl → executes motor commands to follow waypoints
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from drone_ai.simulation.physics import QuadrotorPhysics, DroneState
from drone_ai.simulation.world import World, Obstacle
from drone_ai.modules.flycontrol.environment import FlyControlEnv, TaskType
from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig
from drone_ai.modules.pathfinder.algorithms import PathPlanner
from drone_ai.modules.perception.detector import PerceptionAI
from drone_ai.modules.perception.tracker import ObjectTracker
from drone_ai.modules.manager.planner import MissionPlanner, Priority


class SystemState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    FLYING = "flying"
    DELIVERING = "delivering"
    RETURNING = "returning"
    LANDED = "landed"
    CRASHED = "crashed"


@dataclass
class GradeConfig:
    """Which grade model each module is using (for experiments)."""
    flycontrol: str = "P"
    pathfinder: str = "P"
    perception: str = "P"
    manager: str = "P"

    def summary(self) -> str:
        return (f"FlyControl={self.flycontrol}  Pathfinder={self.pathfinder}  "
                f"Perception={self.perception}  Manager={self.manager}")


@dataclass
class FlightStatus:
    state: SystemState = SystemState.IDLE
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    battery: float = 1.0
    waypoint_idx: int = 0
    total_waypoints: int = 0
    obstacles_detected: int = 0
    deliveries_done: int = 0
    deliveries_pending: int = 0
    crashed: bool = False
    step: int = 0


class DroneAI:
    """Fully autonomous drone with 4-layer AI architecture."""

    BASE = np.array([0.0, 0.0, 0.0])

    def __init__(
        self,
        grades: Optional[GradeConfig] = None,
        flycontrol_model: Optional[str] = None,
        perception_model: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.grades = grades or GradeConfig()
        self._rng = np.random.default_rng(seed)

        self.world = World()
        self.physics = QuadrotorPhysics()

        self.manager = MissionPlanner(
            base_position=self.BASE.copy(),
            grade=self.grades.manager,
            seed=int(self._rng.integers(0, 2**31)),
        )
        self.pathfinder = PathPlanner(self.world)
        self.perception = PerceptionAI(
            grade=self.grades.perception,
            seed=int(self._rng.integers(0, 2**31)),
        )
        self.tracker = ObjectTracker()

        # If no trained model: use PD controller (untrained PPO produces garbage)
        self._agent: Optional[PPOAgent] = None
        if flycontrol_model:
            self._agent = PPOAgent.from_file(flycontrol_model)

        # State
        self.system_state = SystemState.IDLE
        self._path: List[np.ndarray] = []
        self._path_idx: int = 0
        self._step = 0
        self._position_history: List[np.ndarray] = []

    def reset(self, position: Optional[np.ndarray] = None):
        start = position if position is not None else self.BASE.copy() + np.array([0, 0, 2])
        self.physics.reset(position=start)
        self.manager.reset()
        self.tracker.clear()
        self.world.clear()
        self.system_state = SystemState.IDLE
        self._path = []
        self._path_idx = 0
        self._step = 0
        self._position_history = [start.copy()]

    def add_delivery(
        self, target, priority: Priority = Priority.NORMAL, weight: float = 1.0
    ):
        t = np.array(target, dtype=float)
        return self.manager.add_delivery(t, priority, weight)

    def set_obstacles(self, obstacles: List[Obstacle]):
        self.world.set_obstacles(obstacles)
        self.pathfinder.update_world(self.world)

    def step(self) -> Tuple[DroneState, bool]:
        """Run one simulation step. Returns (state, done)."""
        self._step += 1
        state = self.physics.state

        if state.crashed:
            self.system_state = SystemState.CRASHED
            return state, True

        # 1. Perception — detect obstacles, update world model
        detections = self.perception.detect(state.position, self.world)
        tracks = self.tracker.update(detections)
        detected_obstacles = self.perception.detections_to_obstacles(detections)
        perception_world = World()
        perception_world.set_obstacles(detected_obstacles)
        self.pathfinder.update_world(perception_world)

        # 2. Manager — decide what to do
        if self.system_state == SystemState.IDLE:
            chosen = self.manager.select_next(state.position)
            if chosen is not None:
                self.system_state = SystemState.PLANNING
                self._current_delivery = chosen
                self._target_position = chosen.target.copy()
                self._target_position[2] = max(chosen.target[2], 8.0)
            elif self.manager.is_complete():
                if np.linalg.norm(state.position - self.BASE) > 3.0:
                    self.system_state = SystemState.RETURNING
                    self._plan_to(self.BASE + np.array([0, 0, 8]))
                else:
                    self.system_state = SystemState.LANDED
                    return state, True

        # 3. PathFinder — compute path
        if self.system_state == SystemState.PLANNING:
            self._plan_to(self._target_position)
            self.system_state = SystemState.FLYING

        # 4. FlyControl — execute movement
        if self.system_state in (SystemState.FLYING, SystemState.RETURNING):
            done_waypoint = self._fly_step(state)
            if done_waypoint:
                if self.system_state == SystemState.FLYING:
                    # Reached destination
                    dist_to_delivery = np.linalg.norm(state.position[:2] - self._target_position[:2])
                    success = dist_to_delivery < 5.0
                    self.manager.complete_current(success)
                    self.system_state = SystemState.IDLE
                else:
                    self.system_state = SystemState.LANDED
                    return state, True

        self.manager.update(self.physics.DT, state.position, state.battery)
        self._position_history.append(state.position.copy())
        return state, False

    def _plan_to(self, target: np.ndarray):
        start = self.physics.state.position.copy()
        start[2] = max(start[2], 5.0)
        self._path = self.pathfinder.plan(start, target)
        self._path_idx = 0

    def _fly_step(self, state: DroneState) -> bool:
        if not self._path or self._path_idx >= len(self._path):
            return True

        target_wp = self._path[self._path_idx]
        dist = float(np.linalg.norm(state.position - target_wp))

        if dist < 2.5:
            self._path_idx += 1
            if self._path_idx >= len(self._path):
                return True

        # FlyControl: use RL agent or PD controller based on flycontrol grade
        action = self._compute_action(state, target_wp)
        self.physics.step(action)
        return False

    def _compute_action(self, state: DroneState, target: np.ndarray) -> np.ndarray:
        # Build observation for the agent
        from drone_ai.modules.flycontrol.environment import OBS_DIM
        _, obs_dist = self.world.nearest_obstacle(state.position)
        rel = target - state.position
        obs = np.array([
            state.position[0] / 50.0,
            state.position[1] / 50.0,
            state.position[2] / 50.0,
            state.velocity[0] / 15.0,
            state.velocity[1] / 15.0,
            state.velocity[2] / 15.0,
            np.clip(state.orientation[0] / np.pi, -1, 1),
            np.clip(state.orientation[1] / np.pi, -1, 1),
            np.clip(state.orientation[2] / np.pi, -1, 1),
            state.angular_velocity[0] / 6.0,
            state.angular_velocity[1] / 6.0,
            state.angular_velocity[2] / 6.0,
            rel[0] / 50.0,
            rel[1] / 50.0,
            rel[2] / 50.0,
            np.linalg.norm(rel) / 100.0,
            state.battery,
            min(obs_dist / 20.0, 1.0),
            0.0,
        ], dtype=np.float32)

        if self._agent is not None:
            action, _ = self._agent.select_action(obs, deterministic=True)
            return action

        return self._pd_controller(state, target)

    def _pd_controller(self, state: DroneState, target: np.ndarray) -> np.ndarray:
        """Cascaded position→attitude PD controller (fallback for untrained flight)."""
        # Outer loop: position → desired acceleration
        pos_err = target - state.position
        vel_err = -state.velocity
        kp_pos = np.array([0.8, 0.8, 3.0])
        kd_pos = np.array([1.2, 1.2, 2.0])
        desired_acc = kp_pos * pos_err + kd_pos * vel_err
        desired_acc = np.clip(desired_acc, -6.0, 6.0)
        desired_acc[2] += self.physics.G  # compensate gravity

        # Thrust magnitude (project desired acc onto body-z)
        thrust_mag = self.physics.MASS * float(np.linalg.norm(desired_acc))
        thrust_norm = np.clip(thrust_mag / self.physics.MAX_THRUST, 0.0, 1.0)

        # Desired roll/pitch from horizontal acceleration
        yaw = state.orientation[2]
        ax_body = desired_acc[0] * np.cos(yaw) + desired_acc[1] * np.sin(yaw)
        ay_body = -desired_acc[0] * np.sin(yaw) + desired_acc[1] * np.cos(yaw)
        desired_pitch = np.arctan2(ax_body, desired_acc[2] + 1e-6)
        desired_roll = -np.arctan2(ay_body, desired_acc[2] + 1e-6)
        desired_pitch = np.clip(desired_pitch, -0.4, 0.4)
        desired_roll = np.clip(desired_roll, -0.4, 0.4)

        # Inner loop: attitude → angular acceleration
        kp_att = 4.0
        kd_att = 1.5
        roll_cmd = kp_att * (desired_roll - state.orientation[0]) - kd_att * state.angular_velocity[0]
        pitch_cmd = kp_att * (desired_pitch - state.orientation[1]) - kd_att * state.angular_velocity[1]
        yaw_cmd = -kd_att * state.angular_velocity[2]

        # Motor mixing — signs must match physics torque equations:
        # roll_torque  = ( m1 - m2 - m3 + m4) * MAX_TORQUE
        # pitch_torque = (-m1 - m2 + m3 + m4) * MAX_TORQUE
        # yaw_torque   = (-m1 + m2 - m3 + m4) * YAW_TORQUE
        k = 0.15
        m1 = thrust_norm + k * (-pitch_cmd + roll_cmd - yaw_cmd)
        m2 = thrust_norm + k * (-pitch_cmd - roll_cmd + yaw_cmd)
        m3 = thrust_norm + k * ( pitch_cmd - roll_cmd - yaw_cmd)
        m4 = thrust_norm + k * ( pitch_cmd + roll_cmd + yaw_cmd)
        return np.clip([m1, m2, m3, m4], 0.0, 1.0).astype(np.float32)

    def run(self, max_steps: int = 20000, verbose: bool = False) -> Dict[str, Any]:
        """Run a full mission. Returns episode stats.

        Assumes reset() + deliveries + obstacles were set beforehand.
        """
        crashes = 0
        for s in range(max_steps):
            state, done = self.step()
            if verbose and s % 200 == 0:
                st = self.get_status()
                print(f"  Step {s}: {st.state.value}  pos={state.position.round(1)}  "
                      f"deliveries={st.deliveries_done}/{st.deliveries_done+st.deliveries_pending}")
            if done:
                if state.crashed:
                    crashes += 1
                break

        summary = self.manager.get_summary()
        summary["steps"] = self._step
        summary["crashed"] = self.physics.state.crashed
        summary["system_state"] = self.system_state.value
        summary["grades"] = self.grades.summary()
        return summary

    def get_status(self) -> FlightStatus:
        s = self.physics.state
        target = self._path[self._path_idx] if self._path and self._path_idx < len(self._path) else np.zeros(3)
        mgr_summary = self.manager.get_summary()
        return FlightStatus(
            state=self.system_state,
            position=s.position.copy(),
            target=target,
            battery=s.battery,
            waypoint_idx=self._path_idx,
            total_waypoints=len(self._path),
            deliveries_done=mgr_summary["completed"],
            deliveries_pending=mgr_summary["pending"],
            crashed=s.crashed,
            step=self._step,
        )
