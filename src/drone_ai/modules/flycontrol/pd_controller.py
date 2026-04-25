"""Analytic PD controller — imitation-signal source for BC warm-up.

The cascaded position→attitude PD loop from `drone.py._pd_controller`
lifted into a pure function so the FlyControl trainer can pre-train
the PPO actor against its actions before stochastic PPO takes over.

Why: a fresh PPO actor with `hover-biased init + log_std=-1.0` still
explores aggressively enough to tumble the drone within ~20 steps —
the -50 crash reward dominates every update and learning never gets
off the ground. PLAN.md §8 prescribes fixing this with BC warm-up
(not by softening the crash penalty). This module is that warm-up's
teacher.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from drone_ai.simulation.physics import DroneState, QuadrotorPhysics


def _angle_diff(a: float, b: float) -> float:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def pd_action(state: DroneState, target: np.ndarray,
              physics: QuadrotorPhysics) -> np.ndarray:
    """Compute a 4-motor throttle [0,1]^4 that steers the drone toward
    `target`. Logic mirrors `DroneAI._pd_controller` so the PPO actor
    learns the exact fallback controller's behavior."""
    pos_err = target - state.position
    vel_err = -state.velocity
    kp_pos = np.array([1.2, 1.2, 4.0])
    kd_pos = np.array([1.8, 1.8, 2.5])
    desired_acc = kp_pos * pos_err + kd_pos * vel_err
    desired_acc = np.clip(desired_acc, -10.0, 10.0)
    desired_acc[2] += physics.G

    thrust_mag = physics.MASS * float(np.linalg.norm(desired_acc))
    thrust_norm = float(np.clip(thrust_mag / physics.MAX_THRUST, 0.0, 1.0))

    horiz_dist = float(np.linalg.norm(pos_err[:2]))
    if horiz_dist > 2.0:
        desired_yaw = float(np.arctan2(pos_err[1], pos_err[0]))
    else:
        desired_yaw = float(state.orientation[2])

    yaw = float(state.orientation[2])
    ax_world, ay_world = float(desired_acc[0]), float(desired_acc[1])
    ax_body =  ax_world * np.cos(yaw) + ay_world * np.sin(yaw)
    ay_body = -ax_world * np.sin(yaw) + ay_world * np.cos(yaw)

    az = float(desired_acc[2])
    desired_pitch = float(np.arctan2( ax_body, max(az, 1e-3)))
    desired_roll  = float(np.arctan2(-ay_body, max(az, 1e-3)))
    desired_pitch = float(np.clip(desired_pitch, -0.6, 0.6))
    desired_roll  = float(np.clip(desired_roll,  -0.6, 0.6))

    kp_att = 6.0
    kd_att = 2.0
    kp_yaw = 2.5
    kd_yaw = 0.8
    roll_cmd  = kp_att * (desired_roll  - state.orientation[0]) - kd_att * state.angular_velocity[0]
    pitch_cmd = kp_att * (desired_pitch - state.orientation[1]) - kd_att * state.angular_velocity[1]
    yaw_err = _angle_diff(desired_yaw, yaw)
    yaw_cmd = kp_yaw * yaw_err - kd_yaw * state.angular_velocity[2]

    # Motor mixing (signs match physics torque equations — see
    # simulation/physics.py step() for the canonical version).
    k = 0.22
    m1 = thrust_norm + k * (-pitch_cmd + roll_cmd - yaw_cmd)
    m2 = thrust_norm + k * (-pitch_cmd - roll_cmd + yaw_cmd)
    m3 = thrust_norm + k * ( pitch_cmd - roll_cmd - yaw_cmd)
    m4 = thrust_norm + k * ( pitch_cmd + roll_cmd + yaw_cmd)
    return np.clip([m1, m2, m3, m4], 0.0, 1.0).astype(np.float32)


def collect_pd_rollouts(
    env,
    n_episodes: int = 10,
    max_steps: int = 1500,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the PD controller in `env` for `n_episodes` and return a
    supervised dataset of (obs, action, reward, done).

    The observations are the env's 25-D vector as seen by the PPO
    actor (normalized), and the actions are the PD controller's
    outputs in [0,1]^4 — identical action space to the policy. The
    reward + done streams come back so the critic can be warmed up
    against Monte Carlo returns at the same time the actor is BC-
    trained on the action targets.
    """
    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    rew_buf: List[float] = []
    done_buf: List[bool] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            action = pd_action(env.physics.state, env.target, env.physics)
            obs_buf.append(obs.copy())
            act_buf.append(action.copy())
            obs, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            rew_buf.append(float(reward))
            done_buf.append(done)
            if done:
                break
    if not obs_buf:
        # Degenerate case — return empty arrays with the correct shapes.
        return (
            np.zeros((0, env.observation_space.shape[0]), dtype=np.float32),
            np.zeros((0, env.action_space.shape[0]), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.bool_),
        )
    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(act_buf, dtype=np.float32),
        np.asarray(rew_buf, dtype=np.float32),
        np.asarray(done_buf, dtype=np.bool_),
    )
