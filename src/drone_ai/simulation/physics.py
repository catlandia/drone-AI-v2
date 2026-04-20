"""Quadrotor physics simulation.

X-configuration motor layout:
    Motor 1 (front-left,  CCW): +roll, -pitch, -yaw
    Motor 2 (front-right, CW):  -roll, -pitch, +yaw
    Motor 3 (rear-right,  CCW): -roll, +pitch, -yaw
    Motor 4 (rear-left,   CW):  +roll, +pitch, +yaw

Action: [m1, m2, m3, m4] in [0, 1]
State: position(3), velocity(3), orientation(3), angular_velocity(3) = 12D
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DroneState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll, pitch, yaw
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # p, q, r
    battery: float = 1.0
    crashed: bool = False
    t: float = 0.0

    def copy(self) -> "DroneState":
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            battery=self.battery,
            crashed=self.crashed,
            t=self.t,
        )

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.position, self.velocity, self.orientation, self.angular_velocity
        ]).astype(np.float32)


class QuadrotorPhysics:
    """Realistic quadrotor physics via Euler integration."""

    # Physical constants
    MASS = 0.5          # kg
    G = 9.81            # m/s²
    ARM = 0.23          # m (motor arm length)
    IXX = 0.0196        # kg·m² (roll inertia)
    IYY = 0.0196        # kg·m² (pitch inertia)
    IZZ = 0.0264        # kg·m² (yaw inertia)
    MAX_THRUST = 14.0   # N (total, hover ≈ mass*g = 4.9N → ~35% throttle)
    MAX_TORQUE = 0.5    # N·m
    YAW_TORQUE = 0.1    # N·m
    DT = 0.02           # s (50 Hz)
    MAX_VEL = 15.0      # m/s clamp
    MAX_ANG_VEL = 6.0   # rad/s clamp
    MAX_TILT = np.pi / 3  # 60° max tilt before crash
    BATTERY_DRAIN = 5e-5  # per step at full throttle

    def __init__(self, dt: float = DT):
        self.dt = dt
        self.state = DroneState()
        self._i_inv = np.array([1/self.IXX, 1/self.IYY, 1/self.IZZ])

    def reset(self, position: Optional[np.ndarray] = None, yaw: float = 0.0):
        self.state = DroneState(
            position=position.copy() if position is not None else np.zeros(3),
            orientation=np.array([0.0, 0.0, yaw]),
        )
        return self.state

    def step(self, action: np.ndarray) -> DroneState:
        """Step simulation with motor commands [0,1]^4."""
        if self.state.crashed:
            return self.state

        m = np.clip(action, 0.0, 1.0)

        # Compute forces and torques in body frame
        thrust = np.sum(m) * self.MAX_THRUST / 4.0
        roll_torque  = (m[0] - m[1] - m[2] + m[3]) * self.MAX_TORQUE
        pitch_torque = (-m[0] - m[1] + m[2] + m[3]) * self.MAX_TORQUE
        yaw_torque   = (-m[0] + m[1] - m[2] + m[3]) * self.YAW_TORQUE

        r, p, y = self.state.orientation
        R = self._rotation_matrix(r, p, y)

        # Linear dynamics
        thrust_world = R @ np.array([0.0, 0.0, thrust])
        gravity = np.array([0.0, 0.0, -self.G * self.MASS])
        accel = (thrust_world + gravity) / self.MASS

        # Angular dynamics
        I = np.array([self.IXX, self.IYY, self.IZZ])
        w = self.state.angular_velocity
        gyro = np.cross(w, I * w)
        torque = np.array([roll_torque, pitch_torque, yaw_torque])
        alpha = self._i_inv * (torque - gyro)

        # Euler integration
        self.state.velocity += accel * self.dt
        self.state.position += self.state.velocity * self.dt
        self.state.angular_velocity += alpha * self.dt
        self.state.orientation += self.state.angular_velocity * self.dt

        # Clamp for stability
        self.state.velocity = np.clip(self.state.velocity, -self.MAX_VEL, self.MAX_VEL)
        self.state.angular_velocity = np.clip(self.state.angular_velocity, -self.MAX_ANG_VEL, self.MAX_ANG_VEL)

        # Ground collision
        if self.state.position[2] < 0.0:
            self.state.position[2] = 0.0
            if np.abs(self.state.velocity[2]) > 3.0:
                self.state.crashed = True
            else:
                self.state.velocity[2] = 0.0

        # Tilt crash
        if (np.abs(self.state.orientation[0]) > self.MAX_TILT or
                np.abs(self.state.orientation[1]) > self.MAX_TILT):
            self.state.crashed = True

        # Battery
        self.state.battery = max(0.0, self.state.battery - self.BATTERY_DRAIN * np.mean(m))
        self.state.t += self.dt

        return self.state

    @staticmethod
    def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ZYX Euler angle rotation matrix (body → world)."""
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr            ],
        ])
