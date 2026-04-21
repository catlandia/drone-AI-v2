"""Quadrotor physics simulation — FPV-racing-style 5" quad dynamics.

Modeled after a typical 5-inch freestyle/racing FPV drone (e.g. Iflight
Nazgul, TBS Source One). Includes the key dynamics that matter for
sim-to-real transfer of RL policies:

  1. First-order motor response lag (TAU_MOTOR ≈ 35 ms).
  2. Quadratic aerodynamic drag on linear velocity.
  3. Linear angular drag on body rates.
  4. Battery voltage sag: max thrust drops as battery depletes.

X-configuration motor layout (motor index → location / spin direction):
    Motor 1: front-left,  CCW   (+roll, -pitch, -yaw)
    Motor 2: front-right, CW    (-roll, -pitch, +yaw)
    Motor 3: rear-right,  CCW   (-roll, +pitch, -yaw)
    Motor 4: rear-left,   CW    (+roll, +pitch, +yaw)

Action: [m1, m2, m3, m4] normalized throttle in [0, 1].
State: position(3), velocity(3), orientation(3), angular_velocity(3) = 12D,
       plus motor_state(4) and battery — these are integrated internally.
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
    motor_state: np.ndarray = field(default_factory=lambda: np.zeros(4))  # filtered throttle

    def copy(self) -> "DroneState":
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            battery=self.battery,
            crashed=self.crashed,
            t=self.t,
            motor_state=self.motor_state.copy(),
        )

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.position, self.velocity, self.orientation, self.angular_velocity
        ]).astype(np.float32)


class QuadrotorPhysics:
    """FPV-style quadrotor physics, Euler-integrated at 50 Hz."""

    # ---- Frame & propulsion ------------------------------------------
    MASS = 0.60          # kg (5" FPV with 1300mAh 4S battery)
    G = 9.81             # m/s²
    ARM = 0.12           # m (motor arm from center — 5" spacing)
    # Inertia (cuboid approximation, realistic for 5" freestyle)
    IXX = 0.0048         # kg·m² (roll)
    IYY = 0.0048         # kg·m² (pitch)
    IZZ = 0.0090         # kg·m² (yaw, higher — most of the mass is in the plane)
    # Thrust — 4 motors, ~2800 KV on 4S, EMAX 2207 + HQ 5"
    MAX_THRUST_PER_MOTOR = 7.0   # N at full battery (≈ 700g per motor)
    MAX_TORQUE = 0.18            # N·m (roll/pitch differential)
    YAW_TORQUE = 0.06            # N·m (yaw differential — smaller than roll)
    # Motor dynamics: first-order response lag
    TAU_MOTOR = 0.035            # s (motor + ESC spin-up time constant)
    # Aerodynamic drag
    DRAG_LIN = 0.25              # quadratic drag coefficient (N / (m/s)²)
    DRAG_ANG = 0.008             # linear angular drag (N·m / (rad/s))
    # Battery: max thrust scales with sqrt(voltage), approximated linearly
    BAT_THRUST_MIN = 0.80        # at 0% battery, thrust drops to 80% of max
    BATTERY_DRAIN = 5e-5         # per step at full throttle
    # Integration / safety
    DT = 0.02                    # s (50 Hz)
    MAX_VEL = 28.0               # m/s clamp (real FPV can do 30+)
    MAX_ANG_VEL = 14.0           # rad/s clamp (real FPV can do 17+)
    # Crash conditions. Custom FPV drones are expensive and slow to rebuild,
    # so crashing is a real event, not a soft reset.
    HARD_IMPACT_VZ = 8.0         # m/s into ground = crash
    INVERSION_TILT = np.pi / 2   # ±90°: propellers pointing down → dead

    def __init__(self, dt: float = DT):
        self.dt = dt
        self.state = DroneState()
        self._i_inv = np.array([1 / self.IXX, 1 / self.IYY, 1 / self.IZZ])

    @property
    def MAX_THRUST(self) -> float:
        """Total max thrust (backwards-compat alias for other modules)."""
        return 4.0 * self.MAX_THRUST_PER_MOTOR

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

        cmd = np.clip(action, 0.0, 1.0)

        # 1. Motor lag: first-order filter toward commanded throttle.
        alpha = 1.0 - np.exp(-self.dt / max(self.TAU_MOTOR, 1e-4))
        self.state.motor_state += (cmd - self.state.motor_state) * alpha
        m = self.state.motor_state

        # 2. Battery sag: thrust envelope drops as battery depletes.
        bat_factor = self.BAT_THRUST_MIN + (1.0 - self.BAT_THRUST_MIN) * self.state.battery
        thrust_per_motor = m * self.MAX_THRUST_PER_MOTOR * bat_factor  # (4,)

        # 3. Total body-frame thrust along +z_body.
        thrust = float(np.sum(thrust_per_motor))

        # 4. Torques from motor differences (per-motor thrust × lever arm fractions).
        # We normalize by MAX_THRUST_PER_MOTOR so MAX_TORQUE keeps the same scale
        # regardless of battery state — the real drone has the same leverage.
        mn = thrust_per_motor / self.MAX_THRUST_PER_MOTOR
        roll_torque  = (mn[0] - mn[1] - mn[2] + mn[3]) * self.MAX_TORQUE
        pitch_torque = (-mn[0] - mn[1] + mn[2] + mn[3]) * self.MAX_TORQUE
        yaw_torque   = (-mn[0] + mn[1] - mn[2] + mn[3]) * self.YAW_TORQUE

        # 5. Rotation to world.
        r, p, y = self.state.orientation
        R = self._rotation_matrix(r, p, y)

        # 6. Linear dynamics: thrust + gravity + aerodynamic drag.
        thrust_world = R @ np.array([0.0, 0.0, thrust])
        gravity = np.array([0.0, 0.0, -self.G * self.MASS])
        v = self.state.velocity
        drag_force = -self.DRAG_LIN * np.linalg.norm(v) * v
        accel = (thrust_world + gravity + drag_force) / self.MASS

        # 7. Angular dynamics: body torques − gyroscopic − angular drag.
        I = np.array([self.IXX, self.IYY, self.IZZ])
        w = self.state.angular_velocity
        gyro = np.cross(w, I * w)
        drag_torque = -self.DRAG_ANG * w
        torque = np.array([roll_torque, pitch_torque, yaw_torque])
        alpha_w = self._i_inv * (torque + drag_torque - gyro)

        # 8. Euler integration.
        self.state.velocity += accel * self.dt
        self.state.position += self.state.velocity * self.dt
        self.state.angular_velocity += alpha_w * self.dt
        self.state.orientation += self.state.angular_velocity * self.dt

        # 9. Clamp for stability.
        self.state.velocity = np.clip(self.state.velocity, -self.MAX_VEL, self.MAX_VEL)
        self.state.angular_velocity = np.clip(self.state.angular_velocity,
                                              -self.MAX_ANG_VEL, self.MAX_ANG_VEL)

        # 10. Wrap yaw to (-π, π] — keeps representation bounded.
        y = self.state.orientation[2]
        self.state.orientation[2] = (y + np.pi) % (2 * np.pi) - np.pi

        # 11. Ground collision — only hard impacts crash. A real FPV drone
        #     can touch down firmly without instantly breaking, and most
        #     importantly an RL policy needs lots of near-ground exploration
        #     during the learning phase without getting punished terminally.
        if self.state.position[2] < 0.0:
            self.state.position[2] = 0.0
            if np.abs(self.state.velocity[2]) > self.HARD_IMPACT_VZ:
                self.state.crashed = True
            else:
                # Bounce/settle: kill vertical velocity, damp lateral.
                self.state.velocity[2] = 0.0
                self.state.velocity[0] *= 0.6
                self.state.velocity[1] *= 0.6

        # Wrap roll/pitch to (-π, π] BEFORE the inversion check — otherwise
        # integrating past ±π wraps around silently and the crash never fires.
        for i in (0, 1):
            a = self.state.orientation[i]
            self.state.orientation[i] = (a + np.pi) % (2 * np.pi) - np.pi

        # Inversion crash: props pointing down (|roll| ≥ 90° or |pitch| ≥ 90°)
        # means the thrust vector cannot push the drone up anymore. A real
        # bespoke FPV quad at that attitude is going down; treat it as dead.
        if (np.abs(self.state.orientation[0]) >= self.INVERSION_TILT or
                np.abs(self.state.orientation[1]) >= self.INVERSION_TILT):
            self.state.crashed = True

        # 13. Battery drain proportional to mean throttle.
        self.state.battery = max(0.0, self.state.battery - self.BATTERY_DRAIN * float(np.mean(m)))
        self.state.t += self.dt

        return self.state

    @staticmethod
    def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ZYX Euler rotation matrix (body → world)."""
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr            ],
        ])
