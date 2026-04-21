"""Quadrotor physics simulation — FPV-racing-style 5" quad dynamics.

Modeled after a typical 5-inch freestyle/racing FPV drone (e.g. Iflight
Nazgul, TBS Source One). Phase 1 realism upgrades:

  1. First-order motor response lag (TAU_MOTOR ≈ 35 ms).
  2. Full moment-of-inertia tensor (still diagonal for an X-quad, but
     exposed as `I_TENSOR` so later off-diagonal terms fit).
  3. Gyroscopic coupling from SPINNING PROPELLERS (not just airframe).
  4. Anisotropic aerodynamic drag (forward ≠ sideways ≠ vertical).
  5. Reactive yaw torque when the COLLECTIVE thrust changes
     (rotor-angular-momentum kick, not just the differential).
  6. Wind + von-Kármán-ish turbulence.
  7. Ground effect (extra lift when hovering near the ground).
  8. Prop damage / motor wear — per-motor efficiency factor that
     drifts down with usage and can fail outright.
  9. Battery voltage sag depending on state-of-charge AND temperature.

X-configuration motor layout (motor index → location / spin direction):
    Motor 1: front-left,  CCW   (+roll, -pitch, -yaw)
    Motor 2: front-right, CW    (-roll, -pitch, +yaw)
    Motor 3: rear-right,  CCW   (-roll, +pitch, -yaw)
    Motor 4: rear-left,   CW    (+roll, +pitch, +yaw)

Action: [m1, m2, m3, m4] normalized throttle in [0, 1].
State: position(3), velocity(3), orientation(3), angular_velocity(3) = 12D,
       plus per-step linear acceleration(3), motor_state(4),
       motor_health(4), battery scalar, battery_temp scalar,
       wind(3).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DroneState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world-frame linear
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll, pitch, yaw
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # p, q, r
    battery: float = 1.0
    battery_temp: float = 25.0           # °C
    crashed: bool = False
    t: float = 0.0
    motor_state: np.ndarray = field(default_factory=lambda: np.zeros(4))    # filtered throttle
    motor_health: np.ndarray = field(default_factory=lambda: np.ones(4))    # 1.0 = new, 0.0 = dead
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3))           # world-frame wind vel

    def copy(self) -> "DroneState":
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            battery=self.battery,
            battery_temp=self.battery_temp,
            crashed=self.crashed,
            t=self.t,
            motor_state=self.motor_state.copy(),
            motor_health=self.motor_health.copy(),
            wind=self.wind.copy(),
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

    # Moment-of-inertia tensor. Diagonal on an X-quad but treated as a
    # full 3x3 so future off-diagonal products of inertia fit without
    # breaking the integrator.
    IXX = 0.0048         # kg·m² (roll)
    IYY = 0.0048         # kg·m² (pitch)
    IZZ = 0.0090         # kg·m² (yaw — most mass is in the plane)

    # Thrust — 4 motors, ~2800 KV on 4S, EMAX 2207 + HQ 5"
    MAX_THRUST_PER_MOTOR = 7.0   # N at full battery
    MAX_TORQUE = 0.18            # N·m (roll/pitch differential)
    YAW_TORQUE = 0.06            # N·m (yaw differential — smaller than roll)

    # Rotor inertia (one prop + bell). Matters for (a) gyroscopic
    # coupling from spinning props and (b) reactive yaw torque when
    # collective thrust ramps.
    ROTOR_INERTIA = 6e-6         # kg·m² per rotor
    ROTOR_MAX_OMEGA = 3500.0     # rad/s at full throttle on a fresh 4S

    # Motor dynamics
    TAU_MOTOR = 0.035            # s (motor + ESC spin-up time constant)

    # Anisotropic drag. 5" FPV: forward face is small, planform (top-
    # down) is large, sides are in between.  Units: N / (m/s)².
    DRAG_COEFF_BODY = np.array([0.20, 0.20, 0.40])
    DRAG_ANG = 0.008             # linear angular drag (N·m / (rad/s))

    # Battery
    BAT_THRUST_MIN = 0.80        # at 0% battery, thrust drops to 80% of max
    BATTERY_DRAIN = 5e-5         # per step at full throttle
    # Sag increases when the battery is cold. Factor applied as
    #   bat_factor *= 1 - TEMP_SAG * max(0, ref - temp) / ref
    BAT_TEMP_REF = 25.0          # °C (nominal)
    BAT_TEMP_SAG = 0.30          # up to 30% worse at 0 °C
    # Battery temp dynamics: warms with draw, cools to ambient.
    BAT_HEAT_RATE = 0.05         # °C per step per unit mean throttle
    BAT_COOL_RATE = 0.002        # °C per step toward ambient

    # Wind + turbulence (m/s). Steady field + gusty component.
    # These defaults mean "indoor calm." Set via set_wind() at reset.
    WIND_MEAN = np.array([0.0, 0.0, 0.0])
    WIND_GUST_STD = 0.0          # gust amplitude
    WIND_GUST_TAU = 1.5          # s, gust correlation time

    # Ground effect: T_eff = T * (1 + GE_GAIN * max(0, 1 - z/GE_HEIGHT)²)
    GE_GAIN = 0.15               # +15% thrust at z≈0
    GE_HEIGHT = 0.5              # m — effect vanishes above ~0.5 m

    # Prop wear: slow drift of motor_health down; small chance of sudden
    # failure. Disabled by default — enable via set_wear(True) when
    # training long-horizon reliability.
    WEAR_RATE = 0.0              # per step, per unit throttle (tune)
    FAILURE_PROB = 0.0           # per step, per motor (tune)

    # Integration / safety
    DT = 0.02                    # s (50 Hz)
    MAX_VEL = 28.0               # m/s clamp
    MAX_ANG_VEL = 14.0           # rad/s clamp

    # Crash conditions
    HARD_IMPACT_VZ = 8.0         # m/s into ground = crash
    INVERSION_TILT = np.pi / 2   # ±90°: props down → dead

    def __init__(self, dt: float = DT):
        self.dt = dt
        self.state = DroneState()
        self._I = np.diag([self.IXX, self.IYY, self.IZZ])
        self._I_inv = np.linalg.inv(self._I)
        # Cached previous thrust for reactive yaw torque.
        self._prev_thrust_per_motor = np.zeros(4)
        # Persistent gust noise for OU-like turbulence.
        self._gust = np.zeros(3)
        self._rng = np.random.default_rng()

    @property
    def MAX_THRUST(self) -> float:
        """Total max thrust (backwards-compat alias for other modules)."""
        return 4.0 * self.MAX_THRUST_PER_MOTOR

    @property
    def I_tensor(self) -> np.ndarray:
        return self._I

    # ------------------------------------------------------------------

    def reset(self, position: Optional[np.ndarray] = None, yaw: float = 0.0,
              seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = DroneState(
            position=position.copy() if position is not None else np.zeros(3),
            orientation=np.array([0.0, 0.0, yaw]),
            battery_temp=float(self.BAT_TEMP_REF),
        )
        self._prev_thrust_per_motor = np.zeros(4)
        self._gust = np.zeros(3)
        return self.state

    def set_wind(self, mean: np.ndarray, gust_std: float = 0.0,
                 gust_tau: float = 1.5):
        """Runtime setter for environmental wind."""
        self.WIND_MEAN = np.asarray(mean, dtype=np.float64)
        self.WIND_GUST_STD = float(gust_std)
        self.WIND_GUST_TAU = float(gust_tau)

    def set_wear(self, enable: bool, rate: float = 1e-7,
                 failure_prob: float = 1e-6):
        """Enable prop wear and sudden-failure models."""
        self.WEAR_RATE = float(rate) if enable else 0.0
        self.FAILURE_PROB = float(failure_prob) if enable else 0.0

    def braking_distance(self, v: Optional[np.ndarray] = None) -> float:
        """Straight-line braking distance at current (or given) velocity.

        Approximation: decelerate at ~0.8g since an FPV drone at full
        tilt can pull ~1g horizontally, and we leave margin so Pathfinder
        and Manager don't plan flush to the limit. Returns metres.
        """
        if v is None:
            v = self.state.velocity
        speed = float(np.linalg.norm(v))
        if speed < 1e-3:
            return 0.0
        a_brake = 0.8 * self.G
        return (speed * speed) / (2.0 * a_brake)

    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> DroneState:
        """Step simulation with motor commands [0,1]^4."""
        if self.state.crashed:
            return self.state

        cmd = np.clip(action, 0.0, 1.0)

        # 1. Motor lag: first-order filter toward commanded throttle.
        alpha = 1.0 - np.exp(-self.dt / max(self.TAU_MOTOR, 1e-4))
        self.state.motor_state += (cmd - self.state.motor_state) * alpha
        m = self.state.motor_state

        # 2. Motor health (wear + sudden failure). Health is a per-motor
        # efficiency multiplier; healthy = 1.0, dead = 0.0.
        if self.WEAR_RATE > 0.0:
            self.state.motor_health = np.clip(
                self.state.motor_health - self.WEAR_RATE * m, 0.0, 1.0
            )
        if self.FAILURE_PROB > 0.0:
            # Each motor has an independent small chance to seize.
            fail = self._rng.random(4) < self.FAILURE_PROB
            self.state.motor_health = np.where(fail, 0.0, self.state.motor_health)

        # 3. Battery sag — base + temperature term.
        soc = self.state.battery
        bat_factor = self.BAT_THRUST_MIN + (1.0 - self.BAT_THRUST_MIN) * soc
        cold = max(0.0, self.BAT_TEMP_REF - self.state.battery_temp) / self.BAT_TEMP_REF
        bat_factor *= max(0.2, 1.0 - self.BAT_TEMP_SAG * cold)

        thrust_per_motor = (
            m * self.state.motor_health * self.MAX_THRUST_PER_MOTOR * bat_factor
        )  # (4,)

        # 4. Total body-frame thrust along +z_body.
        thrust = float(np.sum(thrust_per_motor))

        # 5. Torques from motor differences.
        mn = thrust_per_motor / self.MAX_THRUST_PER_MOTOR
        roll_torque  = (mn[0] - mn[1] - mn[2] + mn[3]) * self.MAX_TORQUE
        pitch_torque = (-mn[0] - mn[1] + mn[2] + mn[3]) * self.MAX_TORQUE
        yaw_torque_diff = (-mn[0] + mn[1] - mn[2] + mn[3]) * self.YAW_TORQUE

        # 5b. Reactive yaw torque when the COLLECTIVE thrust ramps.
        # Spinning up rotors 1+3 (CCW) and 2+4 (CW) reacts against each
        # other, but the CCW–CW pairs don't match exactly during
        # collective thrust changes, which shows up on real drones as a
        # small "punchout yaw kick." Model it as ROTOR_INERTIA × d(omega)/dt
        # summed across the asymmetry.
        d_thrust = thrust_per_motor - self._prev_thrust_per_motor
        # CCW = motors 1, 3 (indices 0, 2); CW = motors 2, 4 (indices 1, 3).
        # Yaw reaction is opposite to the net rotor acceleration of the
        # spin direction that is speeding up.
        d_omega_ccw = np.sum(d_thrust[[0, 2]]) / self.MAX_THRUST_PER_MOTOR * self.ROTOR_MAX_OMEGA
        d_omega_cw  = np.sum(d_thrust[[1, 3]]) / self.MAX_THRUST_PER_MOTOR * self.ROTOR_MAX_OMEGA
        reactive_yaw = -self.ROTOR_INERTIA * (d_omega_ccw - d_omega_cw) / max(self.dt, 1e-4)
        yaw_torque = yaw_torque_diff + reactive_yaw
        self._prev_thrust_per_motor = thrust_per_motor.copy()

        # 6. Rotation to world.
        r, p, y = self.state.orientation
        R = self._rotation_matrix(r, p, y)

        # 7. Wind update (OU-like gust). Then airspeed = v - wind.
        if self.WIND_GUST_STD > 0.0:
            beta = self.dt / max(self.WIND_GUST_TAU, 1e-3)
            self._gust += -beta * self._gust + np.sqrt(2.0 * beta) * self.WIND_GUST_STD * self._rng.standard_normal(3)
        self.state.wind = self.WIND_MEAN + self._gust
        airspeed_world = self.state.velocity - self.state.wind

        # 8. Anisotropic drag — applied in BODY frame where the
        # coefficients make geometric sense.
        airspeed_body = R.T @ airspeed_world
        drag_body = -self.DRAG_COEFF_BODY * np.abs(airspeed_body) * airspeed_body
        drag_world = R @ drag_body

        # 9. Ground effect: thrust gets a bonus at low altitude.
        z = self.state.position[2]
        ge_frac = max(0.0, 1.0 - z / self.GE_HEIGHT)
        thrust_eff = thrust * (1.0 + self.GE_GAIN * ge_frac * ge_frac)

        # 10. Linear dynamics.
        thrust_world = R @ np.array([0.0, 0.0, thrust_eff])
        gravity = np.array([0.0, 0.0, -self.G * self.MASS])
        accel = (thrust_world + gravity + drag_world) / self.MASS
        self.state.acceleration = accel.copy()

        # 11. Angular dynamics with airframe-gyroscopic + prop-gyroscopic
        # coupling. Prop gyroscopic term: H_prop × ω, where H_prop is the
        # net rotor angular momentum along body +z (CCW minus CW rotors).
        w = self.state.angular_velocity
        I_w = self._I @ w
        gyro_body = np.cross(w, I_w)

        # Net prop angular momentum along body +z (CCW positive).
        omega_ccw = (mn[0] + mn[2]) * self.ROTOR_MAX_OMEGA
        omega_cw  = (mn[1] + mn[3]) * self.ROTOR_MAX_OMEGA
        H_prop = self.ROTOR_INERTIA * (omega_ccw - omega_cw)
        gyro_prop = np.cross(np.array([0.0, 0.0, H_prop]), w)  # shape (3,)

        drag_torque = -self.DRAG_ANG * w
        torque = np.array([roll_torque, pitch_torque, yaw_torque])
        alpha_w = self._I_inv @ (torque + drag_torque - gyro_body - gyro_prop)

        # 12. Euler integration.
        self.state.velocity += accel * self.dt
        self.state.position += self.state.velocity * self.dt
        self.state.angular_velocity += alpha_w * self.dt
        self.state.orientation += self.state.angular_velocity * self.dt

        # 13. Clamp for stability.
        self.state.velocity = np.clip(self.state.velocity, -self.MAX_VEL, self.MAX_VEL)
        self.state.angular_velocity = np.clip(self.state.angular_velocity,
                                              -self.MAX_ANG_VEL, self.MAX_ANG_VEL)

        # 14. Wrap yaw to (-π, π].
        y_ang = self.state.orientation[2]
        self.state.orientation[2] = (y_ang + np.pi) % (2 * np.pi) - np.pi

        # 15. Ground collision — only hard impacts crash.
        if self.state.position[2] < 0.0:
            self.state.position[2] = 0.0
            if np.abs(self.state.velocity[2]) > self.HARD_IMPACT_VZ:
                self.state.crashed = True
            else:
                # Bounce/settle: kill vertical velocity, damp lateral.
                self.state.velocity[2] = 0.0
                self.state.velocity[0] *= 0.6
                self.state.velocity[1] *= 0.6

        # 16. Wrap roll/pitch BEFORE the inversion check.
        for i in (0, 1):
            a = self.state.orientation[i]
            self.state.orientation[i] = (a + np.pi) % (2 * np.pi) - np.pi

        # 17. Inversion crash.
        if (np.abs(self.state.orientation[0]) >= self.INVERSION_TILT or
                np.abs(self.state.orientation[1]) >= self.INVERSION_TILT):
            self.state.crashed = True

        # 18. Battery drain + temperature.
        mean_throttle = float(np.mean(m))
        self.state.battery = max(0.0, self.state.battery - self.BATTERY_DRAIN * mean_throttle)
        self.state.battery_temp += (
            self.BAT_HEAT_RATE * mean_throttle
            - self.BAT_COOL_RATE * (self.state.battery_temp - self.BAT_TEMP_REF)
        )
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
