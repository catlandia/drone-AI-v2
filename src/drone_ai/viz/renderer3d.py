"""3D software renderer for the drone simulator (pygame-only, no OpenGL).

Uses manual perspective projection and painter's-algorithm depth sorting.
Designed for drones and obstacles — not a general engine.

Public API:
    r = Renderer(width=1280, height=800, title="...")
    r.handle_events(dt)                     # returns False when user quits
    r.draw_scene(state, target, path, world, hud_lines=[...])
    r.flip()
    r.close()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from drone_ai.simulation.physics import DroneState
from drone_ai.simulation.world import Obstacle, World


# ---- Colors ---------------------------------------------------------------

BG           = (18, 22, 30)
GROUND_A     = (42, 48, 60)
GROUND_B     = (36, 42, 52)
GRID         = (55, 64, 80)
GRID_AXIS_X  = (200, 80,  80)
GRID_AXIS_Y  = (80,  200, 80)
SKY_TOP      = (30, 40, 55)
SKY_BOT      = (14, 18, 26)

DRONE_BODY   = (70, 180, 90)
DRONE_BODY_D = (35, 110, 55)
DRONE_ARM    = (50, 55, 65)
DRONE_CANOPY = (140, 230, 170)
MOTOR        = (30, 32, 38)
PROP         = (210, 220, 230)
PROP_TRAIL   = (120, 150, 180)

OBS_TOP      = (110, 120, 140)
OBS_SIDE     = (85, 92, 108)
OBS_EDGE     = (140, 155, 180)

TARGET       = (255, 200, 80)
TARGET_DIM   = (120, 95, 40)
PATH         = (140, 175, 220)
TRAIL        = (80, 130, 180)
WAYPOINT     = (255, 230, 100)

TEXT         = (225, 230, 240)
TEXT_DIM     = (150, 160, 175)
TEXT_OK      = (120, 220, 150)
TEXT_WARN    = (240, 180, 90)
TEXT_BAD     = (240, 100, 110)
PANEL_BG     = (22, 26, 34, 200)


# ---- Camera ---------------------------------------------------------------

class CameraMode(Enum):
    FOLLOW = "follow"    # Behind the drone, smoothly tracking
    FREE   = "free"      # Orbit around a pivot
    FPV    = "fpv"       # From the drone, looking forward
    TOPDOWN = "topdown"  # Straight down


@dataclass
class Camera:
    """Orbit / follow / FPV camera."""
    pos: np.ndarray = field(default_factory=lambda: np.array([15.0, -15.0, 12.0]))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    fov_deg: float = 60.0
    mode: CameraMode = CameraMode.FOLLOW
    # Orbit params (used in FOLLOW/FREE)
    distance: float = 25.0
    azimuth: float = -0.9       # yaw around world z
    elevation: float = 0.55     # pitch above horizon
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def update_follow(self, drone_pos: np.ndarray, drone_yaw: float, dt: float):
        # Pivot smoothly lerps toward drone
        self.pivot += (drone_pos - self.pivot) * min(1.0, dt * 3.0)
        # Gently trail behind the drone's yaw
        target_az = drone_yaw - math.pi / 2.0
        self.azimuth += _angle_diff(target_az, self.azimuth) * min(1.0, dt * 1.5)
        self._recompute_orbit()

    def update_free(self, dt: float):
        self._recompute_orbit()

    def set_topdown(self, pivot: np.ndarray):
        self.pivot = pivot.copy()
        self.pos = pivot + np.array([0.0, 0.001, 60.0])
        self.target = pivot.copy()
        self.up = np.array([0.0, 1.0, 0.0])

    def set_fpv(self, drone_pos: np.ndarray, drone_yaw: float, drone_pitch: float):
        forward = np.array([math.cos(drone_yaw) * math.cos(drone_pitch),
                            math.sin(drone_yaw) * math.cos(drone_pitch),
                            math.sin(drone_pitch)])
        self.pos = drone_pos + np.array([0.0, 0.0, 0.08])
        self.target = self.pos + forward * 5.0
        self.up = np.array([0.0, 0.0, 1.0])

    def _recompute_orbit(self):
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        ce, se = math.cos(self.elevation), math.sin(self.elevation)
        offset = np.array([self.distance * ca * ce,
                           self.distance * sa * ce,
                           self.distance * se])
        self.pos = self.pivot + offset
        self.target = self.pivot.copy()
        self.up = np.array([0.0, 0.0, 1.0])

    def zoom(self, factor: float):
        self.distance = float(np.clip(self.distance * factor, 3.0, 200.0))

    def rotate(self, d_az: float, d_el: float):
        self.azimuth += d_az
        self.elevation = float(np.clip(self.elevation + d_el, -1.4, 1.4))


def _angle_diff(a: float, b: float) -> float:
    """Shortest signed difference a-b in (-pi, pi]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


# ---- Projection -----------------------------------------------------------

class Projector:
    """Builds view matrix and perspective-projects 3D points to 2D screen."""

    def __init__(self, width: int, height: int, near: float = 0.2, far: float = 500.0):
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        self.view = np.eye(4)
        self.fov_scale = 1.0  # half-height / near at fov

    def update(self, cam: Camera):
        # Right-handed view matrix (camera looks down -Z in view space)
        f = cam.target - cam.pos
        n = np.linalg.norm(f)
        if n < 1e-6:
            f = np.array([1.0, 0.0, 0.0])
        else:
            f = f / n
        s = np.cross(f, cam.up)
        sn = np.linalg.norm(s)
        if sn < 1e-6:
            s = np.array([1.0, 0.0, 0.0])
        else:
            s = s / sn
        u = np.cross(s, f)

        m = np.eye(4)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[:3, 3] = -m[:3, :3] @ cam.pos
        self.view = m

        # fov → pixel scale
        fov_rad = math.radians(cam.fov_deg)
        self.fov_scale = (self.height * 0.5) / math.tan(fov_rad * 0.5)

    def project(self, p: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Return (x_screen, y_screen, view_depth) or None if behind camera."""
        v = self.view @ np.array([p[0], p[1], p[2], 1.0])
        # View space: camera looks along -Z, so in-front points have negative Z.
        if v[2] >= -self.near:
            return None
        z = -v[2]
        x = self.width * 0.5 + v[0] / z * self.fov_scale
        y = self.height * 0.5 - v[1] / z * self.fov_scale
        return (x, y, z)

    def project_many(self, pts: Sequence[np.ndarray]) -> List[Optional[Tuple[float, float, float]]]:
        return [self.project(p) for p in pts]


# ---- Draw queue (painter's algorithm) -------------------------------------

@dataclass
class _Drawable:
    depth: float
    kind: str          # 'poly' | 'line' | 'dot' | 'text' | 'circle'
    points: list
    color: tuple
    width: int = 1
    fill: bool = True
    text: str = ""


# ---- Renderer -------------------------------------------------------------

class Renderer:
    """Owns the pygame window, the camera, the projector, and the draw loop."""

    def __init__(self, width: int = 1280, height: int = 800, title: str = "Drone AI — 3D"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_md = pygame.font.SysFont("Consolas", 16)
        self.font_sm = pygame.font.SysFont("Consolas", 13)

        self.camera = Camera()
        self.projector = Projector(width, height)
        self._drawlist: List[_Drawable] = []
        self._pressed = set()
        self._prop_phase = 0.0
        self._sim_speed_index = 1  # index into SPEEDS
        self.paused = False
        self.show_trail = True
        self.show_hud = True

    # ---- Events ---------------------------------------------------------

    SPEEDS = [1, 2, 5, 10, 20]

    @property
    def sim_speed(self) -> int:
        return self.SPEEDS[self._sim_speed_index]

    def handle_events(self, dt: float) -> bool:
        keys = pygame.key.get_pressed()
        # continuous camera controls
        if self.camera.mode in (CameraMode.FREE, CameraMode.FOLLOW):
            rate = 1.8 * dt
            if keys[pygame.K_LEFT]:  self.camera.rotate(-rate, 0)
            if keys[pygame.K_RIGHT]: self.camera.rotate( rate, 0)
            if keys[pygame.K_UP]:    self.camera.rotate(0,  rate)
            if keys[pygame.K_DOWN]:  self.camera.rotate(0, -rate)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key == pygame.K_1:
                    self.camera.mode = CameraMode.FOLLOW
                elif ev.key == pygame.K_2:
                    self.camera.mode = CameraMode.FREE
                elif ev.key == pygame.K_3:
                    self.camera.mode = CameraMode.FPV
                elif ev.key == pygame.K_4:
                    self.camera.mode = CameraMode.TOPDOWN
                elif ev.key == pygame.K_t:
                    self.show_trail = not self.show_trail
                elif ev.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.camera.zoom(1 / 1.15)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.camera.zoom(1.15)
                elif ev.key == pygame.K_LEFTBRACKET:
                    if self._sim_speed_index > 0:
                        self._sim_speed_index -= 1
                elif ev.key == pygame.K_RIGHTBRACKET:
                    if self._sim_speed_index < len(self.SPEEDS) - 1:
                        self._sim_speed_index += 1
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 4:
                    self.camera.zoom(1 / 1.1)
                elif ev.button == 5:
                    self.camera.zoom(1.1)
        return True

    # ---- Scene ----------------------------------------------------------

    def draw_scene(
        self,
        state: DroneState,
        target: Optional[np.ndarray],
        path: Optional[Sequence[np.ndarray]],
        world: Optional[World],
        trail: Optional[Sequence[np.ndarray]] = None,
        waypoints: Optional[Sequence[np.ndarray]] = None,
        hud: Optional[dict] = None,
    ):
        # Animate props (independent of sim)
        self._prop_phase = (self._prop_phase + 0.6) % (2 * math.pi)

        # Update camera
        dt = self.clock.get_time() / 1000.0
        if self.camera.mode == CameraMode.FOLLOW:
            self.camera.update_follow(state.position, float(state.orientation[2]), dt)
        elif self.camera.mode == CameraMode.FREE:
            self.camera.update_free(dt)
        elif self.camera.mode == CameraMode.FPV:
            self.camera.set_fpv(state.position, float(state.orientation[2]),
                                float(state.orientation[1]))
        elif self.camera.mode == CameraMode.TOPDOWN:
            self.camera.set_topdown(state.position)

        self.projector.update(self.camera)
        self._drawlist.clear()

        self._draw_sky()
        self._add_ground()
        if world is not None:
            for ob in world.obstacles:
                self._add_obstacle(ob)
        if trail is not None and self.show_trail:
            self._add_trail(trail)
        if path:
            self._add_path(path, waypoints)
        if target is not None:
            self._add_target_marker(target)
        self._add_drone(state)

        self._drawlist.sort(key=lambda d: -d.depth)
        self._flush_drawlist()

        if self.show_hud:
            self._draw_hud(state, hud)

    # ---- Sky / ground ---------------------------------------------------

    def _draw_sky(self):
        # Vertical gradient fill (fast, done directly to screen)
        steps = 32
        band_h = self.height // steps
        for i in range(steps):
            t = i / (steps - 1)
            c = (
                int(SKY_BOT[0] + (SKY_TOP[0] - SKY_BOT[0]) * (1 - t)),
                int(SKY_BOT[1] + (SKY_TOP[1] - SKY_BOT[1]) * (1 - t)),
                int(SKY_BOT[2] + (SKY_TOP[2] - SKY_BOT[2]) * (1 - t)),
            )
            pygame.draw.rect(self.screen, c, (0, i * band_h, self.width, band_h + 1))

    def _add_ground(self):
        # Checkerboard-like big squares + grid lines
        step = 10.0
        half = 80.0
        n = int(half / step)
        # Checkerboard
        for i in range(-n, n):
            for j in range(-n, n):
                x0, y0 = i * step, j * step
                x1, y1 = x0 + step, y0 + step
                color = GROUND_A if (i + j) % 2 == 0 else GROUND_B
                corners = [
                    np.array([x0, y0, 0.0]),
                    np.array([x1, y0, 0.0]),
                    np.array([x1, y1, 0.0]),
                    np.array([x0, y1, 0.0]),
                ]
                projected = [self.projector.project(p) for p in corners]
                if any(p is None for p in projected):
                    continue
                pts = [(p[0], p[1]) for p in projected]  # type: ignore
                avg_z = sum(p[2] for p in projected) / 4.0  # type: ignore
                self._drawlist.append(_Drawable(
                    depth=avg_z + 0.01, kind='poly', points=pts, color=color, fill=True,
                ))
        # Grid lines
        for i in range(-n, n + 1):
            x = i * step
            p1 = np.array([x, -half, 0.0])
            p2 = np.array([x,  half, 0.0])
            self._add_line(p1, p2, GRID_AXIS_Y if x == 0 else GRID, width=2 if x == 0 else 1)
        for j in range(-n, n + 1):
            y = j * step
            p1 = np.array([-half, y, 0.0])
            p2 = np.array([ half, y, 0.0])
            self._add_line(p1, p2, GRID_AXIS_X if y == 0 else GRID, width=2 if y == 0 else 1)

    # ---- Primitives -----------------------------------------------------

    def _add_line(self, p1: np.ndarray, p2: np.ndarray, color, width: int = 1):
        a = self.projector.project(p1)
        b = self.projector.project(p2)
        if a is None or b is None:
            return
        depth = 0.5 * (a[2] + b[2])
        self._drawlist.append(_Drawable(
            depth=depth, kind='line',
            points=[(a[0], a[1]), (b[0], b[1])], color=color, width=width,
        ))

    def _add_dot(self, p: np.ndarray, color, radius: int = 3):
        a = self.projector.project(p)
        if a is None:
            return
        self._drawlist.append(_Drawable(
            depth=a[2], kind='dot', points=[(a[0], a[1])], color=color, width=radius,
        ))

    def _add_poly(self, vertices: Sequence[np.ndarray], color, fill: bool = True):
        projected = [self.projector.project(p) for p in vertices]
        if any(p is None for p in projected):
            return
        pts = [(p[0], p[1]) for p in projected]  # type: ignore
        depth = sum(p[2] for p in projected) / len(projected)  # type: ignore
        self._drawlist.append(_Drawable(
            depth=depth, kind='poly', points=pts, color=color, fill=fill,
        ))

    # ---- Complex objects -----------------------------------------------

    def _add_obstacle(self, ob: Obstacle):
        # Draw as extruded box with shaded top / sides
        cx, cy, cz = ob.position
        sx, sy, sz = ob.size
        x0, x1 = cx - sx, cx + sx
        y0, y1 = cy - sy, cy + sy
        z0, z1 = max(0.0, cz - sz), cz + sz
        # 8 corners
        c = [np.array([x, y, z]) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
        # side faces
        faces = [
            ([c[0], c[1], c[3], c[2]], OBS_SIDE),  # bottom (rarely visible)
            ([c[4], c[5], c[7], c[6]], OBS_TOP),   # top
            ([c[0], c[1], c[5], c[4]], OBS_SIDE),  # y-low
            ([c[2], c[3], c[7], c[6]], OBS_SIDE),  # y-high
            ([c[0], c[2], c[6], c[4]], OBS_SIDE),  # x-low
            ([c[1], c[3], c[7], c[5]], OBS_SIDE),  # x-high
        ]
        for verts, col in faces:
            self._add_poly(verts, col, fill=True)
        # edges
        edges = [
            (c[0], c[1]), (c[1], c[3]), (c[3], c[2]), (c[2], c[0]),
            (c[4], c[5]), (c[5], c[7]), (c[7], c[6]), (c[6], c[4]),
            (c[0], c[4]), (c[1], c[5]), (c[2], c[6]), (c[3], c[7]),
        ]
        for p1, p2 in edges:
            self._add_line(p1, p2, OBS_EDGE, width=1)

    def _add_drone(self, state: DroneState):
        pos = state.position
        roll, pitch, yaw = state.orientation
        R = _rotation_matrix(roll, pitch, yaw)

        arm = 0.22
        body_half = np.array([0.10, 0.10, 0.04])
        # Body as small cuboid, rotated
        offsets = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
        ], dtype=float) * body_half
        body = [pos + R @ o for o in offsets]
        body_faces = [
            ([body[4], body[5], body[6], body[7]], DRONE_CANOPY),  # top
            ([body[0], body[1], body[5], body[4]], DRONE_BODY),
            ([body[1], body[2], body[6], body[5]], DRONE_BODY_D),
            ([body[2], body[3], body[7], body[6]], DRONE_BODY),
            ([body[3], body[0], body[4], body[7]], DRONE_BODY_D),
            ([body[0], body[1], body[2], body[3]], DRONE_BODY_D),  # bottom
        ]
        for verts, col in body_faces:
            self._add_poly(verts, col, fill=True)

        # 4 motors at X-config corners (FL, FR, RR, RL)
        motor_offsets = np.array([
            [ arm,  arm, 0.0],   # front-left (world +x +y)
            [ arm, -arm, 0.0],   # front-right
            [-arm, -arm, 0.0],   # rear-right
            [-arm,  arm, 0.0],   # rear-left
        ])
        for i, off in enumerate(motor_offsets):
            world_off = R @ off
            motor_pos = pos + world_off
            # arm
            self._add_line(pos, motor_pos, DRONE_ARM, width=3)
            # motor disk — small filled ngon in rotor plane
            prop_r = 0.13
            n_sides = 12
            circle_pts = []
            for k in range(n_sides):
                a = 2 * math.pi * k / n_sides
                local = np.array([prop_r * math.cos(a), prop_r * math.sin(a), 0.0])
                circle_pts.append(motor_pos + R @ local)
            self._add_poly(circle_pts, MOTOR, fill=True)

            # Spinning propeller — a pair of line arcs rotating
            prop_phase = self._prop_phase + (i * 0.7)
            # Draw a blurred disk (light color) + two crossing lines
            # Two-blade prop
            for blade in range(2):
                ang = prop_phase + blade * math.pi
                local1 = np.array([prop_r * math.cos(ang), prop_r * math.sin(ang), 0.02])
                local2 = np.array([prop_r * math.cos(ang + math.pi),
                                   prop_r * math.sin(ang + math.pi), 0.02])
                self._add_line(motor_pos + R @ local1, motor_pos + R @ local2,
                               PROP, width=2)

    def _add_target_marker(self, target: np.ndarray):
        # Vertical pillar + ring
        top = target + np.array([0.0, 0.0, 2.0])
        self._add_line(target, top, TARGET, width=2)
        ring_r = 0.9
        prev = target + np.array([ring_r, 0.0, 0.05])
        n = 16
        for k in range(1, n + 1):
            a = 2 * math.pi * k / n
            nxt = target + np.array([ring_r * math.cos(a), ring_r * math.sin(a), 0.05])
            self._add_line(prev, nxt, TARGET, width=2)
            prev = nxt
        self._add_dot(top + np.array([0.0, 0.0, 0.1]), TARGET, radius=4)

    def _add_trail(self, trail: Sequence[np.ndarray]):
        if len(trail) < 2:
            return
        step = max(1, len(trail) // 200)
        pts = list(trail[::step])
        for i in range(1, len(pts)):
            self._add_line(pts[i - 1], pts[i], TRAIL, width=1)

    def _add_path(self, path: Sequence[np.ndarray], waypoints: Optional[Sequence[np.ndarray]] = None):
        for i in range(1, len(path)):
            self._add_line(path[i - 1], path[i], PATH, width=1)
        pts = waypoints if waypoints is not None else path
        for p in pts:
            self._add_dot(p, WAYPOINT, radius=3)

    # ---- Flush ----------------------------------------------------------

    def _flush_drawlist(self):
        sc = self.screen
        for d in self._drawlist:
            if d.kind == 'poly':
                if len(d.points) < 3:
                    continue
                # Cull offscreen
                if not _any_on_screen(d.points, self.width, self.height):
                    continue
                if d.fill:
                    try:
                        pygame.draw.polygon(sc, d.color, d.points)
                    except Exception:
                        pass
                else:
                    pygame.draw.polygon(sc, d.color, d.points, 1)
            elif d.kind == 'line':
                pygame.draw.line(sc, d.color, d.points[0], d.points[1], d.width)
            elif d.kind == 'dot':
                pygame.draw.circle(sc, d.color, (int(d.points[0][0]), int(d.points[0][1])), d.width)

    # ---- HUD ------------------------------------------------------------

    def _draw_hud(self, state: DroneState, hud: Optional[dict]):
        # Left HUD — telemetry
        lines = [
            ("pos",     f"{state.position[0]:+.1f}, {state.position[1]:+.1f}, {state.position[2]:+.1f} m", TEXT),
            ("vel",     f"{float(np.linalg.norm(state.velocity)):.2f} m/s", TEXT),
            ("tilt",    f"{math.degrees(state.orientation[0]):+.0f}° / {math.degrees(state.orientation[1]):+.0f}°", TEXT),
            ("battery", f"{state.battery*100:.0f}%",
             TEXT_OK if state.battery > 0.4 else TEXT_WARN if state.battery > 0.15 else TEXT_BAD),
            ("camera",  self.camera.mode.value, TEXT_DIM),
            ("speed",   f"{self.sim_speed}x", TEXT_DIM),
        ]
        y = 12
        for k, v, col in lines:
            ks = self.font_sm.render(k, True, TEXT_DIM)
            vs = self.font_md.render(v, True, col)
            self.screen.blit(ks, (16, y + 3))
            self.screen.blit(vs, (78, y))
            y += 20

        # Top-center status badge (task / phase)
        if hud:
            title = hud.get("title", "")
            if title:
                t = self.font_lg.render(title, True, TEXT)
                self.screen.blit(t, (self.width // 2 - t.get_width() // 2, 10))
            sub = hud.get("subtitle", "")
            if sub:
                t = self.font_md.render(sub, True, TEXT_DIM)
                self.screen.blit(t, (self.width // 2 - t.get_width() // 2, 36))
            # Right-side: metrics
            metrics = hud.get("metrics", [])
            y = 12
            for label, val, col in metrics:
                ls = self.font_sm.render(label, True, TEXT_DIM)
                vs = self.font_md.render(str(val), True, col or TEXT)
                self.screen.blit(ls, (self.width - 220, y + 3))
                self.screen.blit(vs, (self.width - 140, y))
                y += 20

        # Bottom help bar
        help_text = ("[1/2/3/4] cam  [Space] pause  [T] trail  [H] hud  "
                     "[[]/][] speed  [+/-] zoom  [Arrows] orbit  [Q] quit")
        hs = self.font_sm.render(help_text, True, TEXT_DIM)
        self.screen.blit(hs, (self.width // 2 - hs.get_width() // 2, self.height - 20))

        if self.paused:
            t = self.font_lg.render("PAUSED", True, TEXT_WARN)
            self.screen.blit(t, (self.width // 2 - t.get_width() // 2, self.height // 2 - 12))

    def flip(self):
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass


# ---- Helpers --------------------------------------------------------------

def _any_on_screen(pts: Sequence[Tuple[float, float]], w: int, h: int) -> bool:
    minx = min(p[0] for p in pts); maxx = max(p[0] for p in pts)
    miny = min(p[1] for p in pts); maxy = max(p[1] for p in pts)
    return not (maxx < 0 or minx > w or maxy < 0 or miny > h)


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler (body → world). Matches simulation/physics.py."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ])
