"""Visual pygame app — top-down drone simulator with a click-to-add-delivery UI.

Controls:
    Left click  : add NORMAL delivery at cursor
    Right click : add URGENT delivery at cursor
    Middle click: add CRITICAL delivery at cursor
    Space       : pause / resume
    R           : reset world and drone (keeps deliveries)
    N           : new scenario (clear + regenerate obstacles + sample deliveries)
    C           : clear all deliveries
    O           : add 5 more random obstacles
    1-5         : simulation speed (1x, 2x, 5x, 10x, 20x)
    +/-         : zoom in / out
    Arrows      : pan view
    Esc / Q     : quit
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np

try:
    import pygame
except ImportError as e:  # pragma: no cover
    print("pygame is required. Install with:  pip install pygame", file=sys.stderr)
    raise

from drone_ai.drone import DroneAI, SystemState
from drone_ai.modules.manager.planner import Priority
from drone_ai.simulation.world import World, Obstacle


# ---- Colors ---------------------------------------------------------------

BG           = (18, 22, 30)
GRID         = (35, 42, 55)
GRID_AXIS    = (70, 80, 100)
PANEL_BG     = (26, 30, 40)
PANEL_BORDER = (70, 80, 100)
TEXT         = (220, 225, 235)
TEXT_DIM     = (150, 160, 175)
TEXT_OK      = (120, 220, 150)
TEXT_WARN    = (240, 180, 90)
TEXT_BAD     = (240, 100, 110)

DRONE        = (90, 200, 250)
DRONE_TRAIL  = (90, 200, 250, 120)
PATH         = (160, 180, 220)
BASE         = (120, 220, 150)
OBSTACLE     = (90, 100, 120)
OBSTACLE_OUT = (130, 145, 170)

PRIO_COLORS = {
    Priority.NORMAL:   (200, 210, 230),
    Priority.URGENT:   (250, 170, 70),
    Priority.CRITICAL: (250, 90, 100),
}


# ---- Camera ---------------------------------------------------------------

class Camera:
    def __init__(self, width: int, height: int, pan_px: Tuple[int, int] = (0, 0)):
        self.view_width = width
        self.view_height = height
        self.scale = 4.0  # pixels per meter
        self.cx = 0.0  # world center
        self.cy = 0.0
        self.pan_px = pan_px  # where (0,0) of view sits in screen coords

    def world_to_screen(self, pos: np.ndarray) -> Tuple[int, int]:
        sx = self.pan_px[0] + self.view_width  // 2 + (pos[0] - self.cx) * self.scale
        sy = self.pan_px[1] + self.view_height // 2 - (pos[1] - self.cy) * self.scale
        return int(sx), int(sy)

    def screen_to_world(self, sx: int, sy: int) -> np.ndarray:
        x = (sx - self.pan_px[0] - self.view_width  // 2) / self.scale + self.cx
        y = (self.pan_px[1] + self.view_height // 2 - sy) / self.scale + self.cy
        return np.array([x, y, 0.0])

    def zoom(self, factor: float):
        self.scale = float(np.clip(self.scale * factor, 0.8, 40.0))

    def pan(self, dx: float, dy: float):
        self.cx += dx
        self.cy += dy


# ---- App ------------------------------------------------------------------

WINDOW_W = 1280
WINDOW_H = 800
PANEL_W = 320
VIEW_W = WINDOW_W - PANEL_W
VIEW_H = WINDOW_H


class App:
    def __init__(self, seed: int = 0, flycontrol_model: str | None = None):
        pygame.init()
        pygame.display.set_caption("Drone AI — autonomous delivery simulator")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_md = pygame.font.SysFont("Consolas", 16)
        self.font_sm = pygame.font.SysFont("Consolas", 13)

        self.camera = Camera(VIEW_W, VIEW_H)
        self.seed = seed
        self.flycontrol_model = flycontrol_model
        self.drone = DroneAI(flycontrol_model=flycontrol_model, seed=seed)

        self.paused = False
        self.sim_speed = 2  # sim steps per frame
        self._new_scenario(reset_camera=True)

    # ---- Scenario setup --------------------------------------------------

    def _new_scenario(self, reset_camera: bool = False):
        self.drone = DroneAI(flycontrol_model=self.flycontrol_model, seed=self.seed)
        self.drone.reset()
        # Obstacles
        rng = np.random.default_rng(self.seed)
        w = World()
        w.generate_random_obstacles(12, rng)
        self.drone.set_obstacles(w.obstacles)
        # Sample deliveries
        self.drone.add_delivery([30.0, 20.0, 0.0], Priority.URGENT)
        self.drone.add_delivery([-25.0, 35.0, 0.0], Priority.NORMAL)
        self.drone.add_delivery([40.0, -30.0, 0.0], Priority.CRITICAL)
        self.seed += 1
        if reset_camera:
            self.camera.cx = 0.0
            self.camera.cy = 0.0
            self.camera.scale = 4.0

    def _reset_drone_only(self):
        obstacles = list(self.drone.world.obstacles)
        pending = [(d.target.copy(), d.priority, d.weight)
                   for d in self.drone.manager.state.pending]
        cur = self.drone.manager.state.current
        if cur is not None:
            pending.insert(0, (cur.target.copy(), cur.priority, cur.weight))
        self.drone = DroneAI(flycontrol_model=self.flycontrol_model, seed=self.seed)
        self.drone.reset()
        self.drone.set_obstacles(obstacles)
        for target, priority, weight in pending:
            self.drone.add_delivery(target, priority, weight)

    def _add_delivery_at_cursor(self, mx: int, my: int, priority: Priority):
        if mx > VIEW_W:
            return
        world = self.camera.screen_to_world(mx, my)
        self.drone.add_delivery(world, priority)

    def _add_random_obstacles(self, n: int = 5):
        rng = np.random.default_rng(self.seed + 99)
        new_obs = []
        for _ in range(n):
            pos = rng.uniform(np.array([-80.0, -80.0, 0.0]), np.array([80.0, 80.0, 0.0]))
            pos[2] = rng.uniform(2.0, 20.0)
            sz = rng.uniform(1.5, 6.0, size=3)
            sz[2] = rng.uniform(5.0, 30.0)
            new_obs.append(Obstacle(position=pos, size=sz))
        combined = list(self.drone.world.obstacles) + new_obs
        self.drone.set_obstacles(combined)
        self.seed += 1

    # ---- Main loop -------------------------------------------------------

    def run(self):
        running = True
        while running:
            running = self._handle_events()
            if not self.paused:
                for _ in range(self.sim_speed):
                    state, done = self.drone.step()
                    if done:
                        break
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_drone_only()
                elif event.key == pygame.K_n:
                    self._new_scenario()
                elif event.key == pygame.K_c:
                    self.drone.manager.reset()
                elif event.key == pygame.K_o:
                    self._add_random_obstacles()
                elif event.key == pygame.K_1:
                    self.sim_speed = 1
                elif event.key == pygame.K_2:
                    self.sim_speed = 2
                elif event.key == pygame.K_3:
                    self.sim_speed = 5
                elif event.key == pygame.K_4:
                    self.sim_speed = 10
                elif event.key == pygame.K_5:
                    self.sim_speed = 20
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.camera.zoom(1.15)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.camera.zoom(1/1.15)
                elif event.key == pygame.K_LEFT:
                    self.camera.pan(-10, 0)
                elif event.key == pygame.K_RIGHT:
                    self.camera.pan(10, 0)
                elif event.key == pygame.K_UP:
                    self.camera.pan(0, 10)
                elif event.key == pygame.K_DOWN:
                    self.camera.pan(0, -10)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:
                    self._add_delivery_at_cursor(mx, my, Priority.NORMAL)
                elif event.button == 2:
                    self._add_delivery_at_cursor(mx, my, Priority.CRITICAL)
                elif event.button == 3:
                    self._add_delivery_at_cursor(mx, my, Priority.URGENT)
                elif event.button == 4:
                    self.camera.zoom(1.1)
                elif event.button == 5:
                    self.camera.zoom(1/1.1)
        return True

    # ---- Drawing ---------------------------------------------------------

    def _draw(self):
        self.screen.fill(BG)
        self._draw_grid()
        self._draw_obstacles()
        self._draw_trail()
        self._draw_path()
        self._draw_deliveries()
        self._draw_base()
        self._draw_drone()
        self._draw_panel()

    def _draw_grid(self):
        step = 10  # meters
        # Find world-space range visible
        tl = self.camera.screen_to_world(0, 0)
        br = self.camera.screen_to_world(VIEW_W, VIEW_H)
        x_min = int(np.floor(min(tl[0], br[0]) / step)) * step
        x_max = int(np.ceil (max(tl[0], br[0]) / step)) * step
        y_min = int(np.floor(min(tl[1], br[1]) / step)) * step
        y_max = int(np.ceil (max(tl[1], br[1]) / step)) * step
        for x in range(x_min, x_max + 1, step):
            s1 = self.camera.world_to_screen(np.array([x, y_min, 0]))
            s2 = self.camera.world_to_screen(np.array([x, y_max, 0]))
            color = GRID_AXIS if x == 0 else GRID
            pygame.draw.line(self.screen, color, s1, s2, 1)
        for y in range(y_min, y_max + 1, step):
            s1 = self.camera.world_to_screen(np.array([x_min, y, 0]))
            s2 = self.camera.world_to_screen(np.array([x_max, y, 0]))
            color = GRID_AXIS if y == 0 else GRID
            pygame.draw.line(self.screen, color, s1, s2, 1)

    def _draw_obstacles(self):
        for ob in self.drone.world.obstacles:
            cx, cy = self.camera.world_to_screen(ob.position)
            w = max(2, int(2 * ob.size[0] * self.camera.scale))
            h = max(2, int(2 * ob.size[1] * self.camera.scale))
            rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
            # Clip to viewport before drawing
            if rect.right < 0 or rect.left > VIEW_W or rect.bottom < 0 or rect.top > VIEW_H:
                continue
            pygame.draw.rect(self.screen, OBSTACLE, rect)
            pygame.draw.rect(self.screen, OBSTACLE_OUT, rect, 1)

    def _draw_base(self):
        bx, by = self.camera.world_to_screen(self.drone.BASE)
        pygame.draw.circle(self.screen, BASE, (bx, by), 8, 2)
        pygame.draw.circle(self.screen, BASE, (bx, by), 3)
        label = self.font_sm.render("BASE", True, BASE)
        self.screen.blit(label, (bx + 10, by - 8))

    def _draw_deliveries(self):
        st = self.drone.manager.state
        pending = list(st.pending)
        if st.current is not None:
            pending.append(st.current)
        for d in pending:
            col = PRIO_COLORS.get(d.priority, PRIO_COLORS[Priority.NORMAL])
            x, y = self.camera.world_to_screen(d.target)
            is_current = (d is st.current)
            pygame.draw.circle(self.screen, col, (x, y), 7 if is_current else 6)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 7 if is_current else 6,
                               2 if is_current else 1)
            label = self.font_sm.render(d.priority.name[0], True, (20, 20, 28))
            self.screen.blit(label, label.get_rect(center=(x, y)))
        for d in st.completed:
            x, y = self.camera.world_to_screen(d.target)
            pygame.draw.circle(self.screen, TEXT_OK, (x, y), 4, 1)
        for d in st.failed:
            x, y = self.camera.world_to_screen(d.target)
            pygame.draw.circle(self.screen, TEXT_BAD, (x, y), 4, 1)

    def _draw_path(self):
        if not self.drone._path or len(self.drone._path) < 2:
            return
        pts = [self.camera.world_to_screen(p) for p in self.drone._path]
        pygame.draw.lines(self.screen, PATH, False, pts, 1)
        for p in pts[self.drone._path_idx:]:
            pygame.draw.circle(self.screen, PATH, p, 2)

    def _draw_trail(self):
        hist = self.drone._position_history
        if len(hist) < 2:
            return
        step = max(1, len(hist) // 400)
        pts = [self.camera.world_to_screen(p) for p in hist[::step]]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, (60, 110, 150), False, pts, 1)

    def _draw_drone(self):
        s = self.drone.physics.state
        x, y = self.camera.world_to_screen(s.position)
        yaw = float(s.orientation[2])
        r = 9
        # Triangle pointing in heading direction (world +x is right, +y is up)
        tip = (x + r * np.cos(yaw), y - r * np.sin(yaw))
        left = (x + r * np.cos(yaw + 2.5), y - r * np.sin(yaw + 2.5))
        right = (x + r * np.cos(yaw - 2.5), y - r * np.sin(yaw - 2.5))
        color = TEXT_BAD if s.crashed else DRONE
        pygame.draw.polygon(self.screen, color, [tip, left, right])
        pygame.draw.polygon(self.screen, (255, 255, 255), [tip, left, right], 1)
        # Altitude text
        alt = self.font_sm.render(f"z={s.position[2]:.1f}m", True, TEXT_DIM)
        self.screen.blit(alt, (x + 12, y + 6))

    # ---- Side panel ------------------------------------------------------

    def _draw_panel(self):
        x0 = VIEW_W
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, PANEL_W, WINDOW_H))
        pygame.draw.line(self.screen, PANEL_BORDER, (x0, 0), (x0, WINDOW_H), 2)

        st = self.drone.get_status()
        s = self.drone.physics.state
        mgr = self.drone.manager.get_summary()

        y = 16
        title = self.font_lg.render("DRONE AI", True, TEXT)
        self.screen.blit(title, (x0 + 16, y)); y += 28
        sub = self.font_sm.render(f"4-layer autonomous delivery", True, TEXT_DIM)
        self.screen.blit(sub, (x0 + 16, y)); y += 20

        # Status box
        y = self._section(x0, y, "STATUS")
        state_color = (
            TEXT_BAD if self.drone.system_state == SystemState.CRASHED
            else TEXT_OK if self.drone.system_state == SystemState.LANDED
            else TEXT_WARN if self.paused
            else TEXT
        )
        state_txt = f"{self.drone.system_state.value.upper()}"
        if self.paused:
            state_txt += " (paused)"
        y = self._kv(x0, y, "state", state_txt, value_color=state_color)
        y = self._kv(x0, y, "position", f"{s.position[0]:+.1f}, {s.position[1]:+.1f}, {s.position[2]:+.1f}")
        y = self._kv(x0, y, "velocity", f"{np.linalg.norm(s.velocity):.2f} m/s")
        y = self._kv(x0, y, "tilt",     f"{np.degrees(s.orientation[0]):+.1f}° / {np.degrees(s.orientation[1]):+.1f}°")
        bat_color = TEXT_OK if s.battery > 0.4 else TEXT_WARN if s.battery > 0.15 else TEXT_BAD
        y = self._kv(x0, y, "battery",  f"{s.battery*100:.1f}%", value_color=bat_color)
        y = self._kv(x0, y, "step",     f"{st.step}")
        y += 6

        # Mission box
        y = self._section(x0, y, "MISSION")
        y = self._kv(x0, y, "done",     f"{mgr['completed']}")
        y = self._kv(x0, y, "failed",   f"{mgr['failed']}", value_color=TEXT_BAD if mgr['failed'] else TEXT)
        y = self._kv(x0, y, "pending",  f"{mgr['pending']}")
        y = self._kv(x0, y, "total km", f"{mgr['total_distance_m']/1000:.2f}")
        y += 6

        # Controls box
        y = self._section(x0, y, "SIM SPEED")
        speeds = [("1", 1), ("2", 2), ("5", 5), ("10", 10), ("20", 20)]
        bx = x0 + 16
        for key, val in speeds:
            active = self.sim_speed == val
            bg = (70, 110, 160) if active else (50, 58, 75)
            fg = (255, 255, 255) if active else TEXT_DIM
            pygame.draw.rect(self.screen, bg, (bx, y, 44, 24), border_radius=4)
            lbl = self.font_md.render(f"{val}x", True, fg)
            self.screen.blit(lbl, lbl.get_rect(center=(bx + 22, y + 12)))
            bx += 50
        y += 32

        # Help box
        y = self._section(x0, y, "CONTROLS")
        help_lines = [
            ("L-click",  "add normal delivery"),
            ("R-click",  "add urgent delivery"),
            ("M-click",  "add critical delivery"),
            ("Space",    "pause / resume"),
            ("R",        "reset drone"),
            ("N",        "new scenario"),
            ("C",        "clear deliveries"),
            ("O",        "+5 obstacles"),
            ("+ / -",    "zoom"),
            ("Arrows",   "pan"),
            ("1-5",      "sim speed"),
            ("Q / Esc",  "quit"),
        ]
        for key, desc in help_lines:
            k = self.font_sm.render(key, True, TEXT)
            d = self.font_sm.render(desc, True, TEXT_DIM)
            self.screen.blit(k, (x0 + 16, y))
            self.screen.blit(d, (x0 + 100, y))
            y += 16

    def _section(self, x0: int, y: int, title: str) -> int:
        hdr = self.font_md.render(title, True, TEXT_DIM)
        self.screen.blit(hdr, (x0 + 16, y))
        y += 20
        pygame.draw.line(self.screen, PANEL_BORDER, (x0 + 16, y - 2), (x0 + PANEL_W - 16, y - 2), 1)
        return y + 4

    def _kv(self, x0: int, y: int, key: str, val: str, value_color=TEXT) -> int:
        k = self.font_sm.render(key, True, TEXT_DIM)
        v = self.font_md.render(val, True, value_color)
        self.screen.blit(k, (x0 + 16, y + 2))
        self.screen.blit(v, (x0 + 110, y))
        return y + 20


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Drone AI visual simulator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None, help="Trained FlyControl .pt file (optional)")
    args = parser.parse_args()
    App(seed=args.seed, flycontrol_model=args.model).run()


if __name__ == "__main__":
    main()
