"""Stage launcher — the main window you see when you start the app.

Pick a training stage (or free-fly demo) and a new window opens running it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

import pygame

from drone_ai.viz.trainer_ui import STAGE_DEFS, TrainConfig, TrainerUI


# ---- Styling --------------------------------------------------------------

BG            = (14, 18, 26)
PANEL         = (22, 28, 38)
PANEL_HOVER   = (32, 42, 58)
PANEL_ACTIVE  = (50, 95, 150)
BORDER        = (60, 72, 95)
TEXT          = (225, 230, 240)
TEXT_DIM      = (150, 160, 175)
TEXT_ACCENT   = (120, 220, 150)
TEXT_TITLE    = (180, 220, 250)


@dataclass
class StageCard:
    key: str
    title: str
    subtitle: str
    description: str
    accent: tuple


STAGE_CARDS: List[StageCard] = [
    StageCard("hover", "Hover",
              "Learn to stay still",
              "Basic attitude + altitude hold. The drone must hover near a target point.",
              (120, 220, 150)),
    StageCard("waypoint", "Waypoint",
              "Navigate to scattered targets",
              "Harder hover variant — target drifts further out each episode.",
              (140, 200, 230)),
    StageCard("delivery", "Delivery",
              "Pickup → dropzone, one package",
              "Fly to a pickup point, then to a dropzone. Single-delivery task.",
              (240, 170, 90)),
    StageCard("delivery_route", "Delivery Route",
              "Multi-stop with obstacles",
              "2-6 deliveries, up to 20 obstacles. Route optimization + avoidance.",
              (220, 140, 210)),
    StageCard("deployment", "Deployment Ready",
              "Full difficulty + domain rand",
              "Max difficulty, randomized physics. Prepares policy for sim-to-real.",
              (240, 100, 110)),
    StageCard("demo", "Free-Fly Demo",
              "Run with the built-in PD controller",
              "No training — just watch the drone fly its demo mission in 3D.",
              (180, 220, 250)),
]


WINDOW_W = 1100
WINDOW_H = 720


class Launcher:
    def __init__(self):
        self._init_pygame()
        self.hover_idx: Optional[int] = None
        self.selected_idx: int = 0
        self.total_updates = 400  # default per-stage training budget

    def _init_pygame(self):
        """Idempotent pygame + window + font init.

        Separated so we can call it again after launching a child window,
        in case the child tore down subsystems we still depend on (fonts
        are the common casualty when anything in the stack ever calls
        pygame.quit instead of pygame.display.quit).
        """
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Drone AI — Launcher")
        self.clock = pygame.time.Clock()
        self.font_xl = pygame.font.SysFont("Consolas", 34, bold=True)
        self.font_lg = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_md = pygame.font.SysFont("Consolas", 16)
        self.font_sm = pygame.font.SysFont("Consolas", 13)

    def run(self) -> None:
        running = True
        try:
            while running:
                running = self._handle_events()
                self._draw()
                pygame.display.flip()
                self.clock.tick(60)
        finally:
            # Final cleanup — only the top-level owner calls pygame.quit.
            try:
                pygame.quit()
            except Exception:
                pass

    # ---- Events ---------------------------------------------------------

    def _handle_events(self) -> bool:
        mx, my = pygame.mouse.get_pos()
        self.hover_idx = self._card_at(mx, my)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif ev.key in (pygame.K_DOWN, pygame.K_RIGHT):
                    self.selected_idx = (self.selected_idx + 1) % len(STAGE_CARDS)
                elif ev.key in (pygame.K_UP, pygame.K_LEFT):
                    self.selected_idx = (self.selected_idx - 1) % len(STAGE_CARDS)
                elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._launch(STAGE_CARDS[self.selected_idx])
                elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                    self.total_updates = min(10000, self.total_updates + 100)
                elif ev.key == pygame.K_MINUS:
                    self.total_updates = max(50, self.total_updates - 100)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if self.hover_idx is not None:
                    self.selected_idx = self.hover_idx
                    self._launch(STAGE_CARDS[self.hover_idx])
        return True

    # ---- Launch ---------------------------------------------------------

    def _launch(self, card: StageCard):
        # Hide launcher while child runs.
        pygame.display.quit()
        try:
            if card.key == "demo":
                _run_demo()
            else:
                cfg = TrainConfig(stage=card.key, total_updates=self.total_updates)
                TrainerUI(cfg).run()
        except Exception as e:
            # Never let a child crash kill the launcher — surface it instead.
            print(f"[launcher] stage '{card.key}' raised: {e}")
        finally:
            # Reopen the launcher window. Full re-init guards against children
            # that tore down fonts or other subsystems we depend on.
            self._init_pygame()

    # ---- Layout ---------------------------------------------------------

    GRID_COLS = 3
    CARD_W = 320
    CARD_H = 170
    CARD_GAP = 20
    GRID_X = 50
    GRID_Y = 170

    def _card_rect(self, idx: int) -> pygame.Rect:
        col = idx % self.GRID_COLS
        row = idx // self.GRID_COLS
        x = self.GRID_X + col * (self.CARD_W + self.CARD_GAP)
        y = self.GRID_Y + row * (self.CARD_H + self.CARD_GAP)
        return pygame.Rect(x, y, self.CARD_W, self.CARD_H)

    def _card_at(self, mx: int, my: int) -> Optional[int]:
        for i in range(len(STAGE_CARDS)):
            if self._card_rect(i).collidepoint(mx, my):
                return i
        return None

    # ---- Draw -----------------------------------------------------------

    def _draw(self):
        self.screen.fill(BG)
        # Header
        title = self.font_xl.render("DRONE AI", True, TEXT_TITLE)
        self.screen.blit(title, (50, 40))
        sub = self.font_md.render("Pick a stage to train. Each stage opens a live 3D training window.",
                                  True, TEXT_DIM)
        self.screen.blit(sub, (50, 84))

        # Budget control
        budget_y = 118
        budget = self.font_md.render(f"Training budget: {self.total_updates} updates",
                                     True, TEXT_DIM)
        self.screen.blit(budget, (50, budget_y))
        hint = self.font_sm.render("  [+/-] adjust   [Enter] launch selected   [Q] quit",
                                   True, TEXT_DIM)
        self.screen.blit(hint, (50, budget_y + 20))

        # Cards
        for i, card in enumerate(STAGE_CARDS):
            rect = self._card_rect(i)
            is_sel = (i == self.selected_idx)
            is_hover = (i == self.hover_idx)
            bg = PANEL_ACTIVE if is_sel else PANEL_HOVER if is_hover else PANEL
            pygame.draw.rect(self.screen, bg, rect, border_radius=8)
            pygame.draw.rect(self.screen, card.accent, rect, 2, border_radius=8)
            # Title
            t = self.font_lg.render(card.title, True, TEXT)
            self.screen.blit(t, (rect.x + 16, rect.y + 14))
            # Accent chip
            chip_w = 48
            pygame.draw.rect(self.screen, card.accent,
                             (rect.right - chip_w - 14, rect.y + 18, chip_w, 8),
                             border_radius=4)
            # Subtitle
            s = self.font_md.render(card.subtitle, True, TEXT_ACCENT)
            self.screen.blit(s, (rect.x + 16, rect.y + 50))
            # Description (wrap)
            self._blit_wrapped(card.description, rect.x + 16, rect.y + 80,
                               rect.width - 32, self.font_sm, TEXT_DIM)
            # Key badge bottom-right
            badge = f"{card.key}"
            b = self.font_sm.render(badge, True, TEXT_DIM)
            self.screen.blit(b, (rect.right - b.get_width() - 14, rect.bottom - 22))

    def _blit_wrapped(self, text: str, x: int, y: int, max_w: int, font, color):
        words = text.split(" ")
        line = ""
        dy = 0
        for w in words:
            trial = (line + " " + w).strip()
            if font.size(trial)[0] > max_w:
                surf = font.render(line, True, color)
                self.screen.blit(surf, (x, y + dy))
                dy += font.get_height() + 2
                line = w
            else:
                line = trial
        if line:
            surf = font.render(line, True, color)
            self.screen.blit(surf, (x, y + dy))


# ---- Free-fly demo --------------------------------------------------------

def _run_demo():
    """Run the built-in DroneAI (PD controller + full 4-layer stack) with 3D viz.

    Auto-generates a fresh mission (new obstacles + deliveries) whenever
    the current mission finishes or the drone crashes — so the viewer
    never just "stops flying."
    """
    import numpy as np
    from drone_ai.drone import DroneAI
    from drone_ai.modules.manager.planner import Priority
    from drone_ai.simulation.world import World
    from drone_ai.viz.renderer3d import Renderer

    missions_completed = 0
    crashes = 0
    run_seed = [0]

    def fresh_mission() -> DroneAI:
        seed = run_seed[0]
        run_seed[0] += 1
        d = DroneAI(seed=seed)
        d.reset()
        rng = np.random.default_rng(seed)
        w = World()
        w.generate_random_obstacles(rng.integers(6, 14), rng)
        d.set_obstacles(w.obstacles)
        # Random deliveries (3-5) spread around base
        n = int(rng.integers(3, 6))
        priorities = [Priority.NORMAL, Priority.URGENT, Priority.CRITICAL]
        for _ in range(n):
            angle = float(rng.uniform(0, 2 * np.pi))
            dist = float(rng.uniform(20, 55))
            target = [dist * np.cos(angle), dist * np.sin(angle), 0.0]
            d.add_delivery(target, priorities[int(rng.integers(0, 3))])
        return d

    drone = fresh_mission()
    renderer = Renderer(title="Drone AI — Free-fly Demo")
    running = True
    while running:
        running = renderer.handle_events(1 / 60)
        if not running:
            break
        if not renderer.paused:
            for _ in range(renderer.sim_speed):
                _, done = drone.step()
                if done:
                    if drone.physics.state.crashed:
                        crashes += 1
                    else:
                        missions_completed += 1
                    drone = fresh_mission()
                    break
        st = drone.get_status()
        mgr_state = drone.manager.state
        waypoints = [d.target for d in mgr_state.pending]
        if mgr_state.current is not None:
            waypoints.insert(0, mgr_state.current.target)
        hud = {
            "title":    "FREE-FLY DEMO",
            "subtitle": "4-layer stack with PD controller (no training)",
            "metrics": [
                ("state",    st.state.value, None),
                ("step",     str(st.step), None),
                ("done",     str(st.deliveries_done), None),
                ("pending",  str(st.deliveries_pending), None),
                ("missions", str(missions_completed), None),
                ("crashes",  str(crashes), None),
            ],
        }
        renderer.draw_scene(
            state=drone.physics.state,
            target=drone._path[drone._path_idx] if drone._path and drone._path_idx < len(drone._path) else None,
            path=drone._path,
            world=drone.world,
            trail=drone._position_history,
            waypoints=waypoints,
            hud=hud,
        )
        renderer.flip()
    renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Drone AI launcher")
    parser.add_argument("--stage", default=None,
                        help="Skip the menu and launch this stage directly "
                             "(hover/waypoint/delivery/delivery_route/deployment/demo)")
    parser.add_argument("--updates", type=int, default=400)
    args = parser.parse_args()

    if args.stage:
        exit_code = 0
        try:
            if args.stage == "demo":
                _run_demo()
            elif args.stage in STAGE_DEFS:
                TrainerUI(TrainConfig(stage=args.stage, total_updates=args.updates)).run()
            else:
                print(f"Unknown stage: {args.stage}", file=sys.stderr)
                exit_code = 2
        finally:
            try:
                pygame.quit()
            except Exception:
                pass
            sys.exit(exit_code)
    else:
        Launcher().run()


if __name__ == "__main__":
    main()
