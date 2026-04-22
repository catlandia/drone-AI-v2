"""Stage launcher — the main window you see when you start the app.

Pick a card and a new window opens running that layer's training /
benchmark. Three sections:

  FlyControl curriculum (Layer 4)
    hover → waypoint → delivery → delivery_route → deployment
    Each stage warm-starts from the previous stage's latest checkpoint.

  Module benchmarks (Layers 1, 2, 3, 5)
    Manager / Pathfinder / Perception / Adaptive
    Each runs its standalone benchmark and writes a tier-named .pt
    to models/<module>/ plus a row to models/runs.csv.

  Phase 2 ops (Layers 6, 7, 8) + Free-fly demo
    Storage / Personality / Swarm / Demo
    Storage round-trips synthetic field rows; Personality auto-exports
    from the latest FlyControl checkpoint; Swarm runs the coordinator
    on synthetic multi-drone scenes; Demo runs the PD controller live.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pygame

from drone_ai.grading import RunLogger, parse_model_name
from drone_ai.viz.trainer_ui import (
    STAGE_DEFS, STAGE_ORDER, TrainConfig, TrainerUI,
    latest_flycontrol_checkpoint,
)


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
    key: str                # routes to the launch dispatcher
    title: str
    subtitle: str
    description: str
    accent: tuple
    badge: str = ""         # small tag e.g. "Layer 4" / "demo"
    section: str = "main"   # section heading bucket


# Section keys / labels in display order.
SECTIONS = [
    ("flycontrol", "FlyControl Curriculum (Layer 4)"),
    ("modules",    "Module Benchmarks (Layers 1, 2, 3, 5)"),
    ("phase2",     "Phase 2 Ops + Demo"),
]


STAGE_CARDS: List[StageCard] = [
    # ---- FlyControl curriculum (Layer 4) -----------------------------------
    StageCard("hover", "Hover", "Learn to stay still",
              "Basic attitude + altitude hold. The drone must hover near a target point.",
              (120, 220, 150), badge="L4 · stage 1", section="flycontrol"),
    StageCard("waypoint", "Waypoint", "Navigate to scattered targets",
              "Harder hover variant — target drifts further out each episode.",
              (140, 200, 230), badge="L4 · stage 2", section="flycontrol"),
    StageCard("delivery", "Delivery", "Pickup → dropzone, one package",
              "Fly to a pickup point, then to a dropzone. Single-delivery task.",
              (240, 170, 90), badge="L4 · stage 3", section="flycontrol"),
    StageCard("delivery_route", "Delivery Route", "Multi-stop with obstacles",
              "2-6 deliveries, up to 20 obstacles. Route optimization + avoidance.",
              (220, 140, 210), badge="L4 · stage 4", section="flycontrol"),
    StageCard("deployment", "Deployment Ready", "Full difficulty + domain rand",
              "Max difficulty, randomized physics. Prepares policy for sim-to-real.",
              (240, 100, 110), badge="L4 · stage 5", section="flycontrol"),

    # ---- Module benchmarks (Layers 1, 2, 3, 5) ----------------------------
    StageCard("manager", "Manager", "Mission planning",
              "Greedy-TSP routing, priority queue, pre-flight feasibility. Layer 1 benchmark.",
              (200, 180, 100), badge="L1", section="modules"),
    StageCard("pathfinder", "Pathfinder", "A* + RRT planning",
              "Path optimality + collision-avoidance benchmark across random worlds.",
              (160, 200, 100), badge="L2", section="modules"),
    StageCard("perception", "Perception", "Obstacle detection",
              "Detection rate, false positives, position error across scenes. Sub-models split.",
              (100, 200, 200), badge="L3", section="modules"),
    StageCard("adaptive", "Adaptive", "Online fine-tune",
              "Warden-guarded online learning vs frozen baseline on a perturbed env. Layer 5.",
              (220, 110, 220), badge="L5", section="modules"),

    # ---- Phase 2 ops (Layers 6, 7, 8) + demo -----------------------------
    StageCard("storage", "Storage", "Field-log round-trip",
              "Layer 6: write synthetic field rows, read them back, verify the on-drone log.",
              (180, 130, 100), badge="L6", section="phase2"),
    StageCard("personality", "Personality", "Export delta artifact",
              "Layer 7: build a transferable personality from the latest FlyControl checkpoint.",
              (130, 180, 220), badge="L7", section="phase2"),
    StageCard("swarm", "Swarm", "Multi-drone coordinator",
              "Layer 8: visual mutual avoidance + swarm-mate-failed contingency benchmark.",
              (220, 200, 120), badge="L8", section="phase2"),
    StageCard("demo", "Free-Fly Demo", "Run the full stack live",
              "No training — just watch the drone fly its demo mission in 3D.",
              (180, 220, 250), badge="demo", section="phase2"),
]


WINDOW_W = 1280
WINDOW_H = 980

RUN_LOG_PATH = "models/runs.csv"
MODELS_ROOT = "models"


def _load_recent_runs(limit: int = 6) -> List[Dict[str, str]]:
    try:
        rows = RunLogger(RUN_LOG_PATH).read_all()
    except Exception:
        rows = []
    return rows[-limit:][::-1]


def _latest_flycontrol_per_stage() -> Dict[str, str]:
    """Newest flycontrol checkpoint filename for each curriculum stage."""
    latest: Dict[str, str] = {}
    for stage in STAGE_ORDER:
        path = latest_flycontrol_checkpoint(MODELS_ROOT, stage)
        if path:
            latest[stage] = os.path.basename(path)
    return latest


# ---- Per-layer dispatch ----------------------------------------------------

def _run_module(key: str) -> None:
    """Dispatch a non-FlyControl card. Each branch runs a benchmark
    that prints to stdout and writes a runs.csv row + a tier-named
    .pt under models/<module>/. The launcher reopens after the call
    returns and refreshes its strip from the new files."""
    if key == "manager":
        from drone_ai.modules.manager.train import run_training
        run_training(grade="P", trials=20, save_dir="models/manager")
    elif key == "pathfinder":
        from drone_ai.modules.pathfinder.train import run_training
        run_training(trials=30, save_dir="models/pathfinder")
    elif key == "perception":
        from drone_ai.modules.perception.train import run_training, run_submodels
        run_training(grade="P", trials=80, save_dir="models/perception")
        run_submodels(grade="P", trials=40, save_dir="models/perception")
    elif key == "adaptive":
        from drone_ai.modules.adaptive.train import run_training
        run_training(model_path=None, episodes=3, save_dir="models/adaptive")
    elif key == "storage":
        from drone_ai.modules.storage.train import run_training
        run_training(n_missions=20, save_dir="models/storage")
    elif key == "personality":
        from drone_ai.modules.personality.train import run_training
        run_training(save_dir="models/personality")
    elif key == "swarm":
        from drone_ai.modules.swarm.train import run_training
        run_training(trials=30, n_drones=4, save_dir="models/swarm")
    elif key == "demo":
        _run_demo()
    else:
        raise ValueError(f"unknown module key: {key}")


class Launcher:
    def __init__(self):
        self._init_pygame()
        self.hover_idx: Optional[int] = None
        self.selected_idx: int = 0
        self.total_updates = 400  # FlyControl per-stage budget
        self._recent_runs: List[Dict[str, str]] = _load_recent_runs()
        self._latest_ckpt: Dict[str, str] = _latest_flycontrol_per_stage()

    def _init_pygame(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Drone AI — Launcher")
        self.clock = pygame.time.Clock()
        self.font_xl = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_lg = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_md = pygame.font.SysFont("Consolas", 14)
        self.font_sm = pygame.font.SysFont("Consolas", 12)

    def run(self) -> None:
        running = True
        try:
            while running:
                running = self._handle_events()
                self._draw()
                pygame.display.flip()
                self.clock.tick(60)
        finally:
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
            if card.section == "flycontrol":
                cfg = TrainConfig(stage=card.key, total_updates=self.total_updates)
                TrainerUI(cfg).run()
            else:
                # Non-FlyControl cards run as headless benchmarks. They
                # print progress to stdout and finish in seconds.
                print(f"\n[launcher] running '{card.key}' benchmark…")
                t0 = time.monotonic()
                _run_module(card.key)
                print(f"[launcher] '{card.key}' done in {time.monotonic()-t0:.1f}s")
        except Exception as e:
            import traceback
            print(f"[launcher] stage '{card.key}' raised: {e}")
            traceback.print_exc()
        finally:
            self._init_pygame()
            self._recent_runs = _load_recent_runs()
            self._latest_ckpt = _latest_flycontrol_per_stage()

    # ---- Layout ---------------------------------------------------------

    GRID_COLS = 5
    CARD_W = 230
    CARD_H = 145
    CARD_GAP = 14
    GRID_X = 40
    SECTION_H_PAD = 26   # vertical padding under each section header
    HEADER_H = 24        # height of the section header text

    def _section_layout(self) -> List[tuple]:
        """Return [(section_key, y_start, [(card_index, rect), …]), …]
        in display order. Cards within a section flow GRID_COLS-wide.
        """
        # First pass: collect indices per section.
        per_section: Dict[str, List[int]] = {k: [] for k, _ in SECTIONS}
        for i, card in enumerate(STAGE_CARDS):
            per_section.setdefault(card.section, []).append(i)

        layout = []
        y = 168  # below the header / budget controls
        for key, label in SECTIONS:
            ids = per_section.get(key, [])
            if not ids:
                continue
            section_y = y
            y += self.HEADER_H + self.SECTION_H_PAD
            cards_in_section = []
            for slot, idx in enumerate(ids):
                col = slot % self.GRID_COLS
                row = slot // self.GRID_COLS
                x = self.GRID_X + col * (self.CARD_W + self.CARD_GAP)
                cy = y + row * (self.CARD_H + self.CARD_GAP)
                cards_in_section.append((idx, pygame.Rect(x, cy, self.CARD_W, self.CARD_H)))
            n_rows = (len(ids) + self.GRID_COLS - 1) // self.GRID_COLS
            y += n_rows * (self.CARD_H + self.CARD_GAP) + 8
            layout.append((key, section_y, cards_in_section))
        return layout

    def _card_at(self, mx: int, my: int) -> Optional[int]:
        for _, _, cards in self._section_layout():
            for idx, rect in cards:
                if rect.collidepoint(mx, my):
                    return idx
        return None

    def _card_rect(self, target_idx: int) -> Optional[pygame.Rect]:
        for _, _, cards in self._section_layout():
            for idx, rect in cards:
                if idx == target_idx:
                    return rect
        return None

    # ---- Draw -----------------------------------------------------------

    def _draw(self):
        self.screen.fill(BG)
        # Header
        title = self.font_xl.render("DRONE AI — 8 Layer Stack", True, TEXT_TITLE)
        self.screen.blit(title, (40, 36))
        sub = self.font_md.render(
            "Pick a card. FlyControl stages open a 3D training window; "
            "module benchmarks run headless and write to models/runs.csv.",
            True, TEXT_DIM,
        )
        self.screen.blit(sub, (40, 78))

        # Budget control (only meaningful for FlyControl stages)
        budget_y = 110
        budget = self.font_md.render(
            f"FlyControl training budget: {self.total_updates} updates",
            True, TEXT_DIM,
        )
        self.screen.blit(budget, (40, budget_y))
        hint = self.font_sm.render(
            "  [+/-] adjust budget   [Enter] launch selected   [Q] quit",
            True, TEXT_DIM,
        )
        self.screen.blit(hint, (40, budget_y + 18))

        # Section headers + cards
        layout = self._section_layout()
        section_label_lookup = dict(SECTIONS)
        for key, header_y, cards in layout:
            label = section_label_lookup.get(key, key)
            hdr = self.font_lg.render(label, True, TEXT_TITLE)
            self.screen.blit(hdr, (self.GRID_X, header_y))
            pygame.draw.line(
                self.screen, BORDER,
                (self.GRID_X, header_y + self.HEADER_H + 4),
                (WINDOW_W - self.GRID_X, header_y + self.HEADER_H + 4),
                1,
            )
            for idx, rect in cards:
                self._draw_card(idx, rect)

        # Run-log + saved-checkpoint strip across the bottom.
        self._draw_run_strip()

    def _draw_card(self, idx: int, rect: pygame.Rect):
        card = STAGE_CARDS[idx]
        is_sel = (idx == self.selected_idx)
        is_hover = (idx == self.hover_idx)
        bg = PANEL_ACTIVE if is_sel else PANEL_HOVER if is_hover else PANEL
        pygame.draw.rect(self.screen, bg, rect, border_radius=8)
        pygame.draw.rect(self.screen, card.accent, rect, 2, border_radius=8)
        # Title
        t = self.font_lg.render(card.title, True, TEXT)
        self.screen.blit(t, (rect.x + 12, rect.y + 10))
        # Accent chip
        chip_w = 36
        pygame.draw.rect(self.screen, card.accent,
                         (rect.right - chip_w - 12, rect.y + 14, chip_w, 6),
                         border_radius=3)
        # Subtitle
        s = self.font_md.render(card.subtitle, True, TEXT_ACCENT)
        self.screen.blit(s, (rect.x + 12, rect.y + 38))
        # Description (wrap)
        self._blit_wrapped(card.description, rect.x + 12, rect.y + 62,
                           rect.width - 24, self.font_sm, TEXT_DIM)
        # Layer / key badges
        if card.badge:
            badge = self.font_sm.render(card.badge, True, card.accent)
            self.screen.blit(badge, (rect.x + 12, rect.bottom - 18))
        key_b = self.font_sm.render(card.key, True, TEXT_DIM)
        self.screen.blit(key_b, (rect.right - key_b.get_width() - 12, rect.bottom - 18))

    def _draw_run_strip(self):
        # Compute strip start from the layout so it always sits below
        # the last card row, no matter how many sections / cards exist.
        layout = self._section_layout()
        bottom_of_cards = 0
        for _, _, cards in layout:
            for _, rect in cards:
                bottom_of_cards = max(bottom_of_cards, rect.bottom)
        strip_y = bottom_of_cards + 18
        strip_h = WINDOW_H - strip_y - 14
        if strip_h < 80:
            return  # no room — skip rather than overlap
        strip_rect = pygame.Rect(40, strip_y, WINDOW_W - 80, strip_h)
        pygame.draw.rect(self.screen, PANEL, strip_rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, strip_rect, 1, border_radius=8)

        col_w = strip_rect.width // 2
        # ---- Recent runs (left) ----
        lx = strip_rect.x + 14
        ly = strip_rect.y + 8
        self.screen.blit(
            self.font_md.render("Recent runs (models/runs.csv)", True, TEXT_TITLE),
            (lx, ly),
        )
        ly += 22
        if not self._recent_runs:
            self.screen.blit(
                self.font_sm.render("  no runs yet — launch a card to record one",
                                    True, TEXT_DIM),
                (lx, ly),
            )
        else:
            header = f"  {'date':<12}{'module':<13}{'stage':<16}{'grade':<6}{'best':>8}{'min':>6}"
            self.screen.blit(self.font_sm.render(header, True, TEXT_DIM), (lx, ly))
            ly += 14
            for row in self._recent_runs:
                line = (
                    f"  {row.get('date',''):<12}"
                    f"{row.get('module',''):<13}"
                    f"{row.get('stage',''):<16}"
                    f"{row.get('grade',''):<6}"
                    f"{row.get('best_score',''):>8}"
                    f"{row.get('minutes',''):>6}"
                )
                self.screen.blit(self.font_sm.render(line, True, TEXT), (lx, ly))
                ly += 14

        # ---- Curriculum chain (right) ----
        rx = strip_rect.x + col_w + 14
        ry = strip_rect.y + 8
        self.screen.blit(
            self.font_md.render("FlyControl curriculum chain (models/flycontrol/<stage>/)",
                                True, TEXT_TITLE),
            (rx, ry),
        )
        ry += 22
        warm_source: Optional[str] = None
        for stage in STAGE_ORDER:
            name = self._latest_ckpt.get(stage)
            if name:
                warm_source = stage
                label = f"  {stage:<16}{name}"
                color = TEXT
            elif warm_source is not None:
                label = f"  {stage:<16}— (warm from {warm_source})"
                color = TEXT_ACCENT
            else:
                label = f"  {stage:<16}— (fresh)"
                color = TEXT_DIM
            self.screen.blit(self.font_sm.render(label, True, color), (rx, ry))
            ry += 14

    def _blit_wrapped(self, text: str, x: int, y: int, max_w: int, font, color):
        words = text.split(" ")
        line = ""
        dy = 0
        for w in words:
            trial = (line + " " + w).strip()
            if font.size(trial)[0] > max_w:
                surf = font.render(line, True, color)
                self.screen.blit(surf, (x, y + dy))
                dy += font.get_height() + 1
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
    parser.add_argument(
        "--stage", default=None,
        help=("Skip the menu and launch this card directly. "
              "Accepts any card key — FlyControl stages "
              "(hover/waypoint/delivery/delivery_route/deployment), "
              "module benchmarks (manager/pathfinder/perception/adaptive/"
              "storage/personality/swarm), or demo."),
    )
    parser.add_argument("--updates", type=int, default=400)
    args = parser.parse_args()

    if args.stage:
        exit_code = 0
        try:
            if args.stage in STAGE_DEFS:
                TrainerUI(TrainConfig(stage=args.stage, total_updates=args.updates)).run()
            elif args.stage == "demo":
                _run_demo()
            else:
                # Non-FlyControl module benchmark.
                try:
                    _run_module(args.stage)
                except ValueError:
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
