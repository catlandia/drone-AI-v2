"""Stage launcher — the main window you see when you start the app.

State machine:
  MENU      — pick a card.
  PICKER    — confirm the launch and pick the base checkpoint to
              warm-start from (or "fresh" / "auto").
  RUNNING   — work runs (FlyControl stages open the live 3D trainer
              window in the same process; benchmarks run in a worker
              thread with stdout streamed into a panel).
  RESULTS   — the most recent run's grade/score stays on screen until
              the user dismisses it. No auto-close.

The launcher itself does not exit between launches — every transition
returns here.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
import traceback
from collections import deque
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import pygame

from drone_ai.grading import RunLogger, parse_model_name
from drone_ai.viz.trainer_ui import (
    STAGE_DEFS, STAGE_ORDER, TrainConfig, TrainerUI,
    flycontrol_stage_dir, latest_flycontrol_checkpoint,
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
TEXT_WARN     = (240, 180, 90)


@dataclass
class StageCard:
    key: str
    title: str
    subtitle: str
    description: str
    accent: tuple
    badge: str = ""
    section: str = "main"
    # Directory paths (relative to MODELS_ROOT) the picker scans for
    # warm-start candidates. Empty list = no warm-start needed.
    base_dirs: List[str] = field(default_factory=list)
    # Free-form text shown in the picker — what the base provides.
    base_hint: str = ""


SECTIONS = [
    ("flycontrol", "FlyControl Curriculum (Layer 4)"),
    ("modules",    "Module Benchmarks (Layers 1, 2, 3, 5)"),
    ("phase2",     "Phase 2 Ops + Demo"),
]

# Cycle for the [P] population key on the menu. 1 = single-drone (the
# original behavior). Higher values run a parallel population inside the
# same TrainerUI window — see `TrainerUI.population_size`.
POPULATION_STEPS: List[int] = [1, 2, 4, 6, 8, 12]


def _next_population_step(current: int) -> int:
    if current in POPULATION_STEPS:
        i = POPULATION_STEPS.index(current)
        return POPULATION_STEPS[(i + 1) % len(POPULATION_STEPS)]
    return POPULATION_STEPS[0]


# Per-FlyControl-stage warm-start dirs: this stage's own dir (resume)
# plus every earlier stage in the chain (curriculum step-up).
def _flycontrol_base_dirs(stage: str) -> List[str]:
    if stage not in STAGE_ORDER:
        return [f"flycontrol/{stage}"]
    idx = STAGE_ORDER.index(stage)
    dirs = [f"flycontrol/{stage}"]
    for prev in reversed(STAGE_ORDER[:idx]):
        dirs.append(f"flycontrol/{prev}")
    return dirs


STAGE_CARDS: List[StageCard] = [
    # ---- FlyControl curriculum (Layer 4) -----------------------------------
    StageCard("hover", "Hover", "Learn to stay still",
              "Basic attitude + altitude hold. The drone must hover near a target point.",
              (120, 220, 150), badge="L4 · stage 1", section="flycontrol",
              base_dirs=_flycontrol_base_dirs("hover"),
              base_hint="Resume hover, or start fresh — hover has no curriculum predecessor."),
    StageCard("waypoint", "Waypoint", "Navigate to scattered targets",
              "Harder hover variant — target drifts further out each episode.",
              (140, 200, 230), badge="L4 · stage 2", section="flycontrol",
              base_dirs=_flycontrol_base_dirs("waypoint"),
              base_hint="Warm-start from a hover or waypoint checkpoint."),
    StageCard("delivery", "Delivery", "Pickup → dropzone, one package",
              "Fly to a pickup point, then to a dropzone. Single-delivery task.",
              (240, 170, 90), badge="L4 · stage 3", section="flycontrol",
              base_dirs=_flycontrol_base_dirs("delivery"),
              base_hint="Warm-start from a waypoint / delivery checkpoint."),
    StageCard("delivery_route", "Delivery Route", "Multi-stop with obstacles",
              "2-6 deliveries, up to 20 obstacles. Route optimization + avoidance.",
              (220, 140, 210), badge="L4 · stage 4", section="flycontrol",
              base_dirs=_flycontrol_base_dirs("delivery_route"),
              base_hint="Warm-start from a delivery / route checkpoint."),
    StageCard("deployment", "Deployment Ready", "Full difficulty + domain rand",
              "Max difficulty, randomized physics. Prepares policy for sim-to-real.",
              (240, 100, 110), badge="L4 · stage 5", section="flycontrol",
              base_dirs=_flycontrol_base_dirs("deployment"),
              base_hint="Warm-start from delivery_route. Final curriculum stage."),

    # ---- Module benchmarks (Layers 1, 2, 3, 5) ----------------------------
    StageCard("manager", "Manager", "Mission planning",
              "Greedy-TSP routing, priority queue, pre-flight feasibility. Layer 1 benchmark.",
              (200, 180, 100), badge="L1", section="modules",
              base_dirs=["manager"],
              base_hint="Manager benchmarks are stateless — pick any prior grade or fresh."),
    StageCard("pathfinder", "Pathfinder", "A* + RRT planning",
              "Path optimality + collision-avoidance benchmark across random worlds.",
              (160, 200, 100), badge="L2", section="modules",
              base_dirs=["pathfinder"],
              base_hint="Pathfinder is rules-based; the base only sets the version slot."),
    StageCard("perception", "Perception", "Obstacle detection",
              "Detection rate, false positives, position error across scenes. Sub-models split.",
              (100, 200, 200), badge="L3", section="modules",
              base_dirs=["perception"],
              base_hint="Perception benchmark; sub-models grade independently."),
    StageCard("adaptive", "Adaptive", "Online fine-tune",
              "Warden-guarded online learning vs frozen baseline on a perturbed env. Layer 5.",
              (220, 110, 220), badge="L5", section="modules",
              base_dirs=[f"flycontrol/{s}" for s in STAGE_ORDER],
              base_hint="Pick a FlyControl checkpoint to adapt. Auto = newest across all stages."),

    # ---- Phase 2 ops (Layers 6, 7, 8) + demo -----------------------------
    StageCard("storage", "Storage", "Field-log round-trip",
              "Layer 6: write synthetic field rows, read them back, verify the on-drone log.",
              (180, 130, 100), badge="L6", section="phase2",
              base_dirs=["storage"],
              base_hint="Storage has no model — the picker is informational only."),
    StageCard("personality", "Personality", "Export delta artifact",
              "Layer 7: build a transferable personality from the latest FlyControl checkpoint.",
              (130, 180, 220), badge="L7", section="phase2",
              base_dirs=[f"flycontrol/{s}" for s in STAGE_ORDER],
              base_hint="Pick the FlyControl baseline the personality is computed against."),
    StageCard("swarm", "Swarm", "Multi-drone coordinator",
              "Layer 8: visual mutual avoidance + swarm-mate-failed contingency benchmark.",
              (220, 200, 120), badge="L8", section="phase2",
              base_dirs=["swarm"],
              base_hint="Swarm is rules-based; the picker only confirms launch."),
    StageCard("demo", "Free-Fly Demo", "Run the full stack live",
              "No training — just watch the drone fly its demo mission in 3D.",
              (180, 220, 250), badge="demo", section="phase2",
              base_dirs=[f"flycontrol/{s}" for s in STAGE_ORDER],
              base_hint="Optional: pick a FlyControl model to fly. Fresh uses the PD controller."),
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
    latest: Dict[str, str] = {}
    for stage in STAGE_ORDER:
        path = latest_flycontrol_checkpoint(MODELS_ROOT, stage)
        if path:
            latest[stage] = os.path.basename(path)
    return latest


def _scan_checkpoints(base_dirs: List[str]) -> List[Tuple[str, str]]:
    """Return [(label, absolute_path), …] of valid .pt files under
    each base dir. Newest first — the picker uses this order so the
    user can confirm the auto-selection by pressing Enter."""
    found: List[Tuple[float, str, str]] = []
    for rel in base_dirs:
        full = os.path.join(MODELS_ROOT, rel)
        if not os.path.isdir(full):
            continue
        for fname in os.listdir(full):
            parsed = parse_model_name(fname)
            if not parsed:
                continue
            path = os.path.join(full, fname)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            label = f"{rel}/{fname}"
            found.append((mtime, label, path))
    found.sort(reverse=True)
    return [(lbl, p) for _, lbl, p in found]


# ---- Per-card runner ------------------------------------------------------

def _run_card(
    card: StageCard,
    base_path: Optional[str],
    total_updates: int,
    population: int = 1,
) -> None:
    """Execute the work for a card.

    Every card now opens its own full-screen inspector so the user can
    actually SEE what the module is doing — nothing is terminal-only.
    Each inspector:
      - Owns its own pygame display (created after the launcher tears
        down its window — see `_launch_selected`).
      - Runs the same underlying benchmark/training logic as before,
        so the same .pt checkpoints + runs.csv rows still get
        produced.
      - Shows its own final-results modal before returning. The
        launcher then re-creates its window and returns to MENU.

    The non-UI command-line entry points (`modules/*/train.py::main`)
    still exist for batch/CI use, but interactive launches always go
    through the inspector path now.
    """
    is_test = base_path is None
    tag = "test" if is_test else ""

    if card.section == "flycontrol":
        cfg = TrainConfig(
            stage=card.key,
            total_updates=total_updates,
            warm_start_path=base_path,
            hold_on_finish=True,
            test_run=is_test,
            run_tag=tag,
            population=max(1, int(population)),
        )
        TrainerUI(cfg).run()
        return

    if card.key == "manager":
        from drone_ai.viz.inspector_manager import run_manager_inspector
        run_manager_inspector(grade="P", trials=20, run_tag=tag)
    elif card.key == "pathfinder":
        from drone_ai.viz.inspector_pathfinder import run_pathfinder_inspector
        run_pathfinder_inspector(trials=30, run_tag=tag)
    elif card.key == "perception":
        from drone_ai.viz.inspector_perception import run_perception_inspector
        run_perception_inspector(grade="P", trials=30, run_tag=tag)
    elif card.key == "adaptive":
        from drone_ai.viz.inspector_adaptive import run_adaptive_inspector
        run_adaptive_inspector(episodes=6, model_path=base_path, run_tag=tag)
    elif card.key == "storage":
        from drone_ai.viz.inspector_storage import run_storage_inspector
        run_storage_inspector(n_missions=60, run_tag=tag)
    elif card.key == "personality":
        from drone_ai.viz.inspector_personality import run_personality_inspector
        run_personality_inspector(n_siblings=5, run_tag=tag)
    elif card.key == "swarm":
        from drone_ai.viz.inspector_swarm import run_swarm_inspector
        run_swarm_inspector(trials=40, n_drones=4, run_tag=tag)
    elif card.key == "demo":
        _run_demo(base_path)
    else:
        raise ValueError(f"unknown card key: {card.key}")


# ---- Worker for non-FlyControl benchmarks ---------------------------------

class BenchmarkWorker:
    """Runs the work in a daemon thread. Stdout/stderr routed into a
    bounded deque so the launcher's running-view can tail the last N
    lines without blocking on IO."""

    def __init__(self, card: StageCard, base_path: Optional[str], total_updates: int):
        self.card = card
        self.base_path = base_path
        self.total_updates = total_updates
        self.lines: Deque[str] = deque(maxlen=200)
        self._buf = io.StringIO()
        self._done = threading.Event()
        self._error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        class _Tee(io.TextIOBase):
            def __init__(self, sink: Deque[str]):
                super().__init__()
                self._sink = sink
                self._line = ""
            def write(self, s):
                if not s:
                    return 0
                self._line += s
                while "\n" in self._line:
                    nl, self._line = self._line.split("\n", 1)
                    if nl.strip():
                        self._sink.append(nl)
                return len(s)
            def flush(self):
                if self._line.strip():
                    self._sink.append(self._line)
                    self._line = ""

        tee = _Tee(self.lines)
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                _run_card(self.card, self.base_path, self.total_updates)
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            self.lines.append(self._error)
            for tb in traceback.format_exc().splitlines():
                self.lines.append(tb)
        finally:
            try:
                tee.flush()
            except Exception:
                pass
            self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def error(self) -> Optional[str]:
        return self._error


# ---- Launcher state machine -----------------------------------------------

class LauncherState:
    MENU = "menu"
    PICKER = "picker"
    RUNNING = "running"
    RESULTS = "results"


class Launcher:
    def __init__(self):
        self._init_pygame()
        self.state = LauncherState.MENU
        self.hover_idx: Optional[int] = None
        self.selected_idx: int = 0
        self.total_updates = 400
        # Population size for FlyControl training. 1 = single-drone (the
        # original behavior). N > 1 trains N drones in parallel inside
        # the same window, picks the best at the end. The cycle steps
        # land on values that look natural: 1, 2, 4, 6, 8, 12.
        self.population: int = 1
        self._recent_runs: List[Dict[str, str]] = _load_recent_runs()
        self._latest_ckpt: Dict[str, str] = _latest_flycontrol_per_stage()

        # Picker state
        self._picker_card: Optional[StageCard] = None
        self._picker_options: List[Tuple[str, Optional[str]]] = []
        self._picker_idx: int = 0

        # Runner / results state
        self._worker: Optional[BenchmarkWorker] = None
        self._run_t0: float = 0.0
        self._results_card: Optional[StageCard] = None
        self._results_lines: List[Tuple[str, str]] = []

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
                # State updates between event handling and draw.
                if self.state == LauncherState.RUNNING:
                    self._tick_runner()
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
        if self.state == LauncherState.MENU:
            return self._handle_menu_events()
        elif self.state == LauncherState.PICKER:
            return self._handle_picker_events()
        elif self.state == LauncherState.RUNNING:
            return self._handle_running_events()
        elif self.state == LauncherState.RESULTS:
            return self._handle_results_events()
        return True

    def _handle_menu_events(self) -> bool:
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
                    self._open_picker(STAGE_CARDS[self.selected_idx])
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.total_updates = min(10000, self.total_updates + 100)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.total_updates = max(50, self.total_updates - 100)
                elif ev.key == pygame.K_p:
                    self.population = _next_population_step(self.population)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if self.hover_idx is not None:
                    self.selected_idx = self.hover_idx
                    self._open_picker(STAGE_CARDS[self.hover_idx])
        return True

    def _handle_picker_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE,):
                    self.state = LauncherState.MENU
                elif ev.key == pygame.K_DOWN:
                    self._picker_idx = (self._picker_idx + 1) % len(self._picker_options)
                elif ev.key == pygame.K_UP:
                    self._picker_idx = (self._picker_idx - 1) % len(self._picker_options)
                elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._launch_selected()
        return True

    def _handle_running_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                # Esc during a benchmark is informational only — the
                # worker thread is daemonized but we shouldn't bail
                # out of the launcher mid-run. Show a hint instead.
                pass
        return True

    def _handle_results_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                self._refresh_strip()
                self.state = LauncherState.MENU
        return True

    # ---- Picker ---------------------------------------------------------

    def _open_picker(self, card: StageCard) -> None:
        self._picker_card = card
        # Picker options: explicit fresh + auto-newest + every found ckpt.
        opts: List[Tuple[str, Optional[str]]] = []
        opts.append(("(fresh — start without a base model)", None))
        scanned = _scan_checkpoints(card.base_dirs)
        if scanned:
            # Auto = the same as the first scanned (newest), but labeled
            # so the user understands the difference.
            opts.append((f"(auto — newest: {os.path.basename(scanned[0][1])})", scanned[0][1]))
            for label, path in scanned:
                opts.append((label, path))
        self._picker_options = opts
        # Default to "auto" if a checkpoint exists, else "fresh".
        self._picker_idx = 1 if len(opts) > 1 else 0
        self.state = LauncherState.PICKER

    def _launch_selected(self) -> None:
        card = self._picker_card
        if card is None:
            return
        _, path = self._picker_options[self._picker_idx]

        # Every card opens its own inspector window (the launcher used
        # to stream terminal output for non-FlyControl cards; that's
        # gone now because the user asked for a proper UI on every
        # module). Tear down the launcher display, run the card, then
        # re-init and come back to MENU. Each inspector shows its own
        # results modal so we don't need the launcher's RESULTS state.
        pygame.display.quit()
        try:
            _run_card(card, path, self.total_updates, self.population)
        except Exception as e:
            print(f"[launcher] card '{card.key}' raised: {e}")
            traceback.print_exc()
        finally:
            self._init_pygame()
            self._refresh_strip()
            self.state = LauncherState.MENU

    # ---- Runner tick ----------------------------------------------------

    def _tick_runner(self) -> None:
        if self._worker is None:
            return
        if self._worker.done:
            self._build_results()
            self.state = LauncherState.RESULTS
            self._worker = None

    def _build_results(self) -> None:
        elapsed = time.monotonic() - self._run_t0
        card = self._results_card
        rows = _load_recent_runs(limit=12)
        is_test = self._worker is not None and self._worker.base_path is None
        # Find the most recent row that matches this card's module.
        latest = None
        if card is not None:
            for row in rows:
                module = row.get("module", "")
                if module == card.key or (card.key in ("perception",) and module == "perception"):
                    latest = row
                    break
        lines: List[Tuple[str, str]] = []
        title = card.title if card is not None else "Run"
        lines.append((f"TEST TRAINING — {title}" if is_test else title, "title"))
        if card is not None:
            lines.append((card.subtitle, "dim"))
        if is_test:
            lines.append(
                ("no base model — results are a training-machinery check", "warn")
            )
        lines.append(("", "dim"))
        if self._worker is not None and self._worker.error:
            lines.append((f"FAILED: {self._worker.error}", "warn"))
        elif latest is not None:
            grade = latest.get("grade", "—")
            best = latest.get("best_score", "—")
            avg = latest.get("avg_score", "—")
            std = latest.get("std_score", "—")
            overall = latest.get("overall_score", "—")
            stage = latest.get("stage", "—")
            tag = latest.get("run_tag", "")
            lines.append((f"Grade: {grade}", "accent"))
            lines.append((f"Overall:    {overall}   (drives the grade)", "text"))
            lines.append((f"Avg score:  {avg}", "text"))
            lines.append((f"Std score:  {std}", "text"))
            lines.append((f"Best score: {best}   (tiebreaker only)", "dim"))
            lines.append((f"Stage:      {stage}", "text"))
            if tag:
                lines.append((f"Tag:        {tag}", "dim"))
        else:
            lines.append(("(no runs.csv row found for this card)", "warn"))
        lines.append((f"Elapsed:    {elapsed:.1f}s", "text"))
        lines.append(("", "dim"))
        lines.append(("Press any key / click to return to the launcher.", "accent"))
        self._results_lines = lines

    def _refresh_strip(self) -> None:
        self._recent_runs = _load_recent_runs()
        self._latest_ckpt = _latest_flycontrol_per_stage()

    # ---- Layout ---------------------------------------------------------

    GRID_COLS = 5
    CARD_W = 230
    CARD_H = 145
    CARD_GAP = 14
    GRID_X = 40
    SECTION_H_PAD = 26
    HEADER_H = 24

    def _section_layout(self) -> List[Tuple[str, int, List[Tuple[int, pygame.Rect]]]]:
        per_section: Dict[str, List[int]] = {k: [] for k, _ in SECTIONS}
        for i, card in enumerate(STAGE_CARDS):
            per_section.setdefault(card.section, []).append(i)
        layout: List[Tuple[str, int, List[Tuple[int, pygame.Rect]]]] = []
        y = 168
        for key, _ in SECTIONS:
            ids = per_section.get(key, [])
            if not ids:
                continue
            section_y = y
            y += self.HEADER_H + self.SECTION_H_PAD
            cards_in_section: List[Tuple[int, pygame.Rect]] = []
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

    # ---- Draw -----------------------------------------------------------

    def _draw(self):
        self.screen.fill(BG)
        self._draw_menu()
        if self.state == LauncherState.PICKER:
            self._draw_picker_modal()
        elif self.state == LauncherState.RUNNING:
            self._draw_running_modal()
        elif self.state == LauncherState.RESULTS:
            self._draw_results_modal()

    def _draw_menu(self):
        title = self.font_xl.render("DRONE AI — 8 Layer Stack", True, TEXT_TITLE)
        self.screen.blit(title, (40, 36))
        sub = self.font_md.render(
            "Pick a card → confirm the base model → run. Nothing auto-exits; the results screen waits for you.",
            True, TEXT_DIM,
        )
        self.screen.blit(sub, (40, 78))

        budget_y = 110
        pop_suffix = (
            f"   ·   population: {self.population} drones"
            if self.population > 1 else ""
        )
        budget = self.font_md.render(
            f"FlyControl training budget: {self.total_updates} updates{pop_suffix}",
            True, TEXT_DIM,
        )
        self.screen.blit(budget, (40, budget_y))
        hint = self.font_sm.render(
            "  [+/-] adjust budget   [P] cycle population   "
            "[Enter] open picker   [Esc/Q] quit",
            True, TEXT_DIM,
        )
        self.screen.blit(hint, (40, budget_y + 18))

        section_label_lookup = dict(SECTIONS)
        for key, header_y, cards in self._section_layout():
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
        self._draw_run_strip()

    def _draw_card(self, idx: int, rect: pygame.Rect):
        card = STAGE_CARDS[idx]
        is_sel = (idx == self.selected_idx)
        is_hover = (idx == self.hover_idx)
        bg = PANEL_ACTIVE if is_sel else PANEL_HOVER if is_hover else PANEL
        pygame.draw.rect(self.screen, bg, rect, border_radius=8)
        pygame.draw.rect(self.screen, card.accent, rect, 2, border_radius=8)
        t = self.font_lg.render(card.title, True, TEXT)
        self.screen.blit(t, (rect.x + 12, rect.y + 10))
        chip_w = 36
        pygame.draw.rect(self.screen, card.accent,
                         (rect.right - chip_w - 12, rect.y + 14, chip_w, 6),
                         border_radius=3)
        s = self.font_md.render(card.subtitle, True, TEXT_ACCENT)
        self.screen.blit(s, (rect.x + 12, rect.y + 38))
        self._blit_wrapped(card.description, rect.x + 12, rect.y + 62,
                           rect.width - 24, self.font_sm, TEXT_DIM)
        if card.badge:
            badge = self.font_sm.render(card.badge, True, card.accent)
            self.screen.blit(badge, (rect.x + 12, rect.bottom - 18))
        key_b = self.font_sm.render(card.key, True, TEXT_DIM)
        self.screen.blit(key_b, (rect.right - key_b.get_width() - 12, rect.bottom - 18))

    def _draw_run_strip(self):
        layout = self._section_layout()
        bottom_of_cards = 0
        for _, _, cards in layout:
            for _, rect in cards:
                bottom_of_cards = max(bottom_of_cards, rect.bottom)
        strip_y = bottom_of_cards + 18
        strip_h = WINDOW_H - strip_y - 14
        if strip_h < 80:
            return
        strip_rect = pygame.Rect(40, strip_y, WINDOW_W - 80, strip_h)
        pygame.draw.rect(self.screen, PANEL, strip_rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER, strip_rect, 1, border_radius=8)

        col_w = strip_rect.width // 2
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
            header = f"  {'date':<12}{'module':<13}{'stage':<16}{'grade':<6}{'avg':>8}{'std':>7}{'best':>8}{'min':>6}"
            self.screen.blit(self.font_sm.render(header, True, TEXT_DIM), (lx, ly))
            ly += 14
            for row in self._recent_runs:
                line = (
                    f"  {row.get('date',''):<12}"
                    f"{row.get('module',''):<13}"
                    f"{row.get('stage',''):<16}"
                    f"{row.get('grade',''):<6}"
                    f"{row.get('avg_score',''):>8}"
                    f"{row.get('std_score',''):>7}"
                    f"{row.get('best_score',''):>8}"
                    f"{row.get('minutes',''):>6}"
                )
                self.screen.blit(self.font_sm.render(line, True, TEXT), (lx, ly))
                ly += 14

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

    # ---- Modals ---------------------------------------------------------

    def _modal_rect(self, w: int = 760, h: int = 540) -> pygame.Rect:
        return pygame.Rect(
            (WINDOW_W - w) // 2,
            (WINDOW_H - h) // 2,
            w, h,
        )

    def _draw_modal_bg(self, rect: pygame.Rect, accent=BORDER) -> None:
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((10, 14, 22, 180))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, (24, 30, 42), rect, border_radius=10)
        pygame.draw.rect(self.screen, accent, rect, 2, border_radius=10)

    def _draw_picker_modal(self):
        card = self._picker_card
        if card is None:
            return
        rect = self._modal_rect(820, 580)
        self._draw_modal_bg(rect, accent=card.accent)
        # Current selection — if "(fresh)" is highlighted we call the
        # run a TEST TRAINING up front so the user knows what they'll
        # get before hitting Enter.
        _, current_path = (
            self._picker_options[self._picker_idx]
            if self._picker_options else (None, None)
        )
        is_test = current_path is None
        title_text = (
            f"Test Training — {card.title}" if is_test else f"Launch — {card.title}"
        )
        t = self.font_xl.render(title_text, True, TEXT_WARN if is_test else TEXT_TITLE)
        self.screen.blit(t, (rect.x + 24, rect.y + 18))
        sub = self.font_md.render(card.subtitle, True, TEXT_ACCENT)
        self.screen.blit(sub, (rect.x + 24, rect.y + 58))
        if is_test:
            note = self.font_sm.render(
                "No base model selected — run exercises the training "
                "machinery and is tagged 'test' in runs.csv.",
                True, TEXT_WARN,
            )
            self.screen.blit(note, (rect.x + 24, rect.y + 80))
        elif card.base_hint:
            h = self.font_sm.render(card.base_hint, True, TEXT_DIM)
            self.screen.blit(h, (rect.x + 24, rect.y + 80))

        # Options list
        opts_y = rect.y + 120
        self.screen.blit(
            self.font_md.render("Pick a base model:", True, TEXT_TITLE),
            (rect.x + 24, opts_y - 24),
        )
        list_h = rect.height - 200
        max_visible = max(1, list_h // 22)
        first = max(0, min(len(self._picker_options) - max_visible,
                           self._picker_idx - max_visible // 2))
        for i in range(first, min(first + max_visible, len(self._picker_options))):
            label, _ = self._picker_options[i]
            is_sel = (i == self._picker_idx)
            row_y = opts_y + (i - first) * 22
            row_rect = pygame.Rect(rect.x + 16, row_y, rect.width - 32, 20)
            if is_sel:
                pygame.draw.rect(self.screen, (45, 75, 120), row_rect, border_radius=4)
            text_color = TEXT if is_sel else TEXT_DIM
            # Truncate long labels.
            display = label
            max_chars = (rect.width - 60) // 8
            if len(display) > max_chars:
                display = "…" + display[-max_chars + 1:]
            self.screen.blit(self.font_sm.render(display, True, text_color),
                             (rect.x + 24, row_y + 2))

        # Footer
        footer_y = rect.bottom - 36
        if card.section == "flycontrol":
            pop_suffix = (
                f"  ·  pop {self.population}" if self.population > 1 else ""
            )
            tip = (
                f"  Budget: {self.total_updates} updates{pop_suffix}  "
                "[↑/↓] pick  [Enter] launch  [Esc] back"
            )
        else:
            tip = "  [↑/↓] pick  [Enter] launch  [Esc] back"
        self.screen.blit(self.font_sm.render(tip, True, TEXT_DIM), (rect.x + 24, footer_y))

    def _draw_running_modal(self):
        card = self._results_card
        if card is None:
            return
        rect = self._modal_rect(820, 580)
        self._draw_modal_bg(rect, accent=card.accent)
        is_test = self._worker is not None and self._worker.base_path is None
        title_text = (
            f"Running Test — {card.title}" if is_test else f"Running — {card.title}"
        )
        t = self.font_xl.render(title_text, True, TEXT_WARN if is_test else TEXT_TITLE)
        self.screen.blit(t, (rect.x + 24, rect.y + 18))
        elapsed = time.monotonic() - self._run_t0
        spin = ".." + "." * (int(elapsed * 4) % 4)
        sub = self.font_md.render(
            f"{card.subtitle}  ·  {elapsed:.1f}s {spin}", True, TEXT_ACCENT,
        )
        self.screen.blit(sub, (rect.x + 24, rect.y + 58))

        log_y = rect.y + 96
        self.screen.blit(
            self.font_md.render("Live output", True, TEXT_TITLE),
            (rect.x + 24, log_y - 22),
        )
        log_rect = pygame.Rect(rect.x + 16, log_y, rect.width - 32, rect.height - 140)
        pygame.draw.rect(self.screen, (12, 16, 24), log_rect, border_radius=6)
        pygame.draw.rect(self.screen, BORDER, log_rect, 1, border_radius=6)

        if self._worker is not None:
            lines = list(self._worker.lines)
            line_h = 14
            max_visible = (log_rect.height - 12) // line_h
            visible = lines[-max_visible:] if len(lines) > max_visible else lines
            ly = log_rect.y + 6
            for line in visible:
                # Truncate long lines so they don't overflow the panel.
                disp = line if len(line) <= 110 else line[:107] + "…"
                self.screen.blit(self.font_sm.render(disp, True, TEXT), (log_rect.x + 8, ly))
                ly += line_h

        footer = self.font_sm.render(
            "  Worker thread is running — the launcher waits here until it finishes.",
            True, TEXT_DIM,
        )
        self.screen.blit(footer, (rect.x + 24, rect.bottom - 28))

    def _draw_results_modal(self):
        rect = self._modal_rect(680, 420)
        accent = self._results_card.accent if self._results_card else BORDER
        self._draw_modal_bg(rect, accent=accent)
        styles = {
            "title":  (self.font_xl, TEXT_TITLE),
            "accent": (self.font_lg, TEXT_ACCENT),
            "warn":   (self.font_lg, TEXT_WARN),
            "text":   (self.font_md, TEXT),
            "dim":    (self.font_sm, TEXT_DIM),
        }
        y = rect.y + 24
        for text, style in self._results_lines:
            font, color = styles.get(style, styles["text"])
            surf = font.render(text, True, color)
            self.screen.blit(surf, (rect.x + (rect.width - surf.get_width()) // 2, y))
            y += font.get_height() + 6

    # ---- Helpers --------------------------------------------------------

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

def _run_demo(model_path: Optional[str] = None):
    """Run the built-in DroneAI (PD controller + full 4-layer stack) with 3D viz.

    Auto-generates a fresh mission whenever the current one ends.
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
        d = DroneAI(seed=seed, flycontrol_model=model_path)
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
              "FlyControl stages, module benchmarks, or 'demo'."),
    )
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument(
        "--warm-start", default=None,
        help="Optional explicit warm-start checkpoint path for FlyControl stages.",
    )
    parser.add_argument(
        "--population", type=int, default=1,
        help=("Number of drones to train in parallel inside the FlyControl "
              "viz (1 = single-drone, default). Picks the best of population "
              "at the end and saves that one. Ignored for non-FlyControl cards."),
    )
    args = parser.parse_args()

    if args.stage:
        exit_code = 0
        try:
            if args.stage in STAGE_DEFS:
                cfg = TrainConfig(
                    stage=args.stage,
                    total_updates=args.updates,
                    warm_start_path=args.warm_start,
                    hold_on_finish=True,
                    population=max(1, int(args.population)),
                )
                TrainerUI(cfg).run()
            else:
                # Look up the card (or fall back to demo) and run inline.
                card = next((c for c in STAGE_CARDS if c.key == args.stage), None)
                if card is None:
                    print(f"Unknown stage: {args.stage}", file=sys.stderr)
                    exit_code = 2
                else:
                    _run_card(card, args.warm_start, args.updates, args.population)
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
