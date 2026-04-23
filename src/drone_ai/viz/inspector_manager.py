"""Manager visual inspector — watch the mission queue get served.

Per trial:
  1. Drop N delivery requests into the world (colored by priority).
  2. Run the planner step-by-step: each step the user (or autoplay)
     picks the next delivery and flies the drone there.
  3. Draw the current drone position, visited trail, and remaining
     queue, plus a mission-queue list in the sidebar so the user can
     see WHY the planner chose the one it did (priority + distance).

Wraps the same scoring logic the terminal benchmark uses.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pygame

from drone_ai.grading import (
    ModelGrader, ManagerMetrics, RunLogger, RunRecord,
    generate_model_name, next_version,
)
from drone_ai.modules.manager.planner import (
    DeliveryRequest, MissionPlanner, Priority,
)
from drone_ai.viz.inspector_common import (
    BG, BORDER, PANEL, TEXT, TEXT_ACCENT, TEXT_DIM, TEXT_OK, TEXT_WARN, TEXT_BAD,
    InspectorBase, RunningStats, TopDownProjector, draw_grid,
)


BASE_COLOR = (150, 220, 240)
DRONE_COLOR = (120, 220, 150)
VISITED = (90, 120, 80)
EDGE = (40, 50, 66)

PRIORITY_COLORS = {
    Priority.NORMAL:   (150, 180, 220),
    Priority.URGENT:   (240, 200, 100),
    Priority.CRITICAL: (240, 100, 110),
}
PRIORITY_LABEL = {
    Priority.NORMAL: "N", Priority.URGENT: "U", Priority.CRITICAL: "C",
}


class ManagerInspector(InspectorBase):
    """Walk through delivery-queue benchmarks with a visible queue."""

    DELIVERIES_PER_TRIAL = 6

    def __init__(
        self,
        grade: str = "P",
        n_trials: int = 20,
        seed: int = 42,
        save_dir: str = "models/manager",
        run_tag: str = "",
    ):
        super().__init__(
            title=f"Manager — mission planner @ grade {grade}",
            subtitle="Drop deliveries, watch the queue drain. Priority then distance.",
            total_trials=n_trials,
            autoplay_hz=3.0,
        )
        self.input_grade = grade
        self.n_trials = n_trials
        self.seed = seed
        self.save_dir = save_dir
        self.run_tag = run_tag

        self.rng = np.random.default_rng(seed)
        self.base = np.zeros(3)
        self.planner = MissionPlanner(base_position=self.base, grade=grade, seed=seed)

        # Per-trial state (the inspector walks trial-by-trial, not pick-by-pick,
        # so the visual can show the full mission unfold between Space-presses).
        self.targets: List[np.ndarray] = []
        self.priorities: List[Priority] = []
        self.order: List[DeliveryRequest] = []
        self.actual_dist: float = 0.0
        self.opt_dist: float = 0.0
        self.priority_ok: int = 0
        self.proj: Optional[TopDownProjector] = None

        # Running stats.
        self.stats_completion = RunningStats()
        self.stats_efficiency = RunningStats()
        self.stats_priority = RunningStats()

        # End-of-run.
        self._final_grade: Optional[str] = None
        self._final_score: float = 0.0
        self._saved_path: Optional[str] = None

    # ---- scenario -------------------------------------------------------

    def setup(self) -> None:
        self._new_scenario()

    def _new_scenario(self) -> None:
        self.planner.reset()
        self.targets = []
        self.priorities = []
        for _ in range(self.DELIVERIES_PER_TRIAL):
            t = self.rng.uniform([-100, -100, 0], [100, 100, 0]).astype(float)
            p = self.rng.choice([Priority.NORMAL, Priority.URGENT, Priority.CRITICAL])
            self.planner.add_delivery(t, p)
            self.targets.append(t)
            self.priorities.append(p)

        pos = self.base.copy()
        self.order = []
        self.actual_dist = 0.0
        for _ in range(self.DELIVERIES_PER_TRIAL + 2):
            chosen = self.planner.select_next(pos)
            if chosen is None:
                break
            self.actual_dist += float(np.linalg.norm(chosen.target - pos))
            self.order.append(chosen)
            pos = chosen.target.copy()
            self.planner.update(0.1, pos, 0.9)
            self.planner.complete_current(success=True)

        self.opt_dist = self._greedy_tsp(self.targets, self.base)
        # Priority adherence — same rule the terminal benchmark uses.
        self.priority_ok = 0
        for i, req in enumerate(self.order):
            later = [self.order[j].priority.value
                     for j in range(i + 1, len(self.order))]
            if not later or req.priority.value >= max(later, default=0):
                self.priority_ok += 1

        comp = len(self.planner.state.completed) / self.DELIVERIES_PER_TRIAL
        eff = min(self.opt_dist / max(self.actual_dist, 1.0), 1.0)
        pri = self.priority_ok / max(len(self.order), 1)
        self.stats_completion.push(comp)
        self.stats_efficiency.push(eff)
        self.stats_priority.push(pri)

        pts = [self.base[:2]] + [t[:2] for t in self.targets]
        self.proj = TopDownProjector.from_points(self.view_rect, pts,
                                                 padding_world=20.0)

    def step(self) -> bool:
        self.trial_idx += 1
        if self.trial_idx >= self.n_trials:
            self._finalize()
            return False
        self._new_scenario()
        return True

    # ---- render ---------------------------------------------------------

    def render(self, surface: pygame.Surface, view_rect: pygame.Rect) -> None:
        if self.proj is None:
            return
        draw_grid(surface, self.proj, step=25.0)

        # Base station.
        bx, by = self.proj.to_screen(self.base[0], self.base[1])
        pygame.draw.rect(surface, BASE_COLOR, (bx - 7, by - 7, 14, 14))
        lbl = self.font_sm.render("BASE", True, TEXT_DIM)
        surface.blit(lbl, (bx + 10, by - 8))

        # Delivery markers, colored by priority, labeled with visit order.
        visited_idx = {id(o): i + 1 for i, o in enumerate(self.order)}
        for idx, (t, p) in enumerate(zip(self.targets, self.priorities)):
            tx, ty = self.proj.to_screen(t[0], t[1])
            color = PRIORITY_COLORS[p]
            pygame.draw.circle(surface, color, (tx, ty), 9)
            pygame.draw.circle(surface, EDGE, (tx, ty), 9, 1)
            lab = PRIORITY_LABEL[p]
            label_surf = self.font_sm.render(lab, True, (20, 24, 32))
            surface.blit(label_surf, (tx - 4, ty - 6))

        # Route polyline base → visited.
        if self.order:
            pts = [(bx, by)] + [self.proj.to_screen(r.target[0], r.target[1])
                                for r in self.order]
            pygame.draw.lines(surface, VISITED, False, pts, 2)
            for i, (px, py) in enumerate(pts[1:], start=1):
                idx_surf = self.font_sm.render(str(i), True, TEXT)
                surface.blit(idx_surf, (px + 10, py + 6))

        caption = (
            "BASE (cyan) → pickups (N=normal, U=urgent, C=critical). "
            "Line order = planner's chosen sequence."
        )
        s = self.font_sm.render(caption, True, TEXT_DIM)
        surface.blit(s, (view_rect.x + 10, view_rect.bottom - 18))

    def sidebar_lines(self) -> List[Tuple[str, str, str]]:
        comp = self.stats_completion.last()
        eff = self.stats_efficiency.last()
        pri = self.stats_priority.last()
        return [
            ("grade",         self.input_grade, "accent"),
            ("deliveries",    f"{len(self.planner.state.completed)}/{self.DELIVERIES_PER_TRIAL}", "text"),
            ("this dist",     f"{self.actual_dist:.1f} m", "text"),
            ("optimal dist",  f"{self.opt_dist:.1f} m", "dim"),
            ("trip efficiency", f"{eff*100:5.1f}%", "ok" if eff > 0.75 else "warn"),
            ("priority adher.", f"{pri*100:5.1f}%", "ok" if pri > 0.75 else "warn"),
            ("",              "", "dim"),
            ("cum completion",f"{self.stats_completion.mean()*100:5.1f}%", "ok"),
            ("cum efficiency",f"{self.stats_efficiency.mean()*100:5.1f}%", "text"),
            ("cum priority",  f"{self.stats_priority.mean()*100:5.1f}%", "text"),
        ]

    # ---- finalize -------------------------------------------------------

    def _finalize(self) -> None:
        metrics = ManagerMetrics(
            completion_rate=self.stats_completion.mean(),
            distance_efficiency=self.stats_efficiency.mean(),
            priority_score=self.stats_priority.mean(),
            battery_waste=max(0.0, 1.0 - self.stats_efficiency.mean()),
        )
        grader = ModelGrader()
        grade, score = grader.grade_manager(metrics)
        self._final_grade = grade
        self._final_score = score

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        version = next_version(self.save_dir, "manager")
        fname = generate_model_name(grade, "manager", version)
        out = Path(self.save_dir) / fname
        import torch
        torch.save({"grade": grade, "input_grade": self.input_grade, "score": score},
                   str(out))
        with open(out.with_suffix(".json"), "w") as f:
            json.dump({
                "grade": grade, "score": score,
                "input_grade": self.input_grade,
                "metrics": {
                    "completion_rate": metrics.completion_rate,
                    "distance_efficiency": metrics.distance_efficiency,
                    "priority_score": metrics.priority_score,
                    "battery_waste": metrics.battery_waste,
                },
                "timestamp": datetime.now().isoformat(),
                "model_file": fname,
            }, f, indent=2)
        self._saved_path = str(out)
        try:
            RunLogger().append(RunRecord(
                module="manager", stage="benchmark",
                best_score=score, avg_score=score, grade=grade,
                minutes=(time.monotonic() - self._start_t) / 60.0,
                episodes=self.n_trials, run_tag=self.run_tag,
            ))
        except Exception as e:
            print(f"[manager-ui] run-log append failed: {e}")

    def final_summary(self) -> List[Tuple[str, str]]:
        if self._final_grade is None:
            return [("Run did not finish.", "warn")]
        return [
            (f"Grade: {self._final_grade}   Score: {self._final_score:.1f}", "accent"),
            (f"Completion: {self.stats_completion.mean()*100:.1f}%", "text"),
            (f"Efficiency: {self.stats_efficiency.mean()*100:.1f}%", "text"),
            (f"Priority: {self.stats_priority.mean()*100:.1f}%", "text"),
            (f"Saved: {self._saved_path}", "dim"),
        ]

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _greedy_tsp(targets: List[np.ndarray], start: np.ndarray) -> float:
        pos = start.copy()
        remaining = list(range(len(targets)))
        total = 0.0
        while remaining:
            dists = [float(np.linalg.norm(targets[i] - pos)) for i in remaining]
            best = remaining[int(np.argmin(dists))]
            total += float(np.linalg.norm(targets[best] - pos))
            pos = targets[best]
            remaining.remove(best)
        return total


def run_manager_inspector(grade: str = "P", trials: int = 20, seed: int = 42,
                          save_dir: str = "models/manager",
                          run_tag: str = "") -> Tuple[str, float]:
    ui = ManagerInspector(
        grade=grade, n_trials=trials, seed=seed,
        save_dir=save_dir, run_tag=run_tag,
    )
    ui.run()
    return ui._final_grade or "W", ui._final_score
