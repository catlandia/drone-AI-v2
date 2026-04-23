"""Swarm visual inspector — watch the Layer-8 coordinator decide.

Each step: a random swarm plan is generated, a random contact is
dropped near one of the drones, and the coordinator is asked what to
do. We draw the top-down scene (drones, routes, contact, avoidance
radius) and colour-code the coordinator's returned action so the user
can see whether it matched the ground-truth expected action.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from drone_ai.grading import (
    RunLogger, RunRecord, generate_model_name, next_version,
    score_to_universal_grade,
)
from drone_ai.modules.swarm import (
    DroneAssignment, DroneRole, SwarmCoordinator, VisualContact,
    build_swarm_plan,
)
from drone_ai.modules.swarm.coordinator import AvoidanceKind
from drone_ai.modules.swarm.train import _expected_action, _random_contact, _random_plan
from drone_ai.viz.inspector_common import (
    BG, BORDER, PANEL, TEXT, TEXT_ACCENT, TEXT_DIM, TEXT_OK, TEXT_WARN, TEXT_BAD,
    InspectorBase, RunningStats, TopDownProjector, draw_grid,
)


DRONE_COLOR = (120, 220, 150)
DRONE_LC    = (240, 110, 110)  # life-critical drone
ROUTE       = (120, 150, 200)
CONTACT     = (240, 180, 90)
RADIUS      = (80, 130, 180, 70)

ACTION_COLORS = {
    AvoidanceKind.CONTINUE:      (150, 160, 175),
    AvoidanceKind.BANK_RIGHT:    (140, 200, 255),
    AvoidanceKind.CLIMB:         (120, 220, 150),
    AvoidanceKind.YIELD:         (240, 180, 90),
    AvoidanceKind.DIVERT_TO_MARK:(220, 140, 210),
}


class SwarmInspector(InspectorBase):
    def __init__(self, n_trials: int = 40, n_drones: int = 4, seed: int = 42,
                 save_dir: str = "models/swarm", run_tag: str = ""):
        super().__init__(
            title="Swarm — mutual avoidance coordinator",
            subtitle="Each step: random plan + contact. See the coordinator's decision.",
            total_trials=n_trials,
            autoplay_hz=2.0,
        )
        self.n_trials = n_trials
        self.n_drones = n_drones
        self.seed = seed
        self.save_dir = save_dir
        self.run_tag = run_tag
        self.rng = np.random.default_rng(seed)

        # Aggregates matching the terminal benchmark.
        self.total_avoid = 0
        self.correct_avoid = 0
        self.false_trigger = 0
        self.missed_trigger = 0

        # Current step state.
        self.plan = None
        self.coord: Optional[SwarmCoordinator] = None
        self.self_id: str = ""
        self.self_pos: Optional[np.ndarray] = None
        self.contact: Optional[VisualContact] = None
        self.got_action: AvoidanceKind = AvoidanceKind.CONTINUE
        self.expected: AvoidanceKind = AvoidanceKind.CONTINUE
        self.proj: Optional[TopDownProjector] = None

        self._final_grade: Optional[str] = None
        self._final_score: float = 0.0
        self._saved_path: Optional[str] = None

    def setup(self) -> None:
        self._new_step()

    def _new_step(self) -> None:
        self.plan = _random_plan(self.rng, self.n_drones)
        drone_ids = list(self.plan.drone_ids())
        self.self_id = drone_ids[0]
        self.self_pos = self.plan.assignments[self.self_id].route[0].astype(np.float32)
        self.coord = SwarmCoordinator(self.plan, self.self_id)
        inside = bool(self.rng.random() > 0.3)
        self.contact = _random_contact(self.rng, self.self_pos, force_inside_radius=inside)
        assignment = self.plan.assignments[self.self_id]
        high_pri = (assignment.mission_class == "LIFE_CRITICAL"
                    or assignment.role == DroneRole.PRIMARY)
        self.expected = _expected_action(self.self_pos, self.contact, high_pri)
        action = self.coord.step(self.self_pos, [self.contact])
        self.got_action = action.kind if action is not None else AvoidanceKind.CONTINUE

        self.total_avoid += 1
        if self.got_action == self.expected:
            self.correct_avoid += 1
        elif self.expected == AvoidanceKind.CONTINUE:
            self.false_trigger += 1
        elif self.got_action == AvoidanceKind.CONTINUE:
            self.missed_trigger += 1

        pts = [self.self_pos[:2], self.contact.position[:2]]
        for d_id, asn in self.plan.assignments.items():
            for wp in asn.route:
                pts.append(wp[:2])
        self.proj = TopDownProjector.from_points(self.view_rect, pts,
                                                 padding_world=8.0)

    def step(self) -> bool:
        self.trial_idx += 1
        if self.trial_idx >= self.n_trials:
            self._finalize()
            return False
        self._new_step()
        return True

    def render(self, surface: pygame.Surface, view_rect: pygame.Rect) -> None:
        if self.proj is None or self.plan is None or self.self_pos is None:
            return
        draw_grid(surface, self.proj, step=10.0)

        # All drones + their routes.
        for d_id, asn in self.plan.assignments.items():
            color = DRONE_LC if asn.mission_class == "LIFE_CRITICAL" else DRONE_COLOR
            for i in range(len(asn.route) - 1):
                a = self.proj.to_screen(asn.route[i][0], asn.route[i][1])
                b = self.proj.to_screen(asn.route[i + 1][0], asn.route[i + 1][1])
                pygame.draw.line(surface, ROUTE, a, b, 1)
            sx, sy = self.proj.to_screen(asn.route[0][0], asn.route[0][1])
            pygame.draw.circle(surface, color, (sx, sy), 6)
            pygame.draw.circle(surface, (0, 0, 0), (sx, sy), 6, 1)
            if d_id == self.self_id:
                pygame.draw.circle(surface, (255, 255, 255), (sx, sy), 10, 2)

        # Avoidance radius ring around self.
        sx, sy = self.proj.to_screen(self.self_pos[0], self.self_pos[1])
        r_px = self.proj.size_px(8.0)  # SwarmCoordinator.AVOID_RADIUS
        ring = pygame.Surface((r_px * 2, r_px * 2), pygame.SRCALPHA)
        pygame.draw.circle(ring, RADIUS, (r_px, r_px), r_px)
        surface.blit(ring, (sx - r_px, sy - r_px))

        # Contact.
        cx, cy = self.proj.to_screen(self.contact.position[0], self.contact.position[1])
        pygame.draw.circle(surface, CONTACT, (cx, cy), 6)
        pygame.draw.line(surface, CONTACT, (sx, sy), (cx, cy), 1)

        # Chosen action vector + label.
        action_color = ACTION_COLORS.get(self.got_action, (200, 200, 200))
        pygame.draw.circle(surface, action_color, (sx, sy), 16, 3)
        label = self.font_md.render(
            f"action: {self.got_action.name}  "
            f"expected: {self.expected.name}",
            True,
            TEXT_OK if self.got_action == self.expected else TEXT_BAD,
        )
        surface.blit(label, (view_rect.x + 14, view_rect.y + 14))

        caption = (
            "white ring = self drone   orange = contact   "
            "blue disc = avoidance radius   coloured ring = chosen action"
        )
        s = self.font_sm.render(caption, True, TEXT_DIM)
        surface.blit(s, (view_rect.x + 10, view_rect.bottom - 18))

    def sidebar_lines(self) -> List[Tuple[str, str, str]]:
        acc = 100.0 * self.correct_avoid / max(self.total_avoid, 1)
        ft = 100.0 * self.false_trigger / max(self.total_avoid, 1)
        mt = 100.0 * self.missed_trigger / max(self.total_avoid, 1)
        match = "YES" if self.got_action == self.expected else "NO"
        return [
            ("contact rng",  f"{self.contact.range_m:.1f} m" if self.contact else "—", "text"),
            ("closing",      f"{self.contact.closing_speed:.1f} m/s" if self.contact else "—", "text"),
            ("expected",     self.expected.name, "dim"),
            ("got",          self.got_action.name, "text"),
            ("match",        match, "ok" if match == "YES" else "bad"),
            ("",             "", "dim"),
            ("accuracy",     f"{acc:.1f}%", "ok" if acc > 80 else "warn"),
            ("false trig",   f"{ft:.1f}%",  "ok" if ft < 10 else "warn"),
            ("missed trig",  f"{mt:.1f}%",  "ok" if mt < 10 else "warn"),
        ]

    def _finalize(self) -> None:
        # Simplified score from this single-threat stream — same weights
        # the terminal benchmark uses for avoidance (divert is sampled
        # separately by the CLI benchmark; not modeled here).
        correct_rate = self.correct_avoid / max(self.total_avoid, 1)
        ft_rate = self.false_trigger / max(self.total_avoid, 1)
        mt_rate = self.missed_trigger / max(self.total_avoid, 1)
        raw = correct_rate * 0.75 - ft_rate * 0.15 - mt_rate * 0.15
        score = max(0.0, min(1.0, 0.25 + raw)) * 800.0
        grade = score_to_universal_grade(score)
        self._final_grade = grade
        self._final_score = score

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        version = next_version(self.save_dir, "swarm")
        fname = generate_model_name(grade, "swarm", version)
        out = Path(self.save_dir) / fname
        metrics = {
            "avoidance_correctness": correct_rate,
            "false_trigger_rate": ft_rate,
            "missed_trigger_rate": mt_rate,
            "trials": self.n_trials,
            "drones_per_trial": self.n_drones,
            "ui_mode": "single_threat_visual",
        }
        import torch
        torch.save({"grade": grade, "score": score, "metrics": metrics}, str(out))
        with open(out.with_suffix(".json"), "w") as f:
            json.dump({
                "grade": grade, "score": score, "metrics": metrics,
                "timestamp": datetime.now().isoformat(), "model_file": fname,
            }, f, indent=2)
        self._saved_path = str(out)
        try:
            RunLogger().append(RunRecord(
                module="swarm", stage="coordinator",
                best_score=score, avg_score=score, grade=grade,
                minutes=(time.monotonic() - self._start_t) / 60.0,
                episodes=self.n_trials, run_tag=self.run_tag,
            ))
        except Exception as e:
            print(f"[swarm-ui] run-log append failed: {e}")

    def final_summary(self) -> List[Tuple[str, str]]:
        if self._final_grade is None:
            return [("Run did not finish.", "warn")]
        return [
            (f"Grade: {self._final_grade}   Score: {self._final_score:.1f}", "accent"),
            (f"Correct avoidance: "
             f"{100.0 * self.correct_avoid / max(self.total_avoid, 1):.1f}%", "text"),
            (f"Saved: {self._saved_path}", "dim"),
        ]


def run_swarm_inspector(trials: int = 40, n_drones: int = 4, seed: int = 42,
                        save_dir: str = "models/swarm",
                        run_tag: str = "") -> Tuple[str, float]:
    ui = SwarmInspector(trials, n_drones, seed, save_dir, run_tag)
    ui.run()
    return ui._final_grade or "W", ui._final_score
