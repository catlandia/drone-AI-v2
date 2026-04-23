"""Personality visual inspector — see the delta transfer.

Per-step pipeline:
  1. Find baseline (newest FlyControl ckpt or fresh init).
  2. Mutate baseline → "proven" drone.
  3. Export personality delta.
  4. For each noisy sibling: clone baseline + noise, apply delta,
     measure residual against proven.
  5. Aggregate + grade.

The right pane shows residual-per-sibling as a bar chart so the user
can see the transfer quality drop (or not) as sibling noise grows.
"""

from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pygame
import torch

from drone_ai.grading import (
    RunLogger, RunRecord, generate_model_name, next_version,
    parse_model_name, score_to_universal_grade,
)
from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig
from drone_ai.modules.flycontrol.environment import ACT_DIM, OBS_DIM
from drone_ai.modules.personality import (
    Personality, apply_personality, export_personality,
)
from drone_ai.modules.personality.train import (
    _mutate, _newest_flycontrol_checkpoint, _recovery_residual,
)
from drone_ai.viz.inspector_common import (
    TEXT, TEXT_ACCENT, TEXT_DIM, TEXT_OK, TEXT_WARN, TEXT_BAD,
)
from drone_ai.viz.inspector_structure import Arrow, Box, StructureInspector


PIPELINE = [
    "pick baseline",
    "mutate proven",
    "export delta",
    "apply to siblings",
    "score residuals",
]


class PersonalityInspector(StructureInspector):
    def __init__(self, n_siblings: int = 5, seed: int = 42,
                 save_dir: str = "models/personality", run_tag: str = ""):
        super().__init__(
            title="Personality — delta transfer across siblings",
            subtitle="Export delta from proven drone, apply to noisy siblings, measure recovery.",
            total_trials=4 + n_siblings,
            autoplay_hz=1.5,
        )
        self.n_siblings = n_siblings
        self.seed = seed
        self.save_dir = save_dir
        self.run_tag = run_tag
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.active_stage = 0
        self.baseline: Optional[PPOAgent] = None
        self.baseline_name: str = ""
        self.proven: Optional[PPOAgent] = None
        self.personality: Optional[Personality] = None
        self.residuals: List[float] = []
        self.trivial_residual: float = 0.0

        self._final_grade: Optional[str] = None
        self._final_score: float = 0.0
        self._saved_path: Optional[str] = None
        self._artifact_path: Optional[str] = None

    def setup(self) -> None:
        self.push_event("ready — press space to step through pipeline", "dim")

    def structure_diagram(self) -> Tuple[List[Box], List[Arrow]]:
        # Pipeline index by stage count.
        if self.active_stage <= 0:
            idx = 0
        elif self.active_stage == 1:
            idx = 1
        elif self.active_stage == 2:
            idx = 2
        elif self.active_stage < 3 + self.n_siblings:
            idx = 3
        else:
            idx = 4
        n = len(PIPELINE)
        boxes = [
            Box(label, x=i / max(1, n - 1), y=0.5,
                highlight=(i == idx) and not self.finished)
            for i, label in enumerate(PIPELINE)
        ]
        arrows = [Arrow(i, i + 1) for i in range(n - 1)]
        return boxes, arrows

    def current_thinking(self) -> List[Tuple[str, str]]:
        mean_res = float(np.mean(self.residuals)) if self.residuals else 0.0
        return [
            ("baseline",       self.baseline_name or "—"),
            ("siblings done",  f"{len(self.residuals)}/{self.n_siblings}"),
            ("trivial resid.", f"{self.trivial_residual:.3e}"),
            ("mean sibling",   f"{mean_res:.3f}"),
            ("tensors in delta",
             str(len(self.personality.weight_deltas)) if self.personality else "—"),
        ]

    def sidebar_lines(self) -> List[Tuple[str, str, str]]:
        mean_res = float(np.mean(self.residuals)) if self.residuals else 0.0
        max_res = float(np.max(self.residuals)) if self.residuals else 0.0
        effective = 0.7 * mean_res + 0.3 * max_res
        est_score = max(0.0, min(1.0, 1.0 - effective)) * 800.0
        return [
            ("siblings planned", str(self.n_siblings), "text"),
            ("siblings done",    str(len(self.residuals)), "text"),
            ("mean residual",    f"{mean_res:.3f}", "ok" if mean_res < 0.5 else "warn"),
            ("max residual",     f"{max_res:.3f}", "dim"),
            ("",                 "", "dim"),
            ("est. score",       f"{est_score:.1f}", "accent"),
        ]

    # Override to add a per-sibling residual bar chart beneath the diagram.
    def render(self, surface: pygame.Surface, view_rect: pygame.Rect) -> None:
        top = pygame.Rect(view_rect.x, view_rect.y,
                          view_rect.width, int(view_rect.height * 0.55))
        bot = pygame.Rect(view_rect.x, top.bottom + 4,
                          view_rect.width, view_rect.height - top.height - 4)
        super().render(surface, top)

        pygame.draw.rect(surface, (28, 34, 48), bot, border_radius=6)
        pygame.draw.rect(surface, (60, 72, 95), bot, 1, border_radius=6)
        title = self.font_md.render(
            "Per-sibling residual (lower = better transfer)", True, TEXT_ACCENT,
        )
        surface.blit(title, (bot.x + 8, bot.y + 6))
        chart = pygame.Rect(bot.x + 8, bot.y + 28, bot.width - 16, bot.height - 36)
        pygame.draw.rect(surface, (40, 50, 66), chart, 1)
        if self.residuals:
            amp = max(max(self.residuals), 1.0)
            bar_w = max(8, chart.width // max(self.n_siblings, 1) - 4)
            for i, v in enumerate(self.residuals):
                h = int((v / amp) * (chart.height - 12))
                x = chart.x + 4 + i * (bar_w + 4)
                color = (120, 220, 150) if v < 0.5 else (240, 180, 90) if v < 1.0 \
                    else (240, 110, 110)
                pygame.draw.rect(surface, color, (x, chart.bottom - h - 4, bar_w, h))
                lbl = self.font_sm.render(f"s{i+1}", True, TEXT_DIM)
                surface.blit(lbl, (x, chart.bottom - 2))
        else:
            msg = self.font_sm.render("(no siblings evaluated yet)", True, TEXT_DIM)
            surface.blit(msg, (chart.x + 6, chart.y + 8))

    def step(self) -> bool:
        try:
            if self.active_stage == 0:
                p = _newest_flycontrol_checkpoint()
                if p is None:
                    self.baseline = PPOAgent(OBS_DIM, ACT_DIM, PPOConfig())
                    self.baseline_name = "fresh_init"
                    self.push_event("no FC ckpt — using fresh init", "warn")
                else:
                    self.baseline = PPOAgent.from_file(p)
                    self.baseline_name = os.path.basename(p)
                    self.push_event(f"baseline: {p}", "ok")
            elif self.active_stage == 1:
                self.proven = _mutate(self.baseline, 0.04, self.rng)
                self.push_event("proven drone generated (baseline + 0.04 noise)", "text")
            elif self.active_stage == 2:
                self.personality = export_personality(
                    self.proven, self.baseline,
                    source_drone_id="bench",
                    baseline_name=self.baseline_name,
                    confidence=0.6,
                )
                self.push_event(
                    f"personality exported — {len(self.personality.weight_deltas)} tensors",
                    "ok",
                )
                # Trivial sanity residual (applies to baseline itself).
                trivial = self.baseline.clone()
                apply_personality(trivial, self.personality)
                self.trivial_residual = _recovery_residual(
                    self.proven, trivial, self.baseline,
                )
                self.push_event(f"trivial residual = {self.trivial_residual:.3e}", "dim")
            elif self.active_stage < 3 + self.n_siblings:
                sibling = _mutate(self.baseline, 0.02, self.rng)
                target = sibling.clone()
                apply_personality(target, self.personality)
                res = _recovery_residual(self.proven, target, self.baseline)
                self.residuals.append(res)
                self.push_event(
                    f"sibling {len(self.residuals)}/{self.n_siblings}  residual={res:.3f}",
                    "ok" if res < 0.5 else "warn",
                )
            else:
                self._finalize()
                self.active_stage += 1
                return False
        except Exception as e:
            self.push_event(f"step {self.active_stage} crashed: {e}", "bad")
            self._finalize()
            return False
        self.active_stage += 1
        self.trial_idx = self.active_stage
        return True

    def _finalize(self) -> None:
        if not self.residuals:
            self._final_grade = "W"
            self._final_score = 0.0
            return
        mean_res = float(np.mean(self.residuals))
        max_res = float(np.max(self.residuals))
        effective = 0.7 * mean_res + 0.3 * max_res
        score = max(0.0, min(1.0, 1.0 - effective)) * 800.0
        grade = score_to_universal_grade(score)
        self._final_grade = grade
        self._final_score = score

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        artifact = Path(self.save_dir) / f"bench_{int(self.seed)}.personality.pt"
        try:
            self.personality.save(str(artifact))
            self._artifact_path = str(artifact)
        except Exception:
            pass
        version = next_version(self.save_dir, "personality")
        fname = generate_model_name(grade, "personality", version)
        out = Path(self.save_dir) / fname
        metrics = {
            "baseline": self.baseline_name,
            "n_siblings": self.n_siblings,
            "mean_residual": mean_res,
            "max_residual": max_res,
            "trivial_residual": self.trivial_residual,
            "applied_tensors": (len(self.personality.weight_deltas)
                                if self.personality else 0),
            "score": score, "grade": grade,
            "artifact_path": self._artifact_path,
        }
        torch.save({"grade": grade, "score": score, "metrics": metrics}, str(out))
        with open(out.with_suffix(".json"), "w") as f:
            json.dump({
                "grade": grade, "score": score, "metrics": metrics,
                "timestamp": datetime.now().isoformat(), "model_file": fname,
            }, f, indent=2)
        self._saved_path = str(out)
        try:
            RunLogger().append(RunRecord(
                module="personality", stage="sibling_transfer",
                best_score=score, avg_score=score, grade=grade,
                minutes=(time.monotonic() - self._start_t) / 60.0,
                episodes=1, run_tag=self.run_tag,
            ))
        except Exception as e:
            self.push_event(f"run-log append failed: {e}", "warn")

    def final_summary(self) -> List[Tuple[str, str]]:
        if self._final_grade is None:
            return [("Run did not finish.", "warn")]
        mean_res = float(np.mean(self.residuals)) if self.residuals else 0.0
        return [
            (f"Grade: {self._final_grade}   Score: {self._final_score:.1f}", "accent"),
            (f"Mean residual: {mean_res:.3f}", "text"),
            (f"Saved: {self._saved_path}", "dim"),
        ]


def run_personality_inspector(n_siblings: int = 5, seed: int = 42,
                              save_dir: str = "models/personality",
                              run_tag: str = "") -> Tuple[str, float]:
    ui = PersonalityInspector(n_siblings, seed, save_dir, run_tag)
    ui.run()
    return ui._final_grade or "W", ui._final_score
