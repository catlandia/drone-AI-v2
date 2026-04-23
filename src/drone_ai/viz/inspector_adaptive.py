"""Adaptive visual inspector — watch the online learner recover.

Runs the same baseline-vs-adapted comparison as the terminal benchmark
but plots per-episode rewards as two bars so the user can SEE the gap
close (or not). The pipeline diagram on the left shows: load baseline
→ perturb env → run baseline → run adapted → compute delta.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pygame
import torch

from drone_ai.grading import (
    AdaptiveMetricsGrading, ModelGrader, RunLogger, RunRecord,
    generate_model_name, next_version,
)
from drone_ai.modules.adaptive.learner import AdaptiveConfig, AdaptiveLearner
from drone_ai.modules.adaptive.train import (
    _newest_flycontrol_checkpoint, _perturb_env,
)
from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig
from drone_ai.modules.flycontrol.environment import (
    ACT_DIM, FlyControlEnv, OBS_DIM, TaskType,
)
from drone_ai.viz.inspector_structure import Arrow, Box, StructureInspector
from drone_ai.viz.inspector_common import (
    TEXT, TEXT_ACCENT, TEXT_DIM, TEXT_OK, TEXT_WARN, TEXT_BAD,
)


PIPELINE = [
    "load baseline",
    "perturb env",
    "run baseline",
    "run adapted",
    "score delta",
]


class AdaptiveInspector(StructureInspector):
    """Adaptive benchmark with live per-episode reward bars."""

    def __init__(self, episodes: int = 8, task: TaskType = TaskType.HOVER,
                 seed: int = 42, save_dir: str = "models/adaptive",
                 model_path: Optional[str] = None, run_tag: str = ""):
        # Trial count is: 1 pipeline-step + `episodes` baseline + `episodes` adapted
        # + 1 score step.
        super().__init__(
            title="Adaptive — online recovery vs frozen baseline",
            subtitle="Perturbed env; baseline is frozen, adapted is online-tuned.",
            total_trials=2 + 2 * episodes + 1,
            autoplay_hz=1.5,
        )
        self.episodes = episodes
        self.task = task
        self.seed = seed
        self.save_dir = save_dir
        self.run_tag = run_tag
        self.model_path = model_path

        self.active_stage = 0
        self.baseline_rewards: List[float] = []
        self.adapted_rewards: List[float] = []
        self.baseline_agent: Optional[PPOAgent] = None
        self.adapted_agent: Optional[PPOAgent] = None
        self.learner: Optional[AdaptiveLearner] = None
        self.env: Optional[FlyControlEnv] = None
        self.episode_idx = 0

        self._final_grade: Optional[str] = None
        self._final_score: float = 0.0
        self._saved_path: Optional[str] = None

    def setup(self) -> None:
        if self.model_path is None:
            self.model_path = _newest_flycontrol_checkpoint()
        if self.model_path is None:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            seed_path = str(Path(self.save_dir) / "fresh_baseline.pt")
            agent = PPOAgent(OBS_DIM, ACT_DIM, PPOConfig())
            agent.save(seed_path)
            self.model_path = seed_path
            self.push_event(f"no FC ckpt — wrote fresh baseline: {seed_path}", "warn")
        else:
            self.push_event(f"baseline: {self.model_path}", "dim")

    def structure_diagram(self) -> Tuple[List[Box], List[Arrow]]:
        # Stage mapping onto the pipeline boxes.
        pipeline_idx = 0
        if self.active_stage == 0:
            pipeline_idx = 0
        elif self.active_stage == 1:
            pipeline_idx = 1
        elif 1 < self.active_stage <= 1 + self.episodes:
            pipeline_idx = 2
        elif 1 + self.episodes < self.active_stage <= 1 + 2 * self.episodes:
            pipeline_idx = 3
        else:
            pipeline_idx = 4

        n = len(PIPELINE)
        boxes = [
            Box(label, x=i / max(1, n - 1), y=0.5,
                highlight=(i == pipeline_idx) and not self.finished)
            for i, label in enumerate(PIPELINE)
        ]
        arrows = [Arrow(i, i + 1) for i in range(n - 1)]
        return boxes, arrows

    def current_thinking(self) -> List[Tuple[str, str]]:
        b_mean = float(np.mean(self.baseline_rewards)) if self.baseline_rewards else 0.0
        a_mean = float(np.mean(self.adapted_rewards)) if self.adapted_rewards else 0.0
        stage_label = PIPELINE[min(self.active_stage, len(PIPELINE) - 1)] \
            if self.active_stage < len(PIPELINE) else "score delta"
        return [
            ("stage",           stage_label),
            ("baseline so far", f"{len(self.baseline_rewards)}/{self.episodes}"),
            ("adapted so far",  f"{len(self.adapted_rewards)}/{self.episodes}"),
            ("baseline mean",   f"{b_mean:+.1f}"),
            ("adapted mean",    f"{a_mean:+.1f}"),
            ("recovery",        f"{(a_mean - b_mean):+.1f}"),
        ]

    def sidebar_lines(self) -> List[Tuple[str, str, str]]:
        b_mean = float(np.mean(self.baseline_rewards)) if self.baseline_rewards else 0.0
        a_mean = float(np.mean(self.adapted_rewards)) if self.adapted_rewards else 0.0
        return [
            ("baseline ep",   str(len(self.baseline_rewards)), "text"),
            ("adapted ep",    str(len(self.adapted_rewards)), "text"),
            ("baseline mean", f"{b_mean:+.1f}", "dim"),
            ("adapted mean",  f"{a_mean:+.1f}", "accent"),
            ("",              "", "dim"),
            ("delta",         f"{(a_mean - b_mean):+.1f}",
                              "ok" if a_mean > b_mean else "warn"),
        ]

    # Override render: draw the structure diagram AND a bar chart below it.
    def render(self, surface: pygame.Surface, view_rect: pygame.Rect) -> None:
        # Top half: structure + thinking (as StructureInspector).
        top = pygame.Rect(view_rect.x, view_rect.y,
                          view_rect.width, int(view_rect.height * 0.5))
        bot = pygame.Rect(view_rect.x, top.bottom + 4,
                          view_rect.width, view_rect.height - top.height - 4)
        super().render(surface, top)

        # Bottom: side-by-side reward bar charts.
        pygame.draw.rect(surface, (28, 34, 48), bot, border_radius=6)
        pygame.draw.rect(surface, (60, 72, 95), bot, 1, border_radius=6)
        half = bot.width // 2
        self._draw_bars(surface,
                        pygame.Rect(bot.x + 8, bot.y + 6, half - 12, bot.height - 12),
                        "Baseline (frozen)", self.baseline_rewards,
                        (150, 160, 175))
        self._draw_bars(surface,
                        pygame.Rect(bot.x + half + 4, bot.y + 6,
                                    half - 12, bot.height - 12),
                        "Adapted (online)", self.adapted_rewards,
                        (120, 220, 150))

    def _draw_bars(self, surface, rect, title, values, color):
        title_surf = self.font_md.render(title, True, TEXT_ACCENT)
        surface.blit(title_surf, (rect.x + 6, rect.y + 2))
        chart = pygame.Rect(rect.x + 6, rect.y + 26, rect.width - 12, rect.height - 34)
        pygame.draw.rect(surface, (40, 50, 66), chart, 1)
        if not values:
            msg = self.font_sm.render("(no episodes yet)", True, TEXT_DIM)
            surface.blit(msg, (chart.x + 6, chart.y + 8))
            return
        # Scale bars to the largest magnitude observed.
        amp = max(1.0, max(abs(v) for v in values))
        mid = chart.y + chart.height // 2
        bar_w = max(4, chart.width // max(len(values), 1) - 2)
        for i, v in enumerate(values):
            h = int((abs(v) / amp) * (chart.height // 2 - 4))
            x = chart.x + 2 + i * (bar_w + 2)
            if v >= 0:
                pygame.draw.rect(surface, color, (x, mid - h, bar_w, h))
            else:
                pygame.draw.rect(surface, (180, 100, 100), (x, mid, bar_w, h))
        # Baseline zero line.
        pygame.draw.line(surface, (90, 100, 120), (chart.x, mid),
                         (chart.right, mid), 1)

    def step(self) -> bool:
        try:
            if self.active_stage == 0:
                # Load baseline.
                self.baseline_agent = PPOAgent.from_file(self.model_path)
                self.adapted_agent = PPOAgent.from_file(self.model_path)
                self.push_event("loaded baseline + clone for adaptation", "ok")
            elif self.active_stage == 1:
                # Perturb env (create + perturb; rebuilt per-episode).
                self.push_event("env perturbed: mass *1.2, batt_min 0.70", "warn")
            elif 1 < self.active_stage <= 1 + self.episodes:
                # Baseline episode.
                env = FlyControlEnv(task=self.task, difficulty=0.5,
                                    seed=self.seed + self.active_stage)
                _perturb_env(env)
                r = self._run_one(env, self.baseline_agent, learner=None)
                self.baseline_rewards.append(r)
                self.push_event(
                    f"baseline ep {len(self.baseline_rewards)}/{self.episodes}  r={r:+.1f}",
                    "text",
                )
            elif 1 + self.episodes < self.active_stage <= 1 + 2 * self.episodes:
                # Adapted episode.
                env = FlyControlEnv(task=self.task, difficulty=0.5,
                                    seed=self.seed + self.active_stage)
                _perturb_env(env)
                if self.learner is None:
                    self.learner = AdaptiveLearner(
                        self.adapted_agent, AdaptiveConfig(enabled=True),
                    )
                r = self._run_one(env, self.adapted_agent, learner=self.learner)
                self.adapted_rewards.append(r)
                self.push_event(
                    f"adapted ep {len(self.adapted_rewards)}/{self.episodes}  r={r:+.1f}",
                    "ok" if r > (self.baseline_rewards[-1] if self.baseline_rewards else -1e9) else "warn",
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

    def _run_one(self, env: FlyControlEnv, agent: PPOAgent,
                 learner: Optional[AdaptiveLearner]) -> float:
        obs, _ = env.reset()
        total = 0.0
        while True:
            if learner is not None:
                action, info = learner.select_action(obs, deterministic=False)
            else:
                action, info = agent.select_action(obs, deterministic=True)
            next_obs, r, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            if learner is not None:
                learner.observe(obs, action, float(r), info, done)
            total += float(r)
            obs = next_obs
            if done:
                if learner is not None:
                    learner.end_episode(next_obs, total)
                break
        return total

    def _finalize(self) -> None:
        b_mean = float(np.mean(self.baseline_rewards)) if self.baseline_rewards else 0.0
        a_mean = float(np.mean(self.adapted_rewards)) if self.adapted_rewards else 0.0
        recovery = (a_mean - b_mean) / max(abs(b_mean), 1.0)
        stability = 1.0 - (float(np.std(self.adapted_rewards))
                           / (abs(a_mean) + 1.0) if self.adapted_rewards else 1.0)
        stability = max(0.0, min(1.0, stability))
        metrics = AdaptiveMetricsGrading(
            baseline_score=b_mean,
            adapted_score=a_mean,
            recovery_rate=recovery,
            stability=stability,
        )
        grader = ModelGrader()
        grade, score = grader.grade_adaptive(metrics)
        self._final_grade = grade
        self._final_score = score

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        version = next_version(self.save_dir, "adaptive")
        fname = generate_model_name(grade, "adaptive", version)
        out = Path(self.save_dir) / fname
        torch.save({
            "grade": grade, "score": score,
            "baseline_path": self.model_path,
            "metrics": {
                "baseline_score": b_mean,
                "adapted_score": a_mean,
                "recovery_rate": recovery,
                "stability": stability,
            },
        }, str(out))
        with open(out.with_suffix(".json"), "w") as f:
            json.dump({
                "grade": grade, "score": score,
                "baseline_path": self.model_path,
                "metrics": {
                    "baseline_score": b_mean, "adapted_score": a_mean,
                    "recovery_rate": recovery, "stability": stability,
                },
                "timestamp": datetime.now().isoformat(), "model_file": fname,
            }, f, indent=2)
        self._saved_path = str(out)
        try:
            RunLogger().append(RunRecord(
                module="adaptive", stage=str(self.task.value),
                best_score=score, avg_score=a_mean, grade=grade,
                minutes=(time.monotonic() - self._start_t) / 60.0,
                episodes=self.episodes, run_tag=self.run_tag,
            ))
        except Exception as e:
            self.push_event(f"run-log append failed: {e}", "warn")

    def final_summary(self) -> List[Tuple[str, str]]:
        if self._final_grade is None:
            return [("Run did not finish.", "warn")]
        b_mean = float(np.mean(self.baseline_rewards)) if self.baseline_rewards else 0.0
        a_mean = float(np.mean(self.adapted_rewards)) if self.adapted_rewards else 0.0
        return [
            (f"Grade: {self._final_grade}   Score: {self._final_score:.1f}", "accent"),
            (f"Baseline: {b_mean:+.1f}", "text"),
            (f"Adapted:  {a_mean:+.1f}", "text"),
            (f"Recovery: {a_mean - b_mean:+.1f}",
             "ok" if a_mean > b_mean else "warn"),
            (f"Saved: {self._saved_path}", "dim"),
        ]


def run_adaptive_inspector(episodes: int = 8, seed: int = 42,
                           save_dir: str = "models/adaptive",
                           model_path: Optional[str] = None,
                           run_tag: str = "") -> Tuple[str, float]:
    ui = AdaptiveInspector(
        episodes=episodes, seed=seed, save_dir=save_dir,
        model_path=model_path, run_tag=run_tag,
    )
    ui.run()
    return ui._final_grade or "W", ui._final_score
