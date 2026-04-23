"""Perception visual inspector.

Shows what the perception module "recognizes" on each frame:
  - drone position + detection range as a circle
  - ground-truth obstacles within range (green outlines)
  - detections the model returned (blue filled, with error line to
    the matched ground-truth)
  - false positives (red X)
  - misses (ground-truth that had no detection within 5 m — orange)

The same accuracy metrics the terminal benchmark computes update in
the sidebar so the user can SEE the "F+ perception misses half the
obstacles" story unfold instead of staring at a single final number.
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
    ModelGrader, PerceptionMetrics, RunLogger, RunRecord,
    generate_model_name, next_version,
)
from drone_ai.modules.perception.detector import PerceptionAI
from drone_ai.simulation.world import World
from drone_ai.viz.inspector_common import (
    BG, BORDER, PANEL, TEXT, TEXT_ACCENT, TEXT_DIM, TEXT_OK, TEXT_WARN, TEXT_BAD,
    InspectorBase, RunningStats, TopDownProjector, draw_grid,
)


DRONE_COLOR = (120, 220, 150)
RANGE_COLOR = (80, 130, 180, 90)
TRUE_OBSTACLE = (90, 200, 120)
DETECTION = (140, 175, 230)
ERROR_LINE = (220, 200, 110)
FALSE_POS = (240, 110, 110)
MISSED    = (240, 160, 90)


class PerceptionInspector(InspectorBase):
    """Step through perception trials, visualizing each detection."""

    MATCH_RADIUS = 5.0  # same threshold the terminal benchmark uses

    def __init__(
        self,
        grade: str = "P",
        n_trials: int = 30,
        seed: int = 42,
        save_dir: str = "models/perception",
        run_tag: str = "",
    ):
        super().__init__(
            title=f"Perception — simulated @ grade {grade}",
            subtitle="Each frame: random scene → detect → score TP/FP/miss.",
            total_trials=n_trials,
            autoplay_hz=2.0,
        )
        self.input_grade = grade
        self.n_trials = n_trials
        self.seed = seed
        self.save_dir = save_dir
        self.run_tag = run_tag

        self.rng = np.random.default_rng(seed)
        self.perception = PerceptionAI(grade=grade, seed=seed)
        self.world = World()

        # Current-frame state.
        self.drone_pos: Optional[np.ndarray] = None
        self.nearby: List = []
        self.detections: List = []
        self.tps: int = 0
        self.fps_this: int = 0
        self.missed_idx: List[int] = []
        self.errors: List[float] = []
        self.last_ms = 0.0
        self.proj: Optional[TopDownProjector] = None

        # Rolling totals.
        self.total_true = 0
        self.total_tp = 0
        self.total_fp = 0
        self.stats_err = RunningStats()
        self.stats_ms = RunningStats()

        # End-of-run artifacts.
        self._final_grade: Optional[str] = None
        self._final_score: float = 0.0
        self._saved_path: Optional[str] = None

    # ---- scenario -------------------------------------------------------

    def setup(self) -> None:
        self._new_scenario()

    def _new_scenario(self) -> None:
        self.world.clear()
        n_obs = int(self.rng.integers(5, 20))
        self.world.generate_random_obstacles(n_obs, self.rng)
        self.drone_pos = self.rng.uniform([-30, -30, 5], [30, 30, 30])
        self.nearby = self.world.obstacles_in_radius(
            self.drone_pos, self.perception.detection_range,
        )
        t0 = time.perf_counter()
        self.detections = self.perception.detect(self.drone_pos, self.world)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.stats_ms.push(self.last_ms)

        # Match detections to ground-truth.
        self.tps = 0
        self.fps_this = 0
        matched = set()
        self.errors = []
        for det in self.detections:
            dists = [float(np.linalg.norm(det.position - o.position)) for o in self.nearby]
            if dists and min(dists) < self.MATCH_RADIUS:
                idx = int(np.argmin(dists))
                matched.add(idx)
                self.tps += 1
                self.errors.append(dists[idx])
            else:
                self.fps_this += 1
        self.missed_idx = [i for i in range(len(self.nearby)) if i not in matched]

        # Update global aggregates.
        self.total_true += len(self.nearby)
        self.total_tp += self.tps
        self.total_fp += self.fps_this
        for e in self.errors:
            self.stats_err.push(e)

        # Reproject so view always contains drone + obstacles.
        pts = [self.drone_pos[:2]] + [o.position[:2] for o in self.nearby]
        self.proj = TopDownProjector.from_points(self.view_rect, pts,
                                                 padding_world=15.0)

    def step(self) -> bool:
        self.trial_idx += 1
        if self.trial_idx >= self.n_trials:
            self._finalize()
            return False
        self._new_scenario()
        return True

    # ---- render ---------------------------------------------------------

    def render(self, surface: pygame.Surface, view_rect: pygame.Rect) -> None:
        if self.proj is None or self.drone_pos is None:
            return
        draw_grid(surface, self.proj, step=20.0)

        # Detection range as a filled circle with alpha.
        cx, cy = self.proj.to_screen(self.drone_pos[0], self.drone_pos[1])
        r_px = self.proj.size_px(self.perception.detection_range)
        range_surf = pygame.Surface((r_px * 2, r_px * 2), pygame.SRCALPHA)
        pygame.draw.circle(range_surf, RANGE_COLOR, (r_px, r_px), r_px)
        surface.blit(range_surf, (cx - r_px, cy - r_px))

        # True obstacles within range (outline only so detections show through).
        for i, o in enumerate(self.nearby):
            ox, oy = self.proj.to_screen(o.position[0], o.position[1])
            sz = max(3, self.proj.size_px(float(o.size[0])))
            color = MISSED if i in self.missed_idx else TRUE_OBSTACLE
            pygame.draw.circle(surface, color, (ox, oy), sz, 2)

        # Detections.
        for det in self.detections:
            dx, dy = self.proj.to_screen(det.position[0], det.position[1])
            matched = False
            if self.nearby:
                dists = [float(np.linalg.norm(det.position - o.position)) for o in self.nearby]
                if min(dists) < self.MATCH_RADIUS:
                    matched = True
                    idx = int(np.argmin(dists))
                    ox, oy = self.proj.to_screen(
                        self.nearby[idx].position[0], self.nearby[idx].position[1],
                    )
                    pygame.draw.line(surface, ERROR_LINE, (dx, dy), (ox, oy), 1)
            if matched:
                pygame.draw.circle(surface, DETECTION, (dx, dy), 4)
            else:
                # False positive: red X.
                pygame.draw.line(surface, FALSE_POS, (dx - 4, dy - 4), (dx + 4, dy + 4), 2)
                pygame.draw.line(surface, FALSE_POS, (dx - 4, dy + 4), (dx + 4, dy - 4), 2)

        # Drone marker last so it's on top.
        pygame.draw.circle(surface, DRONE_COLOR, (cx, cy), 6)
        pygame.draw.circle(surface, (0, 0, 0), (cx, cy), 6, 1)

        legend = (
            "drone (green dot)   range (blue disc)   "
            "true (green O)   detection (blue)   error-line (yellow)   "
            "missed (orange O)   false-pos (red X)"
        )
        surf = self.font_sm.render(legend, True, TEXT_DIM)
        surface.blit(surf, (view_rect.x + 10, view_rect.bottom - 18))

    def sidebar_lines(self) -> List[Tuple[str, str, str]]:
        tp_rate = 100.0 * self.total_tp / max(self.total_true, 1)
        fp_rate = 100.0 * self.total_fp / max(self.total_tp + self.total_fp, 1)
        err_mean = self.stats_err.mean() if self.stats_err.values else 0.0
        fps = 1000.0 / max(self.stats_ms.mean(), 0.001) if self.stats_ms.values else 0.0
        return [
            ("input grade",    self.input_grade, "accent"),
            ("this-frame TP",  str(self.tps), "ok"),
            ("this-frame FP",  str(self.fps_this),
                               "bad" if self.fps_this else "dim"),
            ("this-frame miss",str(len(self.missed_idx)),
                               "warn" if self.missed_idx else "dim"),
            ("",               "",                "dim"),
            ("cum detection",  f"{tp_rate:5.1f}%",
                               "ok" if tp_rate > 80 else "warn"),
            ("cum false-pos",  f"{fp_rate:5.1f}%",
                               "ok" if fp_rate < 10 else "warn"),
            ("avg pos error",  f"{err_mean:.2f} m",
                               "ok" if err_mean < 2 else "warn"),
            ("est FPS",        f"{fps:.0f}", "text"),
        ]

    # ---- finalize -------------------------------------------------------

    def _finalize(self) -> None:
        tp_rate = 100.0 * self.total_tp / max(self.total_true, 1)
        fp_rate = 100.0 * self.total_fp / max(self.total_tp + self.total_fp, 1)
        err_mean = self.stats_err.mean() if self.stats_err.values else 10.0
        fps = 1000.0 / max(self.stats_ms.mean(), 0.001) if self.stats_ms.values else 1.0

        metrics = PerceptionMetrics(
            detection_accuracy=tp_rate,
            false_positive_rate=fp_rate,
            position_error=err_mean,
            fps=fps,
        )
        grader = ModelGrader()
        grade, score = grader.grade_perception(metrics)
        self._final_grade = grade
        self._final_score = score

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        version = next_version(self.save_dir, "perception")
        fname = generate_model_name(grade, "perception", version)
        out = Path(self.save_dir) / fname
        import torch
        torch.save({
            "grade": grade, "input_grade": self.input_grade,
            "score": score,
            "metrics": {
                "detection_accuracy": tp_rate,
                "false_positive_rate": fp_rate,
                "position_error": err_mean,
                "fps": fps,
            },
        }, str(out))
        with open(out.with_suffix(".json"), "w") as f:
            json.dump({
                "grade": grade, "score": score,
                "input_grade": self.input_grade,
                "metrics": {
                    "detection_accuracy": tp_rate,
                    "false_positive_rate": fp_rate,
                    "position_error": err_mean,
                    "fps": fps,
                },
                "timestamp": datetime.now().isoformat(),
                "model_file": fname,
            }, f, indent=2)
        self._saved_path = str(out)
        try:
            RunLogger().append(RunRecord(
                module="perception", stage="benchmark",
                best_score=score, avg_score=score, grade=grade,
                minutes=(time.monotonic() - self._start_t) / 60.0,
                episodes=self.n_trials, run_tag=self.run_tag,
            ))
        except Exception as e:
            print(f"[perception-ui] run-log append failed: {e}")

    def final_summary(self) -> List[Tuple[str, str]]:
        if self._final_grade is None:
            return [("Run did not finish.", "warn")]
        tp_rate = 100.0 * self.total_tp / max(self.total_true, 1)
        return [
            (f"Grade: {self._final_grade}   Score: {self._final_score:.1f}", "accent"),
            (f"Detection: {tp_rate:.1f}%", "text"),
            (f"Pos error: {self.stats_err.mean():.2f} m", "text"),
            (f"Saved: {self._saved_path}", "dim"),
        ]


def run_perception_inspector(grade: str = "P", trials: int = 30,
                             seed: int = 42,
                             save_dir: str = "models/perception",
                             run_tag: str = "") -> Tuple[str, float]:
    ui = PerceptionInspector(
        grade=grade, n_trials=trials, seed=seed,
        save_dir=save_dir, run_tag=run_tag,
    )
    ui.run()
    return ui._final_grade or "W", ui._final_score
