"""Pathfinder visual inspector.

Runs the same benchmark as `modules/pathfinder/train.py` but with a
top-down view of each scenario so the user can actually SEE what the
planner does on each trial: the world, obstacles, start/goal, and the
path the planner returned. Running stats update in the sidebar.

Keeps the terminal version (train.py) working — this is a second entry
point that writes the same .pt + runs.csv row at the end.
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
    ModelGrader, PathfinderMetrics, RunLogger, RunRecord,
    generate_model_name, next_version,
)
from drone_ai.modules.pathfinder.algorithms import PathPlanner
from drone_ai.simulation.world import Obstacle, World
from drone_ai.viz.inspector_common import (
    BG, BORDER, PANEL, TEXT, TEXT_ACCENT, TEXT_DIM, TEXT_OK, TEXT_WARN,
    InspectorBase, RunningStats, TopDownProjector, draw_grid,
)


START_COLOR = (110, 220, 150)
GOAL_COLOR = (240, 110, 110)
OBSTACLE_COLOR = (110, 120, 140)
OBSTACLE_EDGE = (170, 180, 200)
PATH_COLOR = (140, 175, 230)
PATH_POINT = (200, 220, 255)


class PathfinderInspector(InspectorBase):
    """Step through each benchmark trial with a visible planner result."""

    def __init__(
        self,
        n_trials: int = 30,
        seed: int = 42,
        save_dir: str = "models/pathfinder",
        run_tag: str = "",
    ):
        super().__init__(
            title="Pathfinder — A* + RRT planner",
            subtitle="Each trial: random obstacles → plan start→goal → measure.",
            total_trials=n_trials,
            autoplay_hz=2.0,
        )
        self.n_trials = n_trials
        self.seed = seed
        self.save_dir = save_dir
        self.run_tag = run_tag
        self.rng = np.random.default_rng(seed)
        self.world = World()
        self.planner = PathPlanner(self.world)

        # Current scenario state.
        self.start: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None
        self.path: List[np.ndarray] = []
        self.plan_ms = 0.0
        self.optimal = 0.0
        self.length = 0.0
        self.collides = False
        self.proj: Optional[TopDownProjector] = None

        # Rolling stats.
        self.stats_optimality = RunningStats()
        self.stats_plan_ms = RunningStats()
        self.stats_clear = 0
        self.stats_counted = 0

        # End-of-run artifacts.
        self._final_grade: Optional[str] = None
        self._final_score: float = 0.0
        self._saved_path: Optional[str] = None

    # ---- scenario setup -------------------------------------------------

    def setup(self) -> None:
        # Prime the first scenario before the loop kicks in.
        self._new_scenario()

    def _new_scenario(self) -> None:
        self.world.clear()
        n_obs = int(self.rng.integers(5, 25))
        self.world.generate_random_obstacles(n_obs, self.rng)
        # Keep trying until start/goal are far enough apart — same
        # rule as the terminal benchmark (`if optimal < 5: continue`).
        for _ in range(20):
            self.start = self.world.random_free_point(self.rng)
            self.goal = self.world.random_free_point(self.rng)
            self.optimal = float(np.linalg.norm(self.goal - self.start))
            if self.optimal >= 5.0:
                break
        t0 = time.perf_counter()
        path = self.planner.plan(self.start, self.goal)
        self.plan_ms = (time.perf_counter() - t0) * 1000.0
        self.path = list(path) if path else []
        self.length = self._path_length(self.path)
        self.collides = self._path_collides(self.path)

        # Update rolling stats. Unreachable goals (no path returned) do
        # NOT contribute an optimality sample — the terminal benchmark
        # also only counts paths with >= 2 points.
        if self.path and len(self.path) >= 2:
            self.stats_counted += 1
            ratio = self.length / max(self.optimal, 1.0)
            self.stats_optimality.push(ratio)
            if not self.collides:
                self.stats_clear += 1
        self.stats_plan_ms.push(self.plan_ms)

        # Rebuild the 2D projector so the view fits the current scene.
        # Everything we hand the projector must be a 2-tuple (x, y) —
        # mixing 3D `DroneState`/obstacle positions with hand-built 2D
        # extents makes numpy's autoshape reject the list.
        pts = [(float(self.start[0]), float(self.start[1])),
               (float(self.goal[0]),  float(self.goal[1]))]
        for o in self.world.obstacles:
            pts.append((float(o.position[0] + o.size[0]),
                        float(o.position[1] + o.size[1])))
            pts.append((float(o.position[0] - o.size[0]),
                        float(o.position[1] - o.size[1])))
        self.proj = TopDownProjector.from_points(self.view_rect, pts)

    # ---- step / render --------------------------------------------------

    def step(self) -> bool:
        self.trial_idx += 1
        if self.trial_idx >= self.n_trials:
            self._finalize()
            return False
        self._new_scenario()
        return True

    def render(self, surface: pygame.Surface, view_rect: pygame.Rect) -> None:
        if self.proj is None:
            return
        draw_grid(surface, self.proj, step=25.0)
        # Obstacles (drawn as 2D top-down boxes; we ignore z for display).
        for o in self.world.obstacles:
            x0 = o.position[0] - o.size[0]
            x1 = o.position[0] + o.size[0]
            y0 = o.position[1] - o.size[1]
            y1 = o.position[1] + o.size[1]
            sx0, sy0 = self.proj.to_screen(x0, y1)  # top-left in screen
            sx1, sy1 = self.proj.to_screen(x1, y0)
            rect = pygame.Rect(sx0, sy0, max(2, sx1 - sx0), max(2, sy1 - sy0))
            pygame.draw.rect(surface, OBSTACLE_COLOR, rect)
            pygame.draw.rect(surface, OBSTACLE_EDGE, rect, 1)

        # Path polyline.
        if len(self.path) >= 2:
            pts = [self.proj.to_screen(p[0], p[1]) for p in self.path]
            pygame.draw.lines(surface, PATH_COLOR, False, pts, 2)
            for p in pts:
                pygame.draw.circle(surface, PATH_POINT, p, 2)
        elif self.start is not None and self.goal is not None:
            # Dashed "no path found" line, rendered as a dim dotted polyline.
            a = self.proj.to_screen(self.start[0], self.start[1])
            b = self.proj.to_screen(self.goal[0], self.goal[1])
            pygame.draw.line(surface, (120, 70, 80), a, b, 1)

        # Start / goal markers.
        if self.start is not None:
            s = self.proj.to_screen(self.start[0], self.start[1])
            pygame.draw.circle(surface, START_COLOR, s, 7)
            pygame.draw.circle(surface, (0, 0, 0), s, 7, 1)
        if self.goal is not None:
            g = self.proj.to_screen(self.goal[0], self.goal[1])
            pygame.draw.circle(surface, GOAL_COLOR, g, 7)
            pygame.draw.circle(surface, (0, 0, 0), g, 7, 1)

        # Inline caption so the screen is self-explanatory.
        caption = (
            "start (green) → goal (red)   "
            "obstacles (grey)   path (blue)"
        )
        surf = self.font_sm.render(caption, True, TEXT_DIM)
        surface.blit(surf, (view_rect.x + 10, view_rect.bottom - 18))

    def sidebar_lines(self) -> List[Tuple[str, str, str]]:
        clear_pct = 100.0 * self.stats_clear / max(self.stats_counted, 1)
        status = "collision!" if self.collides else "clear"
        status_style = "bad" if self.collides else "ok"
        found = bool(self.path and len(self.path) >= 2)
        return [
            ("obstacles",     str(len(self.world.obstacles)), "text"),
            ("straight-line", f"{self.optimal:5.1f} m", "dim"),
            ("path length",   f"{self.length:5.1f} m" if found else "—",
                              "text" if found else "warn"),
            ("path found",    "yes" if found else "NO",
                              "ok" if found else "bad"),
            ("path clear",    status, status_style),
            ("plan time",     f"{self.plan_ms:5.1f} ms", "text"),
            ("",              "",  "dim"),
            ("avg optimality",f"{self.stats_optimality.mean():.2f} (1.0 = perfect)", "accent"),
            ("avg plan time", f"{self.stats_plan_ms.mean():.1f} ms", "text"),
            ("collision-free",f"{clear_pct:.1f}%", "ok" if clear_pct > 90 else "warn"),
        ]

    # ---- finalize -------------------------------------------------------

    def _finalize(self) -> None:
        avg_opt = self.stats_optimality.mean() if self.stats_optimality.values else 5.0
        avoid = self.stats_clear / max(self.stats_counted, 1)
        avg_ms = self.stats_plan_ms.mean() if self.stats_plan_ms.values else 999.0
        metrics = PathfinderMetrics(
            path_optimality=avg_opt,
            avoidance_rate=avoid,
            planning_ms=avg_ms,
        )
        grader = ModelGrader()
        grade, score = grader.grade_pathfinder(metrics)
        self._final_grade = grade
        self._final_score = score

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        version = next_version(self.save_dir, "pathfinder")
        fname = generate_model_name(grade, "pathfinder", version)
        out = Path(self.save_dir) / fname
        import torch
        torch.save({"grade": grade, "algorithm": "A*+RRT", "score": score}, str(out))
        with open(out.with_suffix(".json"), "w") as f:
            json.dump({
                "grade": grade, "score": score,
                "metrics": {
                    "path_optimality": avg_opt,
                    "avoidance_rate": avoid,
                    "planning_ms": avg_ms,
                },
                "timestamp": datetime.now().isoformat(),
                "model_file": fname,
                "algorithm": "A*+RRT",
            }, f, indent=2)
        self._saved_path = str(out)
        try:
            RunLogger().append(RunRecord(
                module="pathfinder", stage="benchmark",
                best_score=score, avg_score=score,
                grade=grade, minutes=(time.monotonic() - self._start_t) / 60.0,
                episodes=self.n_trials, run_tag=self.run_tag,
            ))
        except Exception as e:
            print(f"[pathfinder-ui] run-log append failed: {e}")

    def final_summary(self) -> List[Tuple[str, str]]:
        if self._final_grade is None:
            return [("Run did not finish.", "warn")]
        return [
            (f"Grade: {self._final_grade}   Score: {self._final_score:.1f}", "accent"),
            (f"Optimality: {self.stats_optimality.mean():.2f}", "text"),
            (f"Collision-free: "
             f"{100.0 * self.stats_clear / max(self.stats_counted, 1):.1f}%", "text"),
            (f"Plan time: {self.stats_plan_ms.mean():.1f} ms", "text"),
            (f"Saved: {self._saved_path}", "dim"),
        ]

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _path_length(path: List[np.ndarray]) -> float:
        if not path or len(path) < 2:
            return 0.0
        return float(sum(np.linalg.norm(path[i + 1] - path[i])
                         for i in range(len(path) - 1)))

    def _path_collides(self, path: List[np.ndarray], margin: float = 0.3) -> bool:
        return any(self.world.in_collision(p, margin) for p in path)


def run_pathfinder_inspector(trials: int = 30, seed: int = 42,
                             save_dir: str = "models/pathfinder",
                             run_tag: str = "") -> Tuple[str, float]:
    ui = PathfinderInspector(
        n_trials=trials, seed=seed, save_dir=save_dir, run_tag=run_tag,
    )
    ui.run()
    return ui._final_grade or "W", ui._final_score
