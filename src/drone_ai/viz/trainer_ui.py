"""Live 3D training UI — runs PPO training on a chosen task while rendering
the current policy's behavior in a 3D window.

Every render frame we step the env with the current policy; every N steps
a PPO update runs. Rewards and episode stats accumulate into the HUD.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

import os

from drone_ai.grading import (
    RunLogger, RunRecord, score_to_flycontrol_grade,
    generate_model_name, next_version, parse_model_name,
)
from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig
from drone_ai.modules.flycontrol.environment import (
    FlyControlEnv, TaskType, OBS_DIM, ACT_DIM,
)
from drone_ai.viz.renderer3d import Renderer


# Curriculum chain — each stage warm-starts from the previous stage's
# latest checkpoint. Hover trains from scratch; deployment sits at the
# end and keeps learning on top of delivery_route's base.
STAGE_ORDER: List[str] = [
    "hover", "waypoint", "delivery", "delivery_route", "deployment",
]


STAGE_DEFS: Dict[str, Dict] = {
    "hover": {
        "title": "HOVER",
        "subtitle": "Learn to maintain a stable target position",
        "task": TaskType.HOVER,
        "difficulty": 0.3,
        "domain_rand": False,
    },
    "waypoint": {
        "title": "WAYPOINT",
        "subtitle": "Navigate between scattered targets",
        "task": TaskType.HOVER,
        "difficulty": 0.9,
        "domain_rand": False,
    },
    "delivery": {
        "title": "DELIVERY",
        "subtitle": "Pickup → dropzone, single package",
        "task": TaskType.DELIVERY,
        "difficulty": 0.5,
        "domain_rand": False,
    },
    "delivery_route": {
        "title": "DELIVERY ROUTE",
        "subtitle": "Multi-stop delivery with obstacles",
        "task": TaskType.DELIVERY_ROUTE,
        "difficulty": 0.6,
        "domain_rand": False,
    },
    "deployment": {
        "title": "DEPLOYMENT READY",
        "subtitle": "Full difficulty + domain randomization",
        "task": TaskType.DEPLOYMENT,
        "difficulty": 1.0,
        "domain_rand": True,
    },
}


def flycontrol_stage_dir(models_root: str, stage: str) -> str:
    """Where a stage's flycontrol checkpoints live on disk.

    Stage subfolders keep the curriculum auditable: every hover run ends
    up in models/flycontrol/hover/, every waypoint run in
    models/flycontrol/waypoint/, etc. Lets the launcher show per-stage
    progress and lets the next stage find a specific predecessor base.
    """
    return os.path.join(models_root, "flycontrol", stage)


def latest_flycontrol_checkpoint(models_root: str, stage: str) -> Optional[str]:
    """Return absolute path to the newest flycontrol checkpoint for this
    stage (by version), or None if the stage has no checkpoints yet."""
    sdir = flycontrol_stage_dir(models_root, stage)
    if not os.path.isdir(sdir):
        return None
    best_v = -1
    best_name = ""
    for fname in os.listdir(sdir):
        parsed = parse_model_name(fname)
        if parsed and parsed["module"] == "flycontrol" and parsed["version"] > best_v:
            best_v = parsed["version"]
            best_name = fname
    return os.path.join(sdir, best_name) if best_name else None


def resolve_warm_start(models_root: str, stage: str) -> Optional[str]:
    """Pick the checkpoint to warm-start this stage from.

    First preference: this stage's own latest checkpoint (resume training
    where we left off). Fallback: the nearest earlier stage's latest
    checkpoint (curriculum step-up). Returns None for hover on a fresh
    install — hover has no predecessor and no prior run.
    """
    own = latest_flycontrol_checkpoint(models_root, stage)
    if own:
        return own
    if stage not in STAGE_ORDER:
        return None
    idx = STAGE_ORDER.index(stage)
    for prev in reversed(STAGE_ORDER[:idx]):
        prev_ckpt = latest_flycontrol_checkpoint(models_root, prev)
        if prev_ckpt:
            return prev_ckpt
    return None


@dataclass
class TrainConfig:
    stage: str = "hover"
    seed: int = 42
    steps_per_update: int = 512
    total_updates: int = 1000
    save_path: Optional[str] = None
    run_tag: str = ""                        # free-form tag for the run log
    log_path: str = "models/runs.csv"
    # Optional explicit warm-start checkpoint path. If None, the trainer
    # falls back to resolve_warm_start() (this stage's newest, then the
    # nearest earlier stage's). Set by the launcher's pre-launch picker
    # so the user can pin the base model instead of accepting the
    # automatic choice.
    warm_start_path: Optional[str] = None
    # If True, after training completes the trainer holds the window
    # open on a results screen until the user dismisses it. The launcher
    # turns this on so missions don't auto-exit the moment they finish.
    hold_on_finish: bool = True
    # If True, the run is labeled "TEST TRAINING" everywhere the user
    # sees it — HUD title, results screen, runs.csv run_tag. The launcher
    # sets this when the user picked "(fresh)" in the pre-launch picker,
    # so a run without a real baseline can be recognized at a glance
    # (useful when the curriculum chain is still empty and you just want
    # to exercise the training machinery).
    test_run: bool = False


class TrainerUI:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.stage_def = STAGE_DEFS.get(cfg.stage, STAGE_DEFS["hover"])
        self.env = FlyControlEnv(
            task=self.stage_def["task"],
            difficulty=self.stage_def["difficulty"],
            domain_randomization=self.stage_def["domain_rand"],
            seed=cfg.seed,
        )
        self.agent = PPOAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM, config=PPOConfig())
        # Warm-start order (PLAN.md §4 curriculum chain):
        #   1. Explicit cfg.warm_start_path — picked by the launcher's
        #      pre-launch picker. The user wins.
        #   2. resolve_warm_start() — this stage's newest, then the
        #      nearest earlier stage's. Lets deployment keep refining
        #      on top of delivery_route instead of training from scratch.
        self.base_model_name: Optional[str] = None
        models_root = os.path.dirname(cfg.log_path) or "models"
        warm = cfg.warm_start_path or resolve_warm_start(models_root, cfg.stage)
        if warm is not None:
            try:
                self.agent.load(warm)
                self.base_model_name = os.path.basename(warm)
                print(f"[trainer] warm-started {cfg.stage} from {warm}")
            except Exception as e:
                print(f"[trainer] warm-start skipped ({warm}): {e}")
        prefix = "Test Training" if cfg.test_run else "Training"
        self.renderer = Renderer(title=f"{prefix} — {self.stage_def['title']}")

        self.obs, _ = self.env.reset(seed=cfg.seed)
        self.ep_reward = 0.0
        self.ep_len = 0
        self.episode_idx = 0
        self.update_idx = 0
        self.steps_since_update = 0
        self.best_ep_reward = -float("inf")
        self.recent_rewards: List[float] = []
        self.latest_loss: Optional[float] = None
        # Run-log bookkeeping: capture wall-clock so the log has minutes.
        self._start_time = time.monotonic()
        self._all_rewards: List[float] = []

    # ------------------------------------------------------------------

    def run(self) -> bool:
        running = True
        finished_naturally = False
        save_path: Optional[str] = None
        try:
            while running and self.update_idx < self.cfg.total_updates:
                running = self.renderer.handle_events(1 / 60)
                if not running:
                    break
                if not self.renderer.paused:
                    for _ in range(self.renderer.sim_speed):
                        self._collect_step()
                        if self.steps_since_update >= self.cfg.steps_per_update:
                            self._do_update()
                            break
                self._render_frame()
                self.renderer.flip()

            finished_naturally = self.update_idx >= self.cfg.total_updates

            # Always persist the trained weights — previously the UI path
            # left save_path=None so training runs vanished on exit. Name
            # the file with the tier-list convention so runs.csv rows and
            # disk checkpoints share a grade + date + version.
            save_path = self.cfg.save_path or self._auto_save_path()
            try:
                self.agent.save(save_path)
                print(f"[trainer] saved checkpoint -> {save_path}")
            except Exception as e:
                print(f"[trainer] save failed: {e}")

            # Hold-on-finish results screen: the user asked for the
            # window not to disappear the second training ends. Wait
            # for an explicit dismiss instead of closing immediately.
            if (
                finished_naturally
                and self.cfg.hold_on_finish
                and self.renderer.is_open()
            ):
                self._show_results_screen(save_path)
        finally:
            # Always record the run, even if the user quit early — the
            # minutes and best-score so far are still meaningful data points.
            self._log_run()
            self.renderer.close()
        return finished_naturally

    def _show_results_screen(self, save_path: Optional[str]) -> None:
        """Block on a results panel until the user presses any key /
        clicks / closes the window. Lets them read the final grade
        before returning to the launcher."""
        from drone_ai.grading import GRADE_NAMES, GRADE_DESCRIPTIONS
        avg = (sum(self.recent_rewards) / len(self.recent_rewards)) if self.recent_rewards else 0.0
        best = self.best_ep_reward if self.best_ep_reward != -float("inf") else 0.0
        grade = score_to_flycontrol_grade(best) if self.best_ep_reward != -float("inf") else "—"
        minutes = (time.monotonic() - self._start_time) / 60.0
        is_test = self.cfg.test_run or not self.base_model_name
        title = (
            f"TEST TRAINING — {self.stage_def['title']}"
            if is_test else f"FlyControl — {self.stage_def['title']}"
        )
        results_lines = [
            (title, "title"),
            (f"Stage: {self.cfg.stage}" +
             ("   ·   no base model — results are a training-machinery check" if is_test else ""),
             "dim"),
            ("", "dim"),
            (f"Grade:    {grade}  ({GRADE_NAMES.get(grade,'')})", "accent"),
            (f"Best R:   {best:+.1f}", "text"),
            (f"Avg R:    {avg:+.1f}", "text"),
            (f"Updates:  {self.update_idx}/{self.cfg.total_updates}", "text"),
            (f"Episodes: {self.episode_idx}", "text"),
            (f"Time:     {minutes:.1f} min", "text"),
            ("", "dim"),
            (f"Saved: {save_path or '(none)'}", "dim"),
            ("", "dim"),
            ("Press any key / click / close to return to the launcher.", "accent"),
        ]
        # Run the renderer's idle loop until the user dismisses it.
        if hasattr(self.renderer, "show_modal_text"):
            self.renderer.show_modal_text(results_lines)
            return
        # Fallback for older renderers: just keep the last frame visible
        # until the user closes the window via the renderer's own input.
        try:
            keep_open = True
            while keep_open and self.renderer.is_open():
                keep_open = self.renderer.handle_events(1 / 60)
                self._render_frame()
                self.renderer.flip()
        except Exception:
            pass

    def _auto_save_path(self) -> str:
        # Per-stage subfolder (models/flycontrol/<stage>/) so the curriculum
        # chain stays auditable — each stage's checkpoints live separately
        # and the next stage can pick up from the specific predecessor.
        models_root = os.path.dirname(self.cfg.log_path) or "models"
        stage_dir = flycontrol_stage_dir(models_root, self.cfg.stage)
        os.makedirs(stage_dir, exist_ok=True)
        best = self.best_ep_reward if self.best_ep_reward != -float("inf") else 0.0
        grade = score_to_flycontrol_grade(best)
        version = next_version(stage_dir, "flycontrol")
        return os.path.join(stage_dir, generate_model_name(grade, "flycontrol", version))

    def _log_run(self) -> None:
        minutes = (time.monotonic() - self._start_time) / 60.0
        avg = float(np.mean(self._all_rewards)) if self._all_rewards else 0.0
        best = self.best_ep_reward if self.best_ep_reward != -float("inf") else 0.0
        grade = score_to_flycontrol_grade(best)
        # A fresh-base run (test_run or no base_model_name) is tagged
        # "test" so the runs.csv reader can filter it out when comparing
        # real curriculum steps.
        is_test = self.cfg.test_run or not self.base_model_name
        tag = self.cfg.run_tag or ""
        if is_test and "test" not in tag.split(","):
            tag = f"{tag},test".lstrip(",")
        rec = RunRecord(
            module="flycontrol",
            stage=self.cfg.stage,
            best_score=best,
            avg_score=avg,
            grade=grade,
            minutes=minutes,
            updates=self.update_idx,
            episodes=self.episode_idx,
            run_tag=tag,
        )
        try:
            RunLogger(self.cfg.log_path).append(rec)
        except Exception as e:
            print(f"[trainer] run-log append failed: {e}")

    # ------------------------------------------------------------------

    def _collect_step(self):
        action, info = self.agent.select_action(self.obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done = bool(terminated or truncated)
        self.agent.store(self.obs, action, float(reward), info["value"], info["log_prob"], done)

        self.ep_reward += float(reward)
        self.ep_len += 1
        self.steps_since_update += 1
        self.obs = next_obs

        if done:
            self.recent_rewards.append(self.ep_reward)
            self._all_rewards.append(self.ep_reward)
            if len(self.recent_rewards) > 20:
                self.recent_rewards.pop(0)
            if self.ep_reward > self.best_ep_reward:
                self.best_ep_reward = self.ep_reward
            self.ep_reward = 0.0
            self.ep_len = 0
            self.episode_idx += 1
            self.obs, _ = self.env.reset(seed=self.cfg.seed + self.episode_idx)

    def _do_update(self):
        try:
            stats = self.agent.update(self.obs)
            self.latest_loss = stats.get("loss")
        except Exception as e:
            print(f"[trainer] update skipped: {e}")
        self.steps_since_update = 0
        self.update_idx += 1

    # ------------------------------------------------------------------

    def _render_frame(self):
        avg = (sum(self.recent_rewards) / len(self.recent_rewards)) if self.recent_rewards else 0.0
        best = self.best_ep_reward if self.best_ep_reward != -float("inf") else 0.0
        grade = score_to_flycontrol_grade(best) if self.best_ep_reward != -float("inf") else "—"
        minutes = (time.monotonic() - self._start_time) / 60.0
        metrics = [
            ("update",  f"{self.update_idx}/{self.cfg.total_updates}", None),
            ("buffer",  f"{self.steps_since_update}/{self.cfg.steps_per_update}", None),
            ("episode", str(self.episode_idx), None),
            ("ep step", str(self.ep_len), None),
            ("ep R",    f"{self.ep_reward:+.1f}", None),
            ("avg R",   f"{avg:+.1f}", None),
            ("best R",  f"{best:+.1f}"
                        if self.best_ep_reward != -float("inf") else "—", None),
            ("grade",   grade, None),
            ("time",    f"{minutes:.1f} min", None),
        ]
        if self.latest_loss is not None:
            metrics.append(("loss", f"{self.latest_loss:.3f}", None))

        subtitle = self.stage_def["subtitle"]
        if self.base_model_name:
            subtitle = f"{subtitle}   •   base: {self.base_model_name}"
        else:
            # No base — label as a test run so the user can tell at a
            # glance this isn't a real curriculum step.
            subtitle = f"{subtitle}   •   base: fresh (TEST)"
        title_prefix = "TEST TRAINING" if (self.cfg.test_run or not self.base_model_name) else "TRAINING"
        hud = {
            "title":    f"{title_prefix} — {self.stage_def['title']}",
            "subtitle": subtitle,
            "metrics":  metrics,
        }
        self.renderer.draw_scene(
            state=self.env.physics.state,
            target=self.env.target,
            path=None,
            world=self.env.world,
            trail=self.env.position_history,
            waypoints=self.env.waypoints if self.env.waypoints else None,
            hud=hud,
        )


def run_trainer(stage: str, **kwargs):
    cfg = TrainConfig(stage=stage, **kwargs)
    TrainerUI(cfg).run()
