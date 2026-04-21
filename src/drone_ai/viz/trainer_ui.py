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
    generate_model_name, next_version,
)
from drone_ai.modules.flycontrol.agent import PPOAgent, PPOConfig
from drone_ai.modules.flycontrol.environment import (
    FlyControlEnv, TaskType, OBS_DIM, ACT_DIM,
)
from drone_ai.viz.renderer3d import Renderer


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


@dataclass
class TrainConfig:
    stage: str = "hover"
    seed: int = 42
    steps_per_update: int = 512
    total_updates: int = 1000
    save_path: Optional[str] = None
    run_tag: str = ""                        # free-form tag for the run log
    log_path: str = "models/runs.csv"


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
        self.renderer = Renderer(title=f"Training — {self.stage_def['title']}")

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
        finally:
            # Always record the run, even if the user quit early — the
            # minutes and best-score so far are still meaningful data points.
            self._log_run()
            self.renderer.close()
        return self.update_idx >= self.cfg.total_updates

    def _auto_save_path(self) -> str:
        # Checkpoints live per-module: models/flycontrol/, models/perception/,
        # etc. The run log sits at the top (models/runs.csv) and covers all
        # modules, so we have to join in the module folder explicitly.
        models_root = os.path.dirname(self.cfg.log_path) or "models"
        module = "flycontrol"
        module_dir = os.path.join(models_root, module)
        os.makedirs(module_dir, exist_ok=True)
        best = self.best_ep_reward if self.best_ep_reward != -float("inf") else 0.0
        grade = score_to_flycontrol_grade(best)
        version = next_version(module_dir, module)
        return os.path.join(module_dir, generate_model_name(grade, module, version))

    def _log_run(self) -> None:
        minutes = (time.monotonic() - self._start_time) / 60.0
        avg = float(np.mean(self._all_rewards)) if self._all_rewards else 0.0
        best = self.best_ep_reward if self.best_ep_reward != -float("inf") else 0.0
        grade = score_to_flycontrol_grade(best)
        rec = RunRecord(
            module="flycontrol",
            stage=self.cfg.stage,
            best_score=best,
            avg_score=avg,
            grade=grade,
            minutes=minutes,
            updates=self.update_idx,
            episodes=self.episode_idx,
            run_tag=self.cfg.run_tag,
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

        hud = {
            "title":    f"TRAINING — {self.stage_def['title']}",
            "subtitle": self.stage_def["subtitle"],
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
