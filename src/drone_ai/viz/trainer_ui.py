"""Live 3D training UI — runs PPO training on a chosen task while rendering
the current policy's behavior in a 3D window.

Every render frame we step the env with the current policy; every N steps
a PPO update runs. Rewards and episode stats accumulate into the HUD.

When `TrainConfig.population > 1`, N drones train in parallel inside the
same window. BC warm-up runs once on drone 0; the rest are cloned from
that warmed-up policy and mutated so they diverge. Camera follows the
current leader (highest recent-mean reward). At the end, the best-of-
population by consistency score is saved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import os

from drone_ai.grading import (
    RunLogger, RunRecord, score_to_flycontrol_grade,
    generate_model_name, next_version, parse_model_name,
    consistency_score,
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
    # Behavior-cloning warm-up before PPO kicks in. Uses the PD
    # controller as the imitation teacher so the actor has a sane
    # starting policy instead of tumbling the drone on random noise.
    # PLAN.md §8 + §18 explicitly calls for this. Defaults on for
    # fresh (no-warm-start) runs; skipped when loading a real ckpt
    # since we don't want to wipe learned weights.
    bc_warmup: bool = True
    bc_episodes: int = 6
    bc_epochs: int = 60
    # Population-based training. N drones train in parallel inside the
    # same window — same selection-pressure idea as the CLI population
    # trainer (drone_ai.modules.flycontrol.train), with live 3D viz.
    # population == 1 reproduces the original single-drone behavior.
    population: int = 1
    # Mutation applied to clones 1..N-1 after seeding from drone 0, so
    # the population starts diverse instead of N identical copies. Same
    # defaults as PPOAgent.mutate.
    mutate_noise_std: float = 0.05
    mutate_prob: float = 0.1
    # Periodic evolution within a single run. Every `evolve_every`
    # per-drone updates (i.e. when *every* drone has done at least
    # this many PPO updates since the last evolution), rank by
    # consistency score, replace the bottom half with mutated clones
    # of the top half. 0 disables mid-run evolution — population just
    # trains in parallel and the best-of-N is saved at the end. Only
    # has effect when population > 1.
    #
    # Bumped from 25 to 50: at 25 each drone only had ~8-10 episodes
    # between culls, and ranking on a 20-episode recent window with
    # such few samples meant fitness was dominated by single-episode
    # luck. Selection oscillated, the leader kept changing, and grades
    # bounced around the noise floor instead of climbing.
    evolve_every: int = 50
    # Per-drone PPO buffer-size jitter. Each drone gets
    # `steps_per_update + i * stagger_steps` so their PPO updates fire
    # on different ticks instead of all clumping on the same frame.
    # Removes the visible stutter when N drones all finish their buffer
    # at once. 0 disables the offset.
    stagger_steps: int = 8


# ---- Per-drone state -------------------------------------------------------

@dataclass
class _DroneSlot:
    """One drone in the population — its env, agent, and per-drone counters."""
    env: FlyControlEnv
    agent: PPOAgent
    obs: np.ndarray
    # Per-drone PPO buffer size. Set from cfg.steps_per_update plus a
    # per-index stagger offset so each drone's update fires on a
    # different tick — see TrainConfig.stagger_steps.
    n_steps: int = 512
    ep_reward: float = 0.0
    ep_len: int = 0
    episode_idx: int = 0
    update_idx: int = 0
    steps_since_update: int = 0
    best_ep_reward: float = float("-inf")
    recent_rewards: List[float] = field(default_factory=list)
    all_rewards: List[float] = field(default_factory=list)
    latest_loss: Optional[float] = None
    # Per-drone update count at the most recent evolution event. Used
    # to gate the next evolution: we only evolve once *every* drone has
    # done at least `evolve_every` updates since this checkpoint.
    last_evolve_update: int = 0


# Distinct hues for up to 12 drones in the HUD / peer markers. Leader is
# always drawn with the renderer's normal drone colors; peers borrow these.
PEER_COLORS: List[Tuple[int, int, int]] = [
    (120, 220, 150), (140, 200, 230), (240, 170, 90),
    (220, 140, 210), (240, 100, 110), (200, 180, 100),
    (160, 200, 100), (100, 200, 200), (220, 110, 220),
    (180, 130, 100), (130, 180, 220), (220, 200, 120),
]


class TrainerUI:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.stage_def = STAGE_DEFS.get(cfg.stage, STAGE_DEFS["hover"])
        n = max(1, int(cfg.population))
        # Build N envs (each drone gets its own physics instance) and N
        # agents. Drone 0 is the seed; others are cloned + mutated from
        # it after BC warm-up so the population starts diverse.
        envs = [
            FlyControlEnv(
                task=self.stage_def["task"],
                difficulty=self.stage_def["difficulty"],
                domain_randomization=self.stage_def["domain_rand"],
                seed=cfg.seed + i,
            )
            for i in range(n)
        ]
        seed_agent = PPOAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM, config=PPOConfig())
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
                seed_agent.load(warm)
                self.base_model_name = os.path.basename(warm)
                print(f"[trainer] warm-started {cfg.stage} from {warm}")
            except Exception as e:
                print(f"[trainer] warm-start skipped ({warm}): {e}")
        prefix = "Test Training" if cfg.test_run else "Training"
        title_suffix = f" × {n}" if n > 1 else ""
        self.renderer = Renderer(
            title=f"{prefix} — {self.stage_def['title']}{title_suffix}"
        )
        # Device reminder — when this prints "cpu" the user is running
        # the CPU-only torch wheel. Reinstall with the CUDA index URL
        # (see docs/quickstart.md) to get GPU acceleration.
        print(f"[trainer] device: {seed_agent.device}  ·  population: {n}")

        # Build the population. Drone 0 owns the seed agent; the rest
        # start as clones and get mutated *after* BC warm-up so they
        # inherit hover competence before diverging. Each drone gets
        # its own n_steps with a small stagger so their PPO updates
        # don't all fire on the same render frame.
        self.drones: List[_DroneSlot] = []
        for i in range(n):
            agent = seed_agent if i == 0 else seed_agent.clone()
            obs, _ = envs[i].reset(seed=cfg.seed + i)
            self.drones.append(_DroneSlot(
                env=envs[i],
                agent=agent,
                obs=obs,
                n_steps=cfg.steps_per_update + i * max(0, cfg.stagger_steps),
            ))
        self._leader_idx_cache = 0

        # Run-log bookkeeping: capture wall-clock so the log has minutes.
        self._start_time = time.monotonic()
        # BC warm-up status — exposed on the HUD so the user can tell
        # the actor is being imitation-trained before PPO kicks in.
        self._bc_status: Optional[str] = None
        self._bc_loss: Optional[float] = None
        # Only warm-up when there's no real base. Loading a trained
        # checkpoint and then MSE-pulling it back toward the PD action
        # would destroy what the previous stage learned.
        self._should_bc = (
            self.cfg.bc_warmup and self.base_model_name is None
        )
        # When BC won't run (warm-started from disk), diversify the
        # clones now so the population doesn't start as N identical
        # copies. With BC, diversification fires after warm-up.
        if not self._should_bc:
            self._diversify_population()

    # ---- Population helpers --------------------------------------------

    @property
    def population_size(self) -> int:
        return len(self.drones)

    def _diversify_population(self) -> None:
        """Replace clones 1..N-1 with mutated copies of drone 0 so the
        population starts with shared skill but distinct policies.

        Called after BC warm-up (so all drones inherit hover competence)
        or right after a warm-start checkpoint load (same idea — give
        every drone the base, then perturb to create selection signal)."""
        if self.population_size <= 1:
            return
        seed = self.drones[0].agent
        for i in range(1, self.population_size):
            self.drones[i].agent = seed.mutate(
                noise_std=self.cfg.mutate_noise_std,
                prob=self.cfg.mutate_prob,
            )

    def _leader_index(self) -> int:
        """Drone with the highest current fitness. Camera and HUD
        follow this index, recomputed every frame so the leader can
        change as the population evolves."""
        best_idx = 0
        best_score = float("-inf")
        for i, d in enumerate(self.drones):
            score = self._drone_fitness(d)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _total_updates_done(self) -> int:
        return sum(d.update_idx for d in self.drones)

    def _drone_fitness(self, d: _DroneSlot) -> float:
        """Current fitness for evolution ranking and leader selection.

        Uses the mean of `recent_rewards` (sliding window of the last 20
        episodes) — *not* the cumulative consistency score over the
        whole run. The cumulative version reward early luck and let a
        degrading champion stay on top forever, which broke mid-run
        evolution: the BC-warmed lineage kept being preserved while
        successful mutations were culled because they carried the
        failed episode history of whichever drone they replaced.
        """
        if d.recent_rewards:
            return sum(d.recent_rewards) / len(d.recent_rewards)
        if d.best_ep_reward != float("-inf"):
            return d.best_ep_reward
        return float("-inf")

    def _maybe_evolve(self) -> None:
        """Mid-run evolution. Once every drone has done at least
        `cfg.evolve_every` PPO updates since the last cull, rank by
        current fitness (recent-window mean) and replace the bottom
        half with mutated clones of the top half. Replaced drones get
        a complete fresh-slate reset — their predecessor's reward
        history shouldn't poison the new policy's selection signal.
        Their env keeps flying so the viz doesn't teleport.

        Skipped when population is too small to evolve (need at least 2
        survivors and 1 victim) or when evolve_every == 0.
        """
        n = self.population_size
        if n < 2 or self.cfg.evolve_every <= 0:
            return
        # Gate on the *minimum* per-drone update delta — only evolve
        # once every drone has had at least evolve_every updates of
        # selection-eligible training since the last event.
        deltas = [d.update_idx - d.last_evolve_update for d in self.drones]
        if min(deltas) < self.cfg.evolve_every:
            return

        ranked = sorted(
            range(n),
            key=lambda i: self._drone_fitness(self.drones[i]),
            reverse=True,
        )
        n_keep = max(1, n // 2)
        survivors = ranked[:n_keep]
        victims = ranked[n_keep:]

        for j, victim_idx in enumerate(victims):
            parent = self.drones[survivors[j % n_keep]].agent
            child = parent.mutate(
                noise_std=self.cfg.mutate_noise_std,
                prob=self.cfg.mutate_prob,
            )
            slot = self.drones[victim_idx]
            slot.agent = child
            # Full fresh-slate reset for the new policy. Without this,
            # the predecessor's failed episodes stayed in all_rewards
            # and best_ep_reward, so a great mutation got dragged down
            # by a history it didn't earn.
            slot.recent_rewards = []
            slot.all_rewards = []
            slot.best_ep_reward = float("-inf")
            slot.episode_idx = 0
            slot.steps_since_update = 0
            slot.latest_loss = None
            slot.last_evolve_update = slot.update_idx

        # Survivors also reset their evolution checkpoint so the next
        # gate measures from this generation onward.
        for survivor_idx in survivors:
            self.drones[survivor_idx].last_evolve_update = (
                self.drones[survivor_idx].update_idx
            )

        if survivors:
            top_fit = self._drone_fitness(self.drones[survivors[0]])
            print(
                f"[evolve] gen mark @ updates={self._total_updates_done()}: "
                f"survivors={survivors}, victims={victims}, "
                f"top recent={top_fit:+.1f}"
            )

    # Convenience properties for the rendering / results / log code that
    # used to read self.<x> directly. They report the *leader's* numbers
    # so single-drone runs look identical to before.
    @property
    def update_idx(self) -> int:
        return self._total_updates_done()

    @property
    def episode_idx(self) -> int:
        return sum(d.episode_idx for d in self.drones)

    # ------------------------------------------------------------------

    def run(self) -> bool:
        running = True
        finished_naturally = False
        save_path: Optional[str] = None
        try:
            # Step 0: Behavior-cloning warm-up (only for fresh starts).
            # In population mode, BC runs once on drone 0 and the rest
            # are diversified clones — see _run_bc_warmup.
            if self._should_bc:
                running = self._run_bc_warmup()
                # Rewind every drone's env so the first PPO step doesn't
                # inherit the PD rollout's final state.
                if running:
                    for i, d in enumerate(self.drones):
                        d.obs, _ = d.env.reset(seed=self.cfg.seed + i)

            # Outer loop: total_updates is the *sum* across the
            # population. A 6-drone run with the same total_updates
            # therefore takes ~the same wall time as a 1-drone run, but
            # spreads the work across multiple parallel policies and
            # picks the winner at the end.
            while running and self._total_updates_done() < self.cfg.total_updates:
                running = self.renderer.handle_events(1 / 60)
                if not running:
                    break
                if not self.renderer.paused:
                    for _ in range(self.renderer.sim_speed):
                        # Step every drone once per tick. Each drone
                        # owns its own env + buffer; updates fire as
                        # individual buffers fill. Per-drone n_steps
                        # is staggered so updates don't clump on the
                        # same frame and stutter the viz.
                        for d in self.drones:
                            self._collect_step(d)
                            if d.steps_since_update >= d.n_steps:
                                self._do_update(d)
                        # Mid-run evolution: once every drone has done
                        # `evolve_every` updates since the last cull,
                        # replace the bottom half with mutated clones
                        # of the top half. Skips when population <= 1
                        # or evolve_every == 0.
                        self._maybe_evolve()
                        if self._total_updates_done() >= self.cfg.total_updates:
                            break
                self._render_frame()
                self.renderer.flip()

            finished_naturally = self._total_updates_done() >= self.cfg.total_updates

            # Always persist the trained weights — previously the UI path
            # left save_path=None so training runs vanished on exit. Name
            # the file with the tier-list convention so runs.csv rows and
            # disk checkpoints share a grade + date + version. In
            # population mode, save the winner (best consistency score).
            save_path = self.cfg.save_path or self._auto_save_path()
            try:
                winner = self._winner_drone()
                winner.agent.save(save_path)
                print(f"[trainer] saved checkpoint -> {save_path}")
                if self.population_size > 1:
                    print(
                        f"[trainer] winner = drone {self.drones.index(winner)} "
                        f"of {self.population_size}"
                    )
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

    def _drone_stats(self, d: _DroneSlot) -> Dict[str, float]:
        avg = float(np.mean(d.all_rewards)) if d.all_rewards else 0.0
        std = float(np.std(d.all_rewards)) if len(d.all_rewards) > 1 else 0.0
        best = d.best_ep_reward if d.best_ep_reward != float("-inf") else avg
        overall = consistency_score(best, avg, std) if d.all_rewards else 0.0
        return {"avg": avg, "std": std, "best": best, "overall": overall}

    def _winner_drone(self) -> _DroneSlot:
        """Pick the population member to save. Ranks by current fitness
        (recent-window mean) so a drone that started strong but is now
        degraded doesn't win on the strength of stale episodes; ties
        broken by cumulative consistency score."""
        scored = sorted(
            self.drones,
            key=lambda d: (
                self._drone_fitness(d),
                self._drone_stats(d)["overall"],
            ),
            reverse=True,
        )
        return scored[0]

    def _show_results_screen(self, save_path: Optional[str]) -> None:
        """Block on a results panel until the user presses any key /
        clicks / closes the window. Lets them read the final grade
        before returning to the launcher."""
        from drone_ai.grading import GRADE_NAMES
        winner = self._winner_drone()
        st = self._drone_stats(winner)
        avg, std, best, overall = st["avg"], st["std"], st["best"], st["overall"]
        grade = (
            score_to_flycontrol_grade(overall) if winner.all_rewards else "—"
        )
        minutes = (time.monotonic() - self._start_time) / 60.0
        is_test = self.cfg.test_run or not self.base_model_name
        title = (
            f"TEST TRAINING — {self.stage_def['title']}"
            if is_test else f"FlyControl — {self.stage_def['title']}"
        )
        pop_line: List[Tuple[str, str]] = []
        if self.population_size > 1:
            others = [self._drone_stats(d)["overall"] for d in self.drones]
            pop_line = [(
                f"Population: {self.population_size} drones   "
                f"winner overall {overall:+.1f}   "
                f"others avg {float(np.mean(others)):+.1f} / "
                f"min {min(others):+.1f}",
                "dim",
            )]
        results_lines = [
            (title, "title"),
            (f"Stage: {self.cfg.stage}" +
             ("   ·   no base model — results are a training-machinery check" if is_test else ""),
             "dim"),
            *pop_line,
            ("", "dim"),
            (f"Grade:    {grade}  ({GRADE_NAMES.get(grade,'')})", "accent"),
            (f"Overall:  {overall:+.1f}   (consistency-weighted, drives the grade)", "text"),
            (f"Avg R:    {avg:+.1f}", "text"),
            (f"Std R:    {std:.1f}   (lower = more consistent)", "text"),
            (f"Best R:   {best:+.1f}   (tiebreaker only)", "dim"),
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
        grade = self._current_grade()
        version = next_version(stage_dir, "flycontrol")
        return os.path.join(stage_dir, generate_model_name(grade, "flycontrol", version))

    def _current_grade(self) -> str:
        """Grade from the consistency-weighted overall score of the
        current leader. Keeps a policy from earning an S-tier filename on
        the strength of one freak episode when the rest tumbled."""
        winner = self._winner_drone()
        if not winner.all_rewards:
            return score_to_flycontrol_grade(0.0)
        st = self._drone_stats(winner)
        return score_to_flycontrol_grade(st["overall"])

    def _log_run(self) -> None:
        minutes = (time.monotonic() - self._start_time) / 60.0
        winner = self._winner_drone()
        st = self._drone_stats(winner)
        avg, std, best, overall = st["avg"], st["std"], st["best"], st["overall"]
        grade = (
            score_to_flycontrol_grade(overall) if winner.all_rewards
            else score_to_flycontrol_grade(0.0)
        )
        # A fresh-base run (test_run or no base_model_name) is tagged
        # "test" so the runs.csv reader can filter it out when comparing
        # real curriculum steps. Population mode adds a "popN" tag so
        # the run log makes the parallel-evolution context obvious.
        is_test = self.cfg.test_run or not self.base_model_name
        tag = self.cfg.run_tag or ""
        if is_test and "test" not in tag.split(","):
            tag = f"{tag},test".lstrip(",")
        if self.population_size > 1:
            tag = f"{tag},pop{self.population_size}".lstrip(",")
        rec = RunRecord(
            module="flycontrol",
            stage=self.cfg.stage,
            best_score=best,
            avg_score=avg,
            std_score=std,
            overall_score=overall,
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

    def _run_bc_warmup(self) -> bool:
        """Collect PD-controller rollouts, SL-train the actor on them.

        Shows progress in the HUD so the user can tell what's
        happening (BC can take a few seconds and the drone is not
        moving on-screen while the SL epochs churn). Returns False if
        the user closed the window during warm-up.

        In population mode, BC runs once on drone 0 and the rest of the
        population is then re-cloned + mutated from the warmed-up agent
        so every drone inherits hover competence without paying the BC
        cost N times.
        """
        from drone_ai.modules.flycontrol.pd_controller import collect_pd_rollouts

        seed_drone = self.drones[0]

        # --- collect rollouts from the PD controller ---
        self._bc_status = "Collecting PD rollouts…"
        self._render_frame()
        self.renderer.flip()
        pd_obs, pd_acts, pd_rews, pd_dones = collect_pd_rollouts(
            seed_drone.env,
            n_episodes=self.cfg.bc_episodes,
            max_steps=1500,
            seed=self.cfg.seed,
        )
        print(f"[bc] collected {len(pd_obs)} (obs, act, r, done) pairs from "
              f"{self.cfg.bc_episodes} PD episodes")

        if len(pd_obs) == 0:
            self._bc_status = None
            self._diversify_population()
            return True

        # --- SL pretraining ---
        n_epochs = self.cfg.bc_epochs
        running = [True]

        def cb(epoch: int, loss: float) -> None:
            self._bc_loss = float(loss)
            self._bc_status = f"BC epoch {epoch}/{n_epochs}   loss={loss:.4f}"
            # Pump events so the window stays responsive. The renderer
            # returning False means the user closed the window.
            alive = self.renderer.handle_events(0.0)
            if not alive:
                running[0] = False
                raise KeyboardInterrupt("bc warmup cancelled by user")
            self._render_frame()
            self.renderer.flip()

        try:
            stats = seed_drone.agent.bc_warmup(
                pd_obs, pd_acts,
                n_epochs=n_epochs,
                batch_size=128,
                lr=1e-3,
                progress_cb=cb,
                rewards=pd_rews,
                dones=pd_dones,
            )
            print(
                f"[bc] done — actor loss {stats.get('loss', float('nan')):.4f}"
                f"   critic loss {stats.get('critic_loss', float('nan')):.4f}"
            )
        except KeyboardInterrupt:
            print("[bc] cancelled by user")
            return False
        finally:
            self._bc_status = None
        # Re-seed the population from the warmed-up drone 0 so peers
        # inherit hover competence, then mutate to diversify.
        self._diversify_population()
        return running[0]

    def _collect_step(self, d: _DroneSlot) -> None:
        action, info = d.agent.select_action(d.obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = d.env.step(action)
        done = bool(terminated or truncated)
        d.agent.store(d.obs, action, float(reward), info["value"], info["log_prob"], done)

        d.ep_reward += float(reward)
        d.ep_len += 1
        d.steps_since_update += 1
        d.obs = next_obs

        if done:
            d.recent_rewards.append(d.ep_reward)
            d.all_rewards.append(d.ep_reward)
            # 30-wide window keeps the recent-fitness signal stable
            # against single-episode outliers; raising from 20 cut
            # selection-noise oscillation in population mode.
            if len(d.recent_rewards) > 30:
                d.recent_rewards.pop(0)
            if d.ep_reward > d.best_ep_reward:
                d.best_ep_reward = d.ep_reward
            d.ep_reward = 0.0
            d.ep_len = 0
            d.episode_idx += 1
            # Per-drone seed offset keeps drones from synchronising onto
            # the same episode RNG even when their indices line up.
            d.obs, _ = d.env.reset(seed=self.cfg.seed + d.episode_idx * len(self.drones))

    def _do_update(self, d: _DroneSlot) -> None:
        try:
            stats = d.agent.update(d.obs)
            d.latest_loss = stats.get("loss")
        except Exception as e:
            print(f"[trainer] update skipped: {e}")
        d.steps_since_update = 0
        d.update_idx += 1

    # ------------------------------------------------------------------

    def _render_frame(self):
        leader_idx = self._leader_index()
        leader = self.drones[leader_idx]
        self._leader_idx_cache = leader_idx
        avg_recent = (
            sum(leader.recent_rewards) / len(leader.recent_rewards)
            if leader.recent_rewards else 0.0
        )
        st = self._drone_stats(leader)
        avg_all, std_all, best, overall = (
            st["avg"], st["std"], st["best"], st["overall"]
        )
        grade = (
            score_to_flycontrol_grade(overall) if leader.all_rewards else "—"
        )
        minutes = (time.monotonic() - self._start_time) / 60.0
        metrics = [
            ("update",  f"{self.update_idx}/{self.cfg.total_updates}", None),
            ("buffer",  f"{leader.steps_since_update}/{leader.n_steps}", None),
            ("episode", str(leader.episode_idx), None),
            ("ep step", str(leader.ep_len), None),
            ("ep R",    f"{leader.ep_reward:+.1f}", None),
            ("avg R",   f"{avg_recent:+.1f}", None),
            ("std R",   f"{std_all:.1f}", None),
            ("overall", f"{overall:+.1f}" if leader.all_rewards else "—", None),
            ("best R",  f"{best:+.1f}"
                        if leader.best_ep_reward != float("-inf") else "—", None),
            ("grade",   grade, None),
            ("time",    f"{minutes:.1f} min", None),
        ]
        if leader.latest_loss is not None:
            metrics.append(("loss", f"{leader.latest_loss:.3f}", None))
        if self.population_size > 1:
            others_overall = [self._drone_stats(d)["overall"] for d in self.drones]
            metrics.append((
                "pop", f"#{leader_idx+1}/{self.population_size}", None,
            ))
            metrics.append((
                "pop avg", f"{float(np.mean(others_overall)):+.1f}", None,
            ))

        subtitle = self.stage_def["subtitle"]
        if self.base_model_name:
            subtitle = f"{subtitle}   •   base: {self.base_model_name}"
        else:
            # No base — label as a test run so the user can tell at a
            # glance this isn't a real curriculum step.
            subtitle = f"{subtitle}   •   base: fresh (TEST)"
        if self.population_size > 1:
            subtitle = f"{subtitle}   •   pop: {self.population_size}"
        if self._bc_status is not None:
            subtitle = f"{subtitle}   •   {self._bc_status}"
        title_prefix = "TEST TRAINING" if (self.cfg.test_run or not self.base_model_name) else "TRAINING"
        hud = {
            "title":    f"{title_prefix} — {self.stage_def['title']}",
            "subtitle": subtitle,
            "metrics":  metrics,
        }
        # Peer markers: every non-leader drone is rendered as a small
        # colored dot at its current world position, so the user can
        # watch the population diverge / converge in real time.
        peers: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        if self.population_size > 1:
            for i, d in enumerate(self.drones):
                if i == leader_idx:
                    continue
                color = PEER_COLORS[i % len(PEER_COLORS)]
                peers.append((np.asarray(d.env.physics.state.position), color))
        self.renderer.draw_scene(
            state=leader.env.physics.state,
            target=leader.env.target,
            path=None,
            world=leader.env.world,
            trail=leader.env.position_history,
            waypoints=leader.env.waypoints if leader.env.waypoints else None,
            hud=hud,
            peer_drones=peers if peers else None,
        )


def run_trainer(stage: str, **kwargs):
    cfg = TrainConfig(stage=stage, **kwargs)
    TrainerUI(cfg).run()
