"""Storage visual inspector — stream the Layer-6 log stress test.

Shows the four stages of the storage benchmark (round-trip write/read,
malformed-JSON tolerance, truncation tolerance, per-drone isolation)
as a live pipeline with pass/fail status per check. The user can see
the log grow and which assertions tripped without reading stdout.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from drone_ai.grading import (
    RunLogger, RunRecord, generate_model_name, next_version,
    score_to_universal_grade,
)
from drone_ai.modules.storage import (
    MissionOutcome, MissionRecord, Storage, UpdateRecord, UpstreamCause,
)
from drone_ai.modules.storage.train import (
    _corrupt_tail, _inject_malformed, _write_synthetic,
)
from drone_ai.viz.inspector_structure import Arrow, Box, StructureInspector


STAGES = [
    "write synthetic",
    "round-trip read",
    "inject malformed",
    "truncate tail",
    "drone isolation",
]


class StorageInspector(StructureInspector):
    def __init__(self, n_missions: int = 60, seed: int = 42,
                 save_dir: str = "models/storage", run_tag: str = ""):
        super().__init__(
            title="Storage — Layer 6 field log stress",
            subtitle="Writes, corrupts, and re-reads to prove the reader survives.",
            total_trials=len(STAGES),
            autoplay_hz=1.2,
        )
        self.n_missions = n_missions
        self.seed = seed
        self.save_dir = save_dir
        self.run_tag = run_tag
        self.rng = np.random.default_rng(seed)

        self.tmp = tempfile.mkdtemp(prefix="drone_storage_ui_")
        self.storage = Storage("bench", root=self.tmp)
        self.active_stage = 0
        self.checks: List[Tuple[str, bool]] = []
        self.miss_w = self.upd_w = self.deliv_w = self.crash_w = 0
        self.removed = 0

        self._final_grade: Optional[str] = None
        self._final_score: float = 0.0
        self._saved_path: Optional[str] = None

    def setup(self) -> None:
        self.push_event(f"scratch dir: {self.tmp}", "dim")

    def structure_diagram(self) -> Tuple[List[Box], List[Arrow]]:
        boxes: List[Box] = []
        n = len(STAGES)
        for i, label in enumerate(STAGES):
            hi = (i == min(self.active_stage, n - 1)) and not self.finished
            status = ""
            # Find pass/fail state from self.checks by prefix.
            prefix = label.split(" ")[0]
            for name, ok in self.checks:
                if name.startswith(prefix) or prefix in name:
                    status = "ok" if ok else "bad"
                    break
            boxes.append(Box(label, x=(i / max(1, n - 1)),
                             y=0.5, highlight=hi, status=status))
        arrows = [Arrow(i, i + 1) for i in range(n - 1)]
        return boxes, arrows

    def current_thinking(self) -> List[Tuple[str, str]]:
        label = STAGES[min(self.active_stage, len(STAGES) - 1)]
        passed = sum(1 for _, ok in self.checks if ok)
        return [
            ("stage",          label),
            ("missions written", str(self.miss_w)),
            ("updates written",  str(self.upd_w)),
            ("bytes trimmed",    str(self.removed)),
            ("checks",           f"{passed}/{len(self.checks)}"),
        ]

    def sidebar_lines(self) -> List[Tuple[str, str, str]]:
        ok_count = sum(1 for _, ok in self.checks if ok)
        return [
            ("missions/run",  str(self.n_missions), "text"),
            ("checks passed", f"{ok_count}/{len(self.checks)}",
                              "ok" if ok_count == len(self.checks) else "warn"),
            ("",              "", "dim"),
            *[
                (name, "ok" if ok else "FAIL", "ok" if ok else "bad")
                for name, ok in self.checks[-6:]
            ],
        ]

    def step(self) -> bool:
        try:
            if self.active_stage == 0:
                self.miss_w, self.upd_w, self.deliv_w, self.crash_w = _write_synthetic(
                    self.storage, self.n_missions, self.rng,
                )
                self.push_event(
                    f"wrote {self.miss_w} missions, {self.upd_w} updates", "ok",
                )
            elif self.active_stage == 1:
                s = self.storage.summary()
                self._check("round_trip_missions", s["missions_total"] == self.miss_w)
                self._check("round_trip_updates",  s["updates_total"] == self.upd_w)
                self._check("round_trip_delivered",
                            s["missions_by_outcome"]["delivered"] == self.deliv_w)
                self._check("round_trip_crashed",
                            s["missions_by_outcome"]["crashed"] == self.crash_w)
            elif self.active_stage == 2:
                _inject_malformed(self.storage.path, n=5)
                try:
                    s2 = self.storage.summary()
                    ok = (s2["missions_total"] >= self.miss_w - 1
                          and s2["updates_total"] >= self.upd_w - 1)
                    self._check("malformed_tolerated", ok)
                except Exception as e:
                    self._check(f"malformed_tolerated[{type(e).__name__}]", False)
            elif self.active_stage == 3:
                self.removed = _corrupt_tail(self.storage.path, fraction=0.05)
                try:
                    s3 = self.storage.summary()
                    loss = 1.0 - (s3["missions_total"] / max(self.miss_w, 1))
                    self._check(
                        f"truncation_loss_under_15pct(loss={loss:.2%})",
                        loss < 0.15,
                    )
                except Exception as e:
                    self._check(f"truncation_crash[{type(e).__name__}]", False)
            elif self.active_stage == 4:
                other = Storage("other", root=self.tmp)
                other.record_mission(MissionRecord(
                    mission_id="solo", outcome=MissionOutcome.DELIVERED,
                    deadline_type="SOFT", mission_class="STANDARD",
                ))
                ok = (other.summary()["missions_total"] == 1
                      and self.storage.summary()["missions_total"] > 0)
                self._check("drone_isolation", ok)
        except Exception as e:
            self.push_event(f"stage {self.active_stage} crashed: {e}", "bad")

        self.active_stage += 1
        self.trial_idx = self.active_stage
        if self.active_stage >= len(STAGES):
            self._finalize()
            return False
        return True

    def _check(self, name: str, ok: bool) -> None:
        self.checks.append((name, ok))
        self.push_event(f"[{'ok' if ok else 'FAIL'}] {name}", "ok" if ok else "bad")

    def _finalize(self) -> None:
        import shutil
        try:
            shutil.rmtree(self.tmp, ignore_errors=True)
        except Exception:
            pass
        passed = sum(1 for _, ok in self.checks if ok)
        total = len(self.checks) or 1
        score = (passed / total) * 800.0
        grade = score_to_universal_grade(score)
        self._final_grade = grade
        self._final_score = score

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        version = next_version(self.save_dir, "storage")
        fname = generate_model_name(grade, "storage", version)
        out = Path(self.save_dir) / fname
        metrics = {
            "checks_passed": passed, "checks_total": total,
            "check_details": [{"name": n, "ok": ok} for n, ok in self.checks],
            "missions_written": self.miss_w,
            "updates_written": self.upd_w,
            "tail_bytes_trimmed": self.removed,
        }
        torch.save({"grade": grade, "score": score, "metrics": metrics}, str(out))
        with open(out.with_suffix(".json"), "w") as f:
            json.dump({
                "grade": grade, "score": score, "metrics": metrics,
                "timestamp": datetime.now().isoformat(), "model_file": fname,
            }, f, indent=2, default=str)
        self._saved_path = str(out)
        try:
            RunLogger().append(RunRecord(
                module="storage", stage="stress",
                best_score=score, avg_score=score, grade=grade,
                minutes=(time.monotonic() - self._start_t) / 60.0,
                episodes=self.n_missions, run_tag=self.run_tag,
            ))
        except Exception as e:
            self.push_event(f"run-log append failed: {e}", "warn")

    def final_summary(self) -> List[Tuple[str, str]]:
        if self._final_grade is None:
            return [("Run did not finish.", "warn")]
        passed = sum(1 for _, ok in self.checks if ok)
        return [
            (f"Grade: {self._final_grade}   Score: {self._final_score:.1f}", "accent"),
            (f"Checks: {passed}/{len(self.checks)}", "text"),
            (f"Saved: {self._saved_path}", "dim"),
        ]


def run_storage_inspector(n_missions: int = 60, seed: int = 42,
                          save_dir: str = "models/storage",
                          run_tag: str = "") -> Tuple[str, float]:
    ui = StorageInspector(n_missions, seed, save_dir, run_tag)
    ui.run()
    return ui._final_grade or "W", ui._final_score
