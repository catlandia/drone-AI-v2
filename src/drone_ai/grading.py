"""Unified grading system for all drone AI modules.

Model naming convention: {Grade} {DD-MM-YYYY} {module} v{N}.pt
Example: A+ 20-04-2026 flycontrol v1.pt

Every training or evaluation run should also be appended to the combined
run log (see RunLogger) so runs can be compared over time — the user cares
about how much each tweak actually moves the score.
"""

import csv
import re
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


GRADE_ORDER = [
    "P", "S+", "S", "S-", "A+", "A", "A-",
    "B+", "B", "B-", "C+", "C", "C-",
    "D+", "D", "D-", "F+", "F", "F-", "W"
]

GRADE_NAMES = {
    "P":  "PERFECT",
    "S+": "SUPREME+", "S": "SUPREME",   "S-": "SUPREME-",
    "A+": "ALPHA+",   "A": "ALPHA",     "A-": "ALPHA-",
    "B+": "BETTER+",  "B": "BETTER",    "B-": "BETTER-",
    "C+": "COOL+",    "C": "COOL",      "C-": "COOL-",
    "D+": "DELUSIONAL+", "D": "DELUSIONAL", "D-": "DELUSIONAL-",
    "F+": "FAILURE+", "F": "FAILURE",   "F-": "FAILURE-",
    "W":  "WORST",
}

GRADE_DESCRIPTIONS = {
    "P":  "Flawless — production ready",
    "S+": "Near perfect performance",
    "S":  "Outstanding results",
    "S-": "Excellent, almost supreme",
    "A+": "Top tier performer",
    "A":  "Dominant performance",
    "A-": "Strong performance",
    "B+": "Impressive, above average",
    "B":  "Solid performance",
    "B-": "Getting there",
    "C+": "Functional with issues",
    "C":  "Minimal viable",
    "C-": "Below expectations",
    "D+": "Poor but trying",
    "D":  "Very poor",
    "D-": "Barely functional",
    "F+": "Failed but tried",
    "F":  "Complete failure",
    "F-": "Spectacular failure",
    "W":  "Does not work at all",
}


@dataclass
class FlyControlMetrics:
    hover_score: float       # avg reward on hover task
    delivery_score: float    # avg reward on delivery task
    route_score: float       # avg reward on delivery_route task
    deploy_score: float      # avg reward on deployment_ready task


@dataclass
class PathfinderMetrics:
    path_optimality: float   # actual_length / optimal_length (lower is better, 1.0 = perfect)
    avoidance_rate: float    # fraction of runs with no collision (0-1)
    planning_ms: float       # average planning time in milliseconds


@dataclass
class PerceptionMetrics:
    detection_accuracy: float  # % of obstacles detected (0-100)
    false_positive_rate: float # % of false detections (0-100)
    position_error: float      # avg position error in meters
    fps: float                 # frames per second


@dataclass
class ManagerMetrics:
    completion_rate: float     # fraction of deliveries completed (0-1)
    distance_efficiency: float # optimal_distance / actual_distance (0-1)
    priority_score: float      # fraction of high-priority done first (0-1)
    battery_waste: float       # wasted battery fraction (0-1, lower is better)


@dataclass
class AdaptiveMetricsGrading:
    """Grade the Adaptive module by how much it improves a policy in the
    field. Measured as the delta between frozen and adapted rewards on
    an out-of-distribution environment."""
    baseline_score: float      # avg episode reward with adaptation OFF
    adapted_score: float       # avg episode reward with adaptation ON
    recovery_rate: float       # (adapted - baseline) / max(|baseline|, 1)
    stability: float           # 1 - stddev(adapted) / (|mean|+1), clipped 0..1


def score_flycontrol(metrics: FlyControlMetrics) -> float:
    weights = {"hover": 0.15, "delivery": 0.25, "route": 0.30, "deploy": 0.30}
    return (
        metrics.hover_score * weights["hover"] +
        metrics.delivery_score * weights["delivery"] +
        metrics.route_score * weights["route"] +
        metrics.deploy_score * weights["deploy"]
    )


def score_pathfinder(metrics: PathfinderMetrics) -> float:
    optimality_score = max(0.0, (2.0 - metrics.path_optimality) / 1.0) * 300
    avoidance_score = metrics.avoidance_rate * 400
    speed_score = max(0.0, (500.0 - metrics.planning_ms) / 500.0) * 100
    return optimality_score + avoidance_score + speed_score


def score_perception(metrics: PerceptionMetrics) -> float:
    accuracy_score = metrics.detection_accuracy * 4.0
    fp_penalty = metrics.false_positive_rate * 2.0
    error_penalty = min(metrics.position_error * 20.0, 200.0)
    speed_score = min(metrics.fps * 5.0, 100.0)
    return accuracy_score - fp_penalty - error_penalty + speed_score


def score_manager(metrics: ManagerMetrics) -> float:
    completion = metrics.completion_rate * 400
    efficiency = metrics.distance_efficiency * 200
    priority = metrics.priority_score * 150
    battery = (1.0 - metrics.battery_waste) * 50
    return completion + efficiency + priority + battery


def score_adaptive(metrics: AdaptiveMetricsGrading) -> float:
    """Adaptive score = how much it recovered performance on OOD tasks,
    plus a stability bonus so a wildly-oscillating learner doesn't grade
    well just because its average recovered."""
    recovery = max(0.0, metrics.recovery_rate) * 500.0
    delta = max(0.0, metrics.adapted_score - metrics.baseline_score) * 0.5
    stability = max(0.0, min(1.0, metrics.stability)) * 100.0
    return recovery + delta + stability


_FLYCONTROL_THRESHOLDS = [
    ("P",  800), ("S+", 700), ("S",  600), ("S-", 550),
    ("A+", 500), ("A",  450), ("A-", 400),
    ("B+", 350), ("B",  300), ("B-", 250),
    ("C+", 200), ("C",  150), ("C-", 100),
    ("D+",  75), ("D",   50), ("D-",  25),
    ("F+",  10), ("F",    0), ("F-", -50),
]

_UNIVERSAL_THRESHOLDS = [
    ("P",  760), ("S+", 665), ("S",  570), ("S-", 522),
    ("A+", 475), ("A",  428), ("A-", 380),
    ("B+", 333), ("B",  285), ("B-", 238),
    ("C+", 190), ("C",  143), ("C-",  95),
    ("D+",  71), ("D",   48), ("D-",  24),
    ("F+",   9), ("F",    0), ("F-", -50),
]


def _score_to_grade(score: float, thresholds: list) -> str:
    for grade, threshold in thresholds:
        if score >= threshold:
            return grade
    return "W"


# Consistency weighting — how strongly the grade should punish a policy
# that only nails it once in a thousand runs. A score that is great
# once and bad the other 999 times is WORSE than a score that is
# mediocre every single time, because deployment cares about the
# expected per-mission outcome, not the outlier best.
#
# overall = AVG_WEIGHT * avg + BEST_WEIGHT * capped_best − STD_PENALTY * std
#
#   - AVG_WEIGHT = 0.9        → average dominates
#   - BEST_WEIGHT = 0.1       → best is a tiny tiebreaker only
#   - best is capped at avg + BEST_CAP so one lucky run can't shove the
#     grade up; if best is already close to avg, the cap is a no-op
#   - STD_PENALTY * std       → high variance is itself a red flag
#     (usually means the policy is exploiting the reward surface on
#     specific seeds and failing on others)
CONSISTENCY_AVG_WEIGHT = 0.9
CONSISTENCY_BEST_WEIGHT = 0.1
CONSISTENCY_BEST_CAP = 50.0
CONSISTENCY_STD_PENALTY = 0.5


def consistency_score(best: float, avg: float, std: float = 0.0) -> float:
    """Consistency-weighted overall score used to derive the letter grade.

    Prefers policies that score OK every time over policies that score
    great one-in-a-thousand — the latter are almost always reward-hacking
    a single easy seed. See the constants above for the weights.

    Per-episode `std` is optional; if omitted (e.g. a one-shot benchmark
    like Pathfinder where best == avg by construction), the variance
    penalty collapses to zero and the function just returns the mixed
    best/avg.
    """
    capped_best = min(best, avg + CONSISTENCY_BEST_CAP)
    mixed = CONSISTENCY_AVG_WEIGHT * avg + CONSISTENCY_BEST_WEIGHT * capped_best
    return mixed - CONSISTENCY_STD_PENALTY * max(0.0, std)


class ModelGrader:
    """Evaluates and grades any drone AI module."""

    def grade_flycontrol(self, metrics: FlyControlMetrics) -> Tuple[str, float]:
        score = score_flycontrol(metrics)
        return _score_to_grade(score, _FLYCONTROL_THRESHOLDS), score

    def grade_pathfinder(self, metrics: PathfinderMetrics) -> Tuple[str, float]:
        score = score_pathfinder(metrics)
        return _score_to_grade(score, _UNIVERSAL_THRESHOLDS), score

    def grade_perception(self, metrics: PerceptionMetrics) -> Tuple[str, float]:
        score = score_perception(metrics)
        return _score_to_grade(score, _UNIVERSAL_THRESHOLDS), score

    def grade_manager(self, metrics: ManagerMetrics) -> Tuple[str, float]:
        score = score_manager(metrics)
        return _score_to_grade(score, _UNIVERSAL_THRESHOLDS), score

    def grade_adaptive(self, metrics: AdaptiveMetricsGrading) -> Tuple[str, float]:
        score = score_adaptive(metrics)
        return _score_to_grade(score, _UNIVERSAL_THRESHOLDS), score

    def report(self, module: str, grade: str, score: float) -> str:
        name = GRADE_NAMES.get(grade, grade)
        desc = GRADE_DESCRIPTIONS.get(grade, "")
        return (
            f"\n{'='*50}\n"
            f"  {module.upper()} GRADE REPORT\n"
            f"{'='*50}\n"
            f"  Grade: {grade} — {name}\n"
            f"  Score: {score:.1f}\n"
            f"  {desc}\n"
            f"{'='*50}\n"
        )

    @staticmethod
    def compare(g1: str, g2: str) -> int:
        """Returns 1 if g1 better, -1 if worse, 0 if equal."""
        try:
            i1, i2 = GRADE_ORDER.index(g1), GRADE_ORDER.index(g2)
            return 0 if i1 == i2 else (1 if i1 < i2 else -1)
        except ValueError:
            return 0


def generate_model_name(grade: str, module: str, version: int, date: Optional[datetime] = None) -> str:
    """Generate: {Grade} {DD-MM-YYYY} {module} v{N}.pt"""
    d = date or datetime.now()
    return f"{grade} {d.strftime('%d-%m-%Y')} {module} v{version}.pt"


def parse_model_name(filename: str) -> Optional[Dict]:
    """Parse model filename into components."""
    pattern = r"^([PSABCDFW][+-]?) (\d{2}-\d{2}-\d{4}) (\w+) v(\d+)\.pt$"
    m = re.match(pattern, filename)
    if not m:
        return None
    try:
        date = datetime.strptime(m.group(2), "%d-%m-%Y")
    except ValueError:
        return None
    return {"grade": m.group(1), "date": date, "module": m.group(3), "version": int(m.group(4))}


def next_version(model_dir: str, module: str) -> int:
    """Get next available version number for a module."""
    max_v = 0
    if os.path.isdir(model_dir):
        for f in os.listdir(model_dir):
            parsed = parse_model_name(f)
            if parsed and parsed["module"] == module:
                max_v = max(max_v, parsed["version"])
    return max_v + 1


def grade_index(grade: str) -> int:
    """Lower index = better grade."""
    try:
        return GRADE_ORDER.index(grade)
    except ValueError:
        return len(GRADE_ORDER)


# ---- Run log --------------------------------------------------------------
# Single append-only CSV that combines every field the user asked for:
# date, minutes trained, numeric score, letter tier, stage, module, run tag.
# The file sits at models/runs.csv so it's easy to diff between tweaks.

RUN_LOG_FIELDS = [
    "timestamp_iso",
    "date",           # DD-MM-YYYY (matches model filename convention)
    "minutes",        # wall-clock training duration, float with one decimal
    "module",         # flycontrol / pathfinder / perception / manager / adaptive
    "stage",          # hover / delivery / ... / eval
    "best_score",     # best numeric score observed in the run
    "avg_score",      # average score across the run
    "std_score",      # stddev of episode scores (0 for single-shot benchmarks)
    "overall_score",  # consistency-weighted score actually used for the grade
    "grade",          # letter tier (P..W) derived from overall_score
    "updates",        # number of PPO/grad updates (or 0 for pure eval)
    "episodes",       # episodes completed
    "run_tag",        # free-form tag for seed/config distinction
]


@dataclass
class RunRecord:
    module: str
    stage: str
    best_score: float
    avg_score: float
    grade: str
    minutes: float
    updates: int = 0
    episodes: int = 0
    run_tag: str = ""
    timestamp: Optional[datetime] = None
    std_score: float = 0.0
    overall_score: Optional[float] = None  # None → computed from best/avg/std

    def to_row(self) -> Dict[str, str]:
        ts = self.timestamp or datetime.now()
        overall = (
            self.overall_score
            if self.overall_score is not None
            else consistency_score(self.best_score, self.avg_score, self.std_score)
        )
        return {
            "timestamp_iso": ts.isoformat(timespec="seconds"),
            "date":          ts.strftime("%d-%m-%Y"),
            "minutes":       f"{self.minutes:.1f}",
            "module":        self.module,
            "stage":          self.stage,
            "best_score":    f"{self.best_score:.2f}",
            "avg_score":     f"{self.avg_score:.2f}",
            "std_score":     f"{self.std_score:.2f}",
            "overall_score": f"{overall:.2f}",
            "grade":         self.grade,
            "updates":       str(self.updates),
            "episodes":      str(self.episodes),
            "run_tag":       self.run_tag,
        }


class RunLogger:
    """Append training/eval runs to a single CSV file.

    Kept deliberately simple: one row per run, one file for the whole
    project. The user wants to compare runs across days and configs.
    """

    def __init__(self, path: str = "models/runs.csv"):
        self.path = path

    def append(self, record: RunRecord) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._migrate_header_if_needed()
        exists = os.path.isfile(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS)
            if not exists:
                w.writeheader()
            w.writerow(record.to_row())

    def _migrate_header_if_needed(self) -> None:
        """If runs.csv exists with an older header (pre-consistency
        scoring), rewrite it in-place with the new columns, filling
        missing cells with empty strings. Keeps historical rows visible
        so the user can still see trend deltas after the grading change.
        """
        if not os.path.isfile(self.path):
            return
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return
            if header == RUN_LOG_FIELDS:
                return
            rows = list(csv.DictReader(open(self.path, "r", newline="", encoding="utf-8")))
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in RUN_LOG_FIELDS})

    def read_all(self) -> List[Dict[str, str]]:
        if not os.path.isfile(self.path):
            return []
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


def score_to_universal_grade(score: float) -> str:
    """Convenience: map a numeric score to a letter grade using the
    universal thresholds (non-flycontrol modules AND as a live HUD
    display during flycontrol training before a proper eval runs)."""
    return _score_to_grade(score, _UNIVERSAL_THRESHOLDS)


def score_to_flycontrol_grade(score: float) -> str:
    return _score_to_grade(score, _FLYCONTROL_THRESHOLDS)
