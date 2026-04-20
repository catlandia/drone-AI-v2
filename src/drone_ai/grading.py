"""Unified grading system for all drone AI modules.

Model naming convention: {Grade} {DD-MM-YYYY} {module} v{N}.pt
Example: A+ 20-04-2026 flycontrol v1.pt
"""

import re
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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
