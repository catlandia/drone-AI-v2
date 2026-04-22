"""Adaptive — Layer 5. Online learning of all layers including itself.

Guarded by a frozen warden, a 20-episode rolling rollback, and a
soft-bound registry that requires N successful sim recoveries before
Layer 5 may push past a soft limit. Hard limits (ground-impact,
reward structure, goals, module toggles) are never touched.

See docs/modules/adaptive.md and PLAN.md §15.
"""

from drone_ai.modules.adaptive.learner import (
    AdaptiveConfig,
    AdaptiveLearner,
    AdaptiveMetrics,
)
from drone_ai.modules.adaptive.warden import Warden, WardenVerdict
from drone_ai.modules.adaptive.rollback import RollbackMonitor, RollbackDecision
from drone_ai.modules.adaptive.soft_bounds import (
    SoftBoundRegistry,
    BoundStatus,
    SOFT_BOUNDS,
    HARD_LIMITS,
)

__all__ = [
    "AdaptiveConfig",
    "AdaptiveLearner",
    "AdaptiveMetrics",
    "Warden",
    "WardenVerdict",
    "RollbackMonitor",
    "RollbackDecision",
    "SoftBoundRegistry",
    "BoundStatus",
    "SOFT_BOUNDS",
    "HARD_LIMITS",
]
