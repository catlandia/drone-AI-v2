"""Personality — Layer 7. Transferable delta artifact.

See docs/modules/personality.md. A Personality is the weight deltas
+ accepted hparams + soft-bound promotions of a proven drone,
packaged for A/B transfer to an experimental cohort.
"""

from drone_ai.modules.personality.artifact import (
    Personality,
    export_personality,
    apply_personality,
    select_best_drone,
)

__all__ = [
    "Personality",
    "export_personality",
    "apply_personality",
    "select_best_drone",
]
