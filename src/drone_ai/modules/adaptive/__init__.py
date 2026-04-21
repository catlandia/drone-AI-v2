"""Adaptive — optional 5th AI layer for online / active learning.

The drone can be deployed WITHOUT this module (pure inference on the
frozen trained policy) or WITH this module (continues to fine-tune the
policy from its own field experience). Fully offline: no remote servers,
no satellites. Grades like any other module.
"""

from drone_ai.modules.adaptive.learner import (
    AdaptiveConfig,
    AdaptiveLearner,
    AdaptiveMetrics,
)

__all__ = ["AdaptiveConfig", "AdaptiveLearner", "AdaptiveMetrics"]
