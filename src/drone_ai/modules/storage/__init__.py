"""Storage of Learnings — Layer 6.

Per-drone persistent log of every Layer 5 action during flight, plus
per-mission outcome rows. Uploaded to base station on docking via the
pre-flight handshake path. No mid-flight radio uploads.

See docs/modules/storage.md.
"""

from drone_ai.modules.storage.log import (
    Storage,
    UpdateRecord,
    MissionRecord,
    MissionOutcome,
    UpstreamCause,
)

__all__ = [
    "Storage",
    "UpdateRecord",
    "MissionRecord",
    "MissionOutcome",
    "UpstreamCause",
]
