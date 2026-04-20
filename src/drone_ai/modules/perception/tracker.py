"""Multi-object tracker using Kalman filter.

Associates new detections to existing tracks using nearest-neighbor matching.
Each track maintains a Kalman filter for position/velocity estimation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from drone_ai.modules.perception.detector import Detection


@dataclass
class Track:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    confidence: float = 1.0
    age: int = 0
    hits: int = 1
    misses: int = 0

    # Kalman state: [x, y, z, vx, vy, vz]
    x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    P: np.ndarray = field(default_factory=lambda: np.eye(6) * 10.0)


class ObjectTracker:
    """Simple Kalman-filter multi-object tracker."""

    def __init__(
        self,
        dt: float = 0.02,
        max_misses: int = 10,
        match_distance: float = 5.0,
    ):
        self.dt = dt
        self.max_misses = max_misses
        self.match_distance = match_distance
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0

        # Kalman matrices
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        self.Q = np.eye(6) * 0.1   # process noise
        self.R = np.eye(3) * 1.0   # measurement noise

    def update(self, detections: List[Detection]) -> List[Track]:
        # Predict all tracks
        for track in self._tracks.values():
            track.x = self.F @ track.x
            track.P = self.F @ track.P @ self.F.T + self.Q
            track.age += 1

        # Match detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._match(detections)

        # Update matched tracks
        for det_idx, track_id in matched:
            det = detections[det_idx]
            track = self._tracks[track_id]
            z = det.position
            S = self.H @ track.P @ self.H.T + self.R
            K = track.P @ self.H.T @ np.linalg.inv(S)
            track.x += K @ (z - self.H @ track.x)
            track.P = (np.eye(6) - K @ self.H) @ track.P
            track.position = track.x[:3].copy()
            track.velocity = track.x[3:].copy()
            track.hits += 1
            track.misses = 0
            track.confidence = det.confidence

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            tid = self._next_id
            self._next_id += 1
            x = np.zeros(6)
            x[:3] = det.position
            track = Track(track_id=tid, position=det.position.copy(), x=x)
            self._tracks[tid] = track

        # Mark misses, remove dead tracks
        for tid in unmatched_tracks:
            self._tracks[tid].misses += 1
        dead = [tid for tid, t in self._tracks.items() if t.misses > self.max_misses]
        for tid in dead:
            del self._tracks[tid]

        # Return confirmed tracks
        return [t for t in self._tracks.values() if t.hits >= 2]

    def _match(self, detections: List[Detection]):
        if not self._tracks or not detections:
            return [], list(range(len(detections))), list(self._tracks.keys())

        track_ids = list(self._tracks.keys())
        track_positions = np.array([self._tracks[tid].x[:3] for tid in track_ids])
        det_positions = np.array([d.position for d in detections])

        # Cost matrix
        cost = np.linalg.norm(
            det_positions[:, None, :] - track_positions[None, :, :], axis=2
        )

        matched = []
        used_dets = set()
        used_tracks = set()

        for det_i, track_j in sorted(np.ndindex(*cost.shape), key=lambda x: cost[x]):
            if det_i in used_dets or track_j in used_tracks:
                continue
            if cost[det_i, track_j] < self.match_distance:
                matched.append((det_i, track_ids[track_j]))
                used_dets.add(det_i)
                used_tracks.add(track_j)

        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_tracks = [track_ids[j] for j in range(len(track_ids)) if j not in used_tracks]
        return matched, unmatched_dets, unmatched_tracks

    def get_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.hits >= 2]

    def clear(self):
        self._tracks.clear()
        self._next_id = 0
