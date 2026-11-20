"""Evaluación hand_to_body con la heurística HAR de pasillo (pre pocket v2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from har_actions.detect import _features, _hand_to_body
from har_actions.pose import smooth


@dataclass(frozen=True)
class LegacyHandToBodyResult:
    hand_to_body: bool
    n_frames: int
    duration_s: float
    frame_start: int
    frame_end: int
    detail: str


def evaluate_legacy(poses: np.ndarray, *, fps: float, aspect: float) -> LegacyHandToBodyResult:
    """Misma lógica que `detect()` con backend legacy (poses suavizadas, _hand_to_body)."""
    smoothed = smooth(poses, window=3)
    features = _features(smoothed, fps, aspect)
    actions = _hand_to_body(features)
    n = len(poses)
    if not actions:
        return LegacyHandToBodyResult(
            hand_to_body=False,
            n_frames=n,
            duration_s=0.0,
            frame_start=-1,
            frame_end=-1,
            detail="",
        )
    best = max(actions, key=lambda a: a.frame_end - a.frame_start)
    dur = (best.frame_end - best.frame_start + 1) / max(fps, 1e-6)
    return LegacyHandToBodyResult(
        hand_to_body=True,
        n_frames=n,
        duration_s=dur,
        frame_start=best.frame_start,
        frame_end=best.frame_end,
        detail=best.detail,
    )