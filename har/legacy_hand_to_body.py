"""Evaluación hand_to_body con la heurística HAR de pasillo (pre pocket v2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from har_actions.detect import _features, _hand_to_body
from har_actions.pose import smooth


@dataclass(frozen=True)
class LegacyHandToBodyResult:
    hand_to_body: bool
    """Score 0–1: duración del gesto + evidencia far/close (legacy v2)."""
    confidence: float
    n_frames: int
    duration_s: float
    frame_start: int
    frame_end: int
    detail: str


def _legacy_geometry_strength(features: dict) -> float:
    fps = features["fps"]
    n = features["n"]
    far = np.zeros(n, dtype=bool)
    close = np.zeros(n, dtype=bool)
    for arm in features["arms"].values():
        use = arm["ok"] & np.isfinite(arm["dist_axis"]) & np.isfinite(arm["dist_hip"])
        far |= use & (arm["dist_axis"] >= 0.50)
        close |= use & (arm["dist_hip"] <= 0.25)
    if far.sum() < 1.0 * fps or close.sum() < 2.0 * fps:
        return 0.0
    return min(1.0, 0.45 * float(far.mean()) + 0.55 * float(close.mean()))


def legacy_hand_to_body_confidence(
    *,
    hand_to_body: bool,
    duration_s: float,
    clip_s: float,
    geometry_strength: float,
) -> float:
    if not hand_to_body:
        return 0.0
    dur_part = min(1.0, duration_s / max(clip_s * 0.28, 0.28))
    return round(min(1.0, 0.55 * dur_part + 0.45 * geometry_strength), 4)


def evaluate_legacy(poses: np.ndarray, *, fps: float, aspect: float) -> LegacyHandToBodyResult:
    """Misma lógica que `detect()` con backend legacy (poses suavizadas, _hand_to_body)."""
    smoothed = smooth(poses, window=3)
    features = _features(smoothed, fps, aspect)
    actions = _hand_to_body(features)
    n = len(poses)
    clip_s = n / max(fps, 1e-6)
    geom = _legacy_geometry_strength(features)
    if not actions:
        return LegacyHandToBodyResult(
            hand_to_body=False,
            confidence=0.0,
            n_frames=n,
            duration_s=0.0,
            frame_start=-1,
            frame_end=-1,
            detail="",
        )
    best = max(actions, key=lambda a: a.frame_end - a.frame_start)
    dur = (best.frame_end - best.frame_start + 1) / max(fps, 1e-6)
    conf = legacy_hand_to_body_confidence(
        hand_to_body=True,
        duration_s=dur,
        clip_s=clip_s,
        geometry_strength=geom,
    )
    return LegacyHandToBodyResult(
        hand_to_body=True,
        confidence=conf,
        n_frames=n,
        duration_s=dur,
        frame_start=best.frame_start,
        frame_end=best.frame_end,
        detail=best.detail,
    )