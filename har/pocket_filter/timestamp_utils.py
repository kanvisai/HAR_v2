"""Validación y normalización de pose_frame_timestamps para pocket."""
from __future__ import annotations

import numpy as np

# Timestamps Unix ~1e9; umbrales de clip en segundos razonables.
_ABS_EPOCH_MIN = 1.0e4


def prepare_frame_timestamps(
    timestamps: np.ndarray | None,
    n_frames: int,
    fps: float,
) -> tuple[np.ndarray | None, str | None]:
    """
    Devuelve timestamps listos para pocket (t=0 al inicio del chunk) o None → usar fps.

    ``reason`` solo si se descartan (para logs); None si OK o si entrada era None.
    """
    if timestamps is None:
        return None, None
    if n_frames < 1:
        return None, "empty_clip"
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(ts) != n_frames:
        return None, "length_mismatch"

    if not np.all(np.isfinite(ts)):
        return None, "non_finite"

    if np.any(np.diff(ts) < -1e-9):
        return None, "non_monotonic"

    ts = ts.copy()
    if float(ts[-1]) > _ABS_EPOCH_MIN or float(ts[0]) > _ABS_EPOCH_MIN:
        ts -= float(ts[0])

    span = float(ts[-1] - ts[0]) if n_frames > 1 else 0.0
    expected = (n_frames - 1) / max(float(fps), 1e-6)
    if expected > 0:
        ratio = span / expected
        if ratio < 0.35 or ratio > 3.5:
            return None, "span_vs_fps"

    if n_frames >= 2:
        diffs = np.diff(ts)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return None, "zero_dt"
        med = float(np.median(diffs))
        if med < 1e-5 or med > 2.0:
            return None, "dt_out_of_range"
        if med > 0 and (float(np.max(diffs)) / med > 8.0):
            return None, "dt_spike"

    return ts, None