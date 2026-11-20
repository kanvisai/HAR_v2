"""Validez temporal: valid_mask fila, confianza, coords congeladas."""
from __future__ import annotations

import numpy as np

WRIST_L, WRIST_R = 4, 5
HIP_L, HIP_R = 6, 7
SHOULDER_L, SHOULDER_R = 0, 1


def frame_exportable_mask(
    n: int,
    row_mask: np.ndarray | None,
) -> np.ndarray:
    """True si el frame se exportó / es evaluable (valid_mask Kanvis (T,))."""
    if row_mask is None:
        return np.ones(n, dtype=bool)
    m = np.asarray(row_mask, dtype=bool)
    if m.shape == (n,):
        return m
    if m.ndim == 2 and m.shape[0] == n:
        return m.any(axis=1)
    return np.ones(n, dtype=bool)


def wrist_confidence_mask(
    conf: np.ndarray | None,
    wrist_ix: int,
    min_conf: float = 0.50,
) -> np.ndarray | None:
    if conf is None or conf.ndim != 2 or conf.shape[1] < 8:
        return None
    return np.isfinite(conf[:, wrist_ix]) & (conf[:, wrist_ix] >= min_conf)


def frozen_wrist_mask(
    poses: np.ndarray,
    wrist_ix: int,
    *,
    repeat_frames: int,
    torso_eps: float,
    aspect: float,
) -> np.ndarray:
    """Frames con muñeca repetida exacta mientras el torso se mueve → no evidencia."""
    t = len(poses)
    out = np.zeros(t, dtype=bool)
    if t < repeat_frames:
        return out

    def torso_center(i: int) -> tuple[float, float] | None:
        p = poses[i]
        if not all(np.isfinite(p[k]).all() for k in (SHOULDER_L, SHOULDER_R, HIP_L, HIP_R)):
            return None
        w = p[SHOULDER_L].copy()
        w[0] *= aspect
        ww = p[SHOULDER_R].copy()
        ww[0] *= aspect
        h = p[HIP_L].copy()
        h[0] *= aspect
        hh = p[HIP_R].copy()
        hh[0] *= aspect
        s = 0.5 * (w + ww)
        hp = 0.5 * (h + hh)
        return float(s[0]), float(s[1]), float(hp[0]), float(hp[1])

    streak = 1
    prev_w = None
    prev_torso = None
    for i in range(t):
        w = poses[i, wrist_ix].copy()
        w[0] *= aspect
        if not np.isfinite(w).all():
            streak = 1
            prev_w = None
            continue
        torso = torso_center(i)
        if prev_w is not None and np.allclose(w, prev_w, atol=1e-8, rtol=0):
            streak += 1
            if (
                streak >= repeat_frames
                and torso is not None
                and prev_torso is not None
            ):
                move = max(
                    abs(torso[0] - prev_torso[0]),
                    abs(torso[1] - prev_torso[1]),
                    abs(torso[2] - prev_torso[2]),
                    abs(torso[3] - prev_torso[3]),
                )
                if move > torso_eps:
                    out[i] = True
        else:
            streak = 1
        prev_w = w
        prev_torso = torso
    return out