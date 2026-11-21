"""Features cinemáticas del esqueleto (P1-1): velocidades, ángulos, monotonía de aproximación."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pocket_filter.config import PocketConfig
from pocket_filter.evaluate import (
    HIP_L,
    HIP_R,
    WRIST_L,
    WRIST_R,
    _dist_series,
    slice_indices,
)
from pocket_filter.masks import frame_exportable_mask
from pocket_filter.types import PocketResult, PairEvidence

SHOULDER_L, SHOULDER_R = 0, 1
ELBOW_L, ELBOW_R = 2, 3


@dataclass
class ClipKinematics:
    """Resumen auditable para batch/CSV; sin red neuronal."""

    wrist_radial_vel_min: float = 0.0
    approach_monotone_frac: float = 0.0
    hip_speed_norm: float = 0.0
    elbow_angle_deg: float = 0.0
    wrist_axis_dist_min: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


def _wrist_pocket_ix(pair: PairEvidence) -> tuple[int, int]:
    wi = WRIST_L if pair.wrist == "left" else WRIST_R
    pi = HIP_L if pair.pocket == "left" else HIP_R
    return wi, pi


def _elbow_angle_deg(poses: np.ndarray, t: int, wrist_ix: int, aspect: float) -> float:
    if wrist_ix == WRIST_L:
        s, e, w = SHOULDER_L, ELBOW_L, WRIST_L
    else:
        s, e, w = SHOULDER_R, ELBOW_R, WRIST_R
    if t < 0 or t >= len(poses):
        return 0.0
    p = poses[t]
    if not np.isfinite(p[s:e + 1]).all() or not np.isfinite(p[w]).all():
        return 0.0
    sh, el, wr = p[s].copy(), p[e].copy(), p[w].copy()
    for v in (sh, el, wr):
        v[0] *= aspect
    u = sh - el
    v = wr - el
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return 0.0
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _wrist_axis_dist(poses: np.ndarray, t: int, wrist_ix: int, aspect: float) -> float:
    if t < 0 or t >= len(poses):
        return 0.0
    s_ix = SHOULDER_L if wrist_ix == WRIST_L else SHOULDER_R
    h_ix = HIP_L if wrist_ix == WRIST_L else HIP_R
    p = poses[t]
    if not np.isfinite(p[s_ix:h_ix + 1]).all() or not np.isfinite(p[wrist_ix]).all():
        return 1.0
    sh, hi, wr = p[s_ix].copy(), p[h_ix].copy(), p[wrist_ix].copy()
    for v in (sh, hi, wr):
        v[0] *= aspect
    axis = hi - sh
    ln = float(np.linalg.norm(axis))
    if ln < 1e-8:
        return 1.0
    axis = axis / ln
    rel = wr - sh
    perp = rel - axis * float(np.dot(rel, axis))
    return float(np.linalg.norm(perp) / ln)


def compute_clip_kinematics(
    poses: np.ndarray,
    *,
    result: PocketResult,
    fps: float,
    aspect: float,
    frame_height: float,
    cfg: PocketConfig,
    timestamps: np.ndarray | None,
    row_mask: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
) -> ClipKinematics:
    out = ClipKinematics()
    pair = result.best_pair
    if pair is None:
        return out

    t = len(poses)
    exportable = frame_exportable_mask(t, row_mask)
    eval_start, eval_end = result.eval_start, result.eval_end
    eval_mask = np.zeros(t, dtype=bool)
    eval_mask[eval_start : eval_end + 1] = True

    wi, pi = _wrist_pocket_ix(pair)
    d, valid, l_norm, hip_c = _dist_series(
        poses, wi, pi, aspect, frame_height, cfg, eval_mask, exportable, keypoint_conf
    )

    dt = 1.0 / max(fps, 1e-6)
    if timestamps is not None and len(timestamps) > 1:
        diffs = np.diff(timestamps.astype(np.float64))
        diffs = diffs[diffs > 0]
        if len(diffs):
            dt = float(np.median(diffs))

    vel = np.full(t, np.nan)
    for i in range(1, t):
        if valid[i] and valid[i - 1] and np.isfinite(d[i]) and np.isfinite(d[i - 1]):
            vel[i] = (d[i] - d[i - 1]) / dt
    if np.any(np.isfinite(vel)):
        out.wrist_radial_vel_min = float(np.nanmin(vel))

    dec_steps = 0
    total_steps = 0
    for i in range(1, t):
        if valid[i] and valid[i - 1] and np.isfinite(d[i]) and np.isfinite(d[i - 1]):
            total_steps += 1
            if d[i] < d[i - 1]:
                dec_steps += 1
    if total_steps:
        out.approach_monotone_frac = dec_steps / total_steps

    hip_sp = []
    for i in range(1, t):
        if not eval_mask[i] or not eval_mask[i - 1]:
            continue
        if np.isfinite(hip_c[i]).all() and np.isfinite(hip_c[i - 1]).all():
            ln = l_norm[i] if np.isfinite(l_norm[i]) and l_norm[i] > 1e-6 else 1.0
            hip_sp.append(float(np.linalg.norm(hip_c[i] - hip_c[i - 1]) / ln) / dt)
    if hip_sp:
        out.hip_speed_norm = float(np.median(hip_sp))

    if np.any(valid & np.isfinite(d)):
        t_peak = int(np.nanargmin(np.where(valid, d, np.inf)))
    else:
        t_peak = eval_end
    out.elbow_angle_deg = _elbow_angle_deg(poses, t_peak, wi, aspect)
    out.wrist_axis_dist_min = _wrist_axis_dist(poses, t_peak, wi, aspect)
    return out