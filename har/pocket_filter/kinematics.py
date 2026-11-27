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
    near_dwell_s: float = 0.0
    swing_peaks: float = 0.0
    approach_depth: float = 0.0

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


def _swing_peak_count(d: np.ndarray, valid: np.ndarray, *, min_prominence: float = 0.04) -> int:
    """Picos locales en distancia muñeca–bolsillo (braceo / oscilación)."""
    peaks = 0
    n = len(d)
    for i in range(1, n - 1):
        if not (valid[i] and valid[i - 1] and valid[i + 1]):
            continue
        if not (np.isfinite(d[i - 1 : i + 2]).all()):
            continue
        if d[i] > d[i - 1] and d[i] > d[i + 1]:
            trough = min(d[i - 1], d[i + 1])
            if d[i] - trough >= min_prominence:
                peaks += 1
    return peaks


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


def _median_frame_dt(
    fps: float,
    timestamps: np.ndarray | None,
) -> float:
    dt = 1.0 / max(fps, 1e-6)
    if timestamps is not None and len(timestamps) > 1:
        diffs = np.diff(timestamps.astype(np.float64))
        diffs = diffs[diffs > 0]
        if len(diffs):
            dt = float(np.median(diffs))
    return dt


def _hip_speed_norm_series(
    poses: np.ndarray,
    eval_mask: np.ndarray,
    *,
    dt: float,
) -> float:
    hip_sp: list[float] = []
    t = len(poses)
    for i in range(1, t):
        if not eval_mask[i] or not eval_mask[i - 1]:
            continue
        hip_l = poses[i, HIP_L]
        hip_r = poses[i, HIP_R]
        hip_l0 = poses[i - 1, HIP_L]
        hip_r0 = poses[i - 1, HIP_R]
        if not (np.isfinite(hip_l).all() and np.isfinite(hip_r).all()):
            continue
        if not (np.isfinite(hip_l0).all() and np.isfinite(hip_r0).all()):
            continue
        hip_c = (hip_l + hip_r) * 0.5
        hip_c0 = (hip_l0 + hip_r0) * 0.5
        sh_l, sh_r = poses[i, SHOULDER_L], poses[i, SHOULDER_R]
        if not (np.isfinite(sh_l).all() and np.isfinite(sh_r).all()):
            ln = 1.0
        else:
            ln = float(np.linalg.norm(sh_l - sh_r))
            if ln < 1e-6:
                ln = 1.0
        hip_sp.append(float(np.linalg.norm(hip_c - hip_c0) / ln) / dt)
    if not hip_sp:
        return 0.0
    return float(np.median(hip_sp))


def build_movement_summary(
    kin: ClipKinematics,
    *,
    walk_frac: float = 0.0,
    stop_frac: float = 0.0,
    hand_to_body_raw: bool = False,
    hand_to_body: bool = False,
    hand_to_body_confidence: float = 0.0,
) -> dict[str, float | str | bool]:
    """Señales de ritmo para C3 (heurísticas auditables, no umbral de robo)."""
    hip = float(kin.hip_speed_norm)
    wrist_min = float(kin.wrist_radial_vel_min)
    approach_speed = float(max(0.0, -wrist_min))

    if hip <= 0.22 or stop_frac >= 0.45:
        body_pace = "quiet"
    elif hip <= 0.52:
        body_pace = "walking"
    else:
        body_pace = "fast"

    if approach_speed <= 1e-6:
        approach_pace = "none"
    elif approach_speed < 0.35:
        approach_pace = "slow"
    elif approach_speed < 0.85:
        approach_pace = "moderate"
    else:
        approach_pace = "snap"

    has_gesture = hand_to_body_raw or hand_to_body
    if has_gesture and hand_to_body_confidence >= 0.55:
        if body_pace == "fast" or approach_pace == "snap":
            tempo_hint = "hurried"
        elif (
            body_pace == "quiet"
            and (approach_pace in ("slow", "moderate", "none") or kin.near_dwell_s >= 0.35)
        ):
            tempo_hint = "natural"
        else:
            tempo_hint = "neutral"
    elif body_pace == "fast" or (walk_frac >= 0.35 and hip > 0.45):
        tempo_hint = "hurried"
    elif body_pace == "quiet" and hip < 0.18:
        tempo_hint = "natural"
    else:
        tempo_hint = "neutral"

    return {
        "hip_speed_norm": round(hip, 4),
        "wrist_approach_vel_min": round(wrist_min, 4),
        "wrist_approach_speed": round(approach_speed, 4),
        "near_dwell_s": round(float(kin.near_dwell_s), 4),
        "approach_monotone_frac": round(float(kin.approach_monotone_frac), 4),
        "swing_peaks": round(float(kin.swing_peaks), 4),
        "approach_depth": round(float(kin.approach_depth), 4),
        "walk_frac": round(float(walk_frac), 4),
        "stop_frac": round(float(stop_frac), 4),
        "body_pace": body_pace,
        "approach_pace": approach_pace,
        "tempo_hint": tempo_hint,
        "has_pocket_pair_kin": bool(approach_speed > 0 or kin.near_dwell_s > 0),
    }


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

    t = len(poses)
    exportable = frame_exportable_mask(t, row_mask)
    eval_start, eval_end = result.eval_start, result.eval_end
    eval_mask = np.zeros(t, dtype=bool)
    eval_mask[eval_start : eval_end + 1] = True
    dt = _median_frame_dt(fps, timestamps)
    out.hip_speed_norm = _hip_speed_norm_series(poses, eval_mask, dt=dt)
    if pair is None:
        return out

    wi, pi = _wrist_pocket_ix(pair)
    d, valid, l_norm, hip_c = _dist_series(
        poses, wi, pi, aspect, frame_height, cfg, eval_mask, exportable, keypoint_conf
    )

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

    dwell = 0.0
    for i in range(t):
        if valid[i] and np.isfinite(d[i]) and d[i] <= cfg.r_near:
            dwell += dt
    out.near_dwell_s = dwell
    out.swing_peaks = float(_swing_peak_count(d, valid))

    if np.any(valid & np.isfinite(d)):
        d_valid = d[valid & np.isfinite(d)]
        out.approach_depth = float(np.nanmax(d_valid) - np.nanmin(d_valid))

    if np.any(valid & np.isfinite(d)):
        t_peak = int(np.nanargmin(np.where(valid, d, np.inf)))
    else:
        t_peak = eval_end
    out.elbow_angle_deg = _elbow_angle_deg(poses, t_peak, wi, aspect)
    out.wrist_axis_dist_min = _wrist_axis_dist(poses, t_peak, wi, aspect)
    return out