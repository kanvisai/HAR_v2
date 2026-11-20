"""
Evaluador determinista muñeca → bolsillo (pocket v2).
"""
from __future__ import annotations

import numpy as np

from pocket_filter.config import PocketConfig, load_pocket_config
from pocket_filter.fuse import fuse_pairs
from pocket_filter.masks import frame_exportable_mask, frozen_wrist_mask, wrist_confidence_mask
from pocket_filter.types import PairEvidence, PocketResult, Verdict

SHOULDER_L, SHOULDER_R = 0, 1
WRIST_L, WRIST_R = 4, 5
HIP_L, HIP_R = 6, 7

IPSI = {("left", "left"), ("right", "right")}


def min_valid_ratio(n_frames: int, fps: float, cfg: PocketConfig) -> float:
    dur = n_frames / max(fps, 1e-6)
    if dur <= cfg.short_clip_s:
        return cfg.min_valid_ratio_short_clip
    if dur >= cfg.long_clip_s:
        return cfg.min_valid_ratio_long_clip
    span = cfg.long_clip_s - cfg.short_clip_s
    if span <= 0:
        return cfg.min_valid_ratio_long_clip
    t = (dur - cfg.short_clip_s) / span
    return cfg.min_valid_ratio_short_clip + t * (
        cfg.min_valid_ratio_long_clip - cfg.min_valid_ratio_short_clip
    )


def slice_indices(
    n: int,
    fps: float,
    interval_s: tuple[float, float] | None,
    context_pad_s: float,
    timestamps: np.ndarray | None,
) -> tuple[int, int]:
    if interval_s is None:
        return 0, n - 1
    t0, t1 = interval_s

    def index_at(t: float) -> int:
        if timestamps is not None and len(timestamps) == n:
            return int(np.clip(np.searchsorted(timestamps, t, side="left"), 0, n - 1))
        return int(np.clip(round(t * fps), 0, n - 1))

    i0 = index_at(max(0.0, t0 - context_pad_s))
    i1 = index_at(t1 + context_pad_s)
    return i0, max(i0, i1)


def _frame_geometry(
    pose: np.ndarray, aspect: float, frame_height: float, cfg: PocketConfig
):
    p = pose.copy()
    for idx in (SHOULDER_L, SHOULDER_R, WRIST_L, WRIST_R, HIP_L, HIP_R):
        p[idx, 0] *= aspect

    needed = (SHOULDER_L, SHOULDER_R, HIP_L, HIP_R)
    if not all(np.isfinite(p[i]).all() for i in needed):
        return False, "MISSING_TORSO", np.nan, None, None, None, None, np.nan

    s = 0.5 * (p[SHOULDER_L] + p[SHOULDER_R])
    h = 0.5 * (p[HIP_L] + p[HIP_R])
    sh = h - s
    l_norm = float(np.linalg.norm(sh))
    if l_norm < 1e-4:
        return False, "TORSO_TOO_SMALL", np.nan, None, None, None, None, np.nan

    l_px = l_norm * frame_height
    if l_px < cfg.min_torso_px:
        return False, "TORSO_BELOW_MIN_PX", l_px, None, None, None, None, np.nan

    u = sh / l_norm
    width = float(np.linalg.norm(p[SHOULDER_L] - p[SHOULDER_R]))
    frontal = width / l_norm
    if frontal < cfg.side_view_ratio:
        return False, "SIDE_VIEW", l_px, s, h, u, None, frontal

    return True, "", l_px, s, h, u, (p[HIP_L], p[HIP_R]), frontal


def _pocket_center(
    hip: np.ndarray,
    u: np.ndarray,
    l_norm: float,
    shoulder_side: int,
    cfg: PocketConfig,
) -> np.ndarray:
    center = hip + cfg.pocket_offset * l_norm * u
    lateral = (cfg.pocket_lateral_offset * l_norm) * (-1.0 if shoulder_side == SHOULDER_L else 1.0)
    center[0] += lateral
    return center


def _dist_series(
    poses: np.ndarray,
    wrist_ix: int,
    pocket_ix: int,
    aspect: float,
    frame_height: float,
    cfg: PocketConfig,
    eval_mask: np.ndarray,
    exportable: np.ndarray,
    conf: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = len(poses)
    d = np.full(t, np.nan)
    valid = np.zeros(t, dtype=bool)
    l_norm = np.full(t, np.nan)
    hip_c = np.full((t, 2), np.nan)
    shoulder_side = SHOULDER_L if pocket_ix == HIP_L else SHOULDER_R
    frozen = frozen_wrist_mask(
        poses,
        wrist_ix,
        repeat_frames=cfg.frozen_repeat_frames,
        torso_eps=cfg.frozen_torso_eps,
        aspect=aspect,
    )
    conf_mask = wrist_confidence_mask(conf, wrist_ix)

    for i in range(t):
        if not eval_mask[i]:
            continue
        ok, reason, *_rest = _frame_geometry(poses[i], aspect, frame_height, cfg)
        if not ok:
            continue
        _, _, _, s, h, u, hips, _ = _frame_geometry(poses[i], aspect, frame_height, cfg)
        hip_l, hip_r = hips
        pocket_hip = hip_l if pocket_ix == HIP_L else hip_r
        wrist = poses[i, wrist_ix].copy()
        wrist[0] *= aspect
        if not np.isfinite(wrist).all():
            continue
        if frozen[i]:
            continue
        if conf_mask is not None and not conf_mask[i]:
            continue
        ln = float(np.linalg.norm(h - s))
        b = _pocket_center(pocket_hip, u, ln, shoulder_side, cfg)
        d[i] = float(np.linalg.norm(wrist - b) / ln)
        valid[i] = exportable[i]
        l_norm[i] = ln
        hip_c[i] = h

    return d, valid, l_norm, hip_c


def _nominal_period(timestamps: np.ndarray | None, fps: float) -> float:
    if timestamps is not None and len(timestamps) >= 2:
        diffs = np.diff(timestamps.astype(np.float64))
        diffs = diffs[diffs > 0]
        if len(diffs):
            return float(np.median(diffs))
    return 1.0 / max(fps, 1e-6)


def _time_at(index: int, timestamps: np.ndarray | None, fps: float) -> float:
    if timestamps is not None and 0 <= index < len(timestamps):
        return float(timestamps[index])
    return index / max(fps, 1e-6)


def _identity_break(i: int, hip_c, l_norm, period: float, cfg: PocketConfig) -> bool:
    if i <= 0:
        return False
    if not (np.isfinite(hip_c[i]).all() and np.isfinite(hip_c[i - 1]).all()):
        return False
    if not (np.isfinite(l_norm[i]) and np.isfinite(l_norm[i - 1])):
        return False
    jump = float(np.linalg.norm(hip_c[i] - hip_c[i - 1]))
    if jump > cfg.identity_hip_jump_L * l_norm[i] and period <= cfg.identity_jump_window_s:
        return True
    if l_norm[i - 1] > 1e-6:
        scale_chg = abs(l_norm[i] - l_norm[i - 1]) / l_norm[i - 1]
        if scale_chg > cfg.scale_jump_ratio:
            return True
    return False


def _origin_qualifies(d_origin: float, d_entry: float, cfg: PocketConfig) -> bool:
    drop = d_origin - d_entry
    if drop < cfg.delta_min:
        return False
    if d_origin >= cfg.r_far:
        return True
    soft_far = cfg.r_near + cfg.delta_min * 0.5
    return d_origin >= soft_far and drop >= cfg.delta_min


def _invalid_gap_frames(i: int, j: int, valid: np.ndarray, period: float, fps: float, cfg: PocketConfig) -> int:
    max_miss = max(2, int(round(cfg.max_time_gap_factor * period * fps)))
    miss = 0
    for k in range(i + 1, j):
        if not valid[k]:
            miss += 1
            if miss > max_miss:
                return miss
    return miss


def _time_continuous(i: int, j: int, valid: np.ndarray, period: float, cfg: PocketConfig) -> bool:
    max_miss = max(2, int(round(cfg.max_time_gap_factor * period * max(1.0, 1.0 / period))))
    miss = 0
    for k in range(i + 1, j + 1):
        if not valid[k]:
            miss += 1
            if miss > max_miss:
                return False
    return True


def _valid_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _evaluate_pair(
    poses: np.ndarray,
    wrist: str,
    pocket: str,
    *,
    fps: float,
    aspect: float,
    frame_height: float,
    cfg: PocketConfig,
    timestamps: np.ndarray | None,
    eval_mask: np.ndarray,
    exportable: np.ndarray,
    conf: np.ndarray | None,
    eval_start: int,
    eval_end: int,
) -> PairEvidence:
    wrist_ix = WRIST_L if wrist == "left" else WRIST_R
    pocket_ix = HIP_L if pocket == "left" else HIP_R
    ipsi = (wrist, pocket) in IPSI

    d, valid, l_norm, hip_c = _dist_series(
        poses, wrist_ix, pocket_ix, aspect, frame_height, cfg, eval_mask, exportable, conf
    )
    t = len(d)
    period = _nominal_period(timestamps, fps)
    eval_len = max(1, eval_end - eval_start + 1)
    eval_slice = valid[eval_start : eval_end + 1]
    valid_ratio = float(eval_slice.sum() / eval_len)
    min_ratio = min_valid_ratio(eval_len, fps, cfg)

    base = PairEvidence(
        wrist, pocket, "INDETERMINADO", "SIN_DATOS_VALIDOS", valid_ratio=valid_ratio, ipsilateral=ipsi  # type: ignore
    )
    if eval_slice.sum() == 0:
        return base

    for start, end in _valid_runs(valid):
        if end < eval_start or start > eval_end:
            continue
        start = max(start, eval_start)
        end = min(end, eval_end)

        for i in range(max(start + 1, eval_start + 1), min(end, eval_end) + 1):
            if not (valid[i] and valid[i - 1]):
                continue
            if d[i] > cfg.r_near or d[i - 1] > cfg.r_near:
                continue

            d_entry = float(d[i])
            win_start_t = _time_at(i, timestamps, fps) - cfg.approach_window_s
            win_end_t = _time_at(i, timestamps, fps) - cfg.approach_min_lead_s

            origin_idx = None
            origin_d = None
            for j in range(start, i):
                if not valid[j]:
                    continue
                t_j = _time_at(j, timestamps, fps)
                if t_j < win_start_t or t_j > win_end_t:
                    continue
                if not _origin_qualifies(float(d[j]), d_entry, cfg):
                    continue
                if _invalid_gap_frames(j, i, valid, period, fps, cfg) > max(
                    2, int(round(cfg.max_time_gap_factor * period * fps))
                ):
                    continue
                bad_id = False
                for k in range(j + 1, i + 1):
                    if _identity_break(k, hip_c, l_norm, period, cfg):
                        bad_id = True
                        break
                if bad_id:
                    continue
                if not _time_continuous(j, i, valid, period, cfg):
                    continue
                if origin_idx is None or d[j] > (origin_d or 0):
                    origin_idx = j
                    origin_d = float(d[j])

            if origin_idx is None:
                continue

            seg = d[start : end + 1]
            seg_v = valid[start : end + 1]
            d_min = float(np.nanmin(seg[seg_v])) if seg_v.any() else None
            t_ret = None
            for k in range(i, end + 1):
                if valid[k] and d[k] >= cfg.r_far:
                    t_ret = _time_at(k, timestamps, fps)
                    break

            return PairEvidence(
                wrist=wrist,  # type: ignore
                pocket=pocket,  # type: ignore
                verdict="COMPATIBLE",
                motivo="APROXIMACION_OBSERVADA",
                t_origen=_time_at(origin_idx, timestamps, fps),
                d_origen=origin_d,
                t_entrada=_time_at(i, timestamps, fps),
                d_entrada=d_entry,
                d_min=d_min,
                t_retirada=t_ret,
                valid_ratio=valid_ratio,
                approach_delta=(origin_d - d_entry) if origin_d is not None else None,
                ipsilateral=ipsi,
            )

    min_contact = max(cfg.contact_consecutive, int(round(cfg.sustained_contact_s * fps)))
    run_i = eval_start
    while run_i <= eval_end:
        if not (valid[run_i] and np.isfinite(d[run_i]) and d[run_i] <= cfg.r_near):
            run_i += 1
            continue
        run_j = run_i
        while (
            run_j <= eval_end
            and valid[run_j]
            and np.isfinite(d[run_j])
            and d[run_j] <= cfg.r_near
        ):
            run_j += 1
        if run_j - run_i >= min_contact:
            segment = d[run_i:run_j]
            d_min_run = float(np.min(segment))
            if d_min_run <= cfg.deep_contact_r:
                return PairEvidence(
                    wrist=wrist,  # type: ignore
                    pocket=pocket,  # type: ignore
                    verdict="COMPATIBLE",
                    motivo="CONTACTO_BOLSILLO",
                    t_entrada=_time_at(run_i, timestamps, fps),
                    d_entrada=float(d[run_i]),
                    d_min=d_min_run,
                    t_retirada=_time_at(run_j - 1, timestamps, fps),
                    valid_ratio=valid_ratio,
                    ipsilateral=ipsi,
                )
        run_i = run_j

    side_n = sum(
        1
        for i in range(eval_start, eval_end + 1)
        if _frame_geometry(poses[i], aspect, frame_height, cfg)[1] == "SIDE_VIEW"
    )
    if side_n > eval_len * 0.5:
        return PairEvidence(
            wrist, pocket, "INDETERMINADO", "VISTA_NO_APTA", valid_ratio=valid_ratio, ipsilateral=ipsi  # type: ignore
        )

    if valid_ratio < min_ratio:
        return PairEvidence(
            wrist, pocket, "INDETERMINADO", "EVIDENCIA_PARCIAL", valid_ratio=valid_ratio, ipsilateral=ipsi  # type: ignore
        )

    eval_valid = valid[eval_start : eval_end + 1]
    eval_d = d[eval_start : eval_end + 1]
    if eval_valid.sum() >= cfg.contact_consecutive:
        finite_d = eval_d[eval_valid & np.isfinite(eval_d)]
        if len(finite_d) and np.all(finite_d >= cfg.r_far):
            gap_frames = (~eval_valid).sum()
            if gap_frames == 0 or gap_frames / eval_len <= (1.0 - min_ratio):
                return PairEvidence(
                    wrist,
                    pocket,
                    "INCOMPATIBLE",
                    "LEJOS_TODO_EL_GESTO",
                    valid_ratio=valid_ratio,
                    ipsilateral=ipsi,  # type: ignore
                )

    return PairEvidence(
        wrist, pocket, "INDETERMINADO", "ZONA_GRIS", valid_ratio=valid_ratio, ipsilateral=ipsi  # type: ignore
    )


def evaluate_pocket(
    poses: np.ndarray,
    *,
    fps: float = 12.5,
    aspect: float = 16.0 / 9.0,
    frame_height: float = 1080.0,
    frame_width: float | None = None,
    config: PocketConfig | None = None,
    timestamps: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
    keypoint_conf: np.ndarray | None = None,
    interval_s: tuple[float, float] | None = None,
    context_pad_s: float = 1.0,
) -> PocketResult:
    if poses.ndim != 3 or poses.shape[1] != 8 or poses.shape[2] < 2:
        raise ValueError(f"Expected poses (T, 8, 2+), got {poses.shape}")
    cfg = config or load_pocket_config()
    if frame_width is not None and frame_height > 0:
        aspect = frame_width / frame_height

    t = len(poses)
    exportable = frame_exportable_mask(t, row_mask)
    eval_start, eval_end = slice_indices(t, fps, interval_s, context_pad_s, timestamps)
    eval_mask = np.zeros(t, dtype=bool)
    eval_mask[eval_start : eval_end + 1] = True

    pairs: list[PairEvidence] = []
    for wrist in ("left", "right"):
        for pocket in ("left", "right"):
            pairs.append(
                _evaluate_pair(
                    poses,
                    wrist,
                    pocket,
                    fps=fps,
                    aspect=aspect,
                    frame_height=frame_height,
                    cfg=cfg,
                    timestamps=timestamps,
                    eval_mask=eval_mask,
                    exportable=exportable,
                    conf=keypoint_conf,
                    eval_start=eval_start,
                    eval_end=eval_end,
                )
            )

    resultado, motivo, best = fuse_pairs(pairs, cfg)
    t0 = _time_at(eval_start, timestamps, fps)
    t1 = _time_at(eval_end, timestamps, fps)
    return PocketResult(
        resultado=resultado,
        hand_to_body=resultado == "COMPATIBLE",
        filter_fp_safe=resultado == "INCOMPATIBLE",
        motivo=motivo,
        version=cfg.version,
        fps=float(fps),
        n_frames=t,
        eval_start=eval_start,
        eval_end=eval_end,
        pairs=pairs,
        best_pair=best,
        intervalo_s=(t0, t1),
    )


# Re-export for backwards compatibility
from pocket_filter.types import pocket_result_to_dict  # noqa: E402

__all__ = [
    "PocketConfig",
    "evaluate_pocket",
    "pocket_result_to_dict",
    "load_pocket_config",
    "min_valid_ratio",
]