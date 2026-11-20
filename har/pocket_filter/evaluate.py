"""
Evaluador determinista muñeca → bolsillo (spec Ignacio v1).

Poses: (T, 8, 2) normalizadas, y hacia abajo. Orden COCO 5–12.
Sin suavizado en la decisión (usar poses originales).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

SHOULDER_L, SHOULDER_R = 0, 1
WRIST_L, WRIST_R = 4, 5
HIP_L, HIP_R = 6, 7

WristId = Literal["left", "right"]
PocketId = Literal["left", "right"]
Verdict = Literal["COMPATIBLE", "INCOMPATIBLE", "INDETERMINADO"]


@dataclass
class PocketConfig:
    pocket_offset: float = 0.08
    r_near: float = 0.30
    r_far: float = 0.55
    delta_min: float = 0.20
    approach_window_s: float = 1.50
    approach_min_lead_s: float = 0.10
    contact_consecutive: int = 2
    min_torso_px: float = 40.0
    side_view_ratio: float = 0.25
    identity_hip_jump_L: float = 0.75
    identity_jump_window_s: float = 0.15
    scale_jump_ratio: float = 0.40
    max_time_gap_factor: float = 1.5
    version: str = "ignacio-obs-1"


@dataclass
class PairEvidence:
    wrist: WristId
    pocket: PocketId
    verdict: Verdict
    motivo: str
    t_origen: float | None = None
    d_origen: float | None = None
    t_entrada: float | None = None
    d_entrada: float | None = None
    d_min: float | None = None
    t_retirada: float | None = None
    valid_ratio: float = 0.0


@dataclass
class PocketResult:
    resultado: Verdict
    hand_to_body: bool
    motivo: str
    version: str
    fps: float
    n_frames: int
    pairs: list[PairEvidence] = field(default_factory=list)
    intervalo_s: tuple[float, float] = (0.0, 0.0)
    legacy_hand_to_body: bool | None = None

    @property
    def accion_propuesta(self) -> str:
        if self.resultado == "COMPATIBLE":
            return "CONSERVAR"
        if self.resultado == "INCOMPATIBLE":
            return "CERRAR_HIPOTESIS_BOLSILLO"
        return "ABSTENERSE"


def pocket_result_to_dict(result: PocketResult) -> dict[str, Any]:
    data = asdict(result)
    data["accion_propuesta"] = result.accion_propuesta
    return data


def _scale_x(poses: np.ndarray, aspect: float) -> np.ndarray:
    out = np.array(poses, dtype=np.float64, copy=True)
    out[:, :, 0] *= aspect
    return out


def _frame_geometry(
    pose: np.ndarray, aspect: float, frame_height: float, cfg: PocketConfig
) -> tuple[bool, str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Valid frame: shoulders, hips, scale. Returns torso_ok, reason, L_px, S, H, u, frontal ratio."""
    p = pose.copy()
    p[0, 0] *= aspect
    p[1, 0] *= aspect
    p[4, 0] *= aspect
    p[5, 0] *= aspect
    p[6, 0] *= aspect
    p[7, 0] *= aspect

    needed = (SHOULDER_L, SHOULDER_R, HIP_L, HIP_R)
    if not all(np.isfinite(p[i]).all() for i in needed):
        return False, "MISSING_TORSO", np.nan, None, None, None, None, np.nan  # type: ignore

    s = 0.5 * (p[SHOULDER_L] + p[SHOULDER_R])
    h = 0.5 * (p[HIP_L] + p[HIP_R])
    sh = h - s
    l_norm = float(np.linalg.norm(sh))
    if l_norm < 1e-4:
        return False, "TORSO_TOO_SMALL", np.nan, None, None, None, None, np.nan  # type: ignore

    l_px = l_norm * frame_height
    if l_px < cfg.min_torso_px:
        return False, "TORSO_BELOW_MIN_PX", l_px, None, None, None, None, np.nan  # type: ignore

    u = sh / l_norm
    width = float(np.linalg.norm(p[SHOULDER_L] - p[SHOULDER_R]))
    frontal = width / l_norm
    if frontal < cfg.side_view_ratio:
        return False, "SIDE_VIEW", l_px, s, h, u, None, frontal  # type: ignore

    hip_l = p[HIP_L]
    hip_r = p[HIP_R]
    return True, "", l_px, s, h, u, (hip_l, hip_r), frontal


def _pocket_center(hip: np.ndarray, u: np.ndarray, l_norm: float, cfg: PocketConfig) -> np.ndarray:
    return hip + cfg.pocket_offset * l_norm * u


def _dist_series(
    poses: np.ndarray,
    wrist_ix: int,
    pocket_ix: int,
    aspect: float,
    frame_height: float,
    cfg: PocketConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: d (nan invalid), valid mask, L_norm, hip center for discontinuity."""
    t = len(poses)
    d = np.full(t, np.nan)
    valid = np.zeros(t, dtype=bool)
    l_norm = np.full(t, np.nan)
    hip_c = np.full((t, 2), np.nan)

    for i in range(t):
        ok, reason, l_px, s, h, u, hips, _frontal = _frame_geometry(poses[i], aspect, frame_height, cfg)
        if not ok:
            continue
        assert hips is not None and u is not None and h is not None
        hip_l, hip_r = hips
        pocket_hip = hip_l if pocket_ix == HIP_L else hip_r
        wrist = poses[i, wrist_ix].copy()
        wrist[0] *= aspect
        if not np.isfinite(wrist).all():
            continue
        ln = float(np.linalg.norm(h - s))
        b = _pocket_center(pocket_hip, u, ln, cfg)
        d[i] = float(np.linalg.norm(wrist - b) / ln)
        valid[i] = True
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


def _identity_break(
    i: int,
    hip_c: np.ndarray,
    l_norm: np.ndarray,
    fps: float,
    period: float,
    cfg: PocketConfig,
) -> bool:
    if i <= 0:
        return False
    if not (np.isfinite(hip_c[i]).all() and np.isfinite(hip_c[i - 1]).all()):
        return False
    if not (np.isfinite(l_norm[i]) and np.isfinite(l_norm[i - 1])):
        return False
    dt = period
    if dt > cfg.identity_jump_window_s * 2:
        pass
    jump = float(np.linalg.norm(hip_c[i] - hip_c[i - 1]))
    if jump > cfg.identity_hip_jump_L * l_norm[i] and dt <= cfg.identity_jump_window_s:
        return True
    if l_norm[i - 1] > 1e-6:
        scale_chg = abs(l_norm[i] - l_norm[i - 1]) / l_norm[i - 1]
        if scale_chg > cfg.scale_jump_ratio:
            return True
    return False


def _valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i, n = 0, len(valid)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _evaluate_pair(
    poses: np.ndarray,
    wrist: WristId,
    pocket: PocketId,
    *,
    fps: float,
    aspect: float,
    frame_height: float,
    cfg: PocketConfig,
    timestamps: np.ndarray | None,
    wrist_observed: np.ndarray | None,
) -> PairEvidence:
    wrist_ix = WRIST_L if wrist == "left" else WRIST_R
    pocket_ix = HIP_L if pocket == "left" else HIP_R
    d, valid, l_norm, hip_c = _dist_series(poses, wrist_ix, pocket_ix, aspect, frame_height, cfg)
    t = len(d)
    period = _nominal_period(timestamps, fps)
    max_gap = cfg.max_time_gap_factor * period

    if wrist_observed is not None and len(wrist_observed) == t:
        valid &= wrist_observed.astype(bool)

    motivo = "SIN_DATOS_VALIDOS"
    if valid.sum() == 0:
        return PairEvidence(wrist, pocket, "INDETERMINADO", motivo, valid_ratio=0.0)

    compatible_evidence: PairEvidence | None = None

    for start, end in _valid_runs(valid):
        segment_d = d[start : end + 1]
        seg_valid = valid[start : end + 1]
        n_seg = len(segment_d)

        for i in range(start + 1, end + 1):
            if not (valid[i] and valid[i - 1]):
                continue
            if d[i] > cfg.r_near or d[i - 1] > cfg.r_near:
                continue

            entry_idx = i
            d_entry = float(d[entry_idx])
            win_start_t = _time_at(entry_idx, timestamps, fps) - cfg.approach_window_s
            win_end_t = _time_at(entry_idx, timestamps, fps) - cfg.approach_min_lead_s

            origin_idx = None
            origin_d = None
            for j in range(start, entry_idx):
                if not valid[j]:
                    continue
                t_j = _time_at(j, timestamps, fps)
                if t_j < win_start_t or t_j > win_end_t:
                    continue
                if d[j] < cfg.r_far:
                    continue
                if d[j] - d_entry < cfg.delta_min:
                    continue
                gap_ok = True
                for k in range(j + 1, entry_idx):
                    if not valid[k]:
                        gap_ok = False
                        break
                    if _identity_break(k, hip_c, l_norm, fps, period, cfg):
                        gap_ok = False
                        break
                if not gap_ok:
                    continue
                if origin_idx is None or d[j] > origin_d:  # type: ignore
                    origin_idx = j
                    origin_d = float(d[j])

            if origin_idx is None:
                continue

            d_min = float(np.nanmin(segment_d[seg_valid]))
            after = segment_d[segment_d >= cfg.r_far]
            t_ret = None
            for k in range(entry_idx, end + 1):
                if valid[k] and d[k] >= cfg.r_far:
                    t_ret = _time_at(k, timestamps, fps)
                    break

            ev = PairEvidence(
                wrist=wrist,
                pocket=pocket,
                verdict="COMPATIBLE",
                motivo="APROXIMACION_OBSERVADA",
                t_origen=_time_at(origin_idx, timestamps, fps),
                d_origen=origin_d,
                t_entrada=_time_at(entry_idx, timestamps, fps),
                d_entrada=d_entry,
                d_min=d_min,
                t_retirada=t_ret,
                valid_ratio=float(seg_valid.mean()),
            )
            compatible_evidence = ev
            break

        if compatible_evidence:
            break

    if compatible_evidence:
        return compatible_evidence

    all_far = True
    side_view_blocks = 0
    for i in range(t):
        ok_geom, reason, *_ = _frame_geometry(poses[i], aspect, frame_height, cfg)
        if reason == "SIDE_VIEW":
            side_view_blocks += 1
        if not valid[i]:
            continue
        if np.isfinite(d[i]) and d[i] < cfg.r_far:
            all_far = False
            break

    valid_ratio = float(valid.sum() / max(t, 1))
    if side_view_blocks > t * 0.5:
        return PairEvidence(
            wrist, pocket, "INDETERMINADO", "VISTA_NO_APTA", valid_ratio=valid_ratio
        )

    if valid_ratio < 0.7 or valid.sum() < cfg.contact_consecutive:
        return PairEvidence(
            wrist, pocket, "INDETERMINADO", "EVIDENCIA_PARCIAL", valid_ratio=valid_ratio
        )

    if all_far and valid.all():
        runs = _valid_runs(valid)
        if len(runs) == 1 and runs[0][0] == 0 and runs[0][1] == t - 1:
            return PairEvidence(
                wrist, pocket, "INCOMPATIBLE", "LEJOS_TODO_EL_GESTO", valid_ratio=valid_ratio
            )

    return PairEvidence(
        wrist, pocket, "INDETERMINADO", "ZONA_GRIS", valid_ratio=valid_ratio
    )


def _fuse_pairs(pairs: list[PairEvidence]) -> tuple[Verdict, str]:
    if any(p.verdict == "COMPATIBLE" for p in pairs):
        best = next(p for p in pairs if p.verdict == "COMPATIBLE")
        return "COMPATIBLE", best.motivo
    if pairs and all(p.verdict == "INCOMPATIBLE" for p in pairs):
        return "INCOMPATIBLE", "LEJOS_TODO_EL_GESTO"
    reasons = {p.motivo for p in pairs if p.verdict == "INDETERMINADO"}
    motivo = sorted(reasons)[0] if reasons else "INDETERMINADO"
    return "INDETERMINADO", motivo


def evaluate_pocket(
    poses: np.ndarray,
    *,
    fps: float = 12.5,
    aspect: float = 16.0 / 9.0,
    frame_height: float = 1080.0,
    frame_width: float | None = None,
    config: PocketConfig | None = None,
    timestamps: np.ndarray | None = None,
    wrist_mask: np.ndarray | None = None,
) -> PocketResult:
    """
    Evalúa el clip completo. ``hand_to_body`` es True solo si resultado == COMPATIBLE.
    """
    if poses.ndim != 3 or poses.shape[1:] != (8, 2):
        raise ValueError(f"Expected poses (T, 8, 2), got {poses.shape}")
    cfg = config or PocketConfig()
    if frame_width is not None and frame_height > 0:
        aspect = frame_width / frame_height

    t = len(poses)
    wrist_obs: dict[WristId, np.ndarray | None] = {"left": None, "right": None}
    if wrist_mask is not None and wrist_mask.shape == (t, 8):
        wrist_obs["left"] = wrist_mask[:, WRIST_L]
        wrist_obs["right"] = wrist_mask[:, WRIST_R]

    pairs: list[PairEvidence] = []
    for wrist in ("left", "right"):
        for pocket in ("left", "right"):
            obs = wrist_obs[wrist]
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
                    wrist_observed=obs,
                )
            )

    resultado, motivo = _fuse_pairs(pairs)
    t_end = _time_at(t - 1, timestamps, fps) if t else 0.0
    return PocketResult(
        resultado=resultado,
        hand_to_body=resultado == "COMPATIBLE",
        motivo=motivo,
        version=cfg.version,
        fps=float(fps),
        n_frames=t,
        pairs=pairs,
        intervalo_s=(0.0, t_end),
    )