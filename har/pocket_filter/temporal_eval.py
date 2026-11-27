"""
Evaluación temporal por ventanas (chunks ~8 s en producción).

Busca el mejor pico de evidencia hand_to_body en subintervalos,
en lugar de diluir con todo el buffer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pocket_filter.action_policy import apply_action_policy
from pocket_filter.config import CONFIG_PATH, PocketConfig, load_pocket_config
from pocket_filter.evaluate import evaluate_pocket
from pocket_filter.types import PocketResult

_VERDICT_RANK = {"COMPATIBLE": 2, "INDETERMINADO": 1, "INCOMPATIBLE": 0}


_STRONG_MOTIVES = frozenset({"CONTACTO_BOLSILLO"})
_AXIS_APPROX = frozenset({"ACERCAMIENTO_EJE_TORSO", "APROXIMACION_OBSERVADA"})


@dataclass
class TemporalConfig:
    enabled: bool = True
    window_s: float = 2.5
    step_s: float = 1.25
    context_pad_s: float = 0.35
    """Si true, al fusionar ventanas vetamos confirmaciones con marcha en la ventana ganadora."""
    walk_gate_enabled: bool = False
    walk_gate_window_frac: float = 0.38
    walk_gate_window_min_s: float = 0.85
    walk_gate_contact_min_confidence: float = 0.64


def load_temporal_config(path: Path | None = None) -> TemporalConfig:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        return TemporalConfig()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data.get("temporal", {})
    fields = {f.name for f in TemporalConfig.__dataclass_fields__.values()}  # type: ignore
    kwargs = {k: v for k, v in block.items() if k in fields}
    return TemporalConfig(**kwargs)


def sliding_windows(
    clip_s: float,
    *,
    window_s: float,
    step_s: float,
) -> list[tuple[float, float]]:
    if clip_s <= 0 or window_s <= 0 or step_s <= 0:
        return [(0.0, max(clip_s, window_s))]
    if clip_s <= window_s:
        return [(0.0, clip_s)]
    wins: list[tuple[float, float]] = []
    t = 0.0
    while t < clip_s:
        t1 = min(clip_s, t + window_s)
        if t1 - t >= min(window_s * 0.5, 0.8):
            wins.append((t, t1))
        t += step_s
        if t1 >= clip_s:
            break
    if not wins:
        wins = [(0.0, clip_s)]
    return wins


def _window_score(r: PocketResult) -> tuple:
    return (
        1 if r.hand_to_body else 0,
        1 if r.hand_to_body_raw else 0,
        r.hand_to_body_confidence,
        _VERDICT_RANK.get(r.resultado, 0),
    )


def _temporal_walk_gate_veto(r: PocketResult, tcfg: TemporalConfig) -> str | None:
    """Veto en ventana ganadora con marcha (hueco entre política 0,44 y pasillo real ~0,38)."""
    if not tcfg.walk_gate_enabled:
        return None
    if (
        r.ctx_walk_frac < tcfg.walk_gate_window_frac
        or r.ctx_walk_s < tcfg.walk_gate_window_min_s
    ):
        return None
    if r.motivo in _STRONG_MOTIVES:
        if r.hand_to_body_confidence >= tcfg.walk_gate_contact_min_confidence:
            return None
        return "TEMPORAL_WALK_GATE"
    if r.motivo in _AXIS_APPROX:
        return "TEMPORAL_WALK_GATE"
    return None


def _best_clear_window(
    windows: list[tuple[tuple[float, float], PocketResult]],
    tcfg: TemporalConfig,
) -> tuple[bool, tuple[float, float] | None, PocketResult | None, str | None]:
    """
    Elige la mejor ventana que confirme hand_to_body y pasa walk_gate.
    Si todas las confirmaciones caen por marcha, hand_to_body queda false.
    """
    clears = [(iv, r) for iv, r in windows if r.hand_to_body]
    if not clears:
        return False, None, None, None
    ranked = sorted(clears, key=lambda x: _window_score(x[1]), reverse=True)
    for iv, r in ranked:
        veto = _temporal_walk_gate_veto(r, tcfg)
        if veto is None:
            return True, iv, r, None
    iv, r = ranked[0]
    return False, iv, r, _temporal_walk_gate_veto(r, tcfg) or "TEMPORAL_WALK_GATE"


def _merge_results(
    full: PocketResult,
    windows: list[tuple[tuple[float, float], PocketResult]],
    *,
    tcfg: TemporalConfig | None = None,
) -> PocketResult:
    tcfg = tcfg or TemporalConfig()
    ranked = sorted(windows, key=lambda x: _window_score(x[1]), reverse=True)
    best_iv, best = ranked[0]

    accepted, clear_iv, clear_r, walk_veto = _best_clear_window(windows, tcfg)
    best_iv, best = ranked[0]
    if tcfg.walk_gate_enabled:
        any_clear = accepted
    else:
        any_clear = bool(best.hand_to_body)
    any_raw = any(bool(r.hand_to_body_raw) for _, r in windows)
    max_conf = max(r.hand_to_body_confidence for _, r in windows)

    full.hand_to_body = any_clear
    full.hand_to_body_raw = any_raw
    full.hand_to_body_confidence = max_conf
    full.motion_suppress = all(r.motion_suppress for _, r in windows) and not any_raw
    if any_clear or any_raw:
        full.motion_suppress = False

    meta = clear_r if (any_clear and clear_r is not None) else best
    full.resultado = meta.resultado
    full.filter_fp_safe = meta.filter_fp_safe
    full.best_pair = meta.best_pair
    full.pairs = meta.pairs
    meta_iv = clear_iv if (any_clear and clear_iv is not None) else best_iv
    full.eval_start = meta.eval_start
    full.eval_end = meta.eval_end
    full.intervalo_s = meta_iv
    if walk_veto and not any_clear and any_clear_raw:
        full.context_veto = walk_veto
    else:
        full.context_veto = meta.context_veto
    full.ctx_walk_frac = meta.ctx_walk_frac
    full.ctx_stop_frac = meta.ctx_stop_frac
    full.ctx_hands_away_frac = meta.ctx_hands_away_frac
    full.ctx_walk_s = meta.ctx_walk_s
    full.ctx_stop_s = meta.ctx_stop_s
    full.motivo = meta.motivo
    full.htb_tier = meta.htb_tier if any_clear else ("none" if not any_raw else best.htb_tier)
    if any_clear:
        full.htb_tier = "alert"
    full.primary_action = meta.primary_action if any_clear else (
        "hand_to_body" if any_raw else best.primary_action
    )
    full.primary_action_confidence = best.primary_action_confidence
    if not any_clear and any_raw:
        full.primary_action = "hand_to_body"
        full.primary_action_confidence = max_conf
    full.kin_hip_speed_norm = meta.kin_hip_speed_norm
    full.kin_near_dwell_s = meta.kin_near_dwell_s
    full.kin_swing_peaks = meta.kin_swing_peaks
    full.kin_approach_depth = meta.kin_approach_depth
    full.kin_approach_monotone_frac = meta.kin_approach_monotone_frac
    full.kin_wrist_radial_vel_min = meta.kin_wrist_radial_vel_min
    full.kin_elbow_angle_deg = meta.kin_elbow_angle_deg
    full.kin_wrist_axis_dist_min = meta.kin_wrist_axis_dist_min
    return full


def evaluate_pocket_temporal(
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
    config_path: Path | None = None,
    temporal: TemporalConfig | None = None,
) -> PocketResult:
    from pocket_filter.timestamp_utils import prepare_frame_timestamps

    cfg = config or load_pocket_config(config_path)
    tcfg = temporal or load_temporal_config(config_path)
    timestamps, _ = prepare_frame_timestamps(timestamps, len(poses), fps)
    clip_s = len(poses) / max(fps, 1e-6)

    full = evaluate_pocket(
        poses,
        fps=fps,
        aspect=aspect,
        frame_height=frame_height,
        frame_width=frame_width,
        config=cfg,
        timestamps=timestamps,
        row_mask=row_mask,
        keypoint_conf=keypoint_conf,
    )
    wins = sliding_windows(clip_s, window_s=tcfg.window_s, step_s=tcfg.step_s)
    if len(wins) <= 1 and clip_s <= tcfg.window_s * 1.1:
        return apply_action_policy(
            full,
            poses,
            fps=fps,
            aspect=aspect,
            frame_height=frame_height,
            timestamps=timestamps,
            row_mask=row_mask,
            keypoint_conf=keypoint_conf,
            pocket_cfg=cfg,
            config_path=config_path,
        )

    evaluated: list[tuple[tuple[float, float], PocketResult]] = []
    for t0, t1 in wins:
        base = evaluate_pocket(
            poses,
            fps=fps,
            aspect=aspect,
            frame_height=frame_height,
            frame_width=frame_width,
            config=cfg,
            timestamps=timestamps,
            row_mask=row_mask,
            keypoint_conf=keypoint_conf,
            interval_s=(t0, t1),
            context_pad_s=tcfg.context_pad_s,
        )
        pol = apply_action_policy(
            base,
            poses,
            fps=fps,
            aspect=aspect,
            frame_height=frame_height,
            timestamps=timestamps,
            row_mask=row_mask,
            keypoint_conf=keypoint_conf,
            pocket_cfg=cfg,
            config_path=config_path,
        )
        evaluated.append(((t0, t1), pol))

    return _merge_results(full, evaluated, tcfg=tcfg)