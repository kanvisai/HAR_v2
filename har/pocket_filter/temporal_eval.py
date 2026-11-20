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


@dataclass
class TemporalConfig:
    enabled: bool = True
    window_s: float = 2.5
    step_s: float = 1.25
    context_pad_s: float = 0.35


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


def _merge_results(
    full: PocketResult,
    windows: list[tuple[tuple[float, float], PocketResult]],
) -> PocketResult:
    ranked = sorted(windows, key=lambda x: _window_score(x[1]), reverse=True)
    best_iv, best = ranked[0]

    any_clear = any(r.hand_to_body for _, r in windows)
    any_raw = any(bool(r.hand_to_body_raw) for _, r in windows)
    max_conf = max(r.hand_to_body_confidence for _, r in windows)

    full.hand_to_body = any_clear
    full.hand_to_body_raw = any_raw
    full.hand_to_body_confidence = max_conf
    full.motion_suppress = all(r.motion_suppress for _, r in windows) and not any_raw
    if any_clear or any_raw:
        full.motion_suppress = False

    full.resultado = best.resultado
    full.motivo = best.motivo
    full.filter_fp_safe = best.filter_fp_safe
    full.best_pair = best.best_pair
    full.pairs = best.pairs
    full.eval_start = best.eval_start
    full.eval_end = best.eval_end
    full.intervalo_s = best_iv
    full.context_veto = best.context_veto
    full.ctx_walk_frac = best.ctx_walk_frac
    full.ctx_stop_frac = best.ctx_stop_frac
    full.ctx_hands_away_frac = best.ctx_hands_away_frac
    full.ctx_walk_s = best.ctx_walk_s
    full.ctx_stop_s = best.ctx_stop_s
    full.htb_tier = best.htb_tier if any_clear else ("none" if not any_raw else best.htb_tier)
    if any_clear:
        full.htb_tier = "alert"
    full.primary_action = best.primary_action if any_clear else ("hand_to_body" if any_raw else best.primary_action)
    full.primary_action_confidence = best.primary_action_confidence
    if not any_clear and any_raw:
        full.primary_action = "hand_to_body"
        full.primary_action_confidence = max_conf
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
    cfg = config or load_pocket_config(config_path)
    tcfg = temporal or load_temporal_config(config_path)
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
            full, poses, fps=fps, aspect=aspect, pocket_cfg=cfg, config_path=config_path
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
            pocket_cfg=cfg,
            config_path=config_path,
        )
        evaluated.append(((t0, t1), pol))

    return _merge_results(full, evaluated)