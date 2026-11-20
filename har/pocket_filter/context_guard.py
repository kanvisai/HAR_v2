"""
Filtro contextual post-pocket: walk / stop / manos alejadas para bajar FP en pasillo.

Usa las mismas máscaras de movimiento que har_actions.detect (categorías 0/1/5).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pocket_filter.config import CONFIG_PATH
from pocket_filter.types import PocketResult

# Umbrales débiles: vetar solo con estas si veto_weak_motives_only
WEAK_MOTIVES = frozenset(
    {
        "APROXIMACION_OBSERVADA",
        "EVIDENCIA_PARCIAL",
        "ZONA_GRIS",
    }
)


@dataclass
class ContextGuardConfig:
    enabled: bool = True
    hands_away_axis_min: float = 0.38
    veto_walk_window_frac: float = 0.45
    veto_walk_clip_min_s: float = 1.5
    veto_stop_window_frac: float = 0.55
    veto_hands_away_window_frac: float = 0.60
    veto_stop_hands_away_window_frac: float = 0.50
    veto_weak_motives_only: bool = True


def load_context_guard_config(path: Path | None = None) -> ContextGuardConfig:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        return ContextGuardConfig()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data.get("context_guard", {})
    fields = {f.name for f in ContextGuardConfig.__dataclass_fields__.values()}  # type: ignore
    kwargs = {k: v for k, v in block.items() if k in fields}
    return ContextGuardConfig(**kwargs)


def _hands_away_mask(features: dict, *, axis_min: float) -> np.ndarray:
    n = features["n"]
    mask = np.zeros(n, dtype=bool)
    for side in ("left", "right"):
        arms = features["arms"][side]
        away = arms["ok"] & (arms["dist_axis"] >= axis_min)
        mask |= away
    return mask


def _mask_fraction(mask: np.ndarray, i0: int, i1: int) -> float:
    if i1 < i0:
        return 0.0
    seg = mask[i0 : i1 + 1]
    if len(seg) == 0:
        return 0.0
    return float(seg.mean())


def _mask_seconds(mask: np.ndarray, fps: float) -> float:
    return float(mask.sum()) / max(fps, 1e-6)


def _gesture_window_frames(result: PocketResult, fps: float) -> tuple[int, int]:
    bp = result.best_pair
    i0, i1 = result.eval_start, result.eval_end
    if bp is not None:
        if bp.t_entrada is not None:
            i0 = max(i0, int(round(bp.t_entrada * fps)))
        if bp.t_retirada is not None:
            i1 = min(i1, int(round(bp.t_retirada * fps)))
        if bp.t_origen is not None and bp.t_entrada is None:
            i0 = max(i0, int(round(bp.t_origen * fps)))
    i0 = max(0, min(i0, result.n_frames - 1))
    i1 = max(i0, min(i1, result.n_frames - 1))
    return i0, i1


def compute_context_signals(
    poses: np.ndarray,
    *,
    fps: float,
    aspect: float,
    window: tuple[int, int],
    gcfg: ContextGuardConfig,
) -> dict[str, float]:
    from har_actions.detect import _dilate, _features, _motion_mask

    features = _features(poses, fps, aspect)
    walk = _motion_mask(features, walking=True)
    stop = _motion_mask(features, walking=False) & ~_dilate(walk, 3)
    hands_away = _hands_away_mask(features, axis_min=gcfg.hands_away_axis_min)

    i0, i1 = window
    return {
        "walk_frac": _mask_fraction(walk, i0, i1),
        "stop_frac": _mask_fraction(stop, i0, i1),
        "hands_away_frac": _mask_fraction(hands_away, i0, i1),
        "walk_s": _mask_seconds(walk, fps),
        "stop_s": _mask_seconds(stop, fps),
        "hands_away_s": _mask_seconds(hands_away, fps),
    }


def _motivo_allows_veto(motivo: str, gcfg: ContextGuardConfig) -> bool:
    if not gcfg.veto_weak_motives_only:
        return True
    return motivo in WEAK_MOTIVES


def decide_context_veto(
    signals: dict[str, float],
    *,
    motivo: str,
    gcfg: ContextGuardConfig,
) -> str | None:
    if not _motivo_allows_veto(motivo, gcfg):
        return None

    if (
        signals["walk_frac"] >= gcfg.veto_walk_window_frac
        and signals["walk_s"] >= gcfg.veto_walk_clip_min_s
    ):
        return "CONTEXT_VETO_WALKING"

    if (
        signals["stop_frac"] >= gcfg.veto_stop_window_frac
        and signals["hands_away_frac"] >= gcfg.veto_stop_hands_away_window_frac
    ):
        return "CONTEXT_VETO_STILL_HANDS_AWAY"

    if signals["hands_away_frac"] >= gcfg.veto_hands_away_window_frac:
        return "CONTEXT_VETO_HANDS_EXTENDED"

    return None


def apply_context_guard(
    result: PocketResult,
    poses: np.ndarray,
    *,
    fps: float,
    aspect: float,
    gcfg: ContextGuardConfig | None = None,
    config_path: Path | None = None,
) -> PocketResult:
    """Actualiza hand_to_body y añade campos de contexto si aplica veto."""
    cfg = gcfg or load_context_guard_config(config_path)
    result.context_guard_enabled = cfg.enabled
    result.hand_to_body_raw = result.hand_to_body

    if not cfg.enabled or not result.hand_to_body:
        result.context_veto = None
        result.ctx_walk_frac = 0.0
        result.ctx_stop_frac = 0.0
        result.ctx_hands_away_frac = 0.0
        result.ctx_walk_s = 0.0
        result.ctx_stop_s = 0.0
        return result

    window = _gesture_window_frames(result, fps)
    signals = compute_context_signals(
        poses, fps=fps, aspect=aspect, window=window, gcfg=cfg
    )
    result.ctx_walk_frac = signals["walk_frac"]
    result.ctx_stop_frac = signals["stop_frac"]
    result.ctx_hands_away_frac = signals["hands_away_frac"]
    result.ctx_walk_s = signals["walk_s"]
    result.ctx_stop_s = signals["stop_s"]

    veto = decide_context_veto(signals, motivo=result.motivo, gcfg=cfg)
    result.context_veto = veto
    if veto:
        result.hand_to_body = False
        result.motivo = f"{result.motivo}+{veto}"
    return result