"""
Política clip: pocket (confianza) + acciones cotidianas HAR1 (walk, stop, …).

``hand_to_body`` final = pocket COMPATIBLE y confianza ≥ umbral (más estricto en motivos débiles).
Si no, se elige la acción cotidiana dominante del detector clásico.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from har_actions.detect import detect_cotidian_actions, gesture_seconds, visible_seconds
from har_actions.pose import smooth
from pocket_filter.confidence import pocket_hand_to_body_confidence

WALK_VETO_MOTIVES = frozenset(
    {
        "ACERCAMIENTO_EJE_TORSO",
        "APROXIMACION_OBSERVADA",
        "EVIDENCIA_PARCIAL",
        "ZONA_GRIS",
    }
)
STRICT_CONF_MOTIVES = frozenset(
    {
        "APROXIMACION_OBSERVADA",
        "EVIDENCIA_PARCIAL",
        "ZONA_GRIS",
    }
)
from pocket_filter.config import CONFIG_PATH, PocketConfig, load_pocket_config
from pocket_filter.context_guard import compute_context_signals, load_context_guard_config
from pocket_filter.evaluate import evaluate_pocket
from pocket_filter.types import PocketResult

COTIDIAN_PRIMARY = (
    "walk",
    "stop",
    "arm_swing",
    "carry",
    "hand_up",
    "turn_torso",
)


@dataclass
class ActionPolicyConfig:
    enabled: bool = True
    hand_to_body_min_confidence: float = 0.62
    weak_motive_min_confidence: float = 0.72
    walk_override_walk_frac: float = 0.42
    walk_override_walk_clip_min_s: float = 1.0
    cotidian_min_visible_frac: float = 0.08
    inconclusive_below_s: float = 0.25


def load_action_policy_config(path: Path | None = None) -> ActionPolicyConfig:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        return ActionPolicyConfig()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data.get("action_policy", {})
    fields = {f.name for f in ActionPolicyConfig.__dataclass_fields__.values()}  # type: ignore
    kwargs = {k: v for k, v in block.items() if k in fields}
    return ActionPolicyConfig(**kwargs)


def _cotidian_durations(
    poses: np.ndarray,
    *,
    fps: float,
    aspect: float,
) -> tuple[dict[str, float], float]:
    smoothed = smooth(poses, window=3)
    actions = detect_cotidian_actions(smoothed, fps, aspect)
    seconds = gesture_seconds(actions, fps)
    visible = max(visible_seconds(poses, fps), 1e-6)
    return seconds, visible


def _best_cotidian(
    seconds: dict[str, float],
    visible: float,
    *,
    min_frac: float,
) -> tuple[str, float]:
    best_id = "inconclusive"
    best_conf = 0.0
    for aid in COTIDIAN_PRIMARY:
        dur = seconds.get(aid, 0.0)
        if dur <= 0:
            continue
        conf = min(1.0, dur / visible)
        if conf >= min_frac and conf > best_conf:
            best_id = aid
            best_conf = conf
    return best_id, round(best_conf, 4)


def apply_action_policy(
    result: PocketResult,
    poses: np.ndarray,
    *,
    fps: float,
    aspect: float,
    pocket_cfg: PocketConfig | None = None,
    policy: ActionPolicyConfig | None = None,
    config_path: Path | None = None,
) -> PocketResult:
    """Rellena confianza, acción primaria y ajusta ``hand_to_body`` si policy enabled."""
    pocket_cfg = pocket_cfg or load_pocket_config(config_path)
    policy = policy or load_action_policy_config(config_path)

    result.hand_to_body_raw = result.hand_to_body
    conf = pocket_hand_to_body_confidence(result, pocket_cfg)
    result.hand_to_body_confidence = conf

    seconds, visible = _cotidian_durations(poses, fps=fps, aspect=aspect)
    result.cotidian_seconds = {k: round(v, 3) for k, v in seconds.items() if v > 0}

    if not policy.enabled:
        result.primary_action = "hand_to_body" if result.hand_to_body else _best_cotidian(
            seconds, visible, min_frac=policy.cotidian_min_visible_frac
        )[0]
        result.primary_action_confidence = conf if result.hand_to_body else 0.0
        return result

    min_conf = policy.hand_to_body_min_confidence
    if result.motivo in STRICT_CONF_MOTIVES:
        min_conf = max(min_conf, policy.weak_motive_min_confidence)

    gcfg = load_context_guard_config(config_path)
    window = (result.eval_start, result.eval_end)
    signals = compute_context_signals(
        poses, fps=fps, aspect=aspect, window=window, gcfg=gcfg
    )
    result.ctx_walk_frac = signals["walk_frac"]
    result.ctx_stop_frac = signals["stop_frac"]
    result.ctx_hands_away_frac = signals["hands_away_frac"]
    result.ctx_walk_s = signals["walk_s"]
    result.ctx_stop_s = signals["stop_s"]

    raw = bool(result.hand_to_body_raw)
    accept_htb = raw and conf >= min_conf

    walk_veto = (
        raw
        and not accept_htb
        and result.motivo in WALK_VETO_MOTIVES
        and signals["walk_frac"] >= policy.walk_override_walk_frac
        and signals["walk_s"] >= policy.walk_override_walk_clip_min_s
    )

    if accept_htb:
        result.hand_to_body = True
        result.primary_action = "hand_to_body"
        result.primary_action_confidence = conf
        result.context_veto = None
        return result

    result.hand_to_body = False
    if walk_veto:
        result.context_veto = "POLICY_WALK_OVER_WEAK_POCKET"
        walk_conf = min(1.0, seconds.get("walk", 0.0) / visible)
        result.primary_action = "walk"
        result.primary_action_confidence = round(max(walk_conf, signals["walk_frac"]), 4)
        return result

    cot_id, cot_conf = _best_cotidian(
        seconds, visible, min_frac=policy.cotidian_min_visible_frac
    )
    if cot_id == "inconclusive" and seconds.get("stop", 0) >= policy.inconclusive_below_s:
        cot_id = "stop"
        cot_conf = round(min(1.0, seconds["stop"] / visible), 4)

    result.primary_action = cot_id
    result.primary_action_confidence = cot_conf
    result.context_veto = (
        "POLICY_LOW_CONFIDENCE" if raw and conf < min_conf else None
    )
    return result


def evaluate_pocket_with_policy(
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
    config_path: Path | None = None,
) -> PocketResult:
    result = evaluate_pocket(
        poses,
        fps=fps,
        aspect=aspect,
        frame_height=frame_height,
        frame_width=frame_width,
        config=config,
        timestamps=timestamps,
        row_mask=row_mask,
        keypoint_conf=keypoint_conf,
        interval_s=interval_s,
        context_pad_s=context_pad_s,
    )
    return apply_action_policy(
        result, poses, fps=fps, aspect=aspect, pocket_cfg=config, config_path=config_path
    )