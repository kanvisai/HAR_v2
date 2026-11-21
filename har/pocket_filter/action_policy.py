"""
Política clip (sin categoría): pocket + confianza + contexto walk/stop HAR1.

Idea: los FP de pasillo suelen ser ``ACERCAMIENTO_EJE`` **caminando**; muchos robos
comparten el mismo motivo **parados**. Se suprime el gesto débil solo con marcha clara;
en quieto basta confianza moderada. Motivos fuertes (bolsillo) pasan con umbral bajo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from har_actions.detect import detect_cotidian_actions, gesture_seconds, visible_seconds
from har_actions.pose import smooth
from pocket_filter.confidence import pocket_hand_to_body_confidence
from pocket_filter.config import CONFIG_PATH, PocketConfig, load_pocket_config
from pocket_filter.context_guard import compute_context_signals, load_context_guard_config
from pocket_filter.evaluate import evaluate_pocket
from pocket_filter.types import PocketResult

STRONG_MOTIVES = frozenset({"CONTACTO_BOLSILLO"})
APPROACH_MOTIVES = frozenset({"APROXIMACION_OBSERVADA"})
AXIS_MOTIVES = frozenset({"ACERCAMIENTO_EJE_TORSO"})
WALK_SUPPRESS_MOTIVES = frozenset(
    {
        "ACERCAMIENTO_EJE_TORSO",
        "APROXIMACION_OBSERVADA",
    }
)

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
    """full = tiers + acciones cotidianas; robbery_confirm = solo boolean hand_to_body."""
    gate_mode: str = "robbery_confirm"
    strong_min_confidence: float = 0.48
    gate_contact_min_confidence_while_walking: float = 0.62
    approx_min_confidence: float = 0.56
    approx_min_confidence_while_walking: float = 0.68
    axis_min_confidence: float = 0.50
    axis_min_confidence_while_walking: float = 0.66
    axis_quiet_stop_frac: float = 0.30
    walk_active_frac: float = 0.44
    walk_active_clip_min_s: float = 1.0
    cotidian_min_visible_frac: float = 0.08
    inconclusive_below_s: float = 0.25
    axis_max_tier: str = "review"
    review_min_confidence: float = 0.48
    review_axis_min_confidence: float = 0.52


def load_action_policy_config(path: Path | None = None) -> ActionPolicyConfig:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        return ActionPolicyConfig()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data.get("action_policy", {})
    fields = {f.name for f in ActionPolicyConfig.__dataclass_fields__.values()}  # type: ignore
    kwargs = {k: v for k, v in block.items() if k in fields}
    # aliases viejos → nuevos (compatibilidad config)
    if "hand_to_body_min_confidence" in block and "axis_min_confidence" not in block:
        kwargs.setdefault("axis_min_confidence", float(block["hand_to_body_min_confidence"]))
    return ActionPolicyConfig(**kwargs)


def _is_walking(signals: dict[str, float], policy: ActionPolicyConfig) -> bool:
    return (
        signals["walk_frac"] >= policy.walk_active_frac
        and signals["walk_s"] >= policy.walk_active_clip_min_s
    )


def _required_htb_confidence(
    motivo: str,
    *,
    walking: bool,
    stop_frac: float,
    policy: ActionPolicyConfig,
) -> float:
    if motivo in STRONG_MOTIVES:
        return policy.strong_min_confidence
    if motivo in APPROACH_MOTIVES:
        if walking:
            return policy.approx_min_confidence_while_walking
        return policy.approx_min_confidence
    if motivo in AXIS_MOTIVES:
        if walking:
            return policy.axis_min_confidence_while_walking
        if stop_frac >= policy.axis_quiet_stop_frac:
            return policy.axis_min_confidence
        return policy.axis_min_confidence + 0.06
    return 0.62


def decide_hand_to_body(
    *,
    pocket_compatible: bool,
    conf: float,
    motivo: str,
    signals: dict[str, float],
    policy: ActionPolicyConfig,
) -> tuple[bool, str | None]:
    """
    Decisión final sin categoría. Devuelve (hand_to_body, veto_code).
    """
    if not pocket_compatible:
        return False, None

    walking = _is_walking(signals, policy)
    need = _required_htb_confidence(
        motivo,
        walking=walking,
        stop_frac=signals["stop_frac"],
        policy=policy,
    )

    if conf >= need:
        return True, None

    if (
        motivo in WALK_SUPPRESS_MOTIVES
        and walking
        and conf < need
    ):
        return False, "POLICY_WALK_OVER_WEAK_POCKET"

    if conf >= need - 0.04 and motivo in AXIS_MOTIVES and not walking:
        return True, None

    return False, "POLICY_LOW_CONFIDENCE" if conf < need else None


def decide_robbery_confirm(
    *,
    pocket_compatible: bool,
    conf: float,
    motivo: str,
    signals: dict[str, float],
    policy: ActionPolicyConfig,
) -> tuple[bool, str | None]:
    """
    Confirmación binaria para candidatos del modelo de robo (sin clasificar pasillo).
    """
    if not pocket_compatible:
        return False, "GATE_NO_POCKET"

    walking = _is_walking(signals, policy)

    if motivo in STRONG_MOTIVES:
        need = policy.strong_min_confidence
        if walking:
            need = max(need, policy.gate_contact_min_confidence_while_walking)
        if conf >= need:
            return True, None
        return False, "GATE_LOW_CONFIDENCE"

    if motivo in APPROACH_MOTIVES:
        if walking:
            return False, "GATE_WALK_NO_CONFIRM"
        if conf >= policy.approx_min_confidence:
            return True, None
        return False, "GATE_LOW_CONFIDENCE"

    if motivo in AXIS_MOTIVES:
        if walking:
            return False, "GATE_WALK_NO_CONFIRM"
        if signals["stop_frac"] < policy.axis_quiet_stop_frac:
            return False, "GATE_AXIS_NOT_QUIET"
        if conf >= policy.axis_min_confidence:
            return True, None
        if conf >= policy.axis_min_confidence - 0.03:
            return True, None
        return False, "GATE_LOW_CONFIDENCE"

    return False, "GATE_WEAK_MOTIVO"


def decide_motion_suppress(
    *,
    pocket_compatible: bool,
    hand_to_body_clear: bool,
    motivo: str,
    signals: dict[str, float],
    policy: ActionPolicyConfig,
    resultado: str = "",
    filter_fp_safe: bool = False,
) -> tuple[bool, str | None]:
    """
    ¿Suprimir alarma del modelo de robo?
    Sí solo si no hay gesto en pocket o es FP típico (gesto débil + marcha).
    Si hay pocket compatible en quieto aunque no sea ‘claro’, no suprimir (recall robos).
    INDETERMINADO no suprime (P0-3 / doctrina Kanvis).
    """
    if hand_to_body_clear:
        return False, None
    res = (resultado or "").strip().upper()
    if res == "INDETERMINADO":
        return False, "KEEP_INDETERMINADO"
    if res == "INCOMPATIBLE" and filter_fp_safe:
        return True, "SUPPRESS_INCOMPATIBLE"
    if not pocket_compatible:
        return True, "SUPPRESS_NO_POCKET"
    if _is_walking(signals, policy) and motivo in WALK_SUPPRESS_MOTIVES:
        return True, "SUPPRESS_WALK_WEAK_GESTURE"
    return False, None


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
        c = min(1.0, dur / visible)
        if c >= min_frac and c > best_conf:
            best_id = aid
            best_conf = c
    return best_id, round(best_conf, 4)


def _qualifies_review(
    *,
    raw: bool,
    conf: float,
    motivo: str,
    veto: str | None,
    policy: ActionPolicyConfig,
) -> bool:
    if not raw or veto == "POLICY_WALK_OVER_WEAK_POCKET":
        return False
    if motivo in AXIS_MOTIVES:
        return conf >= policy.review_axis_min_confidence
    if motivo in STRONG_MOTIVES | APPROACH_MOTIVES:
        return conf >= policy.review_min_confidence
    return conf >= policy.review_min_confidence


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
    pocket_cfg = pocket_cfg or load_pocket_config(config_path)
    policy = policy or load_action_policy_config(config_path)

    result.hand_to_body_raw = result.hand_to_body
    conf = pocket_hand_to_body_confidence(result, pocket_cfg)
    result.hand_to_body_confidence = conf

    if not policy.enabled:
        result.cotidian_seconds = {}
        result.htb_tier = "alert" if result.hand_to_body else "none"
        result.primary_action = "hand_to_body" if result.hand_to_body else "no_gesture"
        result.primary_action_confidence = conf if result.hand_to_body else 0.0
        raw0 = bool(result.hand_to_body_raw)
        result.motion_suppress = not raw0
        return result

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

    if policy.gate_mode == "robbery_confirm":
        accept, veto = decide_robbery_confirm(
            pocket_compatible=raw,
            conf=conf,
            motivo=result.motivo,
            signals=signals,
            policy=policy,
        )
        result.hand_to_body = accept
        result.context_veto = veto
        if accept:
            result.htb_tier = "alert"
        elif raw:
            result.htb_tier = "review"
        else:
            result.htb_tier = "none"
        result.primary_action = "hand_to_body" if (accept or raw) else "no_gesture"
        result.primary_action_confidence = conf if (accept or raw) else 0.0
        result.cotidian_seconds = {}
        sup, sup_code = decide_motion_suppress(
            pocket_compatible=raw,
            hand_to_body_clear=accept,
            motivo=result.motivo,
            signals=signals,
            policy=policy,
            resultado=result.resultado,
            filter_fp_safe=bool(result.filter_fp_safe),
        )
        result.motion_suppress = sup
        if sup_code and not result.context_veto:
            result.context_veto = sup_code
        return result

    seconds, visible = _cotidian_durations(poses, fps=fps, aspect=aspect)
    result.cotidian_seconds = {k: round(v, 3) for k, v in seconds.items() if v > 0}

    accept, veto = decide_hand_to_body(
        pocket_compatible=raw,
        conf=conf,
        motivo=result.motivo,
        signals=signals,
        policy=policy,
    )
    result.context_veto = veto

    axis_cap = (
        accept
        and result.motivo in AXIS_MOTIVES
        and policy.axis_max_tier == "review"
    )

    def _attach_motion_suppress() -> None:
        sup, code = decide_motion_suppress(
            pocket_compatible=raw,
            hand_to_body_clear=bool(result.hand_to_body),
            motivo=result.motivo,
            signals=signals,
            policy=policy,
            resultado=result.resultado,
            filter_fp_safe=bool(result.filter_fp_safe),
        )
        result.motion_suppress = sup
        if code and not result.context_veto:
            result.context_veto = code

    if accept and not axis_cap:
        result.hand_to_body = True
        result.htb_tier = "alert"
        result.primary_action = "hand_to_body"
        result.primary_action_confidence = conf
        _attach_motion_suppress()
        return result

    result.hand_to_body = False

    if axis_cap or _qualifies_review(
        raw=raw, conf=conf, motivo=result.motivo, veto=veto, policy=policy
    ):
        result.htb_tier = "review"
        result.primary_action = "hand_to_body"
        result.primary_action_confidence = conf
        if axis_cap:
            result.context_veto = result.context_veto or "TIER_AXIS_REVIEW_ONLY"
        _attach_motion_suppress()
        return result

    result.htb_tier = "none"

    if veto == "POLICY_WALK_OVER_WEAK_POCKET":
        walk_conf = min(1.0, seconds.get("walk", 0.0) / visible)
        result.primary_action = "walk"
        result.primary_action_confidence = round(
            max(walk_conf, signals["walk_frac"]), 4
        )
        _attach_motion_suppress()
        return result

    cot_id, cot_conf = _best_cotidian(
        seconds, visible, min_frac=policy.cotidian_min_visible_frac
    )
    if cot_id == "inconclusive" and seconds.get("stop", 0) >= policy.inconclusive_below_s:
        cot_id = "stop"
        cot_conf = round(min(1.0, seconds["stop"] / visible), 4)
    result.primary_action = cot_id
    result.primary_action_confidence = cot_conf
    _attach_motion_suppress()
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
    use_temporal: bool | None = None,
) -> PocketResult:
    if interval_s is None and use_temporal is not False:
        from pocket_filter.temporal_eval import load_temporal_config, evaluate_pocket_temporal

        tcfg = load_temporal_config(config_path)
        if (use_temporal is True or tcfg.enabled) and tcfg.enabled:
            return evaluate_pocket_temporal(
                poses,
                fps=fps,
                aspect=aspect,
                frame_height=frame_height,
                frame_width=frame_width,
                config=config,
                timestamps=timestamps,
                row_mask=row_mask,
                keypoint_conf=keypoint_conf,
                config_path=config_path,
            )

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