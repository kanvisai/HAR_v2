"""
Fusión v2 (pocket + política + temporal) + legacy + cotidiano en clips sin categoría.

Regla producción (sin etiqueta):
  1. Siempre evaluar v2 (raw, conf, clear, motion_suppress).
  2. Si clear v2 y conf >= umbral → confiar en v2 (no hace falta legacy para decidir).
  3. Si conf < umbral o clear dudoso → legacy + acción cotidiana dominante;
     otra acción con conf mayor desplaza la confirmación clara.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from har_actions.detect import detect_cotidian_actions, gesture_seconds, visible_seconds
from har_actions.pose import smooth
from legacy_hand_to_body import evaluate_legacy
from pocket_filter.action_policy import COTIDIAN_PRIMARY, evaluate_pocket_with_policy
from pocket_filter.config import CONFIG_PATH, PocketConfig, load_pocket_config
from pocket_filter.types import PocketResult

_DEFAULT_OVERRIDE_ACTIONS = frozenset({"walk", "stop", "arm_swing", "carry", "hand_up", "turn_torso"})


@dataclass
class MixBackendConfig:
    enabled: bool = True
    v2_clear_trust_confidence: float = 0.80
    cotidian_override_min_confidence: float = 0.80
    cotidian_min_visible_frac: float = 0.08
    legacy_agree_promotes_clear: bool = True
    legacy_veto_when_low_conf: bool = True


@dataclass(frozen=True)
class MixFusion:
    hand_to_body: bool
    motion_suppress: bool
    mix_rule: str
    legacy_hand_to_body: bool
    cotidian_action: str
    cotidian_confidence: float


def load_mix_backend_config(path: Path | None = None) -> MixBackendConfig:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        return MixBackendConfig()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data.get("mix_backend", {})
    fields = {f.name for f in MixBackendConfig.__dataclass_fields__.values()}  # type: ignore
    kwargs = {k: v for k, v in block.items() if k in fields}
    return MixBackendConfig(**kwargs)


def _cotidian_peak(
    poses: np.ndarray,
    *,
    fps: float,
    aspect: float,
    min_frac: float,
) -> tuple[str, float]:
    smoothed = smooth(poses, window=3)
    actions = detect_cotidian_actions(smoothed, fps, aspect)
    seconds = gesture_seconds(actions, fps)
    visible = max(visible_seconds(poses, fps), 1e-6)
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


def fuse_mix_verdict(
    v2: PocketResult,
    *,
    legacy_htb: bool,
    cotidian_id: str,
    cotidian_conf: float,
    mcfg: MixBackendConfig,
) -> MixFusion:
    v2_conf = float(v2.hand_to_body_confidence)
    v2_clear = bool(v2.hand_to_body)
    v2_raw = bool(v2.hand_to_body_raw)

    if v2_clear and v2_conf >= mcfg.v2_clear_trust_confidence:
        return MixFusion(
            hand_to_body=True,
            motion_suppress=bool(v2.motion_suppress),
            mix_rule="MIX_V2_HIGH_CONF",
            legacy_hand_to_body=legacy_htb,
            cotidian_action=cotidian_id,
            cotidian_confidence=cotidian_conf,
        )

    if v2_clear and v2_conf < mcfg.v2_clear_trust_confidence:
        if mcfg.legacy_agree_promotes_clear and legacy_htb:
            return MixFusion(
                hand_to_body=True,
                motion_suppress=False,
                mix_rule="MIX_V2_LOW_LEGACY_AGREE",
                legacy_hand_to_body=legacy_htb,
                cotidian_action=cotidian_id,
                cotidian_confidence=cotidian_conf,
            )
        if (
            cotidian_id in _DEFAULT_OVERRIDE_ACTIONS
            and cotidian_conf >= mcfg.cotidian_override_min_confidence
            and cotidian_conf > v2_conf
        ):
            suppress = not v2_raw
            return MixFusion(
                hand_to_body=False,
                motion_suppress=suppress,
                mix_rule=f"MIX_COTIDIAN_OVERRIDE_{cotidian_id.upper()}",
                legacy_hand_to_body=legacy_htb,
                cotidian_action=cotidian_id,
                cotidian_confidence=cotidian_conf,
            )
        if mcfg.legacy_veto_when_low_conf and not legacy_htb:
            suppress = not v2_raw
            return MixFusion(
                hand_to_body=False,
                motion_suppress=suppress,
                mix_rule="MIX_V2_LOW_LEGACY_VETO",
                legacy_hand_to_body=legacy_htb,
                cotidian_action=cotidian_id,
                cotidian_confidence=cotidian_conf,
            )
        return MixFusion(
            hand_to_body=True,
            motion_suppress=bool(v2.motion_suppress),
            mix_rule="MIX_V2_LOW_KEEP_CLEAR",
            legacy_hand_to_body=legacy_htb,
            cotidian_action=cotidian_id,
            cotidian_confidence=cotidian_conf,
        )

    if not v2_raw:
        suppress = True if not v2_clear else bool(v2.motion_suppress)
        if (
            cotidian_id in _DEFAULT_OVERRIDE_ACTIONS
            and cotidian_conf >= mcfg.cotidian_override_min_confidence
        ):
            return MixFusion(
                hand_to_body=False,
                motion_suppress=True,
                mix_rule=f"MIX_NO_CLEAR_COTIDIAN_{cotidian_id.upper()}",
                legacy_hand_to_body=legacy_htb,
                cotidian_action=cotidian_id,
                cotidian_confidence=cotidian_conf,
            )
        return MixFusion(
            hand_to_body=False,
            motion_suppress=suppress,
            mix_rule="MIX_V2_NO_CLEAR",
            legacy_hand_to_body=legacy_htb,
            cotidian_action=cotidian_id,
            cotidian_confidence=cotidian_conf,
        )

    if legacy_htb and v2_conf < mcfg.v2_clear_trust_confidence:
        return MixFusion(
            hand_to_body=False,
            motion_suppress=False,
            mix_rule="MIX_RAW_LEGACY_SUGGESTS",
            legacy_hand_to_body=legacy_htb,
            cotidian_action=cotidian_id,
            cotidian_confidence=cotidian_conf,
        )

    return MixFusion(
        hand_to_body=False,
        motion_suppress=False,
        mix_rule="MIX_RAW_SAM_PATH",
        legacy_hand_to_body=legacy_htb,
        cotidian_action=cotidian_id,
        cotidian_confidence=cotidian_conf,
    )


def evaluate_pocket_mix(
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
    use_temporal: bool | None = None,
) -> PocketResult:
    cfg = config or load_pocket_config(config_path)
    mcfg = load_mix_backend_config(config_path)

    v2 = evaluate_pocket_with_policy(
        poses,
        fps=fps,
        aspect=aspect,
        frame_height=frame_height,
        frame_width=frame_width,
        config=cfg,
        timestamps=timestamps,
        row_mask=row_mask,
        keypoint_conf=keypoint_conf,
        config_path=config_path,
        use_temporal=use_temporal,
    )
    legacy = evaluate_legacy(poses, fps=fps, aspect=aspect)
    cot_id, cot_conf = _cotidian_peak(
        poses,
        fps=fps,
        aspect=aspect,
        min_frac=mcfg.cotidian_min_visible_frac,
    )

    raw_v2 = bool(v2.hand_to_body_raw if v2.hand_to_body_raw is not None else v2.hand_to_body)
    fused = fuse_mix_verdict(
        v2,
        legacy_htb=bool(legacy.hand_to_body),
        cotidian_id=cot_id,
        cotidian_conf=cot_conf,
        mcfg=mcfg,
    )

    v2.hand_to_body_raw = raw_v2
    v2.legacy_hand_to_body = legacy.hand_to_body
    v2.hand_to_body = fused.hand_to_body
    v2.motion_suppress = fused.motion_suppress
    if fused.mix_rule.startswith("MIX_"):
        if v2.context_veto:
            v2.motivo = f"{v2.motivo}|{fused.mix_rule}" if v2.motivo else fused.mix_rule
        else:
            v2.context_veto = fused.mix_rule
    if fused.hand_to_body:
        v2.htb_tier = "alert"
        v2.primary_action = "hand_to_body"
        v2.primary_action_confidence = v2.hand_to_body_confidence
    elif v2.hand_to_body_raw:
        v2.htb_tier = "none"
        if not v2.primary_action or v2.primary_action == "hand_to_body":
            v2.primary_action = "hand_to_body"
            v2.primary_action_confidence = v2.hand_to_body_confidence
    elif fused.cotidian_action != "inconclusive" and fused.cotidian_confidence > 0:
        v2.primary_action = fused.cotidian_action
        v2.primary_action_confidence = fused.cotidian_confidence

    return v2