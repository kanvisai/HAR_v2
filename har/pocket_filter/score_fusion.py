"""
Fusión por scores: v2 (pocket) + legacyv2 (conf) + cotidiano (walk/stop, …).

Sin categoría: confirmar claro con umbrales v2; legacy refuerza; pasillo gana si
cotidiano domina y v2 no es fuerte.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from legacy_hand_to_body import evaluate_legacy
from pocket_filter.action_policy import evaluate_pocket_with_policy
from pocket_filter.config import CONFIG_PATH, PocketConfig, load_pocket_config
from pocket_filter.mix_backend import _cotidian_peak
from pocket_filter.types import PocketResult

_STRONG = frozenset({"CONTACTO_BOLSILLO"})
_APPROX = frozenset({"APROXIMACION_OBSERVADA"})
_COTIDIAN_PASSAGE = frozenset({"walk", "stop", "arm_swing", "carry"})


@dataclass
class ScoreFusionConfig:
    enabled: bool = True
    confirm_clear_min_v2: float = 0.80
    confirm_strong_min_v2: float = 0.55
    confirm_with_legacy_min_v2: float = 0.50
    confirm_legacy_min: float = 0.45
    approx_strong_min_v2: float = 0.52
    suppress_cotidian_min: float = 0.80
    suppress_v2_below: float = 0.55
    suppress_legacy_below: float = 0.35
    cotidian_min_visible_frac: float = 0.08


@dataclass(frozen=True)
class ScoreFusion:
    hand_to_body: bool
    motion_suppress: bool
    rule: str
    legacy_confidence: float
    cotidian_action: str
    cotidian_confidence: float


def load_score_fusion_config(path: Path | None = None) -> ScoreFusionConfig:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        return ScoreFusionConfig()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    block = data.get("score_fusion", {})
    fields = {f.name for f in ScoreFusionConfig.__dataclass_fields__.values()}  # type: ignore
    kwargs = {k: v for k, v in block.items() if k in fields}
    return ScoreFusionConfig(**kwargs)


def _motivo_base(motivo: str) -> str:
    return motivo.split("|", 1)[0].strip()


def _v2_strong_clear(motivo: str, v2_conf: float, cfg: ScoreFusionConfig) -> bool:
    base = _motivo_base(motivo)
    if base in _STRONG:
        return v2_conf >= cfg.confirm_strong_min_v2
    if base in _APPROX:
        return v2_conf >= cfg.approx_strong_min_v2
    return False


def fuse_scores(
    v2: PocketResult,
    *,
    legacy_htb: bool,
    legacy_conf: float,
    cotidian_id: str,
    cotidian_conf: float,
    cfg: ScoreFusionConfig,
) -> ScoreFusion:
    v2_clear = bool(v2.hand_to_body)
    v2_conf = float(v2.hand_to_body_confidence)
    v2_raw = bool(v2.hand_to_body_raw)
    meta = legacy_conf, cotidian_id, cotidian_conf

    def pack(htb: bool, sup: bool, rule: str) -> ScoreFusion:
        return ScoreFusion(
            hand_to_body=htb,
            motion_suppress=sup,
            rule=rule,
            legacy_confidence=meta[0],
            cotidian_action=meta[1],
            cotidian_confidence=meta[2],
        )

    if v2_clear and v2_conf >= cfg.confirm_clear_min_v2:
        return pack(True, bool(v2.motion_suppress), "SCORE_V2_HIGH")

    if v2_clear and _v2_strong_clear(v2.motivo, v2_conf, cfg):
        return pack(True, bool(v2.motion_suppress), "SCORE_V2_STRONG_MOTIVE")

    if (
        v2_clear
        and v2_conf >= cfg.confirm_with_legacy_min_v2
        and legacy_htb
        and legacy_conf >= cfg.confirm_legacy_min
    ):
        return pack(True, False, "SCORE_V2_LEGACY_BOOST")

    if v2_clear:
        if (
            cotidian_id in _COTIDIAN_PASSAGE
            and cotidian_conf >= cfg.suppress_cotidian_min
            and cotidian_conf > v2_conf
            and not _v2_strong_clear(v2.motivo, v2_conf, cfg)
        ):
            return pack(False, not v2_raw, "SCORE_COTIDIAN_WINS")
        if (
            not legacy_htb
            and legacy_conf < cfg.suppress_legacy_below
            and v2_conf < cfg.suppress_v2_below
            and not _v2_strong_clear(v2.motivo, v2_conf, cfg)
        ):
            return pack(False, not v2_raw, "SCORE_LEGACY_WEAK_VETO")
        return pack(True, bool(v2.motion_suppress), "SCORE_V2_KEEP")

    if v2_raw:
        return pack(False, False, "SCORE_RAW_SAM")

    if cotidian_conf >= cfg.suppress_cotidian_min:
        return pack(False, True, f"SCORE_PASSAGE_{cotidian_id.upper()}")

    if not legacy_htb and legacy_conf < cfg.suppress_legacy_below:
        return pack(False, True, "SCORE_NO_GESTURE")

    return pack(False, bool(v2.motion_suppress), "SCORE_V2_NO_CLEAR")


def evaluate_pocket_scores(
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
    cfg_pocket = config or load_pocket_config(config_path)
    fcfg = load_score_fusion_config(config_path)

    v2 = evaluate_pocket_with_policy(
        poses,
        fps=fps,
        aspect=aspect,
        frame_height=frame_height,
        frame_width=frame_width,
        config=cfg_pocket,
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
        min_frac=fcfg.cotidian_min_visible_frac,
    )

    raw_v2 = bool(v2.hand_to_body_raw if v2.hand_to_body_raw is not None else v2.hand_to_body)
    v2.hand_to_body_v2 = bool(v2.hand_to_body)

    fused = fuse_scores(
        v2,
        legacy_htb=legacy.hand_to_body,
        legacy_conf=legacy.confidence,
        cotidian_id=cot_id,
        cotidian_conf=cot_conf,
        cfg=fcfg,
    )

    v2.hand_to_body_raw = raw_v2
    v2.legacy_hand_to_body = legacy.hand_to_body
    v2.legacy_hand_to_body_confidence = legacy.confidence
    v2.hand_to_body = fused.hand_to_body
    v2.motion_suppress = fused.motion_suppress
    v2.context_veto = fused.rule
    if fused.hand_to_body:
        v2.htb_tier = "alert"
        v2.primary_action = "hand_to_body"
        v2.primary_action_confidence = v2.hand_to_body_confidence
    elif v2.hand_to_body_raw:
        v2.htb_tier = "none"
        v2.primary_action = "hand_to_body"
        v2.primary_action_confidence = v2.hand_to_body_confidence
    elif cot_conf > 0 and cot_id != "inconclusive":
        v2.primary_action = cot_id
        v2.primary_action_confidence = cot_conf

    return v2