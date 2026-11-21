"""Fusión backend mix (v2_first + veto eje/walk)."""
from __future__ import annotations

from pocket_filter.mix_backend import MixBackendConfig, fuse_mix_verdict
from pocket_filter.types import PocketResult


def _v2(clear: bool, conf: float, raw: bool = False, motivo: str = "CONTACTO_BOLSILLO"):
    return PocketResult(
        resultado="COMPATIBLE" if clear or raw else "INCOMPATIBLE",
        hand_to_body=clear,
        hand_to_body_raw=raw or clear,
        filter_fp_safe=False,
        motivo=motivo,
        version="t",
        fps=12.5,
        n_frames=100,
        hand_to_body_confidence=conf,
        motion_suppress=False,
    )


def test_v2_first_keeps_contact_clear():
    cfg = MixBackendConfig(mix_mode="v2_first")
    fused = fuse_mix_verdict(
        _v2(True, 0.55),
        legacy_htb=False,
        cotidian_id="walk",
        cotidian_conf=0.95,
        mcfg=cfg,
    )
    assert fused.hand_to_body and fused.mix_rule == "MIX_V2_PASS"


def test_v2_first_axis_walk_veto():
    cfg = MixBackendConfig(mix_mode="v2_first")
    r = _v2(True, 0.55, motivo="ACERCAMIENTO_EJE_TORSO")
    fused = fuse_mix_verdict(
        r,
        legacy_htb=False,
        cotidian_id="walk",
        cotidian_conf=0.85,
        mcfg=cfg,
    )
    assert not fused.hand_to_body and fused.mix_rule == "MIX_AXIS_WALK_VETO"


def test_v2_first_axis_quiet_keeps_clear():
    cfg = MixBackendConfig(mix_mode="v2_first")
    r = _v2(True, 0.55, motivo="ACERCAMIENTO_EJE_TORSO")
    fused = fuse_mix_verdict(
        r,
        legacy_htb=False,
        cotidian_id="stop",
        cotidian_conf=0.85,
        mcfg=cfg,
    )
    assert fused.hand_to_body and fused.mix_rule == "MIX_V2_PASS"


def test_legacy_arbitrate_rejects_low_conf():
    cfg = MixBackendConfig(mix_mode="legacy_arbitrate", conf_reject_max=0.10)
    fused = fuse_mix_verdict(
        _v2(False, 0.05, raw=False),
        legacy_htb=True,
        cotidian_id="walk",
        cotidian_conf=0.9,
        mcfg=cfg,
    )
    assert not fused.hand_to_body and fused.mix_rule == "MIX_CONF_TOO_LOW"


def test_legacy_arbitrate_band_legacy_no():
    cfg = MixBackendConfig(mix_mode="legacy_arbitrate")
    r = _v2(True, 0.40, raw=True, motivo="ACERCAMIENTO_EJE_TORSO")
    r.hand_to_body_raw = True
    fused = fuse_mix_verdict(
        r,
        legacy_htb=False,
        cotidian_id="inconclusive",
        cotidian_conf=0.0,
        mcfg=cfg,
    )
    assert not fused.hand_to_body and fused.mix_rule == "MIX_LEGACY_ARBITRATE_NO_RAW"


def test_legacy_arbitrate_band_legacy_yes():
    cfg = MixBackendConfig(mix_mode="legacy_arbitrate")
    r = _v2(True, 0.40, raw=True, motivo="ACERCAMIENTO_EJE_TORSO")
    fused = fuse_mix_verdict(
        r,
        legacy_htb=True,
        cotidian_id="inconclusive",
        cotidian_conf=0.0,
        mcfg=cfg,
    )
    assert fused.hand_to_body and fused.mix_rule == "MIX_LEGACY_ARBITRATE_YES"


def test_raw_keeps_sam_path():
    cfg = MixBackendConfig(mix_mode="v2_first")
    fused = fuse_mix_verdict(
        _v2(False, 0.40, raw=True),
        legacy_htb=False,
        cotidian_id="inconclusive",
        cotidian_conf=0.0,
        mcfg=cfg,
    )
    assert not fused.hand_to_body
    assert not fused.motion_suppress
    assert fused.mix_rule == "MIX_RAW_SAM_PATH"