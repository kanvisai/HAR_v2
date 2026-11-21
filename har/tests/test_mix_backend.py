"""Fusión backend mix (v2 + legacy + cotidiano)."""
from __future__ import annotations

from pocket_filter.mix_backend import MixBackendConfig, fuse_mix_verdict
from pocket_filter.types import PocketResult


def _v2(clear: bool, conf: float, raw: bool = False) -> PocketResult:
    return PocketResult(
        resultado="COMPATIBLE" if clear or raw else "INCOMPATIBLE",
        hand_to_body=clear,
        hand_to_body_raw=raw or clear,
        filter_fp_safe=False,
        motivo="CONTACTO_BOLSILLO",
        version="t",
        fps=12.5,
        n_frames=100,
        hand_to_body_confidence=conf,
        motion_suppress=False,
    )


def test_high_conf_trusts_v2_without_legacy():
    cfg = MixBackendConfig(v2_clear_trust_confidence=0.80)
    r = _v2(True, 0.85)
    r.motivo = "ACERCAMIENTO_EJE_TORSO"
    fused = fuse_mix_verdict(
        r,
        legacy_htb=False,
        cotidian_id="walk",
        cotidian_conf=0.9,
        mcfg=cfg,
    )
    assert fused.hand_to_body and fused.mix_rule == "MIX_V2_HIGH_CONF"


def test_low_conf_legacy_veto():
    cfg = MixBackendConfig()
    r = _v2(True, 0.55)
    r.motivo = "ACERCAMIENTO_EJE_TORSO"
    fused = fuse_mix_verdict(
        r,
        legacy_htb=False,
        cotidian_id="inconclusive",
        cotidian_conf=0.1,
        mcfg=cfg,
    )
    assert not fused.hand_to_body and fused.mix_rule == "MIX_V2_LOW_LEGACY_VETO"


def test_contact_keeps_clear_despite_legacy():
    cfg = MixBackendConfig()
    r = _v2(True, 0.55)
    r.motivo = "CONTACTO_BOLSILLO"
    fused = fuse_mix_verdict(
        r,
        legacy_htb=False,
        cotidian_id="walk",
        cotidian_conf=0.95,
        mcfg=cfg,
    )
    assert fused.hand_to_body and fused.mix_rule == "MIX_V2_TRUST_POLICY"


def test_cotidian_overrides_weak_clear():
    cfg = MixBackendConfig()
    r = _v2(True, 0.55)
    r.motivo = "ACERCAMIENTO_EJE_TORSO"
    fused = fuse_mix_verdict(
        r,
        legacy_htb=False,
        cotidian_id="walk",
        cotidian_conf=0.85,
        mcfg=cfg,
    )
    assert not fused.hand_to_body and fused.mix_rule.startswith("MIX_COTIDIAN_OVERRIDE")


def test_raw_keeps_sam_path():
    cfg = MixBackendConfig()
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