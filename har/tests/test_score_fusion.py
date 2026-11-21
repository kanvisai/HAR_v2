"""Backend scores: legacy conf + fusión v2."""
from __future__ import annotations

from legacy_hand_to_body import legacy_hand_to_body_confidence
from pocket_filter.score_fusion import ScoreFusionConfig, fuse_scores
from pocket_filter.types import PocketResult


def test_legacy_confidence_increases_with_duration():
    c = legacy_hand_to_body_confidence(
        hand_to_body=True,
        duration_s=1.0,
        clip_s=8.0,
        geometry_strength=0.7,
    )
    assert 0.4 < c < 0.95


def test_score_high_v2_confirms():
    cfg = ScoreFusionConfig()
    v2 = PocketResult(
        resultado="COMPATIBLE",
        hand_to_body=True,
        hand_to_body_raw=True,
        filter_fp_safe=False,
        motivo="ACERCAMIENTO_EJE_TORSO",
        version="t",
        fps=12.5,
        n_frames=100,
        hand_to_body_confidence=0.85,
        motion_suppress=False,
    )
    fused = fuse_scores(
        v2, legacy_htb=False, legacy_conf=0.0, cotidian_id="walk", cotidian_conf=0.9, cfg=cfg
    )
    assert fused.hand_to_body and fused.rule == "SCORE_V2_HIGH"


def test_score_cotidian_beats_weak_v2():
    cfg = ScoreFusionConfig()
    v2 = PocketResult(
        resultado="COMPATIBLE",
        hand_to_body=True,
        hand_to_body_raw=True,
        filter_fp_safe=False,
        motivo="ACERCAMIENTO_EJE_TORSO",
        version="t",
        fps=12.5,
        n_frames=100,
        hand_to_body_confidence=0.50,
        motion_suppress=False,
    )
    fused = fuse_scores(
        v2, legacy_htb=False, legacy_conf=0.1, cotidian_id="walk", cotidian_conf=0.85, cfg=cfg
    )
    assert not fused.hand_to_body and fused.rule == "SCORE_COTIDIAN_WINS"


def test_score_strong_contact_keeps():
    cfg = ScoreFusionConfig()
    v2 = PocketResult(
        resultado="COMPATIBLE",
        hand_to_body=True,
        hand_to_body_raw=True,
        filter_fp_safe=False,
        motivo="CONTACTO_BOLSILLO",
        version="t",
        fps=12.5,
        n_frames=100,
        hand_to_body_confidence=0.58,
        motion_suppress=False,
    )
    fused = fuse_scores(
        v2, legacy_htb=False, legacy_conf=0.0, cotidian_id="walk", cotidian_conf=0.95, cfg=cfg
    )
    assert fused.hand_to_body and fused.rule == "SCORE_V2_STRONG_MOTIVE"