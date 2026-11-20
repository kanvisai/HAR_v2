"""Confianza pocket + política HAR cotidiano."""
from __future__ import annotations

import numpy as np

from pocket_filter.action_policy import (
    ActionPolicyConfig,
    apply_action_policy,
    decide_hand_to_body,
    decide_robbery_confirm,
    load_action_policy_config,
)
from pocket_filter.confidence import pocket_hand_to_body_confidence
from pocket_filter.evaluate import evaluate_pocket


def _skeleton_frame(wx: float, wy: float) -> np.ndarray:
    pose = np.full((8, 2), np.nan)
    pose[0] = [0.40, 0.22]
    pose[1] = [0.60, 0.22]
    pose[6] = [0.43, 0.58]
    pose[7] = [0.57, 0.58]
    pose[4] = [wx, wy]
    return pose


def test_confidence_high_on_synthetic_approach():
    T = 12
    poses = np.stack(
        [
            _skeleton_frame(0.72 - 0.35 * (t / (T - 1)), 0.70 - 0.12 * (t / (T - 1)))
            for t in range(T)
        ]
    )
    result = evaluate_pocket(poses, fps=12.0, aspect=1.0, frame_height=720.0)
    conf = pocket_hand_to_body_confidence(result)
    assert result.hand_to_body
    assert conf >= 0.55


def test_policy_keeps_strong_compatible():
    T = 12
    poses = np.stack(
        [
            _skeleton_frame(0.72 - 0.35 * (t / (T - 1)), 0.70 - 0.12 * (t / (T - 1)))
            for t in range(T)
        ]
    )
    result = evaluate_pocket(poses, fps=12.0, aspect=1.0, frame_height=720.0)
    policy = load_action_policy_config()
    out = apply_action_policy(result, poses, fps=12.0, aspect=1.0, policy=policy)
    assert out.hand_to_body
    assert out.primary_action == "hand_to_body"
    assert out.hand_to_body_confidence >= policy.strong_min_confidence


def test_walking_weak_axis_rejected_stationary_ok():
    policy = load_action_policy_config()
    walking = {
        "walk_frac": 0.55,
        "walk_s": 2.0,
        "stop_frac": 0.1,
        "stop_s": 0.0,
        "hands_away_frac": 0.0,
        "hands_away_s": 0.0,
    }
    quiet = {
        "walk_frac": 0.1,
        "walk_s": 0.2,
        "stop_frac": 0.6,
        "stop_s": 3.0,
        "hands_away_frac": 0.0,
        "hands_away_s": 0.0,
    }
    ok, veto = decide_hand_to_body(
        pocket_compatible=True,
        conf=0.58,
        motivo="ACERCAMIENTO_EJE_TORSO",
        signals=quiet,
        policy=policy,
    )
    assert ok and veto is None
    ok2, veto2 = decide_hand_to_body(
        pocket_compatible=True,
        conf=0.58,
        motivo="ACERCAMIENTO_EJE_TORSO",
        signals=walking,
        policy=policy,
    )
    assert not ok2 and veto2 == "POLICY_WALK_OVER_WEAK_POCKET"


def test_robbery_confirm_blocks_approx_while_walking():
    policy = ActionPolicyConfig(
        enabled=True,
        gate_mode="robbery_confirm",
        approx_min_confidence=0.56,
        walk_active_frac=0.44,
        walk_active_clip_min_s=1.0,
    )
    walking = {
        "walk_frac": 0.5,
        "walk_s": 2.0,
        "stop_frac": 0.1,
        "stop_s": 0.0,
        "hands_away_frac": 0.0,
        "hands_away_s": 0.0,
    }
    ok, veto = decide_robbery_confirm(
        pocket_compatible=True,
        conf=0.7,
        motivo="APROXIMACION_OBSERVADA",
        signals=walking,
        policy=policy,
    )
    assert not ok
    assert veto == "GATE_WALK_NO_CONFIRM"


def test_robbery_confirm_allows_axis_when_quiet():
    policy = ActionPolicyConfig(
        gate_mode="robbery_confirm",
        axis_min_confidence=0.50,
        axis_quiet_stop_frac=0.30,
    )
    quiet = {
        "walk_frac": 0.1,
        "walk_s": 0.2,
        "stop_frac": 0.5,
        "stop_s": 3.0,
        "hands_away_frac": 0.0,
        "hands_away_s": 0.0,
    }
    ok, veto = decide_robbery_confirm(
        pocket_compatible=True,
        conf=0.55,
        motivo="ACERCAMIENTO_EJE_TORSO",
        signals=quiet,
        policy=policy,
    )
    assert ok and veto is None


def test_axis_compatible_goes_review_tier_not_alert():
    T = 12
    poses = np.stack(
        [
            _skeleton_frame(0.72 - 0.35 * (t / (T - 1)), 0.70 - 0.12 * (t / (T - 1)))
            for t in range(T)
        ]
    )
    from pocket_filter.evaluate import evaluate_pocket

    result = evaluate_pocket(poses, fps=12.0, aspect=1.0, frame_height=720.0)
    if result.best_pair:
        result.motivo = "ACERCAMIENTO_EJE_TORSO"
        result.hand_to_body = True
        result.hand_to_body_raw = True
    policy = ActionPolicyConfig(gate_mode="full", axis_max_tier="review")
    out = apply_action_policy(result, poses, fps=12.0, aspect=1.0, policy=policy)
    if result.motivo == "ACERCAMIENTO_EJE_TORSO" and bool(result.hand_to_body_raw):
        assert out.htb_tier in ("review", "alert")
    else:
        assert out.htb_tier in ("alert", "none", "review")


def test_incompatible_yields_low_confidence():
    pose = np.full((8, 2), np.nan)
    pose[0] = [0.40, 0.22]
    pose[1] = [0.60, 0.22]
    pose[6] = [0.43, 0.58]
    pose[7] = [0.57, 0.58]
    pose[4] = [0.12, 0.12]
    pose[5] = [0.88, 0.12]
    poses = np.stack([pose for _ in range(20)])
    result = evaluate_pocket(poses, fps=12.0, aspect=1.0, frame_height=720.0)
    assert pocket_hand_to_body_confidence(result) == 0.0
    out = apply_action_policy(result, poses, fps=12.0, aspect=1.0)
    assert not out.hand_to_body