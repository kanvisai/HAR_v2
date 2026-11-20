"""Confianza pocket + política HAR cotidiano."""
from __future__ import annotations

import numpy as np

from pocket_filter.action_policy import apply_action_policy, load_action_policy_config
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
    assert out.hand_to_body_confidence >= policy.hand_to_body_min_confidence


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