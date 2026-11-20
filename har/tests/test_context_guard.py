"""Context guard: walk / stop / manos alejadas."""
from __future__ import annotations

import numpy as np

from pocket_filter.context_guard import (
    ContextGuardConfig,
    apply_context_guard,
    decide_context_veto,
)
from pocket_filter.config import PocketConfig
from pocket_filter.evaluate import evaluate_pocket
from pocket_filter.types import PocketResult


def test_veto_walking_on_weak_motive():
    signals = {"walk_frac": 0.6, "walk_s": 2.0, "stop_frac": 0.1, "hands_away_frac": 0.2}
    cfg = ContextGuardConfig(veto_weak_motives_only=True)
    assert decide_context_veto(signals, motivo="APROXIMACION_OBSERVADA", gcfg=cfg) == "CONTEXT_VETO_WALKING"
    assert decide_context_veto(signals, motivo="CONTACTO_BOLSILLO", gcfg=cfg) is None


def test_context_guard_disabled_keeps_hand_to_body():
    T = 12
    poses = np.stack(
        [
            np.array(
                [
                    [0.40, 0.22],
                    [0.60, 0.22],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.72 - 0.35 * (t / (T - 1)), 0.70 - 0.12 * (t / (T - 1))],
                    [np.nan, np.nan],
                    [0.43, 0.58],
                    [0.57, 0.58],
                ]
            )
            for t in range(T)
        ]
    )
    result = evaluate_pocket(poses, fps=12.0, aspect=1.0, frame_height=720.0)
    assert result.hand_to_body
    gcfg = ContextGuardConfig(enabled=False)
    out = apply_context_guard(result, poses, fps=12.0, aspect=1.0, gcfg=gcfg)
    assert out.hand_to_body
    assert out.context_veto is None