"""Puente pocket_filter → har_actions.Action."""
from __future__ import annotations

import numpy as np

from har_actions.detect import Action
from pocket_filter.evaluate import evaluate_pocket
from pocket_filter.config import load_pocket_config


def detect_hand_to_body_actions(
    poses: np.ndarray,
    fps: float,
    aspect: float = 16 / 9,
    frame_height: float = 1080.0,
    row_mask: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> list[Action]:
    result = evaluate_pocket(
        poses,
        fps=fps,
        aspect=aspect,
        frame_height=frame_height,
        config=load_pocket_config(),
        timestamps=timestamps,
        row_mask=row_mask,
    )
    if not result.hand_to_body or result.best_pair is None:
        return []

    bp = result.best_pair
    i0 = result.eval_start
    i1 = result.eval_end
    if bp.t_entrada is not None:
        i0 = int(round(bp.t_entrada * fps))
    if bp.t_retirada is not None:
        i1 = int(round(bp.t_retirada * fps))
    else:
        i1 = min(len(poses) - 1, i1)
    i0 = max(0, min(i0, len(poses) - 1))
    i1 = max(i0, min(i1, len(poses) - 1))

    wrist_side = bp.wrist
    detail = (
        f"Pocket v2: {bp.wrist} wrist → {bp.pocket} pocket. "
        f"Approach from d={bp.d_origen:.2f} to d={bp.d_entrada:.2f}."
    )
    return [
        Action(
            "hand_to_body",
            "Hand to the body",
            wrist_side,
            i0,
            i1,
            detail,
        )
    ]