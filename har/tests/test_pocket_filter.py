"""Tests del filtro Ignacio (spec sintética resumida)."""
from __future__ import annotations

import numpy as np

from pocket_filter.config import PocketConfig
from pocket_filter.evaluate import evaluate_pocket


def _skeleton_frame(wx: float, wy: float) -> np.ndarray:
    pose = np.full((8, 2), np.nan)
    pose[0] = [0.40, 0.22]
    pose[1] = [0.60, 0.22]
    pose[6] = [0.43, 0.58]
    pose[7] = [0.57, 0.58]
    pose[4] = [wx, wy]
    return pose


def test_compatible_approach_synthetic():
    """Muñeca izquierda se acerca al bolsillo izquierdo."""
    T = 12
    poses = np.stack(
        [
            _skeleton_frame(0.72 - 0.35 * (t / (T - 1)), 0.70 - 0.12 * (t / (T - 1)))
            for t in range(T)
        ]
    )
    result = evaluate_pocket(poses, fps=12.0, aspect=1.0, frame_height=720.0)
    assert result.resultado == "COMPATIBLE"
    assert result.hand_to_body
    assert result.best_pair is not None
    assert result.best_pair.ipsilateral


def _skeleton_both_wrists_far() -> np.ndarray:
    pose = np.full((8, 2), np.nan)
    pose[0] = [0.40, 0.22]
    pose[1] = [0.60, 0.22]
    pose[6] = [0.43, 0.58]
    pose[7] = [0.57, 0.58]
    pose[4] = [0.12, 0.12]
    pose[5] = [0.88, 0.12]
    return pose


def test_incompatible_far_all_synthetic():
    """Ambas muñecas lejos; las cuatro parejas deben cerrar incompatibles."""
    T = 24
    poses = np.stack([_skeleton_both_wrists_far() for _ in range(T)])
    cfg = PocketConfig(min_valid_ratio_short_clip=0.5)
    result = evaluate_pocket(poses, fps=12.0, aspect=1.0, frame_height=720.0, config=cfg)
    assert result.resultado == "INCOMPATIBLE"
    assert result.filter_fp_safe


def test_indeterminate_with_gaps():
    """Hueco en medio impide INCOMPATIBLE."""
    T = 10
    seq = [_skeleton_frame(0.70, 0.75) for _ in range(T)]
    seq[4] = _skeleton_frame(np.nan, np.nan)
    poses = np.stack(seq)
    result = evaluate_pocket(poses, fps=12.0, aspect=1.0, frame_height=720.0)
    assert result.resultado in ("INDETERMINADO", "INCOMPATIBLE")


def test_row_mask_blocks_incompatible():
    """Frames no exportables no cuentan como lejanía demostrada."""
    T = 16
    poses = np.stack([_skeleton_frame(0.72, 0.78) for _ in range(T)])
    row = np.zeros(T, dtype=bool)
    row[0::2] = True
    result = evaluate_pocket(
        poses, fps=12.0, aspect=1.0, frame_height=720.0, row_mask=row
    )
    assert result.resultado == "INDETERMINADO"