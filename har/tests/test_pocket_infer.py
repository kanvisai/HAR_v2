from pathlib import Path

import numpy as np

from pocket_infer import infer_from_poses_array, orchestrator_branch
from pocket_filter.types import PocketResult


def test_orchestrator_branch_doctrine():
    r = PocketResult(resultado="INDETERMINADO", hand_to_body=False, filter_fp_safe=False, motivo="ZONA_GRIS", version="t", fps=12, n_frames=10)
    r.hand_to_body_raw = False
    r.motion_suppress = False
    assert orchestrator_branch(r) == "KEEP_INDETERMINADO"

    r.motion_suppress = True
    assert orchestrator_branch(r) == "DISCARD_SUPPRESS"

    r.motion_suppress = False
    r.hand_to_body_raw = True
    assert orchestrator_branch(r) == "REVIEW_RAW"


def test_infer_from_array_smoke():
    poses = np.random.randn(24, 8, 2).astype(np.float64) * 0.1 + 0.5
    result = infer_from_poses_array(poses, fps=12.0, aspect=1.77)
    assert result.n_frames == 24
    assert isinstance(result.hand_to_body, bool)