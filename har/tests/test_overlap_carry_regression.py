"""Regresión: compra / carry no debe confirmar clear (sesión overlap chunk)."""
from pathlib import Path

import pytest

from pocket_infer import infer_from_poses_path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "overlap_carry"


@pytest.mark.parametrize(
    "name",
    ["poses_full_002.npy", "poses_full_003.npy"],
)
def test_carry_overlap_chunks_not_clear(name: str):
    path = FIXTURE_DIR / name
    if not path.is_file():
        pytest.skip("fixtures overlap_carry no presentes")
    r = infer_from_poses_path(path)
    assert r.hand_to_body_raw, f"{name}: debería seguir viendo gesto pocket (raw)"
    assert not r.hand_to_body, f"{name}: no debe confirmar clear sin contacto claro"