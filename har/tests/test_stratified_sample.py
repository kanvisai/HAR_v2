"""Muestra estratificada por categoría en batch."""
from __future__ import annotations

from pathlib import Path

from batch_data_result import ClipRef, stratified_sample_clips


def test_stratified_sample_ten_percent_per_category():
    clips = [
        ClipRef(Path(f"/c{n}/poses_full.npy"), category=n % 3, data_root=Path("/c"))
        for n in range(100)
    ]
    out = stratified_sample_clips(clips, 0.1, seed=1, min_per_cat=1)
    by_cat: dict[int, int] = {}
    for c in out:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
    assert len(out) == sum(by_cat.values())
    for cat in (0, 1, 2):
        assert 3 <= by_cat[cat] <= 4