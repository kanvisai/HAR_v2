"""Ventanas deslizantes en chunks ~8 s."""
from __future__ import annotations

from pocket_filter.temporal_eval import (
    TemporalConfig,
    load_temporal_config,
    sliding_windows,
)


def test_sliding_windows_8s_chunk():
    wins = sliding_windows(8.0, window_s=2.5, step_s=1.25)
    assert len(wins) >= 5
    assert wins[0][0] == 0.0
    assert wins[-1][1] <= 8.0


def test_short_clip_single_window():
    wins = sliding_windows(2.0, window_s=2.5, step_s=1.25)
    assert wins == [(0.0, 2.0)]


def test_temporal_config_loads():
    cfg = load_temporal_config()
    assert isinstance(cfg, TemporalConfig)