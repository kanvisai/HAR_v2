"""Timestamps: epoch absoluto → t relativo; sidecar inválido → fps."""
from __future__ import annotations

import numpy as np

from pocket_filter.timestamp_utils import prepare_frame_timestamps


def test_epoch_timestamps_normalized_to_relative():
    n, fps = 96, 12.5
    t0 = 1_795_459_449.0
    ts = t0 + np.arange(n, dtype=np.float64) / fps
    out, reason = prepare_frame_timestamps(ts, n, fps)
    assert reason is None
    assert out is not None
    assert out[0] == 0.0
    assert abs(float(out[-1]) - (n - 1) / fps) < 0.05


def test_length_mismatch_rejects():
    ts = np.zeros(50)
    out, reason = prepare_frame_timestamps(ts, 96, 12.5)
    assert out is None and reason == "length_mismatch"


def test_good_relative_timestamps_ok():
    n, fps = 96, 12.0
    ts = np.linspace(0, (n - 1) / fps, n)
    out, reason = prepare_frame_timestamps(ts, n, fps)
    assert reason is None and out is not None