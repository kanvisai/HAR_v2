import numpy as np

from pocket_filter.kinematics import _swing_peak_count


def test_swing_peak_count_simple():
    d = np.array([0.5, 0.4, 0.5, 0.35, 0.48, 0.3, 0.45, 0.32])
    valid = np.ones(len(d), dtype=bool)
    assert _swing_peak_count(d, valid, min_prominence=0.04) == 3