from pocket_filter.kinematics import ClipKinematics, build_movement_summary


def test_tempo_natural_slow_body_high_conf():
    kin = ClipKinematics(
        hip_speed_norm=0.18,
        wrist_radial_vel_min=-0.2,
        near_dwell_s=0.6,
        approach_monotone_frac=0.9,
    )
    m = build_movement_summary(
        kin,
        stop_frac=0.5,
        hand_to_body_raw=True,
        hand_to_body=True,
        hand_to_body_confidence=0.82,
    )
    assert m["body_pace"] == "quiet"
    assert m["approach_pace"] == "slow"
    assert m["tempo_hint"] == "natural"


def test_tempo_hurried_fast_hip_without_gesture():
    kin = ClipKinematics(hip_speed_norm=0.7)
    m = build_movement_summary(kin, walk_frac=0.4, hand_to_body_raw=False, hand_to_body=False)
    assert m["body_pace"] == "fast"
    assert m["tempo_hint"] == "hurried"


def test_hip_speed_without_pocket_pair():
    kin = ClipKinematics(hip_speed_norm=0.31)
    m = build_movement_summary(kin)
    assert m["hip_speed_norm"] == 0.31
    assert m["approach_pace"] == "none"
    assert m["has_pocket_pair_kin"] is False