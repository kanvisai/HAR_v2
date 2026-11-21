from review_ranker_features import FEATURE_NAMES, feature_vector, theft_label


def test_feature_dim():
    row = {
        "hand_to_body_confidence": "0.5",
        "ctx_walk_frac": "0.5",
        "ctx_stop_frac": "0.2",
        "ctx_hands_away_frac": "0.1",
        "motivo": "ACERCAMIENTO_EJE_TORSO",
    }
    v = feature_vector(row)
    assert len(v) == len(FEATURE_NAMES)
    assert v[FEATURE_NAMES.index("walk_x_axis")] == 0.5


def test_theft_label():
    assert theft_label(6) == 1
    assert theft_label(0) == 0