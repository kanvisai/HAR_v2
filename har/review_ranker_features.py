"""Features para ranker REVIEW (sin categoría en inferencia)."""
from __future__ import annotations

MOTIVES = (
    "CONTACTO_BOLSILLO",
    "APROXIMACION_OBSERVADA",
    "ACERCAMIENTO_EJE_TORSO",
)
FEATURE_NAMES = [
    "bias",
    "conf",
    "walk_frac",
    "stop_frac",
    "hands_away_frac",
    "mot_contact",
    "mot_approx",
    "mot_axis",
    "walk_x_axis",
    "walk_x_contact",
    "quiet_x_axis",
]


def _f(s: str | float) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(str(s).strip() or 0)
    except ValueError:
        return 0.0


def feature_vector(row: dict[str, str]) -> list[float]:
    conf = _f(row.get("hand_to_body_confidence"))
    walk = _f(row.get("ctx_walk_frac"))
    stop = _f(row.get("ctx_stop_frac"))
    away = _f(row.get("ctx_hands_away_frac"))
    m = (row.get("motivo") or "").strip()
    is_c = 1.0 if m == "CONTACTO_BOLSILLO" else 0.0
    is_a = 1.0 if m == "APROXIMACION_OBSERVADA" else 0.0
    is_x = 1.0 if m == "ACERCAMIENTO_EJE_TORSO" else 0.0
    quiet = 1.0 if stop >= 0.30 else 0.0
    return [
        1.0,
        conf,
        walk,
        stop,
        away,
        is_c,
        is_a,
        is_x,
        walk * is_x,
        walk * is_c,
        quiet * is_x,
    ]


def theft_label(category: int) -> int:
    return 1 if category in (6, 14) else 0