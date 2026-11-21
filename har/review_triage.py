"""Prioridad de cola REVIEW (raw∧¬clear) sin etiqueta de categoría en runtime."""
from __future__ import annotations

STRONG = frozenset({"CONTACTO_BOLSILLO"})
APPROACH = frozenset({"APROXIMACION_OBSERVADA"})
AXIS = frozenset({"ACERCAMIENTO_EJE_TORSO"})
WALK_FRAC_ACTIVE = 0.44


def review_priority_score(
    motivo: str,
    conf: float,
    walk_frac: float,
    *,
    stop_frac: float = 0.0,
) -> float:
    """
    Mayor = revisar antes (SAM/humano). Heurística P2-b; calibrar con analyze_review_separation.py.
    """
    m = (motivo or "").strip()
    score = float(conf)
    if m in STRONG:
        score += 0.30
    elif m in APPROACH:
        score += 0.15
    elif m in AXIS:
        score -= 0.18
    if walk_frac >= WALK_FRAC_ACTIVE:
        if m in AXIS | APPROACH:
            score -= 0.28
        elif m in STRONG:
            score -= 0.06
    if stop_frac >= 0.30 and m in AXIS:
        score += 0.08
    return round(max(0.0, min(1.5, score)), 4)