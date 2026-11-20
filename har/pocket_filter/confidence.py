"""Score continuo [0,1] para evidencia pocket (sin sustituir el ternario)."""
from __future__ import annotations

from pocket_filter.config import PocketConfig
from pocket_filter.types import PocketResult, PairEvidence

MOTIVE_BASE: dict[str, float] = {
    "CONTACTO_BOLSILLO": 0.92,
    "APROXIMACION_OBSERVADA": 0.82,
    "ACERCAMIENTO_EJE_TORSO": 0.42,
}

WEAK_MOTIVES = frozenset(
    {
        "ACERCAMIENTO_EJE_TORSO",
        "APROXIMACION_OBSERVADA",
        "EVIDENCIA_PARCIAL",
        "ZONA_GRIS",
    }
)

INDETERMINATE_BASE: dict[str, float] = {
    "EVIDENCIA_PARCIAL": 0.28,
    "ZONA_GRIS": 0.18,
    "BALANCEO_BRAZO": 0.12,
    "VISTA_NO_APTA": 0.10,
    "SIN_DATOS_VALIDOS": 0.05,
}


def _evidence_strength(pair: PairEvidence, cfg: PocketConfig) -> float:
    if pair.approach_delta is not None and pair.approach_delta > 0:
        if pair.motivo == "ACERCAMIENTO_EJE_TORSO":
            need = cfg.axis_delta_min
        else:
            need = cfg.delta_min
        ratio = float(pair.approach_delta) / max(need, 1e-6)
        strength = min(1.0, 0.35 + 0.65 * ratio)
    elif pair.d_min is not None and pair.d_origen is not None:
        depth = max(0.0, float(pair.d_origen) - float(pair.d_min))
        need = cfg.delta_min
        strength = min(1.0, 0.4 + 0.6 * depth / max(need, 1e-6))
    else:
        strength = 0.45

    if pair.d_min is not None and pair.d_min <= cfg.deep_contact_r:
        strength = min(1.0, strength + 0.12)
    if pair.ipsilateral:
        strength = min(1.0, strength + 0.05)
    vr = max(0.0, min(1.0, float(pair.valid_ratio)))
    return strength * (0.55 + 0.45 * vr)


def pocket_hand_to_body_confidence(
    result: PocketResult,
    cfg: PocketConfig | None = None,
) -> float:
    """
    Confianza de que el clip contiene hand_to_body según pocket (antes de contexto HAR1).
    COMPATIBLE → alta si motivo fuerte; INCOMPATIBLE → ~0; INDETERMINADO → baja.
    """
    cfg = cfg or PocketConfig()
    if result.resultado == "INCOMPATIBLE":
        return 0.0
    if result.resultado == "INDETERMINADO":
        base = INDETERMINATE_BASE.get(result.motivo, 0.15)
        if result.pairs:
            vr = max(p.valid_ratio for p in result.pairs)
            base *= 0.5 + 0.5 * min(1.0, vr)
        return round(min(0.55, base), 4)

    if result.resultado != "COMPATIBLE" or result.best_pair is None:
        return 0.0

    bp = result.best_pair
    base = MOTIVE_BASE.get(bp.motivo, 0.55)
    if bp.motivo == "ACERCAMIENTO_EJE_TORSO" and bp.approach_delta is not None:
        need = cfg.axis_delta_min
        excess = max(0.0, float(bp.approach_delta) - need)
        base = min(0.68, 0.48 + 0.55 * min(1.0, excess / max(need, 1e-6) + 0.35))
    strength = _evidence_strength(bp, cfg)
    conf = base * strength
    return round(max(0.0, min(1.0, conf)), 4)