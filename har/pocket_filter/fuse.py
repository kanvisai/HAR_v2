"""Fusión de las 4 parejas muñeca×bolsillo."""
from __future__ import annotations

from pocket_filter.config import PocketConfig
from pocket_filter.types import PairEvidence, Verdict

IPSI = {
    ("left", "left"),
    ("right", "right"),
}


def _compatible_score(pair: PairEvidence, cfg: PocketConfig) -> float:
    if pair.d_origen is None or pair.d_entrada is None:
        return 0.0
    delta = float(pair.d_origen - pair.d_entrada)
    if not pair.ipsilateral:
        delta -= cfg.contralateral_delta_bonus
    if pair.d_min is not None:
        delta += max(0.0, cfg.r_near - pair.d_min)
    return delta


def fuse_pairs(
    pairs: list[PairEvidence], cfg: PocketConfig
) -> tuple[Verdict, str, PairEvidence | None]:
    compatibles = [p for p in pairs if p.verdict == "COMPATIBLE"]
    if compatibles:
        if cfg.prefer_ipsilateral:
            ipsi = [p for p in compatibles if p.ipsilateral]
            pool = ipsi if ipsi else compatibles
        else:
            pool = compatibles
        best = max(pool, key=lambda p: _compatible_score(p, cfg))
        return "COMPATIBLE", best.motivo, best

    if pairs and all(p.verdict == "INCOMPATIBLE" for p in pairs):
        return "INCOMPATIBLE", "LEJOS_TODO_EL_GESTO", None

    reasons = {p.motivo for p in pairs if p.verdict == "INDETERMINADO"}
    motivo = sorted(reasons)[0] if reasons else "INDETERMINADO"
    return "INDETERMINADO", motivo, None