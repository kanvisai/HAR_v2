"""Walk gate en fusión temporal (ventana ganadora + marcha)."""
from __future__ import annotations

from pocket_filter.temporal_eval import (
    TemporalConfig,
    _best_clear_window,
    _temporal_walk_gate_veto,
)
from pocket_filter.types import PocketResult


def _win(htb: bool, *, motivo: str, conf: float, walk_frac: float, walk_s: float):
    return (
        (0.0, 2.5),
        PocketResult(
            resultado="COMPATIBLE",
            hand_to_body=htb,
            hand_to_body_raw=htb,
            filter_fp_safe=False,
            motivo=motivo,
            version="t",
            fps=12.5,
            n_frames=100,
            hand_to_body_confidence=conf,
            ctx_walk_frac=walk_frac,
            ctx_walk_s=walk_s,
        ),
    )


def test_walk_gate_vetoes_axis_with_ambiguous_march():
    cfg = TemporalConfig(walk_gate_enabled=True)
    r = _win(True, motivo="ACERCAMIENTO_EJE_TORSO", conf=0.55, walk_frac=0.40, walk_s=1.2)[1]
    assert _temporal_walk_gate_veto(r, cfg) == "TEMPORAL_WALK_GATE"


def test_walk_gate_keeps_strong_contact_high_conf():
    cfg = TemporalConfig(walk_gate_enabled=True)
    r = _win(True, motivo="CONTACTO_BOLSILLO", conf=0.70, walk_frac=0.50, walk_s=2.0)[1]
    assert _temporal_walk_gate_veto(r, cfg) is None


def test_best_clear_falls_back_to_quiet_window():
    cfg = TemporalConfig(walk_gate_enabled=True)
    walking = _win(
        True, motivo="ACERCAMIENTO_EJE_TORSO", conf=0.60, walk_frac=0.50, walk_s=2.0
    )
    quiet = _win(
        True, motivo="CONTACTO_BOLSILLO", conf=0.72, walk_frac=0.15, walk_s=0.2
    )
    ok, iv, r, veto = _best_clear_window([walking, quiet], cfg)
    assert ok and r is not None and r.motivo == "CONTACTO_BOLSILLO"
    assert veto is None