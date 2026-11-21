#!/usr/bin/env python3
"""
Ajuste offline de pesos review_triage sobre all_cohorts.csv (con etiqueta cohort).

No usa categoría en inferencia; solo calibra pesos con muestra etiquetada.

Uso:
  .venv/bin/python3 tune_review_triage.py /tmp/cohorts2/all_cohorts.csv
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

POS = frozenset({"fn_cat6_raw_review", "fn_cat6_indeterminate_keep", "fn_cat14_policy_gap"})
NEG = frozenset({"fp_cat0_raw_review", "fp_cat0_confirm", "pasillo_review_raw"})
MOTIVES = ("CONTACTO_BOLSILLO", "APROXIMACION_OBSERVADA", "ACERCAMIENTO_EJE_TORSO")


def _f(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def score_row(
    r: dict[str, str],
    *,
    w_conf: float,
    w_contact: float,
    w_approx: float,
    w_axis: float,
    w_walk_weak: float,
    w_walk_contact: float,
) -> float:
    m = (r.get("motivo") or "").strip()
    conf = _f(r.get("hand_to_body_confidence"))
    walk = _f(r.get("ctx_walk_frac"))
    s = w_conf * conf
    if m == "CONTACTO_BOLSILLO":
        s += w_contact
    elif m == "APROXIMACION_OBSERVADA":
        s += w_approx
    elif m == "ACERCAMIENTO_EJE_TORSO":
        s += w_axis
    if walk >= 0.44:
        if m in ("ACERCAMIENTO_EJE_TORSO", "APROXIMACION_OBSERVADA"):
            s += w_walk_weak
        elif m == "CONTACTO_BOLSILLO":
            s += w_walk_contact
    return s


def recall_at_k(scored: list[tuple[float, str]], k: int) -> tuple[int, float]:
    top = scored[:k]
    pos = sum(1 for _, lb in top if lb == "pos")
    n_pos = sum(1 for _, lb in scored if lb == "pos")
    return pos, (pos / n_pos if n_pos else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("-k", type=int, default=100)
    args = ap.parse_args()
    with args.csv.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("cohort") in POS | NEG]

    best = None
    grid_conf = (0.8, 1.0, 1.2)
    grid_contact = (0.2, 0.35, 0.5)
    grid_axis = (-0.35, -0.2, -0.1)
    grid_walk = (-0.4, -0.28, -0.15)
    for wc, wct, wax, wwk in product(grid_conf, grid_contact, grid_axis, grid_walk):
        weights = dict(
            w_conf=wc,
            w_contact=wct,
            w_approx=0.12,
            w_axis=wax,
            w_walk_weak=wwk,
            w_walk_contact=-0.05,
        )
        scored = []
        for r in rows:
            coh = r["cohort"]
            lb = "pos" if coh in POS else "neg"
            scored.append((score_row(r, **weights), lb))
        scored.sort(key=lambda x: -x[0])
        p, rec = recall_at_k(scored, args.k)
        prec = p / args.k if args.k else 0
        key = (rec, prec, p)
        if best is None or key > best[0]:
            best = (key, weights, p, prec)

    assert best
    (rec, prec, p), weights, _, _ = best
    print(f"Mejor en grid (K={args.k}): recall_pos={rec:.1%}  prec={prec:.1%}  pos_en_top={p}")
    print("Pesos:", weights)
    print("(Sigue peor que mezclar todo; si recall@100 < 20%, hace falta score Motion o features nuevas P1-1)")


if __name__ == "__main__":
    main()