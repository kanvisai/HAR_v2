#!/usr/bin/env python3
"""
Mide si la prioridad REVIEW separa FN robo vs FP pasillo (muestra etiquetada).

Uso:
  .venv/bin/python3 analyze_review_separation.py /tmp/cohorts/all_cohorts.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from review_triage import review_priority_score

POS_COHORTS = frozenset(
    {
        "fn_cat6_raw_review",
        "fn_cat6_indeterminate_keep",
        "fn_cat14_policy_gap",
    }
)
NEG_COHORTS = frozenset(
    {
        "fp_cat0_raw_review",
        "fp_cat0_confirm",
        "pasillo_review_raw",
    }
)


def _f(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cohorts_csv", type=Path)
    ap.add_argument(
        "--top",
        type=int,
        default=100,
        help="Tamaños de cola a reportar",
    )
    args = ap.parse_args()
    with args.cohorts_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scored: list[tuple[float, str, dict[str, str]]] = []
    for r in rows:
        coh = r.get("cohort", "")
        if coh not in POS_COHORTS | NEG_COHORTS:
            continue
        sc = review_priority_score(
            r.get("motivo", ""),
            _f(r.get("hand_to_body_confidence")),
            _f(r.get("ctx_walk_frac")),
            stop_frac=_f(r.get("ctx_stop_frac")),
        )
        label = "pos" if coh in POS_COHORTS else "neg"
        scored.append((sc, label, r))

    scored.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, lb, _ in scored if lb == "pos")
    n_neg = sum(1 for _, lb, _ in scored if lb == "neg")
    print(f"Items etiquetados: pos={n_pos} (robo)  neg={n_neg} (pasillo FP/review)")
    print(f"Score min/median/max pos: {_stats([s for s, lb, _ in scored if lb == 'pos'])}")
    print(f"Score min/median/max neg: {_stats([s for s, lb, _ in scored if lb == 'neg'])}")

    print("\n=== Captura en cola priorizada (global pos+neg) ===")
    print("  top_K | pos_in_top | recall_pos | neg_in_top | prec_pos")
    for k in (50, 100, 200, 500, len(scored)):
        if k > len(scored):
            continue
        top = scored[:k]
        p = sum(1 for _, lb, _ in top if lb == "pos")
        n = sum(1 for _, lb, _ in top if lb == "neg")
        rec = p / n_pos if n_pos else 0
        prec = p / k if k else 0
        print(f"  {k:5d} | {p:9d} | {rec:8.1%} | {n:10d} | {prec:8.1%}")

    pri = [r for _, lb, r in scored if lb == "pos" and r.get("cohort") == "fn_cat6_raw_review"]
    if pri:
        from collections import Counter

        walk = sum(1 for r in pri if _f(r.get("ctx_walk_frac")) >= 0.44)
        print(f"\nfn_cat6_raw_review en análisis: {len(pri)}")
        print(f"  marcha alta: {walk}/{len(pri)}")
        print(f"  motivos: {Counter(r['motivo'] for r in pri).most_common()}")
        print(
            "\nConclusión: si eje+marcha domina FN y también FP, "
            "subir clear a ciegas falla; usar cola top-K con esta prioridad."
        )


def _stats(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    vals = sorted(vals)
    mid = vals[len(vals) // 2]
    return f"{vals[0]:.3f} / {mid:.3f} / {vals[-1]:.3f}"


if __name__ == "__main__":
    main()