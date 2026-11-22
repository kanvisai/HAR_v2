#!/usr/bin/env python3
"""P0-5: compara métricas clear por categoría según pocket_eval_mode en un CSV."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _b(s: str) -> bool:
    return str(s).strip().lower() == "true"


def _audit_bucket(category: int, clear: bool) -> str:
    if category in (6, 14):
        return "TP" if clear else "FN"
    return "FP" if clear else "TN"


def summarize(rows: list[dict], *, mode: str | None) -> dict[str, Counter]:
    sub = rows
    if mode is not None:
        sub = [r for r in rows if (r.get("pocket_eval_mode") or "temporal") == mode]
    by_cat: dict[int, Counter] = {}
    for r in sub:
        cat = int(r["category"])
        clear = _b(r.get("hand_to_body", ""))
        by_cat.setdefault(cat, Counter())[_audit_bucket(cat, clear)] += 1
    return by_cat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_csv", type=Path)
    ap.add_argument("--categories", type=int, nargs="+", default=[0, 6, 14])
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(args.batch_csv.open(encoding="utf-8")) if not (r.get("error") or "").strip()]
    modes = Counter((r.get("pocket_eval_mode") or "temporal") for r in rows)
    print(f"filas OK: {len(rows)}  pocket_eval_mode: {dict(modes)}\n")

    for mode in sorted(modes.keys()):
        by_cat = summarize(rows, mode=mode)
        print(f"=== modo {mode} ===")
        for cat in args.categories:
            c = by_cat.get(cat, Counter())
            tp, tn, fp, fn = c.get("TP", 0), c.get("TN", 0), c.get("FP", 0), c.get("FN", 0)
            n = tp + tn + fp + fn
            if not n:
                continue
            if cat in (6, 14):
                print(f"  cat {cat:2d}: clear TP={tp} FN={fn}  (n={n})  recall_clear={tp/n:.1%}")
            else:
                print(f"  cat {cat:2d}: TN={tn} FP={fp}  (n={n})  fp_rate={fp/n:.1%}")
        print()


if __name__ == "__main__":
    main()