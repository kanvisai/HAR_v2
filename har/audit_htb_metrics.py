#!/usr/bin/env python3
"""Dual metrics: hand_to_body (clear) vs hand_to_body_raw (gesture in pose)."""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


def _bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _audit(rows: list[dict[str, str]], categories: set[int], col: str) -> dict[int, Counter[str]]:
    out: dict[int, Counter[str]] = {}
    pos = {6, 14}
    for r in rows:
        if r.get("error"):
            continue
        cat = int(r["category"])
        if cat not in categories:
            continue
        exp = cat in pos
        got = _bool(r.get(col, ""))
        if got and exp:
            c = "TP"
        elif got and not exp:
            c = "FP"
        elif not got and not exp:
            c = "TN"
        else:
            c = "FN"
        out.setdefault(cat, Counter())[c] += 1
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <batch.csv>", file=sys.stderr)
        raise SystemExit(2)
    path = Path(sys.argv[1])
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labeled = {0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 14}
    cols = (
        ("CLEAR (hand_to_body)", "hand_to_body"),
        ("RAW (hand_to_body_raw)", "hand_to_body_raw"),
    )
    clear_by_cat: dict[int, Counter[str]] = {}
    for label, col in cols:
        print(f"\n=== {label} ===")
        stats = _audit(rows, labeled, col)
        if col == "hand_to_body":
            clear_by_cat = stats
        for cat in sorted(stats):
            c = stats[cat]
            n = c["TP"] + c["TN"] + c["FP"] + c["FN"]
            print(f"  cat {cat:2d}: TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}  (n={n})")
    print("\n=== Clear vs raw (robbery categories) ===")
    raw = _audit(rows, {6, 14}, "hand_to_body_raw")
    for cat in (6, 14):
        cl = clear_by_cat.get(cat, Counter())
        rw = raw.get(cat, Counter())
        print(
            f"  cat {cat}: clear_TP={cl.get('TP', 0)} raw_TP={rw.get('TP', 0)} "
            f"(raw−clear={rw.get('TP', 0) - cl.get('TP', 0)} extra gesture hits)"
        )


if __name__ == "__main__":
    main()