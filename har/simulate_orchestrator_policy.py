#!/usr/bin/env python3
"""Embudo Orchestrator: legado vs doctrina P0-3 (lee columnas del CSV)."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from fase0_pocket_audit import _orch_branch_doctrine, _orch_branch_legacy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    with args.csv.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]

    print(f"Filas OK: {len(rows)}\n")
    for cat in (0, 6, 14):
        sub = [r for r in rows if int(r["category"]) == cat]
        leg = Counter(_orch_branch_legacy(r) for r in sub)
        doc = Counter(_orch_branch_doctrine(r) for r in sub)
        print(f"=== cat {cat} (n={len(sub)}) ===")
        print(f"  Legado Ap.G:   {dict(leg.most_common())}")
        print(f"  Doctrina P0-3: {dict(doc.most_common())}")

    neg = [r for r in rows if int(r["category"]) not in {6, 14, 8, 10, 11, 12}]
    print(f"\n=== Negativas agregadas (n={len(neg)}) ===")
    print(f"  Legado:   {Counter(_orch_branch_legacy(r) for r in neg)}")
    print(f"  Doctrina: {Counter(_orch_branch_doctrine(r) for r in neg)}")


if __name__ == "__main__":
    main()