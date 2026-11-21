#!/usr/bin/env python3
"""
Compara embudo Orchestrator: política CSV grabada (legacy) vs doctrina P0-3 re-simulada.

En CSV viejo, motion_suppress ya refleja el código anterior. Este script aplica
las reglas P0-3 sobre columnas resultado/raw/clear/filter_fp_safe/ctx_walk_frac.

Uso:
  .venv/bin/python3 simulate_orchestrator_policy.py /tmp/fase0_10pct.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

WALK_SUPPRESS_MOTIVES = frozenset(
    {
        "ACERCAMIENTO_EJE_TORSO",
        "APROXIMACION_OBSERVADA",
        "EVIDENCIA_PARCIAL",
        "ZONA_GRIS",
    }
)
WALK_FRAC = 0.44


def _bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _legacy_branch(r: dict[str, str]) -> str:
    if _bool(r.get("motion_suppress", "")):
        return "DISCARD_SUPPRESS"
    if _bool(r.get("hand_to_body", "")):
        return "CONFIRM"
    if _bool(r.get("hand_to_body_raw", "")):
        return "REVIEW_RAW"
    return "DISCARD_ELSE"


def _doctrine_suppress(r: dict[str, str]) -> tuple[bool, str]:
    if _bool(r.get("hand_to_body", "")):
        return False, ""
    res = (r.get("resultado") or "").strip().upper()
    if res == "INDETERMINADO":
        return False, "KEEP_INDETERMINADO"
    if res == "INCOMPATIBLE" and _bool(r.get("filter_fp_safe", "")):
        return True, "SUPPRESS_INCOMPATIBLE"
    raw = _bool(r.get("hand_to_body_raw", ""))
    if not raw:
        return True, "SUPPRESS_NO_POCKET"
    wf = float(r.get("ctx_walk_frac") or 0)
    motivo = (r.get("motivo") or "").strip()
    if wf >= WALK_FRAC and motivo in WALK_SUPPRESS_MOTIVES:
        return True, "SUPPRESS_WALK_WEAK_GESTURE"
    return False, ""


def _doctrine_branch(r: dict[str, str]) -> str:
    sup, _ = _doctrine_suppress(r)
    if sup:
        return "DISCARD_SUPPRESS"
    if _bool(r.get("hand_to_body", "")):
        return "CONFIRM"
    if _bool(r.get("hand_to_body_raw", "")):
        return "REVIEW_RAW"
    res = (r.get("resultado") or "").strip().upper()
    if res == "INDETERMINADO":
        return "KEEP_INDETERMINADO"
    if res == "INCOMPATIBLE":
        return "KEEP_INCOMPATIBLE_WEAK"
    return "DISCARD_ELSE"


def _funnel(rows: list[dict[str, str]], fn) -> Counter[str]:
    return Counter(fn(r) for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    with args.csv.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]

    cats = {0, 6, 14}
    print(f"Filas OK: {len(rows)}\n")
    for cat in sorted(cats):
        sub = [r for r in rows if int(r["category"]) == cat]
        leg = _funnel(sub, _legacy_branch)
        doc = _funnel(sub, _doctrine_branch)
        print(f"=== cat {cat} (n={len(sub)}) ===")
        print(f"  CSV grabado:     {dict(leg.most_common())}")
        print(f"  Doctrina P0-3:   {dict(doc.most_common())}")

    neg = [r for r in rows if int(r["category"]) not in {6, 14, 8, 10, 11, 12}]
    print(f"\n=== Negativas agregadas (n={len(neg)}) ===")
    print(f"  CSV:      {_funnel(neg, _legacy_branch)}")
    print(f"  Doctrina: {_funnel(neg, _doctrine_branch)}")


if __name__ == "__main__":
    main()