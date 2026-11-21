#!/usr/bin/env python3
"""
Exporta cohortes del CSV batch para fusión / revisión (Fase 2, P1-2).

Uso:
  .venv/bin/python3 export_review_cohort.py /tmp/fase1_p03_10pct.csv -o /tmp/cohorts/
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from fase0_pocket_audit import _orch_branch_doctrine
from review_triage import review_priority_score

OUT_FIELDS = [
    "cohort",
    "orch_action",
    "review_priority",
    "category",
    "path",
    "resultado",
    "motivo",
    "hand_to_body",
    "hand_to_body_raw",
    "hand_to_body_confidence",
    "htb_tier",
    "context_veto",
    "ctx_walk_frac",
    "ctx_stop_frac",
    "motion_suppress",
    "audit_class",
]


def _bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _cohort_name(r: dict[str, str]) -> str | None:
    cat = int(r["category"])
    clear = _bool(r.get("hand_to_body", ""))
    raw = _bool(r.get("hand_to_body_raw", ""))
    res = (r.get("resultado") or "").strip().upper()
    if cat == 6 and not clear and raw:
        return "fn_cat6_raw_review"
    if cat == 6 and not clear and res == "INDETERMINADO":
        return "fn_cat6_indeterminate_keep"
    if cat == 14 and not clear and (raw or res == "INDETERMINADO"):
        return "fn_cat14_policy_gap"
    if cat == 0 and clear:
        return "fp_cat0_confirm"
    if cat == 0 and raw and not clear:
        return "fp_cat0_raw_review"
    orch = _orch_branch_doctrine(r)
    if orch == "REVIEW_RAW" and cat in {0, 5}:
        return "pasillo_review_raw"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("/tmp/cohorts"),
        help="Directorio de salida",
    )
    args = ap.parse_args()
    with args.csv.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict[str, str]]] = {}
    counts: Counter[str] = Counter()

    for r in rows:
        name = _cohort_name(r)
        if not name:
            continue
        out = {
            k: r.get(k, "")
            for k in OUT_FIELDS
            if k not in ("cohort", "orch_action", "review_priority")
        }
        out["cohort"] = name
        out["orch_action"] = _orch_branch_doctrine(r)
        out["review_priority"] = str(
            review_priority_score(
                r.get("motivo", ""),
                float(r.get("hand_to_body_confidence") or 0),
                float(r.get("ctx_walk_frac") or 0),
                stop_frac=float(r.get("ctx_stop_frac") or 0),
            )
        )
        buckets.setdefault(name, []).append(out)
        counts[name] += 1

    merged: list[dict[str, str]] = []
    for name, items in sorted(buckets.items()):
        path = args.output_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
            w.writeheader()
            w.writerows(items)
        merged.extend(items)
        print(f"  {name}: {len(items)} → {path}")

    all_path = args.output_dir / "all_cohorts.csv"
    with all_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(merged)
    print(f"\nTotal filas cohortes: {len(merged)}")
    print(f"Combinado: {all_path}")
    print("\nConteos:", dict(counts.most_common()))

    pri = buckets.get("fn_cat6_raw_review", [])
    if pri:
        mot = Counter(r["motivo"] for r in pri)
        print("\n=== fn_cat6_raw_review (P1-2 prioridad) ===")
        print(f"  n={len(pri)} motivos: {mot.most_common()}")
        walk_hi = sum(1 for r in pri if float(r.get("ctx_walk_frac") or 0) >= 0.44)
        print(f"  ctx_walk_frac >= 0.44: {walk_hi}/{len(pri)}")


if __name__ == "__main__":
    main()