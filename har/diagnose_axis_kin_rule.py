#!/usr/bin/env python3
"""Audita candidatos a excepción kin eje+marcha en un CSV batch (solo etiquetas category)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

AXIS = frozenset({"ACERCAMIENTO_EJE_TORSO"})


def _f(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _b(s: str) -> bool:
    return str(s).strip().lower() == "true"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--dwell-min", type=float, default=0.35)
    ap.add_argument("--swing-max", type=float, default=4.0)
    ap.add_argument("--hip-max", type=float, default=0.0, help="0 = sin tope de cadera")
    ap.add_argument("--walk-frac", type=float, default=0.44)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    axis = [r for r in rows if (r.get("motivo") or "").strip() in AXIS]
    walking = [
        r
        for r in axis
        if _f(r.get("ctx_walk_frac", "0")) >= args.walk_frac
    ]

    def would_pass_v1(r: dict) -> bool:
        dwell = _f(r.get("kin_near_dwell_s", ""))
        swing = _f(r.get("kin_swing_peaks", ""))
        conf = _f(r.get("hand_to_body_confidence", ""))
        if dwell != dwell or swing != swing:
            return False
        ok = dwell >= args.dwell_min and swing <= args.swing_max and conf >= 0.66
        if args.hip_max > 0:
            hip = _f(r.get("kin_hip_speed_norm", ""))
            if hip != hip or hip > args.hip_max:
                return False
        return ok

    buckets = {"cat6_clear": 0, "cat6_fn": 0, "cat0_fp": 0, "cat0_tn": 0, "other": 0}
    for r in walking:
        cat = int(r["category"])
        clear = _b(r.get("hand_to_body", ""))
        would = would_pass_v1(r)
        if not would:
            continue
        if cat == 6 and clear:
            buckets["cat6_clear"] += 1
        elif cat == 6 and not clear:
            buckets["cat6_fn"] += 1
        elif cat == 0 and clear:
            buckets["cat0_fp"] += 1
        elif cat == 0 and not clear:
            buckets["cat0_tn"] += 1
        else:
            buckets["other"] += 1

    print(f"axis+marcha clips: {len(walking)} / {len(axis)} axis total")
    print(f"would_pass_rule: {sum(buckets.values())}  breakdown: {buckets}")
    fp_axis_clear = [r for r in walking if int(r["category"]) == 0 and _b(r.get("hand_to_body", ""))]
    print(f"cat0 axis+marcha clear actual: {len(fp_axis_clear)}")


if __name__ == "__main__":
    main()