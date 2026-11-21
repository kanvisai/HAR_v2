#!/usr/bin/env python3
"""Compara features cinemáticas FN robo vs FP pasillo (solo auditoría con category)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

KIN = (
    "kin_wrist_radial_vel_min",
    "kin_approach_monotone_frac",
    "kin_hip_speed_norm",
    "kin_elbow_angle_deg",
    "kin_wrist_axis_dist_min",
)


def _f(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _bool(s: str) -> bool:
    return str(s).strip().lower() == "true"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_csv", type=Path)
    args = ap.parse_args()
    with args.batch_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pos = [
        r
        for r in rows
        if int(r["category"]) == 6
        and not _bool(r.get("hand_to_body", ""))
        and _bool(r.get("hand_to_body_raw", ""))
    ]
    neg = [
        r
        for r in rows
        if int(r["category"]) == 0
        and _bool(r.get("hand_to_body_raw", ""))
        and not _bool(r.get("hand_to_body", ""))
    ]
    print(f"fn_cat6_raw_review: {len(pos)}  fp_cat0_raw_review: {len(neg)}")
    for col in KIN:
        pv = [_f(r[col]) for r in pos if col in r]
        nv = [_f(r[col]) for r in neg if col in r]
        pv = [x for x in pv if x == x]
        nv = [x for x in nv if x == x]
        if not pv or not nv:
            print(f"  {col}: datos insuficientes")
            continue
        print(
            f"  {col}: pos median {sorted(pv)[len(pv)//2]:.4f}  "
            f"neg median {sorted(nv)[len(nv)//2]:.4f}"
        )


if __name__ == "__main__":
    main()