#!/usr/bin/env python3
"""Audita candidatos a excepción kin eje+marcha en un CSV batch (solo etiquetas category)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

AXIS = frozenset({"ACERCAMIENTO_EJE_TORSO"})
CONF_MIN = 0.66


def _f(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _b(s: str) -> bool:
    return str(s).strip().lower() == "true"


def _median(vals: list[float]) -> float | None:
    vals = [x for x in vals if x == x]
    if not vals:
        return None
    s = sorted(vals)
    return s[len(s) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--dwell-min", type=float, default=0.35)
    ap.add_argument("--swing-max", type=float, default=4.0)
    ap.add_argument("--hip-max", type=float, default=0.0, help="0 = sin tope de cadera")
    ap.add_argument("--walk-frac", type=float, default=0.44)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    fieldnames = rows[0].keys() if rows else []
    has_dwell = "kin_near_dwell_s" in fieldnames
    has_swing = "kin_swing_peaks" in fieldnames

    axis = [r for r in rows if (r.get("motivo") or "").strip() in AXIS]
    walking = [r for r in axis if _f(r.get("ctx_walk_frac", "0")) >= args.walk_frac]

    with_kin = 0
    for r in walking:
        d, s = _f(r.get("kin_near_dwell_s", "")), _f(r.get("kin_swing_peaks", ""))
        if d == d and s == s:
            with_kin += 1

    def would_pass_full(r: dict) -> bool:
        dwell = _f(r.get("kin_near_dwell_s", ""))
        swing = _f(r.get("kin_swing_peaks", ""))
        conf = _f(r.get("hand_to_body_confidence", ""))
        if dwell != dwell or swing != swing:
            return False
        if dwell < args.dwell_min or swing > args.swing_max or conf < CONF_MIN:
            return False
        if args.hip_max > 0:
            hip = _f(r.get("kin_hip_speed_norm", ""))
            if hip != hip or hip > args.hip_max:
                return False
        return True

    def would_pass_hip_proxy(r: dict) -> bool:
        """Solo conf + hip (cuando dwell/swing no están en CSV)."""
        conf = _f(r.get("hand_to_body_confidence", ""))
        if conf < CONF_MIN:
            return False
        if args.hip_max <= 0:
            return True
        hip = _f(r.get("kin_hip_speed_norm", ""))
        return hip == hip and hip <= args.hip_max

    buckets: dict[str, int] = {}
    for r in walking:
        cat = int(r["category"])
        clear = _b(r.get("hand_to_body", ""))
        if not would_pass_full(r):
            continue
        if cat == 6 and clear:
            key = "cat6_clear"
        elif cat == 6 and not clear:
            key = "cat6_fn"
        elif cat == 0 and clear:
            key = "cat0_fp"
        elif cat == 0 and not clear:
            key = "cat0_tn"
        else:
            key = "other"
        buckets[key] = buckets.get(key, 0) + 1

    print(f"axis+marcha clips: {len(walking)} / {len(axis)} axis total")
    print(
        f"CSV kin_near_dwell_s: col={'sí' if has_dwell else 'NO'}  "
        f"kin_swing_peaks: col={'sí' if has_swing else 'NO'}  "
        f"filas con dwell+swing numéricos: {with_kin}/{len(walking)}"
    )
    if with_kin == 0:
        print(
            "→ No se puede simular la regla completa: re-batch con batch_data_result.py "
            "(CSV_FIELDS incluye kin_near_dwell_s / kin_swing_peaks)."
        )

    print(f"would_pass_rule (dwell+swing+conf[+hip]): {sum(buckets.values())}  breakdown: {buckets}")

    fp_axis_clear = [r for r in walking if int(r["category"]) == 0 and _b(r.get("hand_to_body", ""))]
    print(f"cat0 axis+marcha clear actual (policy V1 en ese CSV): {len(fp_axis_clear)}")

    if fp_axis_clear and args.hip_max > 0:
        hips = [_f(r.get("kin_hip_speed_norm", "")) for r in fp_axis_clear]
        hips_ok = [h for h in hips if h == h and h <= args.hip_max]
        hips_hi = [h for h in hips if h == h and h > args.hip_max]
        print(
            f"  proxy V2 solo hip≤{args.hip_max}: quedarían clear {len(hips_ok)}, "
            f"bloqueados por hip {len(hips_hi)} (sin dwell/swing en CSV)"
        )
        med = _median(hips)
        if med is not None:
            print(f"  mediana kin_hip_speed_norm FP cat0 eje+marcha clear: {med:.3f}")

    fn6 = [r for r in walking if int(r["category"]) == 6 and not _b(r.get("hand_to_body", ""))]
    if fn6 and args.hip_max > 0:
        hips = [_f(r.get("kin_hip_speed_norm", "")) for r in fn6]
        would_proxy = sum(1 for r in fn6 if would_pass_hip_proxy(r))
        med = _median(hips)
        print(f"cat6 axis+marcha FN: {len(fn6)}  (proxy solo hip+conf pasarían: {would_proxy})")
        if med is not None:
            print(f"  mediana kin_hip_speed_norm FN cat6 eje+marcha: {med:.3f}")


if __name__ == "__main__":
    main()