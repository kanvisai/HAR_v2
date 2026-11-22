#!/usr/bin/env python3
"""
Tabla completa por categoría (todas las presentes en el CSV) para comparar backends.

Uso:
  .venv/bin/python3 report_category_matrix.py /tmp/compare_v2_gh_FULL.csv
  .venv/bin/python3 report_category_matrix.py \\
      v2=/tmp/compare_v2_gh_FULL.csv \\
      legacy=/tmp/compare_legacy_FULL.csv \\
      mix=/tmp/compare_mix_FULL.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_labels import CATEGORY_NOTES, EXPECTS_HAND_TO_BODY_TRUE
from batch_data_result import audit_row


def _bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]


def _cat_stats(rows: list[dict[str, str]]) -> dict[int, dict[str, object]]:
    by_cat: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_cat[int(r["category"])].append(r)
    out: dict[int, dict[str, object]] = {}
    for cat, subset in sorted(by_cat.items()):
        n = len(subset)
        clear = sum(1 for r in subset if _bool(r.get("hand_to_body", "")))
        raw = sum(1 for r in subset if _bool(r.get("hand_to_body_raw", "")))
        suppress = sum(1 for r in subset if _bool(r.get("motion_suppress", "")))
        ac = Counter()
        for r in subset:
            _, _, klass = audit_row(cat, _bool(r.get("hand_to_body", "")))
            ac[klass] += 1
        out[cat] = {
            "n": n,
            "clear": clear,
            "raw": raw,
            "suppress": suppress,
            "TP": ac["TP"],
            "TN": ac["TN"],
            "FP": ac["FP"],
            "FN": ac["FN"],
            "SKIP": ac["SKIP"],
        }
    return out


def _note(cat: int) -> str:
    return CATEGORY_NOTES.get(cat, "")


def _print_one(label: str, path: Path, stats: dict[int, dict[str, object]]) -> None:
    backend = ""
    sample = _load(path)
    if sample:
        backend = (sample[0].get("backend") or "").strip()
    print(f"\n{'=' * 88}")
    print(f"  {label}  |  archivo: {path}")
    if backend:
        print(f"  backend CSV: {backend}")
    print(f"{'=' * 88}")
    print(
        " cat |    n | clear% |  raw% | supp |  TP   TN   FP   FN | SKIP | nota"
    )
    print("-----+------+--------+-------+------+------------------+------+-----")
    for cat in sorted(stats):
        s = stats[cat]
        n = int(s["n"])
        clear = int(s["clear"])
        raw = int(s["raw"])
        supp = int(s["suppress"])
        tp, tn, fp, fn, skip = (int(s[k]) for k in ("TP", "TN", "FP", "FN", "SKIP"))
        audit_cols = f"{tp:4d} {tn:4d} {fp:4d} {fn:4d}"
        if cat in EXPECTS_HAND_TO_BODY_TRUE:
            mark = "robo"
        elif skip == n:
            mark = "sin_TP/TN"
        else:
            mark = "neg"
        note = (_note(cat) or mark)[:28]
        print(
            f" {cat:3d} | {n:5d} | {100 * clear / n:5.1f}% | "
            f"{100 * raw / n:5.1f}% | {supp:4d} | {audit_cols} | {skip:4d} | {note}"
        )


def _print_side_by_side(labeled: dict[str, Path], all_stats: dict[str, dict[int, dict]]) -> None:
    cats = sorted({c for st in all_stats.values() for c in st})
    print(f"\n{'=' * 88}")
    print("  COMPARATIVA clear% por categoría (misma fila = mismos clips)")
    print(f"{'=' * 88}")
    header = " cat |    n |" + "".join(f" {lab:>8}" for lab in labeled)
    print(header)
    print("-----+------+" + "-" * (9 * len(labeled)))
    for cat in cats:
        ns = [all_stats[lab].get(cat, {}).get("n", 0) for lab in labeled]
        n0 = max(ns) if ns else 0
        if not n0:
            continue
        parts = []
        for lab in labeled:
            st = all_stats[lab].get(cat)
            if not st:
                parts.append(f"{'n/a':>9}")
            else:
                n = int(st["n"])
                clear = int(st["clear"])
                parts.append(f"{100 * clear / n:8.1f}%")
        print(f" {cat:3d} | {n0:5d} |" + "".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Matriz por categoría para CSV de batch")
    parser.add_argument(
        "csvs",
        nargs="*",
        metavar="CSV",
        help="Ruta o etiqueta=ruta (v2=/tmp/x.csv)",
    )
    args = parser.parse_args()
    if not args.csvs:
        parser.error("Indica al menos un CSV")

    labeled: dict[str, Path] = {}
    for item in args.csvs:
        if "=" in item:
            name, p = item.split("=", 1)
            labeled[name.strip()] = Path(p)
        else:
            labeled[Path(item).stem] = Path(item)

    all_stats: dict[str, dict[int, dict]] = {}
    for name, path in labeled.items():
        if not path.is_file():
            raise SystemExit(f"No existe: {path}")
        rows = _load(path)
        stats = _cat_stats(rows)
        all_stats[name] = stats
        _print_one(name, path, stats)

    if len(labeled) > 1:
        _print_side_by_side(labeled, all_stats)


if __name__ == "__main__":
    main()