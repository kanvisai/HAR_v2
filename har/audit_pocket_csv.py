#!/usr/bin/env python3
"""Resumen rápido de un CSV generado por batch_data_result.py."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoría de CSV pocket / legacy batch")
    parser.add_argument("csv_path", type=Path, help="CSV de batch_data_result")
    parser.add_argument(
        "-c",
        "--category",
        type=int,
        nargs="+",
        default=None,
        help="Filtrar categorías (p. ej. 6 14)",
    )
    parser.add_argument(
        "--class",
        dest="audit_class",
        metavar="TP|TN|FP|FN",
        default=None,
        help="Filtrar audit_class",
    )
    parser.add_argument(
        "--list",
        type=int,
        default=0,
        metavar="N",
        help="Listar N rutas (p. ej. FN para revisar)",
    )
    parser.add_argument(
        "--motivos",
        action="store_true",
        help="Contar motivo / resultado en las filas filtradas",
    )
    args = parser.parse_args()

    path = args.csv_path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"No existe: {path}")

    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("CSV vacío o sin cabecera")
        rows = list(reader)

    if args.category is not None:
        allowed = {str(c) for c in args.category}
        rows = [r for r in rows if r.get("category", "") in allowed]

    if args.audit_class:
        rows = [r for r in rows if r.get("audit_class", "") == args.audit_class]

    print(f"Archivo: {path}")
    print(f"Filas (tras filtro): {len(rows)}")
    if not rows:
        return

    by_cat: dict[str, Counter] = {}
    for r in rows:
        ac = r.get("audit_class", "") or "?"
        if ac in ("", "SKIP", "ERROR"):
            continue
        cat = r.get("category", "?")
        by_cat.setdefault(cat, Counter())[ac] += 1

    if by_cat:
        print("\nPor categoría (TP/TN/FP/FN):")
        for cat in sorted(by_cat, key=lambda x: int(x) if x.isdigit() else 999):
            c = by_cat[cat]
            labeled = c["TP"] + c["TN"] + c["FP"] + c["FN"]
            ok = c["TP"] + c["TN"]
            print(
                f"  cat {cat:>2}: {ok}/{labeled}  "
                f"TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}"
            )

    if args.motivos or args.audit_class:
        motivos = Counter(r.get("motivo", "") or "(vacío)" for r in rows)
        print("\nMotivo (top 25):")
        for m, n in motivos.most_common(25):
            print(f"  {n:5d}  {m}")

    if args.list > 0:
        print(f"\nRutas (max {args.list}):")
        for r in rows[: args.list]:
            p = r.get("path", "")
            print(f"  cat={r.get('category')}  {r.get('audit_class')}  {r.get('motivo')}  {p}")


if __name__ == "__main__":
    main()