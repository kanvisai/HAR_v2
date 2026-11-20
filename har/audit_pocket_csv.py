#!/usr/bin/env python3
"""Auditoría y empaquetado de CSV de batch_data_result.py."""
from __future__ import annotations

import argparse
import csv
import random
import zipfile
from collections import Counter
from pathlib import Path


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("CSV vacío o sin cabecera")
        return list(reader)


def filter_rows(
    rows: list[dict[str, str]],
    *,
    categories: list[int] | None,
    audit_class: str | None,
) -> list[dict[str, str]]:
    if categories is not None:
        allowed = {str(c) for c in categories}
        rows = [r for r in rows if r.get("category", "") in allowed]
    if audit_class:
        rows = [r for r in rows if r.get("audit_class", "") == audit_class]
    return rows


def print_summary(rows: list[dict[str, str]], *, motivos: bool) -> None:
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

    if motivos and rows:
        counts = Counter(r.get("motivo", "") or "(vacío)" for r in rows)
        print("\nMotivo (top 25):")
        for m, n in counts.most_common(25):
            print(f"  {n:5d}  {m}")


def print_list(rows: list[dict[str, str]], *, paths_only: bool) -> None:
    print(f"\nListado completo: {len(rows)} fila(s)\n")
    for r in rows:
        p = r.get("path", "")
        if paths_only:
            print(p)
            continue
        print(
            f"cat={r.get('category')}  {r.get('audit_class')}  "
            f"htb={r.get('hand_to_body')}  {r.get('motivo')}  {p}"
        )


def _clip_bundle_files(poses_path: Path) -> list[Path]:
    user_dir = poses_path.parent
    if not user_dir.is_dir():
        return [poses_path] if poses_path.is_file() else []
    files = [p for p in user_dir.iterdir() if p.is_file()]
    return files if files else ([poses_path] if poses_path.is_file() else [])


def _arc_prefix(poses_path: Path, data_root: Path, index: int) -> Path:
    user_dir = poses_path.parent
    try:
        root = data_root.resolve()
        if user_dir.is_relative_to(root):
            return user_dir.relative_to(root)
    except (ValueError, AttributeError):
        pass
    try:
        return user_dir.relative_to(poses_path.parents[2])
    except ValueError:
        return Path(f"clip_{index:04d}") / user_dir.name


def build_zip(rows: list[dict[str, str]], output: Path, *, limit: int) -> None:
    if limit <= 0:
        raise SystemExit("--zip requiere N > 0")
    selected = rows[: min(limit, len(rows))]

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = [
        "Paquete generado por audit_pocket_csv.py",
        f"Clips en el zip: {len(selected)}",
        "",
    ]

    added = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, row in enumerate(selected, start=1):
            poses_path = Path(row.get("path", "")).expanduser()
            if not poses_path.is_file():
                manifest.append(f"#{i} SKIP (no existe): {poses_path}")
                continue

            data_root_s = row.get("data_root", "")
            data_root = Path(data_root_s).resolve() if data_root_s else poses_path.parents[3]
            prefix = _arc_prefix(poses_path, data_root, i)

            manifest.append(
                f"#{i} cat={row.get('category')} {row.get('audit_class')} "
                f"htb={row.get('hand_to_body')} motivo={row.get('motivo')}"
            )
            manifest.append(f"    poses: {poses_path}")
            manifest.append(f"    zip:   {prefix}/")

            for fp in _clip_bundle_files(poses_path):
                arc = str(prefix / fp.name)
                zf.write(fp, arcname=arc)
                added += 1

            manifest.append("")

        zf.writestr("MANIFEST.txt", "\n".join(manifest))

    print(f"Zip: {output.resolve()}")
    print(f"Clips empaquetados: {len(selected)}  |  ficheros añadidos: {added}")
    if len(selected) < limit:
        print(f"Aviso: solo había {len(selected)} filas tras el filtro (pediste {limit}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auditoría de CSV pocket / legacy batch + zip de clips"
    )
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
        metavar="N",
        default=0,
        help="Listar las primeras N rutas (usa --list-all para todas)",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="Listar todas las filas del filtro",
    )
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Con --list / --list-all: solo imprime path (una línea por clip)",
    )
    parser.add_argument(
        "--motivos",
        action="store_true",
        help="Contar motivo en las filas filtradas",
    )
    parser.add_argument(
        "--zip",
        type=int,
        metavar="N",
        default=0,
        help="Empaqueta N clips: carpeta user_* (todos los ficheros junto a poses_full.npy)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pocket_audit_clips.zip"),
        help="Ruta del zip (--zip)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Orden aleatorio antes de --list N o --zip N",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla con --shuffle")
    args = parser.parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"No existe: {csv_path}")

    rows = filter_rows(
        load_rows(csv_path),
        categories=args.category,
        audit_class=args.audit_class,
    )

    print(f"Archivo: {csv_path}")
    print(f"Filas (tras filtro): {len(rows)}")
    if not rows:
        return

    print_summary(rows, motivos=args.motivos)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(rows)

    if args.list_all:
        print_list(rows, paths_only=args.paths_only)
    elif args.list > 0:
        print_list(rows[: args.list], paths_only=args.paths_only)

    if args.zip > 0:
        build_zip(rows, args.output.expanduser().resolve(), limit=args.zip)


if __name__ == "__main__":
    main()