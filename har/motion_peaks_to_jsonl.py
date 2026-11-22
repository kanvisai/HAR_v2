#!/usr/bin/env python3
"""P0-4/P0-5: convierte export C2 o columnas de batch a catálogo JSONL Motion."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _rel_poses_path(path_s: str, data_root: str | None) -> str:
    p = Path(path_s)
    if data_root:
        try:
            return p.relative_to(Path(data_root)).as_posix()
        except ValueError:
            pass
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "data_result" or (part.isdigit() and i + 1 < len(parts)):
            if part == "data_result":
                return "/".join(parts[i + 1 :])
            return "/".join(parts[i:])
    return p.name


def _write_jsonl(records: list[dict], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def from_batch_csv(
    batch_csv: Path,
    *,
    data_root: str | None,
    min_duration: float,
    use_motion_cols: bool,
) -> list[dict]:
    rows = list(csv.DictReader(batch_csv.open(encoding="utf-8")))
    out: list[dict] = []
    for r in rows:
        if (r.get("error") or "").strip():
            continue
        path = (r.get("path") or "").strip()
        if not path:
            continue
        if use_motion_cols:
            t0_s = (r.get("interval_t0") or "").strip()
            t1_s = (r.get("interval_t1") or "").strip()
            score_s = (r.get("motion_peak_score") or "").strip()
        else:
            t0_s = (r.get("eval_interval_t0") or "").strip()
            t1_s = (r.get("eval_interval_t1") or "").strip()
            score_s = (r.get("hand_to_body_confidence") or "").strip()
        if not t0_s or not t1_s:
            continue
        try:
            t0, t1 = float(t0_s), float(t1_s)
        except ValueError:
            continue
        if t1 - t0 < min_duration:
            continue
        rec: dict = {
            "path": _rel_poses_path(path, data_root),
            "t0": round(t0, 4),
            "t1": round(t1, 4),
        }
        if score_s:
            try:
                rec["score"] = round(float(score_s), 4)
            except ValueError:
                pass
        out.append(rec)
    return out


def from_motion_csv(
    motion_csv: Path,
    *,
    path_col: str,
    t0_col: str,
    t1_col: str,
    score_col: str | None,
    min_duration: float,
) -> list[dict]:
    rows = list(csv.DictReader(motion_csv.open(encoding="utf-8")))
    out: list[dict] = []
    for r in rows:
        path = (r.get(path_col) or "").strip()
        if not path:
            continue
        try:
            t0, t1 = float(r[t0_col]), float(r[t1_col])
        except (KeyError, ValueError):
            continue
        if t1 - t0 < min_duration:
            continue
        rec: dict = {"path": path.replace("\\", "/"), "t0": round(t0, 4), "t1": round(t1, 4)}
        if score_col and score_col in r and (r[score_col] or "").strip():
            try:
                rec["score"] = round(float(r[score_col]), 4)
            except ValueError:
                pass
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_batch = sub.add_parser(
        "from-batch",
        help="Usa eval_interval_* del batch temporal (oracle P0-5 sin C2)",
    )
    p_batch.add_argument("batch_csv", type=Path)
    p_batch.add_argument("-o", "--output", type=Path, required=True)
    p_batch.add_argument("--data-root", type=Path, default=None)
    p_batch.add_argument("--min-duration", type=float, default=0.35)
    p_batch.add_argument(
        "--use-motion-cols",
        action="store_true",
        help="Usar interval_t0/t1 del sidecar ya aplicado (re-export)",
    )

    p_c2 = sub.add_parser("from-csv", help="Export genérico C2 (columnas configurables)")
    p_c2.add_argument("motion_csv", type=Path)
    p_c2.add_argument("-o", "--output", type=Path, required=True)
    p_c2.add_argument("--path-col", default="path")
    p_c2.add_argument("--t0-col", default="t0")
    p_c2.add_argument("--t1-col", default="t1")
    p_c2.add_argument("--score-col", default="score")
    p_c2.add_argument("--min-duration", type=float, default=0.35)

    args = ap.parse_args()
    if args.cmd == "from-batch":
        root = str(args.data_root.resolve()) if args.data_root else None
        records = from_batch_csv(
            args.batch_csv,
            data_root=root,
            min_duration=args.min_duration,
            use_motion_cols=args.use_motion_cols,
        )
    else:
        records = from_motion_csv(
            args.motion_csv,
            path_col=args.path_col,
            t0_col=args.t0_col,
            t1_col=args.t1_col,
            score_col=args.score_col or None,
            min_duration=args.min_duration,
        )

    n = _write_jsonl(records, args.output)
    print(f"JSONL escrito: {args.output}  ({n} intervalos)", file=sys.stderr)
    if n == 0:
        print(
            "Aviso: 0 filas — ¿batch con eval_interval_t0/t1? (re-batch tras git pull)",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()