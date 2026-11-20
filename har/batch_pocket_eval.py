#!/usr/bin/env python3
"""Evalúa una lista de poses_full.npy y escribe CSV resumen."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_actions.pose import fps_and_video, frame_aspect
from pocket_filter.config import load_pocket_config
from pocket_filter.evaluate import evaluate_pocket
from pocket_filter.io import load_poses_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="Rutas a poses_full.npy")
    parser.add_argument("-o", "--output", type=Path, default=Path("pocket_batch_results.csv"))
    args = parser.parse_args()

    cfg = load_pocket_config()
    rows = []
    for path in args.paths:
        if not path.is_file():
            rows.append({"path": str(path), "error": "missing"})
            continue
        try:
            poses, conf = load_poses_array(path)
        except ValueError:
            continue
        fps, video = fps_and_video(path)
        aspect = frame_aspect(video)
        vm = path.with_name("valid_mask.npy")
        row_mask = __import__("numpy").load(vm) if vm.is_file() else None
        r = evaluate_pocket(
            poses,
            fps=fps,
            aspect=aspect,
            config=cfg,
            row_mask=row_mask,
            keypoint_conf=conf,
        )
        rows.append(
            {
                "path": str(path),
                "resultado": r.resultado,
                "hand_to_body": int(r.hand_to_body),
                "filter_fp_safe": int(r.filter_fp_safe),
                "motivo": r.motivo,
                "valid_ratio_min": min(p.valid_ratio for p in r.pairs),
            }
        )

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {args.output}  ({len(rows)} filas)")


if __name__ == "__main__":
    main()