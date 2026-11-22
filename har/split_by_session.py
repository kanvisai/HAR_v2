#!/usr/bin/env python3
"""P0-4: holdout de sesiones desde CSV batch (split por session_id, no por clip)."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_csv", type=Path)
    ap.add_argument("-o", "--output-dir", type=Path, required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stratify-category",
        action="store_true",
        help="Reservar ~frac de sesiones por categoría (recomendado)",
    )
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(args.batch_csv.open(encoding="utf-8")) if not (r.get("error") or "").strip()]
    by_sess: dict[tuple[int, str], int] = defaultdict(int)
    for r in rows:
        sess = (r.get("session_id") or "").strip() or "_unknown_"
        by_sess[(int(r["category"]), sess)] += 1

    rng = random.Random(args.seed)
    holdout: set[tuple[int, str]] = set()
    all_keys = list(by_sess.keys())
    if args.stratify_category:
        by_cat: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for key in all_keys:
            by_cat[key[0]].append(key)
        for cat, keys in sorted(by_cat.items()):
            rng.shuffle(keys)
            nh = max(1, int(round(len(keys) * args.holdout_frac))) if len(keys) > 1 else 0
            holdout.update(keys[:nh])
    else:
        rng.shuffle(all_keys)
        nh = max(1, int(round(len(all_keys) * args.holdout_frac)))
        holdout.update(all_keys[:nh])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_csv": str(args.batch_csv),
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "n_sessions_total": len(all_keys),
        "n_sessions_holdout": len(holdout),
        "n_clips_holdout": sum(by_sess[k] for k in holdout),
        "n_clips_train": sum(v for k, v in by_sess.items() if k not in holdout),
        "holdout_sessions": [{"category": c, "session_id": s, "n_clips": by_sess[(c, s)]} for c, s in sorted(holdout)],
    }
    (args.output_dir / "session_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_lines = [f"{c}\t{s}" for c, s in sorted(all_keys) if (c, s) not in holdout]
    hold_lines = [f"{c}\t{s}" for c, s in sorted(holdout)]
    (args.output_dir / "sessions_train.tsv").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (args.output_dir / "sessions_holdout.tsv").write_text("\n".join(hold_lines) + "\n", encoding="utf-8")
    print(
        f"sesiones: total={len(all_keys)} holdout={len(holdout)} "
        f"clips train={manifest['n_clips_train']} holdout={manifest['n_clips_holdout']}"
    )
    print(f"→ {args.output_dir}/session_split_manifest.json")


if __name__ == "__main__":
    main()