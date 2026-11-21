#!/usr/bin/env python3
"""Añade review_ml_score a CSV batch usando review_ranker_weights.json."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from review_ranker_features import feature_vector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_csv", type=Path)
    ap.add_argument("weights_json", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    w = np.array(json.loads(args.weights_json.read_text())["weights"], dtype=np.float64)
    with args.batch_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("CSV vacío")
    fields = list(rows[0].keys())
    if "review_ml_score" not in fields:
        fields.append("review_ml_score")
    for r in rows:
        x = np.array(feature_vector(r), dtype=np.float64)
        z = float(x @ w)
        r["review_ml_score"] = f"{1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))):.6f}"
    with args.output.open("w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"Escrito: {args.output}")


if __name__ == "__main__":
    main()