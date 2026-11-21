#!/usr/bin/env python3
"""
Ranker logístico REVIEW: raw∧¬clear (y tier review) → P(etiqueta robo 6/14).

Solo para priorizar cola; en producción no hay category (entrenamiento offline).

Uso:
  .venv/bin/python3 train_review_ranker.py /tmp/fase2_10pct.csv \
    -o har/review_ranker_weights.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from review_ranker_features import FEATURE_NAMES, feature_vector, theft_label

WALK_ACTIVE = 0.44


def _bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _in_review_pool(row: dict[str, str]) -> bool:
    """Candidatos a cola REVIEW (misma lógica que cohorte principal)."""
    if _bool(row.get("hand_to_body", "")):
        return False
    if _bool(row.get("hand_to_body_raw", "")):
        return True
    if (row.get("resultado") or "").strip().upper() == "INDETERMINADO":
        return int(row.get("category", -1)) in (6, 14)
    return False


def _session_split(rows: list[dict[str, str]], *, test_frac: float, seed: int) -> tuple[list, list]:
    by_sess: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        sid = (r.get("session_id") or "").strip() or r.get("path", "")
        by_sess.setdefault(sid, []).append(r)
    sessions = list(by_sess.keys())
    random.Random(seed).shuffle(sessions)
    n_test = max(1, int(len(sessions) * test_frac))
    test_sess = set(sessions[:n_test])
    train, test = [], []
    for sid, items in by_sess.items():
        (test if sid in test_sess else train).extend(items)
    return train, test


def _fit_logreg(X: np.ndarray, y: np.ndarray, *, lr: float = 0.15, epochs: int = 800) -> np.ndarray:
    w = np.zeros(X.shape[1], dtype=np.float64)
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, int(len(y) - y.sum()))
    sample_w = np.where(y > 0.5, n_neg / n_pos, 1.0)
    for _ in range(epochs):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - y) * sample_w
        grad = X.T @ err / len(y)
        w -= lr * grad
    return w


def _metrics(scored: list[tuple[float, int]], ks: tuple[int, ...]) -> None:
    scored.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, y in scored if y)
    print(f"  pool n={len(scored)}  positivos={n_pos}")
    for k in ks:
        if k > len(scored):
            continue
        top = scored[:k]
        p = sum(1 for _, y in top if y)
        rec = p / n_pos if n_pos else 0
        prec = p / k if k else 0
        print(f"  top_{k:4d}: pos={p:3d}  recall={rec:6.1%}  prec={prec:6.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_csv", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("review_ranker_weights.json"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with args.batch_csv.open(encoding="utf-8") as f:
        all_rows = [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]

    pool = [r for r in all_rows if _in_review_pool(r)]
    print(f"Pool REVIEW entrenamiento: {len(pool)} filas (de {len(all_rows)})")

    train, test = _session_split(pool, test_frac=args.test_frac, seed=args.seed)
    print(f"Split sesión: train={len(train)} test={len(test)} (frac={args.test_frac})")

    def _xy(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([feature_vector(r) for r in rows], dtype=np.float64)
        y = np.array([theft_label(int(r["category"])) for r in rows], dtype=np.float64)
        return X, y

    X_train, y_train = _xy(train)
    w = _fit_logreg(X_train, y_train)

    def score_rows(rows: list[dict[str, str]]) -> list[tuple[float, int]]:
        out = []
        for r in rows:
            x = np.array(feature_vector(r), dtype=np.float64)
            z = float(x @ w)
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            out.append((p, theft_label(int(r["category"]))))
        return out

    print("\n=== Train (in-sample) ===")
    _metrics(score_rows(train), (50, 100, 200))
    print("\n=== Test (sesiones hold-out) ===")
    _metrics(score_rows(test), (50, 100, 200))

    payload = {
        "version": 1,
        "feature_names": FEATURE_NAMES,
        "weights": w.tolist(),
        "trained_from": str(args.batch_csv),
        "n_pool": len(pool),
        "note": "Priorizar REVIEW; no usar category en runtime",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nPesos guardados: {args.output}")


if __name__ == "__main__":
    main()