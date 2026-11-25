#!/usr/bin/env python3
"""
Perfil de acciones HAR legacy (cotidianas) por carpeta de data_result.

Para cada clip: legacy ``hand_to_body``, si no → acción dominante (walk, stop, …)
o ``inconclusive``. Resumen por categorías pedidas, excluyendo hand_to_body e
inconclusive del desglose principal.

Uso:
  .venv/bin/python3 legacy_cotidian_profile.py /path/data_result -c 0 1 5
  .venv/bin/python3 legacy_cotidian_profile.py /path/data_result -c 0 --workers 24 --csv /tmp/legacy_prof.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batch_data_result import (
    _configure_stdio,
    _print_progress,
    discover_clips,
    filter_clips_by_category,
)
from har_actions.detect import detect_cotidian_actions, gesture_seconds, visible_seconds
from har_actions.pose import fps_and_video, frame_aspect, load_poses, smooth
from legacy_hand_to_body import evaluate_legacy
from pocket_filter.io import load_poses_array

COTIDIAN_IDS = ("walk", "stop", "turn_torso", "arm_swing", "carry", "hand_up")
EXCLUDED_FROM_MAIN = frozenset({"hand_to_body", "inconclusive"})
LABEL_ES = {
    "walk": "walk",
    "stop": "stop",
    "turn_torso": "turn_torso",
    "arm_swing": "arm_swing",
    "carry": "carry",
    "hand_up": "hand_up",
    "hand_to_body": "hand_to_body",
    "inconclusive": "inconclusive",
    "error": "error",
}
MIN_VISIBLE_S = 2.0
MIN_FRAC = 0.08
INCONCLUSIVE_STOP_S = 0.25


def _load_poses(path: Path):
    try:
        poses, _ = load_poses_array(path)
        return poses
    except ValueError:
        return load_poses(path)


def classify_clip_legacy(path: Path) -> tuple[str, str]:
    """Devuelve (tipo, error). tipo ∈ COTIDIAN_IDS | hand_to_body | inconclusive | error."""
    try:
        poses = _load_poses(path)
        fps, video = fps_and_video(path)
        aspect = frame_aspect(video)
        if evaluate_legacy(poses, fps=fps, aspect=aspect).hand_to_body:
            return "hand_to_body", ""
        smoothed = smooth(poses, window=3)
        visible = visible_seconds(poses, fps)
        if visible < MIN_VISIBLE_S:
            return "inconclusive", ""
        actions = detect_cotidian_actions(smoothed, fps, aspect)
        seconds = gesture_seconds(actions, fps)
        best_id = "inconclusive"
        best_frac = 0.0
        for aid in COTIDIAN_IDS:
            dur = seconds.get(aid, 0.0)
            if dur <= 0:
                continue
            frac = dur / max(visible, 1e-6)
            if frac >= MIN_FRAC and frac > best_frac:
                best_id = aid
                best_frac = frac
        if best_id == "inconclusive" and seconds.get("stop", 0.0) >= INCONCLUSIVE_STOP_S:
            best_id = "stop"
        return best_id, ""
    except Exception as exc:
        return "error", str(exc)


def _worker(task: tuple[str, int]) -> tuple[int, str, str]:
    path_s, cat = task
    tipo, err = classify_clip_legacy(Path(path_s))
    if err:
        return cat, "error", err
    return cat, tipo, ""


def _aggregate(rows: list[tuple[int, str]]) -> dict[int, Counter[str]]:
    by_cat: dict[int, Counter[str]] = defaultdict(Counter)
    for cat, tipo in rows:
        by_cat[cat][tipo] += 1
    return by_cat


def _print_table(by_cat: dict[int, Counter[str]], *, action_cols: list[str]) -> None:
    cats = sorted(by_cat)
    if not cats:
        print("Sin datos.")
        return

    col_headers = ["cat", "total", "hand_to_body", "inconclusive"] + action_cols
    widths = {h: max(len(h), 8 if h == "cat" else 10) for h in col_headers}

    def cell(key: str, val: str) -> None:
        widths[key] = max(widths[key], len(val))

    rows_out: list[list[str]] = []
    for cat in cats:
        c = by_cat[cat]
        total = sum(c.values())
        htb = c.get("hand_to_body", 0)
        inc = c.get("inconclusive", 0)
        err = c.get("error", 0)
        inc_all = inc + err
        row = [
            str(cat),
            str(total),
            str(htb),
            str(inc_all),
        ]
        for aid in action_cols:
            row.append(str(c.get(aid, 0)))
        for k, v in zip(col_headers, row):
            cell(k, v)
        rows_out.append(row)

    sep = "+" + "+".join("-" * (widths[h] + 2) for h in col_headers) + "+"
    header = "|" + "|".join(f" {h:^{widths[h]}} " for h in col_headers) + "|"

    print()
    print("Resumen HAR legacy — acciones cotidianas (excl. hand_to_body e inconclusive en columnas derechas)")
    print(sep)
    print(header)
    print(sep)
    for row in rows_out:
        print("|" + "|".join(f" {v:>{widths[h]}} " for h, v in zip(col_headers, row)) + "|")
    print(sep)
    print()
    print("Leyenda por fila:")
    for cat in cats:
        c = by_cat[cat]
        total = sum(c.values())
        main = {k: v for k, v in c.items() if k not in EXCLUDED_FROM_MAIN and k != "error"}
        parts = ", ".join(f"{k} {v}" for k, v in sorted(main.items(), key=lambda x: -x[1]))
        htb = c.get("hand_to_body", 0)
        inc = c.get("inconclusive", 0) + c.get("error", 0)
        print(f"  categoría {cat:2d} → {total} clips → excl. hand_to_body {htb}, inconclusive {inc}")
        if parts:
            print(f"               cotidianas: {parts}")
        else:
            print("               cotidianas: (ninguna con tipo claro)")


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description="Perfil acciones HAR legacy por categorías de data_result"
    )
    parser.add_argument(
        "data_roots",
        nargs="+",
        type=Path,
        metavar="DATA_RESULT",
        help="Raíz(es) data_result",
    )
    parser.add_argument(
        "-c",
        "--category",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help="Categorías a analizar (p. ej. -c 0 1 5)",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--csv", type=Path, default=None, help="CSV por clip: path, category, legacy_tipo")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    discovery = discover_clips([p.resolve() for p in args.data_roots], verbose=False, heartbeat=not args.quiet)
    allowed = set(args.category)
    clips = filter_clips_by_category(discovery.clips, allowed)
    if not clips:
        raise SystemExit(f"No hay clips en categorías {sorted(allowed)}")

    tasks = [(str(c.path), c.category) for c in clips]
    results: list[tuple[int, str, str, str]] = []
    errors = 0

    if len(tasks) == 1 or args.workers <= 1:
        for path_s, cat in tasks:
            tipo, err = classify_clip_legacy(Path(path_s))
            results.append((path_s, cat, tipo, err))
            errors += bool(err)
    else:
        import time

        t0 = time.perf_counter()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, (path_s, cat)): (path_s, cat) for path_s, cat in tasks}
            for fut in as_completed(futs):
                path_s, cat = futs[fut]
                cat_o, tipo, err = fut.result()
                results.append((path_s, cat_o, tipo, err))
                if err:
                    errors += 1
                done += 1
                if not args.quiet and done % max(1, len(tasks) // 40) == 0:
                    _print_progress(done, len(tasks), t0, prefix="  ", detail="legacy profile")
        if not args.quiet:
            _print_progress(len(tasks), len(tasks), t0, prefix="  ", detail="completado", errors=errors)

    rows = [(cat, tipo) for _, cat, tipo, _ in results]
    by_cat = _aggregate(rows)

    present_actions: set[str] = set()
    for c in by_cat.values():
        for k, v in c.items():
            if k in COTIDIAN_IDS and v:
                present_actions.add(k)
    action_cols = [a for a in COTIDIAN_IDS if a in present_actions] or list(COTIDIAN_IDS)

    _print_table(by_cat, action_cols=action_cols)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["category", "legacy_tipo", "error", "path"])
            w.writeheader()
            for path_s, cat, tipo, err in sorted(results, key=lambda r: (r[1], r[0])):
                w.writerow(
                    {
                        "category": cat,
                        "legacy_tipo": tipo,
                        "error": err,
                        "path": path_s,
                    }
                )
        print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()