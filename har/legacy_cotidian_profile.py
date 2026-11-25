#!/usr/bin/env python3
"""
Perfil de acciones HAR legacy (cotidianas) por carpeta de data_result.

Clasificación por clip:
  • ``hand_to_body`` — ``evaluate_legacy`` (con confianza 0–1 propia del legacy)
  • si no: acción cotidiana dominante (walk, stop, …) o ``inconclusive``
  • confianza cotidiana = **fracción del clip visible** con ese gesto (no hay softmax)

Uso:
  .venv/bin/python3 legacy_cotidian_profile.py /path/data_result
  .venv/bin/python3 legacy_cotidian_profile.py /path/data_result -c 0 1 5 --csv /tmp/legacy_prof.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
SUMMARY_TYPES = ("hand_to_body", "inconclusive", "error") + COTIDIAN_IDS
MIN_VISIBLE_S = 2.0
MIN_FRAC = 0.08
INCONCLUSIVE_STOP_S = 0.25


@dataclass
class ClipLegacyProfile:
    legacy_tipo: str
    tipo_confidence: float
    hand_to_body_confidence: float
    visible_s: float
    frac: dict[str, float]
    error: str = ""


def _load_poses(path: Path):
    try:
        poses, _ = load_poses_array(path)
        return poses
    except ValueError:
        return load_poses(path)


def classify_clip_legacy(path: Path) -> ClipLegacyProfile:
    """Perfil legacy; ``tipo_confidence`` es heurístico (fracción temporal o conf. HTB)."""
    try:
        poses = _load_poses(path)
        fps, video = fps_and_video(path)
        aspect = frame_aspect(video)
        leg = evaluate_legacy(poses, fps=fps, aspect=aspect)
        visible = visible_seconds(poses, fps)
        empty_frac = {aid: 0.0 for aid in COTIDIAN_IDS}
        if leg.hand_to_body:
            return ClipLegacyProfile(
                legacy_tipo="hand_to_body",
                tipo_confidence=float(leg.confidence),
                hand_to_body_confidence=float(leg.confidence),
                visible_s=round(visible, 3),
                frac=empty_frac,
            )
        smoothed = smooth(poses, window=3)
        if visible < MIN_VISIBLE_S:
            return ClipLegacyProfile(
                legacy_tipo="inconclusive",
                tipo_confidence=0.0,
                hand_to_body_confidence=0.0,
                visible_s=round(visible, 3),
                frac=empty_frac,
            )
        actions = detect_cotidian_actions(smoothed, fps, aspect)
        seconds = gesture_seconds(actions, fps)
        vis = max(visible, 1e-6)
        fracs = {aid: round(seconds.get(aid, 0.0) / vis, 4) for aid in COTIDIAN_IDS}
        best_id = "inconclusive"
        best_frac = 0.0
        for aid in COTIDIAN_IDS:
            frac = fracs[aid]
            if frac >= MIN_FRAC and frac > best_frac:
                best_id = aid
                best_frac = frac
        if best_id == "inconclusive" and seconds.get("stop", 0.0) >= INCONCLUSIVE_STOP_S:
            best_id = "stop"
            best_frac = fracs.get("stop", 0.0)
        return ClipLegacyProfile(
            legacy_tipo=best_id,
            tipo_confidence=round(best_frac, 4),
            hand_to_body_confidence=0.0,
            visible_s=round(visible, 3),
            frac=fracs,
        )
    except Exception as exc:
        return ClipLegacyProfile(
            legacy_tipo="error",
            tipo_confidence=0.0,
            hand_to_body_confidence=0.0,
            visible_s=0.0,
            frac={aid: 0.0 for aid in COTIDIAN_IDS},
            error=str(exc),
        )


def _worker(task: tuple[str, int]) -> tuple[int, ClipLegacyProfile]:
    path_s, cat = task
    return cat, classify_clip_legacy(Path(path_s))


def _print_summary_table(by_cat: dict[int, Counter[str]]) -> None:
    cats = sorted(by_cat)
    if not cats:
        print("Sin datos.")
        return

    col_headers = ["cat", "total", "no_htb%", "htb_n"] + list(SUMMARY_TYPES)
    widths = {h: max(len(h), 6 if h == "cat" else 12) for h in col_headers}

    rows_out: list[list[str]] = []
    totals: Counter[str] = Counter()

    for cat in cats:
        c = by_cat[cat]
        total = sum(c.values())
        htb_n = c.get("hand_to_body", 0)
        no_htb_pct = 100.0 * (total - htb_n) / max(total, 1)
        row = [str(cat), str(total), f"{no_htb_pct:.1f}", str(htb_n)]
        for key in SUMMARY_TYPES:
            n = c.get(key, 0)
            totals[key] += n
            row.append(str(n))
            widths[key] = max(widths[key], len(str(n)))
        rows_out.append(row)

    grand = sum(totals.values())
    htb_all = totals.get("hand_to_body", 0)
    row_tot = [
        "TOTAL",
        str(grand),
        f"{100.0 * (grand - htb_all) / max(grand, 1):.1f}",
        str(htb_all),
    ]
    for key in SUMMARY_TYPES:
        row_tot.append(str(totals.get(key, 0)))
    rows_out.append(row_tot)

    sep = "+" + "+".join("-" * (widths[h] + 2) for h in col_headers) + "+"
    header = "|" + "|".join(f" {h:^{widths[h]}} " for h in col_headers) + "|"

    print()
    print("Resumen HAR legacy por categoría (conteo de clips por tipo asignado)")
    print(sep)
    print(header)
    print(sep)
    for i, row in enumerate(rows_out):
        bold_sep = i == len(rows_out) - 1
        line = "|" + "|".join(f" {v:>{widths[h]}} " for h, v in zip(col_headers, row)) + "|"
        print(line)
        if bold_sep:
            print(sep)

    print()
    print("Lectura rápida (clips por categoría):")
    for cat in cats:
        c = by_cat[cat]
        total = sum(c.values()) or 1
        htb = c.get("hand_to_body", 0)
        inc = c.get("inconclusive", 0) + c.get("error", 0)
        cot = {k: c.get(k, 0) for k in COTIDIAN_IDS if c.get(k, 0)}
        cot_parts = ", ".join(f"{k} {v} ({100 * v / total:.1f}%)" for k, v in sorted(cot.items(), key=lambda x: -x[1]))
        no_htb = total - htb
        print(
            f"  cat {cat:2d}: {total:5d} clips  |  "
            f"**NO hand_to_body** {no_htb:5d} ({100 * no_htb / total:.1f}%)"
            f"  |  hand_to_body {htb:5d} ({100 * htb / total:.1f}%)"
            f"  |  inconclusive+error {inc:5d} ({100 * inc / total:.1f}%)"
        )
        if cot_parts:
            print(f"         cotidianas: {cot_parts}")


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
        default=None,
        metavar="N",
        help="Categorías (omitir = todas las encontradas en data_result)",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV por clip (siempre opcional; la tabla resumen se imprime igual)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    discovery = discover_clips(
        [p.resolve() for p in args.data_roots],
        verbose=False,
        heartbeat=not args.quiet,
    )
    clips = discovery.clips
    if args.category is not None:
        allowed = set(args.category)
        clips = filter_clips_by_category(clips, allowed)
    if not clips:
        raise SystemExit("No hay clips para analizar.")

    cats_found = sorted({c.category for c in clips})
    if not args.quiet:
        print(f"Clips a evaluar: {len(clips)}  |  categorías: {cats_found}\n", flush=True)

    tasks = [(str(c.path), c.category) for c in clips]
    results: list[tuple[str, int, ClipLegacyProfile]] = []
    errors = 0

    if len(tasks) == 1 or args.workers <= 1:
        for path_s, cat in tasks:
            prof = classify_clip_legacy(Path(path_s))
            results.append((path_s, cat, prof))
            errors += bool(prof.error)
    else:
        import time

        t0 = time.perf_counter()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, task): task for task in tasks}
            for fut in as_completed(futs):
                path_s, cat = futs[fut]
                cat_o, prof = fut.result()
                results.append((path_s, cat_o, prof))
                if prof.error:
                    errors += 1
                done += 1
                if not args.quiet and done % max(1, len(tasks) // 40) == 0:
                    _print_progress(done, len(tasks), t0, prefix="  ", detail="legacy profile")
        if not args.quiet:
            _print_progress(len(tasks), len(tasks), t0, prefix="  ", detail="completado", errors=errors)

    by_cat: dict[int, Counter[str]] = defaultdict(Counter)
    for _, cat, prof in results:
        by_cat[cat][prof.legacy_tipo] += 1

    _print_summary_table(by_cat)

    print()
    print(
        "Confianza HAR legacy: hand_to_body tiene score 0–1 en evaluate_legacy; "
        "walk/stop/carry… **no** tienen probabilidad calibrada — en el CSV, "
        "`tipo_confidence` = fracción del tiempo visible con el gesto ganador "
        "(heurística, no % bayesiano)."
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frac_cols = [f"frac_{a}" for a in COTIDIAN_IDS]
        fieldnames = [
            "category",
            "legacy_tipo",
            "tipo_confidence",
            "hand_to_body_confidence",
            "visible_s",
            *frac_cols,
            "error",
            "path",
        ]
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for path_s, cat, prof in sorted(results, key=lambda r: (r[1], r[0])):
                row = {
                    "category": cat,
                    "legacy_tipo": prof.legacy_tipo,
                    "tipo_confidence": prof.tipo_confidence,
                    "hand_to_body_confidence": prof.hand_to_body_confidence,
                    "visible_s": prof.visible_s,
                    "error": prof.error,
                    "path": path_s,
                }
                for aid in COTIDIAN_IDS:
                    row[f"frac_{aid}"] = prof.frac.get(aid, 0.0)
                w.writerow(row)
        print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()