#!/usr/bin/env python3
"""
Barrido de data_result por categorías y evaluación pocket v2 (hand_to_body).

Estructura esperada:
  data_result/<categoría>/<sesión>/user_xx/poses_full.npy

Ejemplos:
  .venv/bin/python3 batch_data_result.py /path/data_result --preflight
  .venv/bin/python3 batch_data_result.py /path/data_result -o resultados.csv --workers 20
  .venv/bin/python3 batch_data_result.py /path/data_result --debug 10
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_actions.pose import fps_and_video, frame_aspect, load_poses
from pocket_filter.config import load_pocket_config
from pocket_filter.evaluate import evaluate_pocket
from pocket_filter.io import load_poses_array

# Categorías donde esperamos hand_to_body (positivos hurto / campaña 6+14)
EXPECTS_HAND_TO_BODY_TRUE: frozenset[int] = frozenset({6, 14})

# Cotidianas / no robo: esperamos hand_to_body = false
EXPECTS_HAND_TO_BODY_FALSE: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 7, 9, 13})

# Sin etiqueta de expectativa fiable en documentación
EXPECTS_NO_LABEL: frozenset[int] = frozenset({8, 10, 11, 12})

CSV_FIELDS = [
    "path",
    "category",
    "elapsed_s",
    "hand_to_body",
    "resultado",
    "filter_fp_safe",
    "motivo",
    "n_frames",
    "expected_hand_to_body",
    "label_match",
    "audit_class",
    "error",
]


@dataclass(frozen=True)
class ClipRef:
    path: Path
    category: int


def discover_clips(data_root: Path) -> list[ClipRef]:
    root = data_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"No es un directorio: {root}")

    found: list[ClipRef] = []
    for cat_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not cat_dir.is_dir():
            continue
        if not cat_dir.name.isdigit():
            continue
        category = int(cat_dir.name)
        for poses_path in cat_dir.rglob("poses_full.npy"):
            parent = poses_path.parent
            if parent.name.startswith("user_"):
                found.append(ClipRef(poses_path.resolve(), category))

    found.sort(key=lambda c: (c.category, str(c.path)))
    return found


def expected_hand_to_body(category: int) -> str:
    if category in EXPECTS_HAND_TO_BODY_TRUE:
        return "true"
    if category in EXPECTS_HAND_TO_BODY_FALSE:
        return "false"
    return ""


def audit_row(category: int, hand_to_body: bool) -> tuple[str, str, str]:
    """Devuelve (expected_label, label_match, audit_class)."""
    if category in EXPECTS_HAND_TO_BODY_TRUE:
        expected = True
    elif category in EXPECTS_HAND_TO_BODY_FALSE:
        expected = False
    else:
        return ("", "", "SKIP")

    if hand_to_body and expected:
        return ("true", "true", "TP")
    if hand_to_body and not expected:
        return ("false", "false", "FP")
    if not hand_to_body and not expected:
        return ("false", "true", "TN")
    return ("true", "false", "FN")


def _load_aux(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    row_mask = None
    vm = path.with_name("valid_mask.npy")
    if vm.is_file():
        row_mask = np.load(vm)
    timestamps = None
    ts = path.with_name("pose_frame_timestamps.npy")
    if ts.is_file():
        timestamps = np.load(ts)
    return row_mask, timestamps


def evaluate_one_clip(
    poses_path: Path,
    category: int,
    *,
    config_path: str | None,
    default_height: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        "path": str(poses_path),
        "category": category,
        "error": "",
    }
    try:
        try:
            poses, conf = load_poses_array(poses_path)
        except ValueError:
            poses = load_poses(poses_path)
            conf = None

        fps, video = fps_and_video(poses_path)
        aspect = frame_aspect(video)
        frame_h = default_height
        row_mask, timestamps = _load_aux(poses_path)
        cfg = load_pocket_config(Path(config_path)) if config_path else load_pocket_config()

        result = evaluate_pocket(
            poses,
            fps=fps,
            aspect=aspect,
            frame_height=frame_h,
            timestamps=timestamps,
            row_mask=row_mask,
            keypoint_conf=conf,
            config=cfg,
        )
        ht = bool(result.hand_to_body)
        exp_s, match_s, audit = audit_row(category, ht)
        row.update(
            {
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "hand_to_body": ht,
                "resultado": result.resultado,
                "filter_fp_safe": bool(result.filter_fp_safe),
                "motivo": result.motivo,
                "n_frames": int(result.n_frames),
                "expected_hand_to_body": exp_s,
                "label_match": match_s,
                "audit_class": audit,
            }
        )
    except Exception as exc:
        row.update(
            {
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "hand_to_body": "",
                "resultado": "",
                "filter_fp_safe": "",
                "motivo": "",
                "n_frames": "",
                "expected_hand_to_body": expected_hand_to_body(category),
                "label_match": "",
                "audit_class": "ERROR",
                "error": str(exc),
            }
        )
    return row


def _worker(task: tuple[str, int, str | None, float]) -> dict[str, Any]:
    path_s, cat, cfg, height = task
    return evaluate_one_clip(Path(path_s), cat, config_path=cfg, default_height=height)


def _format_bool_cell(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "" if value in ("", None) else str(value)


def print_debug_table(rows: list[dict[str, Any]]) -> None:
    headers = CSV_FIELDS
    widths = {h: len(h) for h in headers}
    str_rows: list[dict[str, str]] = []
    for row in rows:
        sr = {h: _format_bool_cell(row.get(h, "")) for h in headers}
        str_rows.append(sr)
        for h in headers:
            widths[h] = min(80, max(widths[h], len(sr[h])))

    def line(parts: list[str]) -> None:
        print(" | ".join(p.ljust(widths[h]) for h, p in zip(headers, parts)))

    line(headers)
    line(["-" * widths[h] for h in headers])
    for sr in str_rows:
        line([sr[h][: widths[h]] for h in headers])


def preflight_report(
    clips: list[ClipRef],
    data_root: Path,
    *,
    workers: int,
    sample_size: int,
    config_path: str | None,
    default_height: float,
) -> None:
    by_cat = Counter(c.category for c in clips)
    print(f"Raíz: {data_root.resolve()}")
    print(f"Total clips (poses_full.npy bajo user_*): {len(clips)}")
    print(f"Workers planificados: {workers}")
    print("\nClips por categoría:")
    for cat in sorted(by_cat):
        note = ""
        if cat in EXPECTS_HAND_TO_BODY_TRUE:
            note = "  [esperado hand_to_body=true]"
        elif cat in EXPECTS_HAND_TO_BODY_FALSE:
            note = "  [esperado hand_to_body=false]"
        elif cat in EXPECTS_NO_LABEL:
            note = "  [sin etiqueta fija]"
        print(f"  {cat:2d}: {by_cat[cat]:5d}{note}")

    if not clips:
        print("\nSin archivos que analizar.")
        return

    n_sample = min(sample_size, len(clips))
    sample = clips[:n_sample] if n_sample == len(clips) else random.sample(clips, n_sample)
    print(f"\nMidiendo {n_sample} clip(s) de muestra (secuencial)...")
    times: list[float] = []
    for clip in sample:
        row = evaluate_one_clip(
            clip.path,
            clip.category,
            config_path=config_path,
            default_height=default_height,
        )
        if row.get("error"):
            print(f"  aviso muestra: {clip.path.name} -> {row['error']}")
            continue
        times.append(float(row["elapsed_s"]))

    if not times:
        print("No se pudo estimar tiempo (errores en muestra).")
        return

    avg = sum(times) / len(times)
    p95 = sorted(times)[int(0.95 * (len(times) - 1))]
    total = len(clips)
    est_serial = avg * total
    est_parallel = avg * (total / max(workers, 1)) * 1.08

    print(f"  media: {avg:.3f} s/clip   p95 muestra: {p95:.3f} s")
    print(f"\nTiempo estimado (solo evaluación pocket, sin I/O extra):")
    print(f"  secuencial:  {est_serial/60:.1f} min ({est_serial:.0f} s)")
    print(f"  paralelo:    {est_parallel/60:.1f} min ({est_parallel:.0f} s)  ({workers} workers)")


def run_batch(
    clips: list[ClipRef],
    *,
    workers: int,
    config_path: str | None,
    default_height: float,
    on_progress: bool,
) -> list[dict[str, Any]]:
    tasks = [(str(c.path), c.category, config_path, default_height) for c in clips]
    rows: list[dict[str, Any]] = []
    done = 0
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if on_progress and (done % max(1, total // 20) == 0 or done == total):
                print(f"  Progreso: {done}/{total}")

    rows.sort(key=lambda r: (int(r["category"]), r["path"]))
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["hand_to_body"] = _format_bool_cell(out.get("hand_to_body"))
            out["filter_fp_safe"] = _format_bool_cell(out.get("filter_fp_safe"))
            writer.writerow(out)


def summarize_audit(rows: list[dict[str, Any]]) -> None:
    by_cat: dict[int, Counter] = {}
    for row in rows:
        if row.get("audit_class") in ("", "SKIP", "ERROR"):
            continue
        cat = int(row["category"])
        by_cat.setdefault(cat, Counter())[row["audit_class"]] += 1

    if not by_cat:
        return
    print("\nResumen auditoría (TP/TN/FP/FN por categoría con etiqueta):")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        labeled = c["TP"] + c["TN"] + c["FP"] + c["FN"]
        ok = c["TP"] + c["TN"]
        print(
            f"  cat {cat:2d}: {ok}/{labeled} aciertos  "
            f"TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Barrido data_result → CSV pocket v2 / hand_to_body"
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Carpeta data_result (contiene 0/, 1/, 2/, ...)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pocket_data_result.csv"),
        help="CSV de salida (modo batch)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Inventario + tiempo estimado (usa --workers para la estimación)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        metavar="N",
        default=0,
        help="Evalúa N clips y muestra tabla en pantalla (sin CSV)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Procesos en paralelo para el batch (default: 20)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=12,
        help="Clips de muestra para timing en --preflight",
    )
    parser.add_argument("--config", type=Path, default=None, help="pocket_filter_config.json")
    parser.add_argument("--height", type=float, default=1080.0)
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestras aleatorias")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Menos mensajes de progreso",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cfg_path = str(args.config.resolve()) if args.config else None

    clips = discover_clips(args.data_root)
    if not clips:
        raise SystemExit(f"No se encontraron poses_full.npy bajo user_* en {args.data_root}")

    if args.preflight and args.debug <= 0:
        preflight_report(
            clips,
            args.data_root,
            workers=args.workers,
            sample_size=args.sample,
            config_path=cfg_path,
            default_height=args.height,
        )
        return

    if args.debug > 0:
        n = min(args.debug, len(clips))
        subset = clips[:n] if n == len(clips) else random.sample(clips, n)
        print(f"Modo debug: {n} clip(s)\n")
        rows = run_batch(
            subset,
            workers=min(args.workers, n),
            config_path=cfg_path,
            default_height=args.height,
            on_progress=not args.quiet,
        )
        print_debug_table(rows)
        summarize_audit(rows)
        return

    print(f"Evaluando {len(clips)} clips con {args.workers} workers...")
    t0 = time.perf_counter()
    rows = run_batch(
        clips,
        workers=args.workers,
        config_path=cfg_path,
        default_height=args.height,
        on_progress=not args.quiet,
    )
    elapsed = time.perf_counter() - t0
    write_csv(rows, args.output)
    print(f"CSV: {args.output.resolve()}")
    print(f"Tiempo real: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    summarize_audit(rows)

    errors = sum(1 for r in rows if r.get("error"))
    if errors:
        print(f"Errores: {errors} (ver columna error)")


if __name__ == "__main__":
    main()