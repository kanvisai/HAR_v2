#!/usr/bin/env python3
"""
Barrido de data_result por categorías y evaluación pocket v2 (hand_to_body).

Estructura esperada:
  data_result/<categoría>/<sesión>/user_xx/poses_full.npy

Ejemplos:
  .venv/bin/python3 batch_data_result.py /path/data_result --preflight
  .venv/bin/python3 batch_data_result.py /path/data_result --csv /tmp/informe.csv --workers 20
  .venv/bin/python3 batch_data_result.py /path/data_result --debug 10
  .venv/bin/python3 batch_data_result.py /path/data_result --debug 10 --csv /tmp/debug_muestra.csv
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
    "data_root",
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
    "hand_to_body_raw",
    "context_veto",
    "ctx_walk_frac",
    "ctx_stop_frac",
    "ctx_hands_away_frac",
    "ctx_walk_s",
    "error",
]


@dataclass(frozen=True)
class ClipRef:
    path: Path
    category: int
    data_root: Path


@dataclass(frozen=True)
class DiscoveryResult:
    clips: list[ClipRef]
    scan_seconds: float


def _scan_one_data_root(root: Path, *, verbose: bool, heartbeat: bool) -> list[ClipRef]:
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"No es un directorio: {root}")

    found: list[ClipRef] = []
    cat_dirs = [
        p
        for p in sorted(root.iterdir(), key=lambda p: p.name)
        if p.is_dir() and p.name.isdigit()
    ]
    if verbose or heartbeat:
        print(f"── Escaneando: {root}", flush=True)
        print(f"   Carpetas de categoría: {len(cat_dirs)}", flush=True)

    for i, cat_dir in enumerate(cat_dirs, start=1):
        category = int(cat_dir.name)
        n_before = len(found)
        if verbose:
            print(f"   cat {category:2d}: buscando poses_full.npy …", flush=True)
        elif heartbeat:
            print(
                f"   [{i}/{len(cat_dirs)}] cat {category:2d}: escaneando …",
                flush=True,
            )
        for poses_path in cat_dir.rglob("poses_full.npy"):
            parent = poses_path.parent
            if parent.name.startswith("user_"):
                found.append(ClipRef(poses_path.resolve(), category, root))
        n_cat = len(found) - n_before
        if verbose and n_cat:
            print(f"   cat {category:2d}: +{n_cat:5d} clips  (acumulado: {len(found)})", flush=True)
        elif heartbeat:
            print(f"        → {n_cat} clips en cat {category}  (acumulado raíz: {len(found)})", flush=True)

    if verbose or heartbeat:
        print(f"   Subtotal en esta raíz: {len(found)} clips\n", flush=True)
    return found


def discover_clips(
    data_roots: list[Path],
    *,
    verbose: bool = False,
    heartbeat: bool = True,
) -> DiscoveryResult:
    t_scan0 = time.perf_counter()
    merged: list[ClipRef] = []
    if heartbeat and not verbose:
        print(
            "Inventario: buscando poses_full.npy (disco de red = minutos sin bloquear; "
            "verás una línea por categoría)…",
            flush=True,
        )
    for root in data_roots:
        merged.extend(_scan_one_data_root(root, verbose=verbose, heartbeat=heartbeat and not verbose))
    merged.sort(key=lambda c: (str(c.data_root), c.category, str(c.path)))
    scan_s = time.perf_counter() - t_scan0
    if verbose or heartbeat:
        print(
            f"Escaneo terminado en {scan_s:.1f} s ({scan_s / 60:.1f} min) — {len(merged)} clips\n",
            flush=True,
        )
        if len(data_roots) > 1:
            print(f"Total combinado ({len(data_roots)} raíz/raíces): {len(merged)} clips\n", flush=True)
    return DiscoveryResult(merged, scan_s)


def _short_path(path: Path, max_len: int = 72) -> str:
    s = str(path)
    if len(s) <= max_len:
        return s
    return "…" + s[-(max_len - 1) :]


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or not (seconds < 1e12):
        return "—"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _print_progress(
    done: int,
    total: int,
    t0: float,
    *,
    prefix: str = "",
    detail: str = "",
    errors: int = 0,
    mean_clip_s: float | None = None,
    workers: int = 1,
) -> None:
    elapsed = max(time.perf_counter() - t0, 1e-6)
    remain = max(total - done, 0)
    pct_done = 100.0 * done / total if total else 0.0
    pct_left = 100.0 * remain / total if total else 0.0
    rate = done / elapsed

    # ETA por ritmo real (clips/s en esta corrida)
    eta_wall_s = remain / rate if rate > 0 else 0.0

    # ETA por media de tiempo CPU por clip (más estable cuando arranca el pool)
    if mean_clip_s is not None and mean_clip_s > 0 and workers > 0 and remain > 0:
        eta_mean_s = (remain * mean_clip_s) / max(workers, 1) * 1.05
    else:
        eta_mean_s = eta_wall_s

    bar_w = 28
    filled = int(bar_w * done / total) if total else 0
    bar = "#" * filled + "-" * (bar_w - filled)
    err_part = f"  |  errores={errors}" if errors else ""
    mean_part = f"  |  media {mean_clip_s:.2f} s/clip" if mean_clip_s else ""

    line = (
        f"{prefix}[{bar}]  {pct_done:5.1f}%  ({done}/{total})  "
        f"|  pendiente {pct_left:5.1f}% ({remain} clips){mean_part}  "
        f"|  restante ~{_fmt_duration(eta_mean_s)}  "
        f"(wall ~{_fmt_duration(eta_wall_s)}){err_part}"
    )
    if detail:
        line += f"\n{prefix}      último: {detail}"
    print(line, flush=True)


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
        "data_root": "",
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
                "hand_to_body_raw": result.hand_to_body_raw
                if result.hand_to_body_raw is not None
                else ht,
                "context_veto": result.context_veto or "",
                "ctx_walk_frac": round(result.ctx_walk_frac, 4),
                "ctx_stop_frac": round(result.ctx_stop_frac, 4),
                "ctx_hands_away_frac": round(result.ctx_hands_away_frac, 4),
                "ctx_walk_s": round(result.ctx_walk_s, 3),
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
                "hand_to_body_raw": "",
                "context_veto": "",
                "ctx_walk_frac": "",
                "ctx_stop_frac": "",
                "ctx_hands_away_frac": "",
                "ctx_walk_s": "",
                "error": str(exc),
            }
        )
    return row


def _worker(task: tuple[str, int, str, str | None, float]) -> dict[str, Any]:
    path_s, cat, data_root_s, cfg, height = task
    row = evaluate_one_clip(Path(path_s), cat, config_path=cfg, default_height=height)
    row["data_root"] = data_root_s
    return row


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
    data_roots: list[Path],
    *,
    scan_seconds: float,
    workers: int,
    sample_size: int,
    config_path: str | None,
    default_height: float,
) -> None:
    t_preflight0 = time.perf_counter()
    print("=== PREFLIGHT pocket v2 / data_result ===\n", flush=True)
    print(
        f"Nota: el escaneo de disco (buscar poses_full.npy) ya consumió "
        f"{scan_seconds / 60:.1f} min; el batch completo lo repetirá una vez al inicio.\n",
        flush=True,
    )
    roots_unique = sorted({c.data_root.resolve() for c in clips})
    if not clips:
        for r in data_roots:
            print(f"  (vacío) {r.resolve()}", flush=True)
        print("\nSin archivos que analizar.")
        return

    by_root: dict[Path, int] = Counter(c.data_root.resolve() for c in clips)
    print("Raíces incluidas:", flush=True)
    for r in roots_unique:
        print(f"  • {r}  →  {by_root[r]} clips", flush=True)

    by_cat = Counter(c.category for c in clips)
    print(f"\nTotal clips (poses_full.npy bajo user_*): {len(clips)}")
    print(f"Workers planificados (batch completo): {workers}")

    print("\nClips por categoría (global):")
    for cat in sorted(by_cat):
        note = ""
        if cat in EXPECTS_HAND_TO_BODY_TRUE:
            note = "  [esperado hand_to_body=true]"
        elif cat in EXPECTS_HAND_TO_BODY_FALSE:
            note = "  [esperado hand_to_body=false]"
        elif cat in EXPECTS_NO_LABEL:
            note = "  [sin etiqueta fija]"
        print(f"  {cat:2d}: {by_cat[cat]:5d}{note}")

    if len(roots_unique) > 1:
        print("\nClips por categoría y raíz:")
        for r in roots_unique:
            sub = [c for c in clips if c.data_root.resolve() == r]
            sub_cat = Counter(c.category for c in sub)
            print(f"  {r}", flush=True)
            for cat in sorted(sub_cat):
                print(f"    cat {cat:2d}: {sub_cat[cat]:5d}")

    if sample_size <= 0:
        print("\n(--sample 0) Sin benchmark de evaluación; solo inventario + escaneo.", flush=True)
        print(f"\nTiempo de este preflight: {time.perf_counter() - t_preflight0:.1f} s", flush=True)
        print("=== fin preflight ===", flush=True)
        return

    n_sample = min(sample_size, len(clips))
    sample = clips[:n_sample] if n_sample == len(clips) else random.sample(clips, n_sample)
    bench_workers = min(workers, n_sample)
    print(
        f"\nBenchmark evaluación: {n_sample} clip(s) con {bench_workers} workers "
        f"(igual que el batch, sin escaneo)...",
        flush=True,
    )
    t_bench0 = time.perf_counter()
    bench_rows = run_batch(
        sample,
        workers=bench_workers,
        config_path=config_path,
        default_height=default_height,
        on_progress=True,
        progress_every=max(1, n_sample // 8),
        phase_label="PREFLIGHT-BENCH",
    )
    bench_wall = time.perf_counter() - t_bench0

    times = [float(r["elapsed_s"]) for r in bench_rows if not r.get("error")]
    sample_errors = sum(1 for r in bench_rows if r.get("error"))
    if not times:
        print("\nNo se pudo estimar tiempo (errores en todas las muestras).", flush=True)
        return

    ok_n = len(times)
    avg = sum(times) / ok_n
    p95 = sorted(times)[int(0.95 * max(ok_n - 1, 0))]
    total = len(clips)

    # Throughput real de la mini-corrida (incluye arranque del pool + I/O de disco)
    bench_throughput = ok_n / max(bench_wall, 1e-6)
    est_eval_parallel = total / bench_throughput
    # Límite inferior teórico (solo CPU por clip, peor caso con colas)
    est_eval_parallel_p95 = (total * p95) / max(bench_workers, 1) * 1.1
    est_eval_parallel = max(est_eval_parallel, est_eval_parallel_p95)

    est_eval_serial = avg * total
    est_batch_total = scan_seconds + est_eval_parallel
    est_batch_serial = scan_seconds + est_eval_serial

    preflight_wall = scan_seconds + (time.perf_counter() - t_preflight0)

    print(f"\nMuestra OK: {ok_n}/{n_sample}  (errores: {sample_errors})", flush=True)
    print(f"  benchmark wall: {bench_wall:.1f} s ({bench_wall / 60:.1f} min)")
    print(f"  throughput medido: {bench_throughput:.3f} clip/s  ({60 / bench_throughput:.2f} s/clip efectivo)")
    print(f"  CPU por clip (media / p95): {avg:.3f} s / {p95:.3f} s")

    print("\n--- Estimación batch COMPLETO (desde cero: escaneo + evaluación) ---", flush=True)
    print(f"  1) Escaneo inventario:     {scan_seconds / 60:.1f} min  ({scan_seconds:.0f} s)")
    print(
        f"  2) Evaluación paralela:     {est_eval_parallel / 60:.1f} min  ({est_eval_parallel:.0f} s)"
        f"  [{workers} workers]"
    )
    print(
        f"  TOTAL estimado:            {est_batch_total / 60:.1f} min  ({est_batch_total:.0f} s)"
    )
    print("\n  (Referencia eval secuencial sin paralelizar:", flush=True)
    print(f"   escaneo + eval ≈ {est_batch_serial / 60:.1f} min)", flush=True)

    print(
        f"\nEste --preflight lleva ~{preflight_wall / 60:.1f} min hasta aquí "
        f"(escaneo {scan_seconds / 60:.1f} + benchmark {bench_wall / 60:.1f}); "
        f"no es comparable solo con la línea 'evaluación paralela'."
    )
    print("\n=== fin preflight ===", flush=True)


def run_batch(
    clips: list[ClipRef],
    *,
    workers: int,
    config_path: str | None,
    default_height: float,
    on_progress: bool,
    progress_every: int = 0,
    phase_label: str = "EVAL",
) -> list[dict[str, Any]]:
    tasks = [
        (str(c.path), c.category, str(c.data_root.resolve()), config_path, default_height)
        for c in clips
    ]
    rows: list[dict[str, Any]] = []
    done = 0
    errors = 0
    elapsed_sum = 0.0
    elapsed_n = 0
    total = len(tasks)
    pool_workers = max(1, workers)
    if progress_every <= 0:
        if total <= 30:
            progress_every = 1
        else:
            progress_every = max(1, min(pool_workers, total // 50 or 1))
    t0 = time.perf_counter()
    last_progress_t = t0

    print(
        f"\n── {phase_label}: {total} clip(s), {pool_workers} worker(s) ──",
        flush=True,
    )
    print(
        "Preparando pool de procesos (cada worker carga NumPy/HAR; "
        "pueden pasar 15–60 s sin nueva línea)…",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=pool_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        print(f"Tareas enviadas: {total}. Esperando resultados…", flush=True)
        got_first = False
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            done += 1
            if not got_first:
                print(
                    f"Primer clip completado tras {time.perf_counter() - t0:.1f} s "
                    f"(cat={row.get('category')} hand_to_body={row.get('hand_to_body')!r})",
                    flush=True,
                )
                got_first = True
            if row.get("error"):
                errors += 1
            else:
                try:
                    elapsed_sum += float(row.get("elapsed_s", 0) or 0)
                    elapsed_n += 1
                except (TypeError, ValueError):
                    pass
            mean_clip = elapsed_sum / elapsed_n if elapsed_n else None
            now = time.perf_counter()
            should_log = (
                done == 1
                or done == total
                or (now - last_progress_t) >= 2.0
                or done % progress_every == 0
            )
            if on_progress and should_log:
                detail = _short_path(Path(row.get("path", "")), 90)
                if row.get("error"):
                    detail += "  ERR"
                else:
                    detail += (
                        f"  htb={_format_bool_cell(row.get('hand_to_body'))}"
                        f" {row.get('audit_class', '')}"
                    )
                _print_progress(
                    done,
                    total,
                    t0,
                    prefix="  ",
                    detail=detail,
                    errors=errors,
                    mean_clip_s=mean_clip,
                    workers=pool_workers,
                )
                last_progress_t = now

    if on_progress and total:
        mean_clip = elapsed_sum / elapsed_n if elapsed_n else None
        _print_progress(
            total,
            total,
            t0,
            prefix="  ",
            detail="completado",
            errors=errors,
            mean_clip_s=mean_clip,
            workers=pool_workers,
        )

    rows.sort(key=lambda r: (r.get("data_root", ""), int(r["category"]), r["path"]))
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


def _configure_stdio() -> None:
    """Evita salida en buffer (SSH) hasta que el proceso termina."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description="Barrido data_result → CSV pocket v2 / hand_to_body"
    )
    parser.add_argument(
        "data_roots",
        type=Path,
        nargs="+",
        metavar="DATA_RESULT",
        help="Una o más carpetas data_result (cada una con 0/, 1/, 2/, ...)",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--csv",
        dest="output",
        type=Path,
        metavar="RUTA_CSV",
        default=Path("pocket_data_result.csv"),
        help="Ruta del CSV de resultados (absoluta o relativa; crea carpetas padre si faltan). "
        "Solo en modo batch (no con --preflight ni --debug).",
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
        help="Evalúa N clips: tabla en pantalla; con --csv también guarda ese subconjunto",
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
        help="Sin heartbeat de escaneo ni barra de progreso en evaluación",
    )
    parser.add_argument(
        "--no-scan-log",
        action="store_true",
        help="No imprimir detalle por categoría al escanear (solo totales)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Imprimir progreso cada N clips (0 = automático)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cfg_path = str(args.config.resolve()) if args.config else None

    verbose_scan = not args.no_scan_log
    if args.preflight and args.debug <= 0:
        verbose_scan = True
    if args.debug > 0 and not args.no_scan_log:
        verbose_scan = True
    heartbeat_scan = not args.quiet

    mode = "preflight" if args.preflight and args.debug <= 0 else ("debug" if args.debug > 0 else "batch")
    roots_banner = ", ".join(str(r) for r in args.data_roots)
    print(f"\n>>> batch_data_result | modo={mode} | raíces: {roots_banner}\n", flush=True)

    discovered = discover_clips(
        args.data_roots,
        verbose=verbose_scan,
        heartbeat=heartbeat_scan,
    )
    clips = discovered.clips
    scan_s = discovered.scan_seconds
    if not clips and not args.preflight:
        roots_s = ", ".join(str(r) for r in args.data_roots)
        raise SystemExit(f"No se encontraron poses_full.npy bajo user_* en: {roots_s}")

    if args.preflight and args.debug <= 0:
        preflight_report(
            clips,
            args.data_roots,
            scan_seconds=scan_s,
            workers=args.workers,
            sample_size=args.sample,
            config_path=cfg_path,
            default_height=args.height,
        )
        return

    if args.debug > 0:
        n = min(args.debug, len(clips))
        subset = clips[:n] if n == len(clips) else random.sample(clips, n)
        print(f"=== DEBUG: evaluar {n} de {len(clips)} clips (semilla {args.seed}) ===\n", flush=True)
        for i, clip in enumerate(subset, start=1):
            try:
                rel = clip.path.relative_to(clip.data_root)
            except ValueError:
                rel = clip.path
            print(f"  {i:2d}. cat={clip.category}  {_short_path(rel, 88)}", flush=True)
        print(flush=True)
        rows = run_batch(
            subset,
            workers=min(args.workers, n),
            config_path=cfg_path,
            default_height=args.height,
            on_progress=not args.quiet,
            progress_every=args.progress_every,
            phase_label="DEBUG",
        )
        print("\n=== Tabla resultado ===\n", flush=True)
        print_debug_table(rows)
        summarize_audit(rows)
        out_path = args.output.expanduser().resolve()
        write_csv(rows, out_path)
        print(f"\n(debug) CSV guardado: {out_path}", flush=True)
        return

    out_path = args.output.expanduser().resolve()
    roots_line = ", ".join(str(r.resolve()) for r in args.data_roots)
    print(
        f"Evaluando {len(clips)} clips | {args.workers} workers | raíces: {roots_line}",
        flush=True,
    )
    print(f"CSV de salida: {out_path}", flush=True)
    if scan_s > 1.0:
        print(f"(Escaneo previo: {scan_s:.1f} s — ya contado antes del batch en esta ejecución)", flush=True)
    t0 = time.perf_counter()
    rows = run_batch(
        clips,
        workers=args.workers,
        config_path=cfg_path,
        default_height=args.height,
        on_progress=not args.quiet,
        progress_every=args.progress_every,
        phase_label="BATCH",
    )
    eval_elapsed = time.perf_counter() - t0
    total_elapsed = scan_s + eval_elapsed
    write_csv(rows, out_path)
    print(f"CSV escrito: {out_path}")
    print(
        f"Tiempo real: eval {eval_elapsed:.1f} s ({eval_elapsed / 60:.1f} min)  "
        f"+ escaneo {scan_s:.1f} s → total {total_elapsed:.1f} s ({total_elapsed / 60:.1f} min)"
    )
    summarize_audit(rows)

    errors = sum(1 for r in rows if r.get("error"))
    if errors:
        print(f"Errores: {errors} (ver columna error)")


if __name__ == "__main__":
    main()