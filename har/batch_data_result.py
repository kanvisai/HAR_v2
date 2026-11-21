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
  .venv/bin/python3 batch_data_result.py /path/data_result --backend legacy --debug 20 --csv /tmp/legacy.csv
  .venv/bin/python3 batch_data_result.py /path/data_result --backend mix -o /tmp/mix.csv --workers 50
  .venv/bin/python3 batch_data_result.py /path/data_result --backend scores --sample-frac 0.1 -o /tmp/scores.csv
  .venv/bin/python3 batch_data_result.py /path/data_result --category 6 --backend v2 -o cat6_v2.csv
  .venv/bin/python3 batch_data_result.py /path/data_result --sample-frac 0.1 --backend mix -o /tmp/mix_10pct.csv
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
from legacy_hand_to_body import evaluate_legacy
from pocket_filter.action_policy import evaluate_pocket_with_policy
from pocket_filter.motion_intervals import lookup_motion_interval, load_motion_interval_catalog
from pocket_filter.mix_backend import evaluate_pocket_mix
from pocket_filter.score_fusion import evaluate_pocket_scores
from pocket_filter.evaluate import evaluate_pocket
from pocket_filter.io import load_poses_array
from pocket_filter.kinematics import compute_clip_kinematics
from pocket_filter.config import PocketConfig

# Categorías donde esperamos hand_to_body (positivos hurto / campaña 6+14)
EXPECTS_HAND_TO_BODY_TRUE: frozenset[int] = frozenset({6, 14})

# Cotidianas / no robo: esperamos hand_to_body = false
EXPECTS_HAND_TO_BODY_FALSE: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 7, 9, 13})

# Sin etiqueta de expectativa fiable en documentación
EXPECTS_NO_LABEL: frozenset[int] = frozenset({8, 10, 11, 12})

_KIN_CSV_KEYS = (
    "kin_wrist_radial_vel_min",
    "kin_approach_monotone_frac",
    "kin_hip_speed_norm",
    "kin_elbow_angle_deg",
    "kin_wrist_axis_dist_min",
    "kin_near_dwell_s",
    "kin_swing_peaks",
    "kin_approach_depth",
)


def _kin_csv_empty() -> dict[str, str]:
    return dict.fromkeys(_KIN_CSV_KEYS, "")


def _kin_csv_from_result(
    poses: np.ndarray,
    result: Any,
    *,
    fps: float,
    aspect: float,
    frame_h: float,
    cfg: PocketConfig,
    timestamps: np.ndarray | None,
    row_mask: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
) -> dict[str, float]:
    k = compute_clip_kinematics(
        poses,
        result=result,
        fps=fps,
        aspect=aspect,
        frame_height=frame_h,
        cfg=cfg,
        timestamps=timestamps,
        row_mask=row_mask,
        keypoint_conf=keypoint_conf,
    )
    d = k.to_dict()
    return {
        "kin_wrist_radial_vel_min": d["wrist_radial_vel_min"],
        "kin_approach_monotone_frac": d["approach_monotone_frac"],
        "kin_hip_speed_norm": d["hip_speed_norm"],
        "kin_elbow_angle_deg": d["elbow_angle_deg"],
        "kin_wrist_axis_dist_min": d["wrist_axis_dist_min"],
        "kin_near_dwell_s": d["near_dwell_s"],
        "kin_swing_peaks": d["swing_peaks"],
        "kin_approach_depth": d["approach_depth"],
    }


def session_id_from_poses_path(poses_path: Path) -> str:
    """`data_result/<cat>/<session>/user_*/poses_full.npy` → session."""
    parent = poses_path.parent
    if parent.name.startswith("user"):
        return parent.parent.name
    return ""


CSV_FIELDS = [
    "data_root",
    "path",
    "category",
    "session_id",
    "backend",
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
    "kin_wrist_radial_vel_min",
    "kin_approach_monotone_frac",
    "kin_hip_speed_norm",
    "kin_elbow_angle_deg",
    "kin_wrist_axis_dist_min",
    "kin_near_dwell_s",
    "kin_swing_peaks",
    "kin_approach_depth",
    "pocket_eval_mode",
    "interval_t0",
    "interval_t1",
    "motion_peak_score",
    "hand_to_body_confidence",
    "primary_action",
    "primary_action_confidence",
    "htb_tier",
    "motion_suppress",
    "legacy_hand_to_body",
    "legacy_hand_to_body_confidence",
    "hand_to_body_v2",
    "audit_class_v2",
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


def filter_clips_by_category(
    clips: list[ClipRef],
    categories: set[int] | frozenset[int],
) -> list[ClipRef]:
    if not categories:
        return clips
    return [c for c in clips if c.category in categories]


def stratified_sample_clips(
    clips: list[ClipRef],
    frac: float,
    *,
    seed: int,
    min_per_cat: int = 1,
) -> list[ClipRef]:
    """~frac de clips por categoría (mínimo min_per_cat si hay datos). Reproducible con seed."""
    if frac <= 0:
        return clips
    if frac >= 1.0:
        return clips
    by_cat: dict[int, list[ClipRef]] = {}
    for clip in clips:
        by_cat.setdefault(clip.category, []).append(clip)
    rng = random.Random(seed)
    sampled: list[ClipRef] = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        n = min(len(pool), max(min_per_cat, int(round(len(pool) * frac))))
        if n >= len(pool):
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, n))
    sampled.sort(key=lambda c: (str(c.data_root), c.category, str(c.path)))
    return sampled


def _print_stratified_sample_report(before: list[ClipRef], after: list[ClipRef], frac: float) -> None:
    print(
        f"\n=== Muestra estratificada: {frac * 100:.1f}% por categoría ===\n",
        flush=True,
    )
    before_c = Counter(c.category for c in before)
    after_c = Counter(c.category for c in after)
    for cat in sorted(before_c):
        print(
            f"  cat {cat:2d}: {after_c.get(cat, 0):5d} / {before_c[cat]:5d}",
            flush=True,
        )
    print(
        f"\nTotal: {len(after)} / {len(before)} clips "
        f"(métricas de auditoría orientativas, no extrapolación automática)\n",
        flush=True,
    )


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


_MOTION_CATALOG: tuple[str, dict] | None = None


def _motion_catalog(path: str | None) -> dict:
    global _MOTION_CATALOG
    if not path:
        return {}
    if _MOTION_CATALOG is None or _MOTION_CATALOG[0] != path:
        _MOTION_CATALOG = (path, load_motion_interval_catalog(Path(path)))
    return _MOTION_CATALOG[1]


def _pocket_eval_meta() -> dict[str, str]:
    return {
        "pocket_eval_mode": "",
        "interval_t0": "",
        "interval_t1": "",
        "motion_peak_score": "",
    }


def evaluate_one_clip(
    poses_path: Path,
    category: int,
    *,
    backend: str,
    config_path: str | None,
    default_height: float,
    motion_intervals_path: str | None = None,
    motion_min_interval_s: float = 0.35,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        "data_root": "",
        "path": str(poses_path),
        "category": category,
        "session_id": session_id_from_poses_path(poses_path),
        "backend": backend,
        "error": "",
        **_pocket_eval_meta(),
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

        if backend == "legacy":
            leg = evaluate_legacy(poses, fps=fps, aspect=aspect)
            ht = bool(leg.hand_to_body)
            exp_s, match_s, audit = audit_row(category, ht)
            motivo = "LEGACY_HAND_TO_BODY" if ht else "LEGACY_NO_GESTURE"
            if ht and leg.detail:
                motivo = f"{motivo}|{leg.detail[:80]}"
            row.update(
                {
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                    "hand_to_body": ht,
                    "resultado": "LEGACY_HIT" if ht else "LEGACY_NONE",
                    "filter_fp_safe": "",
                    "motivo": motivo,
                    "n_frames": int(leg.n_frames),
                    "expected_hand_to_body": exp_s,
                    "label_match": match_s,
                    "audit_class": audit,
                    "hand_to_body_raw": ht,
                    "context_veto": "",
                    "ctx_walk_frac": "",
                    "ctx_stop_frac": "",
                    "ctx_hands_away_frac": "",
                    "ctx_walk_s": "",
                    "hand_to_body_confidence": "",
                    "primary_action": "",
                    "primary_action_confidence": "",
                    "htb_tier": "",
                    "motion_suppress": "",
                    "legacy_hand_to_body": ht,
                    "legacy_hand_to_body_confidence": round(leg.confidence, 4),
                    "hand_to_body_v2": "",
                    "audit_class_v2": "",
                    **_kin_csv_empty(),
                }
            )
        elif backend == "scores":
            cfg_path = Path(config_path) if config_path else None
            cfg = load_pocket_config(cfg_path)
            result = evaluate_pocket_scores(
                poses,
                fps=fps,
                aspect=aspect,
                frame_height=frame_h,
                timestamps=timestamps,
                row_mask=row_mask,
                keypoint_conf=conf,
                config=cfg,
                config_path=cfg_path,
            )
            ht = bool(result.hand_to_body)
            ht_v2 = bool(
                result.hand_to_body_v2 if result.hand_to_body_v2 is not None else ht
            )
            ht_raw = bool(result.hand_to_body_raw)
            exp_s, match_s, audit = audit_row(category, ht)
            _, _, audit_v2 = audit_row(category, ht_v2)
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
                    "hand_to_body_raw": ht_raw,
                    "context_veto": result.context_veto or "",
                    "ctx_walk_frac": round(result.ctx_walk_frac, 4),
                    "ctx_stop_frac": round(result.ctx_stop_frac, 4),
                    "ctx_hands_away_frac": round(result.ctx_hands_away_frac, 4),
                    "ctx_walk_s": round(result.ctx_walk_s, 3),
                    "hand_to_body_confidence": round(result.hand_to_body_confidence, 4),
                    "primary_action": result.primary_action,
                    "primary_action_confidence": round(
                        result.primary_action_confidence, 4
                    ),
                    "htb_tier": result.htb_tier,
                    "motion_suppress": bool(result.motion_suppress),
                    "legacy_hand_to_body": bool(result.legacy_hand_to_body),
                    "legacy_hand_to_body_confidence": round(
                        float(result.legacy_hand_to_body_confidence), 4
                    ),
                    "hand_to_body_v2": ht_v2,
                    "audit_class_v2": audit_v2,
                    **_kin_csv_from_result(
                        poses,
                        result,
                        fps=fps,
                        aspect=aspect,
                        frame_h=frame_h,
                        cfg=cfg,
                        timestamps=timestamps,
                        row_mask=row_mask,
                        keypoint_conf=conf,
                    ),
                }
            )
        elif backend == "mix":
            cfg_path = Path(config_path) if config_path else None
            cfg = load_pocket_config(cfg_path)
            result = evaluate_pocket_mix(
                poses,
                fps=fps,
                aspect=aspect,
                frame_height=frame_h,
                timestamps=timestamps,
                row_mask=row_mask,
                keypoint_conf=conf,
                config=cfg,
                config_path=cfg_path,
            )
            ht = bool(result.hand_to_body)
            ht_v2 = bool(
                result.hand_to_body_v2 if result.hand_to_body_v2 is not None else ht
            )
            ht_raw = bool(result.hand_to_body_raw)
            exp_s, match_s, audit = audit_row(category, ht)
            _, _, audit_v2 = audit_row(category, ht_v2)
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
                    "hand_to_body_raw": ht_raw,
                    "context_veto": result.context_veto or "",
                    "ctx_walk_frac": round(result.ctx_walk_frac, 4),
                    "ctx_stop_frac": round(result.ctx_stop_frac, 4),
                    "ctx_hands_away_frac": round(result.ctx_hands_away_frac, 4),
                    "ctx_walk_s": round(result.ctx_walk_s, 3),
                    "hand_to_body_confidence": round(result.hand_to_body_confidence, 4),
                    "primary_action": result.primary_action,
                    "primary_action_confidence": round(
                        result.primary_action_confidence, 4
                    ),
                    "htb_tier": result.htb_tier,
                    "motion_suppress": bool(result.motion_suppress),
                    "legacy_hand_to_body": bool(result.legacy_hand_to_body),
                    "legacy_hand_to_body_confidence": round(
                        float(result.legacy_hand_to_body_confidence), 4
                    ),
                    "hand_to_body_v2": ht_v2,
                    "audit_class_v2": audit_v2,
                    **_kin_csv_from_result(
                        poses,
                        result,
                        fps=fps,
                        aspect=aspect,
                        frame_h=frame_h,
                        cfg=cfg,
                        timestamps=timestamps,
                        row_mask=row_mask,
                        keypoint_conf=conf,
                    ),
                }
            )
        else:
            cfg_path = Path(config_path) if config_path else None
            cfg = load_pocket_config(cfg_path)
            catalog = _motion_catalog(motion_intervals_path)
            motion_iv = lookup_motion_interval(poses_path, catalog)
            eval_kwargs: dict[str, Any] = {
                "fps": fps,
                "aspect": aspect,
                "frame_height": frame_h,
                "timestamps": timestamps,
                "row_mask": row_mask,
                "keypoint_conf": conf,
                "config": cfg,
                "config_path": cfg_path,
            }
            if motion_iv is not None and motion_iv.duration_s() >= motion_min_interval_s:
                eval_kwargs["interval_s"] = (motion_iv.t0, motion_iv.t1)
                eval_kwargs["use_temporal"] = False
                row["pocket_eval_mode"] = "motion_interval"
                row["interval_t0"] = round(motion_iv.t0, 4)
                row["interval_t1"] = round(motion_iv.t1, 4)
                if motion_iv.score is not None:
                    row["motion_peak_score"] = round(motion_iv.score, 4)
            else:
                row["pocket_eval_mode"] = "temporal"
            result = evaluate_pocket_with_policy(poses, **eval_kwargs)
            ht = bool(result.hand_to_body)
            ht_raw = bool(result.hand_to_body_raw)
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
                    "hand_to_body_raw": ht_raw,
                    "context_veto": result.context_veto or "",
                    "ctx_walk_frac": round(result.ctx_walk_frac, 4),
                    "ctx_stop_frac": round(result.ctx_stop_frac, 4),
                    "ctx_hands_away_frac": round(result.ctx_hands_away_frac, 4),
                    "ctx_walk_s": round(result.ctx_walk_s, 3),
                    "hand_to_body_confidence": round(result.hand_to_body_confidence, 4),
                    "primary_action": result.primary_action,
                    "primary_action_confidence": round(
                        result.primary_action_confidence, 4
                    ),
                    "htb_tier": result.htb_tier,
                    "motion_suppress": bool(result.motion_suppress),
                    "legacy_hand_to_body": "",
                    "legacy_hand_to_body_confidence": "",
                    "hand_to_body_v2": "",
                    "audit_class_v2": "",
                    **_kin_csv_from_result(
                        poses,
                        result,
                        fps=fps,
                        aspect=aspect,
                        frame_h=frame_h,
                        cfg=cfg,
                        timestamps=timestamps,
                        row_mask=row_mask,
                        keypoint_conf=conf,
                    ),
                }
            )
    except Exception as exc:
        row.update(
            {
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "backend": backend,
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
                "hand_to_body_confidence": "",
                "primary_action": "",
                "primary_action_confidence": "",
                "htb_tier": "",
                "motion_suppress": "",
                "legacy_hand_to_body": "",
                "legacy_hand_to_body_confidence": "",
                "hand_to_body_v2": "",
                "audit_class_v2": "",
                "error": str(exc),
                **_kin_csv_empty(),
            }
        )
    return row


def _worker(
    task: tuple[str, int, str, str, str | None, float, str | None, float],
) -> dict[str, Any]:
    path_s, cat, data_root_s, backend, cfg, height, motion_iv_path, motion_min_s = task
    row = evaluate_one_clip(
        Path(path_s),
        cat,
        backend=backend,
        config_path=cfg,
        default_height=height,
        motion_intervals_path=motion_iv_path,
        motion_min_interval_s=motion_min_s,
    )
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
    backend: str,
    scan_seconds: float,
    workers: int,
    sample_size: int,
    config_path: str | None,
    default_height: float,
) -> None:
    t_preflight0 = time.perf_counter()
    print(f"=== PREFLIGHT data_result | backend={backend} ===\n", flush=True)
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
        backend=backend,
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
    backend: str,
    workers: int,
    config_path: str | None,
    default_height: float,
    on_progress: bool,
    progress_every: int = 0,
    phase_label: str = "EVAL",
    motion_intervals_path: str | None = None,
    motion_min_interval_s: float = 0.35,
) -> list[dict[str, Any]]:
    tasks = [
        (
            str(c.path),
            c.category,
            str(c.data_root.resolve()),
            backend,
            config_path,
            default_height,
            motion_intervals_path,
            motion_min_interval_s,
        )
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


def _audit_table(rows: list[dict[str, Any]], class_key: str, title: str) -> None:
    by_cat: dict[int, Counter] = {}
    for row in rows:
        ac = row.get(class_key, "")
        if ac in ("", "SKIP", "ERROR"):
            continue
        cat = int(row["category"])
        by_cat.setdefault(cat, Counter())[ac] += 1
    if not by_cat:
        return
    print(f"\n{title}")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        labeled = c["TP"] + c["TN"] + c["FP"] + c["FN"]
        ok = c["TP"] + c["TN"]
        print(
            f"  cat {cat:2d}: {ok}/{labeled} aciertos  "
            f"TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}"
        )


def summarize_audit(rows: list[dict[str, Any]]) -> None:
    _audit_table(
        rows,
        "audit_class",
        "Resumen auditoría hand_to_body (mix fusionado / v2)",
    )
    if any(str(r.get("audit_class_v2", "")).strip() for r in rows):
        _audit_table(
            rows,
            "audit_class_v2",
            "Resumen auditoría hand_to_body_v2 (solo v2, referencia)",
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
    parser.add_argument(
        "--backend",
        choices=("v2", "legacy", "mix", "scores"),
        default="v2",
        help="v2 | legacy | mix | scores (v2+legacy conf+cotidiano)",
    )
    parser.add_argument("--height", type=float, default=1080.0)
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestras aleatorias")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.0,
        metavar="F",
        help="Evalúa F×100%% de clips **por categoría** (p. ej. 0.1 ≈ 10%%). Requiere al menos 1 clip/cat. "
        "Indicativo; mucho más rápido que barrido completo.",
    )
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
    parser.add_argument(
        "-c",
        "--category",
        type=int,
        nargs="+",
        metavar="N",
        default=None,
        help="Solo estas categorías (p. ej. -c 6 o -c 6 14). Escaneo completo; evalúa solo el subconjunto.",
    )
    parser.add_argument(
        "--motion-intervals",
        type=Path,
        default=None,
        metavar="JSONL",
        help="P0-5: JSONL con path,t0,t1[,score]; si hay hit, evalúa en intervalo (sin ventanas OR).",
    )
    parser.add_argument(
        "--motion-min-interval-s",
        type=float,
        default=0.35,
        help="Duración mínima del intervalo Motion para usarlo (s).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cfg_path = str(args.config.resolve()) if args.config else None
    motion_iv_path = (
        str(args.motion_intervals.expanduser().resolve()) if args.motion_intervals else None
    )

    verbose_scan = not args.no_scan_log
    if args.preflight and args.debug <= 0:
        verbose_scan = True
    if args.debug > 0 and not args.no_scan_log:
        verbose_scan = True
    heartbeat_scan = not args.quiet

    mode = "preflight" if args.preflight and args.debug <= 0 else ("debug" if args.debug > 0 else "batch")
    roots_banner = ", ".join(str(r) for r in args.data_roots)
    print(
        f"\n>>> batch_data_result | modo={mode} | backend={args.backend} | raíces: {roots_banner}\n",
        flush=True,
    )

    discovered = discover_clips(
        args.data_roots,
        verbose=verbose_scan,
        heartbeat=heartbeat_scan,
    )
    clips = discovered.clips
    scan_s = discovered.scan_seconds

    if args.category is not None:
        allowed = frozenset(args.category)
        before = len(clips)
        clips = filter_clips_by_category(clips, allowed)
        print(
            f"Filtro categorías {sorted(allowed)}: {len(clips)} clips a evaluar "
            f"(de {before} en inventario completo)",
            flush=True,
        )
        if clips:
            by_f = Counter(c.category for c in clips)
            for cat in sorted(by_f):
                print(f"  cat {cat:2d}: {by_f[cat]}", flush=True)
        print(flush=True)

    if not clips and not args.preflight:
        roots_s = ", ".join(str(r) for r in args.data_roots)
        raise SystemExit(f"No se encontraron poses_full.npy bajo user_* en: {roots_s}")

    clips_full = clips
    if args.sample_frac > 0:
        if not (0 < args.sample_frac < 1.0):
            raise SystemExit("--sample-frac debe estar entre 0 y 1 (exclusivo), p. ej. 0.1")
        clips = stratified_sample_clips(
            clips, args.sample_frac, seed=args.seed, min_per_cat=1
        )
        _print_stratified_sample_report(clips_full, clips, args.sample_frac)

    if args.preflight and args.debug <= 0:
        preflight_report(
            clips,
            args.data_roots,
            backend=args.backend,
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
            backend=args.backend,
            workers=min(args.workers, n),
            config_path=cfg_path,
            default_height=args.height,
            on_progress=not args.quiet,
            progress_every=args.progress_every,
            phase_label="DEBUG",
            motion_intervals_path=motion_iv_path,
            motion_min_interval_s=args.motion_min_interval_s,
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
    if motion_iv_path:
        print(f"P0-5: sidecar Motion intervals → {motion_iv_path}", flush=True)
    rows = run_batch(
        clips,
        backend=args.backend,
        workers=args.workers,
        config_path=cfg_path,
        default_height=args.height,
        on_progress=not args.quiet,
        progress_every=args.progress_every,
        phase_label="BATCH",
        motion_intervals_path=motion_iv_path,
        motion_min_interval_s=args.motion_min_interval_s,
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