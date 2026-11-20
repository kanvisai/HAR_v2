#!/usr/bin/env python3
"""
Prueba del filtro pocket v2 (aproximación muñeca → bolsillo) sobre poses_full.npy.

Uso:
  .venv/bin/python3 har2_test.py /ruta/poses_full.npy
  .venv/bin/python3 har2_test.py /ruta/poses_full.npy --json -o eval_pocket_output.json
  .venv/bin/python3 har2_test.py /ruta/poses_full.npy --compare-legacy
  .venv/bin/python3 har2_test.py /ruta/poses_full.npy --interval 2.0 6.0

``hand_to_body`` = True solo si el resultado es COMPATIBLE.
``filter_fp_safe`` = True solo si INCOMPATIBLE (candidato a filtrar FP de robo).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from har_actions.detect import detect, detect_mobility_context, gesture_seconds
from har_actions.pose import fps_and_video, frame_aspect, load_poses, smooth
from pocket_filter.config import hand_to_body_backend, load_pocket_config
from pocket_filter.evaluate import evaluate_pocket
from pocket_filter.io import load_poses_array
from pocket_filter.types import pocket_result_to_dict


def _optional_npy(path: Path) -> np.ndarray | None:
    return np.load(path) if path.is_file() else None


def _legacy_hand_to_body(poses: np.ndarray, fps: float, aspect: float) -> bool:
    smoothed = smooth(poses, window=3)
    actions = detect(smoothed, fps, aspect)
    return any(a.id == "hand_to_body" for a in actions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtro pocket v2: ¿hay hand_to_body?")
    parser.add_argument("poses", type=Path, help="poses_full.npy")
    parser.add_argument(
        "--valid-mask",
        type=Path,
        default=None,
        help="valid_mask.npy (T,) o (T,8)",
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=None,
        help="pose_frame_timestamps.npy",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="pocket_filter_config.json",
    )
    parser.add_argument(
        "--interval",
        type=float,
        nargs=2,
        metavar=("T0", "T1"),
        default=None,
        help="Ventana de evaluación en segundos (gesto motion)",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--height", type=float, default=1080.0)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None, help="eval_pocket_output.json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--legacy-compat", action="store_true")
    args = parser.parse_args()

    if not args.poses.is_file():
        raise SystemExit(f"No existe: {args.poses}")

    conf: np.ndarray | None
    try:
        poses, conf = load_poses_array(args.poses)
    except ValueError:
        poses = load_poses(args.poses)
        conf = None

    fps, video = fps_and_video(args.poses)
    if args.fps is not None:
        fps = args.fps
    aspect = frame_aspect(video)
    frame_h = args.height
    frame_w = args.width
    if frame_w is None and video is not None and video.is_file():
        import cv2

        cap = cv2.VideoCapture(str(video))
        frame_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    mask_path = args.valid_mask or args.poses.with_name("valid_mask.npy")
    row_mask = _optional_npy(mask_path) if mask_path.is_file() else None

    ts_path = args.timestamps or args.poses.with_name("pose_frame_timestamps.npy")
    timestamps = _optional_npy(ts_path) if ts_path.is_file() else None

    interval = tuple(args.interval) if args.interval else None
    cfg = load_pocket_config(args.config) if args.config else load_pocket_config()

    result = evaluate_pocket(
        poses,
        fps=fps,
        aspect=aspect,
        frame_height=frame_h,
        frame_width=frame_w,
        config=cfg,
        timestamps=timestamps,
        row_mask=row_mask,
        keypoint_conf=conf,
        interval_s=interval,
    )

    hand_to_body = result.resultado != "INCOMPATIBLE" if args.legacy_compat else result.hand_to_body

    if args.compare_legacy:
        result.legacy_hand_to_body = _legacy_hand_to_body(poses, fps, aspect)

    document = pocket_result_to_dict(result)
    document["poses"] = str(args.poses)
    document["hand_to_body"] = hand_to_body
    document["hand_to_body_backend_config"] = hand_to_body_backend(args.config)

    out_path = args.output or args.poses.with_name("eval_pocket_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(document, ensure_ascii=False, indent=2))
        return

    print(f"Archivo:         {args.poses}")
    print(f"Frames:          {result.n_frames}  eval [{result.eval_start}-{result.eval_end}]  fps: {fps:.2f}")
    print(f"Backend config:  {document['hand_to_body_backend_config']} (detect: legacy|v2)")
    print(f"Resultado:       {result.resultado}")
    print(f"hand_to_body:    {'sí' if hand_to_body else 'no'}")
    if result.hand_to_body_raw is not None and result.hand_to_body_raw != hand_to_body:
        print(f"hand_to_body_raw:  {'sí' if result.hand_to_body_raw else 'no'}  (pocket sin contexto)")
    print(f"filter_fp_safe:  {'sí' if result.filter_fp_safe else 'no'}")
    print(f"Motivo:          {result.motivo}")
    if result.context_guard_enabled:
        mob = gesture_seconds(detect_mobility_context(smooth(poses, window=3), fps, aspect), fps)
        print(
            f"Contexto clip:   walk {mob.get('walk', 0):.1f}s  "
            f"stop {mob.get('stop', 0):.1f}s  "
            f"(ventana gesto: walk {result.ctx_walk_frac:.0%}  "
            f"stop {result.ctx_stop_frac:.0%}  "
            f"manos lejos {result.ctx_hands_away_frac:.0%})"
        )
        if result.context_veto:
            print(f"Context veto:    {result.context_veto}")
    print(f"Acción:          {result.accion_propuesta}")
    print(f"JSON:            {out_path}")
    if result.best_pair:
        bp = result.best_pair
        print(f"Mejor pareja:     {bp.wrist}→bolsillo_{bp.pocket}  Δd={bp.approach_delta}")
    if args.legacy_compat:
        print("(legacy-compat: hand_to_body = no INCOMPATIBLE)")
    if result.pairs:
        print()
        print("Parejas:")
        for p in result.pairs:
            tag = f"{p.wrist}→bolsillo_{p.pocket}"
            extra = ""
            if p.verdict == "COMPATIBLE" and p.d_entrada is not None:
                if p.t_origen is not None and p.d_origen is not None:
                    extra = f"  t {p.t_origen:.2f}→{p.t_entrada:.2f}  d {p.d_origen:.2f}→{p.d_entrada:.2f}"
                else:
                    extra = f"  entrada t={p.t_entrada:.2f}  d_min={p.d_min}"
            print(f"  {tag:22} {p.verdict:14} {p.motivo}{extra}")
    if args.compare_legacy and result.legacy_hand_to_body is not None:
        print()
        print(f"HAR legacy _hand_to_body: {'sí' if result.legacy_hand_to_body else 'no'}")


if __name__ == "__main__":
    main()