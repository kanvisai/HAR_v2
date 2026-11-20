#!/usr/bin/env python3
"""
Prueba del filtro Ignacio (aproximación muñeca → bolsillo) sobre poses_full.npy.

Uso:
  .venv/bin/python3 har2_test.py /ruta/poses_full.npy
  .venv/bin/python3 har2_test.py /ruta/poses_full.npy --json
  .venv/bin/python3 har2_test.py /ruta/poses_full.npy --compare-legacy

``hand_to_body`` = True solo si el resultado es COMPATIBLE (trayectoria lejos→cerca visible).
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

from har_actions.detect import detect
from har_actions.pose import fps_and_video, frame_aspect, load_poses, smooth
from pocket_filter.evaluate import evaluate_pocket, pocket_result_to_dict
from pocket_filter.io import load_poses_array


def _optional_npy(path: Path) -> np.ndarray | None:
    return np.load(path) if path.is_file() else None


def _legacy_hand_to_body(poses_path: Path, poses: np.ndarray, fps: float, aspect: float) -> bool:
    smoothed = smooth(poses, window=3)
    actions = detect(smoothed, fps, aspect)
    return any(a.id == "hand_to_body" for a in actions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtro Ignacio v1: ¿hay hand_to_body?")
    parser.add_argument("poses", type=Path, help="poses_full.npy")
    parser.add_argument(
        "--valid-mask",
        type=Path,
        default=None,
        help="valid_mask.npy o keypoint_mask alineado (opcional)",
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=None,
        help="pose_frame_timestamps.npy (opcional)",
    )
    parser.add_argument("--fps", type=float, default=None, help="Forzar FPS")
    parser.add_argument("--height", type=float, default=1080.0, help="Altura de frame en px")
    parser.add_argument("--width", type=float, default=None, help="Ancho de frame en px")
    parser.add_argument("--json", action="store_true", help="Salida JSON completa")
    parser.add_argument("--compare-legacy", action="store_true", help="Comparar con HAR _hand_to_body")
    parser.add_argument("--legacy-compat", action="store_true", help="Compat: COMPATIBLE o INDETERMINADO → hand_to_body")
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

    mask_path = args.valid_mask
    if mask_path is None:
        sibling = args.poses.with_name("valid_mask.npy")
        if sibling.is_file():
            mask_path = sibling
    wrist_mask = _optional_npy(mask_path) if mask_path else None
    if wrist_mask is not None and wrist_mask.ndim == 1:
        wrist_mask = None
    if wrist_mask is None and conf is not None:
        wrist_mask = np.isfinite(conf) & (conf >= 0.50)

    ts_path = args.timestamps
    if ts_path is None:
        sibling = args.poses.with_name("pose_frame_timestamps.npy")
        if sibling.is_file():
            ts_path = sibling
    timestamps = _optional_npy(ts_path) if ts_path else None

    result = evaluate_pocket(
        poses,
        fps=fps,
        aspect=aspect,
        frame_height=frame_h,
        frame_width=frame_w,
        timestamps=timestamps,
        wrist_mask=wrist_mask,
    )

    if args.legacy_compat:
        hand_to_body = result.resultado != "INCOMPATIBLE"
    else:
        hand_to_body = result.hand_to_body

    if args.compare_legacy:
        result.legacy_hand_to_body = _legacy_hand_to_body(args.poses, poses, fps, aspect)

    document = pocket_result_to_dict(result)
    document["poses"] = str(args.poses)
    document["hand_to_body"] = hand_to_body

    if args.json:
        print(json.dumps(document, ensure_ascii=False, indent=2))
        return

    print(f"Archivo:       {args.poses}")
    print(f"Frames:        {result.n_frames}  fps: {fps:.2f}")
    print(f"Resultado:     {result.resultado}")
    print(f"hand_to_body:  {'sí' if hand_to_body else 'no'}")
    print(f"Motivo:        {result.motivo}")
    print(f"Acción:        {result.accion_propuesta}")
    if args.legacy_compat:
        print("(Modo legacy-compat: hand_to_body = no INCOMPATIBLE)")
    if result.pairs:
        print()
        print("Parejas (muñeca → bolsillo):")
        for p in result.pairs:
            tag = f"{p.wrist}→bolsillo_{p.pocket}"
            extra = ""
            if p.verdict == "COMPATIBLE" and p.d_entrada is not None:
                extra = (
                    f"  origen t={p.t_origen:.2f}s d={p.d_origen:.2f}"
                    f" → entrada t={p.t_entrada:.2f}s d={p.d_entrada:.2f} d_min={p.d_min:.2f}"
                )
            print(f"  {tag:22} {p.verdict:14} {p.motivo}{extra}")
    if args.compare_legacy and result.legacy_hand_to_body is not None:
        leg = "sí" if result.legacy_hand_to_body else "no"
        print()
        print(f"HAR legacy _hand_to_body: {leg}")


if __name__ == "__main__":
    main()