#!/usr/bin/env python3
"""
Dibuja bolsillos, anillos r/R y exporta series d(t) por pareja.

Uso:
  .venv/bin/python3 visualize_pocket.py poses_full.npy -o temp_output/pocket_debug.mp4
  .venv/bin/python3 visualize_pocket.py poses_full.npy --save-series temp_output/d_series.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_actions.pose import fps_and_video, frame_aspect
from pocket_filter.config import load_pocket_config
from pocket_filter.evaluate import _dist_series, _frame_geometry, _pocket_center, slice_indices
from pocket_filter.io import load_poses_array
from pocket_filter.masks import frame_exportable_mask

SHOULDER_L, SHOULDER_R = 0, 1
WRIST_L, WRIST_R = 4, 5
HIP_L, HIP_R = 6, 7


def collect_series(
    poses: np.ndarray,
    fps: float,
    aspect: float,
    frame_height: float,
    row_mask: np.ndarray | None,
    conf: np.ndarray | None,
) -> dict:
    cfg = load_pocket_config()
    t = len(poses)
    exportable = frame_exportable_mask(t, row_mask)
    eval_start, eval_end = slice_indices(t, fps, None, 0.0, None)
    eval_mask = np.zeros(t, dtype=bool)
    eval_mask[eval_start : eval_end + 1] = True
    out = {}
    for wrist, wix in (("left", WRIST_L), ("right", WRIST_R)):
        for pocket, pix in (("left", HIP_L), ("right", HIP_R)):
            d, valid, _, _ = _dist_series(
                poses, wix, pix, aspect, frame_height, cfg, eval_mask, exportable, conf
            )
            key = f"{wrist}_{pocket}"
            out[key] = {
                "d": np.nan_to_num(d, nan=-1).tolist(),
                "valid": valid.astype(int).tolist(),
                "r_near": cfg.r_near,
                "r_far": cfg.r_far,
            }
    return out


def render_video(
    poses: np.ndarray,
    video: Path,
    output: Path,
    fps: float,
    aspect: float,
    frame_height: float,
) -> None:
    cfg = load_pocket_config()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"No se abre vídeo: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    out = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        vfps,
        (w, h),
    )
    t = 0
    while t < len(poses):
        ok, frame = cap.read()
        if not ok:
            break
        pose = poses[t]
        ok_geom, reason, _, s, hc, u, hips, _ = _frame_geometry(pose, aspect, frame_height, cfg)
        if ok_geom and hips is not None and u is not None:
            hip_l, hip_r = hips
            for hip, side in ((hip_l, SHOULDER_L), (hip_r, SHOULDER_R)):
                shoulder_side = SHOULDER_L if side == SHOULDER_L else SHOULDER_R
                ln = float(np.linalg.norm(hc - s))
                b = _pocket_center(hip, u, ln, shoulder_side, cfg)
                for wx, wy in ((b[0], b[1]),):
                    cx, cy = int(wx * w), int(wy * h)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
                    rad_r = int(cfg.r_near * ln * h)
                    rad_R = int(cfg.r_far * ln * h)
                    cv2.circle(frame, (cx, cy), rad_r, (0, 255, 0), 1)
                    cv2.circle(frame, (cx, cy), rad_R, (0, 128, 255), 1)
            for wix, color in ((WRIST_L, (255, 0, 0)), (WRIST_R, (255, 128, 0))):
                if np.isfinite(pose[wix]).all():
                    cx, cy = int(pose[wix, 0] * aspect * w), int(pose[wix, 1] * h)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.putText(
            frame,
            f"f{t}  {reason if not ok_geom else 'ok'}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        out.write(frame)
        t += 1
    cap.release()
    out.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("poses", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "temp_output" / "pocket_debug.mp4")
    parser.add_argument("--save-series", type=Path, default=None)
    parser.add_argument("--height", type=float, default=1080.0)
    args = parser.parse_args()
    poses, conf = load_poses_array(args.poses)
    fps, video = fps_and_video(args.poses)
    aspect = frame_aspect(video)
    row = None
    vm = args.poses.with_name("valid_mask.npy")
    if vm.is_file():
        row = np.load(vm)

    series = collect_series(poses, fps, aspect, args.height, row, conf)
    if args.save_series:
        args.save_series.parent.mkdir(parents=True, exist_ok=True)
        args.save_series.write_text(json.dumps(series, indent=2), encoding="utf-8")
        print(f"Series: {args.save_series}")
    if video and video.is_file():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        render_video(poses, video, args.output, fps, aspect, args.height)
        print(f"Video:  {args.output}")
    else:
        print("Sin clip.mp4: solo series (--save-series).")


if __name__ == "__main__":
    main()