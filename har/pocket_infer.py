"""
API mínima para Kanvis C3: un npy de poses → PocketResult (baseline G+H).

Uso desde C3 (PYTHONPATH = raíz `har/`):

    from pathlib import Path
    from pocket_infer import infer_from_poses_path

    result = infer_from_poses_path(Path("/path/poses_full.npy"))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from har_actions.pose import fps_and_video, frame_aspect
from pocket_filter.action_policy import evaluate_pocket_with_policy
from pocket_filter.config import CONFIG_PATH, load_pocket_config
from pocket_filter.io import load_poses_array
from pocket_filter.types import PocketResult, pocket_result_to_dict


def _load_aux(poses_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    row_mask = None
    vm = poses_path.with_name("valid_mask.npy")
    if vm.is_file():
        row_mask = np.load(vm)
    timestamps = None
    ts = poses_path.with_name("pose_frame_timestamps.npy")
    if ts.is_file():
        timestamps = np.load(ts)
    return row_mask, timestamps


def infer_from_poses_array(
    poses: np.ndarray,
    *,
    poses_path: Path | None = None,
    fps: float | None = None,
    aspect: float | None = None,
    frame_height: float = 1080.0,
    config_path: Path | None = None,
    keypoint_conf: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> PocketResult:
    """Evalúa poses en memoria (misma ruta que batch v2 + temporal + robbery_confirm)."""
    if fps is None:
        if poses_path is None:
            fps = 12.5
        else:
            fps, _ = fps_and_video(poses_path)
    if aspect is None:
        _, video = fps_and_video(poses_path) if poses_path else (fps, None)
        aspect = frame_aspect(video) if poses_path else 16.0 / 9.0
    cfg_path = config_path or CONFIG_PATH
    cfg = load_pocket_config(cfg_path)
    return evaluate_pocket_with_policy(
        poses,
        fps=float(fps),
        aspect=float(aspect),
        frame_height=frame_height,
        timestamps=timestamps,
        row_mask=row_mask,
        keypoint_conf=keypoint_conf,
        config=cfg,
        config_path=cfg_path,
    )


def infer_from_poses_path(
    poses_path: Path,
    *,
    config_path: Path | None = None,
    frame_height: float = 1080.0,
) -> PocketResult:
    """Punto de entrada recomendado para C3: ruta a `poses_full.npy`."""
    poses_path = poses_path.expanduser().resolve()
    try:
        poses, conf = load_poses_array(poses_path)
    except ValueError:
        from har_actions.pose import load_poses

        poses = load_poses(poses_path)
        conf = None
    row_mask, timestamps = _load_aux(poses_path)
    return infer_from_poses_array(
        poses,
        poses_path=poses_path,
        config_path=config_path,
        frame_height=frame_height,
        keypoint_conf=conf,
        row_mask=row_mask,
        timestamps=timestamps,
    )


def infer_to_dict(poses_path: Path, **kwargs) -> dict:
    """Igual que infer_from_poses_path + serialización JSON-friendly."""
    r = infer_from_poses_path(poses_path, **kwargs)
    d = pocket_result_to_dict(r)
    d["poses_path"] = str(poses_path)
    d["orchestrator_branch"] = orchestrator_branch(r)
    return d


def orchestrator_branch(result: PocketResult) -> str:
    """Doctrina P0-3 (sin categoría de carpeta)."""
    if result.motion_suppress:
        return "DISCARD_SUPPRESS"
    if result.hand_to_body:
        return "CONFIRM"
    if result.hand_to_body_raw:
        return "REVIEW_RAW"
    if result.resultado == "INDETERMINADO":
        return "KEEP_INDETERMINADO"
    return "DISCARD_OTHER"