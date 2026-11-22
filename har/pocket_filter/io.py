"""Carga de poses_full.npy compatible con HAR (8,2) y export Kanvis (8,3 xyn)."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

MIN_CONF = 0.50


def _split_poses_raw(raw: np.ndarray, source: str) -> tuple[np.ndarray, np.ndarray | None]:
    if raw.ndim != 3 or raw.shape[1] != 8:
        raise ValueError(f"{source} shape {raw.shape}; se espera (T, 8, 2) o (T, 8, 3).")
    if raw.shape[2] == 2:
        return raw, None
    if raw.shape[2] >= 3:
        xy = raw[:, :, :2].copy()
        conf = raw[:, :, 2]
        for t in range(xy.shape[0]):
            for k in range(8):
                if not np.isfinite(conf[t, k]) or conf[t, k] < MIN_CONF:
                    xy[t, k] = np.nan
        return xy, conf
    raise ValueError(f"{source}: canal 3 inesperado.")


def load_poses_array(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Devuelve (poses (T,8,2), keypoint_conf (T,8) o None).

    Si el archivo trae un tercer canal, se usa como confianza y NaN donde conf < MIN_CONF.
    """
    raw = np.load(path).astype(np.float64)
    return _split_poses_raw(raw, str(path))


def load_poses_array_from_bytes(data: bytes, *, source: str = "poses_full.npy") -> tuple[np.ndarray, np.ndarray | None]:
    """Igual que load_poses_array para buffers (p. ej. objeto MinIO en memoria)."""
    raw = np.load(BytesIO(data), allow_pickle=False).astype(np.float64)
    return _split_poses_raw(raw, source)