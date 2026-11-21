"""P0-5: sidecar JSONL con intervalo Motion por clip (`t0`,`t1` en segundos)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionInterval:
    t0: float
    t1: float
    score: float | None = None

    def duration_s(self) -> float:
        return max(0.0, self.t1 - self.t0)


def _norm_key(path: str) -> str:
    return path.replace("\\", "/").strip()


def load_motion_interval_catalog(path: Path) -> dict[str, MotionInterval]:
    """Carga JSONL: una línea por clip con `path`, `t0`, `t1` y opcional `score`."""
    out: dict[str, MotionInterval] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            p = _norm_key(str(row.get("path", "")))
            if not p:
                continue
            iv = MotionInterval(
                float(row["t0"]),
                float(row["t1"]),
                float(row["score"]) if row.get("score") is not None else None,
            )
            out[p] = iv
            out[Path(p).name] = iv
    return out


def lookup_motion_interval(
    poses_path: Path,
    catalog: dict[str, MotionInterval],
) -> MotionInterval | None:
    if not catalog:
        return None
    candidates = [
        _norm_key(str(poses_path)),
        _norm_key(str(poses_path.resolve())),
    ]
    parents = poses_path.parts
    for i, part in enumerate(parents):
        if part.startswith("user_"):
            tail = _norm_key("/".join(parents[i - 2 :]))
            candidates.append(tail)
            candidates.append(_norm_key("/".join(parents[i - 1 :])))
            break
    candidates.append(poses_path.name)
    for key in candidates:
        if key in catalog:
            return catalog[key]
    return None