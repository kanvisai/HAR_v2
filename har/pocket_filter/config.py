"""Carga de configuración del filtro pocket (hand_to_body v2)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "pocket_filter_config.json"


@dataclass
class PocketConfig:
    pocket_offset: float = 0.08
    pocket_lateral_offset: float = 0.04
    r_near: float = 0.30
    r_far: float = 0.50
    delta_min: float = 0.16
    delta_tuck_extra: float = 0.06
    deep_contact_r: float = 0.22
    sustained_contact_s: float = 0.28
    extend_axis_min: float = 0.42
    tuck_axis_max: float = 0.20
    swing_axis_margin: float = 0.06
    max_near_swing_episodes: int = 2
    approach_window_s: float = 1.50
    approach_min_lead_s: float = 0.10
    contact_consecutive: int = 2
    min_torso_px: float = 40.0
    side_view_ratio: float = 0.25
    identity_hip_jump_L: float = 0.75
    identity_jump_window_s: float = 0.15
    scale_jump_ratio: float = 0.40
    max_time_gap_factor: float = 1.5
    min_valid_ratio_short_clip: float = 0.35
    min_valid_ratio_long_clip: float = 0.65
    short_clip_s: float = 2.0
    long_clip_s: float = 5.0
    prefer_ipsilateral: bool = True
    contralateral_delta_bonus: float = 0.08
    frozen_repeat_frames: int = 5
    frozen_torso_eps: float = 0.008
    version: str = "pocket-v2"


def load_pocket_config(path: Path | None = None) -> PocketConfig:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        return PocketConfig()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    pocket = data.get("pocket", data)
    fields = {f.name for f in PocketConfig.__dataclass_fields__.values()}  # type: ignore
    kwargs = {k: v for k, v in pocket.items() if k in fields}
    if "version" in data and "version" not in kwargs:
        kwargs["version"] = data["version"]
    return PocketConfig(**kwargs)


_V2_ALIASES = frozenset({"v2", "ignacio", "pocket", "pocket_v2", "pocket-v2"})


def _normalize_backend(name: str) -> str:
    n = name.lower().strip()
    if n in _V2_ALIASES:
        return "v2"
    if n == "legacy":
        return "legacy"
    return n


def hand_to_body_backend(path: Path | None = None) -> str:
    """`v2` = pocket_filter; `legacy` = heurística HAR pasillo."""
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        return "v2"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return _normalize_backend(str(data.get("hand_to_body_backend", "v2")))


def hand_to_body_uses_pocket_filter(path: Path | None = None) -> bool:
    return hand_to_body_backend(path) == "v2"