from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

WristId = Literal["left", "right"]
PocketId = Literal["left", "right"]
Verdict = Literal["COMPATIBLE", "INCOMPATIBLE", "INDETERMINADO"]
HtbTier = Literal["none", "review", "alert"]


@dataclass
class PairEvidence:
    wrist: WristId
    pocket: PocketId
    verdict: Verdict
    motivo: str
    t_origen: float | None = None
    d_origen: float | None = None
    t_entrada: float | None = None
    d_entrada: float | None = None
    d_min: float | None = None
    t_retirada: float | None = None
    valid_ratio: float = 0.0
    approach_delta: float | None = None
    ipsilateral: bool = False


@dataclass
class PocketResult:
    resultado: Verdict
    hand_to_body: bool
    filter_fp_safe: bool
    motivo: str
    version: str
    fps: float
    n_frames: int
    eval_start: int = 0
    eval_end: int = 0
    pairs: list[PairEvidence] = field(default_factory=list)
    best_pair: PairEvidence | None = None
    intervalo_s: tuple[float, float] = (0.0, 0.0)
    legacy_hand_to_body: bool | None = None
    legacy_hand_to_body_confidence: float = 0.0
    hand_to_body_v2: bool | None = None
    hand_to_body_raw: bool | None = None
    context_guard_enabled: bool = False
    context_veto: str | None = None
    ctx_walk_frac: float = 0.0
    ctx_stop_frac: float = 0.0
    ctx_hands_away_frac: float = 0.0
    ctx_walk_s: float = 0.0
    ctx_stop_s: float = 0.0
    hand_to_body_confidence: float = 0.0
    primary_action: str = ""
    primary_action_confidence: float = 0.0
    cotidian_seconds: dict[str, float] = field(default_factory=dict)
    htb_tier: HtbTier = "none"
    """Frenar candidato del modelo de robo (FP): sin pocket o pasillo caminando débil."""
    motion_suppress: bool = False
    kin_hip_speed_norm: float = 0.0
    kin_near_dwell_s: float = 0.0
    kin_swing_peaks: float = 0.0
    kin_approach_depth: float = 0.0
    kin_approach_monotone_frac: float = 0.0

    @property
    def accion_propuesta(self) -> str:
        if self.resultado == "COMPATIBLE":
            return "CONSERVAR"
        if self.resultado == "INCOMPATIBLE":
            return "CERRAR_HIPOTESIS_BOLSILLO"
        return "ABSTENERSE"


def pocket_result_to_dict(result: PocketResult) -> dict[str, Any]:
    data = asdict(result)
    data["accion_propuesta"] = result.accion_propuesta
    return data