# Hand-to-body filter — engineering status (HAR_v2)

**Last updated:** 2026-11-21  
**Audience:** engineers extending the pocket filter, motion orchestration, or alternative fusion policies.  
**Related (Spanish ops):** [`HAND_TO_BODY_BASELINE_PRODUCCION.md`](HAND_TO_BODY_BASELINE_PRODUCCION.md), [`HAND_TO_BODY_PROGRESO.md`](HAND_TO_BODY_PROGRESO.md), [`HAND_TO_BODY_SOLUCIONES.md`](HAND_TO_BODY_SOLUCIONES.md) — **§0ter** (problem statement, *soluciones buscadas*).

**Status:** Prototype **shipped and measured**; **product problem not solved** — we need **new engineering approaches**, not only threshold tweaks.

---

## Executive summary — the unsolved problem

### What hurts in production

Kanvis motion (C2) flags **robbery candidates** on ~8 s skeleton chunks. Many are **everyday aisle behaviour**. This repo must add **geometric hand-to-body evidence** so the orchestrator can **confirm**, **send to review**, or **suppress** motion — **without** human category labels at runtime.

If we treat a single **`hand_to_body=true`** as “robbery confirmed”, we are stuck between two failure modes measured on the audit dataset:

| Regime | Cat 6 recall (clear) | Cat 0 FP rate | Problem |
|--------|----------------------|---------------|---------|
| Legacy HAR | ~37% | ~3% | **Misses real thefts** |
| Pocket v2, no policy | ~94% | ~82% | **Floods aisle false alarms** |
| **Frozen G+H (today’s baseline)** | **~84%** | **~9.4%** (~434 FPs) | **Still loses ~671 robberies on clear**; FP still too high for cheap human review |

There is **no configuration found so far** that simultaneously meets stakeholder targets on **one strict boolean**. The agreed **three-signal contract** (`motion_suppress` → `hand_to_body` → `hand_to_body_raw`) is the **intended fix**, but **fusion policy, calibration, and orchestrator wiring are incomplete**.

### What we have already tried (and why it is not enough)

- **Global confidence gates** → recall collapse on cat 6.
- **Temporal windows** → necessary for recall inside 8 s chunks; **increases** aisle FP vs no-temporal gate.
- **`walk_gate`, mix, scores, legacy veto** → small FP gains or **large** recall losses; band arbitration ≈ v2 on sample.
- **Heuristic “confidence” scores** → not calibrated; fusion by threshold **over-suppresses** real thefts.

**Engineering conclusion:** incremental rules on the same pose features are **hitting a ceiling**. We need **new solution designs** (see §8 and Spanish §0ter in `HAND_TO_BODY_SOLUCIONES.md`).

### What “success” still means (not yet achieved)

| Goal | G+H frozen baseline | Gap |
|------|----------------------|-----|
| High theft recall on **clear** confirm | 3521 / 4192 (~84%) | ~671 FN; many with `raw=true` |
| Low aisle FP on **clear** | 434 / 4633 (~9.4%) | Too many escalations if each FP → SAM/human |
| Safe abstention | Pocket ternary + `raw` in CSV | Production path not fully using review tier |
| Reproducible rollback | `BASELINE_PRODUCCION_H.json` | Exists — **does not** mean problem closed |

### Call for solutions

Contributors should propose **measurable** changes that:

1. Reduce **FP on 0/1/5…** without dropping cat **6/14** clear recall below agreed floor (or move recall to **`raw` + review** with quantified cost).
2. Use **motion interval / score** (evaluate pocket on C2 peak, not blind 2.5 s slides only).
3. Replace **one global threshold** with **motive-conditional** or **learned fusion** on existing batch features.
4. Ship with **batch + audit** numbers vs frozen baseline — not sample-only anecdotes.

---

## 1. Context

### 1.1 Product setting

- **Input:** one `poses_full.npy` per **~8 s video chunk** in supermarket aisle footage (Kanvis / pilot).
- **Upstream:** a separate **motion model** flags robbery *candidates*; this repo does **not** train that model.
- **This repo’s job:** geometric **hand-to-body** evidence from skeleton poses to:
  - **Confirm** real concealment gestures (categories **6** and **14** in the audit dataset).
  - **Reject** everyday aisle behaviour (especially category **0** “walk-through”, plus 1, 5, …).
- **Critical constraint:** at runtime there is **no human category label** — only the chunk and motion score.

### 1.2 Two historical backends

| Backend | Implementation | Strengths | Weaknesses |
|---------|------------------|-----------|------------|
| **legacy** | `har_actions/detect.py` → `_hand_to_body` on smoothed poses | Very low FP on aisle clips (~3% on cat 0 full sweep); **fast** | Poor robbery recall (~37% cat 6 on full sweep) |
| **v2 (pocket)** | `pocket_filter/` geometry + policy + optional **temporal windows** | Strong robbery recall (~84% cat 6 full baseline); richer signals (`confidence`, `raw`, `motion_suppress`) | Higher FP on cat 0 vs legacy; **~5× slower** per clip when temporal is on |

There is **no single `hand_to_body` boolean** that simultaneously matches pocket-pure recall on 6 and `robbery_confirm` FP on 0. Production is designed around **multiple signals** (see §5).

### 1.3 Audit dataset

- Primary benchmark:  
  `/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result`
- Evaluation tool: `har/batch_data_result.py` → CSV → `har/audit_pocket_csv.py`
- Labels use folder category IDs (6/14 = expect `hand_to_body=true`; 0/1/5/… = expect false).
- Fast iteration: **`--sample-frac 0.1 --seed 42`** stratified per category (~4 min vs ~40 min full).

---

## 2. What we are trying to achieve

1. **Primary (business):** When motion raises a robbery candidate, reduce **false confirmations** on everyday clips (esp. cat 0) **without** collapsing detection on real thefts (6/14).
2. **Secondary:** Recover gestures that occur **late inside the 8 s chunk** (not diluted by the whole buffer).
3. **Operational:** Document a **frozen config** for rollback and compare experiments reproducibly.
4. **Explicit non-goals (for this filter):** Fine-grained classification of all aisle behaviours as a product feature; training a new neural net on poses here.

**Desired orchestration contract** (agreed with stakeholders):

```text
on motion_robbery_candidate:
  if motion_suppress:           discard (likely aisle FP)
  elif hand_to_body:            hard confirm
  elif hand_to_body_raw:        pocket gesture present → SAM / human review
  else:                         discard
```

`hand_to_body` is **strict**; recall on 6 can be recovered via **`hand_to_body_raw`** without reopening 82% FP rates from pocket-pure.

---

## 3. Architecture today (v2 production stack)

### 3.1 Pipeline (backend `v2`)

1. **Pocket geometry** (`pocket_filter/evaluate.py`) — motives such as `CONTACTO_BOLSILLO`, `APROXIMACION_OBSERVADA`, `ACERCAMIENTO_EJE_TORSO`.
2. **Confidence** (`confidence.py`) — scalar `hand_to_body_confidence` ∈ [0, 1] (heuristic, not calibrated probability).
3. **Action policy** (`action_policy.py`), mode **`gate_mode: robbery_confirm`**:
   - Strong contact: relatively low threshold.
   - Approach / axis while **walking**: blocked or high threshold.
   - Axis while **quiet** (`stop_frac`): allowed with moderate confidence.
4. **Temporal fusion** (`temporal_eval.py`), default **G+H**:
   - Windows **2.5 s**, step **1.25 s** on ~8 s clips.
   - `hand_to_body` = OR across windows passing policy; metadata from best peak.
   - **`walk_gate`** (optional): among confirming windows, veto weak axis if that window is **walking** (reduces cat 0 FP at small recall cost).
5. **Legacy path** (optional backends): separate code path, not mixed into v2 unless using `mix` / `scores`.

### 3.2 Config files

| File | Role |
|------|------|
| `har/pocket_filter_config.json` | Active tuning (may include `walk_gate`, `mix_backend`, `score_fusion`) |
| `har/pocket_filter_config.BASELINE_PRODUCCION_H.json` | **Frozen G+H** reproducing full-sweep **3521/4192** cat 6 TP and **434** cat 0 FP (no walk_gate) |

### 3.3 Batch backends (`--backend`)

| Value | Meaning |
|-------|---------|
| `v2` | Pocket + policy + temporal (default) |
| `legacy` | Old `_hand_to_body` only |
| `mix` | v2 + legacy boolean + cotidian context; fusion mode in `mix_backend.mix_mode` |
| `scores` | v2 + **legacy confidence** + cotidian peaks; rule block `score_fusion` |

**CLI examples:**

```bash
cd har
.venv/bin/python3 batch_data_result.py /path/data_result \
  --backend v2 -o /tmp/out.csv --workers 50

.venv/bin/python3 batch_data_result.py /path/data_result \
  --backend mix --sample-frac 0.1 --seed 42 -o /tmp/mix_10.csv
```

**Single clip:** `har2_test.py poses_full.npy` (add `--backend mix|scores`, `--whole-clip` to disable temporal).

---

## 4. Experiment timeline (what we tried)

| Phase | Idea | Outcome (full dataset unless noted) |
|-------|------|-------------------------------------|
| Pocket pure | No policy | **94%** cat 6, **82%** FP cat 0 — unusable for aisle |
| Policy v1 | Global min confidence | ~49% cat 6 — too harsh |
| Policy v2 | Walk/stop per motive | ~80% cat 6, ~28% FP cat 0 — bridge |
| **G: `robbery_confirm`** | Binary gate for motion | **70%** cat 6, **164** FP cat 0 **without** temporal |
| **H: temporal windows** | Peak in 2.5 s slices | **84%** cat 6, **434** FP cat 0 — **baseline G+H** |
| **I: `temporal.walk_gate`** | Veto confirm if peak window walks | **~83%** cat 6, **~361** FP cat 0 (trade ~57 TP for ~73 FP) |
| **mix `legacy_veto`** (early) | Legacy vetoes low-conf v2 | Pasillo better, **~50%** cat 6 — abandoned |
| **mix `v2_first`** | ≈ v2; veto only axis+walk | On 10% sample: **≈ identical** to v2 |
| **mix `legacy_arbitrate`** | conf &lt;10% → no; [10%,55%) → legacy decides; else v2 rules | 10% sample: **≈ identical** to v2 on 6/0 |
| **`scores` fusion** | Cotidian/legacy scores vs v2 | 10% sample: slight FP gain, **large** cat 6 drop — **not recommended** as-is |

---

## 5. Reference metrics (full sweep, `hand_to_body` column)

Dataset: `datos_limpios_triton/data_result`. Audit column: **`hand_to_body`** (fused clear), not `raw`.

### 5.1 Headline comparison

| Configuration | Cat 6 TP | Cat 6 recall | Cat 0 FP | Cat 0 FP rate | Cat 14 TP |
|---------------|----------|--------------|----------|---------------|-----------|
| Pocket pure | 3957 | ~94% | 3816 | ~82% | ~93% |
| `robbery_confirm`, no temporal | 2925 | ~70% | 164 | ~3.5% | ~299 |
| **G+H baseline (frozen)** | **3521** | **~84%** | **434** | **~9.4%** | **334** |
| G+H + walk gate (measured once) | 3464 | ~83% | 361 | ~7.8% | 328 |
| legacy backend | 1535 | ~37% | 147 | ~3.2% | 162 |

FN on 6 under G+H: **671** (many still have `hand_to_body_raw=true` — check CSV for “gesture in pose” recall).

### 5.2 10% stratified sample (seed 42) — recent backends

Approximate, indicative only:

| Backend | Cat 6 TP (of 419) | Cat 0 FP (of 463) | Notes |
|---------|-------------------|-------------------|-------|
| v2 / mix `v2_first` / `legacy_arbitrate` | 347 (~83%) | 44 (~9.5%) | Matched |
| `scores` | 281 (~67%) | 41 (~8.9%) | Hurts robbery clear |
| early broken `mix` | 265 (~63%) | 31 (~6.7%) | Legacy veto too aggressive |

---

## 6. Interim deployment only — not “problem solved” (2026-11-21)

**Stakeholder priority stated:** minimise FP on non-6/14 categories while keeping robbery detection usable.

Until a solution from §8 lands, we only have an **interim** stack (below). Treat **`BASELINE_PRODUCCION_H.json`** as rollback reference, **not** as acceptance that metrics are good enough for unconstrained auto-confirm.

| Layer | Recommendation |
|-------|----------------|
| **Batch / default config** | **`--backend v2`** with temporal + `robbery_confirm` |
| **FP vs recall knob** | Enable **`temporal.walk_gate`** if accepting ~1.5% fewer cat-6 clears for ~1.6% fewer cat-0 FP vs frozen G+H |
| **Rollback** | `cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json` |
| **Motion integration** | Use **three signals**; measure **`hand_to_body_raw`** on cat 6 for SAM path |
| **mix / scores** | Implemented for research; **`legacy_arbitrate`** is safe but rarely changes 10% sample vs v2; **`scores`** needs redesign (do not veto strong v2 on cotidian alone) |
| **legacy** | Keep for fast baseline / optional second opinion; not as sole `hand_to_body` in prod |

**Compute:** Run full v2 only on **motion candidates** if the orchestrator can gate; legacy is cheap for sanity checks.

---

## 7. Key code map (for contributors)

```
har/
  batch_data_result.py      # --backend v2|legacy|mix|scores, --sample-frac
  audit_pocket_csv.py       # TP/TN/FP/FN; mix prints v2 reference column too
  legacy_hand_to_body.py      # legacy + legacy_hand_to_body_confidence
  pocket_filter/
    evaluate.py               # pocket geometry
    confidence.py             # v2 confidence
    action_policy.py          # robbery_confirm, motion_suppress
    temporal_eval.py          # windows + walk_gate merge
    mix_backend.py            # mix_mode: v2_first | legacy_arbitrate | legacy_veto
    score_fusion.py           # scores backend
  pocket_filter_config.json
  pocket_filter_config.BASELINE_PRODUCCION_H.json
```

**Mix modes (config `mix_backend.mix_mode`):**

- **`v2_first`:** Final clear ≈ v2; only extra veto = weak **axis** + dominant **walk** in peak window.
- **`legacy_arbitrate`:** `conf < 0.10` → no; `[0.10, 0.55)` → legacy boolean arbitrates clear; above strong thresholds → trust v2 (contact/approx/high conf).
- **`legacy_veto`:** Historical; aggressive legacy veto on weak v2 — **deprecated**.

**Scores (`score_fusion`):** Threshold rules comparing v2 conf, legacy conf, cotidian peak; current defaults over-suppress v2 clear on cat 6 in sample.

---

## 8. Open problems / ideas for smarter approaches

1. **Single scalar is insufficient:** Joint optimisation of `hand_to_body`, `raw`, and `motion_suppress` under motion priors — not a single threshold.
2. **Calibration:** All “confidences” are heuristics (time fractions + geometry margins). A small labelled validation set could calibrate isotonic or Platt scaling **per motive**.
3. **Temporal vs motion interval:** If motion provides **[t0, t1]** of suspicion, evaluate pocket **only on that interval** (C2 in design docs) — may beat sliding 2.5 s blind windows.
4. **Category-free fusion that worked in principle but not in practice:**
   - Legacy **boolean veto** when v2 disagrees → kills robbery recall.
   - **Score fusion** with global cotidian wins → same issue.
   - **Band arbitration** (legacy only between 10–55% v2 conf) → safe but **low incidence** in sample; may need wider band or motive-conditioned bands.
5. **axis motive:** Most aisle FP and policy FN tension concentrates on `ACERCAMIENTO_EJE_TORSO` vs `CONTACTO_BOLSILLO`; motive-aware policies outperform global cuts.
6. **Full raw recall:** Report `hand_to_body_raw` TP on cat 6 alongside `hand_to_body` before tightening clear further.
7. **Alternative architectures:** Learned fusion (small GBM) on `{v2_conf, motive, walk_frac, legacy_conf, …}` with clip-level labels; event-level HAR listing actions with timestamps instead of one clip vote.

---

## 9. How to reproduce and extend

```bash
# Full reference sweep (slow)
.venv/bin/python3 batch_data_result.py "$DATA" --backend v2 \
  -o /tmp/gate_prod.csv --workers 50
.venv/bin/python3 audit_pocket_csv.py /tmp/gate_prod.csv

# Quick 10% stratified compare
for b in v2 mix scores; do
  .venv/bin/python3 batch_data_result.py "$DATA" --backend "$b" \
    --sample-frac 0.1 --seed 42 -o "/tmp/${b}_10.csv" --workers 50
done

# Raw / clear dual metrics — see Appendix D
.venv/bin/python3 audit_htb_metrics.py /tmp/gate_prod.csv
```

When changing production behaviour:

1. Copy config → `pocket_filter_config.PRUEBA_<date>.json`.
2. One logical change per experiment.
3. Store CSV with fixed name; update [`HAND_TO_BODY_BASELINE_PRODUCCION.md`](HAND_TO_BODY_BASELINE_PRODUCCION.md) only if replacing the frozen baseline.

---

## 10. Summary table (one page)

| Question | Answer |
|----------|--------|
| **Is the product problem solved?** | **No** — G+H is best **compromise** so far; §Executive summary + §8 |
| What works for **robbery clear**? | v2 + temporal + `robbery_confirm` (~84% cat 6 full) — **671 FN remain** |
| What works for **aisle FP**? | legacy alone (~3% FP, unusable recall); no single boolean hits both targets |
| Frozen baseline numbers? | **3521 TP cat 6**, **434 FP cat 0** — rollback only, **not** success criteria met |
| Best FP/recall trade seen? | G+H + **walk_gate**: marginal; **not** a structural fix |
| mix / scores ready for prod? | **No** — research only; scores hurts cat 6 |
| How should motion use outputs? | **suppress → clear → raw** — **needs new fusion design** (§8) |
| Where to propose solutions? | [`HAND_TO_BODY_SOLUCIONES.md`](HAND_TO_BODY_SOLUCIONES.md) **§0ter** + **§14** template |

**Self-contained handoff:** sections 1–10 plus **appendices A–I** (environment, CSV schema, JSON baselines, raw vs clear metrics, full audit table, rollback, Kanvis contract). Spanish deep-dive remains in the linked `HAND_TO_BODY_*.md` files.

---

## Appendix A — Environment and repo layout

### A.1 Prerequisites

- Python 3 with project venv at `har/.venv`
- Dataset tree: `data_result/<category_id>/<session>/user_*/poses_full.npy`
- Optional sidecars: `valid_mask.npy`, `pose_frame_timestamps.npy`, `meta.json`, video under clip folder

### A.2 Setup (typical server)

```bash
cd ~/kanvis/piloto_v1/HAR_v2/har   # or local clone …/HAR_v2/har
git pull
python3 -m venv .venv              # once
.venv/bin/pip install -r requirements.txt   # if present
.venv/bin/python3 -m pytest tests/ -q
```

### A.3 Data layout

```
data_result/
  0/<session>/user_00/poses_full.npy
  6/...
  14/...
```

**Category semantics (audit only):**  
Expect `hand_to_body=true` for **6, 14**; expect **false** for **0, 1, 2, 3, 4, 5, 7, 9, 13**. Categories **8, 10, 11, 12** → `audit_class=SKIP`.

### A.4 Output artifacts

| Artifact | Purpose |
|----------|---------|
| `gate_temporal.csv` | Reference full sweep **G+H baseline** (frozen config, no walk_gate) |
| `pocket_filter_config.BASELINE_PRODUCCION_H.json` | Byte-stable config for that sweep |
| `har2_test` JSON | Per-clip `eval_pocket_output.json` with `intervalo_s`, scores |

---

## Appendix B — Batch CSV schema

Defined in `har/batch_data_result.py` → `CSV_FIELDS` (stable order).

| # | Column | Type | Meaning |
|---|--------|------|---------|
| 1 | `data_root` | path | Root scanned |
| 2 | `path` | path | `poses_full.npy` |
| 3 | `category` | int | Folder label (audit) |
| 4 | `backend` | str | `v2` \| `legacy` \| `mix` \| `scores` |
| 5 | `elapsed_s` | float | Wall time per clip |
| 6 | `hand_to_body` | bool | **Final clear** (after policy / fusion) |
| 7 | `resultado` | str | Pocket verdict or `LEGACY_*` |
| 8 | `filter_fp_safe` | bool | Pocket INCOMPATIBLE |
| 9 | `motivo` | str | Pocket motive (+ optional `|MIX_*` / rule suffix) |
| 10 | `n_frames` | int | Clip length |
| 11 | `expected_hand_to_body` | str | Label expectation |
| 12 | `label_match` | str | Matches expectation |
| 13 | `audit_class` | str | **TP \| TN \| FP \| FN \| SKIP \| ERROR** for `hand_to_body` |
| 14 | `hand_to_body_raw` | bool | Pocket COMPATIBLE **before** strict fusion (gesture in pose) |
| 15 | `context_veto` | str | Policy veto or fusion rule code |
| 16 | `ctx_walk_frac` | float | Walk fraction in eval window |
| 17 | `ctx_stop_frac` | float | Stop fraction |
| 18 | `ctx_hands_away_frac` | float | Hands away fraction |
| 19 | `ctx_walk_s` | float | Walk seconds |
| 20 | `hand_to_body_confidence` | float | v2 score 0–1 |
| 21 | `primary_action` | str | Cotidian or `hand_to_body` |
| 22 | `primary_action_confidence` | float | Cotidian dominance 0–1 |
| 23 | `htb_tier` | str | `alert` \| `review` \| `none` |
| 24 | `motion_suppress` | bool | Suggest suppressing motion candidate |
| 25 | `legacy_hand_to_body` | bool | Legacy boolean (empty for pure `v2`) |
| 26 | `legacy_hand_to_body_confidence` | float | Legacy v2 score (mix/scores/legacy) |
| 27 | `hand_to_body_v2` | bool | v2 clear **before** mix/scores (empty for pure `v2`) |
| 28 | `audit_class_v2` | str | Audit using `hand_to_body_v2` (mix/scores only) |
| 29 | `error` | str | Exception text if failed |

**Audit logic** (`audit_row`): for labeled categories, compares **`hand_to_body`** (column 6) to expectation — not `raw`, unless you compute separately (Appendix D).

**Awk column numbers** = table **#** column (1-based), e.g. `$3` = category, `$6` = clear, `$14` = raw.

---

## Appendix C — Configuration reference

### C.1 Frozen baseline (`pocket_filter_config.BASELINE_PRODUCCION_H.json`)

Reproduces **3521 / 4192** cat 6 clear TP and **434** cat 0 FP on full `datos_limpios_triton` sweep.

| Block | Key settings |
|-------|----------------|
| `temporal` | `enabled: true`, `window_s: 2.5`, `step_s: 1.25`, **no `walk_gate_*`** |
| `action_policy` | `enabled: true`, `gate_mode: robbery_confirm` |
| `context_guard` | `enabled: false` |

Full file: `har/pocket_filter_config.BASELINE_PRODUCCION_H.json` (reproduces `gate_temporal.csv`).

<details>
<summary>Frozen baseline JSON (click to expand)</summary>

```json
{
  "version": "pocket-v2",
  "hand_to_body_backend": "v2",
  "pocket": {
    "pocket_offset": 0.08,
    "pocket_lateral_offset": 0.04,
    "r_near": 0.3,
    "r_far": 0.5,
    "delta_min": 0.18,
    "delta_tuck_extra": 0.06,
    "deep_contact_r": 0.22,
    "sustained_contact_s": 0.28,
    "extend_axis_min": 0.42,
    "tuck_axis_max": 0.20,
    "max_near_swing_episodes": 2,
    "swing_axis_margin": 0.06,
    "approach_window_s": 1.5,
    "approach_min_lead_s": 0.1,
    "contact_consecutive": 2,
    "min_torso_px": 40.0,
    "side_view_ratio": 0.25,
    "allow_side_view_geometry": true,
    "enable_axis_body_approach": true,
    "axis_extend_min": 0.38,
    "axis_tuck_max": 0.22,
    "axis_delta_min": 0.14,
    "identity_hip_jump_L": 0.75,
    "identity_jump_window_s": 0.15,
    "scale_jump_ratio": 0.4,
    "max_time_gap_factor": 1.5,
    "min_valid_ratio_short_clip": 0.35,
    "min_valid_ratio_long_clip": 0.65,
    "short_clip_s": 2.0,
    "long_clip_s": 5.0,
    "prefer_ipsilateral": true,
    "contralateral_delta_bonus": 0.08,
    "frozen_repeat_frames": 5,
    "frozen_torso_eps": 0.008
  },
  "temporal": {
    "enabled": true,
    "window_s": 2.5,
    "step_s": 1.25,
    "context_pad_s": 0.35
  },
  "action_policy": {
    "enabled": true,
    "gate_mode": "robbery_confirm",
    "gate_contact_min_confidence_while_walking": 0.62,
    "strong_min_confidence": 0.48,
    "approx_min_confidence": 0.56,
    "approx_min_confidence_while_walking": 0.68,
    "axis_min_confidence": 0.50,
    "axis_min_confidence_while_walking": 0.66,
    "axis_quiet_stop_frac": 0.30,
    "walk_active_frac": 0.44,
    "walk_active_clip_min_s": 1.0,
    "cotidian_min_visible_frac": 0.08,
    "inconclusive_below_s": 0.25,
    "axis_max_tier": "review",
    "review_min_confidence": 0.48,
    "review_axis_min_confidence": 0.52
  },
  "context_guard": {
    "enabled": false
  }
}
```

</details>

### C.2 Active config (`pocket_filter_config.json`) — research extras

In addition to baseline pocket + policy + temporal, the active file may include:

| Block | Purpose |
|-------|---------|
| `temporal.walk_gate_*` | Optional FP reduction on weak axis in walking peak window |
| `mix_backend` | `mix_mode`: `legacy_arbitrate` \| `v2_first` \| `legacy_veto` |
| `score_fusion` | Thresholds for `--backend scores` only |

**`mix_backend` defaults (active repo):**

```json
"mix_backend": {
  "mix_mode": "legacy_arbitrate",
  "conf_reject_max": 0.10,
  "conf_arbitrate_max": 0.55,
  "v2_clear_trust_confidence": 0.80,
  "cotidian_override_min_confidence": 0.80,
  "weak_axis_veto_actions": ["walk"],
  "approx_clear_min_confidence": 0.52
}
```

**`score_fusion` (active repo — use with caution):**

```json
"score_fusion": {
  "confirm_clear_min_v2": 0.80,
  "confirm_strong_min_v2": 0.55,
  "suppress_cotidian_min": 0.80,
  "suppress_v2_below": 0.55,
  "suppress_legacy_below": 0.35
}
```

### C.3 Pocket geometry (high level)

Full spec: [`HAND_TO_BODY_SOLUCIONES.md`](HAND_TO_BODY_SOLUCIONES.md). Summary:

- Wrist–pocket distance in torso-normalised units; **near / far** zones; approach **delta**.
- Motives: **`CONTACTO_BOLSILLO`** (strong), **`APROXIMACION_OBSERVADA`**, **`ACERCAMIENTO_EJE_TORSO`** (weak, aisle-sensitive).
- Key toggles: `enable_axis_body_approach`, `allow_side_view_geometry`.
- Default radii: `r_near: 0.3`, `r_far: 0.5`, `delta_min: 0.18` (see JSON `pocket` block).

### C.4 `robbery_confirm` policy (reminder)

| Motive | Clear when |
|--------|------------|
| `CONTACTO_BOLSILLO` | conf ≥ `strong_min_confidence` (~0.48); stricter if walking |
| `APROXIMACION_OBSERVADA` | **not** walking; conf ≥ ~0.56 |
| `ACERCAMIENTO_EJE_TORSO` | quiet (`stop_frac` ≥ 0.30); conf ≥ ~0.50; **not** walking |

---

## Appendix D — Metrics: clear vs raw

### D.1 Why two booleans

| Signal | Meaning | Typical cat 6 recall (full G+H) |
|--------|---------|--------------------------------|
| `hand_to_body` | Policy clear | **~84%** |
| `hand_to_body_raw` | Pocket compatible | **~90–94%** (run on your CSV; not archived in repo) |

Historical note: under **`gate_mode: full`** policy v2, ~**598** cat-6 FNs had `raw=true` while clear was false (pocket saw gesture; policy rejected). Under **G+H**, run the script below on `gate_temporal.csv`.

### D.2 Helper script

Implemented in **`har/audit_htb_metrics.py`**: prints TP/TN/FP/FN per category for **clear** and **raw**, plus `raw−clear` on cats 6 and 14.

```bash
cd har && .venv/bin/python3 audit_htb_metrics.py /tmp/gate_temporal.csv
```

### D.3 One-liners (fixed columns)

```bash
# Cat 6 clear TP
awk -F',' 'NR>1 && $3==6 && tolower($6)=="true" {c++} END{print c}' file.csv
# Cat 6 raw TP
awk -F',' 'NR>1 && $3==6 && tolower($14)=="true" {c++} END{print c}' file.csv
# Cat 0 FP clear
awk -F',' 'NR>1 && $3==0 && tolower($6)=="true" {c++} END{print c}' file.csv
```

---

## Appendix E — Full baseline audit table (G+H frozen, clear column)

From `gate_temporal.csv` / [`HAND_TO_BODY_BASELINE_PRODUCCION.md`](HAND_TO_BODY_BASELINE_PRODUCCION.md):

| Cat | Labeled | TP | TN | FP | FN | Notes |
|-----|---------|----|----|----|----|-------|
| 0 | 4633 | 0 | 4199 | **434** | 0 | 9.4% FP |
| 1 | 758 | 0 | 322 | 289 | 0 | |
| 2 | 1954 | 0 | 394 | 1560 | 0 | long buffers |
| 3 | 2182 | 0 | 459 | 1723 | 0 | |
| 4 | 2403 | 0 | 1102 | 1301 | 0 | |
| 5 | 5602 | 0 | 5171 | **431** | 0 | 7.7% FP |
| **6** | **4192** | **3521** | 0 | 0 | **671** | **84.0%** clear recall |
| 7 | 591 | 0 | 260 | 331 | 0 | |
| 9 | 914 | 0 | 401 | 513 | 0 | |
| 13 | 80 | 0 | 21 | 59 | 0 | |
| **14** | **401** | **334** | 0 | 0 | **67** | **83.3%** |

---

## Appendix F — Rollback and experiment discipline

| Step | Action | Effect |
|------|--------|--------|
| **A** | `cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json` | Restore G+H numbers |
| **B** | `"temporal": { "enabled": false }` | ~2925 TP cat 6, ~164 FP cat 0 |
| **C** | `"gate_mode": "full"` | Policy v2 tiers + cotidian |
| **D** | `"action_policy": { "enabled": false }` | Pocket pure ~94% / 82% FP |
| **E** | `"enable_axis_body_approach": false` | Less axis recall & FP |
| **F** | `git checkout <commit> -- har/pocket_filter/` | Code rollback |

Always re-run batch + `audit_pocket_csv` after changes. Store CSV with a fixed name per experiment.

---

## Appendix G — Kanvis / motion integration (contract)

**Per chunk** (~8 s `poses_full.npy`), evaluation returns:

| Field | Orchestrator use |
|-------|-------------------|
| `motion_suppress` | If true → **do not escalate** motion (aisle FP / no pocket / weak+walk) |
| `hand_to_body` | If true → **hard confirm** robbery gesture |
| `hand_to_body_raw` | If true and not clear → **SAM / human review** |
| `hand_to_body_confidence` | Tie-break / logging |
| `motivo` | Forensics (`CONTACTO_*`, `ACERCAMIENTO_EJE_*`, fusion rule) |
| `intervalo_s` | Peak window (temporal mode) for debugging |

**Suggested runtime (category-free):**

```text
if not motion_candidate: skip pocket entirely  # optional compute save
else: run evaluate_pocket_with_policy (or scores/mix if adopted)

if motion_suppress:     return DISCARD
elif hand_to_body:      return CONFIRM
elif hand_to_body_raw:  return REVIEW_SAM
else:                   return DISCARD
```

Batch `--backend` is for **evaluation**; production embedding typically calls the same Python entrypoints as `har2_test` / `evaluate_pocket_with_policy`.

---

## Appendix H — Pocket v2 motives and confidence (for fusion research)

**`hand_to_body_confidence`** (`confidence.py`): heuristic blend of motive base score, pair geometry, `valid_ratio`. Not calibrated probability.

| Motive | Typical base |
|--------|----------------|
| `CONTACTO_BOLSILLO` | ~0.92 |
| `APROXIMACION_OBSERVADA` | ~0.82 |
| `ACERCAMIENTO_EJE_TORSO` | ~0.42 |

**Legacy confidence** (`legacy_hand_to_body_confidence`): duration of legacy segment + far/close hip–axis geometry (0–1).

**Cotidian confidence** (`primary_action_confidence` / mix cotidian peak): fraction of clip time in dominant `walk` / `stop` / etc.

Fusion research should treat these as **comparable scores only after calibration**, or use **motive-conditional rules** (axis vs contact) as implemented in mix/scores.

---

## Appendix I — Tests and quality

```bash
cd har && .venv/bin/python3 -m pytest tests/ -q
```

Relevant suites: `test_pocket_filter.py`, `test_action_policy.py`, `test_temporal_eval.py`, `test_mix_backend.py`, `test_score_fusion.py`, `test_stratified_sample.py`.

---

## Document map (what this file does *not* duplicate)

| Need | Read |
|------|------|
| Long design / handoff spec (Spanish+options A–J) | [`HAND_TO_BODY_SOLUCIONES.md`](HAND_TO_BODY_SOLUCIONES.md) |
| Frozen metrics + Spanish rollback commands | [`HAND_TO_BODY_BASELINE_PRODUCCION.md`](HAND_TO_BODY_BASELINE_PRODUCCION.md) |
| Experiment diary | [`HAND_TO_BODY_PROGRESO.md`](HAND_TO_BODY_PROGRESO.md) |
| **This file** | English status + appendices A–I for external engineers |