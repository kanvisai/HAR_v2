# Fase 4 — notas cinemática (muestra 10 % / 5 %, seed 42)

**Última actualización:** 2026-11-22

## Números de referencia (qué estamos midiendo)

| Escala | Valor | Nota |
|--------|-------|------|
| Dataset audit | `data_result` (~23k+ clips full) | Etiqueta por **carpeta** (cat 6/14 = robo, 0/5 = pasillo…) |
| Iteración rápida | `--sample-frac 0.05` o `0.10`, **seed 42** | ~1.2k / ~2.4k filas; **comparar tasas**, no conteos absolutos entre fracciones |
| Baseline congelado | `BASELINE_PRODUCCION_H.json` | **G+H** = pocket **v2** + ventanas 2,5 s + `robbery_confirm` + doctrina P0-3 en audit |
| Backend **mix** | `mix` en batch | v2+legacy fusionado; en muestras previas ≈ **v2** en recall/FP — **no** es el baseline de producción documentado |
| Legacy solo | `legacy` | ~3 % FP cat 0, ~37 % recall cat 6 — descartado como confirmador |

### Baseline G+H+P0-3 (muestra **10 %**, seed 42, barrido histórico)

| Métrica | Cat 6 (robo) | Cat 0 (pasillo) | Cat 14 |
|---------|----------------|-----------------|--------|
| **clear** | **352/419 (~84 %)** | TN alta, **FP=58** (~12,5 % de n) | ~35–36/40 |
| **raw** | ~397/419 | ~83 % raw | ~38/40 |
| FP agregado negativas (clear) | — | ~36,8 % | — |

**Contrato runtime (objetivo):** `motion_suppress` → `hand_to_body` (clear) → `hand_to_body_raw` (REVIEW). Clear **no** debe acercarse al v2 sin policy (~82 % FP cat 0).

---

**Fecha:** 2026-11-21  
**CSV:** `fase4_kin_10pct.csv`  
**Clear:** sin cambio vs G+H+P0-3 (cat 6 **352/419**, cat 0 FP **58**).

## Separación global FN vs FP (raw∧¬clear)

| Feature | Mediana FN cat 6 (pos) | Mediana FP cat 0 (neg) | ¿Útil? |
|---------|------------------------|------------------------|--------|
| `kin_wrist_radial_vel_min` | -1.85 | **-2.16** | No (FP más “rápido” hacia bolsillo) |
| `kin_approach_monotone_frac` | 0.50 | 0.50 | No |
| `kin_hip_speed_norm` | **0.62** | **0.76** | Débil (pasillo marcha más) |
| `kin_elbow_angle_deg` | 155 | 156 | No |
| `kin_wrist_axis_dist_min` | **0.15** | 0.18 | Muy débil |

**Conclusión:** con estas 5 magnitudes **no** basta un umbral global para desambiguar REVIEW. Coherente con que el pocket ya consumió distancia/Δ y el residual sea sobre todo **motivo + marcha** (`robbery_confirm`).

## Dirección ingeniería (sin red)

1. **Condicionar por `motivo`** (eje vs contacto) + `kin_hip_speed_norm` / `ctx_walk_frac`.
2. **P1-1b (implementado):** `kin_near_dwell_s`, `kin_swing_peaks`, `kin_approach_depth` + política `kin_axis_dwell_enabled` (eje+marcha). Config: `pocket_filter_config.PRUEBA_KIN_AXIS_DWELL.json`.
3. **P1-1 resto:** retirada post-contacto, antifase bilateral (pendiente).
4. **No** bajar umbral contacto global (Fase 2-c).
5. **Ranker logístico:** opcional cola SAM; no sustituye cinemática ni Motion.

## Comando seguimiento

```bash
.venv/bin/python3 analyze_kinematics_separation.py /tmp/fase4_kin_10pct.csv
```

## Experimento P1-1b V1 (`PRUEBA_KIN_AXIS_DWELL`, 10 % seed 42)

| Métrica | Baseline G+H+P0-3 | V1 dwell | Δ |
|---------|-------------------|----------|---|
| Cat 6 clear | 352 | **359** | +7 |
| Cat 0 FP | **58** | **105** | **+47** |

**No adoptar V1.** Causa: excepción eje+marcha demasiado amplia; FP cat 0 = 67× eje.  
**Bug corregido:** `CSV_FIELDS` en batch no incluía `kin_near_dwell_s` / `kin_swing_peaks` → re-batch tras pull.  
**Siguiente prueba:** `PRUEBA_KIN_AXIS_DWELL_V2.json` (dwell 0,45, swing ≤2, hip ≤0,65) + `diagnose_axis_kin_rule.py`.

## Experimento P2-d (`PRUEBA_KIN_CONTACT_QUIET`, 10 % seed 42)

| Métrica | Baseline G+H+P0-3 | Kin gate | Δ |
|---------|-------------------|----------|---|
| Cat 6 TP | 352 | **356** | **+4** |
| Cat 6 FN | 67 | **63** | −4 |
| Cat 0 FP | 58 | **68** | **+10** |
| Cat 14 TP | 35 | **36** | +1 |

Mejor trade-off que Fase 2-c global (0,55 contacto: +10 TP cat 6, **+49** FP cat 0). **No congelar** sin afinar `kin_contact_max_hip_speed` (probar 0,45) o modo observación.

## Experimento P1-1b V2 (`PRUEBA_KIN_AXIS_DWELL_V2`, muestra **5 %**, seed 42)

**CSV:** p.ej. `/tmp/fase5_axis_dwell_v2_5pct.csv` — **1191** filas, `kin_near_dwell_s` presente en export.

| Métrica | V2 @ 5 % | Baseline ~10 % (referencia) | V1 dwell ~10 % |
|---------|----------|-----------------------------|----------------|
| Cat 6 clear | **175/210 (83,3 %)** | 352/419 (~84 %) | 359/419 (85,7 %) |
| Cat 6 FN | 35 | ~60–67 | 60 |
| Cat 0 FP (clear) | **25/232 (10,8 %)** | 58/463 (~12,5 %) | 105/463 (22,7 %) |
| Embudo cat 6 CONFIRM | 83,3 % | 85,7 % | 85,7 % |
| cat0 eje+marcha clear (diagnose) | **4** | — | ~53 |

**Lectura:** V2 **recupera pasillo** respecto a V1; en tasa cat 0 clear FP va **por debajo** del baseline 10 % en esta corrida (muestra distinta → confirmar al 10 %). Recall cat 6 **~1 pp** por debajo del baseline — trade aceptable **solo** si el 10 % pareado lo confirma.

**Diagnose:** `would_pass_rule` 20 clips (2 cat6 clear, 4 cat0_fp eje+marcha) — regla kin estricta y acotada.

**Decisión tras 10 % (2026-11-22):** ver tabla siguiente; **no sustituye baseline** sin decisión de producto (recall vs FP).

## Experimento P1-1b V2 — confirmación **10 %**, seed 42

**CSV:** `/tmp/fase5_axis_dwell_v2_10pct.csv` — **2373** filas, `PRUEBA_KIN_AXIS_DWELL_V2.json`.

| Métrica | Baseline G+H @ 10 % | V2 kin @ 10 % | Δ |
|---------|---------------------|---------------|---|
| Cat 6 clear | **352/419 (84,0 %)** | **355/419 (84,7 %)** | **+3 TP**, FN 67→**64** |
| Cat 0 FP clear | **58/463 (12,5 %)** | **66/463 (14,3 %)** | **+8 FP** |
| Cat 14 clear | 35–36/40 | **35/40** | ~igual (FN 5) |

**Veredicto:** V2 **recupera ~3 robos** en clear vs baseline con **+8 FP** solo en cat 0 (+1,8 pp tasa). Mucho mejor que V1 (+47 FP). **No domina** baseline: es un **challenger** si producto acepta +8 pasillo por +3 robo en confirmación automática; si no → quedarse en `BASELINE_PRODUCCION_H.json`.

**Embudo P0-3 (mismo CSV, `fase0_pocket_audit`):**

- Global: CONFIRM **47,2 %** | REVIEW_RAW **41,0 %** | KEEP_INDETERMINADO **11,4 %** | DISCARD_SUPPRESS **0,4 %**
- Cat 6: CONFIRM **84,7 %** | REVIEW_RAW **10,0 %** (42) | KEEP_INDETERMINADO **4,8 %** (20)
- Cat 0: CONFIRM **14,3 %** (66 FP) | REVIEW_RAW **68,9 %**
- FN cat 6 **64**: **42/64 (65,6 %)** con `raw=true` (recuperables en REVIEW, no clear); motivos FN: eje **29**, evidencia parcial **13**, contacto **8**; **20/64** pocket INDETERMINADO (árbol legado los **DESCARTARÍA** — argumento P0-3 Kanvis)
- FP cat 0 clear: eje **27**, contacto **20**, aprox **19**

```bash
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json   # producción / comparaciones
```

---

## Estado frente al roadmap (`PROPUESTA_SOLUCIONES_HAND_TO_BODY.md` §8)

| ID | Tema | Estado |
|----|------|--------|
| **P0-1** | Fase 0, embudo, auditorías | **Hecho** (`fase0_pocket_audit`, runbook) |
| **P0-2** | Preguntas producto (coste, C2, suppress) | **Pendiente** equipo |
| **P0-3** | Doctrina INDETERMINADO / REVIEW_RAW | **En código**; **Kanvis Orchestrator** sin cablear |
| **P0-4** | Motion en CSV, split sesión | **Parcial** (`session_id`); falta score/intervalo C2 masivo |
| **P0-5** | Eval pocket en intervalo Motion | **Cableado batch** (`--motion-intervals`); falta **JSONL** C2 |
| **P1-1 / P1-1b** | Features kin + excepción eje | **Implementado**; V2 **candidato** tras 10 % |
| **P2-d** | Kin gate contacto+marcha | Medido; **no** baseline |
| **P1-2** | Fusión GBM/logística | Probado ranker; **no** palanca principal acordada |
| **P1-1 resto** | Antifase, retirada, periodicidad | **Pendiente** |
| **P1-3** | Validez / episodios | Pendiente auditoría código |
| **Hito 4–5** | Observación tienda + canario | Futuro |

## Siguiente paso operativo (P0-4 / P0-5 — en curso)

1. **Pull** → batch 10 % baseline con `eval_interval_t0/t1` en CSV.
2. **P0-4:** `split_by_session.py` sobre el CSV → holdout por sesión.
3. **P0-5 oracle:** `motion_peaks_to_jsonl.py from-batch` + batch `--motion-intervals` → `compare_p05_eval.py` + `fase0_pocket_audit`.
4. **P0-5 real:** cuando C2 entregue CSV → `from-csv` → mismo pipeline.
5. Paralelo **P0-3** Kanvis y **P0-2** costes producto.

Ver [`P0_5_MOTION_INTERVAL.md`](P0_5_MOTION_INTERVAL.md) y [`P0_4_SESSION_SPLIT.md`](P0_4_SESSION_SPLIT.md).