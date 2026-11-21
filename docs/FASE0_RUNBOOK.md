# Fase 0 — runbook (P0-1)

**Objetivo:** decidir con datos si el siguiente paso es P0-3 (árbol), P0-5 (intervalo Motion) o P1-2 (fusión).  
**Sin cambios de código de filtro** en esta fase (solo batch + auditorías).

Referencia: [`PROPUESTA_SOLUCIONES_HAND_TO_BODY.md`](PROPUESTA_SOLUCIONES_HAND_TO_BODY.md) §5 P0-1.

---

## Paso 0 — Entorno y config (auditar una vez)

En el servidor con el dataset:

```bash
cd ~/kanvis/piloto_v1/HAR_v2/har   # ajusta ruta
git pull
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
.venv/bin/python3 -m pytest tests/ -q
```

**Pásame:** salida completa de `pytest` (últimas ~20 líneas) y `head -5 pocket_filter_config.json` (temporal + gate_mode).

---

## Paso 1 — Batch muestra 10 % (rápido, ~4–8 min)

Misma **seed** siempre para comparar corridas:

```bash
DATA=/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result
FRAC=0.10   # o 0.05 para ~mitad de tiempo

.venv/bin/python3 batch_data_result.py "$DATA" \
  --backend v2 \
  --sample-frac "$FRAC" \
  --seed 42 \
  --workers 50 \
  -o /tmp/fase0_${FRAC/./pct}.csv

.venv/bin/python3 audit_pocket_csv.py /tmp/fase0_${FRAC/./pct}.csv
```

**Pásame:**

1. Resumen que imprime `batch_data_result` al inicio (muestra estratificada por cat).
2. Tabla TP/TN/FP/FN de `audit_pocket_csv` (completa o al menos cats 0, 5, 6, 14).
3. Tiempo total del batch (última línea).

**Criterio (muestra):** orden de magnitud coherente con full (~84 % recall cat 6 clear, ~9 % FP cat 0). Si diverge mucho, sube a 0.10 o repite seed 42.

---

## Paso 2 — Métricas clear vs raw

```bash
.venv/bin/python3 audit_htb_metrics.py /tmp/fase0_10pct.csv
```

**Pásame:** las dos tablas CLEAR y RAW + bloque `raw−clear` en cat 6 y 14.

---

## Paso 3 — Informe Fase 0 (embudo + ternario)

```bash
.venv/bin/python3 fase0_pocket_audit.py /tmp/fase0_10pct.csv \
  -o /tmp/fase0_10pct_report.txt
```

**Pásame:** archivo `/tmp/fase0_10pct_report.txt` entero (o pega el contenido en el chat).

---

## Paso 4 — Decisión (lo hacemos juntos al ver el informe)

| Señal en informe | Siguiente implementación |
|------------------|---------------------------|
| `raw` cat 0 ≈ 70–80 % y REVIEW alto | Priorizar P1-2 / triage REVIEW antes de bajar umbrales |
| Muchos FN cat 6 con `resultado=INDETERMINADO` | P0-3 árbol + `htb_tier` review en `robbery_confirm` |
| `DISCARD_ELSE` dominado por INDETERMINADO en cat 6 | Igual P0-3 |
| FP agregado ≫ cat 0 solo | Revisar etiquetas cat 2–4 (§10 propuesta) antes de fusionar |
| FN cat 6 con `raw=false` mayoría | Techo geométrico; intervalo Motion (P0-5) o vídeo (P2) |

---

## Paso opcional — Full sweep (solo si 10 % apunta claro)

```bash
.venv/bin/python3 batch_data_result.py "$DATA" --backend v2 \
  -o /tmp/gate_temporal.csv --workers 50
.venv/bin/python3 fase0_pocket_audit.py /tmp/gate_temporal.csv -o /tmp/fase0_full_report.txt
```

Horas de CPU; hacerlo **después** de validar scripts en 10 %.

---

## Fase 0 — Veredicto (2026-11-21, muestra 10 % seed 42)

| Hallazgo | Evidencia |
|----------|-----------|
| Baseline coherente | Cat 6 clear **84 %**, cat 0 FP **12.5 %** (escala full ~9 %) |
| **H3 confirmado** | Cat 0 **raw 83 %** → **71 %** iría a REVIEW; no usar raw como SAM masivo |
| **DISCARD_ELSE = 0** | Todo lo no-clear va a SUPPRESS o REVIEW; los **277 GATE_NO_POCKET** mezclan INDETERMINADO |
| Recall recuperable | **45/67** FN cat 6 tienen raw=true (política); **20** INDETERMINADO en SUPPRESS |
| FP no es solo cat 0 | Agregado negativas **36.8 %** clear; sin 0/5 **67.5 %** (cat 2–4 dominan) |
| INCOMPATIBLE inútil hoy | **8** clips; `filter_fp_safe` ~0 % — no sirve como palanca FP |

**Siguiente código:** **P0-3** (`action_policy`: INDETERMINADO no `motion_suppress`; `htb_tier=review` si raw∧¬clear).

---

## Fase 1 — P0-3 (auditorías E–G)

Tras `git pull` con el cambio de política:

### Paso E — Re-batch 10 % (mismos flags que Fase 0)

```bash
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
.venv/bin/python3 batch_data_result.py "$DATA" --backend v2 \
  --sample-frac 0.10 --seed 42 --workers 50 -o /tmp/fase1_p03_10pct.csv
```

### Paso F — Mismas auditorías

```bash
.venv/bin/python3 audit_pocket_csv.py /tmp/fase1_p03_10pct.csv
.venv/bin/python3 audit_htb_metrics.py /tmp/fase1_p03_10pct.csv
.venv/bin/python3 fase0_pocket_audit.py /tmp/fase1_p03_10pct.csv -o /tmp/fase1_p03_report.txt
.venv/bin/python3 simulate_orchestrator_policy.py /tmp/fase1_p03_10pct.csv
```

**Validado 2026-11-21 (tu corrida):**

| Check | Fase 0 | Fase 1 P0-3 |
|-------|--------|-------------|
| Cat 6 clear | 352/419 | **igual** |
| Cat 0 FP clear | 58 | **igual** |
| `motion_suppress` | 278 (11.7 %) | **9 (0.4 %)** |
| `htb_tier=review` | 0 | **1003** |
| Embudo doctrina cat 6 | 22 SUPPRESS | **20 KEEP_INDETERMINADO** + 45 REVIEW + 352 CONFIRM |

El informe mostraba `DISCARD_ELSE` porque el árbol **legado** sigue marcando INDETERMINADO como descarte; en Kanvis usar **doctrina P0-3** (`KEEP_INDETERMINADO`). Tras `git pull`, `fase0_pocket_audit` §5 ya imprime doctrina.

### Paso G — Comparación rápida (opcional)

```bash
diff -u <(grep -E 'DISCARD|KEEP|REVIEW|CONFIRM' /tmp/fase0_10pct_report.txt) \
        <(grep -E 'DISCARD|KEEP|REVIEW|CONFIRM' /tmp/fase1_p03_report.txt) || true
```

**Pásame:** Paso F completo (audit + htb_tier en informe + salida `simulate_orchestrator_policy`).

---

## Fase 1 — CERRADA ✓

Embudo doctrina validado en `fase1_p03_report.txt`. Cablear Kanvis con **KEEP_INDETERMINADO**, no `else:DISCARD`.

---

## Fase 2 — cohortes REVIEW + `session_id` (P0-4 / P1-2 prep)

Tras `git pull`:

```bash
.venv/bin/python3 export_review_cohort.py /tmp/fase1_p03_10pct.csv -o /tmp/cohorts/
```

**Pásame:** bloque `fn_cat6_raw_review` (n + motivos + walk_hi).

Re-batch opcional (añade `session_id` al CSV; no cambia métricas):

```bash
.venv/bin/python3 batch_data_result.py "$DATA" --backend v2 \
  --sample-frac 0.10 --seed 42 --workers 50 -o /tmp/fase2_10pct.csv
head -1 /tmp/fase2_10pct.csv   # debe incluir session_id
```

### Fase 2-b — separación cola REVIEW (P1-6 prep)

```bash
git pull
.venv/bin/python3 export_review_cohort.py /tmp/fase2_10pct.csv -o /tmp/cohorts2/
.venv/bin/python3 analyze_review_separation.py /tmp/cohorts2/all_cohorts.csv
```

**Pásame:** tabla `top_K | pos_in_top | recall_pos | prec_pos`.

Interpretación de cohortes (tu corrida): **45 FN** → **30 eje**, **35/45 marcha**; no relajar eje en marcha sin cola.

### Fase 2-c — experimento contacto en marcha (opcional)

```bash
cp pocket_filter_config.PRUEBA_CONTACT_WALK_055.json pocket_filter_config.json
.venv/bin/python3 batch_data_result.py "$DATA" --backend v2 \
  --sample-frac 0.10 --seed 42 -o /tmp/fase2c_contact055.csv --workers 50
.venv/bin/python3 audit_pocket_csv.py /tmp/fase2c_contact055.csv | sed -n '/cat  6/,/cat  7/p'
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
```

Comparar cat 6 TP/FN y cat 0 FP vs `fase2_10pct.csv`.

**Resultado medido 2026-11-21 (10 %):** cat 6 **352→362** (+10 FN 67→57); cat 0 FP **58→107** (+49); cat 5 FP **45→98**. **NO adoptar** como baseline; contacto en marcha sube confirmaciones en pasillo.

### Fase 2-b — resultado heurística

`review_priority` actual: recall pos **4 % @50**, **71 % @500**, precisión **~8 %** global. Pasillo comparte motivos/confianza con FN. Siguiente: `tune_review_triage.py` y/o score C2; no SAM top-50 con reglas solas.