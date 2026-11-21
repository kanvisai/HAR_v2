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

## Fase siguiente (cuando digas “seguimos”)

1. **P0-4 ligero:** columna `session` en CSV (parse path).
2. **P0-3:** simulador de árbol corregido en script (sin Orchestrator).
3. **P0-5:** evaluación en `interval_s` fijo (tests) hasta tener Motion.

Cada fase tendrá su sección en este runbook.