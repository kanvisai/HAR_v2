# P0-5 — evaluar pocket en intervalo Motion

**Estado:** cableado en `batch_data_result.py` (sin sidecar → fallback ventanas temporales igual que hoy).

## Formato sidecar (JSONL)

Una línea por clip:

```json
{"path": "6/<session>/user_xyz/poses_full.npy", "t0": 1.2, "t1": 3.8, "score": 0.91}
```

`path` puede ser sufijo `cat/session/user_*/poses_full.npy` o solo `poses_full.npy` si es único en el catálogo.

## Batch

```bash
python batch_data_result.py "$DATA" --backend v2 \
  --motion-intervals /ruta/motion_peaks.jsonl \
  --motion-min-interval-s 0.35 \
  -o /tmp/p05.csv --workers 50
```

Columnas nuevas: `pocket_eval_mode` (`temporal` | `motion_interval`), `interval_t0`, `interval_t1`, `motion_peak_score`.

## Ablation oracle (sin C2 — ventana ganadora temporal)

Sirve para medir **techo/beneficio** de evaluar en un intervalo fijo antes de tener Motion real.

```bash
cd har
DATA=/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result

# 1) Batch temporal (baseline G+H) — debe incluir eval_interval_t0/t1
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
.venv/bin/python3 batch_data_result.py "$DATA" --backend v2 --sample-frac 0.10 --seed 42 \
  -o /tmp/p05_temporal_10pct.csv --workers 50

# 2) JSONL desde ventanas ganadoras (proxy, NO es C2)
.venv/bin/python3 motion_peaks_to_jsonl.py from-batch /tmp/p05_temporal_10pct.csv \
  -o /tmp/motion_oracle_10pct.jsonl

# 3) Re-eval solo con intervalo (sin OR temporal)
.venv/bin/python3 batch_data_result.py "$DATA" --backend v2 --sample-frac 0.10 --seed 42 \
  --motion-intervals /tmp/motion_oracle_10pct.jsonl \
  -o /tmp/p05_interval_10pct.csv --workers 50

# 4) Comparar modos en el CSV interval (debe haber motion_interval)
.venv/bin/python3 compare_p05_eval.py /tmp/p05_interval_10pct.csv
.venv/bin/python3 fase0_pocket_audit.py /tmp/p05_interval_10pct.csv
```

## Export desde C2 / Kanvis (`motion_peaks_to_jsonl.py from-csv`)

CSV C2 con columnas configurables:

```bash
.venv/bin/python3 motion_peaks_to_jsonl.py from-csv /ruta/c2_peaks.csv -o /tmp/motion_c2.jsonl \
  --path-col poses_path --t0-col peak_t0 --t1-col peak_t1 --score-col p41
```

Misma convención de `path` que en `data_result`. Sin C2, el batch sigue en **temporal**.

## Métricas

Comparación pareada baseline G+H vs mismo CSV con sidecar: recall clear 6/14, FP 0/5, % `INDETERMINADO`, cobertura válida en ventana.

## Smoke test (2026-11-22)

Batch **5 %** seed 42 **sin** `--motion-intervals`, config baseline → `/tmp/p05_smoke.csv` (**1191** filas, ~2 min).

- Cat 6 clear **174/210**; cat 0 FP **22/232** (~9,5 %) — coherente con baseline en muestra reducida.
- Comprobar que `pocket_eval_mode` = `temporal` en todas las filas hasta exista JSONL Motion.