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

## Export desde C2 / Kanvis

Pendiente de equipo: volcar picos Motion al JSONL con la misma convención de rutas que `data_result`. Mientras no exista el fichero, el batch sigue en modo **temporal** (OR 2,5 s).

## Métricas

Comparación pareada baseline G+H vs mismo CSV con sidecar: recall clear 6/14, FP 0/5, % `INDETERMINADO`, cobertura válida en ventana.