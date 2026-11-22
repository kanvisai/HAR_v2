# P0-4 — split por sesión (sin fuga clip↔clip)

Etiquetas del audit van por carpeta, pero los clips de una **misma sesión** están correlacionados. Cualquier ajuste de umbrales o fusión debe usar **holdout de sesiones**, no sorteo de clips.

## Script

```bash
cd har
.venv/bin/python3 split_by_session.py /tmp/fase0_0.10pct.csv -o /tmp/session_split \
  --holdout-frac 0.2 --seed 42 --stratify-category
```

Salida:

- `session_split_manifest.json` — conteos y lista holdout
- `sessions_train.tsv` / `sessions_holdout.tsv` — pares `category<TAB>session_id`

## Uso

- Métricas en **train**: filtrar batch CSV por sesiones en `sessions_train.tsv` (pandas o script futuro).
- **Nunca** afinar sobre `sessions_holdout.tsv`; solo reporte final.

Relacionado: columna `session_id` ya viene en `batch_data_result.py`.