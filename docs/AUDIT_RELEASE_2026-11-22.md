# Auditoría pre-integración C3 — HAR pocket v2

**Fecha:** 2026-11-22  
**Estado:** listo para integrar en Kanvis-C3 (npy → señales). Motion C2 y árbol C3 fuera de alcance de este repo.

## Checks automáticos (repo local)

| Check | Resultado |
|-------|-----------|
| `pytest tests/` | **42 passed** |
| `pocket_filter_config.json` | Presente; **en workspace local difiere** de `BASELINE_PRODUCCION_H` (extra `walk_gate_*`, bloques `score_fusion` / `mix_backend`). **En C3/prod:** copiar solo el baseline H |
| `hand_to_body_backend` | **v2** |
| `action_policy.gate_mode` | **robbery_confirm** |
| `temporal.enabled` | **true** (ventana 2,5 s, paso 1,25 s) |

Comando repro:

```bash
cd har && .venv/bin/python3 -m pytest tests/ -q
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
```

## Barrido offline full (servidor)

| Artefacto | Valor |
|-----------|--------|
| Clips | **23 738** (0 errores) |
| CSV | `/tmp/baseline_g_h_FULL.csv` |
| Informe | `/tmp/baseline_g_h_FULL_fase0.txt` |
| Cohortes | `/tmp/cohorts_full/` |

### Métricas referencia (audit carpeta; cat 6/14 = robo)

| Métrica | Valor |
|---------|--------|
| Cat 6 **clear** | **3521 / 4192 (84,0 %)** |
| Cat 6 **raw** | **3959 / 4192 (94,4 %)** |
| Cat 0 FP **clear** | **434 / 4633 (9,4 %)** |
| Cat 5 FP **clear** | **431 / 5602 (7,7 %)** |
| `motion_suppress` global | **59** (~0,2 %) |

### Embudo doctrina P0-3 (sobre mismo CSV)

| Rama | Global |
|------|--------|
| CONFIRM | 10 656 |
| REVIEW_RAW | 9 991 |
| KEEP_INDETERMINADO | 3 032 |
| DISCARD_SUPPRESS | 59 |

## No usar en integración C3

| Componente | Motivo |
|------------|--------|
| `detect_json.py` → `evaluate_pocket` sin política | No aplica temporal ni `robbery_confirm` |
| `backend mix` / `scores` | Investigación; no baseline |
| `--motion-intervals` sin C2 | Oracle descartado (pierde recall) |
| `PRUEBA_KIN_*` configs | Experimentos rechazados o challenger |

## API recomendada

- **Python:** `pocket_infer.infer_from_poses_path` o `evaluate_pocket_with_policy`
- **CLI prueba:** `har2_test.py poses_full.npy --json -o eval_pocket_output.json`

Ver [`C3_INTEGRATION_HAND_TO_BODY.md`](C3_INTEGRATION_HAND_TO_BODY.md).