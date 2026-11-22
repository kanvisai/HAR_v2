# Integración Kanvis C3 — Hand-to-body (HAR_v2)

Documento para el agente que integre **HAR** en **Kanvis-C3**.  
**Alcance HAR:** `poses_full.npy` (y sidecars opcionales) → señales JSON/Python.  
**Fuera de alcance:** generación del npy (otro servicio), modelo de robo (C2), decisión final de negocio (C3).

---

## 1. Cuándo ejecutar HAR

C3 debe llamar a HAR **solo si** el pipeline ya marcó el chunk como **candidato a robo** (probabilidad Motion u otra regla). HAR **no** clasifica categoría de carpeta ni sabe si es pasillo o hurto.

Entrada típica por chunk (~8 s):

```
<data_root>/<session>/user_*/poses_full.npy
```

Sidecars **opcionales** (misma carpeta):

| Archivo | Uso |
|---------|-----|
| `valid_mask.npy` | Máscara por frame / KP |
| `pose_frame_timestamps.npy` | Tiempo real por frame |
| `video` vía convención pose | Aspect ratio (si existe) |

Formato npy: `(T, 8, 2)` o `(T, 8, 3)` con conf en canal 3 — ver `pocket_filter/io.py`.

---

## 2. Configuración congelada (producción)

Copiar en despliegue:

```bash
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
```

Debe contener:

- `hand_to_body_backend: "v2"`
- `action_policy.enabled: true`, `gate_mode: "robbery_confirm"`
- `temporal.enabled: true`, `window_s: 2.5`, `step_s: 1.25`

**No** usar `mix`, `scores`, ni configs `PRUEBA_*`.

---

## 3. API Python (recomendada)

Añadir al `PYTHONPATH` la carpeta **`har/`** del repo HAR_v2 (o empaquetar como submódulo).

```python
from pathlib import Path
from pocket_infer import infer_from_poses_path, infer_to_dict

poses = Path("/ruta/.../user_0/poses_full.npy")
result = infer_from_poses_path(poses)

# Campos clave para C3
assert result.hand_to_body is not None
clear = bool(result.hand_to_body)          # confirmación fuerte
raw = bool(result.hand_to_body_raw)        # gesto pocket (puede no confirmar)
suppress = bool(result.motion_suppress)     # frenar candidato motion

payload = infer_to_dict(poses)  # incluye orchestrator_branch + pocket_result_to_dict
```

Implementación: `har/pocket_infer.py` → `evaluate_pocket_with_policy` (temporal + política).

### Alternativa CLI (debug)

```bash
cd har
.venv/bin/python3 har2_test.py /path/poses_full.npy --json -o eval_pocket_output.json
```

`har2_test.py` con `--backend v2` (default) es equivalente a producción.

### ❌ No usar para producción

- `detect_json.py` con rama v2: llama `evaluate_pocket` **sin** política ni ventanas temporales.
- `batch_data_result.py` en hot path (solo auditoría offline).

---

## 4. Contrato de salida (campos obligatorios)

Objeto `PocketResult` / JSON (`pocket_result_to_dict` + extras en `har2_test` / `infer_to_dict`):

| Campo | Tipo | Significado |
|-------|------|-------------|
| **`hand_to_body`** | bool | **Clear / confirmar** gesto bolsillo (política aplicada) |
| **`hand_to_body_raw`** | bool | Geometría pocket positiva **sin** último veto de política |
| **`motion_suppress`** | bool | Sugerencia de **no** escalar el candidato Motion |
| **`hand_to_body_confidence`** | float 0–1 | Heurística (no probabilidad calibrada) |
| **`resultado`** | string | `COMPATIBLE` \| `INDETERMINADO` \| `INCOMPATIBLE` |
| **`motivo`** | string | p.ej. `CONTACTO_BOLSILLO`, `ACERCAMIENTO_EJE_TORSO`, `EVIDENCIA_PARCIAL` |
| **`context_veto`** | string \| null | Código si política vetó clear (`GATE_WALK_NO_CONFIRM`, …) |
| **`htb_tier`** | string | `alert` \| `review` \| `none` (diagnóstico; en `robbery_confirm` clear ⇒ `alert`) |
| **`filter_fp_safe`** | bool | INCOMPATIBLE fuerte (casi no usado en audit) |
| **`intervalo_s`** | (t0, t1) | Ventana ganadora temporal (segundos) |
| **`ctx_walk_frac`**, **`ctx_stop_frac`** | float | Contexto marcha/parado en evaluación |

Kin opcional (`kin_*` en batch): no requerido en C3 v1.

---

## 5. Árbol Orchestrator (doctrina P0-3)

C3 debe implementar este orden **sin** etiqueta de categoría:

```text
function branch(htb: PocketResult) -> str:
    if htb.motion_suppress:
        return "DISCARD_SUPPRESS"
    if htb.hand_to_body:
        return "CONFIRM"
    if htb.hand_to_body_raw:
        return "REVIEW_RAW"
    if htb.resultado == "INDETERMINADO":
        return "KEEP_INDETERMINADO"   # nunca DISCARD por ser indeterminado
    return "DISCARD_OTHER"            # INCOMPATIBLE sin raw; revisar política
```

Equivalente en Python: `pocket_infer.orchestrator_branch(result)`.

### Acciones sugeridas en C3 (tú defines negocio)

| Rama | Acción típica |
|------|----------------|
| **CONFIRM** | Alerta / confirmación hurto |
| **REVIEW_RAW** | Cola SAM o humano (priorizar si score Motion alto) |
| **KEEP_INDETERMINADO** | Conservar evento, baja prioridad, no tirar |
| **DISCARD_SUPPRESS** | No escalar |
| **DISCARD_OTHER** | Descartar con log |

**Importante:** No mapear `else → DISCARD` sobre INDETERMINADO (árbol legado). En full sweep **3032** clips son KEEP vs DISCARD_ELSE.

---

## 6. Métricas de referencia (offline, sin categoría en runtime)

Barrido completo `data_result` con baseline G+H (ver `AUDIT_RELEASE_2026-11-22.md`):

- Tras candidato motion, si solo usas **`hand_to_body`**: ~**84 %** recall robo (etiqueta carpeta) vs ~**9,4 %** clear en pasillo etiq. cat 0.
- Si usas **`hand_to_body_raw`** en cola REVIEW: ~**94 %** gesto en cat 6 (audit).
- **`motion_suppress`**: muy raro (~59 / 23738).

C3 puede usar `hand_to_body_confidence` y `motivo` para **ordenar** REVIEW, no como único umbral.

---

## 7. Despliegue en repo C3

1. **Vendor / submodule:** clonar HAR_v2 (ruta `har/`, `pocket_filter_config.BASELINE_PRODUCCION_H.json`).
2. **Dependencias:** mismas que HAR (`numpy`, entorno con `har/requirements` o lock del piloto).
3. **Una llamada por chunk:** `infer_from_poses_path(npy_path)`; cachear config al arranque (`load_pocket_config` una vez).
4. **Latencia:** temporal ~5× clip entero; en piloto aceptable; no hay GPU.
5. **Logging:** guardar `motivo`, `context_veto`, `orchestrator_branch`, versión config (`pocket-v2` en JSON).

Ejemplo mínimo servicio:

```python
def on_motion_candidate(poses_npy: Path) -> dict:
    from pocket_infer import infer_to_dict
    return infer_to_dict(poses_npy)
```

---

## 8. Prueba de humo en C3

```bash
export HAR_ROOT=/path/HAR_v2/har
cd "$HAR_ROOT"
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
.venv/bin/python3 har2_test.py "$POSES_FULL" --json
```

Comparar salida con fila del batch para el mismo path (opcional).

---

## 9. Rollback

```bash
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
```

Desactivar temporal solo si producto exige menos FP a costa de recall (`temporal.enabled: false`).

---

## 10. Referencias en HAR_v2

| Doc | Contenido |
|-----|-----------|
| [`AUDIT_RELEASE_2026-11-22.md`](AUDIT_RELEASE_2026-11-22.md) | Auditoría y números full |
| [`HAND_TO_BODY_BASELINE_PRODUCCION.md`](HAND_TO_BODY_BASELINE_PRODUCCION.md) | Baseline y rollback |
| [`AUDIT_CATEGORY_SEMANTICS.md`](AUDIT_CATEGORY_SEMANTICS.md) | Etiquetas carpeta vs runtime |
| [`PROPUESTA_SOLUCIONES_HAND_TO_BODY.md`](PROPUESTA_SOLUCIONES_HAND_TO_BODY.md) | Roadmap (P0-3 ya reflejado aquí) |

---

## 11. Checklist integración C3

- [ ] `PYTHONPATH` incluye `har/`
- [ ] `pocket_filter_config.json` = baseline G+H
- [ ] Llamada: `infer_from_poses_path` o `evaluate_pocket_with_policy` (no `detect_json` pocket)
- [ ] Árbol P0-3 implementado
- [ ] Logs con `motivo`, `context_veto`, rama orchestrator
- [ ] Prueba humo con npy real del piloto