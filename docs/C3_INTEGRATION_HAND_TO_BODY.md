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
| `pose_frame_timestamps.npy` | Tiempo por frame (**debe ser t relativo al chunk**, o epoch re-anclado en HAR); si inválido → fallback `fps` |
| `video` vía convención pose | Aspect ratio (si existe) |

Formato npy: `(T, 8, 2)` o `(T, 8, 3)` con conf en canal 3 — ver `pocket_filter/io.py`.

### 1.1 Fuente en C3: MinIO

En Kanvis-C3 el npy **no** vendrá del disco del worker: el mensaje de candidato Motion incluye la **clave de objeto** (o prefijo de chunk). C3 debe:

1. Descargar `poses_full.npy` desde MinIO (cliente `minio` / `boto3` / SDK interno).
2. Opcionalmente, en el **mismo prefijo** que el npy:
   - `valid_mask.npy`
   - `pose_frame_timestamps.npy`
   - `meta.json` → campo `fps` (y aspect si el pipeline lo expone).
3. Pasar **`fps`** (y si aplica **`aspect`**) al evaluador: sin `meta.json` local, `infer_from_poses_bytes` usa `fps=12.5` y `aspect=16:9` por defecto — **incorrecto** si el clip real difiere.

Convención de claves (ajustar al bucket real de C3):

```text
{bucket}/{session_id}/chunks/{chunk_id}/poses_full.npy
{bucket}/{session_id}/chunks/{chunk_id}/valid_mask.npy          # opcional
{bucket}/{session_id}/chunks/{chunk_id}/pose_frame_timestamps.npy
{bucket}/{session_id}/chunks/{chunk_id}/meta.json
```

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

### 3.1 MinIO → inferencia en memoria (producción C3)

```python
from pocket_infer import infer_from_poses_bytes, result_to_c3_dict, validate_c3_payload

poses_bytes = minio_client.get_object(bucket, key).read()
mask_bytes = minio_client.get_object(bucket, mask_key).read()  # si existe
ts_bytes = minio_client.get_object(bucket, ts_key).read()      # si existe
meta = json.loads(minio_client.get_object(bucket, meta_key).read())
fps = float(meta.get("fps", 12.5))
aspect = float(meta.get("aspect", 16 / 9))  # o derivar del evento Motion

result = infer_from_poses_bytes(
    poses_bytes,
    valid_mask_bytes=mask_bytes,
    timestamps_bytes=ts_bytes,
    fps=fps,
    aspect=aspect,
    source_label=key,
)
payload = result_to_c3_dict(result, object_key=key)
assert validate_c3_payload(payload) == []
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

### 4.1 Ritmo del movimiento (`movement`)

Cada JSON incluye el objeto **`movement`** (esqueleto + contexto marcha). Sirve para combinar HAR con Motion **sin** sustituir `hand_to_body` ni `motion_suppress`.

| Campo | Tipo | Uso en C3 |
|-------|------|-----------|
| **`hip_speed_norm`** | float | Velocidad de traslación del cuerpo (cadera / ancho hombros, u/s). Alta ⇒ marcha rápida. |
| **`wrist_approach_vel_min`** | float | Velocidad radial muñeca→bolsillo (≤ 0 al acercarse). |
| **`wrist_approach_speed`** | float | `\|min\|` del anterior (más fácil de leer). |
| **`near_dwell_s`** | float | Segundos con muñeca cerca del bolsillo. |
| **`body_pace`** | string | `quiet` \| `walking` \| `fast` |
| **`approach_pace`** | string | `none` \| `slow` \| `moderate` \| `snap` |
| **`tempo_hint`** | string | `natural` \| `neutral` \| `hurried` (heurística auditable) |
| **`walk_frac`**, **`stop_frac`** | float | Mismo contexto que `ctx_*` en el payload raíz. |

Ejemplo de lógica de negocio (orientativa):

- `hand_to_body` + `tempo_hint == "natural"` → prioridad baja / compra cotidiana.
- `tempo_hint == "hurried"` con Motion fuerte → subir prioridad aunque `hand_to_body` sea false.
- Los campos planos **`kin_*`** siguen en el JSON (compatibilidad batch); C3 puede usar solo `movement`.

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

Ejemplo mínimo servicio (MinIO):

```python
def on_motion_candidate(event: dict) -> dict:
    from pocket_infer import infer_from_poses_bytes, result_to_c3_dict, validate_c3_payload

    key = event["poses_object_key"]  # p.ej. session/.../poses_full.npy
    poses_bytes = storage.get_bytes(key)
    fps = float(event.get("fps") or load_meta_fps(key))
    result = infer_from_poses_bytes(poses_bytes, fps=fps, source_label=key)
    payload = result_to_c3_dict(result, object_key=key)
    errs = validate_c3_payload(payload)
    if errs:
        raise ValueError("; ".join(errs))
    return payload
```

---

## 8. Pruebas obligatorias tras integrar en C3

Ejecutar **en el repo C3** después de cablear HAR + MinIO (no sustituyen `pytest` de HAR_v2).

### 8.1 Nivel A — humo local (sin MinIO)

Antes del primer deploy, con un npy en disco (mismo artefacto que luego subirás a MinIO):

```bash
export HAR_ROOT=/path/HAR_v2/har
cd "$HAR_ROOT"
cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
.venv/bin/python3 har2_test.py "$POSES_FULL" --json -o /tmp/htb_smoke.json
```

Comprueba que el JSON tiene `hand_to_body`, `motivo`, `resultado`.

### 8.2 Nivel B — contrato Python (`validate_c3_payload`)

Test unitario en C3 (sin red) reutilizando bytes de un npy fijo en `tests/fixtures/`:

```python
# tests/test_har_pocket_minio.py
import json
from pathlib import Path

import pytest
from pocket_infer import infer_from_poses_bytes, result_to_c3_dict, validate_c3_payload

FIXTURE = Path(__file__).parent / "fixtures" / "poses_full_smoke.npy"
GOLDEN = Path(__file__).parent / "fixtures" / "htb_smoke_golden.json"

@pytest.mark.har
def test_pocket_contract_from_bytes():
    data = FIXTURE.read_bytes()
    payload = result_to_c3_dict(
        infer_from_poses_bytes(data, fps=12.5, aspect=16 / 9),
        object_key="test/poses_full.npy",
    )
    assert validate_c3_payload(payload) == []

@pytest.mark.har
def test_pocket_golden_branch():
    """Regenerar GOLDEN solo al cambiar baseline H explícitamente."""
    data = FIXTURE.read_bytes()
    payload = result_to_c3_dict(infer_from_poses_bytes(data, fps=12.5), object_key="test/poses_full.npy")
    golden = json.loads(GOLDEN.read_text())
    for field in ("orchestrator_branch", "hand_to_body", "hand_to_body_raw", "motivo", "n_frames"):
        assert payload[field] == golden[field], field
```

**Golden:** generar una vez con HAR en pin:

```bash
cd "$HAR_ROOT" && cp pocket_filter_config.BASELINE_PRODUCCION_H.json pocket_filter_config.json
python3 -c "
from pathlib import Path; import json
from pocket_infer import infer_from_poses_bytes, result_to_c3_dict
p = result_to_c3_dict(infer_from_poses_bytes(Path('$FIXTURE').read_bytes(), fps=12.5), object_key='test')
Path('tests/fixtures/htb_smoke_golden.json').write_text(json.dumps({k:p[k] for k in ('orchestrator_branch','hand_to_body','hand_to_body_raw','motivo','n_frames')}, indent=2))
"
```

### 8.3 Nivel C — integración MinIO real (`@pytest.mark.integration`)

Variables de entorno (ejemplo):

| Variable | Uso |
|----------|-----|
| `MINIO_ENDPOINT` | URL del cluster piloto |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | Credenciales CI o dev |
| `HAR_MINIO_BUCKET` | Bucket con chunks del piloto |
| `HAR_MINIO_POSES_KEY` | Clave de un npy conocido (robo o pasillo) |
| `HAR_MINIO_META_KEY` | Opcional; si vacío, derivar `meta.json` del prefijo del npy |

Esqueleto:

```python
@pytest.mark.integration
def test_har_from_minio_object(minio_client):
    key = os.environ["HAR_MINIO_POSES_KEY"]
    poses_bytes = minio_client.get_object(os.environ["HAR_MINIO_BUCKET"], key).read()
    fps = 12.5
    meta_key = key.rsplit("/", 1)[0] + "/meta.json"
    try:
        fps = float(json.loads(minio_client.get_object(bucket, meta_key).read())["fps"])
    except Exception:
        pass
    payload = result_to_c3_dict(
        infer_from_poses_bytes(poses_bytes, fps=fps, source_label=key),
        object_key=key,
    )
    assert validate_c3_payload(payload) == []
    assert payload["n_frames"] >= 8
```

**Criterios de aceptación MinIO:**

- Descarga OK y forma `(T, 8, 2|3)` con `T > 0`.
- Sin excepción HAR; `validate_c3_payload` vacío.
- Log de `orchestrator_branch` + latencia p95 anotada en el ticket de integración.

### 8.4 Nivel D — replay staging (opcional)

Lista de 10–20 claves MinIO (mix pasillo/robo del piloto): el worker C3 las procesa en modo dry-run (sin alertar) y vuelca CSV `object_key,orchestrator_branch,motivo,latency_ms`. Comparar distribución de ramas con órdenes de magnitud del informe full (`CONFIRM` >> `DISCARD_SUPPRESS`).

### 8.5 CI C3 recomendado

| Job | Cuándo |
|-----|--------|
| `pytest -m har` | Cada PR (fixture local, sin MinIO) |
| `pytest -m integration` | Nightly o manual post-deploy (MinIO piloto) |

Marcar dependencia: submodule/vendor `HAR_v2/har` + `pocket_filter_config.BASELINE_PRODUCCION_H.json` copiado en job de CI.

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
- [ ] Descarga MinIO: `poses_full.npy` + `meta.json` (fps) en el mismo prefijo
- [ ] Integración usa `infer_from_poses_bytes` (no solo path local)
- [ ] `pytest -m har` con fixture + golden en repo C3
- [ ] `pytest -m integration` contra MinIO piloto (`HAR_MINIO_POSES_KEY`)
- [ ] `validate_c3_payload` en el hot path o en tests