# Baseline producción — pocket v2 (fase **G+H**)

**Congelado:** 2026-11-21 (métricas barrido `gate_temporal.csv` en `datos_limpios_triton`).  
**No modificar** `pocket_filter_config.json` sin copiar este baseline y re-barrido si se buscan las mismas cifras.

---

## Stack activo

| Pieza | Estado |
|-------|--------|
| Backend | `hand_to_body_backend: v2` |
| Geometría pocket | eje torso + perfil (`enable_axis_body_approach`, `allow_side_view_geometry`) |
| Política | `action_policy.enabled: true`, **`gate_mode: robbery_confirm`** |
| Temporal (chunks ~8 s) | **`temporal.enabled: true`**, ventana **2,5 s**, paso **1,25 s** |
| Context guard | `enabled: false` |

Código: `pocket_filter/temporal_eval.py`, `action_policy.py`, `confidence.py`, `evaluate.py`.

---

## Métricas de referencia (auditoría `hand_to_body`)

Dataset: `/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result`  
CSV: `/tmp/gate_temporal.csv` (o nombre local equivalente tras batch).

| Cat | Aciertos | TP | TN | FP | FN | Notas |
|-----|----------|----|----|----|-----|-------|
| **0** | 4199/4633 | 0 | 4199 | **434** | 0 | **9,4 %** FP |
| **1** | 322/758 | 0 | 322 | 289 | 0 | |
| **2** | 394/1954 | 0 | 394 | 1560 | 0 | buffers largos |
| **3** | 883/2182 | 0 | 459 | 1723 | 0 | |
| **4** | 1102/2403 | 0 | 1102 | 1301 | 0 | |
| **5** | 5171/5602 | 0 | 5171 | **431** | 0 | **7,7 %** FP |
| **6** | **3521/4192** | **3521** | 0 | 0 | **671** | **84,0 %** recall |
| **7** | 260/591 | 0 | 260 | 331 | 0 | |
| **9** | 401/914 | 0 | 401 | 513 | 0 | |
| **13** | 21/80 | 0 | 21 | 59 | 0 | |
| **14** | **334/401** | **334** | 0 | 0 | **67** | **83,3 %** recall |

Comparativa rápida cat. **6** / **0**:

| Versión | Cat. 6 TP | Cat. 0 FP |
|---------|-----------|-----------|
| Pocket puro | 3957 | 3816 |
| `robbery_confirm` sin ventanas | 2925 | 164 |
| **Baseline G+H (actual)** | **3521** | **434** |

---

## Contrato salida (Kanvis / motion)

Un **npy = un chunk ~8 s**.

```text
if candidato_modelo_robo:
    if motion_suppress:        → descartar (FP)
    elif hand_to_body:         → gesto claro (confirmar)
    elif hand_to_body_raw:    → hay gesto pocket; SAM/revisión
    else:                      → descartar
```

Columnas batch relevantes: `hand_to_body`, `hand_to_body_raw`, `hand_to_body_confidence`, `motion_suppress`, `motivo`, `intervalo` vía `eval_*` y JSON `intervalo_s` en `har2_test`.

---

## Rollback (de más suave a más agresivo)

### A) Recuperar solo config sin git

```bash
cp har/pocket_filter_config.BASELINE_PRODUCCION_H.json har/pocket_filter_config.json
```

### B) Desactivar ventanas (volver a barrido sin temporal, ~2925 TP cat 6)

En `pocket_filter_config.json`:

```json
"temporal": { "enabled": false }
```

Referencia CSV histórico: `gate_robbery_v2.csv`.

### C) Modo política anterior (tiers + cotidiano)

```json
"action_policy": { "gate_mode": "full" }
```

### D) Solo pocket (sin política) — recall ~94 % cat 6, FP cat 0 ~82 %

```json
"action_policy": { "enabled": false }
```

### E) Sin acercamiento por eje torso

```json
"pocket": { "enable_axis_body_approach": false }
```

### F) Git

```bash
cd har && git log --oneline -15
# git checkout <commit> -- pocket_filter/ pocket_filter_config.json
```

Siempre **re-ejecutar batch** y `audit_pocket_csv` tras un rollback.

---

## Comandos reproducibilidad

```bash
cd ~/kanvis/piloto_v1/HAR_v2/har
git pull
.venv/bin/python3 -m pytest tests/ -q

.venv/bin/python3 batch_data_result.py \
  /media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result \
  --backend v2 \
  -o /tmp/gate_temporal.csv \
  --workers 100

.venv/bin/python3 audit_pocket_csv.py /tmp/gate_temporal.csv
```

Prueba unitaria en un chunk:

```bash
.venv/bin/python3 har2_test.py /ruta/user_0/poses_full.npy
# Ventana pico: …   Modo política: robbery_confirm
# Comparar con todo el clip:
.venv/bin/python3 har2_test.py /ruta/user_0/poses_full.npy --whole-clip
```

---

## Archivos asociados

| Archivo | Rol |
|---------|-----|
| `har/pocket_filter_config.json` | Config **activa** |
| `har/pocket_filter_config.BASELINE_PRODUCCION_H.json` | **Copia congelada** baseline G+H |
| `docs/HAND_TO_BODY_PROGRESO.md` | Historial experimentos |
| `docs/HAND_TO_BODY_SOLUCIONES.md` | Diseño / handoff |

---

## Regla de cambios

1. Copiar `pocket_filter_config.json` → `pocket_filter_config.PRUEBA_<fecha>.json`.  
2. Un cambio (o un bloque) por experimento.  
3. Barrido `-c 0 6 14` o completo; guardar CSV con nombre fijo.  
4. Actualizar tabla en este doc **solo** si el cambio reemplaza baseline en producción.

---

## Experimento I (posterior al baseline)

`temporal.walk_gate_enabled: true` en `pocket_filter_config.json` activo; **no** está en `BASELINE_PRODUCCION_H.json`. Ver fase I en [`HAND_TO_BODY_PROGRESO.md`](HAND_TO_BODY_PROGRESO.md).