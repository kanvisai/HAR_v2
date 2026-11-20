# Progreso calibración `hand_to_body` (pocket v2 + política)

**Última actualización:** 2026-03-20 (sesión agente).  
**Objetivo:** recordar qué se probó, métricas y cómo volver atrás.

---

## Línea temporal (commits / features)

| Fase | Qué se hizo | Archivos / config |
|------|-------------|-------------------|
| **A** | Pocket v2 perfil + `ACERCAMIENTO_EJE_TORSO` | `pocket_filter/evaluate.py`, `allow_side_view_geometry`, `axis_*` en `pocket_filter_config.json` |
| **B** | Barrido sin política | `batch_data_result.py`, backend v2 |
| **C** | Confianza 0–1 + política v1 (umbral global) | `confidence.py`, `action_policy.py` — mató recall (~49 % cat 6) |
| **D** | Política v2 walk vs stop (sin categoría) | `action_policy.py` umbrales por motivo + marcha |
| **E** | HAR1 cotidiano + resumen `har2_test` | `detect_cotidian_actions`, `htb_tier`, `accion_recomendada` |
| **F** (esta prueba) | **`htb_tier`**: `alert` / `review` / `none` — eje no alarma solo | `action_policy.py`, CSV `htb_tier` |

---

## Métricas de referencia (para comparar barridos)

Dataset grande: `/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result`

| Config | cat 6 recall | cat 14 | cat 0 FP | Notas |
|--------|----------------|--------|----------|-------|
| Pocket puro (axis) | **94 %** (3957/4192) | 93 % | **82 %** FP (3816) | ~23k clips |
| Política v1 | 49 % | 51 % | **20 %** FP (921) | conf fija 0.62 |
| Política v2 | **80 %** (3359/4192) | 80 % | **28 %** FP (1278) | walk/stop |
| FN cat 6 v2 | 235 pocket miss + **598** política (`raw=true`) | | | `pocket_all_policy_v2.csv` |

Dataset prueba: `/media/8TB/.../data_clips_triton_pose/data_result` (muestra distinta)

| | cat 6 | cat 0 FP | FP motivos (aprox / contacto / eje) |
|---|-------|----------|--------------------------------------|
| `rune_policy.csv` | 75 % (159/212) | 38 % (1040/2763) | 540 / 442 / 296 |

---

## Cómo volver atrás

1. **Solo pocket (sin política)** — en `pocket_filter_config.json`:
   ```json
   "action_policy": { "enabled": false }
   ```
   El batch seguirá evaluando pero `hand_to_body` = pocket COMPATIBLE.

2. **Sin eje torso** — `enable_axis_body_approach: false` (recall 6 baja, FP bajan).

3. **Git:** `git log --oneline har/pocket_filter/` y revert del commit concreto; no tocar datos en 8TB.

4. **CSV columnas nuevas:** `hand_to_body_raw`, `hand_to_body_confidence`, `primary_action`, `htb_tier` — ignorables si consumes solo `hand_to_body`.

---

## Contrato de salida (producción sin categoría)

| Campo | Uso |
|-------|-----|
| `hand_to_body` | Alarma dura (política) |
| `hand_to_body_raw` | Pocket COMPATIBLE |
| `hand_to_body_confidence` | 0–1 |
| `htb_tier` | `alert` \| `review` \| `none` — eje débil → como mucho `review` |
| `primary_action` | walk, stop, … si no alerta |
| `accion_recomendada` / `accion_confianza_pct` | `har2_test` / JSON |

---

## Próximo experimento (F)

- Barrido completo con `htb_tier` en CSV.
- Medir: TP si `hand_to_body OR htb_tier=review` en cat 6; FP en 0 solo `alert`.
- Ajustar `review_min_confidence`, `review_axis_min` en `action_policy`.

---

## Comandos de prueba (copiar log)

Ver sección **“Suite de pruebas y log”** al final de este archivo (actualizada con cada fase).

---

## Suite de pruebas y log

Ejecutar en el servidor (tras `git pull`):

```bash
cd ~/kanvis/piloto_v1/HAR_v2/har
LOG=/tmp/har_progress_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "=== git ==="
git rev-parse HEAD
git log -1 --oneline

echo "=== pytest ==="
.venv/bin/python3 -m pytest tests/ -v

echo "=== har2_test muestra (cat4) ==="
.venv/bin/python3 har2_test.py \
  "/media/8TB/DatosEntrenamiento/data_clips_triton_pose/data_result/4/nodate_ac7bc87b3c_clip_buffer_000000_000016_c4_d1698c/user_0/poses_full.npy"

echo "=== barrido calibración 0+6+14 ==="
.venv/bin/python3 batch_data_result.py \
  "/media/8TB/DatosEntrenamiento/data_clips_triton_pose/data_result" \
  -c 0 6 14 --backend v2 \
  -o /tmp/har_progress_batch.csv --workers 100

echo "=== auditoría ==="
.venv/bin/python3 audit_pocket_csv.py /tmp/har_progress_batch.csv

echo "=== FP cat0 motivo ==="
awk -F',' 'NR>1 && $3==0 && $13=="FP" {print $9}' /tmp/har_progress_batch.csv | sort | uniq -c | sort -rn

echo "=== htb_tier cat6 FN/TP ==="
awk -F',' 'NR>1 && $3==6 {print $13,$24}' /tmp/har_progress_batch.csv | sort | uniq -c | sort -rn | head -20

echo "LOG: $LOG"
```

Barrido **completo** (largo, opcional):

```bash
.venv/bin/python3 batch_data_result.py \
  /media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result \
  --backend v2 -o /tmp/har_full_$(date +%Y%m%d).csv --workers 100
```

Pásame el archivo: `cat /tmp/har_progress_*.log` o las últimas líneas de auditoría + `htb_tier`.