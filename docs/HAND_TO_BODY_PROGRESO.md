# Progreso calibración `hand_to_body` (pocket v2 + política)

**Última actualización:** 2026-11-21 (sesión agente).  
**Objetivo:** recordar qué se probó, métricas y cómo volver atrás.

> **Baseline producción congelada (G+H):** ver [`HAND_TO_BODY_BASELINE_PRODUCCION.md`](HAND_TO_BODY_BASELINE_PRODUCCION.md).  
> Config copia: `har/pocket_filter_config.BASELINE_PRODUCCION_H.json`.  
> CSV referencia: `gate_temporal.csv` — cat **6: 3521/4192**, cat **0 FP: 434**.

---

## Línea temporal (commits / features)

| Fase | Qué se hizo | Archivos / config |
|------|-------------|-------------------|
| **A** | Pocket v2 perfil + `ACERCAMIENTO_EJE_TORSO` | `pocket_filter/evaluate.py`, `allow_side_view_geometry`, `axis_*` en `pocket_filter_config.json` |
| **B** | Barrido sin política | `batch_data_result.py`, backend v2 |
| **C** | Confianza 0–1 + política v1 (umbral global) | `confidence.py`, `action_policy.py` — mató recall (~49 % cat 6) |
| **D** | Política v2 walk vs stop (sin categoría) | `action_policy.py` umbrales por motivo + marcha |
| **E** | HAR1 cotidiano + resumen `har2_test` | `detect_cotidian_actions`, `htb_tier`, `accion_recomendada` |
| **F** | **`htb_tier`**: `alert` / `review` / `none` | `action_policy.py` |
| **G** | **`gate_mode: robbery_confirm`** — filtro motion + `motion_suppress` | |
| **H** | **Ventanas 2,5 s / paso 1,25 s** en chunks ~8 s | `temporal_eval.py` |
| **G+H** ★ **baseline prod** | Temporal + `robbery_confirm` + `motion_suppress` | Ver `HAND_TO_BODY_BASELINE_PRODUCCION.md` |

---

## Métricas de referencia (para comparar barridos)

Dataset grande: `/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result`

| Config | cat 6 recall | cat 14 | cat 0 FP | Notas |
|--------|----------------|--------|----------|-------|
| Pocket puro (axis) | **94 %** (3957/4192) | 93 % | **82 %** FP (3816) | ~23k clips |
| Política v1 | 49 % | 51 % | **20 %** FP (921) | conf fija 0.62 |
| Política v2 | **80 %** (3359/4192) | 80 % | **28 %** FP (1278) | walk/stop |
| FN cat 6 v2 | 235 pocket miss + **598** política (`raw=true`) | | | `pocket_all_policy_v2.csv` |
| **G+H temporal** ★ | **84 %** (3521/4192) | **83 %** (334/401) | **434** (9,4 %) | `gate_temporal.csv` |
| G sin ventanas | 70 % (2925) | 75 % (299) | **164** (3,5 %) | `gate_robbery_v2.csv` |

Dataset prueba: `/media/8TB/.../data_clips_triton_pose/data_result` (muestra distinta)

| | cat 6 | cat 0 FP | FP motivos (aprox / contacto / eje) |
|---|-------|----------|--------------------------------------|
| `rune_policy.csv` | 75 % (159/212) | 38 % (1040/2763) | 540 / 442 / 296 |
| `har_progress_batch` (F, axis→review) | **49 %** (103/212) alarma | **27 %** FP (756/2763) | **0 / 392 / 364** eje fuera de alert |

---

## Cómo volver atrás

**Detalle completo y métricas congeladas:** [`HAND_TO_BODY_BASELINE_PRODUCCION.md`](HAND_TO_BODY_BASELINE_PRODUCCION.md).

1. **Restaurar config baseline:** `cp har/pocket_filter_config.BASELINE_PRODUCCION_H.json har/pocket_filter_config.json`
2. **Solo quitar ventanas:** `"temporal": { "enabled": false }` → métricas ~`gate_robbery_v2.csv`
3. **Política antigua:** `"gate_mode": "full"`
4. **Sin política:** `"action_policy": { "enabled": false }` → pocket puro
5. **Git:** `git log --oneline har/pocket_filter/` + checkout selectivo si hace falta

---

## Modo `robbery_confirm` (G)

- **Confirmar** (`hand_to_body=true`): `CONTACTO`; `APROXIMACION` solo sin marcha; `EJE` solo parado (`stop_frac`) y confianza.
- **Nunca confirmar** aprox/eje **caminando**; contacto caminando exige conf ≥ `gate_contact_min_confidence_while_walking`.
- **`hand_to_body_raw`**: pocket COMPATIBLE (auditoría).
- Integración motion (recomendada):
  ```text
  if modelo_robo:
      if motion_suppress:  # sin pocket o caminando + gesto débil
          descartar FP
      elif hand_to_body:  # gesto claro
          escalar
      elif hand_to_body_raw:  # hay gesto, no tan claro
          SAM / revisión (recall cat 6)
      else:
          descartar
  ```
- **Un solo boolean no alcanza** ~3957 TP en 6 y ~164 FP en 0: usar `hand_to_body` (claro) + `hand_to_body_raw` (pocket) + `motion_suppress`.

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

## Fase **J** — `--backend mix`

v2 + legacy + cotidiano cuando la confianza v2 &lt; **0,80** (sin categoría). Config `mix_backend` en JSON. Barrido:

```bash
.venv/bin/python3 batch_data_result.py .../data_result --backend mix -o /tmp/mix.csv
.venv/bin/python3 har2_test.py clip.npy --backend mix
```

Columna nueva CSV: `legacy_hand_to_body`; regla en `context_veto` (`MIX_*`).

---

## Fase **I** — `temporal.walk_gate` (ventana ganadora + marcha)

**Objetivo:** bajar FP cat. **0** (~434 → ~250) sin perder mucho recall cat. **6** (3521).

| Pieza | Detalle |
|-------|---------|
| Código | `temporal_eval._best_clear_window`, veto `TEMPORAL_WALK_GATE` |
| Config activa | `walk_gate_enabled: true` en `temporal` |
| Rollback G+H | `cp pocket_filter_config.BASELINE_PRODUCCION_H.json` (gate **desactivado**) |

Tras barrido, guardar CSV como `gate_temporal_walk.csv` y comparar cat **0/6/14**.

## Próximos experimentos (sin tocar baseline sin re-barrido)

- Medir `hand_to_body_raw` en cat 6 en CSV para recall “gesto en pose”.
- Ajustar `walk_gate_window_frac` / `walk_gate_contact_min_confidence` según barrido I.

---

## Comandos de prueba (copiar log)

Ver sección **“Suite de pruebas y log”** al final de este archivo (actualizada con cada fase).

---

## Suite de pruebas y log

Ejecutar en el servidor (tras `git pull`):

Barrido rápido (~10% por categoría, indicativo):

```bash
.venv/bin/python3 batch_data_result.py .../data_result --sample-frac 0.1 \
  --backend v2 -o /tmp/gate_10pct.csv --workers 50 --seed 42
.venv/bin/python3 audit_pocket_csv.py /tmp/gate_10pct.csv
```

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

echo "=== recall cat6: alarma + cola review (col 23=htb_tier) ==="
awk -F',' 'NR>1 && $3==6 {
  if ($6=="true" || $23=="review") tp++; else fn++;
  tot++
} END { printf "caught %d / %d = %.1f%%\n", tp+0, tot+0, 100*tp/(tot?tot:1) }' /tmp/har_progress_batch.csv

echo "=== FP cat0 motivo ==="
awk -F',' 'NR>1 && $3==0 && $13=="FP" {print $9}' /tmp/har_progress_batch.csv | sort | uniq -c | sort -rn

echo "=== FP cat0 htb_tier (col 23) ==="
awk -F',' 'NR>1 && $3==0 && $6=="true" {print $23}' /tmp/har_progress_batch.csv | sort | uniq -c

echo "=== FN cat6 raw+tier ==="
awk -F',' 'NR>1 && $3==6 && $13=="FN" {print $14,$23}' /tmp/har_progress_batch.csv | sort | uniq -c | sort -rn | head -15

echo "LOG: $LOG"
```

Barrido **completo** (largo, opcional):

```bash
.venv/bin/python3 batch_data_result.py \
  /media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result \
  --backend v2 -o /tmp/har_full_$(date +%Y%m%d).csv --workers 100
```

Pásame el archivo: `cat /tmp/har_progress_*.log` o las últimas líneas de auditoría + `htb_tier`.