# Fase 4 — notas cinemática (muestra 10 %, seed 42)

**Fecha:** 2026-11-21  
**CSV:** `fase4_kin_10pct.csv`  
**Clear:** sin cambio vs G+H+P0-3 (cat 6 **352/419**, cat 0 FP **58**).

## Separación global FN vs FP (raw∧¬clear)

| Feature | Mediana FN cat 6 (pos) | Mediana FP cat 0 (neg) | ¿Útil? |
|---------|------------------------|------------------------|--------|
| `kin_wrist_radial_vel_min` | -1.85 | **-2.16** | No (FP más “rápido” hacia bolsillo) |
| `kin_approach_monotone_frac` | 0.50 | 0.50 | No |
| `kin_hip_speed_norm` | **0.62** | **0.76** | Débil (pasillo marcha más) |
| `kin_elbow_angle_deg` | 155 | 156 | No |
| `kin_wrist_axis_dist_min` | **0.15** | 0.18 | Muy débil |

**Conclusión:** con estas 5 magnitudes **no** basta un umbral global para desambiguar REVIEW. Coherente con que el pocket ya consumió distancia/Δ y el residual sea sobre todo **motivo + marcha** (`robbery_confirm`).

## Dirección ingeniería (sin red)

1. **Condicionar por `motivo`** (eje vs contacto) + `kin_hip_speed_norm` / `ctx_walk_frac`.
2. **Ampliar P1-1:** tiempo en `r_near`, retirada post-contacto, antifase bilateral, periodicidad braceo (spec §P1-1).
3. **No** bajar umbral contacto global (Fase 2-c).
4. **Ranker logístico:** opcional cola SAM; no sustituye cinemática ni Motion.

## Comando seguimiento

```bash
.venv/bin/python3 analyze_kinematics_separation.py /tmp/fase4_kin_10pct.csv
```