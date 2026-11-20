# Plan de acción técnico — afinar `hand_to_body` / filtro de aproximación a bolsillo (Kanvis)

**Para:** agente desarrollador · **Fecha del plan:** 2026-11-20
**Base:** `HAND_TO_BODY_SOLUCIONES.md` + síntesis de tres respuestas independientes, citadas en el orden en que llegaron:
**A** (plan R1–R7, 4 hitos de 7+ semanas), **B** (plan R01–R14 por fases), **C** (plan R1–R10 en 4 semanas).
**Estado:** solo análisis y especificación. Este documento no contiene código; las reglas y fórmulas son el contrato a implementar.
Las referencias `§x` apuntan a `HAND_TO_BODY_SOLUCIONES.md`.

### Invariantes no negociables (aplican a todo el plan)

| # | Invariante |
|---|-----------|
| I1 | `INDETERMINADO` nunca descarta una alerta. Solo `INCOMPATIBLE` puede proponer cerrar la hipótesis de bolsillo. |
| I2 | Ausencia de `COMPATIBLE` ≠ `INCOMPATIBLE`. `hand_to_body=True` significa solo `COMPATIBLE`; su inverso nunca se usa para descartar. |
| I3 | La decisión usa poses originales: sin suavizado, sin interpolación, sin relleno. La mediana-3 solo existe para dibujar. |
| I4 | Un frame no observado (máscara, NaN, conf baja, hold-last, identidad rota) jamás cuenta como "mano lejos". |
| I5 | Propiedad metamórfica: añadir huecos o invalidar frames nunca puede convertir un resultado en `INCOMPATIBLE`. |
| I6 | No se toca `detect.py` ni la taxonomía c0/c1/c5 en v1. El filtro es un artefacto aparte (`eval_pocket_output.json`) (§6.5, §11.2). |
| I7 | Misma entrada + misma config + misma versión → misma salida, byte a byte. Toda salida lleva `filter_version` y `config_hash`. |

---

## 1. Resumen ejecutivo

- Kanvis detecta hurtos en retail con un pipeline C1 (esqueleto 8 KP) → C2 (motion p41/p48) → C3 (Orchestrator, HAR, SAM3, revisión humana). El modelo de robo genera FP en acciones cotidianas (pasear, mirar el lineal, llevar producto, gestos breves) (§0.3).
- Se pide un filtro determinista sobre esqueleto que responda "¿hubo aproximación visible de una muñeca al bolsillo?" con tres estados, y que solo `INCOMPATIBLE` pueda bajar una alerta (§2.2).
- El prototipo `pocket_filter` + `har2_test.py` demuestra la idea, pero no es desplegable: ignora `valid_mask (T,)`, evalúa el clip entero, no tiene tests, visualización ni calibración, y su `INCOMPATIBLE` exige `valid.all()` sobre todo el NPY (§11.4–11.6).
- Las tres respuestas coinciden en el orden: endurecer datos/ventana/tests → herramientas de medida → observación (sin aplicar descarte) → activación condicionada a recall.
- Esta síntesis añade cuatro cosas que ninguna respuesta cubría: una **auditoría de datos reales previa a calibrar** (R00), un **núcleo temporal v2** con lógica trivaluada, histéresis y márgenes de incertidumbre (R03), un **cierre corporal** que reduce la dependencia de "hipótesis abiertas" (R12) y **tamaños de muestra con intervalos de confianza** para el gate de recall (§6).
- Hipótesis crítica a verificar antes de prometer resultados (§3.4-H1): con el brazo colgando en vista frontal, la muñeca ya está dentro o cerca de `r` del bolsillo. Si se confirma, `INCOMPATIBLE` se concentrará en FP con la mano extendida o elevada, y el caminar/balanceo frontal acabará en `INDETERMINADO`.
- Recomendación principal: endurecer `pocket_filter` como módulo determinista ternario (Opción A), integrarlo post-alarma en modo observación (Opción C) y activar el descarte solo con evidencia estadística de que no se pierden hurtos.

---

## 2. Comprensión del contexto

### 2.1 Pipeline Kanvis y dónde encaja el filtro

```text
C1 Monitoring: tracks + poses_full.npy (T,8,2|3) + valid_mask + timestamps
   → C2 Motion (p41/p48): candidato a hurto por chunk
        → [FILTRO pocket_filter, solo si C2 abre candidato]  ← este trabajo
              COMPATIBLE    → conservar / seguir SAM3
              INDETERMINADO → conservar, registrar motivo
              INCOMPATIBLE  → propone cerrar hipótesis bolsillo (solo política ON, sin otras hipótesis abiertas)
        → C3: SAM3 gate, política de aviso, revisión humana
```

El filtro se evalúa post-alarma sobre la ventana del evento. No corre en cada chunk de `har_premodel` (§4 Opción C).

### 2.2 Qué es "acción rara" en este alcance

No es una anomalía estadística: es un **patrón geométrico del esqueleto ligado a ocultación**. En v1 solo se afina "muñeca que viaja de lejos a la zona de bolsillo lateral" (§0.4). Walk/stop/turn/arm_swing son cotidianos y sirven para explicar FP. Pecho, abrigo y mochila son hipótesis alternativas (v2 o cierre corporal, R12). El filtro no detecta el objeto: eso es SAM3/VLM.

### 2.3 Política propuesta

| Motion | Filtro | Acción propuesta | Acción aplicada |
|--------|--------|------------------|-----------------|
| Sí | `COMPATIBLE` | `CONSERVAR` | `null` en observación; nunca descarta |
| Sí | `INDETERMINADO` (cualquier motivo, incluido `ERROR`) | `ABSTENERSE` | nunca descarta |
| Sí | `INCOMPATIBLE` + sin hipótesis alternativa abierta + política activa + cámara en allowlist + versión aprobada + datos no marcados `geometria_aproximada` ni `tiempo_asumido` | `CERRAR_HIPOTESIS_BOLSILLO` | solo en `modo=activo` |
| No | — | — | no aplica |

La política vive en un módulo puro y separado de la evaluación (R07). Una nueva versión del filtro no activa una política nueva por sí sola.

---

## 3. Diagnóstico del prototipo actual

### 3.1 Fortalezas

- Contrato correcto: tres estados, trayectoria lejos→cerca (no proximidad acumulada), evidencia con tiempos y distancias (§2.2–2.4, §3).
- Ancla en centro de bolsillo (`B_j = cadera_j + 0,08·L·u`) y 4 parejas muñeca×bolsillo, con distancias adimensionales en unidades de torso.
- No usa suavizado en la decisión; codo opcional; `hand_to_body` = solo `COMPATIBLE` (§11.1).
- Separado del HAR de pasillo (§11.2). API pura y reproducible, apta para tests.

### 3.2 Debilidades (§11.5–11.6 y `evaluate.py`)

| Debilidad | Efecto |
|-----------|--------|
| `valid_mask (T,)` ignorado; sin máscara por KP; hold-last no detectado | Poses congeladas pueden simular contacto o lejanía |
| Evalúa el clip completo; `INCOMPATIBLE` exige `valid.all()` + un solo tramo continuo | `INDETERMINADO` masivo en chunks de ~8 s; atribución a gestos ajenos al evento |
| `valid_ratio ≥ 0,7` fijo; `max_time_gap_factor` definido pero inerte | Penaliza clips cortos y cadencias variables |
| Contacto = "2 frames consecutivos" | No equivale a lo mismo a 10, 12 y 15 Hz |
| Umbrales duros en `r`/`R` sin margen de ruido | Cruces espurios en fronteras |
| "Cualquier pareja COMPATIBLE basta"; primer episodio gana | `COMPATIBLE` espurios en cruces de mano; episodios perdidos |
| Sin deshacer letterbox/espejo; escala solo en X | Geometría aproximada sin que el resultado lo diga |
| Sin tests, visualización, YAML, ni integración | Regresiones silenciosas; calibración a ciegas |

### 3.3 Riesgos si se desplegara tal cual

- **FN (el grave):** un `INCOMPATIBLE` obtenido con poses congeladas, huecos o identidad rota cerraría un hurto real.
- **FP:** `COMPATIBLE` por balanceo/cruce de manos refuerza alertas que debían caer; `INDETERMINADO` casi universal en chunks reales anula el valor del filtro.
- **Operación:** sin versión, config ni persistencia por `event_id` no hay auditoría ni rollback.

### 3.4 Hallazgos adicionales de la síntesis (a verificar en R00)

**H1 — Geometría de reposo (crítico).** Con el brazo colgando en vista frontal, la muñeca queda aproximadamente a 0,10–0,25 L del centro de bolsillo (estimación anatómica gruesa: hombro–muñeca ≈ 1,1–1,3 L; el bolsillo está 0,08 L bajo la cadera). Eso es ≤ `r = 0,30 L` o zona gris, de forma permanente. Si se confirma con datos:
- El contacto ("2 muestras con `d ≤ r`") se cumple en reposo. La barrera anti-FP real es el **origen lejano + Δ**, no `r`.
- Caminar o balancear en vista frontal producirá `INDETERMINADO` ("cerca sin origen"), no `INCOMPATIBLE`.
- `INCOMPATIBLE` se concentrará en FP con mano extendida o elevada (alcanzar el lineal, gesticular, sostener producto delante del cuerpo).
- **Antes de comprometer una tasa de FP evitados hay que medir la composición real de los FP** (R00).

**H2 — Palancas distintas.** `COMPATIBLE` e `INDETERMINADO` conservan ambos la alerta. Mejorar la fusión de parejas (R11) y descartar falsos `COMPATIBLE` mejora la **precisión de la evidencia positiva**, no reduce FP. Los FP solo se reducen subiendo `INCOMPATIBLE` seguro: ventana (R02), validez (R01), núcleo con márgenes (R03), geometría (R10) y cierre corporal (R12). La priorización de §4 sigue esa lógica.

**H3 — Cadencia.** El clip real de §11.6 tiene 34 frames. Si abarca ~8 s, la cadencia útil es ~4 Hz, muy por debajo de los 12 Hz de la spec. Con cadencia baja, "2 muestras en `r`" y "origen en [−1,5 s, −0,1 s]" pueden ser inalcanzables y el filtro casi siempre abstendrá. Medirlo antes de calibrar nada.

**H4 — Posible inconsistencia a comprobar.** §11.5 dice que la vista lateral "no bloquea INCOMPATIBLE global de otras parejas", pero §11.4 exige las 4 parejas `INCOMPATIBLE` y la vista lateral se calcula por frame (común a todas). No asumir: fijarlo con un test en R04.

**H5 — Estadística.** 20–50 clips etiquetados (§12.6) solo certifican una pérdida de TP ≤ 6–14 % con 95 % de confianza si no se pierde ninguno (tabla en §6.4). No bastan para activar descarte.

---

## 4. Recomendaciones priorizadas

P0 = necesario para obtener evidencia fiable; P1 = necesario para un piloto sólido; P2 = expansión posterior. "Origen" indica qué respuestas lo proponían (N = aportación nueva de esta síntesis).

| ID | Prio | Recomendación | Impacto esperado | Esfuerzo | Dependencias | Origen |
|----|------|---------------|------------------|----------|--------------|--------|
| R00 | P0 | Auditoría de datos reales C1 (cadencia, máscaras, hold-last, histogramas de `d`, composición FP) | Fija parámetros iniciales con datos; valida H1 y H3 | Bajo (2–3 días) | NPY reales | N |
| R01 | P0 | Contrato de validez y tiempo: `valid_mask (T,)`, máscara por KP, hold-last, timestamps, gaps, identidad | Cero evidencia negativa inventada; menos `EVIDENCIA_PARCIAL` espurio | Medio | R00, export C1 | A,B,C |
| R02 | P0 | Ventana de evento `interval_s` + cobertura temporal para `INCOMPATIBLE` | `INCOMPATIBLE` usable; atribución correcta | Medio-alto | R01, timestamps | A,B,C |
| R03 | P0 | Núcleo temporal v2: símbolos F/M/N/U, histéresis, margen de incertidumbre, todo en segundos | Estabilidad en fronteras; equivalencia entre cadencias | Medio | R01, R04 | N (parcial B) |
| R04 | P0 | Suite de tests: tabla spec + metamórficos + golden + fuzz + latencia | Contrato y anti-regresión | Medio | Spec | A,B,C |
| R05 | P0 | Dataset etiquetado con protocolo + batch runner + comparativas | Métricas reproducibles | Medio | Etiquetado | A,B,C |
| R06 | P0 | Visualización diagnóstica (overlay, anillos, `d(t)`, tira de símbolos) | Calibración y auditoría rápidas | Medio | R01 | A,B,C |
| R07 | P0 | Contrato JSON versionado + política pura separada + persistencia idempotente | Auditoría; nunca descartar por booleano | Medio | R03 | B,A |
| R08 | P0* | Integración Orchestrator en modo observación (*arranca tras PR #3, en paralelo a calibración) | Datos reales y etiquetas sin riesgo | Alto | R07, R02 | A,B,C |
| R09 | P1 | Calibración restringida por recall (rejilla → TPE), config YAML `ignacio-obs-2`, óptimo robusto | Maximizar FP evitados sin perder hurtos | Medio-alto | R05, R06, R00 | A,B,C |
| R10 | P1 | Geometría: píxeles reales, anclaje lateral, `L` robusto por track, agachado, vista graduada, σ por confianza | Más `INCOMPATIBLE` seguros; menos zona gris | Medio | R00, R03 | A,N |
| R11 | P1 | Fusión de parejas: plausibilidad ipsi/contralateral, ranking determinista, multi-episodio | Menos `COMPATIBLE` espurios sin perder cruces legítimos | Medio | R03, R04 | A,B,C |
| R12 | P1 | Cierre corporal (`INCOMPATIBLE_CUERPO`): mano nunca entra en zona torso+bolsillos | Reduce dependencia de "hipótesis abiertas" | Medio | R03, R09 | N |
| R13 | P1 | Validación por cámara/escenario reservados, leave-one-camera-out, IC | Controla sobreajuste | Medio | R05 | B |
| R14 | P2 | Multi-destino v2 (pecho como destino positivo, hipótesis mochila explícita) | Menos FN de ocultación alternativa | Alto | v1 calibrada | A,B,C |
| R15 | P2 | Features auxiliares (velocidad radial robusta, extensión de brazo, ángulo de codo), DTW diagnóstico, ML como segunda opinión | Fronteras difíciles | Alto | Evidencia de que las reglas no bastan | B,C,A |

Orden de ejecución: R00 ∥ (R04+R01+R02) → R03 → R05+R06+R07 → R08 (observación) ∥ R09+R10+R11+R12 → R13 → activación → R14/R15.

### 4.1 Decisiones ante discrepancias entre las respuestas

| Tema | A | B | C | **Decisión** |
|------|---|---|---|--------------|
| Prioridad de calibración | P1 | P1 | P0 (primero) | **P1**: calibrar sin datos limpios ni tests es ajustar a artefactos. Lo P0 es medir (R00, R05). |
| Cuándo empieza observación | Tras calibración | Tras campaña reservada | Semana 4 | **Gate técnico propio** (tests OK, JSON, latencia, 0 crashes). Observación no actúa y genera las etiquetas que la calibración necesita, así que corre en paralelo. |
| Mapear `COMPATIBLE` → `Action hand_to_body` | Flag `legacy\|ignacio` | Separar | Mapear | **No mapear en v1.** Orchestrator lee `eval_pocket_output.json` (§6.5). El flag existe con default `legacy` y no toca c0/c1/c5. |
| `valid_mask` | Máscara de frame + hold-last k≥3 | 3 conceptos (inválido / no apto / apto) + máscara por KP | Propagar (T,) + conf<0,5 | **B + hold-last de A**, con `ε` y `k` medidos en R00. |
| Vista lateral | >50 % ventana → `INDETERMINADO` global | Por pareja | Por pareja | **Ambas**: por pareja y regla global de ventana (R02). |
| Contralaterales | Offset lateral en X | Penalizar sin vetar | Penalizar por codo anómalo | **Regla de cruce por coordenada lateral** (R11); codo solo auxiliar. |
| Tamaño de dataset | 20–50 clips | 20 = smoke; N por recall | ≥20 clips | **B + tabla de IC** (§6.4). |
| `valid_ratio` | Adaptativo/cobertura | Cobertura temporal | Adaptativo | **Cobertura temporal en ventana** + gaps acotados físicamente (R02). |
| ML / DTW | Aplazar | Aplazar | Aplazar | **P2 diagnóstico**; ML solo si las reglas no cierran. |

---

## 5. Propuestas técnicas (detalle)

### R00 — Auditoría de datos reales C1 (P0)

**Problema.** Los parámetros de la spec asumen 12 Hz y poses limpias (§2.3). El único clip real (§11.6) dio `EVIDENCIA_PARCIAL` en las 4 parejas. No conocemos cadencia, semántica de máscaras ni distribución real de `d`.

**Enfoque.** Script `tools/audit_poses.py` sobre todos los NPY disponibles (FP, TP y una muestra aleatoria). Informe md + csv con:
1. Cadencia efectiva con timestamps: mediana de `dt`, p5/p95, % de huecos > 2P.
2. Tabla de contingencia `valid_mask (T,)` × NaN × `conf < 0,5` (¿qué significa realmente "frame no exportable"?).
3. Histograma de `|Δ|` frame a frame por KP para fijar `ε`/`k` de hold-last empíricamente.
4. Histogramas de `d_ij` por clase (caminar, balanceo, lineal, parado, TP), con `r` y `R` marcados. **Valida o refuta H1.**
5. Distribución de `L` en px y de `ρ = ancho_hombros/L` (para `min_torso_px` y vista lateral).
6. Jitter de KP en ventanas casi estáticas → `σ₀` (R03).
7. Percentiles p99/p99,9 de `|dd/dt|` → `v_fis` y `g_max` (R02).
8. Frecuencia de saltos de escala/posición compatibles con ID swap.
9. Composición de los FP etiquetados: % con mano extendida, % en reposo colgando, % con producto, etc.

**Alternativas descartadas.** Calibrar directamente sobre los pocos clips (sobreajuste a artefactos de C1); validar solo con sintéticos (no ve los problemas reales).

**Métrica.** Informe completo y cada parámetro inicial de R01–R03 justificado por un percentil. Decisión explícita sobre H1 y H3.

**Archivos.** `tools/audit_poses.py` (nuevo), `docs/audit_c1_<fecha>.md`. No modifica `pocket_filter`.

### R01 — Contrato de validez y tiempo (P0)

**Problema.** `valid_mask (T,)` se ignora, hold-last puede simular contacto o lejanía, `max_time_gap_factor` no se aplica, el fps se asume (§11.5).

**Enfoque.**
1. Por frame y por KP, dos estados: `OBSERVADO` / `NO_OBSERVADO`. Observado = `valid_mask` de frame ∧ KP finito ∧ `conf ≥ c_min` (0,5 inicial) ∧ no hold-last ∧ sin ruptura de identidad. Distinguir además "válido pero geometría no apta" y "válido y apto".
2. `valid_mask (T,)` se propaga a todos los KP del frame; `(T,8)` se aplica por KP; si llegan ambos, AND. La causa (máscara / conf / hold-last / identidad) queda registrada en la evidencia.
3. Tiempo: usar `pose_frame_timestamps`; `P = mediana(dt)`; cortar tramo si `dt > max_time_gap_factor·P` (hacer efectivo el parámetro existente). Sin timestamps: usar fps declarado y marcar `tiempo_asumido=true`.
4. Hold-last: ≥ `k` muestras consecutivas idénticas (|Δ| < `ε`) en muñeca o torso → `NO_OBSERVADO`. Punto de partida de A: `k=3`, `ε=1e-6`; confirmar con R00 por si la exportación cuantiza coordenadas.
5. Ruptura de identidad: cambio de `track_id`, salto de `L` o del centro del torso por encima de lo plausible, o velocidad de muñeca > `v_fis`. Umbrales desde R00.
6. Geometría en **píxeles reales** (`x·W`, `y·H`, deshaciendo letterbox y espejo) cuando se conocen `W`, `H` y metadatos. Si no, esquema actual (escala X por aspect) con `geometria_aproximada=true`; la política activa lo trata como no apto para descartar.

**Alternativas descartadas.** Interpolar huecos (viola §2.4 y §6.4, inventa trayectorias); usar solo NaN (no cubre poses rellenadas ni frames invalidados fuera); confianza global (pierde diferencias muñeca/torso); tratar frames inválidos como distancia infinita (produce `INCOMPATIBLE` falsos); exigir siempre `(T,8)` (rompe compatibilidad Kanvis, C).

**Métrica.** 0 `INCOMPATIBLE` falsos en tests con huecos y hold-last; propiedad I5 verde; % de frames recuperados en NPY reales; desglose de causas de no-observado por cámara.

**Archivos.** `pocket_filter/io.py`, `pocket_filter/validity.py` (nuevo), `pocket_filter/config.py`, `har/har2_test.py` (`--valid-mask` acepta `(T,)`), fixtures de test. En Kanvis/C1: contrato de exportación de máscara por KP pre-relleno, timestamps, `W/H`, letterbox y espejo.

### R02 — Ventana de evento y cobertura temporal (P0)

**Problema.** Evaluar el NPY completo asocia aproximaciones ajenas al evento y hace casi imposible `INCOMPATIBLE` (`valid.all()`) en chunks reales (§11.5–11.6).

**Enfoque.**
1. API: `evaluate_pocket(..., interval_s=(t0, t1))` evalúa `[t0 − m, t1 + m]` con `window_margin_s = 1,0` (valor de partida de la documentación, no calibrado).
2. Ensamblador de ventana: unir chunks por cámara/sesión/track, ordenar por timestamp, deduplicar solapes por timestamp (tolerancia ½P) sin interpolar; un cambio de `track_id` en la ventana es ruptura de identidad.
3. `INCOMPATIBLE` por pareja requiere, **dentro de la ventana**:
   - cobertura temporal observada ≥ `c_min` (0,90 inicial);
   - ningún hueco > `g_max`, acotado físicamente: `g_max < (R − r − margen)/v_fis` (tiempo mínimo que necesita una mano para ir de lejos a cerca, con `v_fis` de R00);
   - al menos una muestra observada en cada tercio de la ventana;
   - vista apta en ≥ 50 % de la ventana; si más del 50 % es lateral → `INDETERMINADO` global `VISTA_NO_APTA`;
   - todas las muestras observadas lejanas con margen (R03).
4. Dos aproximaciones separadas por retirada (`d ≥ R` sostenido ≥ `τ_ret`) son dos episodios; se listan todos.
5. Si el contexto no basta para demostrar el gesto completo → `INDETERMINADO` (`EVIDENCIA_PARCIAL`).

**Alternativas descartadas.** Bajar `valid_ratio` a 0,3 (ocultaría hurtos con oclusión justo al entrar al bolsillo, A); ventana por nº de frames (falla con cadencia variable); clip completo + postfiltro (atribución incorrecta); ampliar contexto indefinidamente (asociaciones accidentales).

**Métrica.** % de eventos evaluables; reducción de `INDETERMINADO`/`EVIDENCIA_PARCIAL` en FP limpios; **0 `INCOMPATIBLE`** en clips con oclusión en la ventana crítica (sintéticos y TP); concordancia de tiempos de evidencia con la anotación.

**Archivos.** `pocket_filter/evaluate.py`, `pocket_filter/window.py` (nuevo), `pocket_filter/io.py`, `har/har2_test.py` (`--interval-s`). Kanvis: reconstrucción del evento en `har_robbery_review`.

### R03 — Núcleo temporal v2: símbolos, histéresis e incertidumbre (P0)

**Problema.** Las reglas actuales cuentan frames, usan umbrales duros que cruzan por ruido en `d = r` y `d = R`, y "2 frames consecutivos" dura distinto a 10, 12 y 15 Hz (§11.4).

**Enfoque.** Convertir cada serie `d_ij(t)` en una secuencia de símbolos con lógica trivaluada, sin suavizar:

| Símbolo | Condición (σ = incertidumbre de `d`, k = factor de margen) |
|---------|------------------------------------------------------------|
| `U` | No observado (R01) |
| `F` (lejos) | `d − k·σ ≥ R` |
| `N` (cerca) | `d + k·σ ≤ r` |
| `M` (gris) | cualquier otro caso observado |

- `σ` se propaga del jitter de KP (R00) y se infla con baja confianza de muñeca, cadera del lado `j` y torso. Con `k = 0` el símbolo se reduce a los umbrales de la spec.
- Histéresis: se entra en `N` con `d + kσ ≤ r` y se sale con `d − kσ > r + h`.
- **`COMPATIBLE` por pareja:** existe una entrada (≥ 2 muestras `N` consecutivas **y** ≥ `τ_contacto` = 0,15 s) y un origen `F` en `[t_ent − 1,5 s, t_ent − 0,1 s]` con `d_origen − d_entrada ≥ Δ` (más margen `kσ`), sin símbolo `U` mayor que `g_max` entre ambos y sin ruptura de identidad.
- **`INCOMPATIBLE` por pareja:** cobertura de R02 y todos los símbolos observados de la ventana son `F`. Un único `M` o `N` → `INDETERMINADO` (`ZONA_GRIS`), a favor de la seguridad.
- Resto → `INDETERMINADO` con motivo.
- Todo en segundos; la cadencia solo afecta a la resolución.
- `engine=v2` convive con `engine=v1` (`ignacio-obs-1`) para A/B. Con `k=0`, `h=0`, `τ_contacto = 2 muestras` v2 debe reproducir v1 (test de equivalencia).

**Alternativas descartadas.** Suavizar `d` (prohibido para decidir); parchear umbrales duros caso a caso; ML en esta fase (pierde determinismo y auditabilidad, §4 Opción F).

**Métrica.** Equivalencia v2(k=0) ≡ v1 en la tabla sintética; resultado idéntico a 10/12/15 Hz; con ruido gaussiano en fronteras, 0 `INCOMPATIBLE` espurios y cambios de estado mínimos; sin pérdida de recall en el catálogo.

**Archivos.** `pocket_filter/trajectory.py`, `geometry.py`, `reasons.py`, `evaluate.py` (fachada), `config.py`.

### R04 — Suite de tests (P0)

**Problema.** Solo existe una prueba manual (§9.1, §11.5).

**Enfoque.** Escribir los tests **antes** de refactorizar (los que fallen con v1 entran como `xfail` con motivo).

*Tabla de contrato (`tests/test_pocket_spec_table.py`, generador analítico de trayectorias a 12 Hz):*

| # | Caso | Esperado |
|---|------|----------|
| 1 | Lejos → bolsillo → contacto visible | `COMPATIBLE` |
| 2 | Lejos permanente con cobertura completa | `INCOMPATIBLE` |
| 3 | Cerca permanente sin origen observable | `INDETERMINADO` |
| 4 | Contacto de un solo frame o < `τ_contacto` | no `COMPATIBLE` |
| 5 | Hueco > `g_max` durante la aproximación | `INDETERMINADO` |
| 6 | Hueco ≤ `g_max` dentro de lejanía con cobertura ≥ `c_min` | `INCOMPATIBLE` |
| 7 | Ruptura de identidad durante el gesto | `INDETERMINADO` |
| 8 | Vista lateral en > 50 % de la ventana | `INDETERMINADO` (`VISTA_NO_APTA`) |
| 9 | Cruce legítimo de mano al bolsillo contralateral | `COMPATIBLE` |
| 10 | Balanceo de brazos | no `COMPATIBLE` |
| 11 | Sacar móvil (cerca → lejos sin origen lejano) | `INDETERMINADO` |
| 12 | Ajustarse la ropa | frontera documentada |
| 13 | Cesta pegada a la cadera | `COMPATIBLE` geométrico (no es prueba de robo) |
| 14 | Zoom ×0,5 y ×2 | mismo resultado |
| 15 | Espejo | mismo resultado con lados intercambiados |
| 16 | Cadencia 10/12/15 Hz y jitter temporal | mismo resultado |
| 17 | Dos aproximaciones separadas por retirada | 2 episodios |
| 18 | `d = r` y `d = R` exactos | comportamiento documentado (`≤ r` próximo, `≥ R` lejano) |
| 19 | `valid_mask (T,)` invalidando frames de la aproximación | `INDETERMINADO`, nunca `INCOMPATIBLE` |
| 20 | Hold-last durante lejanía o contacto | `NO_OBSERVADO`; ni `COMPATIBLE` ni `INCOMPATIBLE` falsos |
| 21 | Ruido gaussiano en fronteras `r`/`R` | estado estable (R03) |
| 22 | Agacharse (cambia `L`) | `INDETERMINADO` |
| 23 | Dos personas / ID swap | `INDETERMINADO` |

*Propiedades metamórficas (`tests/test_pocket_metamorphic.py`):*
- **M1** invalidar frames (NaN o máscara) nunca produce `INCOMPATIBLE` (I5) ni crea `COMPATIBLE` desde otro estado.
- **M2** escala y traslación de la pose → resultado idéntico.
- **M3** espejo horizontal + intercambio L/R → resultado idéntico con lados intercambiados.
- **M4** re-muestreo de una trayectoria analítica a 10/12/15 Hz → mismo resultado.
- **M5** añadir frames fuera de la ventana no cambia el resultado.
- **M6** permutar el orden de parejas → idéntico (determinismo).
- **M7** insertar frames hold-last ≡ insertar NaN.

*Además:* golden JSON por fixture (versionado); fuzz de 1000 máscaras aleatorias (sin excepciones; ningún `INCOMPATIBLE` con cobertura < `c_min`); benchmark de latencia p95 < 10 ms para T ≤ 150 (marcado para no ser flaky en CI); test que fija la duda H4.

**Alternativas descartadas.** Solo casos positivos; validación manual por CLI (no reproducible); tests que verifican la implementación y no la semántica de los estados.

**Métrica.** 100 % de la tabla y de las propiedades en CI en < 5 s; cobertura de todos los códigos de motivo; 0 regresiones en los casos de abstención.

**Archivos.** `har/tests/test_pocket_spec_table.py`, `test_pocket_synthetic.py`, `test_pocket_metamorphic.py`, `test_pocket_catalogue.py`, `har/pocket_filter/testdata/`.

### R05 — Dataset etiquetado y batch runner (P0)

**Problema.** Sin FP/TP reales etiquetados no se puede saber si el filtro reduce alarmas o solo aumenta `INDETERMINADO` (§7).

**Enfoque.**
- **Manifiesto versionado** (csv): `event_id, camera_id, session_id, track_id, npy, timestamps, mask, clip, t_start, t_end, label {TP_bolsillo, TP_otro_destino, FP_cotidiano, ambiguo}, escenario, observabilidad {gesto_completo_visible, oclusión_muñeca, vista}, hipótesis_alternativas, split {calib, valid, test_reservado}, anotadores`.
- **Protocolo de anotación:** ¿hubo aproximación visible a bolsillo?, ¿robo confirmado?, ¿oclusión?, ¿gesto completo visible? Dos anotadores en TP y ambiguos, con adjudicación. Nunca usar la salida del modelo de robo como etiqueta.
- **Batch runner** (`har/batch_evaluate_pocket.py manifest.csv`): ejecuta legacy, proximidad sola (`d ≤ r`), `ignacio-obs-1`, `ignacio-obs-2` y la ablación Opción D sobre los mismos eventos. Exporta csv por evento, JSON de evidencia, matriz de confusión ternaria (resultado × etiqueta), tablas por cámara/escenario y reproducibilidad (misma config → mismo hash de resultados).
- Si faltan TP reales: grabar un **catálogo controlado** (varios sujetos × ángulos de cámara × gestos × velocidades, con y sin objeto). Sirve para regresión de geometría, **no** para certificar recall de hurtos reales.

**Alternativas descartadas.** Ajustar sobre uno o dos clips; mezclar calibración y cámaras de test; split aleatorio de frames del mismo vídeo.

**Métrica.** Informe automático por lote; reproducibilidad 100 %; matriz por cámara y escenario.

**Archivos.** `har/batch_evaluate_pocket.py`, `har/pocket_filter/manifest.py` (esquema + validación), `docs/protocolo_anotacion_pocket.md`.

### R06 — Visualización diagnóstica (P0)

**Problema.** Calibrar sin ver `B_L/B_R`, `r/R` y `d(t)` es ciego, y un falso `COMPATIBLE` no es explicable (§6.6, §12.2-G).

**Enfoque.** `har/visualize_pocket.py` (o extender `visualize_sup.py`):
- Vídeo con overlay de esqueleto, centros de bolsillo, círculos `r` y `R`, trayectoria de muñeca y color por estado.
- Panel con `d(t)` de las 4 parejas, marcas `t_origen`, `t_entrada`, `t_retirada`, ventana sombreada y **tira de símbolos F/M/N/U** por pareja.
- Frames no observados con su causa (máscara / conf / hold-last / identidad).
- Distinguir visualmente evidencia original y gráficos derivados (la mediana-3 solo punteada).
- Resultado global, motivo, `filter_version` y `config_hash` impresos en el frame.

**Alternativas descartadas.** Solo esqueleto; solo el JSON (oculta errores de proyección o de C1); suavizar para decidir.

**Métrica.** Tiempo de un humano para diagnosticar un falso `COMPATIBLE`/`INCOMPATIBLE`; % de decisiones que los revisores consideran justificadas.

**Archivos.** `har/visualize_pocket.py`, integrable desde `batch_evaluate_pocket.py --render`.

### R07 — Contrato JSON, política pura y persistencia (P0)

**Problema.** Un booleano no representa la incertidumbre y puede llevar a descartes peligrosos (§1.2, §5.2).

**Enfoque.**
- **Esquema JSON versionado** (`pocket_filter/schema/eval_pocket_output.schema.json`, `schema_version`) con evidencia completa; ver §7.2.
- **Módulo `policy.py` puro**: entradas (resultado, flags de calidad, cámara, versión, hipótesis abiertas, modo) → `accion_propuesta` y `accion_aplicada`. Tabla de verdad probada: para todo input con `resultado ≠ INCOMPATIBLE`, `accion_aplicada ≠ CERRAR`. Errores o timeouts se persisten como `ERROR` y la política los trata como `INDETERMINADO`.
- **Persistencia idempotente** por `camera_id + session_id + event_id + filter_version + config_hash`; nunca sobrescribir resultados de versiones anteriores.
- `hand_to_body` en `actions.json` solo si `COMPATIBLE`.

**Alternativas descartadas.** Guardar solo booleano o solo score; sobrescribir al cambiar config; derivar el descarte de `not hand_to_body`.

**Métrica.** 100 % de candidatos con registro o error explícito; reproducibilidad; 0 casos de `INDETERMINADO`/`ERROR` descartados en la tabla de verdad.

**Archivos.** `pocket_filter/policy.py`, `schema/`, serializador en `evaluate.py`; Kanvis: almacenamiento por `event_id`, esquema de review.

### R08 — Integración Orchestrator en modo observación (P0, secuenciada)

**Problema.** El Orchestrator sigue consumiendo `_hand_to_body` legacy y no registra evaluaciones ternarias (§1.2, §9.1).

**Enfoque.** Integración paralela, no sustitutiva (Opción C):
1. C2 abre candidato → el Orchestrator ensambla la ventana (R02) → ejecuta `pocket_filter` → persiste JSON → la política vigente sigue decidiendo. `accion_aplicada = null`.
2. Ejecución fuera del camino crítico, con timeout (p. ej. 50 ms): al expirar, registrar `ERROR` y conservar la alerta.
3. Flags: `POCKET_FILTER_MODE = off | observacion | activo`, allowlist por cámara, lista de versiones aprobadas y kill-switch.
4. Observabilidad: contadores por `resultado/motivo/cámara`, histograma de latencia, errores y ventanas incompletas.
5. Gate SAM3: en observación solo se registra el contrafactual "habría omitido SAM3"; no se bloquea.
6. Revisión humana: mostrar resultado, motivo, evidencia y enlace al overlay. Los veredictos de revisión (`event_id`) alimentan automáticamente el dataset (R05).
7. `har_premodel` no cambia su semántica; no se mezcla con el descarte del filtro.

**Alternativas descartadas.** Sustituir `_hand_to_body` en todas las llamadas; hacer skip de motion antes de calibrar; usar `INDETERMINADO` como descarte por defecto; invocar el filtro en chunks sin alarma (cómputo sin beneficio).

**Métrica.** Latencia p50/p95 < 10 ms (solo filtro, CPU); 100 % de candidatos con JSON o error explícito; 0 descartes en observación; FP potencialmente evitables vs TP potencialmente afectados, por cámara y día.

**Archivos.** `kanvis-orchestrator/app/application/har_robbery_review.py` (punto de invocación), `har_premodel.py` (sin cambios de semántica), gate SAM3, almacenamiento y UI de review. `har/har_actions/detect.py` **sin cambios**.

### R09 — Calibración restringida por recall y YAML versionado (P1)

**Problema.** Los parámetros son los teóricos de la spec y están en código (§6.3, §11.4). Los nuevos (R01–R03) también necesitan valores.

**Enfoque.**
1. Config YAML versionada (`pocket_filter/configs/ignacio-obs-2.yaml`) con esquema validado, unidades en segundos y `config_hash` en cada salida.
2. Rejilla sobre: `offset` de bolsillo (0,00/0,08/0,15), `r` (0,25/0,30/0,35), `R` (0,50/0,55/0,65), `Δ` (0,15/0,20/0,25), ventana de origen (1,0/1,5/2,0 s), `min_torso_px` (32/40/56), vista lateral, y los nuevos: `k` de margen, `h` de histéresis, `τ_contacto`, `g_max`, `c_min`, `window_margin_s`. Evaluar cada punto es barato (ms por evento).
3. Objetivo: maximizar `INCOMPATIBLE` entre FP elegibles **sujeto a** 0 TP perdidos en el set de desarrollo **y** en leave-one-camera-out.
4. Elegir un **óptimo robusto**, no el argmax: entre las configuraciones factibles, la que mantiene la restricción en todos sus vecinos (±1 paso por parámetro). Registrar todas las corridas (semilla fija).
5. TPE/Bayesian solo si la rejilla no basta y el dataset lo justifica.
6. Evaluar **una sola vez** en cámaras reservadas; no volver a ajustar sobre ellas.

**Alternativas descartadas.** Ajuste a ojo o mirando solo FP; optimizar accuracy global (el desbalance FP/TP engaña); elegir la configuración que más `INCOMPATIBLE` produce sin restricción de recall; ML end-to-end (rompe auditabilidad).

**Métrica.** TP perdidos = 0 (restricción); FP evitados entre los elegibles; ratio `INDETERMINADO`; estabilidad entre cámaras; IC del recall (§6.4).

**Archivos.** `pocket_filter/config.py`, `configs/ignacio-obs-2.yaml`, `har/calibrate_pocket.py`, informe en `docs/`.

### R10 — Geometría: ancla, escala y calidad de vista (P1)

**Problema.** Ancla de bolsillo fija, `L` instantáneo ruidoso, vista lateral binaria y escala solo en X pueden crear zona gris o falsos (§11.4, §12.2-A).

**Enfoque (cada punto es una ablación evaluada en el batch):**
1. **Ancla con offset lateral:** `B_j = cadera_j + a·L·u + b·L·e_ext_j` (`e_ext_j` = unitario lateral hacia fuera del lado `j`); rejilla `a ∈ {0; 0,08; 0,15}`, `b ∈ {0; 0,05; 0,10}`.
2. **`L` robusto por track:** comparar `L` instantáneo contra la mediana de `L` en frames válidos de la ventana. Si `L_inst/L_ref < 0,75` → marcar "agachado/inclinado" y `INDETERMINADO`. A/B de usar `L_ref` para normalizar `d` (reduce jitter) frente a `L_inst`.
3. **Vista graduada:** `ρ = ancho_hombros/L` con tres zonas (apta ≥ 0,35 / degradada 0,25–0,35 / no apta < 0,25, valores a calibrar). En zona degradada, `σ` se infla en vez de bloquear.
4. **`σ` por confianza:** `σ = σ₀ / max(conf, c_floor)` para muñeca, cadera `j` y torso. Cubre el bolsillo del lado ocluido sin una regla ad hoc.
5. **Píxeles reales** siempre que haya `W`, `H` y letterbox (R01).

**Alternativas descartadas.** Más parámetros sin ablación; bloquear toda vista no frontal (mata el recall útil de 3/4); exigir siempre el codo.

**Métrica.** Para cada ablación: Δ en `INCOMPATIBLE` de FP elegibles, Δ en TP perdidos (debe ser 0), Δ en `INDETERMINADO` por motivo.

**Archivos.** `pocket_filter/geometry.py`, `validity.py`, `config.py`.

### R11 — Fusión de parejas y multi-episodio (P1)

**Problema.** Varias parejas pueden ser `COMPATIBLE` a la vez (incluidas contralaterales) y solo se reporta el primer episodio (§11.5–11.6).

**Enfoque.** Sin vetar los cruces legítimos y sin exigir codo:
1. Coordenada lateral en marco corporal: `λ(t) = (muñeca − H_medio)·e_lat / L`, con `e_lat = (cadera_D − cadera_I)/‖·‖`.
2. **Regla de cruce:** una pareja contralateral (muñeca `i` → bolsillo `j ≠ i`) solo es `COMPATIBLE` si `λ` de la muñeca cruza la línea media hacia el lado `j` durante la aproximación y, en la entrada, `d_ij ≤ d_ii − margen`. Si no, se degrada a `INDETERMINADO` (`CRUCE_AMBIGUO`).
3. Degradar `COMPATIBLE → INDETERMINADO` **no pierde recall** (ambos conservan la alerta); solo mejora la precisión de la evidencia positiva (H2).
4. Ranking determinista de la pareja ganadora: ipsilateral > contralateral; luego mayor `Δd`; luego menor `d_min`; luego `t_entrada` más temprano; luego orden fijo de parejas.
5. Conservar **todas** las parejas y **todos** los episodios en la evidencia. Episodios separados por retirada.
6. `INCOMPATIBLE` global exige las 4 parejas `INCOMPATIBLE` y datos íntegros; identidad rota o cobertura insuficiente → `INDETERMINADO` global.

**Alternativas descartadas.** "Cualquier pareja gana"; veto duro a muñeca contralateral (FN en cruzar la mano al bolsillo opuesto, §4 catálogo); exigir codo universal (degrada recall por oclusión); tomar solo el primer `COMPATIBLE`.

**Métrica.** Tasa de `COMPATIBLE` espurios en balanceo/cruce; recall de cruces legítimos (sin pérdida); episodios recuperados; ratio `INDETERMINADO` por ambigüedad geométrica.

**Archivos.** `pocket_filter/fuse.py` (nuevo), `evaluate.py`, `reasons.py`; tests de cruces, zoom y espejo.

### R12 — Cierre corporal `INCOMPATIBLE_CUERPO` (P1, técnica nueva)

**Problema.** Si el Orchestrator siempre trata pecho/chaqueta/mochila como hipótesis abierta, un `INCOMPATIBLE` de solo bolsillo nunca cierra nada (§0.5, §4 Opción E, §8). El valor del filtro puede quedar en cero por política.

**Enfoque.** Cerrar de una vez todas las ocultaciones que requieren que la mano llegue al cuerpo, sin necesidad de un detector positivo de pecho:
- Zona corporal `Z` = polígono hombros–caderas inflado un margen `m` (inicial 0,15 L, a calibrar) ∪ discos de radio `r` en torno a `B_L` y `B_R`.
- `INCOMPATIBLE_CUERPO`: con cobertura completa en la ventana (R02), ninguna muñeca observada entra en `Z` con margen `kσ`.
- Es **más estricto** que `INCOMPATIBLE` de bolsillo (más difícil de cumplir), así que no puede añadir pérdida de recall. El lado `COMPATIBLE` no cambia.
- Sigue abierta solo la hipótesis mochila/bolsa/espalda. Si el equipo no acepta cerrarla, la política activa debe exigir además una señal externa para ese caso.

**Alternativas descartadas.** Multi-destino positivo completo (Opción E) en esta fase: más parámetros y se solapa con "llevar producto al pecho" cotidiano; tratar pecho siempre como abierto (anula el filtro).

**Métrica.** % de FP elegibles que pasan de `INDETERMINADO` a `INCOMPATIBLE_CUERPO`; TP perdidos = 0 en dev y leave-one-camera-out; hipótesis cerradas sin efecto final.

**Archivos.** `pocket_filter/geometry.py`, `fuse.py`, `policy.py`, `reasons.py`.

### R13 — Validación por cámara y escenario reservados (P1)

**Problema.** Un filtro geométrico se sobreajusta a cámaras, perspectivas, escalas y calidad de pose (§7.1).

**Enfoque.** Tres particiones por cámara/sesión (calibración, validación, test reservado), nunca por frames de un mismo vídeo. Estratificar por vista (frontal, 3/4, lateral), distancia, oclusión parcial, cesta/bolsa, producto en mano, balanceo, ajuste de ropa y ocultación confirmada. Leave-one-camera-out durante la calibración. Informe con IC por métrica y desglose por cámara; un agregado que oculte degradación en una cámara es No-Go.

**Alternativas descartadas.** Split aleatorio; validar en la misma sesión usada para ajustar; resultado global sin desglose.

**Métrica.** Recall por escenario; FP evitados por escenario; `INDETERMINADO` por motivo y cámara; variabilidad entre cámaras.

**Archivos.** Manifiesto, `batch_evaluate_pocket.py`, informe de validación.

### R14–R15 (P2, solo planificar)

- **R14 Multi-destino v2:** pecho/torso como destino positivo, hipótesis mochila explícita en la fusión. Si solo se evaluó bolsillo y hay hipótesis alternativa abierta, no se concluye `INCOMPATIBLE` global.
- **R15 Features y modelos:** orden experimental: (1) velocidad radial robusta con pendiente de Theil–Sen sobre `d` crudo, como refuerzo; (2) extensión de brazo normalizada por longitud de brazo; (3) ángulos de codo auxiliares; (4) DTW como diagnóstico; (5) candidato "reposo estático fuera de bolsillo" evaluado solo offline; (6) clasificador ligero únicamente si las reglas no cierran la frontera y con validación propia. El ternario no se sustituye por una caja negra.

### 5.1 Respuestas a las 7 preguntas de §12.3

| # | Pregunta | Respuesta |
|---|----------|-----------|
| 1 | ¿Píxel real o norm + aspect en X? | Píxeles reales siempre que haya `W`, `H` y letterbox/espejo. Los ratios `d/L` solo equivalen con píxeles cuadrados y sin letterbox/crop. Si no hay metadatos, `geometria_aproximada=true` y la política no descarta (R01, R10). |
| 2 | ¿Fusionar `valid_mask (T,)` con conf por KP? | AND de máscara de frame, KP finito y `conf ≥ c_min`, más hold-last e identidad. La cobertura se mide en tiempo dentro de la ventana, no en nº de frames (R01, R02). |
| 3 | ¿Regla simple para `COMPATIBLE` contralateral espurio? | Regla de cruce por coordenada lateral `λ` más ranking ipsilateral > contralateral; sin exigir codo; degradar a `INDETERMINADO` (R11). |
| 4 | ¿`valid_ratio` o cobertura adaptativa por cadencia? | Cobertura temporal (`c_min`) + `g_max` acotado físicamente + una muestra por tercio de ventana. El `valid_ratio` fijo desaparece (R02). |
| 5 | ¿`INCOMPATIBLE` con `valid.all()` es demasiado estricto? | Sí. Se sustituye por cobertura y gaps dentro de la ventana del evento, sin exigir el 100 % de frames, pero sin hueco que permita ocultar una aproximación (R02). |
| 6 | ¿Features baratas para pecho vs bolsillo? | Probar altura axial de la muñeca sobre el eje torso y la extensión de brazo como separadores. Son P2 (R15); antes, `INCOMPATIBLE_CUERPO` evita necesitarlos (R12). |
| 7 | ¿Sustituir `_hand_to_body` o llamar `pocket_filter` post-alarma? | Post-alarma, en observación, sin tocar `detect.py` (R08, I6). |

---

## 6. Plan de validación y calibración

### 6.1 Datasets necesarios

| Dataset | Contenido | Uso |
|---------|-----------|-----|
| Sintéticos (12 Hz y otras cadencias) | Tabla de R04 + metamórficos | Contrato y regresión (CI) |
| Catálogo controlado | Gestos reproducibles: guardar en bolsillo, cruzar mano, móvil, balanceo, pecho, cesta, mochila, agacharse, dos personas | Geometría y límites; **no** certifica recall real |
| FP reales de tienda | Motion disparó y la revisión humana dijo "no robo", con `event_id` y ventana | FP evitados y composición de FP |
| TP confirmados | Hurtos reales, separando ocultación en bolsillo de otros destinos | Recall y TP perdidos |
| Indeterminados | Oclusión, lateral, ID swap, pose parcial, chunks cortos | Verificar abstención correcta |
| Cámaras reservadas | Cámaras/sesiones nunca usadas para ajustar | Generalización; una sola evaluación |

### 6.2 Comparativas (mismos eventos, resto de C3 congelado)

| Sistema | Qué se aprende |
|---------|----------------|
| Legacy `_hand_to_body` | Baseline actual |
| Proximidad sola (`d ≤ r`) | Cuánto aporta exigir trayectoria |
| `ignacio-obs-1` (prototipo actual) | Punto de partida |
| `ignacio-obs-2` (endurecido) | Efecto de R01–R03, R10–R12 |
| Opción D (legacy mejorado, opcional) | Valor de un cambio mínimo |
| Ablaciones de `ignacio-obs-2` | Con/sin `valid_mask`; píxeles vs aspect; `L` instantáneo vs `L_ref`; `k=0` vs `k>0`; con/sin cierre corporal |

### 6.3 Criterios go / no-go

**Go a modo observación (gate técnico):** tests de R04 al 100 %; contrato ternario y tabla de verdad de la política correctos; persistencia por `event_id`; latencia p95 < 10 ms; 0 crashes en batch real; no se modifica la política de alertas.

**Go a descarte automático condicionado** (todo a la vez):
- Pérdida de TP por debajo del techo acordado por producto, con el **límite superior del IC** (no el recall puntual) por debajo de ese techo.
- Evaluación independiente por cámara y escenario en cámaras reservadas.
- Revisión en vídeo de cada TP potencialmente perdido.
- Evidencia reproducible de cada `INCOMPATIBLE`.
- Tasa de errores de datos baja y desglosada.
- Configuración congelada y versionada; aprobación del responsable de producto/operación.
- Cámaras en allowlist y rollback inmediato a `observacion` ante cualquier retroceso.

**No-Go (cualquiera):** un `INDETERMINADO` o `ERROR` se convierte en descarte; falta de máscara o timestamp tratada como evidencia negativa; recall por debajo del objetivo; agregados que ocultan una cámara degradada; resultado no reproducible.

### 6.4 Tamaño de muestra para el gate de recall

Con **0 TP perdidos** de `n` TP reales, el límite superior unilateral al 95 % de la tasa de pérdida es `1 − 0,05^(1/n)`:

| TP reales sin ninguna pérdida | Pérdida real máxima (95 %) |
|-------------------------------|----------------------------|
| 20 | 13,9 % |
| 50 | 5,8 % |
| 100 | 3,0 % |
| 150 | 2,0 % |
| 300 | 1,0 % |

Cada TP perdido aumenta ese límite, así que el techo acordado debe fijarse **antes** de medir y expresarse como límite del IC (Clopper–Pearson o Wilson). Los 20 clips de §12.6 son un smoke test, no una campaña de aceptación.

### 6.5 Métricas a reportar (por cámara / día)

| Métrica | Definición | Objetivo |
|---------|------------|----------|
| Recall TP conservados | TP `COMPATIBLE` o `INDETERMINADO` / total TP | ≥ umbral acordado; IC reportado |
| TP perdidos | TP marcados `INCOMPATIBLE` | ≈ 0; revisión 1 a 1 |
| FP elegibles | FP con cobertura suficiente, vista apta y sin hipótesis abiertas | Denominador de FP evitados |
| FP evitados | FP elegibles clasificados `INCOMPATIBLE` | Maximizar |
| % por estado | `COMPATIBLE` / `INCOMPATIBLE` / `INDETERMINADO` | Informativo |
| `INDETERMINADO` por motivo | `EVIDENCIA_PARCIAL`, `VISTA_NO_APTA`, `ZONA_GRIS`, `CRUCE_AMBIGUO`, `IDENTIDAD`, `ERROR`… | Localizar cuellos de botella de C1 |
| Precisión de `COMPATIBLE` | `COMPATIBLE` con aproximación real según anotación | Subir (R11) |
| Hipótesis cerradas sin efecto final | Se cerró bolsillo pero había otra hipótesis abierta | Auditar política |
| Latencia p50/p95 | ms por evento, CPU | p95 < 10 ms |
| Eventos sin evaluación | Errores, metadatos ausentes, asociación fallida | Minimizar |
| Reproducibilidad | Mismo evento y config → mismo resultado | 100 % |
| Contrafactual SAM3 | Llamadas a SAM3 que se habrían evitado | Informativo (ahorro) |

La métrica clave del piloto no es cuántos eventos son `INCOMPATIBLE`, sino **cuántas alertas FP realmente evitables se eliminan con el recall de hurtos intacto y absteniéndose cuando falta evidencia.**

---

## 7. Integración Kanvis

### 7.1 Cambios en Orchestrator

| Componente | Cambio |
|------------|--------|
| `har_premodel.py` | Sin cambios de semántica. HAR cotidiano ligero puede seguir como señal auxiliar. No mezclar con el descarte del filtro. |
| `har_robbery_review.py` | Tras candidato motion, ensamblar la ventana `[t_start − 1 s, t_end + 1 s]` con poses originales sin suavizar, ejecutar `pocket_filter` y persistir. |
| Gate SAM3 | Observación: registrar el contrafactual. Activo: si `INCOMPATIBLE` (o `INCOMPATIBLE_CUERPO`) con política ON y sin hipótesis abiertas → no llamar a SAM3. `INDETERMINADO` por oclusión → SAM3 puede resolver. |
| Persistencia | JSON por `event_id` + versión + `config_hash`. |
| Revisión humana | Mostrar resultado, motivo, evidencia y overlay; capturar veredicto para el dataset. |
| Configuración | Modo, allowlist por cámara, versiones aprobadas, kill-switch. |
| Observabilidad | Contadores por resultado/motivo/cámara, latencia, errores, ventanas incompletas. |
| `detect.py`, c0/c1/c5 | Sin cambios. |

### 7.2 Contrato JSON persistente (claves en español, alineadas con §6.2)

```json
{
  "schema_version": "1",
  "event_id": "event-123",
  "camera_id": "cam-04",
  "session_id": "sess-...",
  "track_id": 17,
  "filter_version": "ignacio-obs-2",
  "config_hash": "sha256:...",
  "engine": "v2",
  "modo": "observacion",
  "ventana": {"t_start": 12.0, "t_end": 17.5, "margen_s": 1.0},
  "bolsillo_izquierdo": "INCOMPATIBLE",
  "bolsillo_derecho": "INCOMPATIBLE",
  "resultado": "INCOMPATIBLE",
  "accion_propuesta": "CERRAR_HIPOTESIS_BOLSILLO",
  "accion_aplicada": null,
  "otras_hipotesis_abiertas": [],
  "calidad": {
    "cobertura": 0.98,
    "max_hueco_s": 0.17,
    "vista_apta_frac": 0.91,
    "tiempo_asumido": false,
    "geometria_aproximada": false,
    "causas_no_observado": {"mascara": 3, "conf": 1, "hold_last": 0, "identidad": 0}
  },
  "evidencia": {
    "pareja_ganadora": null,
    "episodios": [],
    "parejas": [
      {"mano": "izquierda", "lado": "izquierdo", "estado": "INCOMPATIBLE", "motivo": "LEJOS_TODO_EL_GESTO", "d_min": null}
    ]
  },
  "motivo": "LEJOS_TODO_EL_GESTO",
  "motivos_secundarios": [],
  "latencia_ms": 3.2,
  "timestamp_evaluacion": "..."
}
```

Con `COMPATIBLE`, la evidencia incluye por episodio `mano`, `lado`, `t_origen`, `d_origen`, `t_entrada`, `d_entrada`, `d_min`, `t_retirada`. El esquema final debe llevar la evidencia completa de las 4 parejas, no solo la ganadora.

### 7.3 Modos

| Modo | Calcula | Registra propuesta | Cambia la alerta |
|------|---------|--------------------|------------------|
| `off` | No | No | No |
| `observacion` | Sí | Sí | No (`accion_aplicada = null`) |
| `activo` | Sí | Sí | Solo con `INCOMPATIBLE`, cámara en allowlist, versión aprobada, calidad íntegra y sin hipótesis abiertas |

El flag de política (activar descarte) es **separado** del flag técnico de evaluación. Una nueva versión del filtro no activa una política nueva automáticamente.

### 7.4 Separación de responsabilidades

- `actions.json`: `hand_to_body` solo cuando `COMPATIBLE`.
- `eval_pocket_output.json`: ternario y evidencia completa.
- Orchestrator: decide con el contrato ternario, nunca con el booleano.
- HAR de pasillo: conserva su semántica hasta una decisión independiente.

---

## 8. Roadmap sugerido

Los hitos se cierran por criterio, no por fecha; las semanas son una estimación.

| Hito | Contenido | Resultado |
|------|-----------|-----------|
| **0 — Datos (≈ 3 días, en paralelo a todo)** | R00 auditoría; manifiesto y protocolo de anotación (R05); cerrar preguntas de §10 | Informe de datos; parámetros iniciales justificados; veredicto sobre H1 y H3 |
| **1 — Prototipo endurecido (sem. 1–2)** | **PR #1:** tests-first (tabla R04 + M1) + R01 + R02 (`interval_s`). **PR #2:** R03 núcleo v2 con equivalencia a v1 | Filtro consistente con la spec; CI verde; `INCOMPATIBLE` con cobertura real |
| **2 — Herramientas de medida (sem. 3–4)** | **PR #3:** R05 batch + R06 visualización + R07 contrato/política; primera matriz de confusión; ablaciones de R10 | Informe comparativo vs legacy; gate técnico de observación |
| **3 — Observación en tienda (desde sem. 5, en paralelo)** | R08 cableado; persistencia; revisión humana alimenta el dataset; auditoría diaria de métricas | Datos reales sin riesgo operativo |
| **4 — Calibración y validación (sem. 5–7)** | R09 rejilla robusta, R10, R11, R12, R13; congelar `ignacio-obs-2.yaml`; una evaluación en cámaras reservadas | Configuración congelada e informe con IC |
| **5 — Activación condicional (≥ sem. 9)** | Solo si se cumplen los criterios de §6.3: allowlist de cámaras piloto, monitorización semanal, rollback inmediato | Reducción controlada de FP con trazabilidad |
| **6 — v2** | R14 y R15 según los TP/FN observados | Ampliación de destinos y features |

---

## 9. Riesgos residuales

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Geometría de reposo deja casi todos los FP de caminar/balanceo en `INDETERMINADO` (H1) | Media-alta | Medio (valor del filtro menor al esperado) | R00 mide antes de prometer; R10 (ancla/σ) y R12 (cierre corporal); fijar expectativa por composición de FP |
| Cadencia real de NPY muy inferior a 12 Hz (H3) | Media | Alto (abstención casi total) | R00; reglas en segundos (R03); informar a C1 del contrato mínimo |
| Ocultación en bolsillo trasero, pecho, interior de abrigo o mochila | Alta | Alto (FN si se cierra) | Hipótesis abiertas; R12 cierra solo lo corporal; R14 en v2 |
| Calidad de C1 deficiente (ID swaps, hold-last, conf baja) | Alta | Alto | R01 detecta y deriva a `INDETERMINADO`; I4 e I5 como invariantes |
| Muestra de TP insuficiente para certificar recall | Alta | Alto | §6.4: fijar techo por IC, no por recall puntual; catálogo controlado solo para regresión |
| `COMPATIBLE` espurio por balanceo o mano cruzada | Media | Medio | R11 (regla de cruce, ranking); R03 (`Δ` con margen) |
| Cesta o bolsa junto a la cadera → `COMPATIBLE` | Media | Medio | No interpretar `COMPATIBLE` como prueba de hurto; SAM3/VLM decide el objeto |
| Sacar el móvil (cerca → lejos) | Media | Bajo | Queda `INDETERMINADO`; no descarta |
| Sobreajuste de umbrales a ciertas cámaras | Media | Alto | R13: reservadas, leave-one-camera-out, óptimo robusto (R09) |
| Contaminación de c0/c1/c5 | Baja si se separa | Medio | I6: artefacto separado y flag por defecto `legacy` |
| El Orchestrator aplica descarte por booleano o ante `ERROR`/timeout | Baja si hay control; crítica si ocurre | Crítico | `policy.py` puro con tabla de verdad; `ERROR` ≡ `INDETERMINADO`; kill-switch |
| La política siempre ve "hipótesis abierta" y nunca descarta | Media | Alto (valor nulo) | R12 y decisión explícita de producto (pregunta 4) |

---

## 10. Preguntas al equipo producto / datos

1. **Techo de pérdida de TP:** ¿qué pérdida máxima aceptamos para activar descarte (0 % estricto o un valor X), entendida como límite superior del IC al 95 %? (Con 0 % hacen falta ≥ 150 TP reales sin pérdidas para certificar 2 %.)
2. **Export de C1:** ¿se exporta hoy, por evento histórico, la máscara por KP **antes** de relleno/hold-last, `pose_frame_timestamps`, `W/H`, letterbox y espejo? ¿Es `valid_mask (T,)` exactamente "frame no exportable"?
3. **Volumen de etiquetas:** ¿cuántos TP reales confirmados de ocultación en bolsillo y cuántos FP con `event_id` y ventana exacta existen ya, o hay que construirlos y grabar un catálogo controlado?
4. **Cierre de hipótesis:** ¿aceptáis que el descarte v1 cierre bolsillo + torso (`INCOMPATIBLE_CUERPO`) dejando abierta solo mochila/bolsa/espalda? ¿Tiene hoy el Orchestrator alguna señal de "hipótesis abierta"?
5. **Consumidores externos:** ¿algún consumidor de `actions.json` necesita que `hand_to_body` cambie de motor, o podemos limitar v1 al JSON ternario post-alarma?

---

## 11. Conclusión

**Decisión recomendada:** endurecer `pocket_filter` como módulo determinista ternario (datos válidos, ventana de evento, núcleo con márgenes y tests metamórficos), medir con datos reales antes de calibrar, integrarlo en el Orchestrator en modo observación y reservar el descarte exclusivamente para `INCOMPATIBLE` con evidencia estadística de que no se pierden hurtos.

**Siguiente paso concreto:** abrir el **PR #1 = tests-first (tabla R04 + metamórfico M1) + R01 + R02** sobre `har/pocket_filter/`, sin tocar umbrales, `detect.py` ni Orchestrator, y lanzar en paralelo R00 (`tools/audit_poses.py`) sin cambios de código en el filtro.

**Definición de hecho del PR #1:**
- [ ] `pytest har/tests -q` verde (los casos que fallan con v1 quedan como `xfail` con motivo).
- [ ] `valid_mask (T,)` y `(T,8)` soportados; causas de no-observado en la evidencia.
- [ ] `interval_s` opcional en `evaluate_pocket` y `--interval-s` en `har2_test.py`.
- [ ] Hold-last, huecos > `max_time_gap_factor·P` y cambios de `track_id` cortan tramo.
- [ ] `INCOMPATIBLE` sustituye `valid.all()` por cobertura y gaps dentro de la ventana.
- [ ] Propiedad I5 (M1) y test de H4 incluidos.
- [ ] Documentación de casos límite actualizada (`documentation/pocket_approach.txt`).

**Criterio central de éxito:** reducir las falsas alarmas realmente evitables sin convertir incertidumbre en ausencia de hurto.
