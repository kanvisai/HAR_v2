# Propuesta de soluciones — Hand-to-body / filtro de bolsillo (Kanvis HAR_v2)

**Fecha:** 2026-11-21
**Fuentes leídas completas:** `HAND_TO_BODY_SOLUCIONES.md` (§0–§14, leído primero) y `HAND_TO_BODY_ENGINEERING_STATUS.md` (§1–§10 y apéndices A–I).
**Respuestas externas comparadas:** tres respuestas de otras IAs a los mismos documentos, que cito como **Respuesta 1** (propuesta técnica con ternario y evento), **Respuesta 2** (sigue la plantilla §14, recomendaciones R1–R10) y **Respuesta 3** (análisis de un RFC de arquitectura). Las cito por orden de llegada; no sé cuál es de cada IA y no hace falta para el análisis.
**Alcance y límites:** es un análisis de los documentos. No he inspeccionado el código del repo ni ejecutado nada. Las cifras salen de los handoffs; las que calculo yo están marcadas como *cálculo propio*, y lo que es inferencia mía está marcado como *inferencia* para que lo verifiques.
**Formato:** sigo la plantilla de SOLUCIONES §14 (secciones 1–11) y añado la sección 12 con la comparativa de respuestas y el veredicto.

---

## 1. Resumen ejecutivo

- **Qué es Kanvis y qué se resuelve.** Kanvis detecta hurtos en retail con un modelo Motion (C2) sobre chunks de ~8 s. El filtro `hand_to_body` / `pocket_filter` debe, solo con el esqueleto de 8 keypoints, confirmar, mandar a revisión o suprimir cada candidato de Motion, sin etiqueta humana en runtime.
- **Estado.** El prototipo está medido y congelado (G+H: ~84 % de recall en cat. 6, 434 FP en cat. 0, ~9,4 %). Los propios documentos dicen que es un compromiso, no una solución, y que las reglas sobre las mismas features han tocado techo (STATUS §Executive summary, §8).
- **Hallazgo central de este análisis.** El número que más se cita (9,4 % de FP en cat. 0) es el más favorable de la tabla. Agregando todas las categorías etiquetadas como negativas (STATUS Apéndice E), el "FP" del baseline es del **35,0 %** (6 641 de 18 970) y del **66,1 %** en las categorías distintas de 0 y 5 *(cálculo propio)*. Además, el árbol de decisión de producción (STATUS Apéndice G) descarta por su rama `else` justo lo que el contrato ternario prohíbe descartar (INDETERMINADO). Y nadie ha medido todavía cuánta carga mueve el tramo `raw → revisión`.
- **Recomendación principal.** No buscar otro umbral. Construir una **cascada de decisión por evento con coste explícito**: (1) corregir la semántica del árbol (INDETERMINADO nunca se descarta), (2) evaluar sobre el intervalo/score de Motion y en la distribución real de candidatos Motion, y (3) sustituir la puerta única por una **fusión aprendida pequeña, calibrada y monótona** que decida los umbrales por coste, dejando al filtro geométrico como proveedor de evidencia auditable.
- **Orden de trabajo.** Antes de escribir código nuevo, una **Fase 0 de 1–2 días** sobre los CSV que ya existen (`gate_temporal.csv`), porque decide qué rama del plan merece la pena.

**Veredicto sobre las otras respuestas (detalle en §12):** la mejor base conceptual es la Respuesta 1 (disciplina ternaria, ventana de evento, protocolo de datos, coste por 1 000 candidatos). La Respuesta 2 aporta lo más concreto a nivel de ficheros y la fusión aprendida sobre el CSV. La Respuesta 3 no trata este problema. Mi propuesta combina 1 y 2 y corrige lo que ambas pasan por alto.

---

## 2. Comprensión del contexto

### 2.1 Pipeline Kanvis (C1 → C2 → C3) y dónde encaja el filtro

| Capa | Produce | Papel respecto al filtro |
|---|---|---|
| **C1** | Tracks, chunks ~8 s, `poses_full.npy` (8 KP, COCO 5–12, `(T,8,3)` con confianza), `valid_mask (T,)`, timestamps | Entrada. El filtro debe usar poses originales, sin suavizar (SOLUCIONES §5.3) |
| **C2 Motion** | Scores de robo (p41, p48, ensembles) por chunk | Abre el candidato. Es la **única fuente de información que el filtro aún no usa** |
| **C3 Orchestrator** | `har_premodel`, `har_robbery_review`, `har_sam3_gate`, política de aviso | Consume las señales y decide |

El filtro va **después** del candidato Motion y **antes** del aviso fuerte o de SAM3 (SOLUCIONES §2.1, §4 Opción C). El HAR de pasillo (c0/c1/c5) debe quedar separado del filtro de alerta (SOLUCIONES §6.5, §8.6).

### 2.2 Qué es "acción rara" en este alcance

No es rareza estadística. Es **cinemática de ocultación visible en el esqueleto**: trayectoria ordenada lejos → cerca de una muñeca hacia una zona de ocultación (bolsillo lateral en v1; pecho/chaqueta/mochila en v2+). Los cotidianos (caminar con braceo, parar a mirar, llevar producto visible, ajustarse la ropa) sirven para *explicar* FP, no para descartar por nombre de actividad (SOLUCIONES §0.4, §5.3).

Límite físico que conviene tener presente: con 8 keypoints a ~12,5 Hz, una muñeca cerca de la cadera es geométricamente igual si guarda un producto, si saca el móvil o si se recoloca el pantalón. Ninguna regla de distancia separa eso por sí sola. La separación sale de **contexto** (Motion, secuencia previa) o de **otra modalidad** (vídeo, vía SAM3).

### 2.3 Política: cuándo bajar alerta por INCOMPATIBLE y cuándo mantener por INDETERMINADO

Doctrina ya fijada (SOLUCIONES §2.2, §5.2): *ausencia de COMPATIBLE ≠ INCOMPATIBLE*. Solo INCOMPATIBLE con cobertura válida y sin hipótesis alternativa abierta puede suprimir.

**Contradicción detectada entre doctrina y producción.** El árbol de STATUS Apéndice G es:

```text
if motion_suppress:     DISCARD
elif hand_to_body:      CONFIRM
elif hand_to_body_raw:  REVIEW_SAM
else:                   DISCARD
```

`hand_to_body_raw` es "Pocket COMPATIBLE antes de la fusión estricta" (STATUS Apéndice B, col. 14). Por tanto `raw=false` agrupa **INDETERMINADO e INCOMPATIBLE**, y la rama `else: DISCARD` descarta ambos. Además `motion_suppress` incluye explícitamente el caso "no pocket" (Apéndice G). Eso es "no se vio ⇒ no ocurrió" implementado como política, lo contrario de SOLUCIONES §0.3 y §0.7 (*Prudencia*). *(Inferencia a partir de los documentos; verificar en `action_policy.py` y en `har_premodel`.)*

La Respuesta 2 (R2) propone implementar "exactamente" este árbol y, en su §7, dice que no hay que mapear "ausencia de clear" a descarte. Las dos frases no se pueden cumplir a la vez.

---

## 3. Diagnóstico

### 3.1 Fortalezas (lo que ya acierta)

- Geometría de bolsillo proyectado y distancias normalizadas por torso; trayectoria origen → entrada con `delta_min`; salida ternaria con evidencia (SOLUCIONES §2.3–2.4, §11.4).
- Señales ya exportadas y útiles para fusión: `hand_to_body`, `hand_to_body_raw`, `motion_suppress`, `hand_to_body_confidence`, `motivo`, `ctx_walk_frac`, `ctx_stop_frac`, `htb_tier`, `filter_fp_safe` (STATUS Apéndice B).
- Línea base reproducible y congelada (`BASELINE_PRODUCCION_H.json`), batch y auditoría (`batch_data_result.py`, `audit_pocket_csv.py`, `audit_htb_metrics.py`).
- Disciplina de experimentación ya establecida (STATUS Apéndice F): un cambio por experimento, CSV con nombre fijo, rollback por copia de config.
- Ya se midió que la política por motivo funciona mejor que los cortes globales (STATUS §8.5).

### 3.2 Debilidades del prototipo — y cuáles ya podrían estar cerradas

SOLUCIONES §11.5–11.6 describe el prototipo **antes** de la política G+H. El JSON congelado (STATUS Apéndice C.1) ya contiene claves que sugieren que parte de esas brechas se cerró:

| Brecha citada en SOLUCIONES §11.5 / en las respuestas | Indicio de que ya existe (STATUS C.1, Apéndice I) |
|---|---|
| `valid_ratio` fijo 0,7 | `min_valid_ratio_short_clip` 0,35, `min_valid_ratio_long_clip` 0,65, `short_clip_s`, `long_clip_s` |
| Pose congelada / hold-last | `frozen_repeat_frames`, `frozen_torso_eps` |
| Saltos de identidad / escala | `identity_hip_jump_L`, `scale_jump_ratio` |
| Pareja contralateral sin criterio | `prefer_ipsilateral`, `contralateral_delta_bonus` |
| Sin tests | Suites `test_pocket_filter.py`, `test_action_policy.py`, `test_temporal_eval.py`, etc. |
| Sin selección de zona real | `pocket_lateral_offset`, `deep_contact_r`, `sustained_contact_s` |

Las Respuestas 1 y 2 repiten esas brechas como abiertas porque solo leen la prosa de §11.5. **Antes de planificar P0/P1 hay que cruzar cada brecha con el código actual.** Probablemente siguen abiertas: `valid_mask (T,)` de Kanvis, letterbox/espejo, API con intervalo de evento, episodios múltiples y visualización.

### 3.3 Hallazgos del análisis de cifras

**H1 — El FP "bueno" es el mejor caso, no el típico.** *(Cálculo propio sobre STATUS Apéndice E, baseline G+H, columna clear.)*

| Cat | Etiquetados (TN+FP) | FP | Tasa de FP |
|---|---:|---:|---:|
| 0 | 4 633 | 434 | 9,4 % |
| 5 | 5 602 | 431 | 7,7 % |
| 1 | 611* | 289 | 47,3 % |
| 4 | 2 403 | 1 301 | 54,1 % |
| 7 | 591 | 331 | 56,0 % |
| 9 | 914 | 513 | 56,1 % |
| 13 | 80 | 59 | 73,8 % |
| 3 | 2 182 | 1 723 | 79,0 % |
| 2 | 1 954 | 1 560 | 79,8 % |
| **Todas las negativas** | **18 970** | **6 641** | **35,0 %** |
| Negativas salvo 0 y 5 | 8 735 | 5 776 | 66,1 % |

\* La cat. 1 aparece con 758 etiquetados pero TN+FP suma 611; faltan 147 que la tabla no clasifica (¿SKIP/ERROR?).

SOLUCIONES §0ter.4 reconoce "peor en otras cats cotidianas", pero no lo cuantifica en el relato principal, y las tres respuestas hablan solo de cat. 0. Hay dos lecturas posibles y la decisión de producto depende de cuál sea la correcta: (a) son comportamientos cotidianos donde el filtro falla de verdad, o (b) son categorías donde llevar la mano al cuerpo es legítimo y la etiqueta "esperado false" no es una verdad de terreno por gesto. La nota "long buffers" en cat. 2 apunta a que parte del efecto puede ser de longitud de clip.

**H2 — El benchmark no es la distribución de runtime.** El filtro solo corre sobre candidatos de Motion (STATUS Apéndice G). Los clips de `data_result` están etiquetados por carpeta, no condicionados a que Motion haya disparado. Un clip de pasillo que Motion dispara es, por construcción, más parecido a un robo que un clip de pasillo cualquiera. Las tasas de FP de cat. 0 probablemente **subestiman** las de producción. *(Inferencia.)*

**H3 — `raw` casi no discrimina, así que el tramo de revisión puede inundarse.** Pocket sin política da ~94 % en cat. 6 y ~82 % de FP en cat. 0. Como `raw` es el veredicto del pocket antes de la política, su tasa en cat. 0 debería rondar el 82 % o más (la ventana temporal es un OR). El tramo `raw ∧ ¬clear → REVIEW` recibiría entonces del orden de **70 puntos porcentuales de cat. 0**. La frase de STATUS §2 ("recuperar recall con `raw` sin reabrir el 82 % de FP") es cierta solo para CONFIRM: el coste no desaparece, se traslada a SAM/humano. *(Inferencia; STATUS Apéndice D dice que la cifra de `raw` "no está archivada". Es la primera medición de la Fase 0.)*

**H4 — Falta el dato Motion en el CSV.** El esquema (STATUS Apéndice B) no tiene score ni intervalo de Motion. Las líneas de trabajo "fusión con contexto motion" (SOLUCIONES §0ter.5.1) y "fusión multi-señal" no se pueden evaluar hoy en el audit, solo diseñar.

**H5 — El OR temporal infla el FP por construcción.** Ventanas de 2,5 s con paso de 1,25 s sobre ~8 s dan unas 4–5 ventanas por chunk *(cálculo propio)*, y `hand_to_body` es el OR de todas (STATUS §3.1). Cada ventana extra es otra oportunidad de falso positivo. Eso explica mecánicamente por qué H sube el FP de 164 a 434 mientras sube el recall de 2 925 a 3 521. Evaluar una sola ventana, la del intervalo de Motion, reduce el número de "miradas" sin bajar la sensibilidad donde importa.

**H6 — `walk_gate` es una palanca de bajo rendimiento.** Cambia 57 TP por 73 FP evitados (STATUS §4, fila I): 1,28 FP por cada robo perdido. Es un microajuste, no estructural, y la elección entre activarlo o no es de coste, no de ingeniería.

**H7 — La precisión depende de un número que nadie ha dado: la prevalencia de robo entre candidatos Motion.** *(Ilustración con supuestos míos; recall conjunto cat. 6+14 = 3 855/4 593 = 83,9 %, y se asume que los negativos de producción se parecen a los etiquetados.)*

| Prevalencia de robo entre candidatos Motion | Precisión de "clear" con FP de cat. 0 (9,4 %) | Precisión con FP agregado (35,0 %) | Alertas por 1 000 candidatos (9,4 % / 35,0 %) |
|---:|---:|---:|---:|
| 2 % | 15,4 % | 4,7 % | 109 / 360 |
| 5 % | 32,0 % | 11,2 % | 131 / 375 |
| 20 % | 69,1 % | 37,5 % | 243 / 448 |

Mientras no se sepa en qué fila vive Kanvis, "9,4 %" no se puede traducir a carga operativa.

### 3.4 Riesgos si se desplegara tal cual

| Riesgo | Efecto |
|---|---|
| Rama `else: DISCARD` del árbol | FN en oclusión/lateral/pose mala; contradice la doctrina del propio proyecto |
| Tramo REVIEW sin cuantificar (H3) | Carga de SAM/humano potencialmente comparable a la del pocket puro |
| Calibrar sobre cat. 0 (H1, H2) | Umbrales optimistas que no se sostienen en tienda |
| Confianzas heurísticas (Apéndice H) usadas como probabilidad | Umbrales frágiles entre cámaras y días |
| `scores` / `legacy_veto` | Ya medido: destruyen recall en cat. 6 (STATUS §4) |
| Activar descarte por INCOMPATIBLE sin medir su tasa | Su cobertura real es desconocida (`filter_fp_safe` no se reporta en ninguna tabla) |

---

## 4. Recomendaciones priorizadas

| ID | Prio | Recomendación | Impacto esperado | Esfuerzo | Dependencias |
|---|---|---|---|---|---|
| P0-1 | P0 | **Fase 0:** análisis sobre CSV existentes (cruces `resultado`, `raw`, `filter_fp_safe`, `htb_tier`, `motion_suppress` × categoría; embudo del árbol; muestreo visual) | Decide el rumbo con coste casi nulo; confirma o refuta H1–H3 | Bajo | `gate_temporal.csv`, `audit_htb_metrics.py` |
| P0-2 | P0 | Modelo de coste y prevalencia (FP-revisión, FN-robo, coste SAM, prevalencia entre candidatos Motion) | Convierte métricas en carga operativa; fija el punto de operación | Bajo | Producto |
| P0-3 | P0 | Corregir la semántica del árbol: INDETERMINADO nunca se descarta; `accion_propuesta` ≠ `accion_aplicada`; modo observación | Elimina FN por abstención mal tratada; alinea producción con doctrina | Bajo-medio | `action_policy.py`, Orchestrator |
| P0-4 | P0 | Dataset por evento con score/intervalo de Motion unido al CSV; split por sesión/cámara; test reservado | Hace posibles la fusión y la calibración sin fuga | Medio | C2, anotación |
| P0-5 | P0 | Evaluar el pocket sobre el intervalo de Motion (`interval_s`) con fallback a G+H | Menos miradas (H5), más relevancia temporal; única fuente nueva de información | Medio | C2 exporta `[t0,t1]` o frame pico |
| P1-1 | P1 | Features esqueleto baratas nuevas (periodicidad de braceo, velocidad radial, permanencia, retirada, ángulo de codo, altura de muñeca, antifase bilateral) | Separa mejor "braceo al caminar" de "aproximación deliberada" | Medio | Export de series `d(t)` |
| P1-2 | P1 | **Fusión aprendida pequeña** (logística y GBM monótono) calibrada por motivo, con abstención y umbrales por coste | Mejor candidata a mover la frontera recall/FP; reemplaza la puerta única | Medio-alto | P0-1, P0-2, P0-4 |
| P1-3 | P1 | Validez y fusión de parejas: `valid_mask (T,)`, letterbox/espejo, episodios múltiples, INCOMPATIBLE sobre "gesto delimitado" | Más INCOMPATIBLE seguros, menos INDETERMINADO espurio | Medio | Revisión previa del código (§3.2) |
| P1-4 | P1 | Visualización (`B_L/B_R`, anillos r/R, `d(t)`) y tests de la tabla sintética de la spec | Calibración con ojos; evita regresiones | Medio | — |
| P1-5 | P1 | Modo observación en Orchestrator con JSON por `event_id` (versión de filtro, config y política) | Despliegue sombra seguro; mide ahorro y FN potenciales | Medio-alto | C3 |
| P1-6 | P1 | Triage del tramo REVIEW: cola priorizada por score calibrado; SAM3 solo ahí; medir llamadas por 1 000 candidatos | Evita que REVIEW se convierta en el coste oculto | Medio | P1-2, SAM3 gate |
| P2-1 | P2 | Multi-destino v2 (pecho/torso) con hipótesis mochila/cesta abierta | Recupera ocultación no lateral | Alto | Bolsillo calibrado |
| P2-2 | P2 | Segundo canal no esquelético en REVIEW (recorte de manos con SAM/VLM) | Rompe el límite físico de §2.2 | Alto | SAM3 gate |
| P2-3 | P2 | DTW/plantillas y optimización TPE/Bayes solo como diagnóstico y con split fijo | Análisis, no decisión | Medio | Dataset estable |

---

## 5. Propuestas técnicas (detalle)

### P0-1 — Fase 0: análisis sobre los CSV existentes

- **Problema:** no se sabe cuánta carga mueve el tramo REVIEW (H3), cuántos INCOMPATIBLE reales hay, qué son las categorías con FP del 50–80 % (H1) ni cuántos FN de cat. 6 son recuperables.
- **Enfoque propuesto** (sin código nuevo, tablas dinámicas sobre `gate_temporal.csv`):
  1. Cruce `resultado` (COMPATIBLE/INCOMPATIBLE/INDETERMINADO) × `category`, y `motivo` × `category`.
  2. Tasa de `raw` por categoría, en especial cat. 0 y 5 (verifica H3), y tasa de `filter_fp_safe` (INCOMPATIBLE) en cat. 0/5 frente a 6/14.
  3. **Embudo del árbol de Apéndice G por categoría:** % CONFIRM / REVIEW / DISCARD, separando dentro de DISCARD los INDETERMINADO.
  4. Para los 671 FN de cat. 6: cuántos tienen `raw=true`, cuántos INDETERMINADO y cuántos INCOMPATIBLE.
  5. Muestreo visual: unos 30–50 clips de cat. 2, 3 y 4 con `clear=true` (¿es FP real o gesto legítimo?) y unos 100 FN de cat. 6 con `raw=false` (¿hay gesto visible o el etiquetado es ruidoso?).
- **Reglas de decisión tras la Fase 0:**
  - Si `raw` en cat. 0 ronda el 70–80 % y INCOMPATIBLE es residual → el ternario solo no basta; P1-2 y P0-5 pasan a primera línea.
  - Si los INDETERMINADO son buena parte de los FN de cat. 6 → P0-3 recupera recall sin tocar la geometría.
  - Si las cat. 2/3/4 resultan ser gestos legítimos → la métrica de FP se rehace y mucho "problema" desaparece.
  - Si hay muchos FN con `raw=false` y gesto no visible → hay un techo de recall del esqueleto, y la salida es P2-2, no más calibración.
- **Alternativas descartadas:** empezar por implementar P0-5 o P1-2 sin saber en qué rama estamos.
- **Métrica:** tabla de una página con el embudo real por categoría y las respuestas a H1–H3.
- **Archivos:** `gate_temporal.csv`, `audit_htb_metrics.py` (ya existe), una hoja de cálculo o script de pandas de usar y tirar.

### P0-2 — Modelo de coste y prevalencia

- **Problema:** sin prevalencia ni costes relativos no hay forma de decir si 9,4 % o 35 % es aceptable (H7).
- **Enfoque propuesto:** acordar con producto cuatro números: prevalencia de robo entre candidatos Motion, coste de un aviso fuerte falso, coste de un caso en REVIEW (humano y SAM por separado) y coste de un robo perdido. Con ellos, medir cada configuración como **coste por 1 000 candidatos** y **llamadas SAM por 1 000**, no como tasas por categoría.
- **No inventar** el suelo de recall: es una decisión de producto. Las tres respuestas coinciden en esto y es correcto.
- **Alternativas descartadas:** optimizar accuracy global o tasa de INCOMPATIBLE aislada (una tasa alta puede venir de mala cobertura).
- **Métrica:** coste/1 000 y su intervalo de confianza por bootstrap de sesiones.
- **Archivos:** informe de coste, sin cambios de código.

### P0-3 — Semántica correcta del árbol de decisión

- **Problema:** la rama `else: DISCARD` descarta INDETERMINADO (§2.3).
- **Enfoque propuesto:** cinco salidas en lugar de tres.

| Condición | Acción propuesta |
|---|---|
| `clear` | CONFIRM |
| `raw ∧ ¬clear` | REVIEW (prioridad por score calibrado; P1-6) |
| `INDETERMINADO ∧ ¬raw` | KEEP/REVIEW de baja prioridad; **nunca DISCARD** |
| `INCOMPATIBLE` fuerte, cobertura válida, sin hipótesis abierta | Candidato a DISCARD (solo con política ON) |
| Resto | Registrar y conservar |

  `motion_suppress` se redefine: solo vale como descarte cuando proviene de INCOMPATIBLE fuerte (o, más adelante, de un score calibrado muy bajo con cobertura válida), nunca de "no pocket". Todo en **modo observación** primero: `accion_aplicada = null`.
- **Efecto esperado:** recupera recall (los INDETERMINADO dejan de perderse) pero **sube el volumen de REVIEW**; por eso va acoplado a P1-2 y P1-6.
- **Alternativas descartadas:** invertir el booleano; convertir INDETERMINADO en false; `legacy_compat` en producción (SOLUCIONES §5.3, §11.3).
- **Métrica:** cero descartes de INDETERMINADO en test y auditoría; matriz de transición completa de la política.
- **Archivos:** `pocket_filter/action_policy.py`, `har_premodel`, `har_robbery_review`, `har_sam3_gate` (Kanvis).

### P0-4 — Dataset por evento, con Motion y split por sesión

- **Problema:** el audit etiqueta por carpeta y sin Motion (H2, H4); además `n_frames`/longitud de clip puede filtrar la etiqueta (cat. 2 "long buffers").
- **Enfoque propuesto:**
  1. Unir al CSV el score/intervalo de Motion (p41/p48) por clip, si existe para `data_result`.
  2. **Split por sesión desde ya:** la ruta `data_result/<cat>/<session>/user_*` (STATUS Apéndice A.3) permite agrupar por `<session>` sin anotar nada. Separar además cámaras/días de test que nunca se usen para ajustar.
  3. Anotación humana por evento de una submuestra: robo / no robo / dudoso, gesto de bolsillo visible sí/no, destino, calidad de pose, intervalo aproximado.
  4. Incluir FP reales de tienda (Motion sí, humano "no robo") como conjunto aparte, prioritario para validar INCOMPATIBLE.
- **Alternativas descartadas:** split aleatorio por clip (varios clips de una misma sesión en train y test inflan las métricas).
- **Métrica:** nº de eventos por grupo y por cámara; tasas con IC por bootstrap de sesiones.
- **Archivos:** `batch_data_result.py` (columnas `session`, `camera`, scores Motion), script de split.

### P0-5 — Evaluar sobre el intervalo de Motion

- **Problema:** el filtro mira ventanas ciegas de 2,5 s con OR (H5) en lugar de la zona que Motion marca.
- **Enfoque propuesto:** API con `interval_s=(t0,t1)` o frame de pico, ampliado con contexto (partir de ±0,35 s, que es el `context_pad_s` ya presente en C.1, y de ±1 s como en SOLUCIONES §6.4; elegir en validación). Dentro del intervalo se aplican geometría, política y evidencia. Si el intervalo no cubre lo suficiente → INDETERMINADO, no interpolar. Sin intervalo → fallback a G+H, para no romper el batch histórico. Registrar `intervalo_s` usado.
- **Variante si C2 solo da score por chunk:** usar el score del chunk como feature de fusión (P1-2), sin ventana.
- **Alternativas descartadas:** seguir con ventanas ciegas; exigir siempre intervalo (rompe el histórico).
- **Métrica:** comparación pareada con G+H sobre las mismas sesiones: recall clear y raw en 6/14, FP en 0/5 y en el resto, nº medio de ventanas evaluadas, % de INDETERMINADO por cobertura.
- **Archivos:** `pocket_filter/evaluate.py`, `temporal_eval.py`, `har2_test.py`, contrato con C2.

### P1-1 — Features nuevas, baratas y solo de esqueleto

- **Problema:** SOLUCIONES §12.3.6 pregunta si hay features baratas que separen "llevar al pecho" de "bolsillo"; el STATUS dice que las reglas están en el techo de las features actuales.
- **Enfoque propuesto:** exportar al CSV, calculadas sobre series crudas:
  - **Periodicidad de la distancia muñeca–cadera** (pico de autocorrelación o espectro en la banda de braceo) y **antifase bilateral** de ambas muñecas: firma de caminar.
  - **Velocidad radial** `dd/dt` sostenida negativa, % de monotonía de la aproximación, y deceleración al contacto.
  - **Permanencia** dentro de `r_near`, nº de episodios, presencia y velocidad de retirada.
  - **Altura de la muñeca** respecto a cadera/hombro en el origen y **ángulo del codo** en el contacto (sin volver a exigir codo siempre).
  - Velocidad de cadera (marcha) y razón de ancho de hombros (ángulo de vista).
- **Excluir** `n_frames` y cualquier proxy de duración del clip (riesgo de fuga, §P0-4).
- **Alternativas descartadas:** más umbrales sobre las mismas 6 features; red neuronal end-to-end sobre poses (pierde auditabilidad, SOLUCIONES §12.2.H).
- **Métrica:** ganancia de AUC/PR por categoría y por motivo sobre el conjunto de features actual, con split por sesión.
- **Archivos:** `pocket_filter/kinematics.py`, `batch_data_result.py` (columnas `kin_*`).
- **Estado P1-1b (2026-11-22):** implementadas `kin_near_dwell_s`, `kin_swing_peaks`, `kin_approach_depth`; regla motivada `kin_axis_dwell_enabled` en `robbery_confirm` (eje + marcha). Probar con `pocket_filter_config.PRUEBA_KIN_AXIS_DWELL.json` y `analyze_kinematics_separation.py` (bloque eje).

### P1-2 — Fusión aprendida pequeña, calibrada y monótona

- **Problema:** una puerta única sobre confianzas heurísticas ya demostró que sacrifica recall (STATUS §4, §8.1–2).
- **Enfoque propuesto:**
  1. Entradas: features del CSV actual (`v2_conf`, `motivo`, `ctx_*`, `legacy_conf`, `primary_action_confidence`), P1-1 y, si está disponible, el score/posición del pico de Motion.
  2. Modelos: regresión logística regularizada como línea base y GBM con **restricciones de monotonía** (más contacto → más P(robo); más fracción de marcha → menos) para conservar explicabilidad.
  3. **Calibración** isotónica o Platt, **por motivo**, sobre grupos de sesión no usados en el ajuste.
  4. Salida: P(robo | x) y tres umbrales (confirm / review / descarte) elegidos por **coste** (P0-2) con restricción de recall acordada. La banda intermedia hace de abstención.
  5. El filtro geométrico sigue emitiendo evidencia JSON; el modelo decide **dónde se corta**, no qué se vio.
  6. Pesar por categoría para que cat. 5 y cat. 0 (las más grandes) no dominen el entrenamiento, y reportar siempre por categoría, nunca solo agregado.
- **Criterio de abandono (falsable):** si, a igual recall, la fusión no baja el FP respecto a G+H y G+H+walk_gate por encima del ruido (IC por bootstrap de sesiones), el techo de features es real y la salida está en P0-5 (Motion) o P2-2 (vídeo).
- **Alternativas descartadas:** umbral global (49 % de recall), veto legacy (~50 %), `scores` actual (de ~83 % a ~67 % en la muestra del 10 %), todos medidos en STATUS §4–§5.2.
- **Métrica:** frontera de Pareto (recall clear+review vs FP y vs coste/1 000) frente a G+H como punto de referencia; ECE y diagramas de fiabilidad por motivo; estabilidad entre semillas.
- **Archivos:** nuevo `learned_fusion.py` y `--backend learned` en `batch_data_result.py`; el camino determinista sigue siendo el default.

### P1-3 — Validez, fusión de parejas y episodios

- **Problema:** `valid_mask (T,)` ignorado, sin letterbox/espejo, "cualquier pareja COMPATIBLE basta", INCOMPATIBLE sobre todo el clip con `valid.all()` (SOLUCIONES §11.4–11.5).
- **Enfoque propuesto:** primero **auditar contra el código actual** qué de esto sigue abierto (§3.2). Después: propagar `valid_mask (T,)` AND validez por KP AND confianza; geometría en espacio imagen coherente con undo-letterbox/espejo; INCOMPATIBLE definido sobre el **gesto delimitado** con cobertura alta en ese tramo, no sobre el clip entero; ranking de parejas (validez, origen y entrada, caída radial, contacto estable, plausibilidad ipsilateral como prior y no como veto); listar episodios separados por retirada.
- **Alternativas descartadas:** prohibir parejas cruzadas (el cruce real existe); elegir solo `d_min`; `valid_ratio` fijo.
- **Métrica:** % INCOMPATIBLE en FP con vista buena; % INDETERMINADO por causa; COMPATIBLE espurio por cruce; recall en cruce real.
- **Archivos:** `pocket_filter/io.py`, `evaluate.py` (`_fuse_pairs`), C1 sidecars.

### P1-4 — Visualización y tests

- **Problema:** calibrar sin ver `d(t)` es lento y peligroso (SOLUCIONES §6.6).
- **Enfoque propuesto:** overlay de `B_L/B_R`, anillos r/R, trazas de muñeca, curva `d(t)` por pareja con estado de validez y motivo; tabla de tests sintéticos de la spec (lejos→cerca, cerca sin origen, desorden temporal, brazos distintos, cruce, huecos, saltos de ID/escala, frontera r/R, espejo/letterbox, dos episodios, cobertura completa vs incompleta).
- **Métrica:** suite reproducible en CI; tiempo medio de diagnóstico de un FP/FN.
- **Archivos:** `visualize_pocket.py` (o extensión de `visualize_sup.py`), `tests/test_pocket_spec_table.py`.

### P1-5 — Modo observación y contrato JSON

- **Problema:** el contrato está descrito pero no cableado extremo a extremo (SOLUCIONES §9.1; STATUS §10).
- **Enfoque propuesto:** persistir por `camera/session/event_id/filter_version/config_hash`: intervalo, cobertura, resultado ternario por pareja, `raw`, `clear`, `motivo`, score calibrado (si aplica), `accion_propuesta` y `accion_aplicada=null`. Idempotente ante replays. Valores ausentes explícitos (no rellenar con 0). Los destinos pecho/mochila quedan como `NOT_EVALUATED`.
- **Métrica:** % de eventos evaluados, latencia p50/p95, replay determinista, distribución de estados y estimación de FP suprimibles y TP que se habrían perdido.
- **Archivos:** `har_premodel`, `har_robbery_review`, `har_sam3_gate`, esquema JSON.

### P1-6 — Triage del tramo REVIEW

- **Problema:** REVIEW es el coste oculto (H3) y P0-3 lo aumenta al conservar INDETERMINADO.
- **Enfoque propuesto:** ordenar la cola por score calibrado en lugar de binario; SAM3 solo en REVIEW, y con tope por unidad de tiempo si hace falta; SAM3 resuelve los INDETERMINADO por oclusión (SOLUCIONES §4 Opción I); medir llamadas SAM y casos humanos por 1 000 candidatos.
- **Métrica:** carga de REVIEW por 1 000 candidatos; robos recuperados por llamada SAM; tiempo en cola.
- **Archivos:** `har_sam3_gate`, `har_robbery_review`.

---

## 6. Plan de validación y calibración

### 6.1 Datasets

1. Auditoría actual `data_result`, ahora **agrupada por sesión** (muestra estratificada seed 42 para iterar; barrido completo antes de decidir).
2. Cámaras/días reservados nunca usados para ajustar.
3. **FP reales de tienda** (Motion sí, humano "no robo").
4. **TP confirmados**, con y sin oclusión.
5. **Sintéticos y catálogo** de la spec (12 Hz, huecos, fronteras r/R, espejo).
6. Submuestra anotada por evento (P0-4).

### 6.2 Comparativas obligatorias (mismo conjunto de eventos)

| Sistema | Qué mide |
|---|---|
| Legacy | Suelo de FP, techo bajo de recall |
| Pocket sin política | Techo de recall, suelo de FP |
| **G+H congelado** | Baseline de referencia y rollback |
| G+H + `walk_gate` | Palanca de microajuste (H6) |
| G+H + árbol corregido (P0-3) | Efecto de la semántica sobre recall y carga REVIEW |
| + intervalo de Motion (P0-5) | Efecto del contexto de C2 y de menos "miradas" |
| + fusión aprendida (P1-2) | Frontera de Pareto frente a reglas |

### 6.3 Criterios go / no-go

| Modo | GO cuando |
|---|---|
| **Observación** | Contratos y tests del ternario pasan; salidas persistentes con replay determinista; métricas por evento y por cámara; sin supresión aplicada |
| **REVIEW activo** | Carga de REVIEW por 1 000 candidatos cuantificada y aceptada por producto; recall conjunto (clear + review) por encima del suelo que fije producto |
| **Descarte por INCOMPATIBLE** (canario por cámara, con feature flag) | Conjunto reservado con cero (o un límite acordado) de TP que el filtro habría suprimido; revisión visual de **todos** los TP propuestos para suprimir; INDETERMINADO nunca suprime; rollback inmediato |
| **NO-GO** | Pérdida de TP por encima del límite, error de máscara/timestamp, evento no reconstruible marcado INCOMPATIBLE, degradación por cámara o discrepancia de replay |

### 6.4 Métricas a reportar

| Métrica | Nota |
|---|---|
| Recall `clear`, `raw` y conjunto (clear+review) en cat. 6/14 | Por separado; los FN de G+H |
| FP por **cada** categoría negativa y agregado | Nunca solo cat. 0 (H1) |
| % CONFIRM / REVIEW / DISCARD / INDETERMINADO-conservado | El embudo, por categoría |
| INDETERMINADO por causa | Oclusión, máscara, lateral, hueco, cobertura |
| `filter_fp_safe` (INCOMPATIBLE) en FP cotidianos con vista buena | Utilidad real del descarte |
| Coste por 1 000 candidatos y llamadas SAM por 1 000 | Con IC por bootstrap de sesiones |
| ECE / fiabilidad por motivo | Solo si hay calibración |
| Rendimiento por cámara | Recall, FP y abstención |
| Latencia p50/p95 del filtro en CPU | Objetivo de SOLUCIONES §12.1: despreciable |

---

## 7. Integración Kanvis

- **Orchestrator (`har_premodel`, `har_robbery_review`, gates SAM3):** invocar el filtro solo sobre candidatos de Motion; pasar intervalo/score de C2 cuando exista; consumir el árbol de §5/P0-3 en lugar del booleano; SAM3 solo en REVIEW. Mantener separada la taxonomía de pasillo c0/c1/c5 del filtro de alerta.
- **Contrato JSON persistente por `event_id`** (ilustrativo; verificar qué entrega C2 antes de fijarlo):

```json
{
  "event_id": "camera/session/event",
  "filter_version": "pocket-v2+policy-obs-1",
  "config_hash": "...",
  "event_interval": {"start": null, "end": null, "source": "C2|window|none"},
  "coverage": {"valid_fraction": null, "gaps": null},
  "pocket": {"resultado": "COMPATIBLE|INCOMPATIBLE|INDETERMINADO", "raw": true,
             "motivo": "...", "episodes": [], "evidence": {}},
  "motion": {"score": null, "peak_t": null},
  "fusion": {"p_theft_calibrated": null, "model_version": null},
  "decision": {"accion_propuesta": "CONFIRM|REVIEW|KEEP|DISCARD_CANDIDATE",
               "accion_aplicada": null, "modo": "observacion", "reason_codes": []}
}
```

- **Modo observación frente a aplicación:** calcular siempre, `accion_aplicada=null`; medir impacto en cola y robos perdidos; activar por versión de config y por cámara con feature flag; rollback a G+H por copia de `BASELINE_PRODUCCION_H.json` (STATUS Apéndice F).

---

## 8. Roadmap sugerido

| Hito | Contenido | Salida |
|---|---|---|
| **0 · Diagnóstico (primeros días)** | P0-1 (Fase 0), P0-2 (preguntas a producto), auditoría del código vs brechas de §3.2 | Embudo real por categoría; decisión sobre qué rama sigue |
| **1 · Semántica y datos** | P0-3 en modo observación, P0-4 (unir Motion, split por sesión, test reservado, muestra anotada) | Árbol corregido medido; dataset utilizable |
| **2 · Contexto Motion** | P0-5 (API de intervalo con fallback), P1-3 y P1-4 en lo que siga abierto | Comparativa pareada vs G+H |
| **3 · Fusión** | P1-1 (features), P1-2 (logística y GBM monótono, calibración), P1-6 (triage) | Frontera de Pareto y propuesta de operating point por coste |
| **4 · Observación en tienda** | P1-5 cableado en Orchestrator; medición extremo a extremo | Informe sombra con coste real y FN potenciales |
| **5 · Activación condicional** | Canario por cámara; solo INCOMPATIBLE fuerte o score calibrado muy bajo con cobertura válida | Decisión go/no-go con coste medido |
| **6 · Extensiones** | P2-1 multi-destino, P2-2 segundo canal en REVIEW, P2-3 diagnóstico | — |

No fijo duraciones en semanas porque dependen de si C2 puede exportar intervalo/score y de la disponibilidad de anotación.

---

## 9. Riesgos residuales

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Etiquetas por carpeta no reflejan gestos (cat. 2/3/4 y similares) | Media-alta | Alto | Muestreo visual en Fase 0; submuestra anotada por evento |
| C2 no exporta intervalo/score para el histórico | Media | Alto | Fallback a G+H; fusión con score de chunk; priorizar P0-3 y P1-2 |
| Fusión aprendida con fuga (longitud de clip, cat. dominantes) | Media | Alto | Excluir `n_frames`; split por sesión; pesos por categoría; informe por categoría |
| REVIEW satura SAM/humano tras P0-3 | Media-alta | Alto | Cuantificar antes de activar; cola priorizada; tope de SAM |
| Sobreajuste a cámaras del laboratorio | Alta | Alto | Cámaras reservadas; observación larga |
| Calibración inestable con pocos positivos por motivo | Media | Medio | Isotónica solo con soporte suficiente; si no, fiabilidad por bins y logística |
| COMPATIBLE por cesta/bolsa junto a la cadera | Media | Medio | Hipótesis abierta + SAM; no cerrar solo con bolsillo |
| Oclusión/vista lateral sistemática en algunas cámaras | Alta | Medio | INDETERMINADO medido por cámara; mejora de C1, no descarte |
| Contrato `motion_suppress` distinto al documentado | Media | Alto | Verificar en código antes de P0-3 |

---

## 10. Preguntas al equipo de producto y datos

1. ¿Qué entrega C2 por candidato (score p41/p48, intervalo o frame de pico, por chunk o por frame) y existe para los clips de `data_result`?
2. ¿Qué son exactamente las categorías 1, 2, 3, 4, 7, 9 y 13, y en cuáles la etiqueta "esperado false" es realmente verdad de terreno de ausencia de ocultación (H1)?
3. ¿Cuál es la prevalencia de robo real entre los candidatos de Motion en tienda y el coste relativo de un caso en REVIEW (humano y SAM) frente a un robo perdido, y qué suelo de recall se acepta?
4. ¿Qué incluye exactamente hoy `motion_suppress` (¿"no pocket"?) y está ya cableada en el Orchestrator la rama `else: DISCARD`?
5. ¿Existe un conjunto de FP reales de tienda etiquetados por humano, por cámara y día, separado del `data_result` de laboratorio?

---

## 11. Conclusión

**Decisión recomendada:** dejar de perseguir un umbral y pasar a una cascada por evento con coste explícito, empezando por corregir la semántica del árbol (INDETERMINADO nunca se descarta), evaluar en el intervalo de Motion y probar una fusión aprendida calibrada y monótona como candidata a mover la frontera, con G+H como baseline y rollback.

**Siguiente paso concreto:** ejecutar la Fase 0 sobre el barrido completo ya existente:

```bash
cd har && .venv/bin/python3 audit_htb_metrics.py /tmp/gate_temporal.csv
.venv/bin/python3 fase0_pocket_audit.py /tmp/gate_temporal.csv -o /tmp/fase0_report.txt
```

Procedimiento muestra 5–10 %: [`FASE0_RUNBOOK.md`](FASE0_RUNBOOK.md).

---

## 12. Anexo — Comparativa de las tres respuestas y veredicto

### 12.1 Lectura crítica

| | **Respuesta 1** (ternario y evento) | **Respuesta 2** (plantilla §14, R1–R10) | **Respuesta 3** (RFC de arquitectura) |
|---|---|---|---|
| **¿Trata el problema hand-to-body?** | Sí | Sí | **No** |
| **Idea central** | Contrato ternario contractual, ventana de evento Motion, validez de datos, política multi-señal en observación | Cerrar las tres señales (suppress/clear/raw), intervalo de Motion, política por motivo, calibración por motivo, GBM sobre el CSV | Reorganizar el pipeline en C2.1/C2.2/C2.3, MinIO, RabbitMQ/Kafka, C7/C8/C9, MLOps |
| **Fortalezas** | Doctrina de abstención bien defendida; protocolo de datos (split por sesión/cámara, test reservado); métrica de coste por 1 000 candidatos; go/no-go y rollback; no inventa el suelo de recall; sigue el espíritu de SOLUCIONES §2.2 | Sigue la plantilla al pie de la letra; mapea cada propuesta a ficheros concretos; incluye motivo-condicional, calibración y GBM sobre features que ya están en el CSV; da comandos de reproducción | Su cascada de tres fases (prefiltro HAR → ensemble → SAM3 a demanda) coincide con la idea de cascada |
| **Debilidades** | Reproduce en gran parte el plan de endurecimiento de §12; no usa el Apéndice E ni calcula nada; da por abiertas brechas que el JSON congelado sugiere cerradas; no propone un primer experimento barato; relega la fusión aprendida a P2 | Se contradice (implementa el árbol de Apéndice G literal y a la vez dice no mapear ausencia a descarte); afirma que REVIEW recupera FN "sin reabrir el 82 %" sin medirlo; R3 admite que ya existe lo que propone; calibra y entrena sobre etiquetas de carpeta sin controles de fuga; solo mira cat. 0/5 | Analiza otro documento; no menciona `pocket_filter`, G+H, ternario, motivos ni métricas; sus recomendaciones (MinIO, brokers, portal de integradores) no atacan el problema |
| **Aprovechable** | Estructura ternaria, ventana de evento, protocolo de datos, coste/1 000, go/no-go | Motivo-condicional, calibración por motivo, GBM sobre CSV, rutas de ficheros | Solo la confirmación de la arquitectura en cascada |

### 12.2 Coincidencias y divergencias

- **Coinciden las Respuestas 1 y 2 y yo** en: conservar G+H como baseline y rollback; evaluar sobre el intervalo de Motion; política por motivo y no un umbral global; calibración y evaluación por grupos reservados; observación antes de aplicar; visualización y tests; y no inventar un suelo de recall.
- **Divergen** en la prioridad de la fusión aprendida (P2 en la Respuesta 1, P1 en la 2) y en cuánto confiar en el ternario estricto como única vía de supresión. Mi posición: la fusión aprendida va tras la Fase 0, con criterio de abandono explícito, y el ternario sigue siendo la regla de supresión segura porque es la única auditable.

### 12.3 Qué incorporo de cada una y qué añado

| De la Respuesta 1 | De la Respuesta 2 | Añadido por este análisis |
|---|---|---|
| Contrato ternario y `accion_propuesta` ≠ `accion_aplicada` | Calibración por motivo y política por motivo | Fase 0 sobre CSV existentes (P0-1) |
| Ventana de evento con cobertura | GBM sobre las columnas del CSV | Hallazgo H1: FP agregado del 35 % y 66 % fuera de cat. 0/5 |
| Split por sesión/cámara y test reservado | Rutas de ficheros por propuesta | Hallazgo H3: `raw` casi no discrimina; el coste pasa a REVIEW |
| Coste por 1 000 candidatos | Comandos de reproducción | Corrección de la rama `else: DISCARD` (P0-3) |
| Go / no-go y canario | Selección de pareja ganadora | H4–H7: falta Motion en el CSV, OR temporal, `walk_gate`, prevalencia |
| | | Features nuevas (P1-1) y criterio de abandono de la fusión (P1-2) |

### 12.4 Veredicto

**Solución más prometedora:** la cascada por evento con coste explícito, en este orden: **(1)** Fase 0 sobre CSV existentes, **(2)** árbol corregido más intervalo de Motion en modo observación, **(3)** fusión aprendida calibrada y monótona como challenger con criterio de abandono, **(4)** SAM3 solo en REVIEW. Es la única combinación que ataca a la vez las tres causas que los documentos identifican: puerta única, falta de contexto de Motion y confianzas sin calibrar. Además es falsable en cada paso: cada hito termina en una tabla que dice si seguir o no.

**Si solo hubiera que hacer una cosa esta semana:** la Fase 0. Es la de menor coste y la que más puede cambiar el plan, incluida la posibilidad de que parte del "problema" sea de etiquetado y no de geometría.

**Sobre la Respuesta 3:** si quieres una tercera opinión real sobre este problema, habría que volver a lanzar a esa IA con SOLUCIONES y STATUS y la plantilla §14. La respuesta que dio corresponde a otro documento.
