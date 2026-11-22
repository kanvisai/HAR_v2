# Semántica de categorías en el audit (`data_result`)

**Runtime:** el filtro **no ve** la categoría. Las carpetas solo sirven para **medir** en laboratorio.

## Cómo leer TP/TN/FP/FN en batch

| Grupo | Cats | Etiqueta audit `expected_hand_to_body` |
|-------|------|----------------------------------------|
| **Robo** | 6, 14 | `true` — queremos detectar ocultación |
| **Negativas estrictas** | 0, 1, 2, 5, 9, 13 | `false` — clear=true cuenta como **FP** |
| **Gesto legítimo posible** | **3, 4, 7** | **SKIP** — compra normal, llevando producto, empleado reponiendo; **pocket puede dar true** sin ser error |
| **Sin etiqueta dura** | 8, 10, 11, 12 | **SKIP** — pueden implicar hand_to_body en algún frame |

Definición en código: `har/audit_labels.py`.

## Métricas de producto (Kanvis)

- **Confirmar robo:** priorizar recall en 6/14 (clear + raw/REVIEW).
- **No saturar tienda:** FP **duro** en **0** y **5**; en **3/4/7** el filtro debe mandar a **REVIEW** o abstenerse (P0-3), no tratar clear como alarma automática igual que cat 6.

`fase0_pocket_audit.py` §4 vs §4b separa FP estricto vs categorías con gesto esperable.