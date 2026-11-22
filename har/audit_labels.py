"""Semántica de categorías del dataset de audit (carpeta ≠ runtime sin etiqueta)."""
from __future__ import annotations

# Robo / ocultación — esperamos gesto de bolsillo en la muestra de lab
EXPECTS_HAND_TO_BODY_TRUE: frozenset[int] = frozenset({6, 14})

# Pasillo / tráfico sin gesto de bolsillo objetivo en el etiquetado
EXPECTS_HAND_TO_BODY_FALSE: frozenset[int] = frozenset({0, 1, 2, 5, 9, 13})

# Compra, llevar producto, reponer, etc.: el gesto pocket puede ser TRUE en vídeo real
GESTURE_POSSIBLE_CATS: frozenset[int] = frozenset({3, 4, 7})

# Sin etiqueta clara en audit; pueden mostrar hand_to_body legítimo (bolsillo propio, cesta…)
EXPECTS_NO_LABEL: frozenset[int] = frozenset({8, 10, 11, 12})

STRICT_NEGATIVE_CATS: frozenset[int] = EXPECTS_HAND_TO_BODY_FALSE

CATEGORY_NOTES: dict[int, str] = {
    0: "pasillo / sin robo",
    1: "negativo audit",
    2: "buffers largos (revisar etiqueta)",
    3: "compra normal — puede tocar bolsillo/cesta",
    4: "compra llevando — gesto bolsillo frecuente",
    5: "pasillo denso",
    6: "robo",
    7: "empleado reponiendo — puede acercar mano al cuerpo",
    8: "sin etiqueta dura",
    9: "negativo audit",
    10: "sin etiqueta dura — puede implicar hand_to_body",
    13: "negativo audit",
    14: "robo",
}