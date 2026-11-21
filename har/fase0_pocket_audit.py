#!/usr/bin/env python3
"""
Fase 0 (P0-1): auditoría sobre CSV de batch sin código nuevo de filtro.

Salidas: ternario pocket (resultado), raw/clear, embudo Orchestrator (doc STATUS Ap.G),
FN cat 6/14, FP agregado, motivos frecuentes.

Uso:
  .venv/bin/python3 fase0_pocket_audit.py /tmp/gate_10.csv
  .venv/bin/python3 fase0_pocket_audit.py /tmp/gate_10.csv -o /tmp/fase0_report.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

NEGATIVE_CATS = {0, 1, 2, 3, 4, 5, 7, 9, 13}
POSITIVE_CATS = {6, 14}
LABELED = NEGATIVE_CATS | POSITIVE_CATS


def _bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _orch_branch_legacy(r: dict[str, str]) -> str:
    """Árbol STATUS Ap.G original (else descarta INDETERMINADO)."""
    if _bool(r.get("motion_suppress", "")):
        return "DISCARD_SUPPRESS"
    if _bool(r.get("hand_to_body", "")):
        return "CONFIRM"
    if _bool(r.get("hand_to_body_raw", "")):
        return "REVIEW_RAW"
    return "DISCARD_ELSE"


def _orch_branch_doctrine(r: dict[str, str]) -> str:
    """P0-3: motion_suppress del CSV; INDETERMINADO → no descartar (KEEP)."""
    if _bool(r.get("motion_suppress", "")):
        return "DISCARD_SUPPRESS"
    if _bool(r.get("hand_to_body", "")):
        return "CONFIRM"
    if _bool(r.get("hand_to_body_raw", "")):
        return "REVIEW_RAW"
    res = (r.get("resultado") or "").strip().upper()
    if res == "INDETERMINADO":
        return "KEEP_INDETERMINADO"
    if res == "INCOMPATIBLE":
        if _bool(r.get("filter_fp_safe", "")):
            return "DISCARD_INCOMPATIBLE"
        return "KEEP_INCOMPATIBLE_WEAK"
    return "DISCARD_ELSE"


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _print_section(lines: list[str], title: str) -> None:
    lines.append("")
    lines.append(f"=== {title} ===")


def run_audit(rows: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    ok = [r for r in rows if not (r.get("error") or "").strip()]
    err = len(rows) - len(ok)
    lines.append(f"Filas CSV: {len(rows)}  |  OK: {len(ok)}  |  error: {err}")

    _print_section(lines, "1. Meta batch")
    backends = Counter(r.get("backend", "") for r in ok)
    for b, n in backends.most_common():
        lines.append(f"  backend={b or '(vacío)'}: {n}")

    _print_section(lines, "2. resultado (ternario pocket) × categoría")
    by_cat_res: dict[int, Counter[str]] = defaultdict(Counter)
    res_global: Counter[str] = Counter()
    for r in ok:
        cat = int(r["category"])
        if cat not in LABELED:
            continue
        res = (r.get("resultado") or "").strip() or "(vacío)"
        by_cat_res[cat][res] += 1
        res_global[res] += 1
    for cat in sorted(by_cat_res):
        parts = ", ".join(f"{k}={v}" for k, v in by_cat_res[cat].most_common())
        lines.append(f"  cat {cat:2d}: {parts}")
    lines.append(f"  TOTAL: {dict(res_global)}")

    _print_section(lines, "3. Tasas raw / clear / filter_fp_safe por categoría")
    lines.append(
        "  cat |    n | raw% | clear% | fp_safe% | nota"
    )
    for cat in sorted(LABELED):
        subset = [r for r in ok if int(r["category"]) == cat]
        n = len(subset)
        if n == 0:
            continue
        raw_n = sum(1 for r in subset if _bool(r.get("hand_to_body_raw", "")))
        clr_n = sum(1 for r in subset if _bool(r.get("hand_to_body", "")))
        safe_n = sum(1 for r in subset if _bool(r.get("filter_fp_safe", "")))
        note = "robo" if cat in POSITIVE_CATS else "neg"
        lines.append(
            f"  {cat:3d} | {n:5d} | {100 * raw_n / n:5.1f} | "
            f"{100 * clr_n / n:6.1f} | {100 * safe_n / n:8.1f} | {note}"
        )

    _print_section(lines, "4. FP clear (hand_to_body) — categorías negativas")
    fp_total = 0
    neg_total = 0
    for cat in sorted(NEGATIVE_CATS):
        subset = [r for r in ok if int(r["category"]) == cat]
        neg_total += len(subset)
        fp = sum(1 for r in subset if _bool(r.get("hand_to_body", "")))
        fp_total += fp
        if subset:
            lines.append(f"  cat {cat:2d}: FP={fp}/{len(subset)} ({100 * fp / len(subset):.1f}%)")
    if neg_total:
        lines.append(
            f"  AGREGADO negativas: FP={fp_total}/{neg_total} ({100 * fp_total / neg_total:.1f}%)"
        )
        ex_05 = [r for r in ok if int(r["category"]) in NEGATIVE_CATS - {0, 5}]
        fp_ex = sum(1 for r in ex_05 if _bool(r.get("hand_to_body", "")))
        if ex_05:
            lines.append(
                f"  Sin cat 0/5: FP={fp_ex}/{len(ex_05)} ({100 * fp_ex / len(ex_05):.1f}%)"
            )

    _print_section(lines, "5. Embudo Orchestrator — doctrina P0-3 (usar en Kanvis)")
    for label, cats in (
        ("GLOBAL (etiquetadas)", LABELED),
        ("cat 0", {0}),
        ("cat 6", {6}),
        ("cat 14", {14}),
    ):
        subset = [r for r in ok if int(r["category"]) in cats]
        funnel: Counter[str] = Counter(_orch_branch_doctrine(r) for r in subset)
        n = len(subset) or 1
        parts = " | ".join(f"{k}={v} ({100 * v / n:.1f}%)" for k, v in funnel.most_common())
        lines.append(f"  {label}: n={len(subset)} → {parts}")

    _print_section(lines, "5b. Embudo legado (Ap.G con else=DISCARD) — solo referencia")
    for label, cats in (("cat 6", {6}),):
        subset = [r for r in ok if int(r["category"]) in cats]
        funnel = Counter(_orch_branch_legacy(r) for r in subset)
        n = len(subset) or 1
        parts = " | ".join(f"{k}={v} ({100 * v / n:.1f}%)" for k, v in funnel.most_common())
        lines.append(f"  {label}: n={len(subset)} → {parts}")

    _print_section(lines, "6. DISCARD_ELSE (árbol legado) — desglose")
    else_rows = [
        r
        for r in ok
        if int(r["category"]) in LABELED and _orch_branch_legacy(r) == "DISCARD_ELSE"
    ]
    by_res = Counter((r.get("resultado") or "").strip() or "(vacío)" for r in else_rows)
    lines.append(f"  n={len(else_rows)}: {dict(by_res.most_common())}")
    for cat in (0, 6):
        sub = [r for r in else_rows if int(r["category"]) == cat]
        if not sub:
            continue
        cr = Counter((r.get("resultado") or "").strip() or "(vacío)" for r in sub)
        lines.append(f"    cat {cat} DISCARD_ELSE: {dict(cr.most_common())}")

    _print_section(lines, "7. DISCARD_SUPPRESS — motivos (context_veto)")
    sup_rows = [r for r in ok if _orch_branch_doctrine(r) == "DISCARD_SUPPRESS"]
    veto = Counter((r.get("context_veto") or "").strip() or "(vacío)" for r in sup_rows)
    lines.append(f"  n={len(sup_rows)} context_veto top: {veto.most_common(8)}")

    _print_section(lines, "8. FN cat 6 y 14 (clear=false, esperado true)")
    for cat in (6, 14):
        subset = [r for r in ok if int(r["category"]) == cat]
        fn = [r for r in subset if not _bool(r.get("hand_to_body", ""))]
        n = len(subset)
        lines.append(f"  cat {cat}: FN={len(fn)}/{n}")
        if not fn:
            continue
        raw_yes = sum(1 for r in fn if _bool(r.get("hand_to_body_raw", "")))
        lines.append(f"    FN con raw=true: {raw_yes}/{len(fn)} ({100 * raw_yes / len(fn):.1f}%)")
        res = Counter((r.get("resultado") or "").strip() or "(vacío)" for r in fn)
        lines.append(f"    resultado en FN: {dict(res.most_common())}")
        mot = Counter((r.get("motivo") or "").strip() or "(vacío)" for r in fn)
        lines.append(f"    motivo top: {mot.most_common(6)}")

    _print_section(lines, "9. htb_tier × clear (diagnóstico policy)")
    tier = Counter(
        (r.get("htb_tier") or "").strip() or "(vacío)" for r in ok if int(r["category"]) in LABELED
    )
    lines.append(f"  distribución: {dict(tier.most_common())}")
    review_clear = sum(
        1
        for r in ok
        if int(r["category"]) in LABELED
        and (r.get("htb_tier") or "").strip() == "review"
        and _bool(r.get("hand_to_body", ""))
    )
    lines.append(f"  clear=true con tier=review (debería ser raro en robbery_confirm): {review_clear}")

    _print_section(lines, "10. Motivos (clear FP en cat 0, top 12)")
    fp0 = [r for r in ok if int(r["category"]) == 0 and _bool(r.get("hand_to_body", ""))]
    mot0 = Counter((r.get("motivo") or "").strip() or "(vacío)" for r in fp0)
    lines.append(f"  FP cat 0 n={len(fp0)}: {mot0.most_common(12)}")

    _print_section(lines, "11. Interpretación rápida (Fase 0)")
    cat0 = [r for r in ok if int(r["category"]) == 0]
    if cat0:
        raw_rate_0 = sum(1 for r in cat0 if _bool(r.get("hand_to_body_raw", ""))) / len(cat0)
        review_rate_0 = sum(1 for r in cat0 if _orch_branch_doctrine(r) == "REVIEW_RAW") / len(cat0)
        lines.append(
            f"  H3 cat0: raw_rate={raw_rate_0:.1%} → REVIEW branch={review_rate_0:.1%} "
            f"(si raw≈70–80%, el tramo REVIEW puede saturar SAM)"
        )
    fn6 = [r for r in ok if int(r["category"]) == 6 and not _bool(r.get("hand_to_body", ""))]
    if fn6:
        indet_fn = sum(
            1 for r in fn6 if (r.get("resultado") or "").strip() == "INDETERMINADO"
        )
        lines.append(
            f"  P0-3: FN cat6 con pocket INDETERMINADO: {indet_fn}/{len(fn6)} "
            f"— si alto, priorizar doctrina abstención vs DESCARTE"
        )

    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Auditoría Fase 0 sobre CSV batch")
    ap.add_argument("csv", type=Path, help="CSV de batch_data_result.py")
    ap.add_argument("-o", "--output", type=Path, help="Guardar informe en archivo")
    args = ap.parse_args()
    if not args.csv.is_file():
        print(f"no existe: {args.csv}", file=sys.stderr)
        raise SystemExit(2)
    rows = _load(args.csv)
    report = run_audit(rows)
    text = "\n".join(report)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\n(guardado en {args.output})", flush=True)


if __name__ == "__main__":
    main()