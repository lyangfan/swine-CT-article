#!/usr/bin/env python3
"""Summarize ROI half-projection classifier results with cross-grid best selection.

Ported from AutoScientists roi_half_projection_presence_classifier with
spec-mandated changes (§8 + §9):
  - Cross-grid best selection: per endpoint, highest val AUPRC subject to FA≤1 gate
  - Tiebreak: FA↓ → Brier↓ → lexicographic (role, variant, init)
  - No control dependencies; test metrics consumed via --test-eval
  - Writes best_selection.json, copies best model → frozen/<endpoint>.pt
  - Result card includes ≡source caveat (§10); no champion/segmentation claims
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from .common import (  # type: ignore[attr-defined]
        add_common_args,
        command_result,
        print_json,
        resolve_output_root,
        stage_report_dir,
        utc_stamp,
        write_json,
    )
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import (  # type: ignore[import-not-found]
        add_common_args,
        command_result,
        print_json,
        resolve_output_root,
        stage_report_dir,
        utc_stamp,
        write_json,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def _val_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Extract val metrics dict from a training-summary record."""
    return record.get("val_metrics") or record.get("validation_metrics") or record.get("best_metrics") or {}


def _val_auprc(record: dict[str, Any]) -> float:
    return float(_val_metrics(record).get("auprc", -1.0))


def _val_fa(record: dict[str, Any]) -> int:
    return int(_val_metrics(record).get("false_absent_count", 9999))


def _val_brier(record: dict[str, Any]) -> float:
    return float(_val_metrics(record).get("brier", 9999.0))


def _val_fp(record: dict[str, Any]) -> int:
    return int(_val_metrics(record).get("false_present_count", 9999))


# ---------------------------------------------------------------------------
# Cross-grid best selection
# ---------------------------------------------------------------------------


def _selection_key(record: dict[str, Any]) -> tuple[float, int, float, str, str, str]:
    """Sort key for best-model selection.

    Primary: highest AUPRC (negated so ascending sort puts best first).
    Tiebreak: FA↓, Brier↓, then deterministic lexicographic (role, variant, init).
    """
    return (
        -_val_auprc(record),                       # higher AUPRC first
        _val_fa(record),                           # lower FA first
        _val_brier(record),                        # lower Brier first
        str(record.get("roi_role", "")),           # lexicographic
        str(record.get("variant", "")),            # lexicographic
        str(record.get("init_mode", "")),          # lexicographic
    )


def _records(training_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not training_summary:
        return []
    return list(training_summary.get("model_outputs", training_summary.get("results", [])))


def select_best_per_endpoint(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Cross-grid best selection: per endpoint, highest val AUPRC with FA≤1 gate.

    Returns dict[endpoint -> best record].
    """
    by_endpoint: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        endpoint = str(rec.get("endpoint", ""))
        if not endpoint:
            continue
        by_endpoint.setdefault(endpoint, []).append(rec)

    best: dict[str, dict[str, Any]] = {}
    for endpoint, recs in sorted(by_endpoint.items()):
        # Gate: FA ≤ 1
        eligible = [r for r in recs if _val_fa(r) <= 1]
        if not eligible:
            # No model passes the gate — fall back to full pool with a warning
            eligible = recs
        eligible.sort(key=_selection_key)
        best[endpoint] = eligible[0]
    return best


# ---------------------------------------------------------------------------
# Test-eval matching
# ---------------------------------------------------------------------------


def _match_test_metrics(
    test_eval: dict[str, Any] | None,
    endpoint: str,
    role: str,
    variant: str,
    init: str,
) -> dict[str, Any] | None:
    """Extract test metrics for a specific (endpoint, role, variant, init) tuple.

    Supports two test_eval shapes:
      1. List of per-model dicts with endpoint/roi_role/variant/init_mode keys.
      2. Nested dict: {endpoint: {role: {variant: {init: metrics}}}}.
    """
    if not test_eval:
        return None

    # Shape 1: list of per-model records
    results = test_eval.get("results") or test_eval.get("model_outputs") or []
    if isinstance(results, list):
        for rec in results:
            if (
                str(rec.get("endpoint")) == endpoint
                and str(rec.get("roi_role")) == role
                and str(rec.get("variant")) == variant
                and str(rec.get("init_mode")) == init
            ):
                return rec.get("test_metrics") or rec
        return None

    # Shape 2: nested dict
    try:
        return test_eval[endpoint][role][variant][init]  # type: ignore[index]
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Result card
# ---------------------------------------------------------------------------


SOURCE_CAVEAT = (
    "本数据集 presence≡source，故准确性反映 source/FOV 信号而非解剖定位"
)


def _best_rows_table(best: dict[str, dict[str, Any]], label: str) -> str:
    """Render a markdown table of best models with val metrics."""
    lines = [
        f"| endpoint | roi_role | variant | init | AUPRC ({label}) | AUROC ({label}) | FA ({label}) | FP ({label}) | Brier ({label}) | ECE10 ({label}) |",
        f"|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for endpoint in sorted(best):
        rec = best[endpoint]
        metrics = _val_metrics(rec)
        lines.append(
            "| {endpoint} | {role} | {variant} | {init} | {auprc} | {auroc} | {fa} | {fp} | {brier} | {ece} |".format(
                endpoint=endpoint,
                role=rec.get("roi_role", "-"),
                variant=rec.get("variant", "-"),
                init=rec.get("init_mode", "-"),
                auprc=_fmt(metrics.get("auprc")),
                auroc=_fmt(metrics.get("auroc")),
                fa=metrics.get("false_absent_count", "-"),
                fp=metrics.get("false_present_count", "-"),
                brier=_fmt(metrics.get("brier")),
                ece=_fmt(metrics.get("ece_10bin")),
            )
        )
    return "\n".join(lines)


def _test_metrics_table(
    best: dict[str, dict[str, Any]],
    test_eval: dict[str, Any] | None,
) -> str:
    """Render a markdown table of test metrics for the selected best models."""
    if not test_eval:
        return ""

    lines = [
        "",
        "## Test Metrics (held-out test set)",
        "",
        "| endpoint | roi_role | variant | init | AUPRC (test) | AUROC (test) | FA (test) | FP (test) | Brier (test) | ECE10 (test) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    any_found = False
    for endpoint in sorted(best):
        rec = best[endpoint]
        role = str(rec.get("roi_role", ""))
        variant = str(rec.get("variant", ""))
        init = str(rec.get("init_mode", ""))
        tm = _match_test_metrics(test_eval, endpoint, role, variant, init)
        if tm:
            any_found = True
            lines.append(
                "| {endpoint} | {role} | {variant} | {init} | {auprc} | {auroc} | {fa} | {fp} | {brier} | {ece} |".format(
                    endpoint=endpoint,
                    role=role,
                    variant=variant,
                    init=init,
                    auprc=_fmt(tm.get("auprc")),
                    auroc=_fmt(tm.get("auroc")),
                    fa=tm.get("false_absent_count", "-"),
                    fp=tm.get("false_present_count", "-"),
                    brier=_fmt(tm.get("brier")),
                    ece=_fmt(tm.get("ece_10bin")),
                )
            )
        else:
            lines.append(
                "| {endpoint} | {role} | {variant} | {init} | - | - | - | - | - | - |".format(
                    endpoint=endpoint,
                    role=role,
                    variant=variant,
                    init=init,
                )
            )
    if not any_found:
        return ""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve_output_root({}, args.output_root)
    stamp = args.stamp or utc_stamp()

    plan = command_result(
        "DRY_RUN" if args.dry_run else "WRITE",
        "ROI result summary plan validated",
        output_root=str(output_root),
        stamp=stamp,
        training_summary=args.training_summary,
        test_eval=args.test_eval,
        classifier_only=True,
        test_consumed=False,
        segmentation_claim=False,
        champion_claim=False,
    )
    if args.dry_run:
        return plan

    training_summary = _load(args.training_summary)
    test_eval = _load(args.test_eval)

    records = _records(training_summary)
    if not records:
        raise ValueError(
            "training_summary contains no model_outputs/results records; "
            "cannot select best models"
        )

    best = select_best_per_endpoint(records)

    # Detect whether any endpoint fell back (no model met FA≤1 gate)
    fallback_endpoints: list[str] = []
    for endpoint, rec in best.items():
        if _val_fa(rec) > 1:
            fallback_endpoints.append(endpoint)

    plan = command_result(
        "DRY_RUN" if args.dry_run else "WRITE",
        "ROI result summary plan validated",
        output_root=str(output_root),
        stamp=stamp,
        training_summary=args.training_summary,
        test_eval=args.test_eval,
        model_records=len(records),
        best_per_endpoint={
            ep: {
                "roi_role": rec.get("roi_role"),
                "variant": rec.get("variant"),
                "init_mode": rec.get("init_mode"),
                "val_auprc": _val_auprc(rec),
                "val_fa": _val_fa(rec),
            }
            for ep, rec in best.items()
        },
        fallback_endpoints=fallback_endpoints or None,
        classifier_only=True,
        segmentation_claim=False,
        champion_claim=False,
    )
    if args.dry_run:
        return plan

    # ------------------------------------------------------------------
    # Write best_selection.json
    # ------------------------------------------------------------------
    rdir = stage_report_dir(output_root, stamp)
    rdir.mkdir(parents=True, exist_ok=True)

    best_selection: list[dict[str, Any]] = []
    for endpoint in sorted(best):
        rec = best[endpoint]
        # Derive checkpoint path from output_dir (training records use output_dir,
        # not checkpoint_path)
        output_dir = rec.get("output_dir", "")
        checkpoint_path = rec.get("checkpoint_path", "")
        if not checkpoint_path and output_dir:
            checkpoint_path = str(Path(output_dir) / "model_best.pt")
        best_selection.append(
            {
                "endpoint": endpoint,
                "roi_role": rec.get("roi_role"),
                "variant": rec.get("variant"),
                "init_mode": rec.get("init_mode"),
                "checkpoint_path": checkpoint_path,
                "val_metrics": _val_metrics(rec),
            }
        )
    write_json(rdir / "best_selection.json", best_selection)

    # ------------------------------------------------------------------
    # Copy model_best.pt → frozen/<endpoint>.pt
    # ------------------------------------------------------------------
    frozen_dir = output_root / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for endpoint in sorted(best):
        rec = best[endpoint]
        output_dir = rec.get("output_dir", "")
        src = Path(output_dir) / "model_best.pt" if output_dir else Path("")
        if src.exists():
            dst = frozen_dir / f"{endpoint}.pt"
            shutil.copy2(str(src), str(dst))
            copied.append(str(dst))
        else:
            copied.append(f"MISSING: {src}")

    # ------------------------------------------------------------------
    # Write result card
    # ------------------------------------------------------------------
    card_lines = [
        "# ROI / Half-Projection Presence Classifier — Result Card",
        "",
        f"> **{SOURCE_CAVEAT}**",
        "",
        "范围：cross-grid best selection per endpoint (FA≤1 gate, AUPRC↓ tiebreak)。",
        "包含 val 指标与 test 指标（若提供 --test-eval）。不声称 champion、不声称分割改善。",
        "",
        "## Best Models by Endpoint (val metrics)",
        "",
        _best_rows_table(best, "val"),
    ]

    if fallback_endpoints:
        card_lines += [
            "",
            "## Gate Fallback Warning",
            "",
            "以下 endpoint 没有任何候选模型满足 FA≤1 gate，已退回到全量池选最优：",
            "",
        ]
        card_lines.extend([f"- `{ep}`" for ep in fallback_endpoints])

    test_table = _test_metrics_table(best, test_eval)
    if test_table:
        card_lines.append(test_table)

    # Source caveat section
    card_lines += [
        "",
        "## Source / Presence Caveat",
        "",
        f"- {SOURCE_CAVEAT}",
        "",
    ]

    # Frozen models section
    card_lines += [
        "## Frozen Models",
        "",
    ]
    for path in copied:
        card_lines.append(f"- `{path}`")

    card_lines += [
        "",
        "## Claim Boundary",
        "",
        "- 可作为 ROI-localized classifier 在 val 上的 cross-grid 选优证据。",
        "- 不声称 champion、不声称分割改善、不声称下游 gate 有效性。",
        "- test 指标仅用于最终报告；选优过程不接触 test。",
    ]

    (rdir / "result_card.md").write_text("\n".join(card_lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Assemble result manifest
    # ------------------------------------------------------------------
    result = {
        **plan,
        "status": "PASS",
        "best_selection": str(rdir / "best_selection.json"),
        "result_card": str(rdir / "result_card.md"),
        "frozen_models": copied,
        "fallback_endpoints": fallback_endpoints or None,
    }
    write_json(rdir / "run_manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--training-summary",
        default="",
        help="Path to training_summary.json (Stage 5 output)",
    )
    parser.add_argument(
        "--test-eval",
        default="",
        help="Path to eval_test.py output JSON (Stage 6 test metrics)",
    )
    args = parser.parse_args(argv)
    print_json(summarize(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
