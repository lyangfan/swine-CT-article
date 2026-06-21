#!/usr/bin/env python3
"""Statistics for the v1 input-consistency comparison (spec §5.5).

Consumes the per-case long-form CSV produced by run_eval.py and produces:

  1. Per-network per-class Dice + HD95 as mean±std across 3 seeds.
  2. Per-case mean Dice (mean across conditionally-evaluated classes), then
     averaged across 3 seeds → one scalar per (network, case).
  3. Paired Wilcoxon signed-rank on the 39 per-case mean Dice values, for every
     C(4,2)=6 pair of 3D networks, Holm-Bonferroni corrected.
  4. HD95 reported descriptively only (no test). 2D nnUNet excluded from Wilcoxon.

Outputs a summary CSV + a markdown table.

Usage:
    python -m evaluation.run_stats \\
        --input evaluation/results/per_case.csv \\
        --out-dir evaluation/results/
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

NETWORKS_3D = ["nnunet_v1", "mednext_s", "swinunetr", "segformer3d"]
SEEDS = [20260520, 20260521, 20260522]
LABELS = {
    1: "front", 2: "middle", 3: "end", 4: "left_kidney", 5: "right_kidney",
    6: "testis", 7: "thoracic_cavity", 8: "abdominal_and_pelvic_cavity", 9: "head",
}


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def per_network_per_class(rows: list[dict], metric: str):
    """Returns {network: {class_label: [values across (seed, case)]}}."""
    out: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        net = r["network"]
        c = int(r["class_label"])
        v = _to_float(r[metric])
        if np.isnan(v):
            continue
        out.setdefault(net, {}).setdefault(c, []).append(v)
    return out


def per_case_mean_dice(rows: list[dict]):
    """For each (network, seed, case): mean Dice across conditionally-evaluated
    classes. Then average across the 3 seeds → {network: {case_id: scalar}}."""
    # step 1: per (network, seed, case) mean Dice over evaluated classes
    nsc: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["network"], int(r["seed"]), r["case_id"])
        nsc.setdefault(key, []).append(_to_float(r["dice"]))
    nsc_mean = {k: np.nanmean(v) for k, v in nsc.items()}

    # step 2: average across seeds per (network, case)
    nc: dict[str, dict[str, list[float]]] = {}
    for (net, seed, case), val in nsc_mean.items():
        nc.setdefault(net, {}).setdefault(case, []).append(val)
    return {net: {case: float(np.mean(vals)) for case, vals in cases.items()}
            for net, cases in nc.items()}


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni: sort p ascending, enforce monotonicity, compare to
    alpha/(m-i). Returns adjusted p-values in the ORIGINAL order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (m - rank) * pvals[idx]
        corrected = min(corrected, 1.0)
        running_max = max(running_max, corrected)
        adj[idx] = running_max
    return adj


def fmt_pm(mean, std):
    if np.isnan(mean):
        return "—"
    return f"{mean:.4f}±{std:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)

    networks_present = sorted(set(r["network"] for r in rows))
    seeds_present = sorted(set(int(r["seed"]) for r in rows))
    nets_3d = [n for n in NETWORKS_3D if n in networks_present]
    print(f"[stats] networks={networks_present} seeds={seeds_present} rows={len(rows)}")

    # ----- 1. per-network per-class Dice + HD95 (mean±std across seeds) -----
    dice_by = per_network_per_class(rows, "dice")
    hd95_by = per_network_per_class(rows, "hd95")
    summary_fields = ["network", "class_label", "class_name", "dice_mean", "dice_std",
                      "hd95_mean", "hd95_std", "n"]
    summary_path = out_dir / "summary_per_class.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for net in networks_present:
            for c in sorted(LABELS):
                dvals = np.array(dice_by.get(net, {}).get(c, []))
                hvals = np.array([v for v in hd95_by.get(net, {}).get(c, []) if not np.isnan(v)])
                if len(dvals) == 0:
                    continue
                w.writerow({
                    "network": net, "class_label": c, "class_name": LABELS[c],
                    "dice_mean": f"{np.mean(dvals):.6f}", "dice_std": f"{np.std(dvals):.6f}",
                    "hd95_mean": f"{np.mean(hvals):.4f}" if len(hvals) else "nan",
                    "hd95_std": f"{np.std(hvals):.4f}" if len(hvals) else "nan",
                    "n": len(dvals),
                })
    print(f"[stats] per-class summary -> {summary_path}")

    # ----- 2. per-case mean Dice (3-seed averaged) -----
    case_dice = per_case_mean_dice(rows)

    # ----- 3. pairwise Wilcoxon (6 pairs for 4 nets) + Holm-Bonferroni -----
    pairs = list(itertools.combinations(nets_3d, 2))
    test_rows = []
    pvals = []
    for a, b in pairs:
        common = sorted(set(case_dice.get(a, {})) & set(case_dice.get(b, {})))
        if len(common) < 5:
            test_rows.append({"net_a": a, "net_b": b, "n_cases": len(common),
                              "wilcoxon_stat": "", "p_raw": "", "p_holm": "",
                              "mean_a": "", "mean_b": "", "note": "insufficient cases"})
            pvals.append(1.0)
            continue
        xa = np.array([case_dice[a][c] for c in common])
        xb = np.array([case_dice[b][c] for c in common])
        diff = xa - xb
        if np.all(diff == 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
        test_rows.append({
            "net_a": a, "net_b": b, "n_cases": len(common),
            "wilcoxon_stat": f"{stat:.4f}", "p_raw": f"{p:.6g}",
            "mean_a": f"{np.mean(xa):.6f}", "mean_b": f"{np.mean(xb):.6f}",
        })
        pvals.append(p)

    adj = holm_bonferroni(pvals)
    for tr, a in zip(test_rows, adj):
        tr["p_holm"] = f"{a:.6g}"

    test_path = out_dir / "wilcoxon_pairs.csv"
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["net_a", "net_b", "n_cases", "mean_a", "mean_b",
                                          "wilcoxon_stat", "p_raw", "p_holm"])
        w.writeheader()
        for tr in test_rows:
            if "note" not in tr:
                w.writerow({k: tr.get(k, "") for k in
                            ["net_a", "net_b", "n_cases", "mean_a", "mean_b",
                             "wilcoxon_stat", "p_raw", "p_holm"]})
    print(f"[stats] pairwise Wilcoxon -> {test_path}")

    # ----- 4. per-network per-case mean Dice summary (mean±std across cases) -----
    case_summary_path = out_dir / "per_case_mean_dice.csv"
    with open(case_summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["network", "mean_case_dice", "std_case_dice", "n_cases"])
        w.writeheader()
        for net in networks_present:
            vals = np.array(list(case_dice.get(net, {}).values()))
            if len(vals) == 0:
                continue
            w.writerow({"network": net, "mean_case_dice": f"{np.mean(vals):.6f}",
                        "std_case_dice": f"{np.std(vals):.6f}", "n_cases": len(vals)})
    print(f"[stats] per-case mean Dice -> {case_summary_path}")

    # ----- markdown summary -----
    md = ["# v1 input-consistency — results summary", ""]
    md.append("## Per-network mean per-case Dice (3-seed averaged, across 39 test cases)")
    md.append("")
    md.append("| network | mean per-case Dice ± std |")
    md.append("|---|---|")
    for net in networks_present:
        vals = np.array(list(case_dice.get(net, {}).values()))
        if len(vals):
            md.append(f"| {net} | {fmt_pm(np.mean(vals), np.std(vals))} |")
    md.append("")
    md.append("## Pairwise Wilcoxon signed-rank on per-case mean Dice (Holm-Bonferroni)")
    md.append("")
    md.append("| pair | mean A | mean B | p (raw) | p (Holm-Bonferroni) |")
    md.append("|---|---|---|---|---|")
    for tr in test_rows:
        if "note" in tr:
            md.append(f"| {tr['net_a']} vs {tr['net_b']} | — | — | — | {tr.get('note','')} |")
        else:
            md.append(f"| {tr['net_a']} vs {tr['net_b']} | {tr['mean_a']} | {tr['mean_b']} | {tr['p_raw']} | {tr['p_holm']} |")
    md.append("")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[stats] markdown -> {out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
