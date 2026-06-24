#!/usr/bin/env python3
"""Paired statistics for conditional LR-mirror ablation (spec §7.2).

Compares baseline vs condlr per-case kidney Dice/HD95, using Wilcoxon signed-rank
with Holm-Bonferroni correction across 8 tests (2 networks × 2 kidney sides × 2 metrics).

Input: per_case.csv files from run_eval.py (30-column format).
Output: condLR_vs_baseline_paired_stats.csv + summary to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KIDNEY_CLASSES = {4: "left_kidney", 5: "right_kidney"}
METRICS = ["Dice", "HD95"]
NETWORKS = [
    ("swinunetr", "swinunetr_baseline", "swinunetr_condlr"),
    ("nnunet_2d", "nnunet_2d_baseline", "nnunet_2d_condlr"),
]


def load_and_average_seeds(results_dir: Path, network_label: str, seeds: list[int]) -> pd.DataFrame:
    """Load per_case.csv for multiple seeds and average Dice/HD95 per (case_id, class_id)."""
    dfs = []
    for seed in seeds:
        path = results_dir / f"{network_label}_seed{seed}_per_case.csv"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No per_case.csv found for {network_label}")

    combined = pd.concat(dfs, ignore_index=True)

    # Average Dice and HD95 across seeds, per (case_id, class_id)
    agg = combined.groupby(["case_id", "class_id", "class_name"]).agg(
        Dice_mean=("Dice", lambda x: np.nanmean(x)),
        HD95_mean=("HD95", lambda x: np.nanmean(x)),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    return agg


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (idx, p) in enumerate(indexed):
        adjusted[idx] = min(1.0, p * (n - rank))
        # Ensure monotonicity
        if rank > 0:
            prev_idx = indexed[rank - 1][0]
            adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


def run_paired_tests(
    results_dir: Path,
    seeds: list[int],
    output_csv: Path,
) -> None:
    """Run paired Wilcoxon tests for all network × kidney × metric combinations."""
    rows = []
    raw_pvalues = []
    test_keys = []

    for net_name, baseline_label, condlr_label in NETWORKS:
        baseline_df = load_and_average_seeds(results_dir, baseline_label, seeds)
        condlr_df = load_and_average_seeds(results_dir, condlr_label, seeds)

        # Merge on (case_id, class_id)
        merged = baseline_df.merge(
            condlr_df,
            on=["case_id", "class_id", "class_name"],
            suffixes=("_baseline", "_condlr"),
        )

        for class_id, class_name in KIDNEY_CLASSES.items():
            subset = merged[merged["class_id"] == class_id].copy()
            if len(subset) == 0:
                print(f"WARNING: no data for {net_name} class {class_name}", file=sys.stderr)
                continue

            for metric in METRICS:
                baseline_col = f"{metric}_mean_baseline"
                condlr_col = f"{metric}_mean_condlr"

                baseline_vals = subset[baseline_col].values
                condlr_vals = subset[condlr_col].values

                # Drop pairs where either is NaN
                valid = ~(np.isnan(baseline_vals) | np.isnan(condlr_vals))
                b = baseline_vals[valid]
                c = condlr_vals[valid]

                if len(b) < 5:
                    print(f"WARNING: too few valid pairs for {net_name} {class_name} {metric}: {len(b)}", file=sys.stderr)
                    continue

                # Paired differences
                diff = c - b
                mean_diff = float(np.mean(diff))
                median_diff = float(np.median(diff))

                # Wilcoxon signed-rank (one-sided, directional hypothesis)
                # Dice: condlr > baseline → alternative="less" (b - c < 0)
                # HD95: condlr < baseline → alternative="greater" (b - c > 0)
                alt = "less" if metric == "Dice" else "greater"
                try:
                    stat_result = stats.wilcoxon(b, c, alternative=alt, zero_method="wilcox")
                    w_stat = float(stat_result.statistic)
                    p_value = float(stat_result.pvalue)
                except ValueError:
                    w_stat = 0.0
                    p_value = 1.0

                # Effect size: r = z / sqrt(N)
                # Approximate z from W using normal approximation
                n = len(b)
                if n > 0:
                    mu_w = n * (n + 1) / 4
                    sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                    z = (w_stat - mu_w) / sigma_w if sigma_w > 0 else 0
                    effect_r = abs(z) / np.sqrt(n) if n > 0 else 0
                else:
                    effect_r = 0

                row = {
                    "network": net_name,
                    "kidney_side": class_name,
                    "class_id": class_id,
                    "metric": metric,
                    "n_pairs": len(b),
                    "baseline_mean": float(np.mean(b)),
                    "condlr_mean": float(np.mean(c)),
                    "mean_delta": mean_diff,
                    "median_delta": median_diff,
                    "wilcoxon_W": w_stat,
                    "p_value": p_value,
                    "effect_size_r": effect_r,
                    "holm_adjusted_p": None,  # filled later
                    "significant_005": None,  # filled later
                }
                rows.append(row)
                raw_pvalues.append(p_value)
                test_keys.append((net_name, class_name, metric))

    # Apply Holm-Bonferroni correction
    if raw_pvalues:
        adjusted = holm_bonferroni(raw_pvalues)
        for i, row in enumerate(rows):
            row["holm_adjusted_p"] = adjusted[i]
            row["significant_005"] = adjusted[i] < 0.05

    # Write output
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"\nResults written to: {output_csv}")

    # Print summary table
    print("\n" + "=" * 90)
    print("PAIRED STATISTICS SUMMARY")
    print("=" * 90)
    print(f"{'Network':<12} {'Kidney':<12} {'Metric':<6} {'N':>3} {'Baseline':>8} {'Condlr':>8} {'Δ':>8} {'p':>8} {'Holm-p':>8} {'Sig':>4}")
    print("-" * 90)
    for row in rows:
        sig = "✓" if row["significant_005"] else ""
        print(f"{row['network']:<12} {row['kidney_side']:<12} {row['metric']:<6} {row['n_pairs']:>3} "
              f"{row['baseline_mean']:>8.4f} {row['condlr_mean']:>8.4f} {row['mean_delta']:>+8.4f} "
              f"{row['p_value']:>8.4f} {row['holm_adjusted_p']:>8.4f} {sig:>4}")
    print("-" * 90)
    n_sig = sum(1 for r in rows if r["significant_005"])
    print(f"Significant (Holm-adjusted p < 0.05): {n_sig}/{len(rows)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired statistics for condlr ablation")
    ap.add_argument("--results-dir", default="evaluation/results",
                    help="directory containing per_case.csv files")
    ap.add_argument("--output-csv", default="evaluation/results/condlr_vs_baseline_paired_stats.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[20260520, 20260521, 20260522])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)

    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        return 1

    run_paired_tests(results_dir, args.seeds, output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
