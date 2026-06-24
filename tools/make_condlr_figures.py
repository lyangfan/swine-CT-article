#!/usr/bin/env python3
"""Generate figures for the conditional LR-mirror ablation results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("/home/share/hzau/home/liuyangfan/swine-CT-article/evaluation/results")
OUTPUT_DIR = Path("/home/share/hzau/home/liuyangfan/swine-CT-article/evaluation/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [20260520, 20260521, 20260522]
KIDNEY = {4: "left_kidney", 5: "right_kidney"}


def load_data():
    """Load and average per-case data across 3 seeds."""
    data = {}
    for prefix in ["swinunetr_baseline", "swinunetr_condlr", "nnunet_2d_baseline", "nnunet_2d_condlr"]:
        dfs = []
        for seed in SEEDS:
            path = RESULTS_DIR / f"{prefix}_seed{seed}_per_case.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["seed"] = seed
                dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        agg = combined.groupby(["case_id", "class_id"]).agg(
            Dice=("Dice", lambda x: np.nanmean(x)),
            HD95=("HD95", lambda x: np.nanmean(x)),
        ).reset_index()
        data[prefix] = agg
    return data


def plot_kidney_bar(data):
    """Bar chart: baseline vs condlr kidney Dice and HD95."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for col, metric in enumerate(["Dice", "HD95"]):
        for row, class_id in enumerate([4, 5]):
            ax = axes[row][col]
            kidney_name = KIDNEY[class_id]

            swin_b = data["swinunetr_baseline"]
            swin_c = data["swinunetr_condlr"]
            d2_b = data["nnunet_2d_baseline"]
            d2_c = data["nnunet_2d_condlr"]

            vals = []
            labels = []
            for net, b_df, c_df in [("SwinUNETR", swin_b, swin_c), ("2D nnUNet", d2_b, d2_c)]:
                b = b_df[b_df["class_id"] == class_id][metric]
                c = c_df[c_df["class_id"] == class_id][metric]
                vals.extend([b.mean(), c.mean()])
                labels.extend([f"{net}\nbaseline", f"{net}\ncondlr"])

            colors = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"]
            bars = ax.bar(range(4), vals, color=colors, edgecolor="black", linewidth=0.5)

            # Error bars (SEM across cases)
            errs = []
            for net, b_df, c_df in [("SwinUNETR", swin_b, swin_c), ("2D nnUNet", d2_b, d2_c)]:
                b = b_df[b_df["class_id"] == class_id][metric]
                c = c_df[c_df["class_id"] == class_id][metric]
                errs.extend([b.std() / np.sqrt(len(b)), c.std() / np.sqrt(len(c))])
            ax.errorbar(range(4), vals, yerr=errs, fmt="none", ecolor="black", capsize=3)

            ax.set_xticks(range(4))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(metric)
            ax.set_title(f"{kidney_name}", fontsize=11)

            if metric == "Dice":
                ax.set_ylim(0.88, 0.98)

    fig.suptitle("Kidney Dice and HD95: Baseline vs Conditional LR-Mirror", fontsize=13)
    plt.tight_layout()
    path = OUTPUT_DIR / "kidney_bar_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_kidney_box(data):
    """Box plot of paired differences (condlr - baseline)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, metric in enumerate(["Dice", "HD95"]):
        ax = axes[col]
        diffs = []
        labels = []
        for net, b_label, c_label in [("SwinUNETR", "swinunetr_baseline", "swinunetr_condlr"),
                                       ("2D nnUNet", "nnunet_2d_baseline", "nnunet_2d_condlr")]:
            for class_id in [4, 5]:
                b = data[b_label]
                c = data[c_label]
                merged = b.merge(c, on=["case_id", "class_id"], suffixes=("_b", "_c"))
                subset = merged[merged["class_id"] == class_id]
                diff = subset[f"{metric}_c"] - subset[f"{metric}_b"]
                diffs.append(diff.values)
                labels.append(f"{net}\n{KIDNEY[class_id].replace('_kidney','')}")

        bp = ax.boxplot(diffs, labels=labels, patch_artist=True, widths=0.6)
        colors = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylabel(f"Δ {metric} (condlr - baseline)")
        ax.set_title(f"Paired Δ {metric}", fontsize=11)

    fig.suptitle("Kidney Metric Changes: Conditional vs Baseline", fontsize=13)
    plt.tight_layout()
    path = OUTPUT_DIR / "kidney_box_diff.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_scatter_hd95(data):
    """Scatter: baseline HD95 vs condlr HD95 per case."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, (net, b_label, c_label, title) in enumerate([
        ("SwinUNETR", "swinunetr_baseline", "swinunetr_condlr", "SwinUNETR"),
        ("2D nnUNet", "nnunet_2d_baseline", "nnunet_2d_condlr", "nnU-Net 2D"),
    ]):
        ax = axes[col]
        for class_id, color in [(4, "#1f77b4"), (5, "#ff7f0e")]:
            b = data[b_label]
            c = data[c_label]
            merged = b.merge(c, on=["case_id", "class_id"], suffixes=("_b", "_c"))
            subset = merged[merged["class_id"] == class_id]
            ax.scatter(subset["HD95_b"], subset["HD95_c"], alpha=0.5, s=20, color=color,
                       label=KIDNEY[class_id].replace("_kidney", ""))

        lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Baseline HD95")
        ax.set_ylabel("Condlr HD95")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Per-Case HD95: Baseline vs Conditional", fontsize=13)
    plt.tight_layout()
    path = OUTPUT_DIR / "hd95_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_swap_rate():
    """Bar chart of swap rate comparison (per-seed means with SEM)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    labels_list = []
    means = []
    sems = []
    colors_list = []

    for csv_name, net_label in [("swinunetr_kidney_swap.csv", "SwinUNETR"),
                                 ("nnunet_2d_kidney_swap.csv", "nnU-Net 2D")]:
        path = RESULTS_DIR / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Compute per-seed swap rate
        per_seed = df.groupby(["network", "seed"]).agg(
            swap_rate=("swap_rate", "mean"),
        ).reset_index()

        for arm, arm_label in [("baseline", "baseline"), ("condlr", "condlr")]:
            net_name = [n for n in per_seed["network"].unique() if ("condlr" in n) == (arm == "condlr")]
            if not net_name:
                continue
            sub = per_seed[per_seed["network"] == net_name[0]]
            seed_means = sub["swap_rate"].values * 100
            labels_list.append(f"{net_label}\n{arm}")
            means.append(np.mean(seed_means))
            sems.append(np.std(seed_means) / np.sqrt(len(seed_means)))
            colors_list.append("#DD8452" if arm == "condlr" else "#4C72B0")

    x = range(len(labels_list))
    ax.bar(x, means, color=colors_list, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_list, fontsize=8)

    ax.set_ylabel("Mean Swap Rate (%)")
    ax.set_title("Kidney Left/Right Swap Rate: Baseline vs Conditional")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = OUTPUT_DIR / "swap_rate_bar.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_kidney_violin(data):
    """Violin plot: per-case kidney Dice and HD95 distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, metric in enumerate(["Dice", "HD95"]):
        for row, (net, b_label, c_label, net_name) in enumerate([
            ("swinunetr", "swinunetr_baseline", "swinunetr_condlr", "SwinUNETR"),
            ("nnunet_2d", "nnunet_2d_baseline", "nnunet_2d_condlr", "nnU-Net 2D"),
        ]):
            ax = axes[row][col]
            b_df = data[b_label]
            c_df = data[c_label]

            positions = []
            violin_data = []
            tick_labels = []
            colors = []

            for i, (class_id, kidney_name) in enumerate([(4, "left"), (5, "right")]):
                b_vals = b_df[b_df["class_id"] == class_id][metric].dropna().values
                c_vals = c_df[c_df["class_id"] == class_id][metric].dropna().values
                positions.extend([i * 3, i * 3 + 1])
                violin_data.extend([b_vals, c_vals])
                tick_labels.extend([f"{kidney_name}\nbaseline", f"{kidney_name}\ncondlr"])
                colors.extend(["#4C72B0", "#DD8452"])

            parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)

            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_ylabel(metric)
            ax.set_title(net_name, fontsize=11)

            if metric == "HD95":
                ax.set_ylim(bottom=0)

    fig.suptitle("Kidney Dice and HD95 Distribution: Baseline vs Conditional", fontsize=13)
    plt.tight_layout()
    path = OUTPUT_DIR / "kidney_violin.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_diff_violin():
    """Violin plot of paired differences (condlr - baseline)."""
    data = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, metric in enumerate(["Dice", "HD95"]):
        ax = axes[col]
        diffs = []
        labels = []

        for net_label, net_name in [("swinunetr", "SwinUNETR"), ("nnunet_2d", "nnU-Net 2D")]:
            b_df = data[f"{net_label}_baseline"]
            c_df = data[f"{net_label}_condlr"]
            for class_id, kidney_name in [(4, "left"), (5, "right")]:
                merged = b_df[b_df["class_id"] == class_id].merge(
                    c_df[c_df["class_id"] == class_id], on="case_id", suffixes=("_b", "_c"))
                diff = merged[f"{metric}_c"] - merged[f"{metric}_b"]
                diffs.append(diff.dropna().values)
                labels.append(f"{net_name}\n{kidney_name}")

        positions = range(len(diffs))
        parts = ax.violinplot(diffs, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor("#4C72B0" if i < 2 else "#DD8452")
            pc.set_alpha(0.6)

        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"Δ {metric} (condlr - baseline)")
        ax.set_title(f"Paired Δ {metric}", fontsize=11)

    fig.suptitle("Kidney Metric Changes Distribution", fontsize=13)
    plt.tight_layout()
    path = OUTPUT_DIR / "kidney_diff_violin.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    data = load_data()
    plot_kidney_bar(data)
    plot_kidney_box(data)
    plot_kidney_violin(data)
    plot_diff_violin()
    plot_scatter_hd95(data)
    plot_swap_rate()
    print("Done.")
