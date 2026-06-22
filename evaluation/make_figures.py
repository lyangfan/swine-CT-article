#!/usr/bin/env python3
"""Generate ALL result figures from the locked evaluator output.

Reads evaluation/results_locked/*/case_metrics.csv (34 metrics per row).
Outputs figures/fig{1-10}*.png.
"""
import csv, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "evaluation" / "results_locked"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})

NAMES = {1:"front",2:"middle",3:"end",4:"left_kidney",5:"right_kidney",
         6:"testis",7:"thoracic",8:"abd_pelvic",9:"head"}
NETS = ["nnunet_v1","swinunetr","segformer3d","mednext_s","nnunet_2d"]
LABELS = {"nnunet_v1":"nnU-Net v1","swinunetr":"SwinUNETR","segformer3d":"SegFormer3D",
          "mednext_s":"MedNeXt-S","nnunet_2d":"nnU-Net 2D"}
SUBTITLE = {"nnunet_v1":"(3D CNN)","swinunetr":"(3D Transformer)","segformer3d":"(3D Transformer)",
            "mednext_s":"(3D CNN)","nnunet_2d":"(2D ref)"}
COLORS = {"nnunet_v1":"#2196F3","swinunetr":"#FF9800","segformer3d":"#4CAF50",
          "mednext_s":"#9C27B0","nnunet_2d":"#F44336"}
PARAMS = {"nnunet_v1":45.0,"swinunetr":72.8,"segformer3d":4.5,"mednext_s":10.6,"nnunet_2d":41.3}

def f(x):
    try: return float(x)
    except: return np.nan

# ---- load all data ----
rows = []
for path in sorted(glob.glob(str(RES / "*" / "case_metrics.csv"))):
    combo = Path(path).parent.name
    net = combo.split("__seed")[0]
    seed = int(combo.split("__seed")[1])
    for r in csv.DictReader(open(path)):
        r["_network"] = net
        r["_seed"] = seed
        rows.append(r)
print(f"loaded {len(rows)} rows from {len(glob.glob(str(RES/'*'/'case_metrics.csv')))} combos")

def metric_per_class(metric, evaluable_only=True):
    """Returns {network: {class_id: [values]}}."""
    out = {n: {} for n in NETS}
    for n in NETS:
        for c in range(1, 10):
            vals = [f(r[metric]) for r in rows if r["_network"]==n and int(r["class_id"])==c
                    and (not evaluable_only or r.get("is_evaluable")=="True")]
            vals = [v for v in vals if not np.isnan(v)]
            out[n][c] = vals
    return out

# ============ Fig 1: Overall Dice bar ============
pc_dice = {}
for n in NETS:
    case_means = {}
    for r in rows:
        if r["_network"]==n and r.get("is_evaluable")=="True":
            key = (r["_seed"], r["case_id"])
            case_means.setdefault(key, []).append(f(r["Dice"]))
    vals = [np.mean(v) for v in case_means.values()]
    # average across seeds
    seed_means = {}
    for (seed, cid), v in case_means.items():
        seed_means.setdefault(cid, []).append(np.mean(v))
    final = [np.mean(v) for v in seed_means.values()]
    pc_dice[n] = (np.mean(final), np.std(final))

fig, ax = plt.subplots(figsize=(10, 6))
order = ["nnunet_2d","swinunetr","nnunet_v1","segformer3d","mednext_s"]
bars = ax.bar(range(5), [pc_dice[n][0] for n in order], yerr=[pc_dice[n][1] for n in order],
              capsize=5, width=0.55, color=[COLORS[n] for n in order], edgecolor="black", linewidth=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels([f"{LABELS[n]}\n{SUBTITLE[n]}" for n in order], fontsize=11)
ax.set_ylabel("Mean per-case Dice (3-seed avg ± std)", fontsize=12)
ax.set_title("Overall segmentation quality", fontsize=13)
ax.set_ylim(0.90, 0.98)
for bar, n in zip(bars, order):
    ax.text(bar.get_x()+bar.get_width()/2, pc_dice[n][0]+0.001, f"{pc_dice[n][0]:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.savefig(FIG/"fig1_overall_dice.png"); plt.close()

# ============ Fig 2: Per-class Dice heatmap ============
dice_pc = metric_per_class("Dice")
mat = np.array([[np.mean(dice_pc[n][c]) if dice_pc[n][c] else np.nan for c in range(1,10)] for n in NETS])
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.80, vmax=0.99)
ax.set_yticks(range(5)); ax.set_yticklabels([LABELS[n] for n in NETS])
ax.set_xticks(range(9)); ax.set_xticklabels([NAMES[c+1] for c in range(9)], rotation=35, ha="right")
for i in range(5):
    for j in range(9):
        if not np.isnan(mat[i,j]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i,j]<0.89 else "black")
ax.set_title("Per-class Dice (mean, 3 seeds)")
plt.colorbar(im, ax=ax, label="Dice")
fig.savefig(FIG/"fig2_dice_heatmap.png"); plt.close()

# ============ Fig 3: Per-class HD95 heatmap ============
hd95_pc = metric_per_class("HD95")
mat_h = np.array([[np.median(hd95_pc[n][c]) if hd95_pc[n][c] else np.nan for c in range(1,10)] for n in NETS])
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(mat_h, aspect="auto", cmap="RdYlGn_r", vmin=2, vmax=30)
ax.set_yticks(range(5)); ax.set_yticklabels([LABELS[n] for n in NETS])
ax.set_xticks(range(9)); ax.set_xticklabels([NAMES[c+1] for c in range(9)], rotation=35, ha="right")
for i in range(5):
    for j in range(9):
        if not np.isnan(mat_h[i,j]):
            ax.text(j, i, f"{mat_h[i,j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if mat_h[i,j]>18 else "black")
ax.set_title("Per-class HD95 median mm (lower = better)")
plt.colorbar(im, ax=ax, label="HD95 (mm)")
fig.savefig(FIG/"fig3_hd95_heatmap.png"); plt.close()

# ============ Fig 4: Kidney Dice box ============
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for idx, (c, title) in enumerate([(4,"left_kidney"),(5,"right_kidney")]):
    ax = axes[idx]
    data = [dice_pc[n][c] for n in NETS]
    bp = ax.boxplot(data, tick_labels=[f"{LABELS[n]}\n{SUBTITLE[n]}" for n in NETS],
                    showfliers=True, patch_artist=True, widths=0.45)
    for patch, n in zip(bp["boxes"], NETS):
        patch.set_facecolor(COLORS[n]); patch.set_alpha(0.7)
    ax.set_title(f"{title} — Dice (n=117)", fontsize=12)
    ax.set_ylabel("Dice", fontsize=11); ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", labelsize=9, rotation=20)
    for lbl in ax.get_xticklabels(): lbl.set_ha("right")
    for i, d in enumerate(data):
        ax.text(i+1, np.percentile(d,10)-0.05, f"P10={np.percentile(d,10):.2f}",
                ha="center", fontsize=8, color="red", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
fig.savefig(FIG/"fig4_kidney_dice_box.png"); plt.close()

# ============ Fig 5: Absent-FP ============
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for idx, (cls, title) in enumerate([(9,"head FP on TB (absent)"),(6,"testis FP on HZAU (absent)")]):
    ax = axes[idx]
    nets_data = []
    for n in NETS:
        sub = [r for r in rows if r["_network"]==n and int(r["class_id"])==cls and r.get("is_evaluable")!="True"]
        vox = [f(r.get("absent_FP_voxels", "nan")) for r in sub]
        inci = [f(r.get("absent_FP_incidence", "nan")) for r in sub]
        nets_data.append((n, np.nanmean(vox) if vox else 0, np.nanmean(inci)*100 if inci else 0))
    nets_data.sort(key=lambda x: x[1])
    x = range(len(nets_data))
    bars = ax.bar(x, [d[1] for d in nets_data], color=[COLORS[d[0]] for d in nets_data], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[d[0]] for d in nets_data], fontsize=9, rotation=20)
    for lbl in ax.get_xticklabels(): lbl.set_ha("right")
    ax.set_ylabel("Mean FP voxels")
    ax.set_title(title, fontsize=11)
    for bar, d in zip(bars, nets_data):
        ax.text(bar.get_x()+bar.get_width()/2, d[1]+max(xd[1] for xd in nets_data)*0.02,
                f"{d[1]:.0f}\n({d[2]:.0f}%)", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Conditional-class hallucination (FP on cases where class is absent)", fontsize=11)
fig.savefig(FIG/"fig5_absent_fp.png"); plt.close()

# ============ Fig 6: Wilcoxon matrix (from existing results) ============
from scipy.stats import wilcoxon
import itertools
wp_path = ROOT / "evaluation" / "results" / "wilcoxon_pairs.csv"
if wp_path.exists():
    wp = list(csv.DictReader(open(wp_path)))
    nets3d = ["nnunet_v1","swinunetr","segformer3d","mednext_s"]
    nn = len(nets3d)
    pmat = np.full((nn, nn), np.nan)
    for r in wp:
        a, b = r["net_a"], r["net_b"]
        if a in nets3d and b in nets3d:
            pmat[nets3d.index(a), nets3d.index(b)] = float(r["p_holm"])
            pmat[nets3d.index(b), nets3d.index(a)] = float(r["p_holm"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pmat, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(nn)); ax.set_xticklabels([LABELS[x] for x in nets3d], rotation=30, ha="right")
    ax.set_yticks(range(nn)); ax.set_yticklabels([LABELS[x] for x in nets3d])
    for i in range(nn):
        for j in range(nn):
            if i != j and not np.isnan(pmat[i,j]):
                p = pmat[i,j]
                sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
                ax.text(j, i, f"{p:.1e}\n{sig}", ha="center", va="center", fontsize=8,
                        color="white" if p<0.1 else "black")
            elif i==j: ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")
    ax.set_title("Wilcoxon p-values (Holm-Bonferroni)\n*** p<0.001  ** p<0.01  * p<0.05  ns=not sig")
    plt.colorbar(im, ax=ax, label="p (Holm-Bonferroni)")
    fig.savefig(FIG/"fig6_wilcoxon_matrix.png"); plt.close()

# ====== NEW FIGURES (locked evaluator metrics) ======

# ============ Fig 7: Parameter efficiency scatter ============
fig, ax = plt.subplots(figsize=(9, 6))
for n in NETS:
    ax.scatter(PARAMS[n], pc_dice[n][0], s=200, c=COLORS[n], edgecolors="black", linewidth=1, zorder=5)
    ax.annotate(f"{LABELS[n]}\n{PARAMS[n]:.1f}M → {pc_dice[n][0]:.3f}",
                (PARAMS[n], pc_dice[n][0]), textcoords="offset points", xytext=(12, 8), fontsize=9)
ax.set_xlabel("Parameters (M)", fontsize=12)
ax.set_ylabel("Mean per-case Dice", fontsize=12)
ax.set_title("Parameter efficiency — accuracy vs model size", fontsize=13)
ax.set_xscale("log")
ax.set_xlim(3, 100)
ax.grid(alpha=0.3)
fig.savefig(FIG/"fig7_param_efficiency.png"); plt.close()

# ============ Fig 8: Precision-Recall per class ============
prec_pc = metric_per_class("Precision")
rec_pc = metric_per_class("Recall")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for idx, (metric_pc, title, vmin, vmax) in enumerate([
    (prec_pc, "Precision (TP / (TP+FP), fewer false positives)", 0.85, 1.0),
    (rec_pc, "Recall (TP / (TP+FN), fewer false negatives)", 0.85, 1.0)]):
    ax = axes[idx]
    mat = np.array([[np.mean(metric_pc[n][c]) if metric_pc[n][c] else np.nan for c in range(1,10)] for n in NETS])
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(5)); ax.set_yticklabels([LABELS[n] for n in NETS])
    ax.set_xticks(range(9)); ax.set_xticklabels([NAMES[c+1] for c in range(9)], rotation=35, ha="right")
    for i in range(5):
        for j in range(9):
            if not np.isnan(mat[i,j]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i,j]<0.92 else "black")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.savefig(FIG/"fig8_precision_recall_heatmap.png"); plt.close()

# ============ Fig 9: HD95 box plot ALL classes ============
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for idx, c in enumerate(range(1, 10)):
    ax = axes[idx // 3][idx % 3]
    data = [hd95_pc[n][c] for n in NETS]
    bp = ax.boxplot(data, tick_labels=[LABELS[n] for n in NETS], showfliers=False,
                    patch_artist=True, widths=0.5)
    for patch, n in zip(bp["boxes"], NETS):
        patch.set_facecolor(COLORS[n]); patch.set_alpha(0.7)
    ax.set_title(NAMES[c], fontsize=10, fontweight="bold")
    ax.set_ylabel("HD95 (mm)" if idx % 3 == 0 else "")
    ax.tick_params(axis="x", labelsize=7, rotation=25)
    for lbl in ax.get_xticklabels(): lbl.set_ha("right")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("HD95 distribution per class (all 9 classes, 3 seeds pooled, outliers hidden)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG/"fig9_hd95_box_all_classes.png"); plt.close()

# ============ Fig 10: Dice P10 heatmap (tail percentile) ============
mat_p10 = np.array([[np.percentile(dice_pc[n][c], 10) if dice_pc[n][c] else np.nan
                     for c in range(1,10)] for n in NETS])
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(mat_p10, aspect="auto", cmap="RdYlGn", vmin=0.60, vmax=0.98)
ax.set_yticks(range(5)); ax.set_yticklabels([LABELS[n] for n in NETS])
ax.set_xticks(range(9)); ax.set_xticklabels([NAMES[c+1] for c in range(9)], rotation=35, ha="right")
for i in range(5):
    for j in range(9):
        if not np.isnan(mat_p10[i,j]):
            ax.text(j, i, f"{mat_p10[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if mat_p10[i,j]<0.80 else "black")
ax.set_title("Dice P10 (10th percentile — worst-case tail, higher = better)")
plt.colorbar(im, ax=ax, label="Dice P10")
fig.savefig(FIG/"fig10_dice_p10_heatmap.png"); plt.close()

# ============ Fig 11: Kidney left/right confusion ============
ks_path = RES / "kidney_swap.csv"
if ks_path.exists():
    ks_rows = list(csv.DictReader(open(ks_path)))
    # aggregate per network: mean swap_rate, mean lp_dice_gap, swap case incidence
    ks_summary = {}
    for n in NETS:
        sub = [r for r in ks_rows if r["network"] == n]
        if not sub:
            continue
        # per-case (3-seed avg) then network-level
        case_swap = {}
        case_gap = {}
        for r in sub:
            case_swap.setdefault(r["case_id"], []).append(f(r["swap_rate"]))
            case_gap.setdefault(r["case_id"], []).append(f(r["lp_dice_gap"]))
        swap_vals = [np.mean(v) for v in case_swap.values()]
        gap_vals = [np.mean(v) for v in case_gap.values()]
        swap_incidence = np.mean([1 if np.mean(v) > 0 else 0 for v in case_swap.values()])
        ks_summary[n] = {
            "swap_rate": np.mean(swap_vals),
            "swap_rate_std": np.std(swap_vals),
            "lp_gap": np.mean(gap_vals),
            "lp_gap_std": np.std(gap_vals),
            "incidence": swap_incidence,
        }

    if ks_summary:
        nets_ks = [n for n in NETS if n in ks_summary]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # panel 1: swap rate
        ax = axes[0]
        x = range(len(nets_ks))
        bars = ax.bar(x, [ks_summary[n]["swap_rate"] for n in nets_ks],
                      yerr=[ks_summary[n]["swap_rate_std"] for n in nets_ks], capsize=4,
                      color=[COLORS[n] for n in nets_ks], edgecolor="black", linewidth=0.5, width=0.6)
        ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in nets_ks], rotation=20, fontsize=9)
        for lbl in ax.get_xticklabels(): lbl.set_ha("right")
        ax.set_ylabel("Swap rate (fraction of kidney voxels)")
        ax.set_title("Kidney swap rate\n(lower = less L/R confusion)", fontsize=11)
        for bar, n in zip(bars, nets_ks):
            ax.text(bar.get_x()+bar.get_width()/2, ks_summary[n]["swap_rate"]+0.002,
                    f"{ks_summary[n]['swap_rate']:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # panel 2: LP-Dice gap
        ax = axes[1]
        bars = ax.bar(x, [ks_summary[n]["lp_gap"] for n in nets_ks],
                      yerr=[ks_summary[n]["lp_gap_std"] for n in nets_ks], capsize=4,
                      color=[COLORS[n] for n in nets_ks], edgecolor="black", linewidth=0.5, width=0.6)
        ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in nets_ks], rotation=20, fontsize=9)
        for lbl in ax.get_xticklabels(): lbl.set_ha("right")
        ax.set_ylabel("merged_Dice - split_Dice")
        ax.set_title("Laterality-preserving Dice gap\n(higher = more Dice lost from L/R confusion)", fontsize=11)
        for bar, n in zip(bars, nets_ks):
            ax.text(bar.get_x()+bar.get_width()/2, ks_summary[n]["lp_gap"]+0.002,
                    f"{ks_summary[n]['lp_gap']:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # panel 3: confusion case incidence
        ax = axes[2]
        bars = ax.bar(x, [ks_summary[n]["incidence"]*100 for n in nets_ks],
                      color=[COLORS[n] for n in nets_ks], edgecolor="black", linewidth=0.5, width=0.6)
        ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in nets_ks], rotation=20, fontsize=9)
        for lbl in ax.get_xticklabels(): lbl.set_ha("right")
        ax.set_ylabel("Cases with any swap (%)")
        ax.set_title("Confusion incidence\n(% of 39 cases with ≥1 swapped voxel)", fontsize=11)
        for bar, n in zip(bars, nets_ks):
            ax.text(bar.get_x()+bar.get_width()/2, ks_summary[n]["incidence"]*100+1,
                    f"{ks_summary[n]['incidence']*100:.0f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Kidney left/right confusion (class 4 ↔ 5) — 3 metrics", fontsize=13)
        fig.savefig(FIG/"fig11_kidney_swap.png"); plt.close()

print(f"\nFigures generated ({len(list(FIG.glob('*.png')))} total):")
for p in sorted(FIG.glob("*.png")):
    print(f"  {p.name} ({p.stat().st_size//1024} KB)")
