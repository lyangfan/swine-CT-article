#!/usr/bin/env python3
"""Generate all result figures for the v1 input-consistency comparison.
Reads evaluation/results/*.csv, writes figures/*.png."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "evaluation" / "results"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})

NAMES = {1:"front",2:"middle",3:"end",4:"left_kidney",5:"right_kidney",
         6:"testis",7:"thoracic",8:"abd_pelvic",9:"head"}
NETS = ["nnunet_v1","swinunetr","segformer3d","mednext_s","nnunet_2d"]
LABELS = {"nnunet_v1":"nnU-Net v1","swinunetr":"SwinUNETR",
          "segformer3d":"SegFormer3D","mednext_s":"MedNeXt-S",
          "nnunet_2d":"nnU-Net 2D"}
SUBTITLE = {"nnunet_v1":"(3D CNN)","swinunetr":"(3D Transformer)",
            "segformer3d":"(3D Transformer)","mednext_s":"(3D CNN)",
            "nnunet_2d":"(2D ref)"}
COLORS = {"nnunet_v1":"#2196F3","swinunetr":"#FF9800","segformer3d":"#4CAF50",
          "mednext_s":"#9C27B0","nnunet_2d":"#F44336"}

def f(x):
    try: return float(x)
    except: return np.nan

rows = list(csv.DictReader(open(RES/"per_case.csv")))

# ============ Fig 1: Overall mean per-case Dice (bar) ============
pc = {}
for r in csv.DictReader(open(RES/"per_case_mean_dice.csv")):
    pc[r["network"]] = (float(r["mean_case_dice"]), float(r["std_case_dice"]))
fig, ax = plt.subplots(figsize=(10, 6))
order = ["nnunet_2d","swinunetr","nnunet_v1","segformer3d","mednext_s"]
means = [pc[n][0] for n in order]
stds = [pc[n][1] for n in order]
bars = ax.bar(range(len(order)), means, yerr=stds, capsize=5, width=0.55,
              color=[COLORS[n] for n in order], edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{LABELS[n]}\n{SUBTITLE[n]}" for n in order], fontsize=11)
ax.set_ylabel("Mean per-case Dice (3-seed avg ± std)", fontsize=12)
ax.set_title("Overall segmentation quality — per-case mean Dice (39 test cases)", fontsize=13)
ax.set_ylim(0.90, 0.98)
for bar, m, n in zip(bars, means, order):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.001, f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.savefig(FIG/"fig1_overall_dice.png"); plt.close()

# ============ Fig 2: Per-class Dice heatmap ============
mat = np.full((len(NETS), 9), np.nan)
for i, n in enumerate(NETS):
    for c in range(1, 10):
        d = np.array([f(r["dice"]) for r in rows if r["network"]==n and int(r["class_label"])==c and r["status"]=="present"])
        d = d[~np.isnan(d)]
        mat[i, c-1] = np.mean(d) if len(d) else np.nan
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.80, vmax=0.99)
ax.set_yticks(range(len(NETS))); ax.set_yticklabels([LABELS[n] for n in NETS])
ax.set_xticks(range(9)); ax.set_xticklabels([NAMES[c+1] for c in range(9)], rotation=35, ha="right")
for i in range(len(NETS)):
    for j in range(9):
        if not np.isnan(mat[i,j]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i,j]<0.89 else "black")
ax.set_title("Per-class Dice (mean, 3 seeds × 39 test)")
plt.colorbar(im, ax=ax, label="Dice")
fig.savefig(FIG/"fig2_dice_heatmap.png"); plt.close()

# ============ Fig 3: Per-class HD95 heatmap ============
mat_h = np.full((len(NETS), 9), np.nan)
for i, n in enumerate(NETS):
    for c in range(1, 10):
        h = np.array([f(r["hd95"]) for r in rows if r["network"]==n and int(r["class_label"])==c and r["status"]=="present"])
        h = h[~np.isnan(h)]
        mat_h[i, c-1] = np.median(h) if len(h) else np.nan  # median (robust to tail)
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(mat_h, aspect="auto", cmap="RdYlGn_r", vmin=2, vmax=30)
ax.set_yticks(range(len(NETS))); ax.set_yticklabels([LABELS[n] for n in NETS])
ax.set_xticks(range(9)); ax.set_xticklabels([NAMES[c+1] for c in range(9)], rotation=35, ha="right")
for i in range(len(NETS)):
    for j in range(9):
        if not np.isnan(mat_h[i,j]):
            ax.text(j, i, f"{mat_h[i,j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if mat_h[i,j]>18 else "black")
ax.set_title("Per-class HD95 median mm (lower = better)")
plt.colorbar(im, ax=ax, label="HD95 (mm)")
fig.savefig(FIG/"fig3_hd95_heatmap.png"); plt.close()

# ============ Fig 4: Kidney Dice distribution (box plot) ============
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for idx, (c, title) in enumerate([(4,"left_kidney"),(5,"right_kidney")]):
    ax = axes[idx]
    data = []
    for n in NETS:
        d = np.array([f(r["dice"]) for r in rows if r["network"]==n and int(r["class_label"])==c and r["status"]=="present"])
        data.append(d[~np.isnan(d)])
    bp = ax.boxplot(data, tick_labels=[f"{LABELS[n]}\n{SUBTITLE[n]}" for n in NETS],
                    showfliers=True, patch_artist=True, widths=0.45)
    for patch, n in zip(bp["boxes"], NETS):
        patch.set_facecolor(COLORS[n]); patch.set_alpha(0.7)
    ax.set_title(f"{title} — Dice distribution (n=117)", fontsize=12)
    ax.set_ylabel("Dice", fontsize=11); ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    # annotate P10
    for i, d in enumerate(data):
        ax.text(i+1, np.percentile(d,10)-0.05, f"P10={np.percentile(d,10):.2f}",
                ha="center", fontsize=8, color="red", fontweight="bold")
fig.suptitle("Kidney Dice tail behavior — box = IQR, whisker = [P10, P90], red P10 annotated", fontsize=11)
fig.savefig(FIG/"fig4_kidney_dice_box.png"); plt.close()

# ============ Fig 5: Absent-FP hallucination ============
afp = list(csv.DictReader(open(RES/"absent_fp_summary.csv")))
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for idx, (cls, title) in enumerate([(9, "head FP on TB cases\n(head absent — should be 0)"),
                                     (6, "testis FP on HZAU cases\n(testis absent — should be 0)")]):
    ax = axes[idx]
    sub = sorted([r for r in afp if int(r["class_label"])==cls], key=lambda r: float(r["mean_fp_voxels"]))
    nets_o = [r["network"] for r in sub]
    vox = [float(r["mean_fp_voxels"]) for r in sub]
    inci = [float(r["fp_incidence"])*100 for r in sub]
    x = range(len(nets_o))
    bars = ax.bar(x, vox, color=[COLORS[n] for n in nets_o], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in nets_o], fontsize=9)
    ax.set_ylabel("Mean FP voxels")
    ax.set_title(title, fontsize=10)
    for bar, v, ic in zip(bars, vox, inci):
        ax.text(bar.get_x()+bar.get_width()/2, v+max(vox)*0.02, f"{v:.0f}\n({ic:.0f}% inc.)",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Conditional-class hallucination — false-positive on cases where the class is naturally absent", fontsize=11)
fig.savefig(FIG/"fig5_absent_fp.png"); plt.close()

# ============ Fig 6: Wilcoxon significance matrix ============
wp = list(csv.DictReader(open(RES/"wilcoxon_pairs.csv")))
nets3d = ["nnunet_v1","swinunetr","segformer3d","mednext_s"]
n = len(nets3d)
pmat = np.full((n, n), np.nan)
for r in wp:
    a, b = r["net_a"], r["net_b"]
    if a in nets3d and b in nets3d:
        ia, ib = nets3d.index(a), nets3d.index(b)
        p = float(r["p_holm"])
        pmat[ia, ib] = p; pmat[ib, ia] = p
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(pmat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(n)); ax.set_xticklabels([LABELS[x] for x in nets3d], rotation=30, ha="right")
ax.set_yticks(range(n)); ax.set_yticklabels([LABELS[x] for x in nets3d])
for i in range(n):
    for j in range(n):
        if i != j and not np.isnan(pmat[i,j]):
            p = pmat[i,j]
            sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
            ax.text(j, i, f"{p:.1e}\n{sig}", ha="center", va="center", fontsize=8,
                    color="white" if p<0.1 else "black")
        elif i == j:
            ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")
ax.set_title("Wilcoxon signed-rank p-values (Holm-Bonferroni corrected)\n*** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
plt.colorbar(im, ax=ax, label="p (Holm-Bonferroni)")
fig.savefig(FIG/"fig6_wilcoxon_matrix.png"); plt.close()

print("Figures generated:")
for p in sorted(FIG.glob("*.png")):
    print(f"  {p.name} ({p.stat().st_size//1024} KB)")
