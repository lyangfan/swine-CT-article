"""Check left/right kidney confusion (class 4 ↔ class 5 swap) in predictions.

For each (network, seed, test_case):
  - voxels where pred==4 & GT==5 (predicted left, actually right)
  - voxels where pred==5 & GT==4 (predicted right, actually left)
  - as fraction of total kidney voxels (GT 4 + GT 5)
"""
import csv, glob, os, sys
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from collections import defaultdict

ARTICLE = "/home/share/hzau/home/liuyangfan/swine-CT-article"
GT_DIR = f"{ARTICLE}/data/nnunetv1/nnUNet_raw_data/Task601_Article622_Carcass9Class/labelsTs"
PRED_ROOT = f"{ARTICLE}/data/nnunetv1/v1_comparison_predictions"

NETS = ["nnunet_v1","swinunetr","segformer3d","mednext_s","nnunet_2d"]
SEEDS = [20260520, 20260521, 20260522]

def check_one(args):
    net, seed, case_id = args
    pred_path = f"{PRED_ROOT}/{net}__seed{seed}/{case_id}.nii.gz"
    gt_path = f"{GT_DIR}/{case_id}.nii.gz"
    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
        return None
    pred = nib.load(pred_path).get_fdata().astype(np.int16)
    gt = nib.load(gt_path).get_fdata().astype(np.int16)
    if pred.shape != gt.shape:
        return None
    # kidney confusion
    l_as_r = int(np.sum((pred == 4) & (gt == 5)))  # pred left, gt right
    r_as_l = int(np.sum((pred == 5) & (gt == 4)))  # pred right, gt left
    gt_l = int(np.sum(gt == 4))
    gt_r = int(np.sum(gt == 5))
    total_kidney = gt_l + gt_r
    swapped = l_as_r + r_as_l
    swap_rate = swapped / total_kidney if total_kidney > 0 else 0.0
    has_swap = swapped > 0
    # also check: did the model put left kidney on the right side spatially?
    # simple check: centroid of pred-4 vs centroid of gt-4
    return {
        "network": net, "seed": seed, "case_id": case_id,
        "gt_l_voxels": gt_l, "gt_r_voxels": gt_r,
        "pred_l_as_r": l_as_r, "pred_r_as_l": r_as_l,
        "total_swapped": swapped, "swap_rate": swap_rate,
        "has_swap": has_swap,
    }

# enumerate all tasks
tasks = []
for net in NETS:
    for seed in SEEDS:
        for gf in sorted(glob.glob(f"{GT_DIR}/*.nii.gz")):
            cid = os.path.basename(gf).replace(".nii.gz", "")
            tasks.append((net, seed, cid))

print(f"checking {len(tasks)} (network, seed, case) combinations...")
results = []
with Pool(16) as pool:
    for i, r in enumerate(pool.imap_unordered(check_one, tasks)):
        if r:
            results.append(r)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(tasks)} done", flush=True)

# aggregate per network (pool 3 seeds)
print("\n## Left/Right Kidney Confusion (class 4 ↔ 5 swap)\n")
print("| network | cases with swap | total swapped voxels | mean swap rate | max swap voxels (1 case) |")
print("|---|---|---|---|---|")
for net in NETS:
    sub = [r for r in results if r["network"] == net]
    swap_cases = [r for r in sub if r["has_swap"]]
    total_swapped = sum(r["total_swapped"] for r in sub)
    mean_rate = np.mean([r["swap_rate"] for r in sub])
    max_swap = max(r["total_swapped"] for r in sub) if sub else 0
    n_cases_with_swap = len(set(r["case_id"] for r in swap_cases))
    print(f"| {net} | {n_cases_with_swap}/39 | {total_swapped} | {mean_rate:.6f} | {max_swap} |")

# detail: which cases have swaps across networks
print("\n## Cases with kidney swap (any network, any seed)\n")
swap_cases = defaultdict(list)
for r in results:
    if r["has_swap"]:
        swap_cases[r["case_id"]].append(f"{r['network']}/{str(r['seed'])[-2:]}:{r['total_swapped']}")
if swap_cases:
    print("| case_id | networks with swap (net/seed:voxels) |")
    print("|---|---|")
    for cid in sorted(swap_cases):
        detail = "; ".join(swap_cases[cid][:8])
        if len(swap_cases[cid]) > 8: detail += f" +{len(swap_cases[cid])-8} more"
        print(f"| {cid} | {detail} |")
else:
    print("No kidney swaps detected in any prediction.")
