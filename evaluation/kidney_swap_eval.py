#!/usr/bin/env python3
"""Kidney left/right confusion metrics (class 4 ↔ 5 swap).

Three metrics per (network, seed, case), requiring cross-class voxel comparison
(the locked evaluator evaluates each class independently and cannot detect swaps):

1. **swap_rate**: (pred==4&GT==5 + pred==5&GT==4) / (GT==4 + GT==5)
   — fraction of kidney voxels with wrong laterality label.

2. **lp_dice_gap** (laterality-preserving Dice gap):
   Dice(merged{4,5}) - mean(Dice_4, Dice_5)
   — how much Dice is lost PURELY from left/right confusion.
   High merged Dice + low split Dice = good detection, poor laterality.

3. **has_swap**: swap_voxels > 0 (binary, for case-incidence counting).

Output: appends to --output-csv. Parallelized via --num-workers.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from multiprocessing import Pool


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.sum(), b.sum()
    if sa + sb == 0:
        return 1.0
    return 2.0 * (a & b).sum() / (sa + sb)


def eval_one(args):
    pred_path, gt_path, network, seed, case_id = args
    try:
        pred = nib.load(pred_path).get_fdata().astype(np.int16)
        gt = nib.load(gt_path).get_fdata().astype(np.int16)
    except Exception as exc:
        print(f"[warn] {case_id}: {exc}", file=sys.stderr)
        return None
    if pred.shape != gt.shape:
        return None

    # voxel-level swap
    swap_lr = int(np.sum((pred == 4) & (gt == 5)))   # pred left, gt right
    swap_rl = int(np.sum((pred == 5) & (gt == 4)))   # pred right, gt left
    total_gt_kidney = int(np.sum((gt == 4) | (gt == 5)))
    swap_rate = (swap_lr + swap_rl) / total_gt_kidney if total_gt_kidney > 0 else 0.0

    # laterality-preserving dice gap
    merged_pred = (pred == 4) | (pred == 5)
    merged_gt = (gt == 4) | (gt == 5)
    merged_dice = _dice(merged_pred, merged_gt)
    d_left = _dice(pred == 4, gt == 4)
    d_right = _dice(pred == 5, gt == 5)
    split_dice = (d_left + d_right) / 2.0
    lp_dice_gap = merged_dice - split_dice

    return {
        "network": network, "seed": seed, "case_id": case_id,
        "swap_rate": f"{swap_rate:.6f}",
        "lp_dice_gap": f"{lp_dice_gap:.6f}",
        "merged_dice": f"{merged_dice:.6f}",
        "split_dice": f"{split_dice:.6f}",
        "swap_voxels": swap_lr + swap_rl,
        "has_swap": int(swap_lr + swap_rl > 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gt-folder", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    pred_folder = Path(args.predictions)
    gt_folder = Path(args.gt_folder)
    pred_files = sorted(pred_folder.glob("*.nii.gz"))
    tasks = []
    for pf in pred_files:
        cid = pf.name.replace(".nii.gz", "")
        gf = gt_folder / pf.name
        if not gf.exists():
            gf = gt_folder / f"{cid}.nii.gz"
        if gf.exists():
            tasks.append((str(pf), str(gf), args.network, args.seed, cid))

    print(f"[kidney_swap] {args.network} seed={args.seed}: {len(tasks)} cases", flush=True)
    fields = ["network", "seed", "case_id", "swap_rate", "lp_dice_gap",
              "merged_dice", "split_dice", "swap_voxels", "has_swap"]
    write_header = not Path(args.output_csv).exists()
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        with Pool(args.num_workers) as pool:
            for r in pool.imap_unordered(eval_one, tasks):
                if r:
                    w.writerow(r)
                    n += 1
        f.flush()
    print(f"[kidney_swap] wrote {n} rows -> {args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
