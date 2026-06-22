#!/usr/bin/env python3
"""Locked evaluator for the v1 input-consistency comparison (spec §5.5).

HD95 is computed EXACTLY as the project's locked evaluator
(``swine_ct_autonomous_discovery/metrics/evaluate_swine_ct.py``): crop GT+pred
masks to their union bounding box (margin=2), extract surfaces via binary
erosion, build scipy cKDTree on spacing-scaled surface points, bidirectional
nearest-neighbour query (parallel via ``workers``), 95th percentile. Dice is
hard-label SoftDice. Conditional masking: head (9) only on HZAU, testis (6)
only on TB.

Output: a long-form CSV appended to --output-csv. Cases evaluated in parallel
(--num-workers). An flock on the output CSV serialises concurrent eval runs so
two eval jobs can never clobber each other's output (the Stage 6 race that
previously corrupted per_case.csv).

Usage:
    python -m evaluation.run_eval \\
        --predictions <PRED>/<network>__seed<seed> \\
        --network swinunetr --seed 20260520 \\
        --output-csv evaluation/results/per_case.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree

LABELS = {
    1: "front", 2: "middle", 3: "end", 4: "left_kidney", 5: "right_kidney",
    6: "testis", 7: "thoracic_cavity", 8: "abdominal_and_pelvic_cavity", 9: "head",
}
FOREGROUND = sorted(LABELS.keys())
HEAD_ONLY_ON = "HZAU"
TESTIS_ONLY_ON = "TB"


# ---- HD95 — verbatim port of the locked evaluator ------------------------- #
def _mask_surface(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def _crop_to_union(gt_mask: np.ndarray, pred_mask: np.ndarray, margin: int = 2):
    union = gt_mask | pred_mask
    coords = np.where(union)
    slices = []
    for axis, axis_coords in enumerate(coords):
        start = max(0, int(axis_coords.min()) - margin)
        stop = min(union.shape[axis], int(axis_coords.max()) + margin + 1)
        slices.append(slice(start, stop))
    crop = tuple(slices)
    return gt_mask[crop], pred_mask[crop]


def hd95_mm(gt_mask: np.ndarray, pred_mask: np.ndarray, spacing: tuple,
            hd95_workers: int = -1) -> float:
    """95th-percentile symmetric surface distance (mm). Matches the locked
    evaluator bit-for-bit: union-bbox crop (margin 2) → surfaces → cKDTree →
    bidirectional query (workers) → p95. NaN if either side empty."""
    if not np.any(gt_mask | pred_mask):
        return float("nan")
    gt_crop, pred_crop = _crop_to_union(gt_mask, pred_mask)
    gt_surf = _mask_surface(gt_crop)
    pred_surf = _mask_surface(pred_crop)
    if not np.any(gt_surf) or not np.any(pred_surf):
        return float("nan")
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    gt_pts = np.argwhere(gt_surf).astype(np.float32, copy=False) * spacing_arr
    pred_pts = np.argwhere(pred_surf).astype(np.float32, copy=False) * spacing_arr
    gt_tree = cKDTree(gt_pts)
    pred_tree = cKDTree(pred_pts)
    # When hd95_workers <= 1, call query WITHOUT the workers kwarg so scipy uses
    # its plain single-threaded path (passing workers=1 still spins up scipy's
    # internal Pool, which deadlocks inside our multiprocessing.Pool workers).
    if hd95_workers and hd95_workers > 1:
        gt_to_pred, _ = pred_tree.query(gt_pts, k=1, workers=hd95_workers)
        pred_to_gt, _ = gt_tree.query(pred_pts, k=1, workers=hd95_workers)
    else:
        gt_to_pred, _ = pred_tree.query(gt_pts, k=1)
        pred_to_gt, _ = gt_tree.query(pred_pts, k=1)
    distances = np.concatenate([gt_to_pred, pred_to_gt])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def dice_score(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    p = pred == c
    g = gt == c
    sp, sg = p.sum(), g.sum()
    if sp + sg == 0:
        return 1.0
    return 2.0 * (p & g).sum() / (sp + sg)


def evaluate_case(pred_path, gt_path, source, hd95_workers: int = -1):
    pred_nii = nib.load(str(pred_path))
    gt_nii = nib.load(str(gt_path))
    pred = pred_nii.get_fdata().astype(np.int16)
    gt = gt_nii.get_fdata().astype(np.int16)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape} for {pred_path.name}")
    spacing = tuple(float(s) for s in gt_nii.header.get_zooms()[:3])
    rows = []
    for c in FOREGROUND:
        if c == 9 and source != HEAD_ONLY_ON:
            continue
        if c == 6 and source != TESTIS_ONLY_ON:
            continue
        gt_mask = (gt == c)
        pred_mask = (pred == c)
        rows.append((c, LABELS[c], dice_score(pred, gt, c),
                     hd95_mm(gt_mask, pred_mask, spacing, hd95_workers)))
    return rows


def _worker(args):
    pred_path, gt_path, source, network, seed, case_id, hd95_workers = args
    try:
        rows = evaluate_case(pred_path, gt_path, source, hd95_workers)
    except Exception as exc:
        print(f"[warn] {case_id}: {exc}", file=sys.stderr)
        rows = []
    return [(network, seed, case_id, source, c, cn, d, h) for (c, cn, d, h) in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gt-folder", required=True)
    ap.add_argument("--case-metadata", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--hd95-workers", type=int, default=-1,
                    help="cKDTree query workers (-1 = all cores, matches locked evaluator default)")
    args = ap.parse_args()

    import csv as _csv
    from multiprocessing import Pool
    source_of = {}
    with open(args.case_metadata, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            source_of[row["case_id"]] = row["source"]

    pred_folder = Path(args.predictions)
    gt_folder = Path(args.gt_folder)
    pred_files = sorted(pred_folder.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"no predictions in {pred_folder}")
    print(f"[eval] {args.network} seed={args.seed}: {len(pred_files)} predictions vs GT in {gt_folder}", flush=True)

    tasks = []
    for pf in pred_files:
        case_id = pf.name.replace(".nii.gz", "")
        gf = gt_folder / pf.name
        if not gf.exists():
            gf = gt_folder / f"{case_id}.nii.gz"
        if not gf.exists():
            print(f"[warn] GT missing for {case_id}, skipping", file=sys.stderr)
            continue
        tasks.append((str(pf), str(gf), source_of.get(case_id, "UNKNOWN"),
                      args.network, args.seed, case_id, args.hd95_workers))

    out_fields = ["network", "seed", "case_id", "source", "class_label", "class_name", "dice", "hd95"]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    # NOTE: concurrent-eval protection is the non-blocking flock in the DSUB job
    # script (jobs/eval/run_eval_and_stats.sh). Do NOT fcntl.flock the CSV here —
    # it lives on NFS and flock-on-NFS deadlocks on compute nodes.
    n_rows = 0
    write_header = not Path(args.output_csv).exists()
    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        if write_header:
            w.writeheader()
        with Pool(args.num_workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(_worker, tasks)):
                for (net, seed, case_id, source, c, cn, d, h) in rows:
                    w.writerow({
                        "network": net, "seed": seed, "case_id": case_id, "source": source,
                        "class_label": c, "class_name": cn,
                        "dice": f"{d:.6f}", "hd95": f"{h:.4f}" if not np.isnan(h) else "nan",
                    })
                    n_rows += 1
                f.flush()
                print(f"[eval] {args.network} seed={args.seed}: {i+1}/{len(tasks)} cases done ({n_rows} rows)", flush=True)
    print(f"[eval] wrote {n_rows} rows -> {args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
