#!/usr/bin/env python3
"""Locked evaluator for the v1 input-consistency comparison (UNIFIED, single evaluator).

Verbatim port of the project's locked evaluator
(``swine_ct_autonomous_discovery/metrics/evaluate_swine_ct.py``): same HD95
algorithm (union-bbox crop margin=2 → surfaces → cKDTree → bidirectional → p95),
same confusion-matrix metrics (Dice/IoU/Precision/Recall/Specificity/FPR/
FP_GT_ratio/TP_percent/FP_percent/FN_percent/TN_percent/absent_FP_*), same
conditional masking (head=9 only on HZAU, testis=6 only on TB; absent cases emit
absent_FP metrics instead of being skipped). The ONLY addition vs the canonical
evaluator is a ``seed`` column (needed for 3-seed paired stats).

Output: a long-form CSV appended to --output-csv. Cases evaluated in parallel
(--num-workers). Concurrent-eval protection is the non-blocking flock in the
DSUB job script (do NOT fcntl.flock here — NFS deadlocks).

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


def safe_div(numerator: float, denominator: float) -> float:
    """Match evaluate_swine_ct.safe_div: NaN (not 0) when denominator is 0."""
    return float(numerator) / float(denominator) if denominator else float("nan")


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
    """95th-percentile symmetric surface distance (mm). Verbatim port of
    evaluate_swine_ct.hd95_mm (union crop margin=2 → surfaces → cKDTree →
    bidirectional query → p95). NaN if either side empty."""
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
    # Match evaluate_swine_ct: try workers kwarg, fall back if scipy lacks it.
    try:
        gt_to_pred, _ = pred_tree.query(gt_pts, k=1, workers=hd95_workers)
        pred_to_gt, _ = gt_tree.query(pred_pts, k=1, workers=hd95_workers)
    except TypeError:
        gt_to_pred, _ = pred_tree.query(gt_pts, k=1)
        pred_to_gt, _ = gt_tree.query(pred_pts, k=1)
    distances = np.concatenate([gt_to_pred, pred_to_gt])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def _is_class_absent(class_id: int, source: str) -> bool:
    """Conditional masking: head(9) only on HZAU, testis(6) only on TB.
    Matches evaluate_swine_ct.conditional_status (absent on the other cohort)."""
    if class_id == 9:
        return source != HEAD_ONLY_ON
    if class_id == 6:
        return source != TESTIS_ONLY_ON
    return False


def evaluate_case(pred_path, gt_path, source, hd95_workers: int = -1):
    """Compute the full locked-evaluator metric set for one case. Port of
    evaluate_swine_ct.evaluate_one_case (confusion-matrix → all metrics),
    adapted to run_eval's (pred_path, gt_path, source) input."""
    pred_nii = nib.load(str(pred_path))
    gt_nii = nib.load(str(gt_path))
    pred = pred_nii.get_fdata().astype(np.int16)
    gt = gt_nii.get_fdata().astype(np.int16)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape} for {pred_path.name}")
    spacing = tuple(float(s) for s in gt_nii.header.get_zooms()[:3])

    foreground_domain = (gt > 0) | (pred > 0)
    domain_voxels = int(np.count_nonzero(foreground_domain))

    rows = []
    for c in FOREGROUND:
        class_name = LABELS[c]
        absent = _is_class_absent(c, source)
        gt_mask = (gt == c)
        pred_mask = (pred == c)
        gt_voxels = int(np.count_nonzero(gt_mask))
        pred_voxels = int(np.count_nonzero(pred_mask))

        tp = int(np.count_nonzero(foreground_domain & gt_mask & pred_mask))
        fp = int(np.count_nonzero(foreground_domain & ~gt_mask & pred_mask))
        fn = int(np.count_nonzero(foreground_domain & gt_mask & ~pred_mask))
        tn = int(np.count_nonzero(foreground_domain & ~gt_mask & ~pred_mask))

        if absent:
            is_evaluable = False
            dice = iou = precision = recall = specificity = fpr = fp_gt_ratio = hd95 = float("nan")
            missed = False
            absent_fp_voxels = float(pred_voxels)
            absent_fp_rate = safe_div(pred_voxels, domain_voxels)
            absent_fp_incidence = 1.0 if pred_voxels > 0 else 0.0
        else:
            is_evaluable = bool(gt_voxels > 0 or pred_voxels > 0)
            dice = safe_div(2 * tp, 2 * tp + fp + fn)
            iou = safe_div(tp, tp + fp + fn)
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            specificity = safe_div(tn, tn + fp)
            fpr = safe_div(fp, fp + tn)
            fp_gt_ratio = safe_div(fp, gt_voxels)
            missed = bool(gt_voxels > 0 and pred_voxels == 0)
            if gt_voxels > 0 and pred_voxels > 0:
                hd95 = hd95_mm(gt_mask, pred_mask, spacing, hd95_workers)
            else:
                hd95 = float("nan")
            absent_fp_voxels = absent_fp_rate = absent_fp_incidence = float("nan")

        if domain_voxels == 0:
            tp_percent = fp_percent = fn_percent = tn_percent = float("nan")
        else:
            tp_percent = 100.0 * safe_div(tp, domain_voxels)
            fp_percent = 100.0 * safe_div(fp, domain_voxels)
            fn_percent = 100.0 * safe_div(fn, domain_voxels)
            tn_percent = 100.0 * safe_div(tn, domain_voxels)

        rows.append({
            "class_id": c, "class_name": class_name,
            "is_evaluable": is_evaluable, "domain_voxels": domain_voxels,
            "GT_voxels": gt_voxels, "Pred_voxels": pred_voxels,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "TP_percent": tp_percent, "FP_percent": fp_percent,
            "FN_percent": fn_percent, "TN_percent": tn_percent,
            "Dice": dice, "IoU": iou, "Precision": precision, "Recall": recall,
            "Specificity": specificity, "FPR": fpr, "FP_GT_ratio": fp_gt_ratio,
            "HD95": hd95, "missed": missed,
            "absent_FP_voxels": absent_fp_voxels, "absent_FP_rate": absent_fp_rate,
            "absent_FP_incidence": absent_fp_incidence,
        })
    return rows


def _worker(args):
    pred_path, gt_path, source, network, seed, case_id, hd95_workers = args
    try:
        rows = evaluate_case(pred_path, gt_path, source, hd95_workers)
    except Exception as exc:
        print(f"[warn] {case_id}: {exc}", file=sys.stderr)
        return []
    out = []
    for r in rows:
        r["network"] = network
        r["seed"] = seed
        r["case_id"] = case_id
        r["source"] = source
        out.append(r)
    return out


OUT_FIELDS = [
    "network", "seed", "case_id", "source", "class_id", "class_name",
    "is_evaluable", "domain_voxels", "GT_voxels", "Pred_voxels",
    "TP", "FP", "FN", "TN", "TP_percent", "FP_percent", "FN_percent", "TN_percent",
    "Dice", "IoU", "Precision", "Recall", "Specificity", "FPR", "FP_GT_ratio", "HD95",
    "missed", "absent_FP_voxels", "absent_FP_rate", "absent_FP_incidence",
]


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

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    # NOTE: concurrent-eval protection is the non-blocking flock in the DSUB job
    # script (jobs/eval/run_eval_and_stats.sh). Do NOT fcntl.flock the CSV here —
    # it lives on NFS and flock-on-NFS deadlocks on compute nodes.
    n_rows = 0
    write_header = not Path(args.output_csv).exists()
    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if write_header:
            w.writeheader()
        with Pool(args.num_workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(_worker, tasks)):
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in OUT_FIELDS})
                    n_rows += 1
                f.flush()
                print(f"[eval] {args.network} seed={args.seed}: {i+1}/{len(tasks)} cases done ({n_rows} rows)", flush=True)
    print(f"[eval] wrote {n_rows} rows -> {args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
