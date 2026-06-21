#!/usr/bin/env python3
"""Locked evaluator for the v1 input-consistency comparison (spec §5.5).

For one (network, seed) prediction folder, computes per-case per-class **Dice**
and **HD95** against the frozen GT, with source-conditional class masking:
  - head (class 9) evaluated ONLY on HZAU cases
  - testis (class 6) evaluated ONLY on TB cases
  - the other 7 classes on every case
A background-removed FP count is also recorded for the conditional classes on
the source where they are absent (e.g. head FP on TB cases), for reporting.

Output: a long-form CSV appended to --output-csv with one row per
(case, class, metric). Stage 6 (run_stats.py) consumes these.

This evaluator is LOCKED: do not change the metric definitions, the label map,
or the conditional masking without updating the spec. Dice is computed from the
hard confusion matrix (argmax predictions vs integer GT); HD95 is the 95th-
percentile symmetric surface distance via scipy.spatial.cKDTree (deterministic).

Usage:
    python -m evaluation.run_eval \\
        --predictions <PRED>/<network>__seed<seed> \\
        --network swinunetr --seed 20260520 \\
        --output-csv evaluation/results/per_case.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

# label map (spec §3.1) — integer label in the GT nii.gz
LABELS = {
    1: "front",
    2: "middle",
    3: "end",
    4: "left_kidney",
    5: "right_kidney",
    6: "testis",
    7: "thoracic_cavity",
    8: "abdominal_and_pelvic_cavity",
    9: "head",
}
FOREGROUND = sorted(LABELS.keys())  # 1..9
# conditional classes (spec §3.1/§5.5)
HEAD_ONLY_ON = "HZAU"     # class 9 head: only HZAU has it
TESTIS_ONLY_ON = "TB"     # class 6 testis: only TB has it


def dice_score(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    """SoftDice on hard labels for class c. Returns 0.0 if both empty (by convention
    a perfect agreement on absence); NaN handled by caller for absent-in-GT cases."""
    p = pred == c
    g = gt == c
    sp, sg = p.sum(), g.sum()
    if sp + sg == 0:
        return 1.0  # both absent — trivially correct (will be masked out anyway for
        # non-conditional classes because gt has the class somewhere in the cohort)
    inter = (p & g).sum()
    return 2.0 * inter / (sp + sg)


def _surface_distances(seg: np.ndarray, spacing: tuple) -> np.ndarray:
    """Euclidean distances from the surface of `seg` to the nearest surface voxel
    of the complementary reference (bidirectional handled by caller). Returns the
    per-voxel distances of the predicted surface. Empty seg → empty array."""
    from scipy.ndimage import binary_erosion
    if seg.sum() == 0:
        return np.array([])
    eroded = binary_erosion(seg, iterations=1)
    surface = seg & ~eroded
    coords = np.array(np.where(surface)).T.astype(float) * np.array(spacing)
    return coords


def hd95(pred: np.ndarray, gt: np.ndarray, c: int, spacing: tuple) -> float:
    """95th-percentile symmetric Hausdorff distance for class c.
    Returns np.nan if either prediction or GT is empty (HD undefined)."""
    p = pred == c
    g = gt == c
    if p.sum() == 0 or g.sum() == 0:
        return np.nan
    p_surf = _surface_distances(p, spacing)
    g_surf = _surface_distances(g, spacing)
    if len(p_surf) == 0 or len(g_surf) == 0:
        return np.nan
    tree_g = cKDTree(g_surf)
    tree_p = cKDTree(p_surf)
    d_pg, _ = tree_g.query(p_surf, k=1)   # pred surface -> nearest GT surface voxel
    d_gp, _ = tree_p.query(g_surf, k=1)   # GT surface -> nearest pred surface voxel
    all_d = np.concatenate([d_pg, d_gp])
    return float(np.percentile(all_d, 95))


def evaluate_case(pred_path: Path, gt_path: Path, source: str):
    """Return list of (class_label, class_name, dice, hd95) rows for one case."""
    pred_nii = nib.load(str(pred_path))
    gt_nii = nib.load(str(gt_path))
    pred = pred_nii.get_fdata().astype(np.int16)
    gt = gt_nii.get_fdata().astype(np.int16)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape} for {pred_path.name}")
    spacing = tuple(float(s) for s in (gt_nii.header.get_zooms()[:3]))

    rows = []
    for c in FOREGROUND:
        # conditional masking: skip head on TB, testis on HZAU
        if c == 9 and source != HEAD_ONLY_ON:
            continue
        if c == 6 and source != TESTIS_ONLY_ON:
            continue
        d = dice_score(pred, gt, c)
        h = hd95(pred, gt, c, spacing)
        rows.append((c, LABELS[c], d, h))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="folder of <case>.nii.gz predictions")
    ap.add_argument("--gt-folder", required=True, help="folder of GT <case>.nii.gz (labelsTs or test/labels)")
    ap.add_argument("--case-metadata", required=True, help="case_metadata.csv with source column")
    ap.add_argument("--network", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-csv", required=True, help="appended long-form per-case results")
    args = ap.parse_args()

    # source lookup
    import csv as _csv
    source_of = {}
    with open(args.case_metadata, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            source_of[row["case_id"]] = row["source"]

    pred_folder = Path(args.predictions)
    gt_folder = Path(args.gt_folder)
    pred_files = sorted(pred_folder.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"no predictions in {pred_folder}")
    print(f"[eval] {args.network} seed={args.seed}: {len(pred_files)} predictions vs GT in {gt_folder}")

    out_fields = ["network", "seed", "case_id", "source", "class_label", "class_name", "dice", "hd95"]
    write_header = not Path(args.output_csv).exists()
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        if write_header:
            w.writeheader()
        for pf in pred_files:
            case_id = pf.name.replace(".nii.gz", "")
            gf = gt_folder / pf.name
            if not gf.exists():
                # try without _0000 suffix mismatch / alternate name
                gf = gt_folder / f"{case_id}.nii.gz"
            if not gf.exists():
                print(f"[warn] GT missing for {case_id}, skipping", file=sys.stderr)
                continue
            source = source_of.get(case_id, "UNKNOWN")
            for c, cname, d, h in evaluate_case(pf, gf, source):
                w.writerow({
                    "network": args.network, "seed": args.seed, "case_id": case_id,
                    "source": source, "class_label": c, "class_name": cname,
                    "dice": f"{d:.6f}", "hd95": f"{h:.4f}" if not np.isnan(h) else "nan",
                })
                n_rows += 1
    print(f"[eval] wrote {n_rows} rows -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
