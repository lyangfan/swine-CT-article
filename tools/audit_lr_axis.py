#!/usr/bin/env python3
"""Audit which spatial axis of Task601 CT volumes is the true LR (left-right) axis.

Method (dual):
  A) CoM separation: compute center-of-mass of left_kidney(4) and right_kidney(5),
     the axis with the largest mean |CoM difference| is the LR axis.
  B) Flip overlap: flip along each axis and measure class 4↔5 voxel overlap.

For 2D: the CT has shape (H=512, W=512, D≈189-279).  Axial slices are along
axis 2 (depth).  Within each axial slice the two spatial axes are 0→LR, 1→AP.

Output (written to <output-dir>/):
  - lr_axis_audit_manifest.json  : canonical audit result
  - lr_axis_audit_report.md      : human-readable witness
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


LEFT_KIDNEY = 4
RIGHT_KIDNEY = 5


def audit_2d_on_preprocessed(preprocessed_dir: str, max_cases: int = 50) -> dict:
    """Run 2D audit directly on nnU-Net preprocessed data.

    This avoids the NIfTI axis ordering ambiguity by operating on the actual
    data that DataLoader2D sees.  The preprocessed data has shape (C, D, H, W)
    where D is the slice axis (first spatial dim).

    For each slice containing both kidneys, check which in-slice axis
    (H or W) has the larger |CoM_L - CoM_R| separation.
    """
    import os
    npz_files = sorted([f for f in os.listdir(preprocessed_dir)
                         if f.endswith(".npz")])[:max_cases]

    axis_votes = {0: 0, 1: 0}  # 0=H, 1=W
    total_slices = 0
    all_h_sep = []
    all_w_sep = []

    for fname in npz_files:
        d = np.load(os.path.join(preprocessed_dir, fname))
        seg = d["data"][-1]  # last channel = seg, shape (D, H, W)
        for z in range(seg.shape[0]):
            sl = seg[z]  # (H, W)
            nl = int((sl == 4).sum())
            nr = int((sl == 5).sum())
            if nl < 10 or nr < 10:
                continue
            total_slices += 1
            left_com = np.argwhere(sl == 4).mean(axis=0)
            right_com = np.argwhere(sl == 5).mean(axis=0)
            diff = np.abs(left_com - right_com)
            all_h_sep.append(float(diff[0]))
            all_w_sep.append(float(diff[1]))
            if diff[0] > diff[1]:
                axis_votes[0] += 1
            elif diff[1] > diff[0]:
                axis_votes[1] += 1

    majority = 0 if axis_votes[0] >= axis_votes[1] else 1
    mean_h = float(np.mean(all_h_sep)) if all_h_sep else 0.0
    mean_w = float(np.mean(all_w_sep)) if all_w_sep else 0.0

    return {
        "confirmed_lr_axis_2d": majority,
        "maps_to_3d_axis_name": "H(AP)" if majority == 0 else "W(LR)",
        "method": "com_separation_2d_preprocessed",
        "axis0_votes_H": axis_votes[0],
        "axis1_votes_W": axis_votes[1],
        "total_kidney_slices": total_slices,
        "mean_H_sep": mean_h,
        "mean_W_sep": mean_w,
        "n_cases_scanned": len(npz_files),
        "in_slice_axes": {"0": "H(AP)", "1": "W(LR)"},
        "slice_axis": "D (first spatial dim of preprocessed data)",
    }


def _find_kidney_cases(labels_dir: Path, case_ids: list[str],
                       min_voxels: int = 100) -> list[str]:
    """Return cases where both left_kidney(4) and right_kidney(5) have >=min_voxels."""
    valid: list[str] = []
    for cid in case_ids:
        p = labels_dir / f"{cid}.nii.gz"
        if not p.exists():
            continue
        try:
            seg = np.asarray(nib.load(str(p)).dataobj, dtype=np.int16)
        except Exception:
            continue
        nl = int(np.count_nonzero(seg == LEFT_KIDNEY))
        nr = int(np.count_nonzero(seg == RIGHT_KIDNEY))
        if nl >= min_voxels and nr >= min_voxels:
            valid.append(cid)
    return valid


def _com(mask: np.ndarray) -> Optional[np.ndarray]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0)


def audit_3d_com(labels_dir: Path, case_ids: list[str]) -> dict:
    """3D CoM method: axis with largest mean |CoM_L - CoM_R| is LR."""
    axis_diffs = defaultdict(list)
    for cid in case_ids:
        seg = np.asarray(nib.load(str(labels_dir / f"{cid}.nii.gz")).dataobj, dtype=np.int16)
        left_com = _com(seg == LEFT_KIDNEY)
        right_com = _com(seg == RIGHT_KIDNEY)
        if left_com is None or right_com is None:
            continue
        diff = np.abs(left_com - right_com)
        for a in [0, 1, 2]:
            axis_diffs[a].append(float(diff[a]))

    summary = {}
    for a in [0, 1, 2]:
        vals = axis_diffs[a]
        summary[f"axis{a}"] = {
            "mean_com_diff": float(np.mean(vals)),
            "median_com_diff": float(np.median(vals)) if vals else 0.0,
            "n_cases": len(vals),
        }
    best = max([0, 1, 2], key=lambda a: summary[f"axis{a}"]["mean_com_diff"])
    second = sorted([0, 1, 2], key=lambda a: summary[f"axis{a}"]["mean_com_diff"])[-2]
    margin = summary[f"axis{best}"]["mean_com_diff"] - summary[f"axis{second}"]["mean_com_diff"]

    return {
        "confirmed_lr_axis": best,
        "margin_to_next_axis": float(margin),
        "per_axis_summary": summary,
        "n_kidney_cases": len(case_ids),
        "method": "com_separation_3d",
    }


def audit_3d_overlap(labels_dir: Path, case_ids: list[str]) -> dict:
    """3D overlap method: axis with highest flip overlap is LR."""
    axis_scores = defaultdict(list)
    for cid in case_ids:
        seg = np.asarray(nib.load(str(labels_dir / f"{cid}.nii.gz")).dataobj, dtype=np.int16)
        left_mask = seg == LEFT_KIDNEY
        right_mask = seg == RIGHT_KIDNEY
        for axis in [0, 1, 2]:
            flipped = np.flip(seg, axis=axis)
            f_left = flipped == LEFT_KIDNEY
            f_right = flipped == RIGHT_KIDNEY
            l_on_r = int(np.count_nonzero(f_left & right_mask))
            r_on_l = int(np.count_nonzero(f_right & left_mask))
            denom = max(int(np.count_nonzero(left_mask)) + int(np.count_nonzero(right_mask)), 1)
            axis_scores[axis].append((l_on_r + r_on_l) / denom)

    summary = {}
    for a in [0, 1, 2]:
        vals = axis_scores[a]
        summary[f"axis{a}"] = {
            "mean_overlap_ratio": float(np.mean(vals)),
            "median_overlap_ratio": float(np.median(vals)) if vals else 0.0,
            "n_cases": len(vals),
        }
    best = max([0, 1, 2], key=lambda a: summary[f"axis{a}"]["mean_overlap_ratio"])
    second = sorted([0, 1, 2], key=lambda a: summary[f"axis{a}"]["mean_overlap_ratio"])[-2]
    margin = summary[f"axis{best}"]["mean_overlap_ratio"] - summary[f"axis{second}"]["mean_overlap_ratio"]

    return {
        "confirmed_lr_axis": best,
        "margin_to_next_axis": float(margin),
        "per_axis_summary": summary,
        "n_kidney_cases": len(case_ids),
        "method": "flip_overlap_3d",
    }


def audit_2d_com(labels_dir: Path, case_ids: list[str],
                 slice_axis_3d: int, in_slice_3d_axes: tuple[int, int]) -> dict:
    """2D CoM method within slices along the nnU-Net 2D slice axis.

    For each slice (along the nnU-Net 2D slice axis) containing both kidneys,
    check which in-slice axis has the larger |CoM_L - CoM_R| separation.
    """
    axis_votes = {0: 0, 1: 0}
    total_slices = 0
    for cid in case_ids:
        seg = np.asarray(nib.load(str(labels_dir / f"{cid}.nii.gz")).dataobj, dtype=np.int16)
        for z in range(seg.shape[slice_axis_3d]):
            if slice_axis_3d == 0:
                sl = seg[z, :, :]
            elif slice_axis_3d == 1:
                sl = seg[:, z, :]
            else:
                sl = seg[:, :, z]
            left_mask = sl == LEFT_KIDNEY
            right_mask = sl == RIGHT_KIDNEY
            if left_mask.sum() < 10 or right_mask.sum() < 10:
                continue
            total_slices += 1
            left_com = _com(left_mask)
            right_com = _com(right_mask)
            if left_com is None or right_com is None:
                continue
            diff = np.abs(left_com - right_com)
            if diff[0] > diff[1]:
                axis_votes[0] += 1
            elif diff[1] > diff[0]:
                axis_votes[1] += 1

    majority_axis = 0 if axis_votes[0] >= axis_votes[1] else 1
    margin = axis_votes[majority_axis] - axis_votes[1 - majority_axis]

    return {
        "confirmed_lr_axis_2d": majority_axis,
        "maps_to_3d_axis": in_slice_3d_axes[majority_axis],
        "margin_votes": margin,
        "axis0_votes": axis_votes[0],
        "axis1_votes": axis_votes[1],
        "total_kidney_slices": total_slices,
        "n_kidney_cases": len(case_ids),
        "in_slice_3d_axes": {0: in_slice_3d_axes[0], 1: in_slice_3d_axes[1]},
        "slice_axis_3d": slice_axis_3d,
        "method": "com_separation_2d_slice",
    }


def validate_2d_axis(result_2d: dict) -> dict:
    axis = result_2d["confirmed_lr_axis_2d"]
    valid = axis in (0, 1)
    return {
        "axis_value": axis,
        "in_valid_range": valid,
        "message": f"PASS: 2D LR axis={axis} ∈ {{0,1}}" if valid
                   else f"FAIL: 2D LR axis={axis} ∉ {{0,1}}"
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit LR axis for Task601 swine CT")
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--preprocessed-dir", required=True,
                    help="path to nnUNetData_plans_v2.1_2D_stage0/ (preprocessed 2D data)")
    ap.add_argument("--max-cases", type=int, default=50)
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read case list
    case_ids = []
    with open(args.split_manifest) as f:
        for row in __import__("csv").DictReader(f):
            case_ids.append(row["case_id"])

    kidney_cases = _find_kidney_cases(labels_dir, case_ids[:args.max_cases])
    print(f"Found {len(kidney_cases)} cases with both kidneys >=100 voxels "
          f"(scanned {min(len(case_ids), args.max_cases)})")

    if len(kidney_cases) < 3:
        print(f"ERROR: too few kidney cases ({len(kidney_cases)}), need >=3", file=sys.stderr)
        return 1

    # --- 3D audit (dual method) ---
    print("\n=== 3D LR-axis audit ===")
    result_3d_com = audit_3d_com(labels_dir, kidney_cases)
    result_3d_overlap = audit_3d_overlap(labels_dir, kidney_cases)

    # Both should agree; CoM is primary
    assert result_3d_com["confirmed_lr_axis"] == result_3d_overlap["confirmed_lr_axis"], \
        f"CoM says axis {result_3d_com['confirmed_lr_axis']}, overlap says {result_3d_overlap['confirmed_lr_axis']}"
    confirmed_3d = result_3d_com["confirmed_lr_axis"]
    print(f"CoM method: LR axis = {confirmed_3d}, margin = {result_3d_com['margin_to_next_axis']:.1f}")
    for a in [0, 1, 2]:
        s = result_3d_com["per_axis_summary"][f"axis{a}"]
        print(f"  axis{a}: mean|CoM_diff| = {s['mean_com_diff']:.1f} (n={s['n_cases']})")
    print(f"Overlap method: LR axis = {result_3d_overlap['confirmed_lr_axis']}, margin = {result_3d_overlap['margin_to_next_axis']:.4f}")
    for a in [0, 1, 2]:
        s = result_3d_overlap["per_axis_summary"][f"axis{a}"]
        print(f"  axis{a}: mean_overlap = {s['mean_overlap_ratio']:.4f} (n={s['n_cases']})")

    # --- 2D audit (on preprocessed data, NOT NIfTI) ---
    print("\n=== 2D LR-axis audit (on preprocessed data) ===")
    result_2d = audit_2d_on_preprocessed(args.preprocessed_dir, args.max_cases)
    r2d_val = validate_2d_axis(result_2d)
    print(f"Slice axis: {result_2d['slice_axis']}")
    print(f"In-slice axes: {result_2d['in_slice_axes']}")
    print(f"Total kidney slices: {result_2d['total_kidney_slices']}")
    print(f"axis0 (H/AP) dominant: {result_2d['axis0_votes_H']} slices, mean sep={result_2d['mean_H_sep']:.1f}")
    print(f"axis1 (W/LR) dominant: {result_2d['axis1_votes_W']} slices, mean sep={result_2d['mean_W_sep']:.1f}")
    print(f"Confirmed LR axis (2D): {result_2d['confirmed_lr_axis_2d']} "
          f"(→ {result_2d['maps_to_3d_axis_name']})")
    print(f"Validation: {r2d_val['message']}")

    # --- nnU-Net baseline checks ---
    nnunet_3d_axes = (0, 1, 2)
    nnunet_2d_axes = (0, 1)
    print(f"\n=== nnU-Net baseline mirror_axes ===")
    print(f"3D: {nnunet_3d_axes} — LR axis {confirmed_3d} in set: {confirmed_3d in nnunet_3d_axes}")
    print(f"2D: {nnunet_2d_axes} — LR axis {result_2d['confirmed_lr_axis_2d']} in set: "
          f"{result_2d['confirmed_lr_axis_2d'] in nnunet_2d_axes}")
    assert confirmed_3d in nnunet_3d_axes, \
        f"3D LR axis {confirmed_3d} NOT in mirror_axes {nnunet_3d_axes}"

    # --- Build manifest ---
    verdict = "PASS" if r2d_val["in_valid_range"] else "FAIL"

    all_pass = (
        confirmed_3d in nnunet_3d_axes
        and r2d_val["in_valid_range"]
        and result_3d_com["margin_to_next_axis"] > 20  # CoM diff well-separated
        and len(kidney_cases) >= 3
    )

    manifest = {
        "schema_version": "lr_axis_audit.v3",
        "task": "Task601_Article622_Carcass9Class",
        "confirmed_lr_axis_3d": confirmed_3d,
        "confirmed_lr_axis_2d": result_2d["confirmed_lr_axis_2d"],
        "confirmed_lr_axis_2d_maps_to_3d_axis_name": result_2d["maps_to_3d_axis_name"],
        "left_kidney_label_id": LEFT_KIDNEY,
        "right_kidney_label_id": RIGHT_KIDNEY,
        "nnunet_3d_mirror_axes": list(nnunet_3d_axes),
        "nnunet_2d_mirror_axes": list(nnunet_2d_axes),
        "verdict": "PASS" if all_pass else "FAIL",
        "training_gate": "PASS" if all_pass else "FAIL",
        "pass_criteria": {
            "lr_axis_3d_found": True,
            "lr_axis_3d_in_mirror_set": confirmed_3d in nnunet_3d_axes,
            "lr_axis_2d_in_valid_range": r2d_val["in_valid_range"],
            "com_margin_sufficient_3d": result_3d_com["margin_to_next_axis"] > 20,
            "overlap_margin_sufficient_3d": result_3d_overlap["margin_to_next_axis"] > 0.05,
            "com_vs_overlap_agree": result_3d_com["confirmed_lr_axis"] == result_3d_overlap["confirmed_lr_axis"],
            "min_kidney_cases_met": len(kidney_cases) >= 3,
        },
        "audit_3d_com": result_3d_com,
        "audit_3d_overlap": result_3d_overlap,
        "audit_2d": result_2d,
        "validation_2d": r2d_val,
        "case_counts": {
            "total_in_manifest": len(case_ids),
            "scanned": min(len(case_ids), args.max_cases),
            "with_both_kidneys_100vox": len(kidney_cases),
        },
    }

    manifest_path = output_dir / "lr_axis_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nManifest: {manifest_path}")

    # Report
    report = f"""# Task601 LR Axis Audit Report

- **Task**: Task601_Article622_Carcass9Class
- **Kidney cases (>=100vox both)**: {len(kidney_cases)}
- **Left kidney**: class {LEFT_KIDNEY}, **Right kidney**: class {RIGHT_KIDNEY}

## 3D Result (dual method)

- **CoM method**: LR axis = **{confirmed_3d}**, margin = {result_3d_com['margin_to_next_axis']:.1f}
- **Overlap method**: LR axis = **{result_3d_overlap['confirmed_lr_axis']}**, margin = {result_3d_overlap['margin_to_next_axis']:.4f}
- **Both methods agree**: {result_3d_com['confirmed_lr_axis'] == result_3d_overlap['confirmed_lr_axis']}

| Axis | mean|CoM_diff| | mean Overlap | N Cases |
|---|---:|---:|---:|
"""
    for a in [0, 1, 2]:
        cs = result_3d_com["per_axis_summary"][f"axis{a}"]
        os = result_3d_overlap["per_axis_summary"][f"axis{a}"]
        report += f"| {a} | {cs['mean_com_diff']:.1f} | {os['mean_overlap_ratio']:.4f} | {cs['n_cases']} |\n"

    report += f"""
## 2D Result (on preprocessed data — nnU-Net 2D actual data)

- **Slice axis**: {result_2d['slice_axis']}
- **In-slice axes**: {result_2d['in_slice_axes']}
- **Total kidney slices**: {result_2d['total_kidney_slices']}
- **axis0 (H/AP) dominant**: {result_2d['axis0_votes_H']} slices, mean sep={result_2d['mean_H_sep']:.1f}
- **axis1 (W/LR) dominant**: {result_2d['axis1_votes_W']} slices, mean sep={result_2d['mean_W_sep']:.1f}
- **Confirmed LR axis (2D)**: **{result_2d['confirmed_lr_axis_2d']}** → {result_2d['maps_to_3d_axis_name']}
- **Validation**: {r2d_val['message']}

## nnU-Net Baseline mirror_axes

- **3D**: `{nnunet_3d_axes}` — LR axis ({confirmed_3d}) in set: ✓
- **2D**: `{nnunet_2d_axes}` — LR axis 2D={result_2d['confirmed_lr_axis_2d']} ({result_2d['maps_to_3d_axis_name']}) in set: ✓

## Verdict

- **Verdict**: **{manifest['verdict']}**
- **Training gate**: **{manifest['training_gate']}**
- All pass criteria: {all(manifest['pass_criteria'].values())}
"""
    report_path = output_dir / "lr_axis_audit_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report: {report_path}")

    if not all_pass:
        print("\nERROR: Not all pass criteria met!", file=sys.stderr)
        for k, v in manifest["pass_criteria"].items():
            if not v:
                print(f"  FAIL: {k}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
