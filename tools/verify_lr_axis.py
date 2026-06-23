#!/usr/bin/env python3
"""Verify LR axis for Task601 using CoM-based approach."""
import nibabel as nib
import numpy as np
import csv
from pathlib import Path
from collections import defaultdict

LEFT, RIGHT = 4, 5
labels_dir = Path("/home/share/hzau/home/liuyangfan/swine-CT-article/data/train/labels")
split_path = "/home/share/hzau/home/liuyangfan/swine-CT-article/data/splits/split_manifest.csv"

cases = []
with open(split_path) as f:
    for row in csv.DictReader(f):
        cases.append(row["case_id"])

axis_diffs = {0: [], 1: [], 2: []}
kidney_cases = []
for cid in cases:
    p = labels_dir / f"{cid}.nii.gz"
    if not p.exists():
        continue
    seg = np.asarray(nib.load(str(p)).dataobj, dtype=np.int16)
    nl = int(np.count_nonzero(seg == LEFT))
    nr = int(np.count_nonzero(seg == RIGHT))
    if nl >= 100 and nr >= 100:
        kidney_cases.append(cid)
        left_com = np.argwhere(seg == LEFT).mean(axis=0)
        right_com = np.argwhere(seg == RIGHT).mean(axis=0)
        diff = np.abs(left_com - right_com)
        for a in [0, 1, 2]:
            axis_diffs[a].append(float(diff[a]))

print(f"Found {len(kidney_cases)} cases with both kidneys >= 100 voxels")
print()
print("CoM separation per axis (|left_com - right_com|):")
for a in [0, 1, 2]:
    vals = axis_diffs[a]
    print(f"  axis {a}: mean={np.mean(vals):.1f} median={np.median(vals):.1f} min={np.min(vals):.1f} max={np.max(vals):.1f}")

best = max([0, 1, 2], key=lambda a: np.mean(axis_diffs[a]))
print(f"\nConfirmed LR axis (3D): {best} (largest CoM separation)")

# 2D: axial slice axis is axis 2 (cranio-caudal, the dim=251 axis)
# Within an axial slice: 2D axes are [0, 1] = [LR, AP]
slice_axis = 2
in_slice_3d = [0, 1]

count_2d = {0: 0, 1: 0}
for cid in kidney_cases:
    seg = np.asarray(nib.load(str(labels_dir / f"{cid}.nii.gz")).dataobj, dtype=np.int16)
    for z in range(seg.shape[slice_axis]):
        sl = seg[:, :, z]
        left_mask = sl == LEFT
        right_mask = sl == RIGHT
        if left_mask.sum() < 20 or right_mask.sum() < 20:
            continue
        left_com = np.argwhere(left_mask).mean(axis=0)
        right_com = np.argwhere(right_mask).mean(axis=0)
        diff = np.abs(left_com - right_com)
        if diff[0] > diff[1]:
            count_2d[0] += 1
        else:
            count_2d[1] += 1

total_slices = count_2d[0] + count_2d[1]
print(f"\n2D slice-level audit ({len(kidney_cases)} cases, {total_slices} kidney slices):")
print(f"  axis0(LR→3d_0) dominates: {count_2d[0]} slices ({100*count_2d[0]/max(1,total_slices):.1f}%)")
print(f"  axis1(AP→3d_1) dominates: {count_2d[1]} slices ({100*count_2d[1]/max(1,total_slices):.1f}%)")
confirmed_2d = 0 if count_2d[0] >= count_2d[1] else 1
print(f"Confirmed LR axis (2D): {confirmed_2d} (maps to 3D axis {in_slice_3d[confirmed_2d]})")

print()
print("=== Summary ===")
print(f"3D LR axis: {best}")
print(f"2D LR axis: {confirmed_2d} (→ 3D axis {in_slice_3d[confirmed_2d]})")
print("nnU-Net v1 3D mirror_axes default: (0,1,2)")
print(f"  3D LR axis ({best}) in mirror set: {best in (0,1,2)}")
print("nnU-Net v1 2D mirror_axes default: (0,1)")
print(f"  2D LR axis ({confirmed_2d}) in mirror set: {confirmed_2d in (0,1)}")
