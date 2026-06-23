# Task601 LR Axis Audit Report

- **Task**: Task601_Article622_Carcass9Class
- **Kidney cases (>=100vox both)**: 64
- **Left kidney**: class 4, **Right kidney**: class 5

## 3D Result (dual method)

- **CoM method**: LR axis = **0**, margin = 111.6
- **Overlap method**: LR axis = **0**, margin = 0.2477
- **Both methods agree**: True

| Axis | mean|CoM_diff| | mean Overlap | N Cases |
|---|---:|---:|---:|
| 0 | 119.0 | 0.2477 | 64 |
| 1 | 7.4 | 0.0000 | 64 |
| 2 | 4.2 | 0.0000 | 64 |

## 2D Result

- **Slice axis (3D)**: 2 (cranio-caudal, 07069186)
- **In-slice axes**: 2D_0 → 3D_0 (LR), 2D_1 → 3D_1 (AP)
- **Total kidney slices**: 1299
- **axis0 (LR) dominant**: 1299 slices
- **axis1 (AP) dominant**: 0 slices
- **Confirmed LR axis (2D)**: **0** → 3D axis 0
- **Validation**: PASS: 2D LR axis=0 ∈ {0,1}

## nnU-Net Baseline mirror_axes

- **3D**: `(0, 1, 2)` — LR axis (0) in set: ✓
- **2D**: `(0, 1)` — LR axis (0) in set: ✓

## Verdict

- **Verdict**: **PASS**
- **Training gate**: **PASS**
- All pass criteria: True
