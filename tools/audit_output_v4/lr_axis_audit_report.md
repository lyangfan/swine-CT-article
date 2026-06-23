# Task601 LR Axis Audit Report

- **Task**: Task601_Article622_Carcass9Class
- **Kidney cases (>=100vox both)**: 18
- **Left kidney**: class 4, **Right kidney**: class 5

## 3D Result (dual method)

- **CoM method**: LR axis = **0**, margin = 128.8
- **Overlap method**: LR axis = **0**, margin = 0.2119
- **Both methods agree**: True

| Axis | mean|CoM_diff| | mean Overlap | N Cases |
|---|---:|---:|---:|
| 0 | 135.9 | 0.2119 | 18 |
| 1 | 7.1 | 0.0000 | 18 |
| 2 | 6.4 | 0.0000 | 18 |

## 2D Result (on preprocessed data — nnU-Net 2D actual data)

- **Slice axis**: D (first spatial dim of preprocessed data)
- **In-slice axes**: {'0': 'H(AP)', '1': 'W(LR)'}
- **Total kidney slices**: 581
- **axis0 (H/AP) dominant**: 0 slices, mean sep=9.2
- **axis1 (W/LR) dominant**: 581 slices, mean sep=136.3
- **Confirmed LR axis (2D)**: **1** → W(LR)
- **Validation**: PASS: 2D LR axis=1 ∈ {0,1}

## nnU-Net Baseline mirror_axes

- **3D**: `(0, 1, 2)` — LR axis (0) in set: ✓
- **2D**: `(0, 1)` — LR axis 2D=1 (W(LR)) in set: ✓

## Verdict

- **Verdict**: **PASS**
- **Training gate**: **PASS**
- All pass criteria: True
