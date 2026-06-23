# Task601 LR Axis Audit Report (v2.1 — CORRECTED)

- **Task**: Task601_Article622_Carcass9Class
- **Kidney cases**: 64 (>=100vox both)

## 3D Result — CORRECT

- **LR axis = 0** (CoM diff=90.7, AP=6.0, CC=2.2)
- Both CoM and overlap methods agree
- In nnU-Net 3D mirror_axes (0,1,2): YES
- Conditional LR mirror applicable: **YES**

## 2D Result — CORRECTED (v2.1)

### Previous v2 audit ERROR
The original v2 audit tested axial slices (along NIfTI axis 2 = CC/depth), finding that within axial slices, axis 0 is LR. **This was WRONG for nnU-Net 2D** because nnU-Net 2D does NOT slice along axis 2 — it slices along the axis with largest spacing.

### Correct analysis
- nnU-Net 2D spacing: (5.0, 0.977, 0.977) mm
- **Slice axis = 0** (largest spacing = 5.0mm)
- In-slice spatial axes: [1, 2]
  - 2D mirror axis 0 = NIfTI axis 1 (**AP**, CoM diff=6.0)
  - 2D mirror axis 1 = NIfTI axis 2 (**CC**, CoM diff=2.2)
- **LR axis (NIfTI axis 0) = SLICE AXIS — NEVER mirrored in 2D nnU-Net**

### Conclusion
- **2D conditional LR mirror: N/A (not applicable)**
- The 2D arm of this ablation is fundamentally invalid because the LR axis is consumed as the slice dimension
- The 2D condlr results should be DISCARDED
- Only the SwinUNETR (3D) arm provides valid evidence for the conditional LR mirror hypothesis

## 3D LR axis (still valid)
- **confirmed_lr_axis_3d = 0**
- In nnU-Net 3D mirror_axes (0,1,2): YES
- Training gate: PASS
