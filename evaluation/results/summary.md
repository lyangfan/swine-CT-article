# v1 input-consistency — results summary

## Per-network mean per-case Dice (3-seed averaged, across 39 test cases)

| network | mean per-case Dice ± std |
|---|---|
| mednext_s | 0.9288±0.0272 |
| nnunet_2d | 0.9670±0.0401 |
| nnunet_v1 | 0.9604±0.0246 |
| segformer3d | 0.9513±0.0179 |
| swinunetr | 0.9642±0.0185 |

## Pairwise Wilcoxon signed-rank on per-case mean Dice (Holm-Bonferroni)

| pair | mean A | mean B | p (raw) | p (Holm-Bonferroni) |
|---|---|---|---|---|
| nnunet_v1 vs mednext_s | 0.960433 | 0.928817 | 1.62618e-09 | 8.13088e-09 |
| nnunet_v1 vs swinunetr | 0.960433 | 0.964176 | 0.485211 | 0.485211 |
| nnunet_v1 vs segformer3d | 0.960433 | 0.951314 | 0.00043872 | 0.000877441 |
| mednext_s vs swinunetr | 0.928817 | 0.964176 | 3.63798e-12 | 2.18279e-11 |
| mednext_s vs segformer3d | 0.928817 | 0.951314 | 8.6402e-09 | 3.45608e-08 |
| swinunetr vs segformer3d | 0.964176 | 0.951314 | 4.13893e-08 | 1.24168e-07 |
