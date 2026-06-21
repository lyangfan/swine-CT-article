# SWCT06042040 Huawei v1 近期实验交接报告：LR mirror、presence classifier 与 hard-zero gate

Status: FINAL_VALIDATED

Date: 2026-06-21

本文档用于把 2026-06-17 至 2026-06-21 期间围绕 Huawei nnU-Net v1 swine CT 的三条实验线索交接给 Claude Code 或后续执行 agent。范围限定为 Huawei 服务器 `paca_share` 上的 `runs/swct06042040`：

- kidney class 相关的 LR mirror 关闭 / 条件 mirror 实验；
- whole-case / ROI half-projection head/testis presence classifier 训练实验；
- 基于 learned classifier 的 `learned_hard_zero_t050` head probability hard-zero 5-fold OOF 实验。

本文不把任何结果升级为 champion，不做 HZAU 或 nnU-Net v2 对比，不主张 classifier gate 已经可部署。所有结论均以本文列出的文件、job、validator 和 evaluation 产物为准。

## 0. 工作目录与环境

本地 AutoScientists workspace:

```text
/Users/liuyangfan/Documents/work/AutoScientists
```

本地 run workspace:

```text
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040
```

Huawei 项目根目录:

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery
```

Huawei run 根目录:

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040
```

统一实验口径：

- framework: Huawei nnU-Net v1；
- task: `Task520_S0R20_Carcass9Class` / r20；
- baseline anchor: `huawei_nnunetv1_r20_5fold_1000ep/batch_20260609T150506Z`；
- prediction/evaluation: `model_final_checkpoint.model`、single fold、no TTA、no ensemble、disable post-processing、不保存 softmax `.npz`；
- evaluator: `metrics/evaluate_swine_ct.py` 的 locked evaluator 口径；
- split/seed: 沿用 Huawei v1 r20 split，训练实验使用 `base_seed=20260520`，fold4 seed `20260524`；
- 重要限制：不要把这些结果与 HZAU 或 Huawei v2 混合比较。

## 1. 背景：为什么做这些实验

Huawei v1 5-fold OOF 的 baseline 暴露出两类主要问题：

1. Kidney：left/right kidney 存在 fold-dependent tail，fold4 尤其明显，表现为左右肾 Dice/HD95 tail 退化。最初假设之一是 nnU-Net 默认 LR-axis mirror augmentation 可能破坏左右类别语义。
2. Head：在 head absent cases 中 baseline 有大量 head false positive。之前 w15/head-logit penalty 能降低 FP，但仍想验证 whole-case presence classifier 是否可以作为 gate，把 absent case 的 head probability 直接压掉。

因此形成三条实验线：

- 先做 LR mirror 相关 ablation，看关闭或条件关闭 LR-axis mirror 是否能修复 kidney tail；
- 再训练 head/testis presence classifier，验证能否可靠判断 case-level organ presence；
- 最后把 classifier 输出接到 segmentation inference，在 classifier absent 时对 head 做 hard-zero probability gate，评估 5-fold OOF 上的上限效果和副作用。

## 2. 实验 A：Kidney LR mirror 关闭与条件 mirror

### 2.1 实验 A1：LR-safe mirror fold4

目标：只关闭确认后的 LR-axis mirror，其他训练口径与 baseline fold4 保持一致，观察 kidney fold4 tail 是否改善。

关键设计：

- 只改 data augmentation 的 mirror axes；
- 原 nnU-Net v1 mirror axes 为 `[0, 1, 2]`；
- 经审计确认 LR axis 是 axis `2`，因此 effective mirror axes 改为 `[0, 1]`；
- 不改 loss、network、sampler、optimizer、patch size、input channels；
- fold: `4`；
- 训练：1000 epochs，`--fp16 --unpack-data --checkpoint-policy final-best`。

核心 wrapper：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/tools/train_lr_safe_mirror_v1_reviewed.py
SHA256: c7452aaf01c1b90b3c9dababa2480f1760e3bd53c1169ec854a9ad7b93fb4a44
```

主要训练 job：

```text
job id: 563380
state: SUCCEEDED
node: whshare-agent-50
resource: cpu=16;gpu=1
start: 2026/06/17 22:28:09
end:   2026/06/18 16:32:16
```

训练输出：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_lr_safe_mirror_fold4_ablation/batch_20260617T142207Z/lr_safe_mirror_fold4_20260617T142207Z
```

prediction/evaluation 输出：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_lr_safe_mirror_fold4_predict_eval/batch_20260618T170731Z/lr_safe_mirror_fold4
```

审计与见证：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_nnunetv1_lr_safe_mirror_lr_axis_audit_20260617T125831Z/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_nnunetv1_lr_safe_mirror_fold4_ablation_20260617T142207Z/runtime_witness_report.md
```

关键 runtime witness：

- `confirmed_lr_axis=2`；
- original mirror axes `[0,1,2]`；
- effective mirror axes `[0,1]`；
- `loss_change=none`；
- `network_architecture_unchanged=true`；
- `sampler_unchanged=true`；
- `optimizer_schedule_unchanged=true`；
- `cudnn_deterministic=true`，`cudnn_benchmark=false`；
- no prediction/evaluation bundled in training job。

### 2.2 实验 A2：conditional LR mirror fold4

目标：保留大多数 class 的 LR mirror augmentation，但当 patch 中出现 protected kidney class 时禁用 LR-axis mirror。该设计试图保留 mirror augmentation 对非左右语义 class 的泛化收益，同时保护 left/right kidney。

关键设计：

- protected classes: left/right kidney class `4/5`；
- protected condition: patch 中 class `4/5` voxel count >= 1；
- protected sample: axis `2` mirror count 必须为 0；
- non-protected sample: axis `2` mirror 仍允许；
- 不改 loss/network/sampler/optimizer。

核心 wrapper：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/tools/train_conditional_lr_mirror_v1_reviewed.py
SHA256: 203a72a9cb32c082f128fc26191c1e74f846f44d75c8c90fd369f7da9b9aa718
```

主要训练 job：

```text
job id: 563810
state: SUCCEEDED
node: whshare-agent-80
resource: cpu=16;gpu=1
start: 2026/06/19 06:33:48
end:   2026/06/20 00:37:15
```

训练输出：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_conditional_lr_mirror_fold4_ablation/batch_20260618T222025Z/conditional_lr_mirror_fold4_20260618T222025Z
```

prediction/evaluation job：

```text
job id: 563927
state: SUCCEEDED
node: whshare-agent-84
resource: cpu=8;gpu=1
start: 2026/06/20 00:46:34
end:   2026/06/20 01:16:31
```

prediction/evaluation 输出：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_conditional_lr_mirror_fold4_predict_eval/batch_20260619T164450Z/conditional_lr_mirror_fold4
```

runtime witness：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_nnunetv1_conditional_lr_mirror_fold4_ablation_20260618T222025Z/runtime_witness_report.md
```

关键 telemetry：

```json
{
  "axis0_mirror_count": 298,
  "axis1_mirror_count": 282,
  "axis2_mirror_count": 187,
  "nonprotected_axis2_mirror_count": 187,
  "protected_axis2_mirror_count": 0,
  "protected_sample_count": 211,
  "nonprotected_sample_count": 389
}
```

解释：conditional transform 确实在含 kidney 的 protected samples 中关闭了 LR-axis mirror，同时保留了 non-protected samples 的 LR-axis mirror。

### 2.3 Fold4 结果：baseline vs LR-safe vs conditional LR mirror

以下结果均来自 Huawei fold4 official prediction/evaluation，39 cases / 351 case-class rows。

| method | rows | evaluable rows | cases | mean Dice | mean IoU | mean HD95 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_fold4 | 351 | 312 | 39 | 0.968 | 0.943 | 10.0 |
| lr_safe_no_lr_axis_fold4 | 351 | 312 | 39 | 0.977 | 0.956 | 6.0 |
| conditional_lr_mirror_fold4 | 351 | 312 | 39 | 0.977 | 0.956 | 6.1 |

Kidney / testis / head class-level comparison：

口径：以下 `FP/GT` 是 official `case_metrics.csv` 中可评估 rows 的 `FP_GT_ratio` 均值；`FN/GT` 是同一批 rows 上逐 case-class 计算 `FN / GT_voxels` 后再取均值。所有值均可从下方 prediction/evaluation 路径中的 `evaluation/case_metrics.csv` 复算。

| method | class | mean Dice | Dice P10 | mean IoU | mean HD95 | HD95 P90 | FP/GT | FN/GT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_fold4 | left_kidney | 0.933 | 0.873 | 0.890 | 16.6 | 64.7 | 0.051 | 0.068 |
| baseline_fold4 | right_kidney | 0.942 | 0.880 | 0.900 | 18.7 | 46.6 | 0.082 | 0.039 |
| baseline_fold4 | testis | 0.946 | 0.930 | 0.898 | 6.0 | 10.0 | 0.052 | 0.055 |
| baseline_fold4 | head | 0.954 | 0.883 | 0.915 | 15.2 | 36.0 | 0.059 | 0.037 |
| lr_safe_no_lr_axis_fold4 | left_kidney | 0.971 | 0.954 | 0.945 | 2.0 | 3.3 | 0.030 | 0.027 |
| lr_safe_no_lr_axis_fold4 | right_kidney | 0.973 | 0.960 | 0.949 | 1.9 | 2.8 | 0.032 | 0.021 |
| lr_safe_no_lr_axis_fold4 | testis | 0.946 | 0.928 | 0.899 | 6.2 | 10.1 | 0.051 | 0.056 |
| lr_safe_no_lr_axis_fold4 | head | 0.955 | 0.884 | 0.917 | 14.3 | 32.4 | 0.063 | 0.032 |
| conditional_lr_mirror_fold4 | left_kidney | 0.972 | 0.954 | 0.946 | 2.0 | 3.1 | 0.033 | 0.024 |
| conditional_lr_mirror_fold4 | right_kidney | 0.974 | 0.960 | 0.949 | 1.9 | 2.3 | 0.031 | 0.022 |
| conditional_lr_mirror_fold4 | testis | 0.946 | 0.935 | 0.898 | 5.9 | 9.2 | 0.053 | 0.054 |
| conditional_lr_mirror_fold4 | head | 0.952 | 0.888 | 0.913 | 15.1 | 34.2 | 0.051 | 0.046 |

结论：

- fold4 上，关闭 LR-axis mirror 或 conditional 关闭 LR-axis mirror 都显著改善 kidney tail，尤其 HD95 P90 从 baseline 的 `left=64.7 / right=46.6` 降到约 `left=3.1-3.3 / right=2.3-2.8`。
- conditional LR mirror 与完全关闭 LR-axis mirror 的 fold4 效果非常接近；conditional 方案更符合“保护左右 kidney，同时保留其它 class mirror augmentation”的设计目标。
- 这仍是 fold4 pilot，不是 5-fold OOF，不应直接写 champion。

### 2.4 如何复现 LR mirror 实验

训练脚本和 job 包在远端：

```text
# LR-safe full disable LR-axis mirror
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/jobs/huawei_nnunetv1_lr_safe_mirror_fold4_ablation/batch_20260617T142207Z/lr_safe_mirror_fold4_20260617T142207Z

# Conditional LR mirror
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/jobs/huawei_nnunetv1_conditional_lr_mirror_fold4_ablation/batch_20260618T222025Z/conditional_lr_mirror_fold4_20260618T222025Z/hv1_cond_lr_mirror_f4_20260618T222025Z.sh
```

复现时建议新建 batch，不覆盖已有输出。按当前策略，先做 validator/static check，再提交 DSUB：

```bash
ssh paca_share 'bash -lc "dsub -s <new_job_script.sh>"'
```

prediction/evaluation 参考已有 job 包：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_r20_5fold_predict_eval/batch_20260610T114041Z/baseline_fold4/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/jobs/huawei_nnunetv1_lr_safe_mirror_fold4_predict_eval/batch_20260618T170731Z/lr_safe_mirror_fold4/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/jobs/huawei_nnunetv1_conditional_lr_mirror_fold4_predict_eval/batch_20260619T164450Z/
```

第一行是 baseline fold4 official prediction/evaluation 输出目录，用于复算 fold4 对照表；后两行是 LR-safe / conditional LR mirror 的 prediction/evaluation job 包。

复现必须保持：

- `model_final_checkpoint.model`；
- no TTA；
- no ensemble；
- disable post-processing；
- 不保存 softmax；
- fold case set 与 Task520 split 一致。

## 3. 实验 B：ROI/Half-Projection Presence Classifier

### 3.1 目标与设计

目标：训练 whole-case level classifier 判断是否存在 head / testis，用于后续探索 case-level gate 或 conditioning。

输入不是 3D volume，而是 2D projection：

- 从 CT whole case 生成 ROI/half-projection 图；
- endpoint: `head_present` 和 `testis_present`；
- ROI roles 包含 head/testis 的 cranial/caudal/lower/upper half 等；
- image variants 包含 `muscle_only_mean`、`muscle_only_p90`、`foreground_thickness`、`bone_only_mip`、`multi_channel_compact`；
- 模型: ResNet-18；
- init: ImageNet 或 random；
- loss: train-only capped `pos_weight` BCEWithLogits；
- seed: `20260520`；
- split: sealed `6:2:2`，但本轮只使用 train/validation，不消费 held-out test。

### 3.2 工具与产物路径

本地工具：

```text
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/task/tools/roi_half_projection_presence_classifier/train_roi_classifier.py
```

远端工具：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/roi_half_projection_presence_classifier/tools/roi_half_projection_presence_classifier/train_roi_classifier.py
SHA256: 9d14c4a792d24ed2790e42cdcdf1df76e5701344743d1de8f4e0da6ae8c9e4e9
```

远端输出根目录：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/roi_half_projection_presence_classifier
```

关键远端产物：

```text
# R3 projection manifest
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/roi_half_projection_presence_classifier/data/manifests/roi_projection_manifest_20260620T093836Z.csv

# R4 60-model training summary
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/roi_half_projection_presence_classifier/reports/20260620T100428Z/training_summary.json

# R4 models
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/roi_half_projection_presence_classifier/models/batch_20260620T100428Z/

# R5 result summary
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/roi_half_projection_presence_classifier/reports/20260620T103539Z/result_card.md
```

说明：R3 `roi_projection_manifest_20260620T093836Z.csv` 是投影样本清单，包含 `variant` 列；R4 `training_summary.json` 的 `grid` 进一步定义 endpoint / roi_role / variant / init_mode 的 60-model training grid。复现实验时应同时核对 R3 manifest rows 与 R4 grid。

本地 mirror 与 final package：

```text
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/reports/roi_half_projection_presence_classifier_final_evidence_package.md
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/reports/roi_half_projection_presence_classifier_final_validator.md
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/remote/reports/roi_half_projection_presence_classifier/20260620T100428Z/
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/remote/reports/roi_half_projection_presence_classifier/20260620T100428Z/selected_training_histories/selected_classifier_training_curves.png
```

R4 training job：

```text
job id: 563949
state: SUCCEEDED
node: whshare-agent-84
resource: cpu=8;mem=64000;gpu=1
runtime: Torch 2.4.1+cuda118, cuDNN 8600
```

### 3.3 Validator chain

所有阶段均有 validator PASS，见：

```text
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/reports/roi_half_projection_presence_classifier_r1_validator.md
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/reports/roi_half_projection_presence_classifier_r2_validator.md
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/reports/roi_half_projection_presence_classifier_r3_validator.md
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/reports/roi_half_projection_presence_classifier_r4_postrun_validator.md
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/reports/roi_half_projection_presence_classifier_r5_validator.md
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/reports/roi_half_projection_presence_classifier_final_validator.md
```

Final validator status: `PASS`。

### 3.4 数据与训练规模

Final evidence package 给出的关键计数：

| item | value |
|---|---:|
| projection rows | 4710 |
| train rows | 3540 |
| validation rows | 1170 |
| test rows | 0 |
| unique train/validation cases | 157 |
| model grid | 60/60 |
| train prediction rows | 7080 = 60 * 118 |
| validation prediction rows | 2340 = 60 * 39 |
| forbidden outputs | 0 |

### 3.5 Validation-only 结果与 caveat

R5 validation-only best rows：

| endpoint | roi_role | variant | init | AUPRC | AUROC | FA | FP | Brier | ECE10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| head_present | head_caudal_half | bone_only_mip | imagenet | 1.0000 | 1.0000 | 0 | 0 | 0.0001 | 0.0051 |
| head_present | head_cranial_half | bone_only_mip | imagenet | 1.0000 | 1.0000 | 0 | 0 | 0.0008 | 0.0130 |
| testis_present | testis_caudal_half | muscle_only_mean | imagenet | 1.0000 | 1.0000 | 0 | 0 | 0.0002 | 0.0103 |
| testis_present | testis_caudal_lower_half | bone_only_mip | imagenet | 1.0000 | 1.0000 | 0 | 0 | 0.0182 | 0.0821 |

重要 caveat：

- source-only control 已经可以在 validation 上完美分开 head/testis labels；
- bbox/ROI shape-only control 也可完美分开；
- wrong ROI controls 也接近或达到正确 ROI 的 AUPRC。

因此，这个 classifier 只证明了当前 split 下的 feasibility 和 shortcut-positive 风险，不能证明 classifier 学到了可靠解剖定位，也不能直接宣称为可部署 gate。

### 3.6 被 hard-zero 使用的 head classifier

后续 B4 / hard-zero 使用的是：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/roi_half_projection_presence_classifier/models/batch_20260620T100428Z/head_present_head_caudal_half_bone_only_mip_imagenet/model_best.pt
SHA256: a4542abe37d4f93a39d25886e2e4997fc680e868fcc1b6859cb67063f1443a75
```

B4 all-case inference CSV：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_logit_bias_conditioner/classifier_inference/20260620T163835Z_b4_allcase_classifier_inference/b4_allcase_head_present_head_caudal_half_bone_only_mip_model_best_probabilities.csv
```

该 CSV 覆盖 197/197 r20 cases。字段中 `b4_probability_role=inference_only_no_classifier_metric_claim`，说明它只是 full-r20 inference 输入，不是 held-out classifier metric claim。

## 4. 实验 C：learned classifier hard-zero schedule 5-fold OOF

### 4.1 目标与设计

目标：在 segmentation inference 阶段，当 classifier 判断 case-level head absent 时，把 head class probability 置 0，再对剩余 class renormalize，然后 argmax 输出 segmentation。

这是 pre-argmax probability transform，不是 final mask 后处理：

```text
softmax/probability map -> head probability hard-zero -> renormalize -> argmax -> segmentation mask
```

本次只做 head，不做 testis。

核心参数：

```text
variant: learned_hard_zero_t050
bias-source: learned_classifier
bias-schedule: hard_zero
head-class-id: 9
head-threshold: 0.5
checkpoint: model_final_checkpoint.model
folds: single fold, 0-4
policy: no TTA, no ensemble, disable post-processing, no saved softmax
```

核心 wrapper：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/tools/huawei_nnunetv1_logit_bias_predict_eval.py
SHA256: 1c30931f1b0997d01a5034e574a81026e0c7386aab2c956683e12cb9a47e4f05
```

`hard_zero` 语义：

- `p_present < threshold`：返回 `-inf`，目标 class probability 置 0；
- `p_present >= threshold`：bias 为 0，present case 不改；
- 对 class 维度 renormalize；
- telemetry 记录 `zero_probability_count`、`zero_probability_absent_count`、`zero_probability_present_count`。

### 4.2 Validator 与 job package

独立 pre-submit validator：

```text
/Users/liuyangfan/Documents/work/AutoScientists/docs/SWCT06042040-LOGIT-BIAS-CONDITIONER-HARD-ZERO-PRESUBMIT-REVIEW.md
Status: PASS
```

远端 jobs root：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/jobs/huawei_nnunetv1_logit_bias_conditioner/batch_20260620T182857Z_b6_head_hard_zero_oof
```

远端 outputs root：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_logit_bias_conditioner/batch_20260620T182857Z_b6_head_hard_zero_oof
```

远端 reports root：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_v1_logit_bias_hard_zero_20260620T182857Z
```

本地 mirror：

```text
/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/remote/reports/huawei_v1_logit_bias_hard_zero_20260620T182857Z
```

### 4.3 Jobs 与调度

最终有效 jobs：

| fold | job id | state | node | masks | rows |
|---:|---:|---|---|---:|---:|
| 0 | 564113 | SUCCEEDED | whshare-agent-84 | 40 | 360 |
| 1 | 564114 | SUCCEEDED | whshare-agent-84 | 40 | 360 |
| 2 | 564115 | SUCCEEDED | whshare-agent-84 | 39 | 351 |
| 3 | 564116 | SUCCEEDED | whshare-agent-84 | 39 | 351 |
| 4 | 564118 | SUCCEEDED | whshare-agent-49 | 39 | 351 |

注意：

- 原 fold4 job `564117` 被提交到 `whshare-agent-82`，但因为该节点 `CPU_FREE=0`，即使 GPU free，调度器无法满足 `cpu=8;gpu=1`，因此该 pending job 被终止；
- `djob -L 564117` 的最终调度器状态字面值为 `FAILED`，`TASK_EXEC_NODES=-`；它不是有效 fold4 结果，也没有作为 output root 写出 failed sentinel；
- fold4 立即重排为 `564118` 到 `whshare-agent-49`；
- 所有实际运行节点均在 170 以前，且不是 `whshare-agent-174`；
- 5 个 fold 总计 `197 masks / 1773 case-class rows`；
- failed sentinel: 0；
- saved softmax `.npz`: 0。

### 4.4 hard-zero telemetry

| fold | active cases | zero absent | zero present |
|---:|---:|---:|---:|
| 0 | 22 | 22 | 0 |
| 1 | 24 | 24 | 0 |
| 2 | 20 | 20 | 0 |
| 3 | 18 | 18 | 0 |
| 4 | 20 | 20 | 0 |

解释：所有 zero 都发生在 classifier 判定 absent 的 case；present case 没有被 zero，因此 head-present Dice/FN 与 baseline 保持一致。

### 4.5 5-fold OOF 结果

OOF summary 文件：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_v1_logit_bias_hard_zero_20260620T182857Z/hard_zero_variant_summary.csv
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_v1_logit_bias_hard_zero_20260620T182857Z/hard_zero_class_metrics_long.csv
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_v1_logit_bias_hard_zero_20260620T182857Z/learned_hard_zero_t050_oof_case_metrics.csv
```

主表：

| method | cases | rows | all Dice | all IoU | all HD95 | non-head Dice | head-present Dice | head-present Dice P10 | head-present HD95 P90 | head-present FP/GT | head-present FN/GT | head-absent FP cases | head-absent FP voxels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 197 | 1773 | 0.970 | 0.946 | 9.1 | 0.971 | 0.957 | 0.917 | 25.0 | 0.048 | 0.036 | 86 | 1176664 |
| w15 | 197 | 1773 | 0.970 | 0.946 | 9.2 | 0.971 | 0.958 | 0.918 | 25.0 | 0.045 | 0.037 | 47 | 479756 |
| learned_logodds_neg_alpha05 | 197 | 1773 | 0.970 | 0.946 | 9.1 | 0.971 | 0.957 | 0.917 | 25.0 | 0.048 | 0.036 | 24 | 183658 |
| learned_hard_t050_l3 | 197 | 1773 | 0.970 | 0.946 | 9.1 | 0.971 | 0.957 | 0.917 | 25.0 | 0.048 | 0.036 | 25 | 239170 |
| learned_hard_zero_t050 | 197 | 1773 | 0.970 | 0.946 | 9.1 | 0.971 | 0.957 | 0.917 | 25.0 | 0.048 | 0.036 | 0 | 0 |

结论：

- `learned_hard_zero_t050` 在当前 OOF 上把 head-absent FP cases 从 baseline 的 `86/104` 降到 `0/104`，FP voxels 从 `1,176,664` 降到 `0`；
- head-present Dice / FN / HD95 与 baseline 完全一致到当前报告精度；
- non-head mean Dice 有极小数值提升：`0.971167 -> 0.971237`；
- kidney/testis 指标没有变化。

### 4.6 对其它 class 的影响

Hard-zero 与 baseline 的 class-level 变化：

| class | baseline Dice | hard-zero Dice | Δ Dice | baseline HD95 | hard-zero HD95 | Δ HD95 |
|---|---:|---:|---:|---:|---:|---:|
| front | 0.981886 | 0.982415 | +0.000529 | 12.334 | 12.138 | -0.195 |
| middle | 0.976620 | 0.976619 | -0.000001 | 9.806 | 9.809 | +0.004 |
| end | 0.988127 | 0.988127 | 0.000000 | 6.493 | 6.493 | 0.000 |
| left_kidney | 0.941524 | 0.941524 | 0.000000 | 14.498 | 14.498 | 0.000 |
| right_kidney | 0.946944 | 0.946944 | 0.000000 | 15.122 | 15.122 | 0.000 |
| testis | 0.948108 | 0.948108 | 0.000000 | 6.089 | 6.089 | 0.000 |
| thoracic_cavity | 0.986089 | 0.986088 | -0.000001 | 2.383 | 2.383 | 0.000 |
| abdominal_and_pelvic_cavity | 0.989155 | 0.989154 | 0.000000 | 2.638 | 2.638 | 0.000 |
| head present rows | 0.956908 | 0.956908 | 0.000000 | 13.590 | 13.590 | 0.000 |

解释：hard-zero 只在 head absent cases 把 head probability 置 0，再把概率质量重新分配给其它 class。非-head class 的可见变化接近数值噪声；front 有轻微改善，可能来自原本 head FP 占掉的概率回流。

### 4.7 如何复现 hard-zero

使用远端 wrapper：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/tools/huawei_nnunetv1_logit_bias_predict_eval.py
```

核心命令参数：

```bash
python /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/tools/huawei_nnunetv1_logit_bias_predict_eval.py \
  --method learned_hard_zero_t050 \
  --fold <0-4> \
  --task-id 520 \
  --train-output <baseline_fold_train_output> \
  --reference-cases /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/data/reference/r20/evaluation_cases.csv \
  --output-root <new_output_root> \
  --bias-source learned_classifier \
  --classifier-predictions /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_logit_bias_conditioner/classifier_inference/20260620T163835Z_b4_allcase_classifier_inference/b4_allcase_head_present_head_caudal_half_bone_only_mip_model_best_probabilities.csv \
  --bias-schedule hard_zero \
  --enable-head-bias \
  --head-class-id 9 \
  --head-threshold 0.5 \
  --num-threads-preprocessing 6 \
  --num-threads-nifti-save 2 \
  --num-workers 8 \
  --hd95-workers 4
```

实际 batch job scripts：

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/jobs/huawei_nnunetv1_logit_bias_conditioner/batch_20260620T182857Z_b6_head_hard_zero_oof/
```

复现时建议复制这些脚本到新 batch，而不是覆盖原 batch。提交前必须重新检查：

- wrapper SHA 是否仍是 `1c30931f1b0997d01a5034e574a81026e0c7386aab2c956683e12cb9a47e4f05` 或明确记录新 SHA；
- classifier CSV 覆盖 197 cases；
- fold0/1 为 40 cases，fold2/3/4 为 39 cases；
- no TTA、no ensemble、disable post-processing、不保存 softmax；
- 节点约束符合当前用户要求；
- validator PASS 后再提交。

## 5. 重要 caveats 与下一步建议

### 5.1 不要把 classifier hard-zero 当作最终 champion

虽然 `learned_hard_zero_t050` 在 head absent FP 上是当前最干净的结果，但它依赖的 classifier 有明确 caveat：

- classifier 的 validation-only 表现可能受 source/FOV/shape shortcut 驱动；
- B4/hard-zero 使用的是 full-r20 inference-only classifier probabilities；
- 这不是 leakage-free held-out classifier claim；
- 因此 hard-zero 是 gate 上限 / sanity check，不是可以直接写成部署方案或 champion 的证据。

### 5.2 Kidney mirror 实验下一步

fold4 结果强烈支持 LR-axis mirror 是 kidney tail 的重要风险因素。但目前只有 fold4 pilot。下一步如果要走论文主线，应考虑：

- 先把 conditional LR mirror 扩展到剩余 folds，做 official 5-fold OOF；
- 与 baseline、combo kidney loss、beta asym/recall floor 等已有 OOF 做同表比较；
- 检查 conditional LR mirror 是否在其它 folds 保持 kidney 改善，而不是只修 fold4；
- 同时观察 testis/head/large cavity 是否有副作用。

### 5.3 Classifier 下一步

若要让 classifier 成为真正可用的条件模块，需要先解决 shortcut：

- source-stratified 或 same-source validation；
- shape-control matching；
- 更严格的 ROI crop/normalization；
- 不使用 source/FOV 直接可分的信息；
- held-out test 在设计稳定后再开启。

## 6. 快速检查命令

### 6.1 检查 Huawei hard-zero 5-fold 完整性

```bash
ssh paca_share 'bash -lc '"'"'
ROOT=/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_logit_bias_conditioner/batch_20260620T182857Z_b6_head_hard_zero_oof
for id in 564113 564114 564115 564116 564118; do
  echo "### job $id"
  djob -L $id | egrep "JOB_STATE|TASK_EXEC_NODES|TASK_SUCCEEDED_CNT|TASK_FAILED_CNT"
done
for f in 0 1 2 3 4; do
  OUT=$ROOT/learned_hard_zero_t050_fold${f}
  masks=$(find "$OUT/predictions" -maxdepth 1 -name "*.nii.gz" | wc -l | tr -d " ")
  rows=$(($(wc -l < "$OUT/evaluation/case_metrics.csv")-1))
  failed=$(find "$OUT/records" -maxdepth 1 -name "failed*json" | wc -l | tr -d " ")
  npz=$(find "$OUT" -name "*.npz" | wc -l | tr -d " ")
  echo "fold$f masks=$masks rows=$rows failed=$failed npz=$npz"
done
'"'"'
```

Expected:

```text
fold0 masks=40 rows=360 failed=0 npz=0
fold1 masks=40 rows=360 failed=0 npz=0
fold2 masks=39 rows=351 failed=0 npz=0
fold3 masks=39 rows=351 failed=0 npz=0
fold4 masks=39 rows=351 failed=0 npz=0
```

### 6.2 检查 hard-zero summary

```bash
python3 - <<'PY'
import pandas as pd
p="/Users/liuyangfan/Documents/work/AutoScientists/output/swct06042040/remote/reports/huawei_v1_logit_bias_hard_zero_20260620T182857Z/hard_zero_variant_summary.csv"
df=pd.read_csv(p)
print(df[["method","cases","rows","all_mean_dice","head_absent_fp_cases","head_absent_fp_voxels"]].to_string(index=False))
PY
```

### 6.3 检查 LR mirror fold4 结果

```bash
ssh paca_share 'bash -lc '"'"'
for p in \
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_lr_safe_mirror_fold4_predict_eval/batch_20260618T170731Z/lr_safe_mirror_fold4/evaluation/case_metrics.csv \
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_conditional_lr_mirror_fold4_predict_eval/batch_20260619T164450Z/conditional_lr_mirror_fold4/evaluation/case_metrics.csv
do
  echo "$p"
  wc -l "$p"
done
'"'"'
```

## 7. 文件索引

### 7.1 Specs / validators / handoff docs

```text
docs/SWCT06042040-LR-SAFE-MIRROR-ABLATION-SPEC.md
docs/SWCT06042040-LR-SAFE-MIRROR-ABLATION-SPEC-INDEPENDENT-REVIEW.md
docs/SWCT06042040-ROI-HALF-PROJECTION-PRESENCE-CLASSIFIER-SPEC.md
docs/SWCT06042040-ROI-HALF-PROJECTION-PRESENCE-CLASSIFIER-SPEC-REREVIEW.md
docs/SWCT06042040-LOGIT-BIAS-CONDITIONER-B4-LEARNED-CLASSIFIER-OOF-REPORT.md
docs/SWCT06042040-LOGIT-BIAS-CONDITIONER-HARD-ZERO-PRESUBMIT-REVIEW.md
```

### 7.2 Local mirrors

```text
output/swct06042040/remote/reports/huawei_nnunetv1_lr_safe_mirror_lr_axis_audit_20260617T125831Z/
output/swct06042040/remote/reports/huawei_nnunetv1_lr_safe_mirror_fold4_ablation_20260617T142207Z/
output/swct06042040/remote/reports/huawei_nnunetv1_conditional_lr_mirror_fold4_ablation_20260618T222025Z/
output/swct06042040/remote/reports/roi_half_projection_presence_classifier/
output/swct06042040/remote/reports/huawei_v1_logit_bias_b4_20260620T164316Z/
output/swct06042040/remote/reports/huawei_v1_logit_bias_hard_zero_20260620T182857Z/
```

### 7.3 Remote reports

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_nnunetv1_lr_safe_mirror_lr_axis_audit_20260617T125831Z/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_nnunetv1_lr_safe_mirror_fold4_ablation_20260617T142207Z/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_nnunetv1_conditional_lr_mirror_fold4_ablation_20260618T222025Z/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/reports/huawei_v1_logit_bias_hard_zero_20260620T182857Z/
```

### 7.4 Remote outputs

```text
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_lr_safe_mirror_fold4_ablation/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_lr_safe_mirror_fold4_predict_eval/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_conditional_lr_mirror_fold4_ablation/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_conditional_lr_mirror_fold4_predict_eval/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/roi_half_projection_presence_classifier/
/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/huawei_nnunetv1_logit_bias_conditioner/
```

## 8. Claude Code 接手时的建议

1. 不要直接清理这些目录；它们是当前证据链。
2. 如果要继续 kidney 方向，优先把 `conditional_lr_mirror_fold4` 扩展为 5-fold OOF，而不是只看 fold4。
3. 如果要继续 classifier/gate 方向，先解决 shortcut 风险，不要把 `learned_hard_zero_t050` 作为最终方法直接推广。
4. 每次提交新训练或 prediction/evaluation 前，保留当前 validator-gated 习惯：先审查 wrapper/数据/split/seed/policy，再提交；完成后检查 masks、rows、failed sentinels、`.npz`、scheduler state。
5. 所有 board/report 结论应明确 scope：Huawei v1 / Task520 / r20；不要主动跨服务器、跨 v2 比较。
