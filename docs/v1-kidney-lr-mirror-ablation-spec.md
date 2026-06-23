# v1 Kidney LR-mirror conditional ablation —— 正式 Spec

Status: **ERRATA — 2D arm invalidated (2026-06-24)**
Date: 2026-06-23 (original) / 2026-06-24 (errata)
Source: 由 [`v1-kidney-lr-mirror-ablation-spec-draft.md`](v1-kidney-lr-mirror-ablation-spec-draft.md)(设计 6 + B 类 11 + 审查 §13 18 + §14 统一 run_eval.py 6 = 41 项决策)整理;含两轮 subagent 审查(REVIEW.md)修订 + 评估器统一(§14)。

## Errata (2026-06-24): 2D axis audit 审计了错误的切片方向，禁用了错误的 axis

**错误**: Stage 0 的 2D axis audit 硬编码 `SLICE_AXIS_3D=2`（假设轴向切片），但 nnU-Net 2D 实际切片方向由 spacing 决定——对本数据集，spacing=(5.0, 0.977, 0.977)mm，**切片轴 = axis 0 (LR)**。

**关键纠正**: 2D 臂**不是**"不适用"——预处理数据中，单张 2D slice 内**可以同时出现左右肾**（平均每个 case 有 ~18 张 slice 包含两个肾）。但两个肾在 slice 内的分离方向是 **CC (W 轴, axis 1)**，不是 AP (H 轴, axis 0)：
- W 分离: 106-179 voxels（主要）
- H 分离: 1-17 voxels（很小）

**后果**: `conditional_mirror_axis=0` 禁用了 AP mirror（对肾位置影响很小），而真正导致左右混淆的 **CC mirror (axis 1) 没有被禁用**。2D condlr 6 个训练作业结果**全部无效**（禁用了错误的 axis）。

**正确做法**: `conditional_mirror_axis` 应为 **1**（2D 中的 W/CC 轴），而非 0。

**修正**:
- `confirmed_lr_axis_2d` = **1**（不是 0，也不是 N/A）
- 2D condlr 训练/预测/评估结果应**废弃**（用了错误 axis）
- 需要重新训练 2D condlr（`conditional_mirror_axis=1`），或放弃 2D 臂
- 3D SwinUNETR 不受影响（`mirror_axes=(0,1,2)` 包含所有轴）
- 参见修正后的 audit: `tools/audit_output_v2/lr_axis_audit_manifest.json` (v2.1)

关联:
- 主对比 spec:[`v1-input-consistency-spec.md`](v1-input-consistency-spec.md)(环境/路径/公平性协议,本文 §3 引用)
- 决策记录:[`v1-kidney-lr-mirror-ablation-spec-draft.md`](v1-kidney-lr-mirror-ablation-spec-draft.md)
- 先验证据:[`SWCT06042040-HUAWEI-V1-KIDNEY-MIRROR-CLASSIFIER-HARDZERO-HANDOFF.md`](SWCT06042040-HUAWEI-V1-KIDNEY-MIRROR-CLASSIFIER-HARDZERO-HANDOFF.md)
- 实验跟踪:GitHub issue #1;2D 统一入口 commit `c585966`

---

## 1. 目标 + scope

在 issue #1 主对比中分割质量最好的两个网络(**nnU-Net 2D、SwinUNETR**)上,把训练增强里的
LR(left-right)axis mirror 改为**条件禁用**(conditional):仅当 sample 含 kidney class(4/5)时禁用
该 sample 的 LR flip,其余 sample 保留全套 mirror。验证能否改善 kidney 的 tail 指标(Dice P10 /
HD95 P90)与左右混淆(swap rate / LP-Dice gap),同时确认对其它 class 无副作用。

**scope 严格声明:**
- framework = swine-CT-article 的 `framework/`(Task601, 3-seed, locked evaluator);
- 2D 与 3D 统一走 `MultiNetworkTrainer`(commit `c585966`),改 mirror 是同一处代码;
- 网络 = nnU-Net 2D + SwinUNETR 两个(**⚠️ ERRATA 2026-06-24: 2D condlr 因禁用错误 axis 而无效，待决策是否重训或放弃 2D 臂**);
- 论文不 claim "架构无关/普适"；
- **不**与 handoff(Huawei v1 Task520 r20 fold4)结果混比 —— 只作先验与方法论参考;
- **不**碰 head/testis 条件类 gate(handoff 实验 C 的方向)。

## 2. 阵容(2 网络)

| 网络 | dim | optimizer | baseline 来源 |
|---|---|---|---|
| SwinUNETR(V2,MONAI) | 3D | AdamW(4e-4, warmup-cosine) | issue #1 `v1_comparison/swinunetr__seed<seed>/`(复用) |
| nnU-Net 2D | 2D | SGD(0.01, poly) | issue #1 `v1_comparison_2d_root/...`(PACA 版,复用,不重训) |

**训练 job:2 网络 × 3 seed = 6**(seeds 20260520/21/22;baseline 全复用不重训 —— 改前 PACA 与改后 MultiNetworkTrainer 等价已核实,见 §9)。

---

## 3. 环境与路径(执行前必备,引用 v1 spec §3/§4)

所有路径在 Huawei `paca_share`,数据根覆盖到 swine-CT-article:

```bash
NNUNETV1_PYTHON=/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/envs/nnunetv1/bin/python
NNUNETV1_COMPAT_ROOT=/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/scripts/nnunetv1_compat
SETUP=/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
export nnUNet_results=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_results
export PYTHONPATH="$NNUNETV1_COMPAT_ROOT:$PYTHONPATH"

source $SETUP
module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.8.1_cuda11 libs/nccl/2.16.5-1_cuda11.8 libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
swct_nnunetv1_preflight   # 必须通过
```

- Task601 preprocessed:`$nnUNet_preprocessed/Task601_Article622_Carcass9Class/`
  - 3D 用 `nnUNetData_plans_v2.1_stage1/`(patch `[64,160,160]`);2D 用 `nnUNetData_plans_v2.1_2D_stage0/`;
  - split:`splits_final.pkl`(单 fold train=120/val=38,`place_split.py` 已放,2D/3D 共用)。
- pip:`monai`(SwinUNETR)。

---

## 4. 公平性协议(与 baseline 逐项对齐,唯一变量 = mirror 对 kidney-sample 的处理)

| 维度 | 锁定值 | 备注 |
|---|---|---|
| 数据 + split | Task601, fold 0, train=120/val=38 | `place_split.py` 产物,2D/3D 共用 |
| seeds | 20260520 / 21 / 22 | **必须复用**(per-case 配对) |
| patch | SwinUNETR `[64,160,160]` batch2;2D 走 2D plans patch | 各自跟 baseline |
| 其它增强 | moreDA 全套(除 LR mirror 对 kidney-sample) | 只动这一处 |
| sampling | force-fg 0.33 | 不变 |
| budget | 500ep × 250 = 125 000 iters | 不变 |
| loss | DC_and_CE_loss(DS 权重同 baseline) | 不变 |
| optimizer | SwinUNETR=AdamW(4e-4, warmup-cosine);2D=SGD(0.01, poly) | 各自跟 baseline |
| checkpoint | final only | 不变 |
| determinism | cudnn.deterministic=True, benchmark=False | 不变 |
| predict | sliding window 0.5, TTA off, no PP | `framework/predict.py` |
| eval | `run_eval.py`(locked evaluator verbatim port,30 列)+ kidney_swap_eval | 与 issue #1 同口径 |
| trainer 子类 | PACA(旧 2D baseline)与 MultiNetworkTrainer 等价 | 已核实(§9) |

> 公平性叙事:baseline = 标准 moreDA(所有 sample 都可能翻 LR);conditional = 含 kidney 的 sample
> 不翻 LR。唯一变量 = LR mirror 是否对 kidney-sample 生效,归因干净。

---

## 5. conditional 实现(工程核心)

> conditional 不能用"改 mirror_axes 一行"实现 —— `get_moreDA_augmentation` 内部 `MirrorTransform`
> 对整 batch 统一翻转,不支持 per-sample 条件。需要自定义 transform + 模块符号 patch。**整套照搬 handoff
> A2 的三组件**(`train_conditional_lr_mirror_v1_reviewed.py`,已验证)。

### 5.1 代码落点(B-1)
新建 `framework/transforms/conditional_mirror.py`,集中放三组件:
- `ConditionalMirrorTransform` 类(handoff line 212-371);
- `install_conditional_mirror_patch`(handoff line 375-435);
- `factory`(闭包,注入参数返回 ConditionalMirrorTransform)。

### 5.2 transform 注入:模块符号 patch(B-6)
```python
import nnunet.training.data_augmentation.data_augmentation_moreDA as da_more
da_more.MirrorTransform = factory   # 替换 moreDA 模块的 MirrorTransform 符号指向
```
- **不 fork `get_moreDA_augmentation`** —— 该函数内部 `MirrorTransform(...)` 调用自动变成 `factory()`
  → 返回 `ConditionalMirrorTransform`;
- 模块符号级 patch,影响面可控(只 moreDA 模块内,非全局类);
- **per-sample 机制**:batchgenerators transform 本就 per-sample;ConditionalMirrorTransform 读当前
  sample 的 `data_dict["seg"]` → 含 kidney(class 4/5 voxel ≥ 1)则该 sample LR **强制不翻**,
  其余 axis 正常随机;不含 kidney 则全套随机。**DS 顺序(机制)**:DS 下采样由 moreDA pipeline 的
  `DownsampleSegForDSTransform`(`data_augmentation_moreDA.py:145-150`)完成,位置在 `MirrorTransform`(`:111`)
  与 `RenameTransform('seg'→'target')`(`:140`)**之后**;故 MirrorTransform 执行时 `data_dict["seg"]` 仍是
  原始分辨率、key 仍为 `seg`,ConditionalMirrorTransform 判 kidney(class 4/5 voxel ≥ 1)用 full-res patch label。

**install 时序契约(M1,顺序敏感)**:`install_conditional_mirror_patch` 必须在 `get_moreDA_augmentation()`
被调用前完成,且晚于 determinism patch。落点:**`MultiNetworkTrainer.initialize()` 内 `setup_DA_params()`(base_trainer.py:197)
之后、`get_moreDA_augmentation()`(:230)之前**(决策 B);train.py:77 的 `install_v1_determinism_patches` 已先于
initialize 执行。**顺序错 → conditional 静默失效**(telemetry `protected_lr_mirror_count==0` 会兜底,Stage 2 就抓)。

### 5.3 train.py CLI(B-2 + B-10)
```bash
python -m framework.train \
    --network {swinunetr|nnunet_v1} [--network-dim 2d] \
    --seed <s> --config configs/<net>.yaml \
    --output-folder <OUT> \
    --lr-mirror-mode conditional \
    --conditional-mirror-axis <audit_value>   # 3D/2D 不同,来自 §6.1 audit
    [--lr-axis-audit-manifest <path>]         # 可选:启动时校验 axis == manifest
```
- `--lr-mirror-mode {full,conditional}`(default `full`,baseline 路径不变);
- `--conditional-mirror-axis <int>`:从 audit 产的 `confirmed_lr_axis_3d` / `confirmed_lr_axis_2d`;
  **启动校验(R5)**:2D 时 axis 必须 ∈ {0,1},train.py reject axis=2(防 telemetry 假 PASS —— axis 错时保护永不触发但计数==0);
- `MultiNetworkTrainer.__init__` 加 `lr_mirror_mode` 参数,透传。

### 5.4 参数(B-10)
- **固定参数**(trainer 常量 / install_patch 默认,不暴露冗余 CLI):
  `protected_class_ids=(4,5)` / `protected_min_voxels=1` / `log_interval=25` / `p_per_sample=1`;
  `run_name`/`instance_index` 由 install_patch 自动生成。
- **动态参数**:`conditional_mirror_axis`(来自 audit,CLI 传入)。

### 5.5 telemetry + witness(B-7 + B-11)
- **telemetry**:`ConditionalMirrorTransform` 实例属性 `self._window` 累计(per-instance):
  `protected_sample_count` / `nonprotected_sample_count` / `protected_lr_mirror_count`(**必须=0**)/
  `nonprotected_lr_mirror_count` / `axis0/1/2_mirror_count`;
- 输出 `<record_dir>/conditional_mirror_components.jsonl`(**JSONL 流式**,每 log_interval=25 append);
- **witness 链**(对齐 handoff,完整证据链):
  - `preflight.json`:训练前自检(env/config/axis/protected 参数齐全);
  - `setup_witness`(并入 augmentation_witness):setup_DA_params 记录改前 `original mirror_axes` / `do_mirror`;
  - `augmentation_witness.json`:配置 + `remaining_original_mirror_transform_count==0`(硬验证原 MirrorTransform 已替换)+ `protected_lr_mirror_count==0`;
  - `runtime_witness_report.md`:Q7 对齐表的"不变项"落盘(loss/network/sampler/optimizer/cudnn_deterministic 不变);
  - env/done/failed:`MultiNetworkTrainer` 的 train.py 已有。
- **提交前必读**:`protected_lr_mirror_count==0` 且 `remaining_original_mirror_transform_count==0`,
  才能进 Stage 4。

---

## 6. axis audit + 2D smoke + determinism(Stage 0)

### 6.1 LR axis audit(B-5)
- 脚本:**新建 `tools/audit_lr_axis.py`**(新建 `tools/` 目录);
- Task601 ≠ handoff Task520,**必须重测**(不能沿用 axis 2);
- 方法:取双肾都在且左右不对称的 case(HZAU/TB),读 GT label,沿各 axis `np.flip`,看 class 4(left_kidney)
  与 class 5(right_kidney)体素位置是否互换 → 互换的 axis = LR;
- 2D:确认 axial slice 内哪个 axis 是真 LR(2D plans 把所有 slice 混采);
- 产物:`confirmed_lr_axis_3d` / `confirmed_lr_axis_2d` + witness md(manifest);
- **2D 校验(R5)**:`confirmed_lr_axis_2d ∈ {0,1}`(2D mirror_axes=(0,1));audit 脚本 reject axis=2(for 2D),防后续假 PASS。

### 6.2 2D smoke(B-8,实质改动 —— 独立 dim 分支)
`framework/smoke_framework.py` 现有 all-networks 循环**硬编码 3D**(`get_default_configuration("3d_fullres")` /
patch `(64,160,160)` / input `[2,1,64,160,160]`),2D 无法塞进。改动(R4):
- **新增 `--network-dim` 分支**:dim=2d 走**单网络路径**(`get_default_configuration("2d")` + 读 2D plans patch +
  `[2,1,H,W]` 输入),**不复用 all-networks 循环**;
- 删 line 50-51 的 `nnunet_2d` skip(死代码:registry 无 nnunet_2d 条目,2D = `nnunet_v1` + `--network-dim 2d`);
- 2D 断言:`conv_op=Conv2d` 构建 / DS 路数(2D plans pool,打印 `n_outputs`)/ `mirror_axes==(0,1)` / forward `[2,10,H,W]`。

```bash
$NNUNETV1_PYTHON -m framework.smoke_framework --network nnunet_v1 --network-dim 2d --seed 20260520
```

### 6.3 determinism(Q13)
conditional 引入新随机逻辑,同 seed 跑两次比 `state_dict_equal`(handoff 做过)。

---

## 7. 评估与统计

### 7.1 指标(Q8)
- **kidney 重点**:L/R kidney mean Dice、Dice P10、mean HD95、HD95 P90、FP/GT、FN/GT;
  左右混淆 swap rate / LP-Dice gap / 混淆 case 比例(复用 `evaluation/kidney_swap_eval.py`,口径同 issue #1);
- **副作用**:全 9 class mean Dice/IoU/HD95,逐 class baseline vs conditional 的 Δ;条件类 absent-FP;
  特别盯 front(handoff 里关 mirror 后 front 极轻微 +Δ)。

### 7.2 统计口径(Q9)+ 脚本(B-4)
- **新建 `evaluation/run_paired_stats.py`**;**底层 Wilcoxon/Holm 抽共用 helper(R6)**:新建
  `evaluation/stats_helpers.py`,把 `holm_bonferroni`(+ wilcoxon wrapper)从 `run_stats.py` 移过去,
  `run_stats.py` + `run_paired_stats.py` 都 import(refactor run_stats 只移动不改逻辑,前后 smoke 确认输出一致);
- 配对:per-case kidney Dice/HD95,baseline vs conditional,**同 (case, seed)**,先 3-seed 平均;
- 主检验:Wilcoxon signed-rank(双侧);
- Holm 族:2 网络 × 2 kidney 侧(left/right)× 2 指标(Dice, HD95)= **8 检验**(跨 2 网络统一校正);
- **HD95 纳入检验**(primary,与 issue #1 不同 —— tail 是核心 hypothesis);
- 描述性(不检验):HD95 P90、Dice P10、swap rate、LP-gap、混淆 case 比例(分布偏 0/tie 多);
- 输出 `evaluation/results/condlr_vs_baseline_paired_stats.csv`(p / Holm-adj / 效应量)。
- **输入契约(M3,统一 run_eval.py 30 列)**:`per_case.csv` 由 `evaluation/run_eval.py` 产出
  (evaluate_swine_ct.py 的 verbatim port:全 confusion 指标 + `seed` 列),30 列 =
  `network,seed,case_id,source,class_id,class_name,is_evaluable,domain_voxels,GT_voxels,Pred_voxels,TP,FP,FN,TN,
  TP_percent,FP_percent,FN_percent,TN_percent,Dice,IoU,Precision,Recall,Specificity,FPR,FP_GT_ratio,HD95,missed,
  absent_FP_voxels,absent_FP_rate,absent_FP_incidence`;
  join key `(case_id,class_id)`,先 3-seed 平均(Dice & HD95 各自,`np.nanmean` over 3 seed,该 case 全 NaN 才 drop pair);
  两臂标记 baseline(`nnunet_2d`/`swinunetr`)vs conditional(`nnunet_2d_condlr`/`swinunetr_condlr`,由 `run_eval.py --network`
  传入);检验对象 class∈{4,5}(left/right kidney);HD95 NaN → drop 该 pair(继续用 case 其它 class);效应量 r=|z|/√N + mean Δ。
- **单一评估器(§C1 统一)**:baseline + conditional 都跑 `run_eval.py`(产 30 列),schema 一致;不再双轨
  run_eval(8 列)+ evaluate_swine_ct(34 列)。issue#1 旧 8 列 per_case.csv 锁历史(C-5),不碰。

### 7.3 baseline 数据路径 + 评估重跑(C-6)
- **模型/ckpt 不重训(Q3)**:复用 issue#1 baseline ckpt + predictions;
  - SwinUNETR:`v1_comparison/swinunetr__seed<seed>/fold_0/model_final_checkpoint.model` + `v1_comparison_predictions/swinunetr__seed<seed>/`;
  - nnU-Net 2D(PACA 版):`v1_comparison_2d_root/seed<seed>/runs/nnunet_2d__seed<seed>/fold_0/model_final_checkpoint.model` + `v1_comparison_predictions/nnunet_2d__seed<seed>/`。
- **评估重跑(C-6,30 列 schema 一致)**:用新 `run_eval.py` 重新评估上述 baseline predictions(产 30 列),
  与 conditional schema 一致(pair stats 才能配对)。**不是重训模型**,只重评估(6 个 × 39 cases,分钟级)。
  产出放 `evaluation/results/ablation_baseline_30col/`(issue#1 旧 8 列 per_case.csv 保留锁历史,C-5)。

### 7.4 评估命令链(单一 run_eval.py,§C1 统一)
- **评估器**:`evaluation/run_eval.py`(统一,evaluate_swine_ct.py 的 verbatim port + `seed` 列)。baseline +
  conditional 各跑一遍(`--network` 标记 arm),产出 30 列 per_case.csv 供 pair stats:
  ```bash
  $NNUNETV1_PYTHON -m evaluation.run_eval --predictions <pred_dir> --gt-folder <labelsTs> \
    --case-metadata <case_metadata.csv> --network <arm> --seed <s> --output-csv <per_case.csv>
  ```
- **kidney_swap**:`evaluation/kidney_swap_eval.py` 独立跑(吃 predictions,非 cases-csv 下游)。
- **make_figures(C-3)**:改读 `run_eval.py` 的 30 列 per_case.csv(单一数据源,消除 `results_locked/` 双轨);
  列名大写(Dice/IoU/HD95)与原 34 列兼容。
- **build_cases_csv / evaluate_swine_ct(C-4,保留备用)**:canonical locked evaluator 是项目资产,保留作
  verbatim 对照 / 交叉验证;默认不调(run_eval 直接读 predictions)。R3 的 build_cases_csv `--variant` 仅走 canonical 链时用。

### 7.5 阴性/混合结果(Q10)
预设双向假设:conditional 无效/变差 = "全局注意力已隐式处理左右"的有效结论,照实写;2D 与 SwinUNETR
结论相反 = "影响架构相关",同样有价值。禁 confirmation bias。

---

## 8. 工程与调度

### 8.1 训练规模(Q11)
- conditional:SwinUNETR ×3 + nnunet_2d ×3 = **6 job**(baseline 全复用不重训);
- 节点:`agent<170`、**不**用 `whshare-agent-174`;单卡并发;
- early-runtime 检查:dsub 后 1–2min `djob <id>`,FAILED 立刻读 `.err` 修;长任务每 ~10min 复查。

### 8.2 产物命名(B-3,绝不覆盖 baseline)
后缀 **`__condlr`**(conditional,非 `__nolr`):
- ckpt `data/nnunetv1/v1_comparison/{swinunetr,nnunet_2d}__condlr_seed<seed>/fold_0/model_final_checkpoint.model`;
- pred `data/nnunetv1/v1_comparison_predictions/{swinunetr,nnunet_2d}__condlr_seed<seed>/*.nii.gz`;
- eval `evaluation/results/{swinunetr,nnunet_2d}_condlr/`(run_eval 30 列 per_case.csv)。

### 8.3 job 生成(B-9)
扩展 `jobs/train/generate_train_jobs.py` 加 `--variant {baseline,condlr}`(默认 baseline,不破坏现有):
- **axis 注入(R2)**:generator `--variant condlr` 时加 `--lr-axis-manifest <path>`,读 audit manifest 把
  `confirmed_lr_axis_3d`/`_2d` bake 进脚本的 `--conditional-mirror-axis`(与 train.py `--lr-axis-audit-manifest` 同源,单一真相);
- 3D conditional:`_3d_script` 模板 + `--lr-mirror-mode conditional` + `--conditional-mirror-axis <v>` + `__condlr`;
- 2D conditional:**新模板** `framework/train.py --network nnunet_v1 --network-dim 2d --lr-mirror-mode conditional
  --conditional-mirror-axis <v>` + `__condlr`(不走旧 train_paca_deterministic.py,因要触发 --lr-mirror-mode);
- baseline 2D 保持旧 `train_paca_deterministic.py`(复用不重训;trainer 等价已核实);
- predict job:`generate_predict_jobs.py` 加 `--variant condlr`;**2D condlr 用新模板**(2D condlr ckpt 在
  `v1_comparison/nnunet_2d__condlr_seed<seed>/`,与 baseline 的 `v1_comparison_2d_root/` 根目录不同,非"只换后缀"),
  CKPT 指 `v1_comparison/nnunet_2d__condlr_seed<seed>/fold_0/model_final_checkpoint.model`,走
  `framework.predict --network-dim 2d`(predict.py:43 已支持),eval 标签 `nnunet_2d_condlr`;3D condlr 同在
  `v1_comparison/`,加 `__condlr` 后缀即可。

### 8.4 批次提交策略(6 job)
- **提交方式**:简单 loop dsub(不碰 `login_node_orchestrator.sh`,避免影响 issue #1 复现);generate 6 脚本后 loop;
- **并发**:不设硬上限,一次提 6 个,调度器按空闲 GPU 分配(不够的 pending);
- **顺序**:混提(6 job 无依赖,一次性);3 seed 并发;
- **early-runtime**:逐个(每 job 提交后 1-2min `djob <id>` 查 RUNNING);
- **失败重提**:单独(读 .err 修后单独 dsub,不动其它);
- **Stage 依赖**:Stage 3 validator PASS 才提 Stage 4。

---

## 9. 风险、边界与定位

### 9.1 风险清单(Q14)
| 风险 | 说明 | 缓解 |
|---|---|---|
| SwinUNETR conditional 无效/变差 | 全局注意力可能已隐式处理左右 | 双向假设(§7.5),照实报 |
| 2D 逐 slice,机制不同于 3D | 2D tail 可能不是 mirror 导致 | axis audit(§6.1)+ 实验 |
| conditional 实现错误 | kidney 检测/翻转静默失效 | telemetry 硬验证(`protected_lr_mirror_count==0`,§5.5) |
| LR axis audit 做错 | 关错轴 → 实验全废 | audit 脚本 + witness + smoke 打印 + manifest 交叉验证 |
| 2D smoke 未通 | Conv2d 分支首次激活 | Stage 0 前置(§6.2) |
| ~~2D baseline trainer 不一致~~ | ~~PACA vs MultiNetworkTrainer~~ | **已核实等价**:`clip_grad_norm_(12)` 是原生 nnUNetTrainerV2 就有的(line 254/264),唯一差异 He-reinit guard 触发条件 `val Dice==0`(训练崩溃),正常不触发 → 旧 2D baseline 直接复用 |

### 9.2 定位(Q15)与边界(Q16)
- issue #1 主对比的**改进型 ablation**,不是新主对比;叙事:"在最强网络上,通过条件禁用 LR-mirror
  进一步修复 kidney tail / 左右混淆";
- **不**升级 champion;scope 严格 = SwinUNETR + nnU-Net 2D(Q5),**不泛化**到其它架构;
- handoff(Huawei v1 Task520)只作先验,不混比;MedNeXt-S `exp_r` 问题独立,**不碰**。

---

## 10. 落地 Stage(Implementation Stages)

### Stage 0:audit + 2D smoke(前置,必须 PASS)
- [ ] 写 `tools/audit_lr_axis.py`(新建 `tools/`),产 `confirmed_lr_axis_3d` / `_2d` + witness;
- [ ] 打印 baseline `mirror_axes` 坐实(SwinUNETR `(0,1,2)`、nnunet_2d `(0,1)`);
- [ ] 扩展 `framework/smoke_framework.py`(去 nnunet_2d skip + 加 `--network-dim 2d`),2D smoke 通过(build/DS/mirror/forward)。
- **依赖**:commit c585966。**输出**:axis 确认 + 2D smoke PASS。

### Stage 1:2D 纳入 framework
- [x] ✅ 已完成(commit c585966,`train.py` 放开 `--network-dim 2d`)。

### Stage 2:conditional 实现 + telemetry smoke
- [ ] 新建 `framework/transforms/conditional_mirror.py`(照搬 handoff 三组件);
- [ ] `train.py` 加 `--lr-mirror-mode` + `--conditional-mirror-axis`(+ 可选 `--lr-axis-audit-manifest`);
- [ ] `MultiNetworkTrainer` 接线(`lr_mirror_mode` 参数,conditional 调 install_conditional_mirror_patch);
- [ ] witness 链:preflight.json + setup_witness + augmentation_witness.json + runtime_witness_report.md;
- [ ] telemetry smoke:跑几 iter,确认 `protected_lr_mirror_count==0` 且 `remaining_original_mirror_transform_count==0`;
- [ ] determinism:同 seed 跑两次 state_dict_equal(可选但建议)。
- **依赖**:Stage 0。**输出**:conditional 实现 + telemetry/witness 验证 PASS。

### Stage 3:validator 审查(独立)
- [ ] 独立审查(可起 subagent):telemetry=0、axis 正确、forward 正常、baseline 不受影响(`--lr-mirror-mode full` 默认)、witness 链完整。
- **依赖**:Stage 2。**输出**:审查 PASS。

### Stage 4:训练 conditional ×6
- [ ] 扩展 `generate_train_jobs.py` 加 `--variant condlr` + `--lr-axis-manifest`,生成 6 个 conditional job;
- [ ] **loop dsub 提交 6 job**(§8.4),early-runtime 逐个检查(1-2min `djob <id>` 查 RUNNING);
- [ ] 每 job 完成后读 witness/telemetry 确认 `protected_lr_mirror_count==0`。
- **依赖**:Stage 3 PASS。**输出**:6 个 `__condlr` model_final_checkpoint.model。

### Stage 5:预测 test 39
- [ ] `framework/predict.py`(sliding window 0.5,TTA off,no PP)对 test 39 预测,6 组 `__condlr` predictions。
- [ ] **predict 无 conditional 接线(O1)**:`predict.py do_mirroring=False`,增强只在训练;直接用 `__condlr` ckpt,predict 链路透明。
- **依赖**:Stage 4。**输出**:6 组 test 39 nii.gz。

### Stage 6:评估 + 统计(单一 run_eval.py + baseline 评估重跑)
- [ ] **baseline 评估重跑(C-6)**:用 `run_eval.py` 重新评估 issue#1 baseline predictions(产 30 列,放 `evaluation/results/ablation_baseline_30col/`);
- [ ] conditional + baseline(30 列)各过 `run_eval.py` + `kidney_swap_eval.py`;
- [ ] `evaluation/run_paired_stats.py`(新建,读 30 列):Wilcoxon + Holm 8;
- [ ] `make_figures.py` 改读 run_eval 30 列(C-3,消除 results_locked 双轨);
- [ ] 汇总:per-class Δ、kidney tail 改善、显著性、效应量。
- **依赖**:Stage 5。**输出**:`condlr_vs_baseline_paired_stats.csv` + 最终结果表。

---

## 11. 决策溯源(41 项,详见 draft)

**设计层(6)**:Q3 2D 纳入 MultiNetworkTrainer / Q4 只 conditional / Q5 只 2 网络 / Q9 统计(Holm 8,HD95 检验)/
2D baseline 不重训(等价核实)/ Q2 3D+2D axis audit。

**实施层 B 类(11)**:B-1 `framework/transforms/conditional_mirror.py` / B-2 `--lr-mirror-mode` /
B-3 `__condlr` / B-4 `run_paired_stats.py` / B-5 `tools/audit_lr_axis.py` / B-6 模块符号 patch /
B-7 telemetry 对齐 handoff / B-8 扩展 smoke_framework / B-9 `generate_train_jobs.py --variant condlr` /
B-10 固定参数写死 + axis CLI / B-11 witness/record 链补全。

**审查 + 批次(§13,18 项)**:MUST M1-M4(install 时序 / DS 机制 / paired schema / 2D predict 路径)+
RECOMMEND R1-R6(telemetry 改名 / axis 注入 generator / build_cases_csv / 2D smoke 分支 / 2D axis 校验 / stats_helpers)+
OPTIONAL O1(predict 透明)/ O2(等价抽查,不做)+ 批次提交 6 点(loop dsub / 不设上限 / 混提 / 逐个检查 / 单独重提 / Stage 3 PASS)。

**统一 run_eval.py 连锁(§14,6 项)**:C-1 §C1 统一决策 / C-2 M3 schema 8→30 列 / C-3 make_figures 改读 run_eval(单一)/
C-4 build_cases_csv + evaluate_swine_ct 保留备用 / C-5 issue#1 锁旧 8 列历史 / C-6 baseline 评估重跑(30 列,schema 一致)。
