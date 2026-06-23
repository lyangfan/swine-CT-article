# Spec Draft — Kidney LR-mirror conditional ablation(2D nnUNet + SwinUNETR)

Status: §13(35 项)+ §14(C-1..C-6 统一 run_eval.py 连锁)全部闭合;**正式 spec 待同步**(§14 统一架构 + C-1..C-6 决策 → spec §7.2/§7.3/§7.4 + make_figures/CLAUDE.md)。

Date: 2026-06-23

关联:
- 主对比 spec:[`v1-input-consistency-spec.md`](v1-input-consistency-spec.md)(环境/路径/公平性,本文引用)
- 先验证据:[`SWCT06042040-HUAWEI-V1-KIDNEY-MIRROR-CLASSIFIER-HARDZERO-HANDOFF.md`](SWCT06042040-HUAWEI-V1-KIDNEY-MIRROR-CLASSIFIER-HARDZERO-HANDOFF.md)
- 实验跟踪:GitHub issue #1;2D 统一入口 commit `c585966`

---

## 0. 目标 + scope

在 issue #1 最强的两个网络(nnU-Net 2D、SwinUNETR)上,把 LR mirror 改为**条件禁用**
(conditional:仅含 kidney 的 sample 禁 LR flip)。验证能否改善 kidney tail(Dice P10 / HD95 P90)
与左右混淆(swap rate / LP-Dice gap),且对其它 class 无副作用。

- framework = `framework/`(Task601, 3-seed, locked evaluator);2D/3D 统一走 MultiNetworkTrainer(c585966);
- 网络 = nnU-Net 2D + SwinUNETR;不与 handoff(Task520)混比;不碰 head/testis gate。

## 1. 动机

issue #1(3-seed):SwinUNETR kidney Dice 最高(0.934/0.935)但 HD95 P90 26–97mm、52% case 混淆;
nnU-Net 2D HD95 P90 高达 116–120mm、40% 混淆。handoff 先验(关 LR mirror 把 v1 fold4 kidney HD95 P90
从 ~64/46 砸到 ~3/2.8)提示 LR mirror 是 kidney tail 风险因素,但能否迁移到 2D+SwinUNETR 是本实验
要回答的(正负都有效)。

## 2. 环境与路径(事实,引用 v1 spec §3/§4)

- `NNUNETV1_PYTHON` = `swine_ct_autonomous_discovery/envs/nnunetv1/bin/python`;
- setup `swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh`;
- `PYTHONPATH` = `swine_ct_autonomous_discovery/scripts/nnunetv1_compat`(确定性套件);
- 三个 nnUNet data-root 覆盖到 `swine-CT-article/data/nnunetv1` + `SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1`;
- module load: gcc9.3/cuda11.8/cudnn8.8.1/nccl2.16.5/openblas;`swct_nnunetv1_preflight` 必须通过;
- Task601 preprocessed:3D `nnUNetData_plans_v2.1_stage1/`、2D `nnUNetData_plans_v2.1_2D_stage0/`;
  split `splits_final.pkl`(`place_split.py`,2D/3D 共用)。

## 3. 已澄清事实

### Q1. baseline 开 LR mirror?(已确认:开了)
SwinUNETR 继承 nnUNetTrainerV2 3d_fullres 默认 `(0,1,2)`;2D 走 MultiNetworkTrainer 默认 `(0,1)`。
smoke 时 `print(self.data_aug_params["mirror_axes"])` 坐实。

### Q2. LR axis = 第几轴?(已定:3D+2D 都在 Stage 0 audit)
Task601 ≠ handoff Task520,必须重测。方法:取双肾都在且左右不对称的 case,沿各 axis `np.flip`,
看 class4/5 体素是否互换 → 互换轴 = LR;2D 需分辨 axial slice 内哪个轴是真 LR。
产物:`confirmed_lr_axis_3d` / `confirmed_lr_axis_2d` + witness md。

### B-5. ✅ audit 脚本放 `tools/audit_lr_axis.py`(新建 `tools/` 目录)
**决策**:新建 `tools/` 目录,放 `audit_lr_axis.py`。
- 理由:audit 是独立类别"实验工具"(非训练框架、非评估指标);未来 telemetry 验证/determinism smoke
  等集中 tools/;对齐 handoff 先例(`runs/swct06042040/tools/`);不污染 framework/evaluation。
- tools/ 定位:**正式实验工具**(实验复现需要、长期保留),非一次性试探脚本。

---

## 4. 设计决策(已定)

- **Q3**:2D 纳入 MultiNetworkTrainer(c585966);2D baseline 不重训(改前改后等价已核实:clip 原生就有)。
- **Q4**:只做 conditional(不做 full-disable,handoff A1 做过);kidney voxel≥1 禁该 sample LR flip;
  3D 按 patch / 2D 按 slice。
- **Q5**:只 SwinUNETR + nnU-Net 2D;论文不 claim 架构无关。
- **Q7**:与 baseline 逐项对齐(split/seeds/patch/增强/sampling/budget/loss/optimizer/ckpt/determinism/
  predict/eval/trainer 子类);trainer 子类已核实等价。

---

## 5. conditional 实现 —— 待决策(工程核心)

> conditional 不能用"改 mirror_axes 一行"实现 —— `get_moreDA_augmentation` 内部 `MirrorTransform`
> 对整 batch 统一翻转,不支持 per-sample 条件。需要自定义 transform。**具体怎么落,以下待定。**

### B-1. ✅ conditional 代码放 `framework/transforms/conditional_mirror.py`(新建)
**决策**:新建独立文件,照搬 handoff 三组件:`ConditionalMirrorTransform` 类 + `install_conditional_mirror_patch`
+ factory(见 B-6)。
- 理由:transform 是数据增强逻辑,独立于 trainer;不污染 base_trainer;集中自洽、好审查;最大化复用 handoff 已验证代码。
- 需新建 `framework/transforms/` 目录。

### B-2. ✅ train.py 加 `--lr-mirror-mode {full,conditional}`(default `full`)
**决策**:train.py 加 `--lr-mirror-mode` enum,透传给 MultiNetworkTrainer(`lr_mirror_mode=args.lr_mirror_mode`)。
- 理由:enum 可扩展(将来加 no_lr 只加 choice);与 trainer 内部 `lr_mirror_mode` 同名同义,透传直接;job 脚本显式自文档化。

### B-6. ✅ 模块符号 patch(`da_more.MirrorTransform = factory`,对齐 handoff)
**决策**:用 handoff 的 `install_conditional_mirror_patch` —— `da_more.MirrorTransform = factory`(替换 moreDA
模块的 MirrorTransform 符号指向),**不 fork get_moreDA**;factory 闭包注入参数返回 ConditionalMirrorTransform。
- 技术约束(机制保证,非决策):pipeline 位置 = MirrorTransform 原位(get_moreDA_augmentation 内部调用自动
  变成 ConditionalMirrorTransform);seg 访问 = `data_dict["seg"]`;DS 顺序 = 原始分辨率。
- per-sample 机制:batchgenerators transform 本就 per-sample;ConditionalMirrorTransform 读 sample seg →
  含 kidney 则该 sample LR **强制不翻**,其余 axis 正常随机;不含 kidney 则全套随机。handoff A2 telemetry
  证实(`protected_lr_mirror_count=0`)。
- 理由:对齐 handoff(三组件整套照搬);省代码(不复制 get_moreDA);模块符号级 patch 影响面可控(只 moreDA 模块内)。
- **修正记录**:原 B-6 决策 fork(我当初把 monkey-patch 一棍子打成"全局副作用",判断不准);审核对比
  handoff 源码后改为模块符号 patch。

### B-7. ✅ telemetry 对齐 handoff(transform 实例属性 + JSONL 流式 + witness)
**决策**:复用 handoff A2 的 `ConditionalMirrorTransform` 类(line 212-371,已含 telemetry),不新设计。
- 计数:transform **实例属性** `self._window`(per-instance;非全局类变量、非 trainer 属性;计数在 transform 里就近发生);
  含 `protected_sample_count` / `nonprotected_sample_count` / `protected_axis2_mirror_count`(**必须=0**) /
  `nonprotected_axis2_mirror_count` / `axis0/1/2_mirror_count`。
- 输出:`<record_dir>/conditional_mirror_components.jsonl`(**JSONL 流式**,每 log_interval append;
  训练中途可查、崩溃可追溯,优于单 json)。
- witness:`<record_dir>/augmentation_witness.json`(配置 + `remaining_original_mirror_transform_count==0`
  硬验证原 MirrorTransform 已被替换)。
- 提交前必读:`protected_axis2_mirror_count==0` 且 `remaining_original_mirror_transform_count==0`,才能进 Stage 4。
- 理由:handoff A2 已验证输出过该 telemetry;流式鲁棒;复用降低新设计风险。

### B-10. ✅ 参数传递:固定参数写死 + axis 走 CLI
**决策**:固定参数写死(trainer 常量 / install_conditional_mirror_patch 默认),axis 显式 CLI。
- 固定参数:`protected_class_ids=(4,5)` / `protected_min_voxels=1` / `log_interval=25` / `p_per_sample=1`
  (本实验固定,沿用 handoff 默认,不暴露冗余 CLI);`run_name`/`instance_index` 由 install_patch 自动生成。
- 动态参数(axis):train.py 加 `--conditional-mirror-axis <int>`,job 脚本传 audit(Q2)产的
  `confirmed_lr_axis_3d` / `confirmed_lr_axis_2d`。
- 可选交叉验证:`--lr-axis-audit-manifest`,train.py 启动时校验 CLI axis == manifest axis(防手抖;**涵盖审核遗漏 3**)。
- 理由:固定隐式 + 动态显式;比 CLI 全参数简洁,比全自动读 manifest 可查。(**涵盖审核遗漏 5**:telemetry 参数值已定。)

### B-11. ✅ witness/record 链补全(对齐 handoff)
**决策**:补 handoff 有、draft 缺的 3 个 witness/record,形成完整证据链。
- **setup witness**:setup_DA_params 里(conditional 模式)记录改前 `original mirror_axes` / `do_mirror`,
  并入 augmentation_witness(改前→改后配套);
- **runtime_witness_report.md**:把 Q7 对齐表的"不变项"(loss / network / sampler / optimizer /
  cudnn_deterministic 不变)落盘成 witness md;
- **preflight.json**:训练前自检 record(env / config / axis / protected 参数齐全)。
- 落点:归 B-1 的 `framework/transforms/conditional_mirror.py`(install_conditional_mirror_patch / trainer
  initialize 里写),照搬 handoff 模式。
- 完整 record 链:preflight → setup_witness → augmentation_witness → telemetry_jsonl → runtime_witness → env/done(对齐 handoff)。
- 理由:对齐 handoff 已验证机制;可追溯/审计/复现性最强;每个成本低(json/md + 几行,照搬 handoff)。

---

## 6. smoke + determinism

### B-8. ✅ 扩展 `framework/smoke_framework.py` 支持 2D
**决策**:扩展现有 smoke_framework,去 line 50-51 的 `nnunet_2d` skip(c585966 前遗留)+ 加 `--network-dim 2d` 透传 + 2D 断言。
- 2D 断言:2D Generic_UNet 构建(`conv_op=Conv2d`)、DS 路数(2D plans pool 层数,打印 `n_outputs`)、
  `mirror_axes==(0,1)`、forward shape `[2,10,H,W]`。
- 命令:`$NNUNETV1_PYTHON -m framework.smoke_framework --network nnunet_v1 --network-dim 2d --seed 20260520`。
- 理由:最小改动(去 skip + 透传);3D/2D 同入口一致;复用现有 build/forward 断言框架。

### Q13. determinism(已定,建议验证)
conditional 引入新随机逻辑,建议同 seed 跑两次比 state_dict_equal(handoff 做过)。

---

## 7. 评估与统计

### Q8. 指标(已定)
kidney:L/R mean Dice、Dice P10、mean HD95、HD95 P90、FP/GT、FN/GT;swap rate/LP-gap/混淆 case 比例
(`kidney_swap_eval.py`)。副作用:全 9 class Δ、条件类 absent-FP、盯 front。

### Q9. 统计口径(已定)
per-case 3-seed 平均 Wilcoxon;Holm 跨 2 网络×2 kidney×2 指标=8;HD95 纳入检验;P90/swap 等描述性。

### 7.3 baseline 数据路径(事实,配对端)
- SwinUNETR baseline:`v1_comparison/swinunetr__seed<seed>/fold_0/model_final_checkpoint.model` +
  `v1_comparison_predictions/swinunetr__seed<seed>/`;
- 2D baseline(PACA 版,复用):`v1_comparison_2d_root/seed<seed>/runs/nnunet_2d__seed<seed>/` +
  `v1_comparison_predictions/nnunet_2d__seed<seed>/`。

### 7.4 评估命令链(事实,既有工具)
`evaluation/build_cases_csv.py` → `swine_ct_autonomous_discovery/metrics/evaluate_swine_ct.py --cases-csv`
→ `evaluation/kidney_swap_eval.py`。conditional 与 baseline 各跑一遍产出 per_case.csv 供配对。

### B-4. ✅ 新建 `evaluation/run_paired_stats.py`
**决策**:新建独立脚本做 baseline vs conditional 网络内配对统计。
- 输入:baseline + conditional 的 per_case.csv(每网络);per-case kidney Dice/HD95 先 3-seed 平均 →
  Wilcoxon signed-rank;Holm 跨 2 网络×2 kidney×2 指标=8。
- 输出:`evaluation/results_locked/condlr_vs_baseline_paired_stats.csv`(p / Holm-adj / 效应量)。
- 理由:网络间 vs 网络内配对逻辑差异大;不碰 issue #1 的 run_stats.py(还在用,MedNeXt-L 重训中);
  底层 Wilcoxon/Holm 抽成共用 helper(与 run_stats.py 共享,避免重复)。

---

## 8. 工程与调度

### Q11. 训练规模(已定)
conditional:SwinUNETR ×3 + nnunet_2d ×3 = 6 job(baseline 全复用不重训);节点 agent<170、非 174;
单卡并发;early-runtime 检查。

### B-3. ✅ 产物命名后缀 `__condlr`
**决策**:conditional 产物统一用 `__condlr` 后缀。
- ckpt `data/nnunetv1/v1_comparison/{swinunetr,nnunet_2d}__condlr_seed<seed>/fold_0/model_final_checkpoint.model`;
- pred `data/nnunetv1/v1_comparison_predictions/{swinunetr,nnunet_2d}__condlr_seed<seed>/*.nii.gz`;
- eval `evaluation/results_locked/{swinunetr,nnunet_2d}_condlr/`;
- 理由:短且自解释(conditional LR);和 full-disable 的 `__nolr` 对称不混;和 baseline `<net>__seed<s>` 协调。

### Q13. determinism / 复现(见 §6)

### B-9. ✅ 扩展 `generate_train_jobs.py` 加 `--variant condlr`
**决策**:扩展现有 generator,加 `--variant {baseline,condlr}` 开关(默认 baseline,不破坏现有)。
- 3D conditional:`_3d_script` 模板 + `--lr-mirror-mode conditional` + `__condlr` 命名;
- 2D conditional:**新模板** `framework/train.py --network nnunet_v1 --network-dim 2d --lr-mirror-mode conditional` + `__condlr`
  (不走旧 `train_paca_deterministic.py`,因要触发 `--lr-mirror-mode`);
- baseline 2D 保持旧 `train_paca_deterministic.py`(复用不重训,历史产物;trainer 等价已核实 Q3);
- predict job:复用 `generate_predict_jobs.py`,只换 ckpt 路径指向 `__condlr`。
- 理由:复用模板/DSUB header/env;一个 generator 统管;不重复。

---

## 9. 科学风险与定位(已定)

- Q14 风险:conditional 实现错误(靠 telemetry 管控)/ axis audit 错 / 2D smoke 未通 / SwinUNETR 无效 / 2D 机制不同;
  (2D baseline trainer 不一致风险已关闭)。
- Q15:改进型 ablation,scope 严格 2 网络,不泛化,不升级 champion。
- Q16:handoff 只作先验;MedNeXt 不碰。

---

## 10. Stage checklist(B 类待决策项已标注)

| Stage | 任务 | 依赖 | 状态 |
|---|---|---|---|
| 0 | axis audit(B-5 脚本位置待定);打印 baseline mirror_axes;2D smoke(B-8 入口待定) | c585966 | 待做 |
| 1 | ✅ 2D 纳入 framework(c585966) | — | 完成 |
| 2 | conditional 实现(B-1 落点/B-2 CLI/B-6 方法/B-7 telemetry 均待定)+ telemetry smoke + determinism | Stage 0 | 待做 |
| 3 | validator 审查(telemetry=0、axis 正确、forward 正常、baseline 不受影响) | Stage 2 | 待做 |
| 4 | 训练 conditional(命名 B-3 待定)×6 | Stage 3 PASS | 待做 |
| 5 | 预测 test 39(`framework/predict.py`) | Stage 4 | 待做 |
| 6 | eval + kidney_swap + 统计(B-4 脚本方式待定) | Stage 5 | 待做 |

---

## 11. 设计决策溯源(6 项,已定)

Q3 2D 纳入 framework / Q4 只 conditional / Q5 只 2 网络 / Q9 统计口径(Holm 8,HD95 检验)/
2D baseline 不重训 / Q2 3D+2D audit。

## 12. B 类决策清单(11 项,均已拍板)

| # | 问题 | 我的建议 |
|---|---|---|
| ~~B-1~~ | conditional 代码放哪 | ✅ 新建 `framework/transforms/conditional_mirror.py` |
| ~~B-2~~ | train.py CLI | ✅ `--lr-mirror-mode {full,conditional}`(default full) |
| ~~B-3~~ | 产物命名后缀 | ✅ `__condlr` |
| ~~B-4~~ | 统计脚本 | ✅ 新建 `evaluation/run_paired_stats.py` |
| ~~B-5~~ | audit 脚本放哪 | ✅ 新建 `tools/audit_lr_axis.py` |
| ~~B-6~~ | 实现方法 | ✅ 模块符号 patch `da_more.MirrorTransform = factory`(对齐 handoff,照搬 install_conditional_mirror_patch) |
| ~~B-7~~ | telemetry 设计 | ✅ 对齐 handoff:transform 实例属性 + JSONL 流式 + augmentation_witness.json |
| ~~B-8~~ | 2D smoke 入口 | ✅ 扩展 `framework/smoke_framework.py`(去 skip + 加 `--network-dim 2d`) |
| ~~B-9~~ | job 生成 | ✅ 扩展 `generate_train_jobs.py` 加 `--variant condlr` |
| ~~B-10~~ | 参数传递 | ✅ 固定参数 trainer 常量 + axis CLI `--conditional-mirror-axis` |
| ~~B-11~~ | witness/record 链 | ✅ 补 setup witness + runtime_witness + preflight(对齐 handoff) |

> 逐个拍板,拍一个写回一个。全部定完后即可收敛成正式 spec 并执行。

---

## 13. 审查 + 批次提交:待决策问题(subagent 审查 + 用户追加)

> 来源:独立 subagent 审查 [`v1-kidney-lr-mirror-ablation-spec-REVIEW.md`](v1-kidney-lr-mirror-ablation-spec-REVIEW.md) + 用户追加"批次提交作业"。这些是 spec/draft 之前没覆盖的,**待用户逐个拍板**。

### 13.1 🔴 MUST(✅ 均已拍板)

| # | 问题 | 决策 |
|---|---|---|
| ~~M1~~ | install_conditional_mirror_patch 时序 | ✅ **B:放 `MultiNetworkTrainer.initialize()` 内** `setup_DA_params()`(base_trainer.py:197)后、`get_moreDA_augmentation()`(:230)前;晚于 train.py:77 determinism patch。注:偏离 handoff 脚本级位置(handoff 在 run_training L495),但满足同样时序约束(get_moreDA 前 + determinism 后)。 |
| ~~M2~~ | §5.2 技术推理错(我写错) | ✅ 按审查改:DS 下采样由 moreDA pipeline `DownsampleSegForDSTransform`(`:145`)做,MirrorTransform(`:111`)跑时 seg 仍 full-res、key 仍 `seg`(结论对、机制改对)。 |
| ~~M3~~ | run_paired_stats 输入 schema/配对 | ✅ join key `(case_id,class_label)`、两臂标记 `nnunet_2d_condlr`/`swinunetr_condlr`、class∈{4,5}、HD95 NaN **drop pair(继续用 case 其它 class)**、效应量 **r=\|z\|/√N + mean Δ 都报**。 |
| ~~M4~~ | 2D condlr predict 路径 | ✅ `generate_predict_jobs.py` 加 `--variant condlr`;2D condlr predict 新模板,CKPT 指 `v1_comparison/nnunet_2d__condlr_seed<seed>/`,走 `framework.predict --network-dim 2d`,eval 标签 `nnunet_2d_condlr`。 |

### 13.2 🟡 RECOMMEND(打磨)

| # | 问题 | 审查建议 | 待拍板 |
|---|---|---|---|
| ~~R1~~ | telemetry 字段名对 2D 误导 | ✅ **改名 `protected_lr_mirror_count` / `nonprotected_lr_mirror_count`**(逻辑照搬 handoff,字段名泛化;spec §5.5/B-7 断言名同步改;2D 不误导) |
| ~~R2~~ | audit axis 注入 generator | ✅ audit 产 manifest,generator `--variant condlr` + `--lr-axis-manifest <path>` 读 manifest bake 进 `--conditional-mirror-axis`(与 train.py `--lr-axis-audit-manifest` 同源,单一真相) |
| ~~R3~~ | build_cases_csv 不认 `__condlr` | ✅ 加 `--variant condlr`,pred_dir 指向 `__condlr` 目录,`method` 用 M3 标签(`nnunet_2d_condlr`/`swinunetr_condlr`) |
| ~~R4~~ | 2D smoke 改动量低估 | ✅ §6.2 改述:新增 `--network-dim` 分支(dim=2d 走单网络路径 `get_default_configuration("2d")`+2D patch+`[2,1,H,W]` 输入,**不复用 all-networks 循环**;删 L50-51 死代码 skip;2D 断言不变 |
| ~~R5~~ | 2D axis 取值约束(假 PASS 风险) | ✅ `confirmed_lr_axis_2d∈{0,1}`;**audit + train.py 双重校验** reject axis=2(for 2D);防 telemetry 假 PASS(axis 错时保护永不触发但计数==0) |
| ~~R6~~ | 共用 helper 重构 | ✅ 新建 `evaluation/stats_helpers.py`,把 `holm_bonferroni`(+wilcoxon wrapper)移过去,`run_stats.py` + `run_paired_stats.py` 都 import;refactor run_stats 只移动不改逻辑(refactor 前后 smoke 确认输出一致) |

### 13.3 🟢 OPTIONAL

| # | 问题 | 审查建议 | 待选 |
|---|---|---|---|
| ~~O1~~ | predict 无需 conditional 接线 | ✅ 补声明(§5/Stage 5):predict 链路透明(`do_mirroring=False`),直接用 `__condlr` ckpt |
| ~~O2~~ | 等价经验证 | ✅ **不做**(等价已源码核实:clip 原生 254/264 + He-reinit guard 仅 Dice=0 触发;抽查要额外重训 1 seed 且只验证 1 seed;baseline 复用不依赖经验证) |

### 13.4 批次提交作业(用户追加)

6 个 conditional job(SwinUNETR×3 + nnunet_2d×3)的批次提交策略:

| 决策点 | 决策 |
|---|---|
| ~~提交方式~~ | ✅ **(b) 简单 loop dsub**(不碰 orchestrator,避免影响 issue #1 复现);`generate_train_jobs --variant condlr` 生成 6 脚本后 loop dsub |
| ~~并发上限~~ | ✅ 不设硬上限,一次提 6 个,调度器按空闲 GPU 分配(不够的 pending) |
| ~~分批顺序~~ | ✅ 混提(6 job 无依赖,一次性);3 seed 并发 |
| ~~early-runtime 检查~~ | ✅ 逐个(每 job 提交后 1-2min `djob <id>` 查 RUNNING) |
| ~~失败重提~~ | ✅ 单独(读 .err 修后单独 dsub,不动其它) |
| ~~Stage 依赖~~ | ✅ Stage 3 validator PASS 才提 Stage 4 |

---

## 14. 统一 run_eval.py 后的未收口连锁(待逐个决策)

> 背景:§C1 讨论后,实际选了"**统一 run_eval.py**"(把 evaluate_swine_ct.py 的全指标移植进 run_eval.py + HD95 verbatim + `seed` 列,代码已改 + smoke 通过:30 列、352 行、Dice 逐位对齐 issue#1)。这把评估架构从"两套"变成"单一",但 spec/draft 还停留在两套叙事,以下连锁待收口。**逐个拍完,再同步正式 spec。**

### C-1. §C1 决策(统一)写进 draft/spec
- 背景:draft §13 没 §C1;spec §7 还"两套(run_eval + evaluate_swine_ct)"叙事。
- 待:draft 补 §C1 决策记录;spec §7.2/§7.4 改成"单一 run_eval.py"。([无选项,纯同步])

### C-2. M3 schema 更新(8 → 30 列)
- 背景:run_eval.py 现产 **30 列**(network/seed/case_id/source + confusion 全指标)。M3(draft §13.1 + spec §7.2)还写 8 列。
- 待:M3 schema 更新成 30 列;run_paired_stats 读 30 列。([无选项,纯同步])

### C-3. ✅ make_figures 改读 run_eval.py(单一数据源)
**决策**:(a) make_figures 改读 `results/per_case.csv`(run_eval 30 列),消除 `results_locked/` 双轨。
- 理由:延续统一目标;30 列含全指标 + 大写列名兼容 34 列,改动小(路径 + glob);消除双数据源。

### C-4. ✅ build_cases_csv / evaluate_swine_ct 保留备用
**决策**:(a) 保留(不默认调,作 canonical 对照 / 交叉验证)。
- 理由:canonical locked evaluator 是项目资产,run_eval 是其 verbatim port;保留作对照基准(smoke 已用 issue#1 Dice 验证);保留不害(默认走 run_eval,evaluate_swine_ct 不默认调)。

### C-5. ✅ issue #1 锁旧 8 列历史不动
**决策**:(b) issue #1 的 run_stats / 8 列 per_case.csv 锁历史;ablation 用新 run_eval 30 列(run_paired_stats 独立读)。
- 理由:issue #1 已定稿 + MedNeXt-L 重训中,重构 run_stats 有风险;run_paired_stats 是新脚本独立 30 列,不冲突;数值口径一致(算法同),schema 分裂可接受。
- 注:MedNeXt-L 重训后 issue #1 若重评估 30 列,是 issue#1 后续(非本 ablation)。

### C-6. ✅ baseline 评估重跑(30 列)
**决策**:(a) 用新 run_eval.py 重新评估 issue#1 的 baseline predictions(产 30 列),与 conditional schema 一致。
- 理由:schema 一致是 pair stats 配对前提(旧 8 列断裂);重评估 ≠ 重训(Q3 模型不重训,只重评估);baseline 获全指标(报表统一,配合 C-3);成本低(6 predictions × run_eval,分钟级)。
- 注:旧 8 列 per_case.csv 保留(C-5 锁历史);新 30 列 baseline 评估是 ablation 专用。

> 逐个拍,拍完同步正式 spec(§7.2/§7.3/§7.4 + make_figures/CLAUDE.md)。
