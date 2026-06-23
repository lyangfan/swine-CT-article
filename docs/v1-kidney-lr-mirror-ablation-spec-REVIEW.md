# 独立审查报告 — v1 Kidney LR-mirror conditional ablation 正式 Spec

Status: **READ-ONLY REVIEW**(未修改 spec / draft / 任何代码)
Reviewer: 独立 subagent(全新 session,零父上下文)
Date: 2026-06-23
被审:`docs/v1-kidney-lr-mirror-ablation-spec.md`(下称 **spec**)
基准:`docs/v1-kidney-lr-mirror-ablation-spec-draft.md`(下称 **draft**)
交叉参考:`docs/v1-input-consistency-spec.md`、`docs/SWCT06042040-…-HANDOFF.md`、`framework/`、`jobs/`、`evaluation/`、Huawei handoff 源码 + nnUNet v1 源码。

> 审查方法:逐条核对 17 项决策(设计 6 + B 类 11);对每个"落地需要"的工程钩子,读本地代码 + ssh
> `paca_share` 核对 handoff 源码行号、nnUNet 源码、determinism 模块、预处理目录。所有结论附 `file:line` 证据。

---

## 1. 与 draft 冲突(逐项)

**结论:无冲突。** 17 项决策在 spec 中全部存在且与 draft 口径一致。证据如下。

### 设计层 6 项

| 决策 | draft 位置 | spec 位置 | 一致性 |
|---|---|---|---|
| Q3 2D 纳入 MultiNetworkTrainer | draft L20, L61 | spec §1(L23)、§2(L25)、§4(L84) | ✅ |
| Q4 只 conditional(kidney voxel≥1) | draft L62-63 | spec §1(L17-20)、§5.2(L111-114)、§9.1(L243) | ✅(见 §2 微差) |
| Q5 只 2 网络 | draft L64 | spec §1(L25)、§2、§9.2(L251) | ✅ |
| Q9 统计(Holm 8,HD95 检验) | draft L153 | spec §7.2(L191-192) | ✅ "2 网络×2 kidney×2 指标=8"口径完全一致 |
| 2D baseline 不重训(等价核实) | draft L61, L195 | spec §2(L36)、§8.1(L216)、§9.1(L246) | ✅ |
| Q2 3D+2D axis audit | draft L46-49 | spec §6.1(L154-160) | ✅ |

### 实施层 B 类 11 项

| 决策 | draft 位置 | spec 位置 | 一致性 |
|---|---|---|---|
| B-1 `framework/transforms/conditional_mirror.py` | draft L75-79 | spec §5.1(L97-101) | ✅ |
| B-2 `--lr-mirror-mode {full,conditional}` | draft L81-83 | spec §5.3(L116-128) | ✅ default `full` 透传 MultiNetworkTrainer |
| B-3 `__condlr` 后缀 | draft L181-186 | spec §8.2(L220-224) | ✅ |
| B-4 `evaluation/run_paired_stats.py` | draft L165-171 | spec §7.2(L188-194) | ✅(见 §3-M3 输入 schema 缺) |
| B-5 `tools/audit_lr_axis.py` | draft L51-55 | spec §6.1(L155) | ✅ |
| B-6 模块符号 patch `da_more.MirrorTransform = factory` | draft L85-95 | spec §5.2(L103-114) | ✅ |
| B-7 telemetry 对齐 handoff | draft L97-107 | spec §5.5(L136-148) | ✅(见 §3-R1 字段名) |
| B-8 扩展 smoke_framework 2D | draft L134-139 | spec §6.2(L162-172) | ✅(见 §3-R4 改动量) |
| B-9 `generate_train_jobs.py --variant condlr` | draft L190-197 | spec §8.3(L226-232) | ✅(见 §3-R2 axis 注入) |
| B-10 固定参数写死 + axis CLI | draft L109-116 | spec §5.4(L130-134) | ✅ 参数集逐项一致 |
| B-11 witness/record 链 | draft L118-128 | spec §5.5(L141-148) | ✅ preflight→setup→augmentation→runtime 链一致 |

### 行号引用核对(ssh 实测,全部准确)

- spec §5.1「handoff line 212-371 ConditionalMirrorTransform」→ 实测 `class ConditionalMirrorTransform:` 在 **line 212** ✅
- spec §5.1「handoff line 375-435 install_conditional_mirror_patch」→ 实测 `def install_conditional_mirror_patch(` 在 **line 375** ✅;`da_more.MirrorTransform = factory` 在 **line 428** ✅
- spec §9.1「clip_grad_norm_(12) 原生 nnUNetTrainerV2 line 254/264」→ 实测 nnUNetTrainerV2.py **line 254、264** ✅

---

## 2. 不完整(draft 有、spec 没有)

仅 2 处**轻微**内容下沉缺失,均不影响落地:

1. **draft §1 动机数据未进 spec。** draft L25-28 给了 issue #1 的量化动机(SwinUNETR kidney HD95 P90 26–97mm、52% case 混淆;nnU-Net 2D HD95 P90 116–120mm、40% 混淆)。spec §1 只保留定性描述。→ 建议:把这几组数搬进 spec §1,作为"为什么做这个 ablation"的证据锚点(论文方法章节用得上)。

2. **draft Q4「3D 按 patch / 2D 按 slice」的粒度措辞未在 spec 体现。** draft L63 明确"3D 按 patch / 2D 按 slice";spec §5.2(L111)只说"per-sample"。语义上 2D 的 sample 即 slice,二者等价,**不构成冲突**;但 spec 没把"2D 保护是逐 slice 的"写明,审稿人/读者可能疑惑 2D 粒度。→ 建议:§5.2 补一句"2D 下 sample 即 axial slice,保护逐 slice 生效"。

---

## 3. 落地遗漏(分级)

### 🔴 必须补(MUST,落地前必须解决)

#### M1. install_conditional_mirror_patch 的调用时序未钉死 + 与 determinism patch 的交互未写

**事实(实测):**
- `framework/train.py:77` 在构造 trainer **之前**就调 `install_v1_determinism_patches(...)`;该函数(`nnunetv1_compat/determinism.py:135-218`)**包裹 `da_more.get_moreDA_augmentation`** —— 在 line 143 捕获 `original_get_more_da = da_more.get_moreDA_augmentation`(真函数),在 line 217 把 `da_more.get_moreDA_augmentation` 替换成 `seeded_get_moreDA_augmentation`(内部调 `original_get_more_da`,** late-bind** `da_more.MirrorTransform`)。
- `install_conditional_mirror_patch`(handoff L375-435)只做一件事:`da_more.MirrorTransform = factory`(handoff L428)。
- 二者**改的是不同符号**(一个改 `get_moreDA_augmentation` 函数,一个改 `MirrorTransform` 类),**能组合**,但**强顺序敏感**:
  1. 必须先 `install_v1_determinism_patches`(让它捕获到**真** `get_moreDA_augmentation`);
  2. 再 `install_conditional_mirror_patch`(rebind `MirrorTransform`);
  3. 两者都必须在 `get_moreDA_augmentation` **被调用之前**完成 —— 而该调用发生在 `MultiNetworkTrainer.initialize()` 内(`framework/base_trainer.py:230`)。
- handoff 的正确范本(`train_conditional_lr_mirror_v1_reviewed.py` 的 `run_training`)正是这个顺序:determinism patch(L494)→ mirror patch(L495)→ 构造 trainer → initialize。

**spec 的问题:** §5.5(L146)只写「install_conditional_mirror_patch / trainer initialize 里写」,§5.3(L128)写「MultiNetworkTrainer.__init__ 加 lr_mirror_mode 参数,透传」,Stage 2(L270)写「MultiNetworkTrainer 接线(… conditional 调 install_conditional_mirror_patch)」—— **都没钉死**:
- (a) 若放在 `MultiNetworkTrainer.initialize()` 里,必须在 `setup_DA_params()`(`base_trainer.py:197`)**之后**、`get_moreDA_augmentation()`(`base_trainer.py:230`)**之前**;
- (b) 必须晚于 train.py:77 的 determinism patch;
- (c) **若放在 get_moreDA 调用之后,patch 静默失效**(无报错,增强根本没变成 conditional —— 正是 telemetry 要防的那种静默失败)。

**建议:** spec §5 增一节「5.x install 时序契约」,明确:
> `install_v1_determinism_patches`(train.py:77)→ 构造 MultiNetworkTrainer → `initialize()` 内 `setup_DA_params()`(base_trainer.py:197)之后、`get_moreDA_augmentation()`(base_trainer.py:230)之前调 `install_conditional_mirror_patch`。顺序错则 conditional 静默失效(telemetry 的 `protected_axis2_mirror_count==0` 断言会兜住,但应在 Stage 2 telemetry smoke 就抓到,不要拖到 Stage 4)。

**严重度:MED-HIGH**(静默失败风险;但 telemetry witness 会兜底,所以不是"实验做错还不知道",只是"实现者容易踩坑")。

---

#### M2. spec §5.2 的技术推理事实错误(MultipleOutputLoss2 不负责下采样)

**spec 原文(L114):** 「DS 顺序:augmentation 阶段 seg 是原始分辨率(**MultipleOutputLoss2 在 loss 阶段才下采样**),判 kidney 用原始 patch label。」

**事实(实测 moreDA 源码 `data_augmentation_moreDA.py`):**
- L111:`tr_transforms.append(MirrorTransform(params.get("mirror_axes")))` ← MirrorTransform 入队
- L140:`RenameTransform('seg', 'target', True)` ← seg 改名 target
- L145-150:`if deep_supervision_scales is not None: DownsampleSegForDSTransform3/2(...)` ← **DS 下采样由 dataloader pipeline 里的 `DownsampleSegForDSTransform` 做,不是 MultipleOutputLoss2**

**结论:** spec 的**结论正确**(kidney 判定用的是原始分辨率 label —— 因为 MirrorTransform 在 L111、下采样在 L145+,MirrorTransform 跑时 seg 仍是 full-res 且 key 仍叫 `seg`,handoff factory 的 `label_key="seg"` 对得上),但**给出的机制是错的**:下采样发生在 augmentation pipeline(`DownsampleSegForDSTransform`),`MultipleOutputLoss2` 只是拿已下采样的 target 列表按权重算 loss,**它不下采样**。

**风险:** 实现者若信了"MultipleOutputLoss2 在 loss 阶段才下采样",可能误以为 dataloader 不下采样,进而在重构时(如移动 MirrorTransform 位置、或以为可去掉 `DownsampleSegForDSTransform`)把 MirrorTransform 挪到下采样之后 → kidney 判定拿到的是低分辨率 / 已重命名的 target,**静默破坏保护逻辑**。

**建议:** 把 L114 改成准确机制:
> DS 下采样由 moreDA pipeline 的 `DownsampleSegForDSTransform` 完成(`data_augmentation_moreDA.py:145-150`),位置在 `MirrorTransform`(:111)与 `RenameTransform('seg'→'target')`(:140)**之后**。因此 MirrorTransform 执行时 `data_dict["seg"]` 仍是原始分辨率、key 仍为 `seg`,ConditionalMirrorTransform 在此点判 kidney(class 4/5 voxel≥1)用的是 full-res patch label。

**严重度:MED**(结论对、机制错;误导重构的风险)。

---

#### M3. `run_paired_stats.py` 的输入 schema / 配对方式未写

**spec §7.2(L189):** 「输入:baseline + conditional 的 per_case.csv(每网络)」。未给列名、join key、取哪些 class、NaN 怎么处理。

**实测真实 schema**(`evaluation/run_eval.py:176`):`network, seed, case_id, source, class_label, class_name, dice, hd95`。该 CSV 由 locked evaluator 风格的 `run_eval.py` append 产出(每网络每 seed 一份,或合并)。

**必须钉死的点(spec 没写):**
- **两臂如何标记**:`network` 列怎么区分 baseline vs conditional?(baseline 2D 是 `nnunet_2d`、SwinUNETR 是 `swinunetr`;conditional 建议显式 `nnunet_2d_condlr` / `swinunetr_condlr`,需 `run_eval.py --network` 传这个标签 —— 当前 `run_eval.py:142` 的 `--network` 是自由字符串,OK,但 spec 要点名约定)。
- **join key**:`(case_id, class_label)`,先 3-seed 平均(dice & hd95 各自平均)。
- **取哪些 class**:只 `class_label ∈ {4,5}`(left/right kidney),其余 class 只做描述性 Δ(§7.1 已说,但 §7.2 的检验对象要写明 4/5)。
- **HD95 的 NaN**:`run_eval.py:193` 对空 pred/GT 写 `"nan"`。Wilcoxon 不能吃 NaN;`run_stats.py:57` 用 `_to_float→nan→skip`。spec 要写明 paired 检验里 HD95 NaN case 的策略(drop pair?还是该 case 整条剔?)——kidney 几乎不空,影响小,但**要写**。
- **输出列**(spec §7.2 L194 说了 `p / Holm-adj / 效应量`):效应量定义未给(Wilcoxon 的 r = |z|/√N?还是 mean Δ?)。建议补。

**建议:** spec §7.2 增「输入/配对契约」小节,把上述 5 点写死。

**严重度:MED**(不写则 `run_paired_stats.py` 的实现者要自己 reverse-engineer run_eval 输出 + 猜 join 口径,容易和 issue #1 的 `run_stats.py` 口径不一致)。

---

#### M4. 2D conditional 的 predict job 路径与现有 generator 不符

**spec §8.2(L222):** 2D condlr ckpt 落 `data/nnunetv1/v1_comparison/nnunet_2d__condlr_seed<seed>/fold_0/…`(注意根是 `v1_comparison/`,**不是** 2D baseline 的 `v1_comparison_2d_root/`)。

**实测现有 generator(`jobs/predict/generate_predict_jobs.py:87`):** `_2d_predict` 把 CKPT 硬编码到 `v1_comparison_2d_root/seed{seed}/runs/nnunet_2d__seed{seed}/fold_0/model_final_checkpoint.model`(baseline PACA 路径)。且 generator **没有 `--variant` / `--suffix` 开关**(`generate_predict_jobs.py:108-113` 只有 `--network/--seed/--no-2d`)。

**spec §8.3(L232):** 「predict job:复用 `generate_predict_jobs.py`,只换 ckpt 路径指向 `__condlr`」—— 对 **3D** 成立(3D condlr 与 baseline 同在 `v1_comparison/`,只加 `__condlr` 后缀);对 **2D 不成立**:2D condlr ckpt 在 `v1_comparison/nnunet_2d__condlr_seed<seed>/`,2D baseline 在 `v1_comparison_2d_root/…`,**根目录都不同**,不是"只换后缀"。

**建议:** spec §8.3 明确:为 condlr 给 `generate_predict_jobs.py` 加 `--variant condlr`;2D condlr predict 用新模板(或参数化 `_2d_predict`),CKPT 指向 `v1_comparison/nnunet_2d__condlr_seed<seed>/fold_0/model_final_checkpoint.model`,仍走 `framework.predict --network nnunet_v1 --network-dim 2d`(predict.py:43 已支持 `--network-dim 2d`,无需改),eval 标签用 `nnunet_2d_condlr`。

**严重度:MED**(不补则 2D condlr 的 predict job 生成会指错路径,job 启动即 FileNotFoundError)。

---

### 🟡 建议补(RECOMMEND)

#### R1. telemetry 字段名 `protected_axis2_mirror_count` 对 2D 有误导(硬编码 "axis2")

**实测(`ConditionalMirrorTransform._record_axis_mirror`, handoff L294-305):** 计数**逻辑**是参数化的(`if int(axis) == int(self.conditional_mirror_axis)`),但**字段名**硬编码成 `protected_axis2_mirror_count` / `nonprotected_axis2_mirror_count`(handoff L300-302)。

**影响:** 对 3D(LR=axis 2)名实相符;对 **2D**(LR axis ∈ {0,1})**逻辑仍正确** —— `protected_axis2_mirror_count==0` 的断言语义上仍=「conditional 轴在 protected sample 上没被翻」,断言**照样成立**;但字段名里的 "axis2" 对 2D 是**误导**(实际指的是 conditional axis,不是字面的 axis 2)。

**建议:** spec §5.5 / B-7 注明:若"整套照搬"handoff,需在文档里写清"`protected_axis2_mirror_count` 语义=『conditional 轴被翻次数』,名字里的 axis2 是 handoff 3D 遗留命名」;或在移植到 `framework/transforms/conditional_mirror.py` 时把字段重命名为 `protected_conditional_axis_mirror_count`(顺手把 spec §5.5 的断言名同步改)。否则 2D 审稿人看到 "axis2==0" 会困惑。

---

#### R2. audit 的 axis 值如何注入 `generate_train_jobs.py`(axis 链路断点)

**事实:** condlr job 脚本要 bake 字面 axis int 进 `--conditional-mirror-axis <v>`。但该值来自 **Stage 0 audit**(`confirmed_lr_axis_3d` / `_2d`),audit 跑在 generator **之后**(Stage 0 → Stage 4)。

**spec 缺口:** §8.3 / Stage 4 没说 generator **怎么拿到** axis 值(hardcode audit 后写死?读 manifest?CLI 传?)。`--lr-axis-audit-manifest`(§5.3 L124)是 **train.py** 启动时的交叉校验,**不是** generator 的输入。

**建议:** spec §8.3 明确 axis 注入方式 —— 推荐:audit 产出 manifest(含 `confirmed_lr_axis_3d`/`_2d`),generator `--variant condlr` 时 `--lr-axis-manifest <path>` 读 manifest bake 进脚本(与 train.py 的 `--lr-axis-audit-manifest` 同源,单一真相)。

---

#### R3. `build_cases_csv.py` 不认 `__condlr` 后缀

**实测(`evaluation/build_cases_csv.py:41`):** `pred_dir = …/v1_comparison_predictions/{network}__seed{seed}` 硬编码,**无 `--variant`/`--suffix`**。condlr 的 pred 目录是 `{network}__condlr_seed{seed}`(§8.2 L223),且 `method` 列(L63)要区分 baseline/conditional。

**建议:** spec §7.4 注明 `build_cases_csv.py` 需加 `--variant`/`--suffix` 以指向 `__condlr` 目录并打对 method 标签(否则 cases_csv 指向不存在的 pred 目录,locked evaluator 空跑)。

---

#### R4. 2D smoke 的改动量被低估("去 skip"是死代码)

**实测(`framework/smoke_framework.py`):**
- L50-51:`if name == "nnunet_2d": continue` —— 但 registry 里**根本没有 `nnunet_2d` 这个条目**(`framework/registry.py` 只有 `nnunet_v1`,见 `framework/nets/nnunet.py:49`;2D = `nnunet_v1` + `--network-dim 2d`)。**这行 skip 是死代码**,去掉它什么都不启用。
- L38:`get_default_configuration("3d_fullres", …)` 硬编码 3D;
- L44:`patch = (64,160,160)` 硬编码 3D patch;
- L45:`x = torch.randn(2,1,*patch)` 硬编码 3D 输入;
- L49:`for name, spec in sorted(all_networks().items())` —— 遍历**所有**网络。

**spec §6.2(L163-168)** 说「去 line 50-51 的 skip + 加 `--network-dim 2d` 透传 + 2D 断言」。**低估了**:2D smoke 不是"在 all-networks 循环里多跑一个",而是**一条独立的单网络路径**(`--network nnunet_v1 --network-dim 2d`,见 §6.2 命令 L171),需要 `get_default_configuration`/patch/输入 shape 全部按 dim 分支。spec 列的 2D 断言(conv_op=Conv2d / DS 路数 / mirror_axes==(0,1) / forward [2,10,H,W])是对的,但"去 skip + 透传"的措辞让人以为改两行就行。

**建议:** spec §6.2 改述为「新增 `--network-dim` 分支:dim=2d 时走单网络路径(`get_default_configuration("2d")` + 2D plans patch + `[2,1,H,W]` 输入),不复用 all-networks 循环;L50-51 的 nnunet_2d skip 是死代码(registry 无此条目),顺手删」。

---

#### R5. 2D 的 `--conditional-mirror-axis` 取值约束未写

**实测(handoff `run_training` L466):** `if conditional_axis not in (0,1,2): raise`。对 2D,mirror_axes=(0,1)(in-slice),LR axis 必须 ∈ {0,1};传 2 会**静默 no-op**(2D MirrorTransform 根本没有 axis 2,保护永不触发,但 telemetry 的 `protected_axis2_mirror_count` 会一直 0 —— **假 PASS**)。

**建议:** spec §5.3 / §6.1 注明:2D 的 `confirmed_lr_axis_2d` ∈ {0,1},audit 脚本与 train.py 启动校验都要 reject axis=2(for 2D)。

---

#### R6. "共用 Wilcoxon/Holm helper"的重构范围未定

**spec §7.2(L188)/ draft B-4:** 「底层 Wilcoxon/Holm 抽共用 helper(与 `run_stats.py` 共享)」。

**实测(`evaluation/run_stats.py`):** 已有 `holm_bonferroni`(L81-93)+ scipy `wilcoxon`(L162)。但没抽成共享模块。

**建议:** spec §7.2 点名重构:把 `holm_bonferroni`(+ 可选 wilcoxon wrapper)抽到如 `evaluation/stats_helpers.py`,`run_stats.py` 与 `run_paired_stats.py` 都 import —— 否则"共用"落空、两份 Holm 实现漂移。

---

### 🟢 可选(OPTIONAL)

- **O1.** spec 未声明 **predict.py 无需 conditional 改动**。实测 `framework/predict.py` 不调任何增强(`do_mirroring=False`,L120;`mirror_axes` 仅透传但 TTA off),conditional 只影响训练增强,predict 透明。建议 §5 或 Stage 5 补一句"predict 链路无 conditional 接线,直接用 `__condlr` ckpt",消除读者疑虑。
- **O2.** §4 / §9.1 的「PACA 2D baseline 与 MultiNetworkTrainer 等价」是**科学等价假设**(clip 值都是 12:nnUNetTrainerV2 native 在 254/264,MultiNetworkTrainer 自有 run_iteration 在 `base_trainer.py:362/373/381`;唯一差异 He-reinit guard 仅在 val Dice==0 训练崩溃时触发)。spec 已标"核实";非 spec 缺陷。可选:加一句"该等价是 baseline 2D 复用的前提,validator 可在 Stage 3 抽查 1 个 seed 的 2D MultiNetworkTrainer 是否复现 PACA 2D baseline(种子噪声内)"——但这要额外重训,可能超 scope,留着判断。

---

## 4. 内部矛盾

**结论:无阻塞性内部矛盾。** 命名/路径/CLI/stage 依赖自洽:

- §8.2 ckpt 路径 ↔ §7.3 baseline 路径 ↔ §2 阵容表:一致(SwinUNETR baseline `v1_comparison/swinunetr__seed<seed>/`、2D baseline `v1_comparison_2d_root/…`、condlr 统一 `v1_comparison/{net}__condlr_seed<seed>/`)。
- §2 表「baseline 全复用不重训」↔ §8.1「6 job」↔ Stage 4:一致。
- §5.3 CLI(`--lr-mirror-mode` default full)↔ §9.1 风险表「baseline 不受影响(`--lr-mirror-mode full` 默认)」:一致。
- stage 依赖链 0→2→3→4→5→6:自洽;Stage 1 标 ✅(commit c585966)与代码现状一致(`framework/train.py:49` 已有 `--network-dim 2d`)。

**唯一张力(已在 §3-M4 覆盖,非矛盾):** §8.2 把 2D condlr ckpt 放 `v1_comparison/`、2D baseline 在 `v1_comparison_2d_root/` —— 这是**有意**(两条 trainer 路径),spec 也承认;但 predict generator(M4)没跟上。属落地遗漏,非内部矛盾。

---

## 5. 总体结论

**评定:需修订(MED)。**

- ✅ **决策忠实度:PASS。** 17 项决策(6 设计 + 11 B 类)在 spec 中**无一遗漏、无一写反、无一冲突**,行号引用(212 / 375-435 / 254/264)**实测全部准确**。spec 是 draft 的忠实整理版。
- ✅ **架构 / 公平性 / stage 骨架:可落地。** 环境/路径/对齐表/命名/stage 依赖自洽;split(实测 `splits_final.pkl` 已放置)、2D plans(`_2D_stage0` 实测存在)、baseline ckpt(SwinUNETR + 2D PACA 实测都在)三条复用锚点**全部就位**。
- ⚠️ **"落地就绪"未达成:** 4 项 **MUST** 落地遗漏(M1 install 时序 / M2 错误技术推理 / M3 paired-stats schema / M4 2D predict 路径)需先补。其中:
  - **M1 + M2 最关键**:M1 是**静默失败风险**(install 顺序错 → conditional 不生效,telemetry 会兜底但应在 Stage 2 就抓);M2 是**错误机制描述**(结论对、推理错,会误导重构)。
  - **M3 + M4 是规格完整性**:新脚本(`run_paired_stats.py`、2D condlr predict)的输入/路径契约缺失,不补则实现者要自行 reverse-engineer,易与 issue #1 口径漂移。
- 🟡 6 项 RECOMMEND(R1-R6)+ 2 项 OPTIONAL,属打磨,不阻塞。

**建议处置:** 补完 M1-M4(预计 1 轮 spec 修订,M1/M2 各 1 段、M3/M4 各 1 小节)后即可标 IMPLEMENTATION-READY 进入 Stage 0。M2 的修正不仅是文字 —— 它澄清了"为什么 seg 在 mirror 时是 full-res"的真正原因(pipeline 顺序),是后续 validator 审查 transform 集成的判据。

---

*审查仅读不改。所有 Huawei 侧核对经 `ssh paca_share` 实测;本地代码核对经 Read 实测。*

---

## 第二轮审查

Status: **READ-ONLY REVIEW**(未修改 spec / draft / 任何代码)
Reviewer: 独立 subagent(全新 session,零父上下文)
Date: 2026-06-23
被审:`docs/v1-kidney-lr-mirror-ablation-spec.md`(修订后,Status 已标 IMPLEMENTATION-READY)
基准:第一轮 REVIEW 的 MUST(M1-M4)+ RECOMMEND(R1-R6)+ OPTIONAL(O1)+ 批次提交,以及 draft
§13 的 35 项决策。

> 审查方法:逐项核对 M1/M2/M3/M4/R1-R6/O1/批次提交是否正确落到 spec(引 spec 行 + 源码行);
> 本地代码经 Read/grep 实测;Huawei 侧经 `ssh paca_share` 实测 handoff、nnUNetV2、moreDA、determinism、
> evaluate_swine_ct 的源码行号。

---

### 1. 修订应用核对(M1 … 批次,逐项)

| 项 | 结论 | 证据 |
|---|---|---|
| **M1** install 时序 | ✅ 正确应用 | spec §5.2 L118-121 写明「initialize() 内 `setup_DA_params()`(base_trainer.py:197)后、`get_moreDA_augmentation()`(:230)前;晚于 train.py:77 determinism」。行号实测全对:`base_trainer.py:197`=`setup_DA_params()`、`:230`=`get_moreDA_augmentation(...)`;`train.py:77`=`install_v1_determinism_patches(...)`(L66 注释已写"must run BEFORE initialize()")。**机制实测成立**:`determinism.py:143` 捕获真 `get_moreDA_augmentation`、`:198` 调它、`:217-218` 仅替换 `get_moreDA_augmentation` 符号(不碰 `MirrorTransform`)→ mirror patch 的 `da_more.MirrorTransform = factory` 与 determinism patch 改的是**两个独立符号**,late-bind,故放在 initialize() 内(L197↔L230 窗口)成立。偏差(handoff 是脚本级 L495、spec 是 initialize 内)已在 draft §13.1 M1 显式标注,功能等价。 |
| **M2** DS 机制 | ✅ 正确应用 | spec §5.2 L113-116 已改成准确机制:「DS 下采样由 moreDA pipeline `DownsampleSegForDSTransform`(`:145-150`)完成,在 `MirrorTransform`(:111)与 `RenameTransform('seg'→'target')`(:140)**之后**」。原错误推理「MultipleOutputLoss2 在 loss 阶段才下采样」**已删除**。moreDA 行号 ssh 实测全对(`:111`/`:140`/`:145-150`)。结论(seg 在 mirror 时是 full-res、key=seg)现由正确机制支撑。 |
| **M3** run_paired_stats 输入契约 | ✅ 应用,但见 §2-§C1 跨节分歧 | spec §7.2 L206-209 写全:8-col schema、join key `(case_id,class_label)`、两臂 `nnunet_2d_condlr`/`swinunetr_condlr`(由 `run_eval.py --network` 传入)、class∈{4,5}、HD95 NaN→drop pair、效应量 r=|z|/√N + mean Δ。schema 与 `run_eval.py:176` 实测逐字一致(`network,seed,case_id,source,class_label,class_name,dice,hd95`)。⚠️ 但该 schema 只匹配 `run_eval.py`,与 §7.4 的 `evaluate_swine_ct.py` 链路 schema 不兼容(见 §C1)。 |
| **M4** 2D condlr predict 路径 | ✅ 正确应用 | spec §8.3 L251-255 写明:`generate_predict_jobs.py` 加 `--variant condlr`;2D condlr 用新模板;CKPT 指 `v1_comparison/nnunet_2d__condlr_seed<seed>/fold_0/model_final_checkpoint.model`;走 `framework.predict --network-dim 2d`;eval 标签 `nnunet_2d_condlr`。实测支撑:`predict.py:43` 已支持 `--network-dim {3d_fullres,2d}`;`generate_predict_jobs.py:87` 的 2D baseline CKPT 硬编码在 `v1_comparison_2d_root/`(确与 condlr 的 `v1_comparison/` 根不同,M4 的区分必要);`:108-113` 的 CLI 无 `--variant`(确需加)。 |
| **R1** 字段名 `protected_lr_mirror_count` | ✅ 正确应用,**全文一致** | spec 全文用 `protected_lr_mirror_count`/`nonprotected_lr_mirror_count`(§5.5 L146/147/152/155、§9.1 L274、Stage 2 L303、Stage 4 L314),**无 `protected_axis2_mirror_count` 残留**。残留的"axis2"字样均合法:L164「不能沿用 axis 2」指 handoff Task520 的 3D LR=axis2;L147 `axis0/1/2_mirror_count` 是 per-axis 诊断计数器(handoff L264-266 实测确为 `axis0/1/2_mirror_count`)。handoff 源码 L262/263/300/302 确用误导名 `protected_axis2_mirror_count` → 改名有据。(注:draft body B-7 L100-101/106 仍是旧名,但 draft §13 R1 行 L266 已记录改名决策 —— draft body 滞后,非 spec 缺陷。) |
| **R2** audit axis 注入 generator | ✅ 正确应用 | spec §8.3 L245-246:generator `--variant condlr` 时加 `--lr-axis-manifest <path>`,读 audit manifest 把 `confirmed_lr_axis_3d`/`_2d` bake 进 `--conditional-mirror-axis`,与 train.py `--lr-axis-audit-manifest`(§5.3 L131)同源单一真相。 |
| **R3** build_cases_csv `--variant condlr` | ✅ 应用,但与 M3 共同构成 §C1 | spec §7.4 L221-222:`build_cases_csv` 加 `--variant condlr`,pred_dir 指 `{network}__condlr_seed<seed>`(原硬编码 `{network}__seed{seed}`),`method` 列打 `nnunet_2d_condlr`/`swinunetr_condlr`。实测支撑:`build_cases_csv.py:41` pred_dir 确硬编码、`:63` `method=f"{network}_seed{seed}"` 确需改。⚠️ 但 `method` 列是 `evaluate_swine_ct.py` 链路的 arm-label,M3 用的是 `run_eval.py` 的 `network` 列(见 §C1)。 |
| **R4** 独立 dim 分支 | ✅ 正确应用 | spec §6.2 L171-177 已改述为「新增 `--network-dim` 分支:dim=2d 走**单网络路径**(`get_default_configuration("2d")` + 2D plans patch + `[2,1,H,W]` 输入),**不复用 all-networks 循环**;删 L50-51 死代码 skip(registry 无 nnunet_2d)」。实测支撑:`smoke_framework.py:38` 硬编码 `"3d_fullres"`、`:49` `for ... in sorted(all_networks())`、`:50-51` `if name=="nnunet_2d": continue`(死代码)。"去 skip+透传"的低估措辞已纠正。 |
| **R5** 2D axis 双重校验 | ✅ 正确应用,**两侧都写** | audit 侧:spec §6.1 L169「`confirmed_lr_axis_2d∈{0,1}`;audit 脚本 reject axis=2(for 2D)」。train.py 侧:spec §5.3 L135-136「2D 时 axis 必须 ∈ {0,1},train.py reject axis=2」。防 telemetry 假 PASS 的两道闸都在。 |
| **R6** stats_helpers.py refactor | ✅ 正确应用 | spec §7.2 L198-199 写明:新建 `evaluation/stats_helpers.py`,把 `holm_bonferroni`(+wilcoxon wrapper)从 `run_stats.py` 移过去,`run_stats.py`+`run_paired_stats.py` 都 import;refactor run_stats 只移动不改逻辑,前后 smoke 确认输出一致。实测支撑:`run_stats.py` 确有 `holm_bonferroni`(第一轮实测 L81-93)+ scipy `wilcoxon`。 |
| **O1** predict 透明 | ✅ 正确应用 | spec Stage 5 L319「predict 无 conditional 接线(O1):`predict.py do_mirroring=False`,增强只在训练」。实测支撑:`predict.py:120` `do_mirroring=False`(注释 # TTA off)。 |
| **批次提交** | ✅ 正确应用 | 新增 spec §8.4 L257-263(提交方式 loop dsub / 不设硬上限 / 混提 / early-runtime 逐个 / 失败单独重提 / Stage 3 PASS 才提 Stage 4)+ Stage 4 L313「loop dsub 提交 6 job(§8.4),early-runtime 逐个检查」。与 draft §13.4 逐条一致。 |

**小结:12 项修订(M1-M4 + R1-R6 + O1 + 批次)全部落到 spec,逐项正确。** 其中 M1(时序,机制实测成立)、M2(机制,源码行号全对)、R1(全文一致无残留)、R5(双侧校验)四项最关键,均核对到位且无错。

---

### 2. 新冲突 / 矛盾

#### §C1(MED)§7.2(M3)与 §7.4(R3)指向**两个不同的 per_case.csv 生产者**,schema 不兼容

**事实(ssh + Read 实测):**

- **§7.2(M3,L206-209)** 钉死 `run_paired_stats.py` 的输入为 8-col:`network,seed,case_id,source,class_label,class_name,dice,hd95`,arm-label 在 **`network`** 列,「由 `run_eval.py --network` 传入」。该 schema 与 `evaluation/run_eval.py:176` 逐字一致 —— 即 **M3 隐含 per_case.csv 由 `run_eval.py` 产出**。
- **§7.4(L219-222)** 给的评估命令链是 `build_cases_csv.py` → `swine_ct_autonomous_discovery/metrics/evaluate_swine_ct.py --cases-csv` → `kidney_swap_eval.py`,arm-label 在 **`method`** 列(`build_cases_csv.py:63` → R3 改成 `nnunet_2d_condlr`)。
- **两者产出 schema 不兼容(实测)**:canonical locked evaluator `evaluate_swine_ct.py` 输出列是 `method`(L50/L363)/`class_id`(L774,非 `class_label`)/`Dice`(L385,大写)/`HD95`(L392,大写);而 `run_eval.py` 输出是 `network`/`class_label`/小写 `dice`/`hd95`。**两套列名/大小写都不同。**

**后果:**
- `run_paired_stats.py` 按 M3 的 8-col 读 —— 只能读 `run_eval.py` 的输出,**读不了 `evaluate_swine_ct.py` 的输出**(列名不匹配 → KeyError)。
- 若实现者按 §7.4 + CLAUDE.md「评估必须用 locked evaluator(`evaluate_swine_ct.py`)」走 canonical 链,得到的 per_case.csv **不符合 M3 schema** → `run_paired_stats.py` 直接报错。
- 若实现者按 M3 用 `run_eval.py`(与 issue #1 / v1-input-consistency 同口径,`run_eval.py` docstring 自述「Locked evaluator for the v1 input-consistency comparison,HD95 computed EXACTLY as `evaluate_swine_ct.py`」),则 **R3 给 `build_cases_csv.py` 加 `--variant condlr` 的工作落在一条不喂 `run_paired_stats.py` 的链上**(白做 + 留下困惑)。

**定性:** 这是修订**新引入/未收口**的跨节分歧 —— 第一轮 M3 把 `run_eval.py` + 8-col 写进 §7.2,而 R3 同时改 §7.4 的 `build_cases_csv`(`evaluate_swine_ct.py` 链),两节各指一个生产者、两套 schema,spec 没声明哪条链喂 `run_paired_stats.py`。非阻塞性(细读 M3 的实现者会收敛到 `run_eval.py`,且该选择与 issue #1 一致、算法与 canonical locked evaluator 逐字相同),但会造浪费或 KeyError。

**建议(单点修订即可):** spec §7.4 增一句明确 per_case 配对的生产者。推荐与 issue #1 / M3 对齐:
> 「**per_case.csv 配对数据由 `evaluation/run_eval.py --network <label>` 产出**(article repo 的 locked-evaluator 实例,HD95 算法与 `evaluate_swine_ct.py` 逐字一致,与 issue #1 同口径);`run_paired_stats.py` 读其 8-col 输出(M3 schema)。`build_cases_csv.py` + `evaluate_swine_ct.py --cases-csv` 是项目级 locked evaluator 的 cases-csv 入口,本实验**不**用于 paired-stats 输入(R3 的 `--variant condlr` 仅在需要 canonical 34-col 报表时才用)。」

或反向:保留 `evaluate_swine_ct.py` 为唯一生产者,把 M3 schema 改成 `method`/`class_id`/`Dice`/`HD95` 并相应调整 `run_paired_stats.py`。二选一,但必须收口。

---

### 3. 新遗漏(分级)

#### 🔴 必须补
- **无。** 落地必需的钩子(M1 时序 / M2 机制 / M3 schema / M4 路径 / R1-R6 / 批次)均已写入;唯一需收口的是 §C1 的跨节分歧(归到 §2「冲突」,非纯遗漏)。

#### 🟡 建议
- **N1(M3 的"drop 该 pair"在 3-seed 平均语境下歧义)。** spec §7.2 L209「HD95 NaN → drop 该 pair」。但配对是 per-case **先 3-seed 平均**再 baseline-vs-conditional;若某 seed 的 HD95 为 NaN(`run_eval.py:193` 写 `"nan"`),`np.mean` 会把整 case 平均成 NaN(丢 case),`np.nanmean` 则只丢该 seed(留 case)。spec 未指明 mean vs nanmean。kidney 几乎不空、影响小,但应写清(建议:`np.nanmean` over 3 seed,该 case 全 NaN 才 drop pair)。

- **N2(`conditional_mirror_axis` 透传到 trainer `__init__` 未显式写)。** spec §5.3 L136 只说「`MultiNetworkTrainer.__init__` 加 `lr_mirror_mode` 参数,透传」,但 `initialize()` 内调 `install_conditional_mirror_patch(conditional_mirror_axis=...)` 还需要 `conditional_mirror_axis` 也存在 trainer 上(§5.4 说它是动态 CLI 参数)。建议补一句「`__init__` 同时接收 `lr_mirror_mode` + `conditional_mirror_axis`,供 `initialize()` 内 install 使用」。实现者可推断,但写明更稳。

#### 🟢 可选
- **N3(§7.4 链路箭头不精确)。** `kidney_swap_eval.py` 实测是**独立**脚本(直接吃 `--predictions`,非 `--cases-csv`),与 `evaluate_swine_ct.py` 并行而非下游。§7.4「→ kidney_swap」的箭头是松画法,不构成阻塞,可顺带写清「kidney_swap 独立跑(吃 predictions)」。
- **N4(§8.1 "agent<170" 的 dsub 语法)。** spec §8.1 给「节点 agent<170、不用 whshare-agent-174」,但 §8.4 的 loop dsub 未写具体 `-pn` 谓词(CLAUDE.md 只给了 `-pn '!whshare-agent-174'`,"agent<170" 谓词未明确)。非修订引入,实现者可从 CLAUDE.md + generate 脚本补,可选写明。

---

### 4. IMPLEMENTATION-READY 评定

**评定:需修订(MED),但距 IMPLEMENTATION-READY 仅差单点。**

- ✅ **修订忠实度:PASS。** 12 项修订(M1-M4 + R1-R6 + O1 + 批次)逐项正确落到 spec,无遗漏、无写反。最关键的 M1(install 时序,机制 ssh 实测成立)、M2(DS 机制,moreDA 行号 :111/:140/:145-150 实测全对)、R1(全文一致无 `protected_axis2` 残留)、R5(双侧校验)均核对到位。
- ✅ **行号引用:PASS。** spec 引用的源码行号(base_trainer 197/230、train.py 77、moreDA 111/140/145-150、nnUNetTrainerV2 254/264、handoff 212/375/428、predict.py 43/120、smoke_framework 38/49/50-51、build_cases_csv 41/63、generate_predict_jobs 45-47/87/108-113、run_eval 142/176/193)ssh + Read 实测全部准确。
- ⚠️ **唯一阻塞点:§C1(MED)。** §7.2(M3)与 §7.4(R3)各指一个 per_case 生产者(`run_eval.py` vs `evaluate_swine_ct.py`),schema 不兼容(`network`/`class_label`/小写 vs `method`/`class_id`/`Dice`/`HD95`)。spec 必须收口到单一生产者 + 单一 arm-label 列,否则 Stage 6 的 `run_paired_stats.py` 要么白做 R3、要么 KeyError。**这是修完后即可标 IMPLEMENTATION-READY 的单点**(§7.4 加一句声明,见 §2 建议)。
- 🟡 N1-N4 为建议/可选招待,不阻塞。

**处置建议:** 收口 §C1(§7.4 加 1 句明确 per_case 生产者 = `run_eval.py`,与 issue #1 + M3 对齐;或反向改 M3 schema)后,spec 即可进入 Stage 0(audit + 2D smoke)。M1 的时序契约 + M2 的正确机制描述已足够支撑实现者照 spec 落地,不需要第三轮审查。

---

*第二轮审查仅读不改。所有结论附 `file:line`;Huawei 侧核对经 `ssh paca_share` 实测(handoff `train_conditional_lr_mirror_v1_reviewed.py`、`nnUNetTrainerV2.py`、`data_augmentation_moreDA.py`、`determinism.py`、`evaluate_swine_ct.py`),本地代码核对经 Read/grep 实测。*
