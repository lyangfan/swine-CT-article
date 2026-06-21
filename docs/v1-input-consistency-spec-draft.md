# Spec Draft — v1 输入一致性(多网络,网络不写死)

Status: DECIDED — 11/11 题已决策
Date: 2026-06-21

## 目标

文章里所有分割模型读**同一份 nnU-Net v1 预处理 + 同一套 v1 数据/增强管线**,
差异只在模型架构,从而保证公平对比。选 v1 是因为 v2 有无法固定的随机性,
而 v1 的 `nnunetv1_compat` 能做到 `state_dict_equal=true`(确定性可控)。

**已定**:用 v1;固定 6:2:2 split(Task601,已搭建+审计通过);**网络不写死,
后面还会跑其它网络** —— v1 input space 作为公共层,网络以可插拔注册方式接入。

---

## 设计含义(由 Q1 决策派生)

- 写一个**网络无关的 base trainer**(子类 `nnUNetTrainerV2`),v1 数据管线(数据集、
  batchgenerators transforms、patch sampling、foreground oversample)+ 确定性 patch
  全部固定在 base 里。
- 一个**网络注册表/工厂**:每个网络只提供 `build_fn`(arch 参数→网络)+
  `forward_handle`(单输出 / deep-supervision 列表适配)+ 可选 optimizer/loss override。
- 加新网络 = 注册一个 builder,不改 base、不改数据管线。
- 第一个具体网络:SwinUNETR(作为首个 plugin 实现)。

## 统一 nnunetv1 环境(华为)

所有 v1 预处理 / 训练统一用 `swine_ct_autonomous_discovery` 那套**已验证**的 nnunetv1 环境。
参考脚本:`runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh`
(函数 `swct_setup_nnunetv1_env` + `swct_nnunetv1_preflight`)。

**CUDA stack(module load):**
- `compilers/gcc/9.3.0`
- `compilers/cuda/11.8.0`
- `libs/cudnn/8.8.1_cuda11`
- `libs/nccl/2.16.5-1_cuda11.8`

**关键路径:**
- `REMOTE_PROJECT_ROOT = /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery`
- `NNUNETV1_ENV_ROOT = $REMOTE_PROJECT_ROOT/envs/nnunetv1`(python / bin)
- `NNUNETV1_COMPAT_ROOT = $REMOTE_PROJECT_ROOT/scripts/nnunetv1_compat`(确定性套件,加进 `PYTHONPATH`)

**Task601 数据根(覆盖默认;用 `SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1`):**
- `nnUNet_raw_data_base = swine-CT-article/data/nnunetv1`
- `nnUNet_preprocessed   = swine-CT-article/data/nnunetv1/nnUNet_preprocessed`
- `RESULTS_FOLDER        = swine-CT-article/data/nnunetv1/nnUNet_results`
- Task601 raw 必须在 `$nnUNet_raw_data_base/nnUNet_raw_data/Task601_Article622_Carcass9Class/`(v1 硬性布局)

**必要 hack:**
- sklearn `libgomp` `LD_PRELOAD`(`scikit_learn.libs/libgomp.so`,解决符号冲突)
- `OMP_NUM_THREADS / OPENBLAS / MKL / NUMEXPR = 16`

**激活:** 先 export 上面三个 data-root 覆盖 + `SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1`,再 `source setup_nnunetv1_env.sh`,然后跑 `swct_nnunetv1_preflight`(校验 import / CUDA / `nnUNet_train`/`nnUNet_predict` 在 PATH)。**所有 v1 job 前必须 preflight 通过。**

---

## 待决策问题(按依赖顺序)

> 每题:`背景` / `选项` / `我的建议` / `决策`(讨论后回填)

### Q1. 文章模型阵容 ✅

- **背景**:决定整套 v1 input infra 要支持到多通用。
- **选项**:(a) 只 nnU-Net v1 + SwinUNETR;(b) + SAM/MedSAM;(c) + VoCo 等 SSL;(d) 其它
- **我的建议**:(a) 先只做两个,子类化模式天然可扩展。
- **决策**:**通用多网络框架,网络不写死。** v1 input space 作为公共层,网络可插拔注册
  (见上方"设计含义")。Q7(deep supervision)因此变成"每个网络的能力 flag",不是全局决策。

### Q2. 网络初始化方式 ✅

- **背景**:scratch 去掉预训练数据混淆变量(最干净架构对比);pretrained 体现实际性能但引入
  预训练数据来源差异(且不同网络可用权重来源不同,难公平)。
- **选项**:(a) 主对比 scratch;(b) 主对比 pretrained;(c) 两条都做
- **我的建议**:(a) 主对比 scratch,框架 `build_fn` 仍支持权重加载开关。
- **决策**:**所有网络统一 scratch,不做预训练。** 框架主路径不需要预训练权重分支,每个网络
  随机初始化。变量降到只剩架构。

### Q3. nnU-Net v1 对照基线 ✅

- **背景**:nnU-Net v1 这个 baseline 架构怎么来 —— Task601 重训,还是对齐已有 r20 5-fold?
- **选项**:(a) Task601 上重新训练;(b) 对齐已有 r20 5-fold
- **我的建议**:(a) Task601 重新训练(r20 是 5-fold CV,split 协议不同,不可比)。
- **决策**:**在 Task601 上重新训练 nnU-Net v1,严格按我们的 6:2:2 split,500 epochs。**
  nnU-Net v1 走和其它网络完全一样的流水线。**这同时把 Q5(训练预算)定为 500 epochs**
  (公平要求同预算)。

### Q4. 优化器 ✅

- **背景**:多网络下共用一个优化器,还是各自适配?nnU-Net v1 原生 SGD+poly;transformer 系适配
  AdamW(用 SGD 会吃亏)。
- **选项**:(a) 都 SGD+poly;(b) 各自适配;(c) 都 AdamW
- **我的建议**:(b) 各自适配,文档写清。
- **决策**:**各自架构适配,按架构家族分规则**:
  - **CNN 类** → SGD + poly lr(与 nnU-Net v1 一致)
  - **Transformer 类** → AdamW + warmup_cosine
  - 框架注册时每个网络声明架构家族,优化器按家族自动走。LR/调度具体值在 config 里固定并记录。

### Q5. 训练预算 ✅

- **背景**:公平对比要求所有网络同预算。
- **选项**:(a) 都按统一 epochs;(b) 同 total iterations/GPU-hours;(c) 各自原协议
- **我的建议**:(a) 统一 epochs。
- **决策**:**所有网络统一 500 epochs**(Q3 给 nnU-Net v1 定的 500 推广到所有网络)。

### Q6. patch size / batch size ✅

- **背景**:input 一致要求所有网络共用同一个 patch/batch。
- **选项**:(a) 都用 Task601 v1 plans 派生;(b) 固定 [64,160,160]/batch2
- **我的建议**:(a) v1 plans 派生值(canonical、数据适配、与预处理空间对齐)。
- **决策**:**所有网络 patch_size 尽量和 nnU-Net v1 plans 派生值一致**(以 Task601 v1 plans 为
  canonical,所有网络共用)。个别架构若有硬约束(如需被 2^N 整除)取最接近可行值并在 config
  记录偏差,但目标是精确一致。batch size 同理跟 plans。

### Q7. deep supervision(每网络 flag)✅

- **背景**:nnU-Net v1 原生多尺度 DS;SwinUNETR 等单输出。多网络下这是每网络能力。
- **选项**:(a) base 支持两种 forward 协议,网络自声明,不改架构;(b) 给单输出网络加 DS heads(改架构)
- **我的建议**:(a) 接受不对称,各用原版。
- **决策**:**接受不对称,按各网络原版。** nnU-Net v1 保留原生 DS(多尺度输出 +
  MultipleOutputLoss2),单输出网络(SwinUNETR 等)走单输出 + 直接 DC_and_CE。base trainer 的
  `run_iteration` 按网络注册的 forward 协议分支,**不改任何架构**。报告时写明这是架构相关差异。

### Q8. Task601 预处理来源

- **背景**:Task601 自己跑 `nnUNet_plan_and_preprocess -t 601`;或复用 swine_ct_autonomous_discovery
  现成 Task509/520(all197,plans_v2.1)软连接。
- **选项**:(a) Task601 自己跑;(b) 复用现成 v1 预处理
- **我的建议**:**(a) Task601 自己跑**。干净隔离、自文档化,符合"文章 workspace 独立"原则;
  30-60min 一次性成本可接受。
- **决策**:

### Q9. shared_unpacked_cache 复用 ✅

- **背景**:Q8 自己跑 plan_and_preprocess(158-based)后,要不要 unpack npy?复用 197-based 的
  `all197_stage1` 会因归一化口径不同(158 vs 197)导致数值对不上 —— 复用与 Q8 冲突。
- **选项**:(a) 软连接复用(与 Q8 冲突,否决);(b) 自己 unpack;(A' 变体)不 unpack 直接读 npz
- **我的建议**:(b) 自己 unpack(与 Q8 隔离一致)。
- **决策**:**自己 unpack**(`--unpack-data --fp16`)。完全隔离,不碰 shared_unpacked_cache。
  fp16 把 npy 砍到 ~50-68GB(158 例)。归一化口径是 Task601 自己 158-based 的,自洽自文档化。

### Q10. 确定性杆 ✅

- **背景**:v1 的立身之本是确定性(`state_dict_equal=true`)。其它网络是否同杆?
- **选项**:(a) 所有网络都要求 state_dict_equal=true;(b) 只要求 loss 曲线一致
- **我的建议**:(a) 同杆。
- **决策**:**所有网络都要求 `state_dict_equal=true`。** base trainer 装 `install_v1_determinism_patches`
  (cudnn deterministic、seed everything、python hash seed 等),所有注册网络享受同一套确定性保证。
  确定性是框架级、与具体网络无关;SwinUNETR 等无内在不可复现因素,同杆可达。

### Q11. 评估口径 ✅

- **背景**:test 39 评估。现有 locked evaluator(`evaluate_swine_ct.py`),Dice+HD95 + 条件类
  处理(head 只 HZAU、testis 只 TB、head-absent FP)。预测口径 no TTA / no ensemble /
  disable post-processing / 不存 softmax。
- **选项**:(a) 复用现有 locked evaluator + 条件类处理;(b) 另定
- **我的建议**:(a) 复用。
- **决策**:**复用 locked evaluator + 条件类处理。** 所有网络过同一个 evaluator,预测统一
  no TTA / no ensemble / disable post-processing / 不存 softmax。评估侧零变量,公平性闭环。

---

## 第二轮:公平性补充问题(Q12-Q19,讨论中)

> 第一轮定了 split/预算/优化器/patch/DS/确定性/评估的大框架,但参照 PACA
> `hyperparameter_fair_comparison_protocol.md` / `swinunetr_scratch_baseline_spec.md`,
> 还有几个**输入侧 / 训练侧的公平性维度**没显式锁定。逐题补。

### Q12. 数据增广一致性 ✅

- **背景**:PACA 硬性要求所有方法共用同一套 augmentation。`nnUNetTrainerV2` 实际用 `get_moreDA_augmentation`(丰富版),非 sparse default。
- **选项**:(a) 所有网络共用 v1 moreDA;(b) 各网络用自己原版增强
- **我的建议**:(a) 共用 v1 moreDA。
- **决策**:**所有网络锁死共用 nnU-Net v1 moreDA 增广**(SpatialTransform 旋转/缩放/弹性 + GaussianNoise +
  GaussianBlur + BrightnessMultiplicative + Brightness 加性偏移 + ContrastAugmentation + SimulateLowResolution +
  GammaTransform 反转/非反转 + Mirror)。不额外加、不做类别专项、seed 固定、resolved_config 记录全套。
  **调研确认**:nnFormer/SegFormer3D/UNETR/SwinUNETR/MedNeXt/SegResNet/2D-nnUNet 没有一个论文声称某种
  "moreDA 之外、且性能关键"的特殊增广;MONAI 系官方增广是 moreDA 子集。MedNeXt 直接复用 v1 moreDA。

### Q13. 前景过采样 / patch sampling 规则 ✅

- **背景**:PACA 锁定 patch 采样严格复刻 nnU-Net dataloader(force-fg 0.33,无类别专项)。决定 patch 分布,是 input 一致性核心。
- **选项**:(a) 共用 v1 采样规则;(b) 允许调整
- **我的建议**:(a)。
- **决策**:**所有网络共用 v1 dataloader 采样规则**:`oversample_foreground_percent=0.33`、batch 内
  sample0=random crop + sample1=forced-foreground crop(从 .pkl `class_locations` 取中心)、case 采样
  均匀随机。**禁止**类别专项 oversampling / hard-case mining / class-balanced sampling / 动态调整。
  base trainer 继承 `nnUNetDataLoader`,所有网络自动拿到。

### Q14. patch size 架构整除约束 ✅

- **背景**:Q6 要所有网络 patch 跟 v1 plans 一致。transformer 系(SwinUNETR/UNETR/MedNeXt)要求
  每个空间维度被 32 整除。
- **选项**:(a) v1 plans patch 优先,遇硬约束取最近合规值;(b) 强制统一
- **我的建议**:(a)。
- **决策**:**Task601 v1 plan_and_preprocess 完整跑完(DSUB 564154 SUCCEEDED,2026-06-22;plan-only
  564151)。`plans_per_stage` 是 2 stage:stage0 `[128,128,128]`@resampled `[5.10,2.56,2.56]`(各向同性化
  备选)/ **stage1(3d_fullres 实际用)`[64,160,160]`**@原始 `[5,0.97656,0.97656]`(不 resample,各向异性
  pooling `[[1,2,2],[1,2,2],[2,2,2],…]`)。transpose_forward `[0,1,2]`,do_dummy_2D=False。**
  完整预处理产物已生成:`nnUNetData_plans_v2.1_stage0/`(316)、`_stage1/`(316)、`_2D_stage0/`、
  `gt_segmentations/`(158)、2D/3D plans、dataset.json。
  **`[64,160,160]` 天然被 32 整除**(64=32×2,160=32×5)→ SwinUNETR/UNETR/MedNeXt 直接用,**无需调整,
  Q14 整除担心完全消解**。所有网络统一 `[64,160,160]` + batch 2。

### Q15. 损失函数一致性

- **背景**:PACA 锁定 supervised loss = CE + Dice(所有 SwinUNETR 系)。nnU-Net v1 用
  `DC_and_CE_loss`。loss 是训练目标,不一致会引入额外变量。
- **选项**:(a) 所有网络统一 `DC_and_CE_loss`(DS 网络走 MultipleOutputLoss2 加权、单输出网络
  直接算,但底层 loss 函数相同);(b) 允许每网络用自己原版 loss
- **我的建议**:
- **决策**:

### Q16. 模型选择指标(val)

- **背景**:从 val 集选最佳 checkpoint 的 metric 必须预定义,禁 post-hoc 选有利指标(PACA 用
  common-class Dice,排除条件类 head/testis 避免被 HZAU/TB 不对称带偏)。
- **选项**:(a) 预定义 val common-class Dice(front/middle/end/kidney×2/cavity×2 这 7 个
  始终存在的类的 Dice 均值,排除条件类 head/testis);(b) 全类 Dice;(c) 别的
- **我的建议**:
- **决策**:

### Q17. effective batch / gradient accumulation

- **背景**:有些网络显存吃紧,physical batch 可能 <2。要保证 effective batch 一致(公平预算)。
- **选项**:(a) target effective_batch_size = v1 plans batch(通常 2 patches),physical 不够时用
  gradient accumulation 补到 effective=2;(b) 各网络 physical batch 不同就不同
- **我的建议**:
- **决策**:

### Q18. weight decay / scheduler 细节

- **背景**:Q4 定了 optimizer 按 family(CNN→SGD+poly,Transformer→AdamW)。scheduler 跟着 family
  (SGD→poly lr,AdamW→warmup_cosine),weight decay 要不要也锁定?
- **选项**:(a) scheduler 跟 family(Q4 已隐含),weight decay 按 family 固定小值(CNN 用 v1 默认,
  Transformer 如 1e-5),config 记录;(b) 允许搜 weight decay
- **我的建议**:
- **决策**:

### Q19. 2D nnUNet 的处理(阵容扩展引入)

- **背景**:你想加 2D nnUNet。但它是 2D 模型,进不了我们 3D v1 预处理空间(3D 体素重采样/crop
  对 2D 切片模型不适用)。把它硬塞进 3D 框架不公平,也不自然。
- **选项**:(a) 2D nnUNet 作为**单独类别的参考基线**,走它自己的 nnU-Net 2D 预处理,单独报告,
  不进 3D 主对比表;(b) 剔除 2D nnUNet,只做 3D 架构对比;(c) 硬塞(不推荐)
- **我的建议**:
- **决策**:

## 决策汇总(讨论后回填)

| Q | 决策 |
|---|---|
| Q1 模型阵容 | ✅ 通用多网络框架,网络不写死,可插拔注册 |
| Q2 SwinUNETR 初始化 | ✅ 所有网络统一 scratch,不预训练 |
| Q3 nnU-Net 对照基线 | ✅ Task601 重训,严格按 6:2:2 split,500 epochs(连带定下 Q5=500) |
| Q4 优化器 | ✅ 各自适配:CNN→SGD+poly,Transformer→AdamW |
| Q5 训练预算 | ✅ 所有网络统一 500 epochs |
| Q6 patch/batch | ✅ 所有网络 patch_size 尽量跟 nnU-Net v1 plans 派生值一致 |
| Q7 deep supervision | ✅ 接受不对称,各网络用原版(nnU-Net 保留 DS,单输出网络单输出) |
| Q8 Task601 预处理来源 | |
| Q9 shared cache 复用 | ✅ 自己 unpack(--unpack-data --fp16),不复用 197-based cache |
| Q10 确定性杆 | ✅ 所有网络要求 state_dict_equal=true |
| Q11 评估口径 | ✅ 复用 locked evaluator + 条件类处理,预测 no TTA/ensemble/PP |
| Q12 数据增广一致性 | ✅ 所有网络锁死共用 nnU-Net v1 moreDA(全套,无额外/无类别专项) |
| Q13 前景过采样规则 | ✅ 共用 v1 dataloader(force-fg 0.33,无类别专项) |
| Q14 patch 整除约束 | ✅ 实测 v1 patch=[64,160,160] batch2,天然 32 整除,无需调整 |
| Q15 损失函数一致性 | |
| Q16 模型选择指标 | |
| Q17 effective batch | |
| Q18 weight decay/scheduler | |
| Q19 2D nnUNet 处理 | |

---

## 实现路线图(决策全部落地后的 build 顺序)

1. **Task601 v1 预处理**(Q3/Q8/Q9):v1 env 下 `nnUNet_plan_and_preprocess -t 601
   --unpack-data --fp16`(158 Tr case,自己 unpack,fp16,~50-68GB);把 `splits_final.pkl`
   放进 preprocessed 目录(覆盖自动 5-fold)。
2. **网络无关 base trainer**(Q1/Q7/Q10):子类 `nnUNetTrainerV2`,固定 v1 数据管线 + 确定性
   patch;`run_iteration` 按网络注册的 forward 协议分支(DS 列表 / 单输出);所有网络同杆
   `state_dict_equal=true`。
3. **网络注册表**(Q1/Q2/Q4/Q6):每个网络注册 `build_fn`(scratch,Q2)+ forward 协议 + 架构
   家族(CNN/Transformer,Q4);patch/batch 跟 Task601 v1 plans(Q6);统一 500 epochs(Q5)。
   - plugin 1:**nnU-Net v1**(Generic_UNet,DS,CNN→SGD+poly)
   - plugin 2:**SwinUNETR**(单输出,Transformer→AdamW)
4. **配置驱动训练**:每个实验 = 一个 yaml(Task601 预处理 + split + 网络 + 训练参数)。
5. **test 39 预测**(Q11):每网络 no TTA / no ensemble / disable post-processing / 不存 softmax。
6. **评估**(Q11):所有网络过同一个 locked evaluator + 条件类处理。
7. **确定性审计**(Q10):每网络过 `run_determinism_smoke`,要求 `state_dict_equal=true`。
