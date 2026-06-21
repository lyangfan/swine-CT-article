# Spec Draft — v1 输入一致性(多网络,网络不写死)

Status: Q1-Q19 已决策;Q20-Q27(工程/运营/科学缺口)待讨论 + 若干待澄清项
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

## 最终阵容(2026-06-22 定)

**4 个 3D + 1 个 2D**,优先最新 + 每条路线一个代表作:

| 网络 | 年份 | 路线 | 保留理由 |
|---|---|---|---|
| **nnU-Net v1** | 2020 | 经典 CNN(数据驱动 auto-config) | baseline 标准 |
| **MedNeXt-S** | 2023 MICCAI | 现代 ConvNet(ConvNeXt) | 最新卷积 |
| **SwinUNETR** | 2022 | transformer 旗舰(Swin,MONAI) | 最广泛使用 |
| **SegFormer3D-aniso** | 2024 CVPR-W | 最新 transformer(轻量高效) | 最新架构(已修 aniso) |
| **2D nnUNet** | — | 2D 参考(单独类别,Q19) | 2D/3D 维度对比 |

**已砍(2026-06-22):**
- SegResNet(2018)—— 最老,CNN 已有 nnU-Net 代表。
- UNETR(2022)—— SwinUNETR 前身(同作者),SwinUNETR 足够代表 transformer。
- nnFormer(2022)—— 路线重叠 + vendoring 最麻烦(fork 冲突 + patch_size bug)。

**训练 job:5 网络 × 3 seed = 15 个**(原 24 减到 15)。

## 网络架构超参(已 pin,2026-06-22)

> 写 configs/*.yaml 的 arch 参数参考。nnU-Net v1 和 2D nnUNet 全自动(planner),不需要手写。

### nnU-Net v1
- 全部从 Task601 plans 自动(base_features / num_pool / conv_kernels / pool_kernels),已算完(Q14)。

### SwinUNETR(MONAI)
- `in_channels=1, out_channels=10`（Q23）
- `feature_size=48`（PACA swinunetr_scratch.yaml + MONAI 默认）
- `depths=[2,2,2,2], num_heads=[3,6,12,24], window_size=7`
- `norm_name="instance"`
- `drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0`
- **`use_v2=True`**（SwinUNETR-V2,MICCAI 2023,更强;PACA 用此版）
- `spatial_dims=3`

### MedNeXt-S(vendored,create_mednext_v1）
- `num_input_channels=1, num_classes=10`（Q23）
- `model_id="S"` → n_channels=32, block_counts=[2,2,2,2,2,2,2,2,2]（自动）
- `kernel_size=3`（3×3×3,smoke test 验证）
- `exp_r=4`（paper 默认）
- **`deep_supervision=True`**（paper 原版;base trainer 走 DS 列表 forward 协议）
- **`do_res=True`**（paper 原版,MedNeXt block 内残差）
- **`do_res_up_down=True`**（paper 原版,resampling block 残差）
- `norm_type="group"`（paper 默认）

### SegFormer3D-aniso(vendored)
- `in_channels=1, num_classes=10`（Q23）
- 全部用 repo/paper 默认:`sr_ratios=[4,2,1,1], embed_dims=[32,64,160,256], patch_kernel_size=[7,3,3,3], patch_stride=[4,2,2,2], patch_padding=[3,1,1,1], num_heads=[1,2,5,8], depths=[2,2,2,2], mlp_ratios=[4,4,4,4], decoder_head_embedding_dim=256, decoder_dropout=0.0`

### 2D nnUNet
- 全部从 2D plans 自动（`_2D_stage0` 已生成）。

### CNN 家族 optimizer/training 超参（nnU-Net v1 / 2D nnUNet / MedNeXt-S 统一）
- **跟 nnU-Net v1 完全一致**:SGD + poly lr,initial_lr=**0.01**,weight_decay=**3e-4**,momentum=0.99(nnU-Net v1 默认)
- MedNeXt-S 算 **CNN 家族**(ConvNeXt 系,但按 Q4 规则归 CNN → SGD+poly,不是 AdamW)
- 不自定义,全部用 nnU-Net v1 原生默认值

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
- **决策**:**预算口径锚定到「总迭代次数」,所有网络(含 2D nnUNet)统一 125,000 次 optimizer step**。
  nnU-Net v1 每 epoch 固定迭代 `num_iterations_per_epoch=250`(PACA 审计确认),所以
  **500 epochs × 250 = 125,000 iters**。base trainer pin `num_iterations_per_epoch=250`(不允许任何网络改),
  所有 3D 网络 + 2D nnUNet(同 trainer 同 250)都跑 500 ep = 125,000 iters。每 iter = 1 optimizer step;
  3D 每 iter effective batch=2 个 3D patch(Q17),2D 每 iter 是它 plans 的 2D batch(更大,但 2D 是单独类别
  Q19,patch 数差异是 2D/3D 维度差,可接受)。**预算单位是 iterations 不是 epochs**,确保不同网络看到相同的
  optimizer 更新次数。

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

### Q8. Task601 预处理来源 ✅

- **背景**:Task601 自己跑 `nnUNet_plan_and_preprocess -t 601`;或复用 swine_ct_autonomous_discovery
  现成 Task509/520(all197,plans_v2.1)软连接。
- **选项**:(a) Task601 自己跑;(b) 复用现成 v1 预处理
- **我的建议**:**(a) Task601 自己跑**。干净隔离、自文档化,符合"文章 workspace 独立"原则;
  30-60min 一次性成本可接受。
- **决策**:**Task601 自己跑 plan_and_preprocess(158-based,隔离)。实际已完成**(DSUB 564154
  SUCCEEDED,2026-06-22;详见 Q14)。归一化口径是 Task601 自己 158 例算的,自洽。不复用 197-based 产物。

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
  **cudnn 澄清(2026-06-22)**:所有网络(含 transformer)用**跟 nnU-Net v1 完全一致的确定性栈**:
  `cudnn.deterministic=True` + `cudnn.benchmark=False` + seed everything(python/numpy/torch/cuda)+ worker seeds
  (train `[base+1000+i]`/val `[base+2000+i]`)+ PYTHONHASHSEED。**不额外关 TF32 / 不加
  `torch.use_deterministic_algorithms`**(PACA 在 A100 上用这套栈对 nnU-Net v1 验证了 state_dict_equal=true
  + optimizer_state_dict_equal=true,同节点 + 跨节点都过;非 bitwise 的只有 plot_stuff/best_stuff 浮点末位差,
  非本质)。PACA determinism smoke 用的就是这套(setup: A100/cudnn8.8.1/torch2.4.1+cuda118/fp16 AMP)。

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
  **GPU forward smoke test 实测**(DSUB 564162,`source/smoke_test_patch.py`,patch `[2,1,64,160,160]`):
  ✅ **全部 7 个网络 forward 通过**(out `[2,10,64,160,160]`):SwinUNETR / UNETR / SegResNet / MedNeXt-S /
  **nnFormer(patch_size=`[4,4,4]`)** / **SegFormer3D-aniso(修复版)**。
  ⚠️ **nnFormer** 必须用 patch_size=`[4,4,4]`(默认 `[2,4,4]` 因 PatchEmbed `(patch//2)²` bug 失败,属 config 选择)。
  ⚠️ **SegFormer3D** 上游版在非立方 + torch 2.4 下失败(`@torch.jit.script cube_root` 崩 + cube_root 立方假设)。
  已修(`source/SegFormer3D/architectures/segformer3d_aniso.py`):去 jit、cube_root reshape 改显式 (D,H,W) 穿线,
  attention/MLP/decoder 不变 —— 属 I/O reshape 修复非架构改动,scratch 下无权重兼容问题,实测通过。

### Q15. 损失函数一致性 ✅

- **背景**:loss 是训练目标,不一致引入额外变量。PACA 锁 CE+Dice。
- **选项**:(a) 所有网络统一 `DC_and_CE_loss`;(b) 各用原版 loss
- **我的建议**:(a)。
- **决策**:**所有网络统一 nnU-Net v1 的 `DC_and_CE_loss`(SoftDice batch_dice + 加权 CE)。DS 网络
  (nnU-Net/MedNeXt/nnFormer/SegFormer3D-aniso)走 `MultipleOutputLoss2` 多尺度加权,单输出网络
  (SwinUNETR/UNETR/SegResNet)直接算,底层同一函数同一参数。**调研确认**:7 个网络全部原生用
  Dice+CE(nnU-Net/MedNeXt/nnFormer 是 DC_and_CE,SwinUNETR/UNETR/SegResNet/SegFormer3D 是 MONAI
  DiceCELoss),统一到 DC_and_CE 不改变任何网络的训练目标本质,无网络吃亏。

### Q16. 模型选择指标(val)✅

- **背景**:选 checkpoint 的 val metric 要预定义,禁 post-hoc。条件类 head/testis 在 val 上受 HZAU/TB 比例影响。
- **选项**:(a) common-class Dice 选 best;(b) 全类 Dice;(c) 用 final checkpoint 不选
- **我的建议**:(a) common-class。
- **决策**:**所有网络统一用 final checkpoint(第 500 epoch,LR 衰减到底的模型),不按 val 选 best —— 跟
  nnU-Net v1 默认一致。** 彻底消除"选 checkpoint 指标"这个变量(无网络能 cherry-pick 最佳 val epoch)。
  val 集降级为**监控**(看 loss/Dice 曲线确认收敛,不做选择)。test 最终评估(Q11 locked evaluator)
  跑在 final-checkpoint 的预测上。原 common-class Dice 讨论作废。

### Q17. effective batch / gradient accumulation ✅

- **背景**:transformer 系 patch [64,160,160]+batch2 可能 OOM。公平要求 effective batch 一致(Q6 batch=2)。
- **选项**:(a) effective=2 + grad accum 兜底;(b) physical 各异
- **我的建议**:(a)。
- **决策**:**effective_batch_size = 2 patches**(v1 plans 派生,Q6)。physical 不够时用 gradient accumulation
  补到 effective=2(`physical × accum = 2`)。base trainer 支持 grad accum。**已知语义 caveat**:grad accum 的
  batch_dice loss 不完全等价 physical batch 2(两个 micro-batch 各算 Dice 再平均 vs 真 batch 跨 2 样本统计),
  可接受的微小偏差,config + 文章记录。

### Q18. weight decay / scheduler 细节 ✅

- **背景**:Q4 定 optimizer 按家族。scheduler 跟家族(poly 配 SGD、warmup_cosine 配 AdamW)。wd 要否锁?
- **选项**:(a) scheduler 跟家族 + wd 固定小值不搜;(b) 搜 wd
- **我的建议**:(a)。
- **决策**:**scheduler 跟家族,CNN→SGD+poly,Transformer→AdamW+warmup_cosine。weight decay 按家族固定小值不搜**:
  CNN 系 nnU-Net v1 默认 SGD wd(3e-4),Transformer 系 AdamW wd 1e-5 + lr 4e-4(与 PACA v2 一致)。
  optimizer/scheduler/wd/lr 全部写死 config,resolved_config 记录,**零调参变量**。

### Q19. 2D nnUNet 的处理(阵容扩展引入)✅

- **背景**:2D nnUNet 是 2D 模型,进不了 3D v1 预处理空间。3D 主对比已干净(7 网络)。
- **选项**:(a) 单独类别参考;(b) 剔除;(c) 硬塞
- **我的建议**:(a) 单独类别参考。
- **决策**:**2D nnUNet 作为单独类别参考基线**:走 nnU-Net v1 自己的 2D pipeline(`nnUNet_train 2d`,
  独立 2D plans 预处理,与 3D Task601 隔离),单独训练 + 评估 + 报告,**不进 3D 主对比表**。
  文章:3D 主对比表(7 网络统一 `[64,160,160]`)+ 补充"2D nnUNet 参考"小节(不同维度)。

## 第三轮:工程 / 运营 / 科学缺口(Q20-Q27,待讨论)

> Q1-Q19 锁定了公平性维度。但从「spec → 实际跑出 8 个网络结果」还差工程集成、运营调度、
> 科学报告层面的东西。逐题待讨论。

### Q20. 网络集成机制(尤其 nnFormer 的包冲突)✅

- **背景**:nnFormer repo 是 nnU-Net v1 fork(包命名空间冲突);MedNeXt 的 nnunet_mednext 依赖 nnunet(版本风险)。
- **选项**:(a) vendoring 非 MONAI 架构 + MONAI pip;(b) nnFormer 单 env;(c) 改名 hack
- **我的建议**:(a)。
- **决策**:**vendoring 非 MONAI 架构进框架** `framework/nets/<name>/`(复制 + import 改本地,不 pip install 它们的包):
  - MedNeXt:3 个架构文件(mednextv1/ 的 create_mednext_v1 + MedNextV1 + blocks,torch only)
  - SegFormer3D-aniso:1 文件(已修)
  - MONAI 系(SwinUNETR):`pip install monai`(不碰 nnunet,无冲突)
  - **nnFormer 已砍(不再需要 vendoring,省掉 fork 冲突 + timm 依赖)**
  - 彻底消除 nnunet 命名空间歧义 + nnunet 版本被动风险。vendored 代码进 git 可追溯。`source/` 已有完整源码挑文件。

### Q21. 预测/推理路径(非 nnU-Net 网络)✅

- **背景**:nnU-Net 有 `nnUNet_predict`(sliding window + 反预处理回原空间)。非 nnU-Net 网络要同等预测。
- **选项**:(a) base trainer 自带统一 predict(复用 nnU-Net v1 机制);(b) 每网络单独写
- **我的建议**:(a)。
- **决策**:**base trainer 自带统一 predict,复用 nnU-Net v1 的 sliding-window + 反预处理机制**(网络无关,
  喂 patch → net.forward → 拼 → argmax → resample 回原始 spacing → 存 nii.gz)。所有网络走同一套。
  **pin 参数**:sliding window overlap=**0.5**,TTA=**off**(Q11),predict batch=**2**,输出 argmax segmentation
  (反 resample 回原空间)喂 locked evaluator。

### Q22. GPU 分配 + 作业调度 ✅

- **背景**:8 网络(7×3D + 1×2D)各 125,000 iters。单卡训 500ep 约 1-1.5 天/网络。
- **选项**:(a) 单卡/网络(DDP off)+ 批量并发;(b) 多卡 DDP
- **我的建议**:(a)。
- **决策**:**单卡/网络(DDP off),DSUB `gpu=1` 批量并发**(几个空闲 GPU 就并发几个,7 个 3D 能并起来)。
  **节点池 = agent 编号 < 170**(170 及以后不用,含已排除的 174)。OOM 靠 Q17 grad accum 兜底,不切多卡
  (单卡最利于 Q10 state_dict_equal 确定性)。提交前先跑 1 个网络测 iter 速度估总 wall-clock。

### Q23. num_classes 约定 ✅

- **背景**:任务标签 0(bg)+1-9(9 前景)= 10 个标签值。CE+softmax 需覆盖 0-9。
- **选项**:(a) 统一 10(含 bg);(b) 统一 9(覆盖不了 head)
- **我的建议**:(a) 10。
- **决策**:**所有网络 out_channels = 10(含 bg,覆盖 labels 0-9),跟 nnU-Net v1 一致。** 核实确认:
  nnU-Net v1 `nnUNetTrainer.py` L367 `self.num_classes = plans['num_classes'] + 1`(plan num_classes=9 是前景计数,
  trainer +1 = 10 含 bg);Generic_UNet 输出 10 通道。smoke test 7 网络用 out_channels=10 全部 forward 出
  `[2,10,64,160,160]` ✓。无歧义。

### Q24. 随机种子策略 ✅

- **背景**:Q10 要求 state_dict_equal。无 fold seed(单固定 split,fold=0)。要做多 seed 平均 + worker seed。
- **选项**:(a) 3 base seed 平均 + worker seed 确定性派生;(b) 单 seed / 每 network 不同 seed
- **我的建议**:(a)。
- **决策**:**3 个 base seed = 20260520 / 20260521 / 20260522**(第一个延续 PACA swine CT)。无 fold seed(fold=0,
  fold_seed=base_seed)。每个网络跑 **3 次**(3 个 seed),test 上评 final checkpoint,**报告 mean±std 跨 3 次**。
  **worker seed 确定性派生**自 base_seed(`install_v1_determinism_patches`:train workers=`[base+1000+i]`、
  val=`[base+2000+i]`,每 worker 内 random/numpy/torch 全 seed,记 seed_policy JSON)。给定 base_seed 全随机性确定 → state_dict_equal。
  **代价:计算量 3×(共 15 个训练 job:4 网络×3 + 2D nnUNet×3)**;unpacked npy 只生成一次(3 seed 共用)。

### Q25. fp16 unpack 的 dtype 口径(澄清 Q9)✅

- **背景**:Q9 的"--fp16"混淆了 unpack dtype 和训练 AMP。标准 nnU-Net v1 `--unpack-data` 产 fp32 npy;`--fp16` 是训练 AMP flag。
- **选项**:(a) fp32 unpack + fp16 AMP;(b) fp16 unpack(自定义);(c) 全 fp32 训练
- **我的建议**:(a)。
- **决策**:**fp32 unpack(标准 `--unpack-data`,158 例 ~100-136GB,一次性,3 seed 共用)+ fp16 AMP 训练(`--fp16`)**。
  跟 PACA 完全一致(`train_paca_deterministic.py` 就是 unpack_data=True + fp16 + deterministic),
  state_dict_equal 已验证可达(fp16 AMP 的不确定性被 nnunetv1_compat patch 覆盖)。
  **澄清 Q9 笔误**:Q9 的"--fp16"指训练 AMP,unpack 是 fp32。**不一致标记 #3 据此关闭**;#1(unpack 是训练时
  `nnUNet_train --unpack-data`,不是 plan_and_preprocess)也据此明确。

### Q26. 统计显著性检验 ✅

- **背景**:7 网络 × 3 seed。网络间差异是否显著?
- **选项**:(a) 纯描述 mean±std;(b) 配对显著性检验;(c) 两者都报
- **我的建议**:(c)。
- **决策**:**两者都报** —— mean±std(跨 3 seed,描述整体水平)+ **配对 Wilcoxon signed-rank**(跨 39 test case,
  网络两两 C(7,2)=21 对,**Holm-Bonferroni 多重比较校正**)。per-case 指标先 3-seed 平均再做检验。
  Wilcoxon 是医学分割标配(非参数,对 Dice 偏态稳健)。scipy stats 实现,评估后加 stats 脚本。

### Q27. 文章主表结构 + 研究问题(延后到写作阶段)

- **背景**:spec 全是公平对比机制,但文章的科学 claim + 主表结构没定。
- **选项**:(i) 架构家族对比;(ii) +2D/3D 维度;(iii) 公平对比方法论;(iv) 其它
- **我的建议**:(i)+(ii),主表 8 方法 × per-class Dice/HD95 + mean±std + 显著性标记。
- **决策**:**暂缓,写作阶段再定**(用户 2026-06-22)。不影响实验执行(实验产出 per-class 指标,
  写作时按需组织表格)。默认主表:行=8 方法(7×3D + 2D nnUNet 分隔),列=per-class Dice/HD95 + mean,
  数值=3-seed mean±std + Wilcoxon 显著性标记(Q26)。

---

## ⚠️ 待澄清 / 前后不一致标记

1. **Q9 unpack 命令 vs 实际**:Q9/路线图写"`nnUNet_plan_and_preprocess -t 601 --unpack-data --fp16`",
   但**实际跑的 plan_and_preprocess(job 564154)没带 `--unpack-data`**(只产 npz+pkl)。nnU-Net v1 的
   unpack 是**训练时**(`nnUNet_train --unpack-data`)做的,不是 plan_and_preprocess 时。所以:
   - 路线图 step 1 的命令描述不准(把 unpack 混进了 plan_and_preprocess)。
   - unpacked npy **还没生成**,首次训练时才会产(~fp32 100GB+ 或 fp16,见 Q25)。待 Q25 定 dtype。
2. **路线图 plugin 列表过期**:路线图 step 3 只列了 nnU-Net + SwinUNETR 两个 plugin,但 Q1/Q14 已扩到
   7 个网络(nnU-Net/SwinUNETR/UNETR/SegResNet/MedNeXt/nnFormer/SegFormer3D-aniso)+ 2D nnUNet。要更新。
3. **Q9 说 fp16,实际机制不明**:~~见 Q25~~ **已关闭(Q25 决策:fp32 unpack + fp16 AMP 训练,Q9 的 --fp16 指训练 AMP)**。
4. **smoke test 用 num_classes=10,nnU-Net v1 用 9**:见 Q23,要 pin 统一。

### Q28. 目录结构 ✅

- **背景**:实现代码、config、eval、job 脚本要落到仓库,跟现有 data/docs/jobs 协调。
- **选项**:(a) 按功能分层(framework/configs/evaluation/jobs 子目录);(b) 扁平化 code/;(c) 其它
- **我的建议**:(a)。
- **决策**:**按功能分层**:
  - `framework/` —— 实现核心:`base_trainer.py`(网络无关 nnUNetTrainerV2 子类 + v1 数据管线 + 确定性 + grad accum + 2 forward 协议)、`registry.py`(name→build_fn+forward协议+家族+optimizer/loss)、`train.py`(config 驱动入口)、`predict.py`(统一 sliding-window,Q21)、`nets/`(插件 + vendored 架构:SwinUNETR `xxx.py` import monai;MedNeXt/SegFormer3D-aniso `xxx/` vendored 目录,Q20)
  - `configs/` —— 每 network 一个 yaml(arch 参数 + family + 超参;**seed 由 job 传入**,3 seed 复用一 config)
  - `evaluation/` —— `run_eval.py`(locked evaluator)+ `run_stats.py`(Wilcoxon+Holm,Q26)
  - `jobs/{train,predict,eval}/` —— 每 network×seed 的 DSUB 脚本
  - 结果(checkpoint/predictions)→ `data/nnunetv1/nnUNet_results/`(只 Huawei,.gitignore 拦)
  - 加网络 = 加 `framework/nets/` 插件 + 一个 `configs/*.yaml` + job 脚本,不动 base。

## 决策汇总(讨论后回填)

| Q | 决策 |
|---|---|
| Q1 模型阵容 | ✅ 通用多网络框架,网络不写死,可插拔注册 |
| Q2 SwinUNETR 初始化 | ✅ 所有网络统一 scratch,不预训练 |
| Q3 nnU-Net 对照基线 | ✅ Task601 重训,严格按 6:2:2 split,500 epochs(连带定下 Q5=500) |
| Q4 优化器 | ✅ 各自适配:CNN→SGD+poly,Transformer→AdamW |
| Q5 训练预算 | ✅ 锚定总迭代 125,000(500ep×250 iters/epoch,pin 250),含 2D nnUNet |
| Q6 patch/batch | ✅ 所有网络 patch_size 尽量跟 nnU-Net v1 plans 派生值一致 |
| Q7 deep supervision | ✅ 接受不对称,各网络用原版(nnU-Net 保留 DS,单输出网络单输出) |
| Q8 Task601 预处理来源 | ✅ 自己跑(已完成 DSUB 564154,158-based 隔离) |
| Q9 shared cache 复用 | ✅ 自己 unpack(--unpack-data --fp16),不复用 197-based cache |
| Q10 确定性杆 | ✅ 所有网络要求 state_dict_equal=true |
| Q11 评估口径 | ✅ 复用 locked evaluator + 条件类处理,预测 no TTA/ensemble/PP |
| Q12 数据增广一致性 | ✅ 所有网络锁死共用 nnU-Net v1 moreDA(全套,无额外/无类别专项) |
| Q13 前景过采样规则 | ✅ 共用 v1 dataloader(force-fg 0.33,无类别专项) |
| Q14 patch 整除约束 | ✅ 实测 v1 patch=[64,160,160] batch2,天然 32 整除,无需调整 |
| Q15 损失函数一致性 | ✅ 所有网络统一 nnU-Net v1 DC_and_CE_loss(DS 套 MultipleOutputLoss2) |
| Q16 模型选择指标 | ✅ 用 final checkpoint(第500ep),不选 best,val 仅监控(跟 nnU-Net 一致) |
| Q17 effective batch | ✅ effective=2,grad accum 兜底(语义 caveat 记录) |
| Q18 weight decay/scheduler | ✅ scheduler 跟家族,wd 固定小值不搜(CNN 3e-4/T 1e-5,lr 写死) |
| Q19 2D nnUNet 处理 | ✅ 单独类别参考基线,走 nnU-Net 2D pipeline,不进 3D 主对比表 |
| Q20 网络集成(nnFormer 冲突) | ✅ vendoring 非 MONAI 架构 + MONAI pip + timm |
| Q21 预测/推理路径 | ✅ 统一 predict 复用 nnU-Net v1 sliding-window,overlap 0.5,TTA off |
| Q22 GPU 分配 + 调度 | ✅ 单卡/网络 DDP off,批量并发,节点池 agent<170 |
| Q23 num_classes 约定 | ✅ 统一 10(含 bg),跟 nnU-Net v1 plan+1 一致 |
| Q24 随机种子 | ✅ 3 base seed(20260520/21/22)平均,无 fold seed,worker seed 确定性派生,共 24 job |
| Q25 fp16 unpack dtype | ✅ fp32 unpack(--unpack-data)+ fp16 AMP 训练,跟 PACA 一致;关闭不一致#3 |
| Q26 统计显著性检验 | ✅ mean±std + 配对 Wilcoxon(跨 39 case,Holm 校正) |
| Q27 主表 + 研究问题 | ⏸️ 延后到写作阶段(实验产出 per-class 指标,写作时组织表格) |
| Q28 目录结构 | ✅ 按功能分层:framework/(base_trainer+registry+train+predict+nets)+configs/+evaluation/+jobs 子目录 |

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
