# v1 输入一致性多网络公平对比 —— 正式 Spec

Status: IMPLEMENTATION-READY
Date: 2026-06-22
Source: 由 `v1-input-consistency-spec-draft.md`(Q1-Q28 决策)整理而来。

---

## 1. 目标

文章里所有分割模型读**同一份 nnU-Net v1 预处理 + 同一套 v1 数据/增强管线**,
差异只在模型架构,从而保证公平对比。选 v1 是因为 v2 有无法固定的随机性,
而 v1 的 `nnunetv1_compat` 能做到 `state_dict_equal=true`。

**唯一变量:架构。** 输入(数据/split/patch/增广/采样)、训练(预算/loss/optimizer 规则)、
预测(统一 sliding window)、评估(locked evaluator + 统计检验)、确定性(cudnn/seed)
全部锁死。

---

## 2. 最终阵容(4 个 3D + 1 个 2D)

| 网络 | 年份 | 路线 | optimizer 家族 |
|---|---|---|---|
| nnU-Net v1 | 2020 | 经典 CNN(数据驱动 auto-config) | CNN → SGD+poly |
| MedNeXt-S | 2023 MICCAI | 现代 ConvNet(ConvNeXt) | CNN → SGD+poly |
| SwinUNETR(V2) | 2022 | transformer 旗舰(Swin,MONAI) | Transformer → AdamW |
| SegFormer3D-aniso | 2024 CVPR-W | 最新 transformer(轻量高效) | Transformer → AdamW |
| 2D nnUNet | — | 2D 参考(单独类别,不进 3D 主对比表) | CNN → SGD+poly |

**训练 job:5 网络 × 3 seed = 15 个。**

---

## 3. 数据基建

### 3.1 数据源
- **197 例** labeled swine CT(9 类胴体分割 + bg),2 个 source:
  - **label class**:`0` bg / `1` front / `2` middle / `3` end / `4` left_kidney / `5` right_kidney / `6` testis / `7` thoracic_cavity / `8` abdominal_and_pelvic_cavity / `9` head
  - 条件类:`6 testis` 仅 TB 有,`9 head` 仅 HZAU 有;其余 7 类两 source 都有
  - HZAU 93 例(Yorkshire,阉猪,head-present/testis-absent)
  - TB 104 例(4 品种:Yorkshire/Landrace/Pietrain/Duroc 各 26,公猪,head-absent/testis-present)
- 软连接到 `/home/hzau/whcs-share37/liuyangfan/nnunet_medsam_semisup/data/labeled_197/`

### 3.2 Split(固定 6:2:2,已搭建+审计)
- train 120 / val 38 / test 39
- TB 按品种分层(每品种 16/5/5),HZAU 纯随机,seed=42
- canonical: `data/splits/split_manifest.csv`
- 物化: `data/{train,val,test}/{images,labels}/` 软连接(Huawei)
- **test 冻结**(只用于最终评估);val 仅监控(Q16:用 final checkpoint 不选 best)

### 3.3 Task601 v1 预处理(已完成)
- `nnUNet_plan_and_preprocess -t 601`(DSUB 564154 SUCCEEDED)
- 3d_fullres(stage1):patch `[64,160,160]`,batch 2,spacing `[5, 0.97656, 0.97656]`
  (不 resample,各向异性 pooling `[[1,2,2],[1,2,2],[2,2,2],…]`)
- 产物:`data/nnunetv1/nnUNet_preprocessed/Task601_Article622_Carcass9Class/`
  - `nnUNetData_plans_v2.1_stage1/`(316 文件 = 158×npz+pkl)← **3d 训练用**
  - `nnUNetData_plans_v2.1_stage0/`、`_2D_stage0/`、`gt_segmentations/`(158)
  - `nnUNetPlansv2.1_plans_3D.pkl`、`_2D.pkl`、`dataset.json`
- raw: `data/nnunetv1/nnUNet_raw_data/Task601_Article622_Carcass9Class/`
  (imagesTr 158 + imagesTs 39 + splits_final.{json,pkl})
- **自定义 split 待放**:把 `splits_final.pkl`(单 fold train=120/val=38)复制进
  preprocessed 目录覆盖(训练前)。
- **unpacked npy 待生成**:首次训练 `--unpack-data` 时产出(fp32,~100-136GB,3 seed 共用)。

### 3.4 数据布局(Huawei)
```
swine-CT-article/data/nnunetv1/
├── nnUNet_raw_data/Task601_Article622_Carcass9Class/  # raw(imagesTr/Ts/labelsTr/Ts/dataset.json/splits)
├── nnUNet_preprocessed/Task601_Article622_Carcass9Class/  # plan + stage npz + pkl + gt_seg
├── nnUNet_results/  # 训练 checkpoint + predictions(不进 git)
└── build_task601.py  # 可复现搭建脚本
```

---

## 4. 统一环境(Huawei nnunetv1)

参考脚本:`swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh`

**CUDA stack(module load):**
- `compilers/gcc/9.3.0` + `compilers/cuda/11.8.0` + `libs/cudnn/8.8.1_cuda11` + `libs/nccl/2.16.5-1_cuda11.8`
- `libs/openblas/0.3.18_kgcc9.3.1`(部分节点缺 `libopenblas.so.0`,显式 load)

**关键路径:**
- `NNUNETV1_ENV_ROOT = swine_ct_autonomous_discovery/envs/nnunetv1`(python 3.10 + torch 2.4.1+cuda118)
- `NNUNETV1_COMPAT_ROOT = swine_ct_autonomous_discovery/scripts/nnunetv1_compat`(确定性套件,PYTHONPATH)

**Task601 数据根(SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1 覆盖):**
- `nnUNet_raw_data_base = swine-CT-article/data/nnunetv1`
- `nnUNet_preprocessed = swine-CT-article/data/nnunetv1/nnUNet_preprocessed`
- `RESULTS_FOLDER = swine-CT-article/data/nnunetv1/nnUNet_results`

**Hack:**
- sklearn `libgomp` `LD_PRELOAD`(符号冲突)
- `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS = 16`
- `pip install monai`(SwinUNETR)+ `pip install einops`(SegFormer3D)

**所有 v1 job 前必须:** source setup → `swct_nnunetv1_preflight` 通过 → 再执行命令。

---

## 5. 公平性协议(全网络锁死)

### 5.1 输入侧

| 维度 | 锁定值 |
|---|---|
| 数据 + split | Task601,6:2:2,同源同预处理 |
| patch / batch | `[64,160,160]` + effective batch 2(grad accum 兜底) |
| 数据增广 | nnU-Net v1 **moreDA** 全套(SpatialTransform 旋转/缩放/弹性 + GaussianNoise + GaussianBlur + BrightnessMultiplicative + Brightness 加性偏移 + ContrastAugmentation + SimulateLowResolution + GammaTransform 反转/非反转 + Mirror),不额外加、不做类别专项、seed 固定 |
| 前景采样 | `oversample_foreground_percent=0.33`,sample0=random crop + sample1=forced-foreground crop(从 .pkl class_locations),case 采样均匀随机,禁类别专项/hard mining |

### 5.2 训练侧

| 维度 | 锁定值 |
|---|---|
| 初始化 | 全 scratch(无预训练) |
| 预算 | **125,000 iters**(500 ep × 250 iters/epoch,pin `num_iterations_per_epoch=250`) |
| loss | nnU-Net v1 `DC_and_CE_loss`(SoftDice batch_dice + 加权 CE)。DS 网络(nnU-Net/MedNeXt)走 `MultipleOutputLoss2` 多尺度加权;单输出网络(SwinUNETR/SegFormer3D-aniso)直接算。底层同一函数同一参数。DS 权重用 nnU-Net v1 默认(`deep_supervision_scales` 从 plans 派生) |
| checkpoint | **final**(第 500 ep,LR 衰减到底),不选 best。val 仅监控(频率跟 nnU-Net v1 默认:每 epoch 50 val iters) |
| deep supervision | 各网络原版:nnU-Net/MedNeXt 保留 DS(多尺度输出);SwinUNETR/**SegFormer3D-aniso 单输出**(SegFormer3D 原版无 DS heads)。base trainer 按注册的 forward 协议分支,不改架构 |
| effective batch | 2 patches(`physical × grad_accum = 2`;grad accum 的 batch_dice 语义微偏差已记录) |
| num_workers | 跟 nnU-Net v1 默认(训练 8 / val 4,pin 固定不搜) |

### 5.3 确定性

| 维度 | 值 |
|---|---|
| 复现杆 | 所有网络要求 `state_dict_equal=true` + `optimizer_state_dict_equal=true` |
| cudnn | `deterministic=True`,`benchmark=False`(不额外关 TF32 / 不加 deterministic_algorithms,跟 nnU-Net v1 一致) |
| seed | 3 个 base seed:**20260520 / 20260521 / 20260522**(第一个延续 PACA) |
| fold seed | 无(单固定 split,fold=0,fold_seed=base_seed) |
| worker seed | 确定性派生(train `[base+1000+i]`/val `[base+2000+i]`)via `install_v1_determinism_patches` |
| PYTHONHASHSEED | 设为 base_seed |
| fp16 | AMP(`--fp16`),+ 确定性 patch = 可复现(PACA A100 验证 state_dict_equal) |

### 5.4 预测侧

| 维度 | 锁定值 |
|---|---|
| 方式 | 统一 sliding window(复用 nnU-Net v1 预测机制,网络无关) |
| overlap | 0.5 |
| TTA | off(no mirror) |
| ensemble | none |
| post-processing | disabled |
| softmax | 不保存 |
| predict batch | 2 |
| 输出 | argmax segmentation(resample 回原始 spacing)→ nii.gz |

### 5.5 评估侧

| 维度 | 锁定值 |
|---|---|
| evaluator | locked evaluator(`evaluate_swine_ct.py`) |
| 指标 | Dice + HD95(per class) |
| 条件类 | head 只在 HZAU 上评、testis 只在 TB 上评;head-absent FP 单独统计 |
| num_classes | 10(含 bg;nnU-Net v1 plan num_classes=9 前景,trainer +1=10) |
| 统计检验 | mean±std(跨 3 seed)+ 配对 **Wilcoxon signed-rank**(per-case mean Dice **先 3-seed 平均后**做检验,跨 39 test case,4 个 3D 网络两两 C(4,2)=6 对,Holm-Bonferroni 校正;2D nnUNet 不进主检验;HD95 仅描述性 mean±std 不做检验) |

---

## 6. 网络架构 + 训练超参(写 configs/*.yaml 的参考)

### 6.1 nnU-Net v1(Generic_UNet)
- 架构:全从 Task601 plans 自动(base_features / num_pool / conv_kernels / pool_kernels)。
- DS:原生多尺度(走 MultipleOutputLoss2)。
- optimizer:SGD lr=0.01, momentum=0.99, wd=3e-4, poly lr(nnU-Net v1 原版默认)。

### 6.2 SwinUNETR(MONAI,`pip install monai`)
- `in_channels=1, out_channels=10, spatial_dims=3`
- `feature_size=48, depths=[2,2,2,2], num_heads=[3,6,12,24], window_size=7`
- `norm_name="instance", drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0`
- `use_v2=True`(SwinUNETR-V2,MICCAI 2023)
- DS:单输出(base trainer 走 single-output DC_and_CE)
- optimizer:AdamW lr=4e-4, wd=1e-5, warmup_cosine(warmup_ratio=0.05, min_lr_ratio=0.01, min_lr_floor=1e-6)

### 6.3 MedNeXt-S(vendored: mednextv1/ 3 文件)
- `num_input_channels=1, num_classes=10`
- `model_id="S"`(n_channels=32, block_counts=[2,2,2,2,2,2,2,2,2] 自动)
- `kernel_size=3, exp_r=4, norm_type="group"`
- `deep_supervision=True, do_res=True, do_res_up_down=True`(paper 原版)
- DS:多尺度(走 MultipleOutputLoss2)
- optimizer:SGD lr=0.01, momentum=0.99, wd=3e-4, poly lr(跟 nnU-Net v1 一致;MedNeXt 论文用的就是 SGD)

### 6.4 SegFormer3D-aniso(vendored: segformer3d_aniso.py 1 文件)
- `in_channels=1, num_classes=10`
- 全 repo 默认:sr_ratios=[4,2,1,1], embed_dims=[32,64,160,256], patch_kernel_size=[7,3,3,3],
  patch_stride=[4,2,2,2], patch_padding=[3,1,1,1], num_heads=[1,2,5,8], depths=[2,2,2,2],
  mlp_ratios=[4,4,4,4], decoder_head_embedding_dim=256, decoder_dropout=0.0
- aniso 修复:去 `@torch.jit.script cube_root`,显式 (D,H,W) 穿线(attention/MLP/decoder 不变)
- DS:单输出(SegFormer3D 原版无 DS)
- optimizer:AdamW lr=4e-4, wd=1e-5, warmup_cosine(同 SwinUNETR)

### 6.5 2D nnUNet(单独类别参考)
- 架构:从 2D plans 自动(`_2D_stage0` 已生成)
- 用我们的 split(同 train/val/test case)+ 125,000 iters(同预算)
- optimizer:SGD lr=0.01, wd=3e-4, poly lr(nnU-Net v1 默认)
- **不进 3D 主对比表**(单独报告)
- **确定性注入**:2D nnUNet 走原生 `nnUNet_train 2d`(不走 base trainer),seed 通过
  `train_paca_deterministic.py` 或等价 wrapper 传入(`install_v1_determinism_patches` 在
  trainer 初始化前 monkey-patch 全局生效,不需要 base trainer 子类)。3 seed 同 3D。

---

## 7. 网络集成方式(vendoring + MONAI)

| 网络 | 集成方式 |
|---|---|
| nnU-Net v1 | env 自带(nnunet 包) |
| SwinUNETR | `pip install monai`,`from monai.networks.nets import SwinUNETR` |
| MedNeXt-S | **vendored** 3 架构文件进 `framework/nets/mednext/`(不装 nnunet_mednext 包) |
| SegFormer3D-aniso | **vendored** 1 文件进 `framework/nets/segformer3d/` |
| 2D nnUNet | env 自带(`nnUNet_train 2d`) |

base trainer 子类 `nnUNetTrainerV2`,固定 v1 数据管线(moreDA + force-fg 采样)+ 确定性 patch +
grad accum;`run_iteration` 按网络注册的 forward 协议(DS 列表 / 单输出)分支;
`initialize_network` 按注册的 build_fn 构造网络;`initialize_optimizer_and_scheduler` 按注册的家族构造。

---

## 8. 目录结构

```
swine-CT-article/
├── CLAUDE.md
├── data/                    # 数据基建
│   ├── images/labels/       # Huawei 软连接(197 例)
│   ├── train/val/test/      # Huawei 软连接(按 split)
│   ├── manifests/           # case_metadata.csv 等
│   ├── splits/              # split_manifest.csv + make_split.py + materialize_symlinks.sh
│   ├── nnunetv1/            # Task601 raw + preprocessed + results
│   └── README.md
├── framework/               # 实现代码(tracked)
│   ├── base_trainer.py      # 网络无关 nnUNetTrainerV2 子类
│   ├── registry.py          # 网络工厂(name→build_fn+forward协议+家族+optimizer)
│   ├── train.py             # config 驱动训练入口
│   ├── predict.py           # 统一 sliding-window predict
│   └── nets/                # 网络插件 + vendored 架构
│       ├── nnunet.py        # nnU-Net v1 插件
│       ├── swinunetr.py     # MONAI SwinUNETR 插件
│       ├── mednext/         # vendored MedNeXt(3 文件)
│       └── segformer3d/     # SegFormer3D-aniso(1 文件)
├── configs/                 # 每 network 的 yaml(seed 由 job 传入)
│   ├── nnunet_v1.yaml
│   ├── swinunetr.yaml
│   ├── mednext_s.yaml
│   ├── segformer3d.yaml
│   └── nnunet_2d.yaml
├── evaluation/              # 评估 + 统计
│   ├── run_eval.py          # locked evaluator
│   └── run_stats.py         # Wilcoxon + Holm
├── jobs/                    # DSUB 作业脚本
│   ├── preprocess/          # plan_and_preprocess(已完成)
│   ├── smoke/               # patch 兼容性 smoke test(已完成)
│   ├── train/               # 每 network×seed 训练 job
│   ├── predict/             # 每 network×seed 预测 job
│   └── eval/                # 评估 + 统计 job
├── docs/                    # spec + handoff
│   ├── v1-input-consistency-spec.md      # 本文件
│   ├── v1-input-consistency-spec-draft.md # Q1-Q28 决策记录
│   └── SWCT06042040-...-HANDOFF.md       # 历史 nnU-Net v1 实验交接
└── source/                  # [gitignored] 第三方 repo 参考
    ├── MedNeXt/
    ├── SegFormer3D/
    └── smoke_test_patch.py
```

**加网络 = 加 `framework/nets/` 插件 + `configs/*.yaml` + job 脚本,不动 base。**

---

## 9. GPU 分配 + 调度

- **单卡/网络**(DDP off,最利于确定性;OOM 靠 grad accum 兜底)
- **批量并发**(几个空闲 GPU 就并发几个)
- **节点池:agent 编号 < 170**(170 及以后不用,含已排除的 174)
- 每 job:`#DSUB -R "cpu=16;mem=64000;gpu=1" -pn !whshare-agent-174`
- 提交前先跑 1 个网络测 iter 速度估总 wall-clock

---

## 10. 实现阶段(Implementation Stages)

### Stage 0:数据基建(✅ 已完成)
- [x] Task601 raw 搭建(build_task601.py,158 Tr + 39 Ts 软连接)
- [x] split 生成(make_split.py,seed 42,6:2:2)
- [x] split 物化(data/{train,val,test}/ 软连接)
- [x] Task601 v1 plan_and_preprocess(DSUB 564154,patch [64,160,160])
- [x] patch 兼容性 smoke test(DSUB 564162,4 网络 forward 通过)

### Stage 1:放置自定义 split(待做)
- [ ] 把 `splits_final.pkl`(单 fold train=120/val=38)从 raw Task601 目录复制进
      `nnUNet_preprocessed/Task601_Article622_Carcass9Class/`
- **输出**:preprocessed 目录有我们的单 fold split(覆盖任何自动 5-fold)
- **依赖**:Stage 0 ✅

### Stage 2:框架骨架(待做)
- [ ] `framework/base_trainer.py`:子类 `nnUNetTrainerV2`,固定 v1 数据管线(moreDA + force-fg)
      + 确定性 patch(`install_v1_determinism_patches`)+ grad accum 支持 + `run_iteration` 两种 forward
      协议(DS 列表 / 单输出)+ `initialize_network` / `initialize_optimizer_and_scheduler` 按注册信息
- [ ] `framework/registry.py`:name → {build_fn, forward_protocol(DS/single), family(CNN/Transformer),
      optimizer override}
- [ ] `framework/train.py`:读 config + seed,构建 trainer,跑 500 ep
- [ ] `framework/predict.py`:统一 sliding window(overlap 0.5, TTA off,反预处理回原空间)
- **输出**:可注册网络的 base trainer + 训练/预测入口
- **依赖**:Stage 1

### Stage 3:网络插件 + configs(待做)
- [ ] `framework/nets/nnunet.py`:nnU-Net v1 plugin(Generic_UNet,DS,CNN→SGD)
- [ ] `framework/nets/swinunetr.py`:MONAI SwinUNETR plugin(use_v2=True,单输出,Transformer→AdamW)
- [ ] `framework/nets/mednext/`:vendored MedNeXt 3 架构文件 + plugin(DS,CNN→SGD)
- [ ] `framework/nets/segformer3d/`:vendored SegFormer3D-aniso + plugin(单输出,Transformer→AdamW)
- [ ] `configs/*.yaml`:每网络一个(seed 由 job 传入)
- [ ] 每网络 `[2,1,64,160,160]` forward smoke test(base trainer + 网络)
- **输出**:4 个可训练的 3D 网络 plugin + config
- **依赖**:Stage 2 + vendored 源码(`source/`)

### Stage 4:训练(待做,15 个 job)
- [ ] 提交 4 网络 × 3 seed = 12 个 3D 训练 job(DSUB,单卡,agent<170)
- [ ] 2D nnUNet × 3 seed = 3 个 job(`nnUNet_train 2d`,独立 2D plans)
- [ ] 每个 job early check(`djob` 确认 RUNNING 不 FAILED)
- [ ] 每网络确定性 smoke(2 run 同 seed 比 state_dict_equal)——至少对 1 个 seed 做
- **输出**:每个 (网络, seed) 的 `model_final_checkpoint.model`
- **依赖**:Stage 3 + Stage 1(split 放好)

### Stage 5:预测(待做,15 个 job)
- [ ] 每网络每 seed:用 `predict.py`(sliding window)对 test 39 预测
- [ ] 输出 argmax segmentation(resample 回原空间)→ nii.gz
- **输出**:15 组 test 39 predictions(每组 39 个 nii.gz)
- **依赖**:Stage 4(final checkpoint)

### Stage 6:评估 + 统计(待做)
- [ ] 每组 predictions 过 locked evaluator(Dice + HD95 per class + 条件类)
- [ ] 跨 3 seed 聚合 mean±std
- [ ] 配对 Wilcoxon signed-rank(39 case,网络两两,Holm 校正)
- [ ] 汇总表
- **输出**:最终结果表(4 网络 × per-class Dice/HD95 ± std + 显著性)
- **依赖**:Stage 5

### Stage 7:确定性审计(不做)
- 确定性设置(cudnn.deterministic + seeds + worker seeds)在 base trainer 里固定生效,
  不单独跑 determinism smoke 验证。PACA 已在 nnU-Net v1 + A100 上验证过 state_dict_equal=true。

---

## 11. 决策溯源

完整 Q1-Q28 决策记录见 `v1-input-consistency-spec-draft.md`。本 spec 是其整理版。

**关键决策一览:**
- Q1 通用多网络框架 / Q2 全 scratch / Q3 Task601 重训 / Q4 optimizer 按家族
- Q5 125,000 iters / Q6 patch [64,160,160] / Q7 DS 不对称原版 / Q8 自己预处理 / Q9 自己 unpack
- Q10 state_dict_equal / Q11 locked evaluator / Q12 moreDA / Q13 force-fg 0.33
- Q14 实测 patch 兼容 / Q15 DC_and_CE / Q16 final checkpoint / Q17 effective batch 2
- Q18 wd 不搜 / Q19 2D 单独类别 / Q20 vendoring / Q21 统一 predict / Q22 单卡并发
- Q23 num_classes=10 / Q24 3 seed / Q25 fp32 unpack + fp16 AMP / Q26 Wilcoxon
- Q27 研究问题延后 / Q28 目录结构
