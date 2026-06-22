# Spec Draft — Kidney LR-mirror 关闭 ablation(2D nnUNet + SwinUNETR)

Status: **问题清单 + 待决项**,尚未决策。本文目的是把"要落地得到结论必须考虑的
所有问题"摆全,后续基于此整理成正式 spec。

Date: 2026-06-23

关联:
- 主对比 spec:[`v1-input-consistency-spec.md`](v1-input-consistency-spec.md)
- 主对比决策记录:[`v1-input-consistency-spec-draft.md`](v1-input-consistency-spec-draft.md)
- 先验证据:[`SWCT06042040-HUAWEI-V1-KIDNEY-MIRROR-CLASSIFIER-HARDZERO-HANDOFF.md`](SWCT06042040-HUAWEI-V1-KIDNEY-MIRROR-CLASSIFIER-HARDZERO-HANDOFF.md)
- 实验跟踪:GitHub issue #1

---

## 0. 一句话目标 + scope 声明

在 issue #1 主对比中**分割质量最好的两个网络(nnU-Net 2D、SwinUNETR)**上,
把训练增强里的 **LR(left-right)axis mirror 关闭**(或条件关闭),验证能否改善
kidney 的 tail 指标(Dice P10 / HD95 P90)与左右混淆(swap rate / LP-Dice gap),
同时确认对其它 class 无副作用。

**scope 严格声明(必须写进正式 spec):**
- framework = swine-CT-article 的 `framework/`(Task601, 3-seed, locked evaluator);
- 网络 = nnU-Net 2D + SwinUNETR 两个;
- **不**与 handoff 文档里的 Huawei v1(Task520 r20, fold4)结果混比 —— 那是另一个
  task / 另一个 framework / 单 fold,只能作为**先验与方法论参考**,不能直接外推;
- **不**在本实验里碰 head/testis 条件类 gate(那是 handoff 实验 C 的方向)。

## 1. 动机:为什么挑这两个网络、为什么现在做

issue #1 最终结果里,kidney 相关数据(3-seed):

| 网络 | L/R kidney Dice | L-kidney HD95 P90 | R-kidney HD95 P90 | kidney swap rate | 混淆 case 比例 |
|---|---|---|---|---|---|
| SwinUNETR | **0.934 / 0.935**(最高) | 26.6 mm | **97.1 mm** | **2.3%**(最低) | 52 ± 8% |
| nnU-Net 2D | 0.926 / 0.923 | **116.3 mm** | **120.3 mm** | 4.2% | 40 ± 6% |

观察:
- **SwinUNETR**:kidney 平均 Dice 最高、swap 最低,但 **HD95 P90 仍有 26–97 mm 的
  尾部**,且 **52% 的 case 出现左右混淆** —— 平均好,tail/混淆仍有空间。
- **nnU-Net 2D**:Dice 不差,但 **HD95 P90 高达 116–120 mm**(极差 tail),且仍有
  40% case 混淆。tail 极可能是少数 case 左右肾彻底崩。

handoff 在 Huawei v1 fold4 上的先验:关掉 LR-axis mirror 把 kidney HD95 P90 从
`left=64.7 / right=46.6` 砸到 `left≈3.1 / right≈2.8`,Dice 从 ~0.93 提到 ~0.97。
**强烈提示 LR mirror 是 kidney tail 的重要风险因素** —— 但那是在另一个 framework
/ task / 单 fold / 单网络(v1 3D)上,能否迁移到 2D 与 SwinUNETR 是本实验要回答的。

> ⚠️ 注意:这是个**真问题**,不是"关了肯定好"。SwinUNETR 的全局注意力可能已经
> 隐式处理了左右语义(它的 swap 本来就最低),关 mirror 也许没收益甚至变差;2D 逐
> slice 处理,关 LR flip 对左右混淆的机制和 3D 不同。**正负结果都是有效结论。**

---

## 2. 待澄清事实(audit,必须在动手前完成)

### Q1. baseline 这两个网络现在到底有没有开 LR-axis mirror?

- **SwinUNETR**:几乎确定**开了**。`MultiNetworkTrainer.initialize()`(base_trainer.py:197)
  调父类 `setup_DA_params()`,**代码里没有任何对 `mirror_axes` 的覆盖** → 继承
  nnU-Net v1 3d_fullres 默认 `mirror_axes=(0,1,2)`,随后 `get_moreDA_augmentation`
  (base_trainer.py:230)据此构建 `MirrorTransform`。
  - audit 做法:在 SwinUNETR baseline 的 `fold_0/training_log*.txt` 里 grep
    `mirror_axes`,或加一行 `print(self.data_aug_params["mirror_axes"])` 跑 smoke。
    预期 `(0, 1, 2)`。
- **nnU-Net 2D**:**待确认**。2D 走的是 `train_paca_deterministic.py --network 2d`
  这个**不在本仓库**的 wrapper(见 Q3),其 mirror_axes 由原生 `nnUNetTrainerV2`
  的 2d 分支决定(原始 nnU-Net v1 对 2d 默认 `(0,1)`)。必须打开那个 wrapper 确认。
  - **若 baseline 2D 根本没开 LR mirror → 本实验对 2D 无意义**,需重新选题。

### Q2. LR 解剖轴在 patch 里到底是第几个 axis?(axis audit)

handoff 在 Task520 上 `confirmed_lr_axis=2`。但本实验是 **Task601**,patch 不同
(3d_fullres stage1 = `[64,160,160]`),**必须重新 audit,不能沿用 2**。

- 3D(SwinUNETR):patch 通常 `(z, row, col)` = `(axis0, axis1, axis2)`,LR 一般是
  col = axis2,但要拿一个**左右不对称、且 left/right kidney label 都在**的 HZAU 或
  TB case 验证:沿某 axis 翻转 label 后,`left_kidney(class4)` 与 `right_kidney(class5)`
  的体素位置互换 → 那个 axis 就是 LR。
- 2D:数据是逐 slice 的 `(c, row, col)`,mirror axes `(0,1)` 对应 row/col。LR 是其中
  一个,但**还要分清是哪个 slicing 方向的 slice** —— axial slice 的左右才是解剖左右,
  sagittal/coronal slice 的"左右翻转"解剖意义不同。2D nnU-Net 把所有 slice 混采,
  mirror 翻的是 slice 内的轴,需要确认这个轴对 axial slice 而言确实是 LR。
- audit 产物:写一个 `tools/audit_lr_axis.py`,输出 `confirmed_lr_axis_3d` 与
  `confirmed_lr_axis_2d` + witness md(对标 handoff 的 `lr_axis_audit` 报告)。

### Q3. 2D 的训练入口在哪?改 mirror 改哪个文件?(工程拦路虎)

- 已确认:`framework/train.py:15-16` 注释 "This module is 3D-only. The 2D nnUNet
  reference trains via `train_paca_deterministic.py --network 2d` wrapper"。
- 全仓 grep:**`train_paca_deterministic.py` 不在 swine-CT-article 仓库**。它应该
  在 Huawei 端的另一个 workspace(候选:`swine_ct_autonomous_discovery`、或某个
  `paca` 训练脚手架)。issue #1 评论 4 记的 2D 权重路径是
  `data/nnunetv1/v1_comparison_2d_root/seed<seed>/runs/nnunet_2d__seed<seed>/`,
  反推 wrapper 位置。
- **决策点(必须先定)**:
  - (a) 找到那个 wrapper,在它里面改 mirror_axes(保持和 baseline 同源,但改动在本
    仓库之外,破坏"本地 canonical"原则);
  - (b) 把 2D 也纳入 `framework/`(写一个 2D 的 MultiNetworkTrainer 路径 / 或让
    `train.py` 支持 `--network-dim 2d`),**重训 2D baseline 对齐** → 工作量大,但
    本仓库自洽、可复现;
  - (c) 本实验**先只做 SwinUNETR**(3D,落点清晰),2D 留到入口问题解决后补。
  - [建议] 先 (c) 拿 SwinUNETR 结论,同时并行排查 (a) 的 wrapper 位置;2D 是否值得
    为它做 (b),取决于 SwinUNETR 的结果是否成立。

---

## 3. 实验设计决策

### Q4. full-disable 还是 conditional mirror,还是都做?

handoff 两个方案在 fold4 上几乎等价(HD95 P90 都降到 ~3/2.3)。差异在工程与设计:
- **full-disable(LR-safe)**:直接把 LR axis 从 `mirror_axes` 移除。改动 = 一行
  `data_aug_params["mirror_axes"] = tuple(a for a in (...) if a != lr_axis)`。
  风险最低、最易复现,但其它 class 也失去了 LR mirror 的泛化增益。
- **conditional**:仅当 patch 含 kidney class(4/5)时关 LR mirror,其余 sample 保留。
  设计上更优(保护 kidney + 保留其它 class 增益),但 `get_moreDA_augmentation`
  内部固定构建 `MirrorTransform`,要做 per-sample 条件翻转,得 fork / monkey-patch
  该函数或自定义 `MirrorTransform` 子类 —— **工程量大、易引入 bug、且偏离 baseline
  管线**(破坏"唯一变量"的公平性叙事)。
- [建议] **第一阶段只做 full-disable**(2 网络 × 3 seed = 6 job,低风险、handoff 已
  证有效)。若 full-disable 对 kidney 有效**且**对其它 class 有可见副作用,再上
  conditional 作为第二阶段 ablation。不要一开始就上 conditional。

### Q5. 只做这 2 个网络,还是要回扫全部 5 网络?

- 用户已明确:**先在最好的 2 个上做 pilot**。合理 —— 避免一开始就 5×3=15 job
  盲跑。
- [建议] 正式 spec 里写明"分阶段":Phase 1 = SwinUNETR(+ 2D 若 Q3 解决);**若结论
  成立**,Phase 2 扩到其余 3 个网络(nnU-Net v1 / SegFormer3D / MedNeXt-S)做完整
  ablation,才能支撑论文里"LR mirror 对 kidney tail 的影响是架构无关的"这类claim。
  只 2 个网络不足以泛化。

### Q6. 改动的代码落点(SwinUNETR 路径)

[建议] 在 `MultiNetworkTrainer` 加一个 keyword-only 参数 `lr_mirror_mode:
Literal["full","no_lr"] = "full"`,在 `initialize()` 里 `setup_DA_params()` 之后、
`get_moreDA_augmentation()` 之前:

```python
if self.lr_mirror_mode == "no_lr":
    before = self.data_aug_params["mirror_axes"]
    self.data_aug_params["mirror_axes"] = tuple(
        a for a in before if a != self.lr_axis
    )
    self.print_to_log_file(f"LR-mirror disabled: {before} -> {self.data_aug_params['mirror_axes']}")
```

- `self.lr_axis` 来自 Q2 的 audit 结果(写进 config / trainer 常量,不要硬编码 2)。
- baseline 默认 `lr_mirror_mode="full"`,**保证已训完的 baseline 完全不受影响**。
- audit + smoke(打印改前/改后 mirror_axes + 确认 forward 正常)必须在提交前 PASS。

### Q7. 变量控制:与 baseline 逐项对齐清单(公平性生命线)

正式 spec 必须有一张"对齐表",逐项确认 no_lr 实验与 issue #1 baseline **除 mirror
外完全一致**:

| 维度 | 必须一致 | 备注 |
|---|---|---|
| split / fold | Task601, fold 0, train=120/val=38 | 复用 `place_split.py` 产物 |
| seeds | 20260520 / 21 / 22(3 个) | **必须复用**,才能 per-case 配对 |
| patch | `[64,160,160]` + effective batch 2 | 不变 |
| 其它增强 | moreDA 全套(除 mirror) | 只动 mirror_axes |
| sampling | force-fg 0.33 | 不变 |
| budget | 500ep × 250 = 125 000 iters | 不变 |
| loss | DC_and_CE_loss(DS 权重同 baseline) | 不变 |
| optimizer | SwinUNETR=AdamW(4e-4, warmup-cosine) | 不变 |
| checkpoint | final only | 不变 |
| determinism | cudnn.deterministic=True, benchmark=False | 不变 |
| predict | sliding window 0.5, TTA off, no PP | 不变 |
| eval | locked evaluator + kidney_swap_eval | 与 issue #1 同口径 |

> 任何一项偏移都会让"关 mirror"的归因失效。正式 spec 要逐行可勾选。

---

## 4. 评估与统计

### Q8. 报告哪些指标?

**kidney 重点(必须):**
- left/right kidney:mean Dice、Dice P10、mean HD95、HD95 P90、FP/GT、FN/GT;
- 左右混淆:swap rate、LP-Dice gap、混淆 case 比例(per-seed mean±std)—— 复用
  `kidney_swap_eval.py`,口径与 issue #1 完全一致。

**副作用(必须,确认无损):**
- 全 9 class 的 mean Dice / IoU / HD95,逐 class 给 baseline vs no_lr 的 Δ;
- 条件类(head/testis)absent-FP:确认关 mirror 没有让幻觉变多;
- 特别盯 front(handoff 里 v1 关 mirror 后 front 有极轻微 +Δ,可能是 head FP 概率
  回流)—— 本实验无 head gate,但仍值得看。

### Q9. 怎么做统计检验?

- **配对对象**:每个 test case、每个 seed 的 per-case kidney Dice/HD95,baseline vs
  no_lr **同一 (case, seed) 配对** —— 这就是为什么 Q7 要求 seed 必须复用。
- **检验**:Wilcoxon signed-rank(双侧),3-seed 平均后逐 case 配对,或逐 seed 配对
  后再用 Holm-Bonferroni 校正(正式 spec 要定死一种,建议 per-case 3-seed 平均后
  配对,与 issue #1 的 Wilcoxon 口径一致)。
- 样本量:39 cases × 3 seed。比 handoff 的单 fold 39 cases 更强。
- 报告:p 值 + Holm 校正后显著性 + 效应量(kidney HD95 P90 的绝对降幅)。

### Q10. 阴性 / 混合结果怎么处理?

- **必须预设**:若 SwinUNETR 关 mirror 后 kidney 无改善或变差,这**不是失败**,而
  是"transformer 全局注意力已隐式处理左右语义、显式关 mirror 无额外收益"的有效
  结论,照实写进论文。
- 若 2D 改善、SwinUNETR 不改善 → 说明 LR mirror 对 kidney tail 的影响**架构相关**,
  同样是有价值的发现。
- 正式 spec 要写明"双向假设",避免 confirmation bias 只报喜。

---

## 5. 工程与调度

### Q11. 训练规模 / 节点 / 预算

- Phase 1:SwinUNETR × 3 seed = 3 job(若 2D 入口解决,+3 = 6 job)。
- 单 job ≈ baseline 训练时长(SwinUNETR 3D,125k iters)。
- 节点约束:`agent<170`、**不**用 `whshare-agent-174`(CLAUDE.md);单卡并发。
- 提交后按 CLAUDE.md 做 **early-runtime 检查**:dsub 后 1–2min `djob <id>`,FAILED
  立刻读 `.err` 修(常见:flag 拼写、缺 module load、缺执行权限);长任务每 ~10min
  复查。

### Q12. 产物命名与路径(绝不覆盖 baseline)

[建议] 命名后缀 `__nolr`:
- checkpoint:`data/nnunetv1/v1_comparison/swinunetr__nolr_seed<seed>/fold_0/model_final_checkpoint.model`;
- prediction:`data/nnunetv1/v1_comparison_predictions/swinunetr__nolr_seed<seed>/*.nii.gz`;
- eval:`evaluation/results_locked/`(与 baseline 同目录,靠 method 列区分)。
- DSUB job 脚本放 `jobs/train/`、`jobs/predict/`,新文件不覆盖 baseline 脚本。

### Q13. determinism / 复现

- 关 mirror 是给定 seed 下确定性的(augmentation 随机序列会变,但可复现)。
- 保持 `cudnn.deterministic=True`、worker seeds、`PYTHONHASHSEED`(train.py 已设)。
- smoke 阶段建议跑两次同 seed forward,确认改动后仍逐字节确定(可选,handoff 做过)。

---

## 6. 科学风险与论文定位

### Q14. 已知风险清单

| 风险 | 说明 | 缓解 |
|---|---|---|
| SwinUNETR 关 mirror 无效或变差 | 全局注意力可能已隐式处理左右 | 双向假设(Q10),照实报 |
| 2D 逐 slice,关 LR flip 机制不同于 3D | 2D tail 可能不是 mirror 导致 | axis audit(Q2)+ 实验 itself |
| conditional 实现破坏管线公平性 | fork get_moreDA 偏离 baseline | Phase 1 只 full-disable(Q4) |
| LR axis audit 做错 | 关错轴 → 实验全废 | audit 脚本 + witness md + smoke 打印 |
| 2D 入口缺失(Q3) | 无法对 2D 落地 | 先 SwinUNETR,2D 后补 |

### Q15. 在 issue #1 主对比里的定位

- 这是主对比的**改进型 ablation**,不是新主对比。叙事:"在最强网络上,通过关闭
  LR-mirror 进一步修复 kidney tail / 左右混淆"。
- 若成立,可作为 issue #1 结果的**补充改进**写进论文;若不成立,则"现有最强网络已
  无需此 trick"也是有效结论。
- **不**升级为新的 champion 声明,除非 Phase 2 全网络 ablation 也支持。

### Q16. 与 handoff / MedNeXt 配置问题的边界

- handoff(Huawei v1 Task520)只作先验,不混比(已在 scope 声明)。
- issue #1 里 MedNeXt-S 的 `exp_r=4`(非上游 S 的 `exp_r=2`)是另一个独立问题,
  MedNeXt-L 正在重训 —— **不在本实验范围**,本实验不碰 MedNeXt。

---

## 7. 落地阶段(正式 spec 应给出的 stage 表,草案)

| Stage | 内容 | 依赖 | 状态 |
|---|---|---|---|
| 0 | Q1+Q2 audit:打印 baseline mirror_axes + 确认 LR axis(3D,可选 2D) | — | 待做 |
| 1 | Q3:定位 / 决策 2D 训练入口 | Q3 决策 | 待做 |
| 2 | Q6:base_trainer 加 `lr_mirror_mode` + audit 脚本 + smoke(打印 mirror_axes 改前/改后) | Stage 0 | 待做 |
| 3 | validator 审查(改前/改后 mirror_axes、axis 正确、forward 正常、不污染 baseline) | Stage 2 | 待做 |
| 4 | 训练:SwinUNETr__nolr × 3 seed(+ 2D 若 Q3 解决) | Stage 3 PASS | 待做 |
| 5 | 预测:test 39,sliding window 0.5,与 baseline 同口径 | Stage 4 | 待做 |
| 6 | 评估 + 统计(locked evaluator + kidney_swap + Wilcoxon 配对) | Stage 5 | 待做 |
| 7 | (可选)Phase 2 扩到其余 3 网络 | Phase 1 成立 | 待定 |

---

## 8. 待决策清单(请先拍板,再写正式 spec)

1. **Q3**:2D 入口怎么办?—— 先只 SwinUNETR(c),还是先排查 wrapper(a),还是
   把 2D 纳入 framework(b)?
2. **Q4**:第一阶段只 full-disable?—— 同意?
3. **Q5**:Phase 1 只 2 网络、成立后再扩?—— 同意?
4. **Q9**:统计口径 = per-case 3-seed 平均后 Wilcoxon + Holm?—— 同意?
5. **Q2 audit 范围**:3D 必做;2D 的 axis audit 现在做,还是等 Q3 定了再做?

拍完这 5 条,就可以把本 draft 收敛成正式 `v1-kidney-lr-mirror-ablation-spec.md`。
