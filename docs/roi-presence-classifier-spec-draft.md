# Spec Draft — ROI/Half-Projection Presence Classifier（迁移到 article 自有 split）

Status: **DRAFT，全部决策完成（D1-D29 + G1-G6）+ F1-F3 已核查 + 内部一致**（2026-06-23）— implementation-ready，待冻结为正式 spec。

Date: 2026-06-23

参考来源（已通读）：
- 实现：`AutoScientists/output/swct06042040/task/tools/roi_half_projection_presence_classifier/`
  （`generate_roi_projections.py` / `train_roi_classifier.py` / `audit_roi_orientation.py` /
  `run_roi_controls.py` / `summarize_roi_results.py` / `common.py` / `default_config.yaml`）
- 原 spec：`AutoScientists/docs/SWCT06042040-ROI-HALF-PROJECTION-PRESENCE-CLASSIFIER-SPEC.md`
- 原 run 结果 + handoff：`docs/SWCT06042040-HUAWEI-V1-KIDNEY-MIRROR-CLASSIFIER-HARDZERO-HANDOFF.md` §3（实验 B）

---

## 0. 目标 + scope

把 swct06042040 已实现的 **ROI/half-projection presence classifier**（2D lateral
projection + CT-image-only 前景 bbox + 解剖 ROI crop + endpoint-specific ResNet-18
二分类），迁移到 **article 仓库自有的固定 split**（`data/splits/split_manifest.csv`，
seed 42，**120 / 38 / 39**）上训练与评估。

**不**改 nnU-Net segmentation 训练/推理（本 spec 只覆盖 classifier 训练）；classifier 产物
计划用于下游 segmentation gate（D29=(b)），但 gating 的实施/评估是**独立后续 spec**，不在本 spec
范围。本 spec 不主张 segmentation 改善或 champion。

本 draft 只列**需要讨论/决策的问题**，不做任何实施决策。技术约束（只有一种正确做法的
机制）单独标注为「C 类」。

---

## 决策总览（D1-D29，全部已拍板；F1-F3 已核查）

| # | 决策 | 结论 |
|---|---|---|
| F1 | Huawei 图像/label 路径 | article `data/{split}/{images,labels}/` 软连接齐全 → 用此 |
| F2 | runtime stack | env `nnunetv1`(torch2.4.1+cuda118/torchvision0.19.1) + module gcc9.3.0/cuda11.8.0/cudnn8.6.0；**Huawei 无外网，resnet18 权重须预放置** |
| F3 | 原投影同源 | 原 4710 .npy 全在、同源坐实；但仅 157 例（缺原 test 40 例） |
| D1 | 科学目标 | 训练**准确**的 head/testis presence classifier，**接受 shortcut** |
| D2 | endpoint | 维持 head_present / testis_present（≡source） |
| D3 | test | sealed 用于选择 + **一次性最终 test 评估** |
| D4 | seed | 单 seed 20260520 |
| D5 | 工具 | port 进 article `tools/roi_classifier/`（改路径/split/test） |
| D6 | split token | ported 代码接受 article 原生 `val` |
| D7 | presence 标签 | GT 体素扫描（class 9/6）+ 一致性检查 |
| D8 | 派生 manifest | `data/manifests/classifier_split_manifest.csv`，git-tracked，相对路径，全 197 |
| D9/D10 | image/label path | article 相对路径 `data/{split}/{images,labels}/<case>.nii.gz` |
| D11 | 投影 | **全部重新生成**（197 例 × 3 roles × 5 variants = 2955 `.npy`，~1.7G） |
| D12 | 投影参数 | verbatim |
| D13 | orientation audit | article 内重跑（全 197） |
| D14 | label id | head=9, testis=6 |
| D15 | 输出 root | `swine-CT-article/runs/roi_presence_classifier/`；`runs/` 进 .gitignore |
| D16 | grid | **30 models** = 3 correct ROI roles × 5 variants × 2 init（砍 wrong-ROI） |
| D17 | seed（确认） | 20260520 |
| D18 | 训练超参 | verbatim |
| D19 | best-model 选择 | 维持 FA≤1 gate + val AUPRC；额外报告 calibration |
| D20 | 调度 | 单 GPU 顺序 job，agent<170 & 不用 174，loop dsub |
| D21 | stack（确认） | nnunetv1 + cuDNN 8.6.0；预放置 resnet18 权重，保留 imagenet init |
| D22 | 控制组 | **不跑** |
| D23 | 结论口径 | 中性实用（准确 classifier + 一句话 ≡source caveat） |
| D24 | wrong-ROI 阈值 | 作废（无对象） |
| D25 | root 子层 | 不加 article_split/，stamp 区分 |
| D26 | 本地 mirror | 小报告 mirror 到本地 `runs/`（gitignored）；大文件不拉 |
| D27 | GitHub | 只放源（spec/脚本/manifest）；run 输出全 gitignore |
| D28 | 流程 | article spec-draft/spec/REVIEW；R1-R6 映射为里程碑 |
| D29 | downstream | **(b) 计划接 article segmentation gate**（handoff C 路线，独立后续 spec） |
| G1 | 脚本清单 | port 5 个 + 保留改造 `summarize` + new `make_classifier_manifest`/`eval_test`；丢 `run_roi_controls` |
| G2 | test 评估 | `--include-test` flag + 新 `eval_test.py`；指标=val 同款+calibration，thr 0.5 |
| G3 | 跨 grid 选 best | val AUPRC(FA≤1 gate) → FA↓ → Brier↓ → 字典序 |
| G4 | manifest 生成 | 路径从 split+case_id 拼；一致性违反 **stop+上报** |
| G6 | orientation 复核 | **全自动** PASS/FAIL（head cranial/testis caudal/testis lower 跨例一致） |
| G5/G8/G9/G10/G14 | 🔧 技术 pin | 权重 SHA、运行/拉取位置、`__init__.py`、config 字段(label id 9/6 等)、summarize 改造 |

---

## 1. 最关键事实：presence ≡ source（结构性，换 split 解决不了）

来源：`data/README.md §149「关键约束：source = class presence」` + CLAUDE.md 数据节。

- **HZAU 93 例**：阉猪 → **head-present / testis-absent**
- **TB 104 例**：公猪 → **head-absent / testis-present**

因此：

```
head_present = 1  ⟺  source == HZAU      （93 例 present，104 例 absent）
testis_present = 1  ⟺  source == TB       （104 例 present，93 例 absent）
```

**逐例完全相等**，没有 within-source 变化。

直接推论（决定整个实验的科学定位）：

1. 在本数据集上，"head / testis presence 分类" **字面等价于 source 分类（HZAU vs TB）**。
2. **source-only 基线 = oracle**，AUPRC = 1.0 是必然，不是模型学到了什么。
3. 任何基于图像的 classifier 也会靠 FOV / 扫描场 / source artifact 轻松拿 AUPRC≈1.0 ——
   这是**结构性 shortcut**，换 article split 不会改变（同一批 197 例、同样的
   source-presence 混淆）。
4. 原 swct06042040 run 已观测到：source-only / shape-only / wrong-ROI 全部 AUPRC≈1.0
   → 实证 shortcut-positive。

→ **D1 已定**：接受 shortcut，目标是准确的 presence classifier。因此本次的价值定位为：
- 产出一个在 article split 上**分类准确**的 head/testis presence classifier（train/val/test
  近乎完美分离，AUPRC≈1.0、FA≈0 近乎必然）；
- 因 presence≡source，该准确性**反映 source/FOV 信号而非解剖定位** —— 这是结构性事实，报告里
  以一句话 caveat 说明（D23），**不**作 shortcut 诊断、**不**跑控制组（D22）；
- 产物用途 = 下游 segmentation gate 输入（D29=(b)，独立后续 spec）。

---

## 2. 环境与路径（事实）

- article 仓库（本地 canonical）：`/Users/liuyangfan/Documents/work/CT/swine-CT-article`
- article 仓库（Huawei 镜像）：`/home/share/hzau/home/liuyangfan/swine-CT-article`
- 既有实现（AutoScientists，参考用，不是 article 仓库内）：
  `.../AutoScientists/output/swct06042040/task/tools/roi_half_projection_presence_classifier/`
- 原已生成的投影 / 模型（Huawei）：
  `/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040/outputs/roi_half_projection_presence_classifier/`

### article split 真实结构（已读，确认）

`data/splits/split_manifest.csv` 列：

```
case_id, source, source_detail, breed_en, hzau_batch, split
```

- split ∈ {**train, val, test**}；计数 **120 / 38 / 39**；source×split：HZAU 56/18/19、TB 64/20/20。
- **没有** `head_present` / `testis_present` / `image_path` / `label_path` 列。

`data/manifests/case_metadata.csv` 含：

- `image_path`：值如 `/workspace/data/CT/all/07069186.nii.gz` —— **HZAU 服务器路径**（不是 Huawei 路径）。
- `label_path`：值如 `/workspace/data/CT/HZAU_veterinary_hospital/label/07069186.nii.gz`。

---

## 3. 技术约束（C 类，非决策，标注用）

- **C1**. article split 冻结，所有实验共用，不得改动（CLAUDE.md / `make_split.py`）。
- **C2**. presence ≡ source（§1），数据结构决定，改不了。
- **C3**. 投影方向：SimpleITK 读出 z,y,x，沿 **axis 2（x = 解剖 L-R）** 投影；
  `flip_cranial_to_left=True`。这是 image 属性，与 split 无关。
- **C4**. 大文件（CT/label `.nii.gz`、`.npy` 投影阵列、`.pt` checkpoint、preview pack）**永不进
  本地 / GitHub**（CLAUDE.md 同步规则）。本地只存小文件（manifest / report / metrics json / spec / 正式脚本）。
- **C5**. classifier 代码**硬编码** split ∈ {`train`, `validation`}（`common.forbid_test_rows`、
  `train_roi_classifier._load_rows`、`run_roi_controls`）；article 用的是 `val`。→ 必须有适配层（见 D6）。
- **C6**. label class id（CLAUDE.md）：`9 = head`、`6 = testis`。

---

## 4. 决策点（每条：选项 / 背景 / 我的建议；均待拍板）

### A. 科学定位

#### D1（核心）. ✅ 已决策（2026-06-23）

**决定（用户）**：本次实验的目标 = **训练一个能准确区分 `head_present` / `testis_present`
有无的 classifier；接受 shortcut（source/FOV shortcut 不算实验失败、不作为科学阻塞）。**

- 不取 (a) shortcut 缓解诊断口径，也不取 (c) "必须先解决 source 混淆才谈 gate" 的硬要求；
  而是实用口径：要一个能给出 per-case presence 概率、分类准确的 classifier。
- 用户原话：「只想训练一个能够准确区分有没有头和睾丸的 classifier，即使是 shortcut 也没关系」。
- 因 `presence ≡ source`（§1），AUPRC≈1.0、FA≈0 近乎必然 → 真正有区分度的质量维度落到
  **calibration / 阈值 / false-absent 安全性**（尤其若下游要做 gate，见 D29）。

**对后续决策点的连带影响**（逐个到时再确认）：

- **D22/D23（控制与结论）**：source-only / shape-only / wrong-ROI 控制从「必交的科学诊断」
  降级为「可选的诚实性补充报告」；结论口径 =「达到了准确的 presence 分类」（不再写
  shortcut-blocks-gating）。
- **D16（grid）**：wrong-ROI roles 原本服务 shortcut 诊断；shortcut 接受后，是否仍需 4 个 testis
  ROI + wrong 对照需重审（可简化到每 endpoint 一个最佳 ROI/variant）。
- **D29（downstream）**：classifier 是仅作为独立 artifact，还是计划接 segmentation gate
  （handoff 实验 C 路线）—— 决定 false-absent 安全性是否硬指标。

#### D2. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 维持 `head_present` / `testis_present` 两个 endpoint，接受
`presence ≡ source`。不引入 within-source presence 目标（本数据也无现成的）。
由 D1 直接推出。

#### D3. ✅ 已决策（2026-06-23）

**决定（用户）**：**(b)** test-sealed 用于模型选择 + **一次性最终 test 评估**。

- val 用于模型选择；冻结 best model 后对 test 预测 + 评估**一次**，报告 test 指标。
- 符合 article「test 冻结、仅最终评估」惯例。
- **代码耦合**：原 `generate_roi_projections.py` 硬编码禁止 test（`forbid_test_rows` +
  `--allowed-splits train,validation`）→ 需**单独、显式**的 test 投影生成 + 一次性预测评估步骤
  （带自己的 gate，不是顺手放开 test）。这条在 D5（工具）和实施阶段落实。

#### D4. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 单 seed `20260520`（与 article base seed 惯例一致）。
- 理由：`presence≡source` 使任务近乎线性可分，跨 seed 方差≈0，多 seed 只增 3× 训练量无信息量。
- 多 seed 列为 optional（仅当 D29 下游 gate 需要 robust 阈值时再回头加）。

#### D5. ✅ 已决策（2026-06-23）

**决定（用户）**：**(b)** 把脚本 **port 进 article 仓库**，落点 **`tools/roi_classifier/`**。

- 保留已验证的投影/训练/指标逻辑（照搬），改掉硬编码路径 + 加 article split 适配 + test 评估步骤。
- 成为 article canonical 脚本、进 git；不再依赖 AutoScientists workspace。

**port 改动清单**（实施阶段做）：
1. `common.py`：`SPEC_PATH` / `DEFAULT_OUTPUT_ROOT` → article 路径；`EXPECTED_CASE_COUNT=197` 保留。
2. split 适配：见 D6（ported 代码接受 `val`）。
3. **test 评估步骤**（D3）：新增"冻结 best model → 生成 test 投影 → 一次性预测+评估"的显式 gated 流程
   （原 `generate` 硬禁 test，需放开 / 新增入口）。
4. image/label path：用 article `data/{split}/{images,labels}/`（F1）。
5. config：output root → article Huawei 镜像下（D15）。

---

### B. split 与 labels 衔接

#### D6. ✅ 已决策（2026-06-23）

**决定（用户）**：**(b)** ported 代码直接接受 article 原生 `val`。

- `FIRST_ROUND_SPLITS = {"train","val"}`、reserved split = `"test"`；派生 manifest（D8）统一用
  `train/val/test`，与 article `split_manifest.csv` 一致。
- 不做 `val→validation` 翻译层（代码已 port、test 处理本就要改，顺手统一）。
- D5 改变了原 (a) 的算法：既然代码在自己手里，没必要为继承来的 token 维护翻译。

#### D7. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 扫 GT label 文件派生 presence。

- class 9 (head) / class 6 (testis) voxel count > 0 → present=1，否则 0。
- **一致性检查**：GT 派生 presence 必须与 source 列逐例相等；任何违反 = 数据异常，flag。
- 标签定义从 GT 出发（原则正确），shortcut 留在 classifier 一侧（D1）。
- 实现：用 F1 确认的 article `data/{split}/labels/<case>.nii.gz`，一次 ssh 扫 197 例；产出
  `head_present` / `testis_present` 进 D8 派生 manifest。

#### D8. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 派生 manifest 落 `data/manifests/classifier_split_manifest.csv`，**git-tracked**，
配生成脚本；路径用**相对仓库根**。

- schema：`case_id, split, source, source_detail, breed_en, image_path, label_path, head_present, testis_present`
- `split` ∈ {train, val, test}（D6）；覆盖全部 197（D3 要评 test）。
- `image_path` / `label_path`：相对仓库根（如 `data/val/images/07069186.nii.gz`），工具用 `--repo-root` 解析。
- `head_present` / `testis_present`：D7 GT 扫描。
- 生成脚本：`tools/roi_classifier/make_classifier_manifest.py`（port 一部分，见 D5）。

#### D9 / D10. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** image_path / label_path 用 article 仓库相对路径。

- `image_path = data/{split}/images/<case>.nii.gz`
- `label_path = data/{split}/labels/<case>.nii.gz`
- F1 已核查：Huawei 上软连接齐全、解析到 labeled_197 存储；同源已坐实（F3）。
- HZAU 服务器路径 `/workspace/...` 弃用；r20 imagesTr 路径不用。

---

### C. 投影（R3）

#### D11. ✅ 已决策（2026-06-23）

**决定（用户）**：**(b)** 全部重新生成（train+val 158 + test 39 = 197 例 × 3 roles × 5 variants = **2955 `.npy`**，约 **1.7 GB**）。

- F3 同源已坐实，但原 `.npy` 只覆盖 157 例（原 test 40 例无投影）→ 纯复用不完整。
- 重生成让 article 证据链自包含、统一（article 图像路径 + ported 脚本 + article split），无 provenance 耦合。
- 生成可多 worker 并行（~30min 量级）；1.7G 磁盘可接受。
- 原 2.7G 旧 `.npy`（157 例）不引用、不拷贝，仅作 swct06042040 历史产物。

#### D12. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 投影参数 verbatim 复用（`default_config.yaml` 原样带过来）。

- foreground `HU>-800`、CT bed 计入、2D fill、cleanup `0.001`、bbox pad `0.05`、ROI 50/50、
  resize-pad `224×224`、`padding_value=0`、bilinear、单通道复制 3 通道。
- 与原 run 可比；D1 下任务平凡可分，调参数无收益反增 confound。

#### D13. ✅ 已决策（2026-06-23）

**决定（用户）**：**(b)** 在 article 仓库重跑 `audit_roi_orientation.py`（全 197 例）。

- 用 D8 派生 manifest + article 图像/label 路径；产出 article-native audit artifact。
- 与 D11（全重新生成、自包含）一致；分钟级成本。
- 产出后作为 ported `generate_roi_projections.py --orientation-verified` 的 gate。

#### D14. ✅ 已决策（2026-06-23）

**决定（用户）**：确认 **head label id = 9、testis label id = 6**（C6 技术事实）。
audit / D7 presence 扫描 / orientation 质心统一用这两个 id。

#### D15. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 新建顶层 `swine-CT-article/runs/roi_presence_classifier/`。

- 子结构（沿用原 layout）：`data/{manifests,projections/<STAMP>,previews/<STAMP>}/`、
  `models/batch_<STAMP>/`、`reports/<STAMP>/`、`jobs/batch_<STAMP>/`。
- **`runs/` 整个加进 `.gitignore`**（大 `.npy`/`.pt`/preview 不进 git；只 mirror 小报告/manifest，D26）。
- article 仓库 Huawei 端原本无 `runs/` 先例（seg 输出在 `data/nnunetv1/`）；classifier 独立隔离至此。

---

### D. 训练（R4）

#### D16. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a) 中度简化 = 30 models**。

- 3 correct ROI roles（`head_cranial_half`、`testis_caudal_half`、`testis_caudal_lower_half`）
  × 5 variants × 2 init（imagenet, random）= 30。
- 砍掉 3 个 wrong/control ROI roles（`head_caudal_half`、`testis_cranial_half`、
  `testis_caudal_upper_half`）—— 它们仅服务 shortcut 定位诊断，D1 已放弃。
- `common.py` 的 `ROI_ROLES_BY_ENDPOINT` / `EXPECTED_MODEL_COUNT` 需相应改：head 1 role、
  testis 2 roles → `EXPECTED_MODEL_COUNT = 3 × 5 × 2 = 30`。
- val 仍按 variant/init 选每 endpoint 最佳（D19）。

#### D17. ✅ 已决策（2026-06-23）

**决定（用户）**：seed = **`20260520`**（D4 已定单 seed，此处确认沿用）。

#### D18. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 训练超参 verbatim（`default_config.yaml` training 段原样）。

- AdamW（imagenet 3e-4 / random 1e-3, wd 1e-4）、cosine+5ep warmup、max 100ep、patience 15、
  batch 16、BCEWithLogitsLoss+capped pos_weight(5.0)、AMP、deterministic cuDNN。
- article split 下 pos_weight ≈ 1.14/0.88（近均衡），不触发 cap。

#### D19. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 维持原选择口径（FA safety gate + val AUPRC）+ 额外报告 calibration。

- 退化下原逻辑仍合理（FA 优先 + 首达峰值胜出）。
- **额外报告**选中模型的 Brier / ECE（透明度 + 潜在 gate 参考），但**不**纳入选择。
- 不发明新口径，与原 run 可比。

#### D20. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 单 GPU 顺序 job，resource `cpu=8;mem=64000;gpu=1`，node `agent<170` &
不用 `whshare-agent-174`，loop dsub，early-runtime 检查。

- 与原 classifier policy + article kidney-lr-mirror §248 惯例一致。
- **调度结构**（实施阶段）：CPU 阶段（D7 presence 扫描 / D13 orientation audit / D11 projection 生成）
  各为/合并为 CPU job；GPU 阶段（D16 训练 + D3 test 一次性评估）为 GPU job；阶段间 validator/REVIEW gate 串行。

#### D21. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 预放置 resnet18 ImageNet 权重，保留 imagenet init。

- **env**：`swine_ct_autonomous_discovery/envs/nnunetv1`（torch 2.4.1+cuda118 / torchvision 0.19.1 / cuDNN 8600，实测与原 run 一致）。
- **module**：`module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.6.0_cuda11`。
- **权重**：本地 Mac 下载 `resnet18-f37072fd.pth`（~45MB）→ rsync 到 Huawei
  `~/.cache/torch/hub/checkpoints/`（Huawei 无外网，job 内下载不可行）；记录 SHA 证清白。
- D16 双 init（imagenet + random）完整保留。

---

### E. 控制与解释（R5）

#### D22. ✅ 已决策（2026-06-23）

**决定（用户）**：**(b) 不跑控制组。**

- D1 已接受 shortcut、不做 shortcut 诊断 → source-only / shape-only / correct-vs-wrong ROI 控制组全部不跑。
- `run_roi_controls.py` 不纳入本次 port（或 port 但不调用）。
- 诚实性 caveat 改由报告里一句话说明（见 D23），不靠控制组数据支撑。

#### D23. ✅ 已决策（2026-06-23）

**决定（用户）**：**(c)** 中性实用口径。

- 结论主轴 =「训练得到准确的 head/testis presence classifier，train/val/test 近乎完美分离」。
- 一句话诚实 caveat：「本数据集 presence≡source，故准确性反映 source/FOV 信号而非解剖定位」
  （不靠控制组数据，不写成诊断实验）。
- 不主张 gating / champion（留 D29 授权）。

#### D24. ✅ 作废（2026-06-23）

wrong-ROI delta 0.05 阈值**无对象**：D16 砍掉 wrong-ROI roles、D22 不跑控制组 → 无 correct-vs-wrong
ROI 比较。此条 N/A，不纳入 port。

---

### F. 产物与同步

#### D25. ✅ 已决策（2026-06-23）

**决定（用户）**：**不加** `article_split/` 子层。

- 直接用 D15 的 `runs/roi_presence_classifier/`；stamp 子目录（`batch_<STAMP>`）已天然区分每次 run。
- 这是 article 唯一的 classifier run，无同目录混淆。

#### D26. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** mirror 小报告到本地，大文件不拉。

- mirror（本地 `runs/roi_presence_classifier/`，gitignored 核查副本）：`training_summary.json`、
  per-model `metrics_summary.json`/`run_manifest.json`、`training_history.csv`、
  train/val/test `predictions.csv`、orientation audit json/md、projection manifest csv。
- **不** mirror：`.npy` 投影（~1.7 GB）、`.pt` checkpoint、preview PNG pack（CLAUDE.md EXCLUDES）。
- rsync 拉用通用 EXCLUDES，不加 `--delete`。

#### D27. ✅ 已决策（2026-06-23）

**决定（用户）**：git 只放可复现源；run 输出全 gitignore。

- **进 git**：`docs/roi-presence-classifier-spec*.md`、`tools/roi_classifier/`（脚本+config）、
  `data/manifests/classifier_split_manifest.csv` + 生成脚本。
- **不进 git**（`runs/` 整个 gitignored）：投影 `.npy`、checkpoint `.pt`、preview PNG、mirror 的小报告。
- **.gitignore 增项**：`runs/`、`*.npy`（已有 `*.pt`/`*.pth`）。

---

### G. 流程与 downstream

#### D28. ✅ 已决策（2026-06-23）

**决定（用户）**：**(b)** 用 article spec-draft/spec/REVIEW 流程。

- 原 R1-R6 stage 映射成实施里程碑，每个后接 REVIEW/检查 gate；原 validator checklist 项
  （orientation 验证、禁 test、grid 完整、node 合规、无 forbidden side-effect）作为 REVIEW checklist 保留。
- **实施里程碑序列**：① port 工具+config（D5，static 检查）→ ② 派生 manifest（D7+D8，一致性检查）
  → ③ orientation audit（D13，REVIEW gate）→ ④ projection 生成 train/val（D11/D12）→
  ⑤ 训练 30 models（D16/D18，runtime 检查）→ ⑥ test 一次性评估（D3，最终报告 D23 口径）。

#### D29. ✅ 已决策（2026-06-23）

**决定（用户）**：**(b)** classifier 计划接 article segmentation 做 gate（handoff 实验 C 路线）。

- 本次 spec 的产物（准确 presence classifier）**用途 = 下游 segmentation gate 的输入**。
- **scope 边界**：本 spec 覆盖 **classifier 训练**（D1-D28）；gating 的实施与评估（pre-argmax
  probability transform、article split 上 head-absent FP→0 评估、副作用）作为**后续独立 spec**
  （handoff C port 到 article split）。若用户希望把 gating 也并入本 spec，需另行确认扩 scope。
- **与既有决策的耦合**：
  - D1（shortcut 接受）+ D29(b)：gate 会"有效"（head-absent FP→0），但本质按 source 信号压 →
    D23 的 caveat 在 downstream gating 报告里要更显眼（gate 有效性来自 source，非解剖）。
  - D19（FA 优先 + 报告 calibration）：gate 场景下 false-absent 安全性是硬指标，D19 的 FA-priority
    选择正好对齐；calibration 报告为 gate 阈值调优留余地。
  - D22（不跑控制组）：downstream gating 评估时，"source-only 即可达 oracle"这一点会在 gating
    报告里以一句话说明，不靠 classifier 侧控制组。

---

## 5. Huawei 端只读核查结果（2026-06-23 已确认）

### F1（D9/D10 路径）✅ 已确认

- article 仓库在 Huawei 已物化 `data/{train,val,test}/{images,labels}/`，**计数全对**
  （images + labels 各 120 / 38 / 39）；这些目录是**真目录**，但**每个文件是软连接**。
- 软连接解析到 canonical 存储：
  `/home/share/hzau/whcs-share37/liuyangfan/nnunet_medsam_semisup/data/labeled_197/{images,labels}/<case>.nii.gz`。
- HZAU 命名 `07069186.nii.gz`，TB 命名 `136021_122090.nii.gz`，两 source 都在。
- case_metadata 里的 HZAU 服务器路径 `/workspace/data/CT/all/...` 在 Huawei **不存在**（`/workspace`
  是 hzau_gpu 路径）→ 派生 manifest 的 `image_path` 必须改用 Huawei 路径。
- **结论**：D9/D10 用 article 仓库 `swine-CT-article/data/{split}/images|labels/<case>.nii.gz`
  （软连接形式）即可；不需走 swct06042040 r20 路径。

### F2（D21 runtime stack）✅ 已确认（含 1 个待办）

- 原 R4 实际跑的：`torch 2.4.1+cuda118`、`cuDNN 8600`、`cuda 11.8`、`cudnn_deterministic=True`。
- 默认 base python（anaconda3）**无 torch**；候选 env：`swine_ct_autonomous_discovery/envs/nnunetv1`、
  `software/anaconda3/envs/torch2`、`vision` —— 三者 import torch 仅因**登录 shell 未载 cuDNN** 失败
  （`libcudnn.so.8` not found），job 里 `module load` 即修复。
- module 配方（`module avail` 实测）：
  `compilers/gcc/9.3.0` + `compilers/cuda/11.8.0` + `libs/cudnn/8.6.0_cuda11`（= 原 run 的 8600）；
  CLAUDE.md verified 栈用 `libs/cudnn/8.8.1_cuda11`（也可）。
- ⚠️ **ImageNet resnet18 权重未缓存 + Huawei 无互联网**：实测 `module load` 后
  `torchvision.models.resnet18(weights=IMAGENET1K_V1)` 触发下载
  `https://download.pytorch.org/models/resnet18-f37072fd.pth`（~45MB），但**节点无外网**
  （`URLError: gaierror(-2, 'Name or service not known')`）→ **必须预先放置**该 `.pth` 到
  `~/.cache/torch/hub/checkpoints/`（从有网的本地 Mac 下载后 rsync 到 Huawei）。
- ✅ **env + module 实测（2026-06-23）**：`swine_ct_autonomous_discovery/envs/nnunetv1` 在
  `module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.6.0_cuda11` 下 =
  **torch 2.4.1+cuda118 / torchvision 0.19.1+cuda118 / cuDNN 8600**，与原 run 完全一致 → 用此 env。
  （`torch2` env = torch 2.1.0，不匹配，不用。）

### F3（D11 复用投影）✅ 已确认

- 原 swct06042040 投影 `.npy` **全部还在**：4710 个（= 157 cases × 6 roles × 5 variants），位于
  `.../roi_half_projection_presence_classifier/data/projections/20260620T093836Z/`。
- 原 manifest：4710 rows，split counts `train 3540 / validation 1170`，**157 unique cases**
  （→ 原 split 为 118 train / 39 val / **40 test**，与 article 的 120/38/**39** 是**不同 split**，
  seed 20260520 vs 42，逐例分配不同）。
- 原 manifest `image_path` 前缀：
  `.../swine_ct_autonomous_discovery/data/data_root/nnunet/nnUNet_raw/Dataset101_Carcass9Class/imagesTr/<case>_0000.nii.gz`
  （nnU-Net `_0000` 后缀）。
- **同源核查（决定性）**：重叠 case `07069189`，原 r20 `imagesTr/07069189_0000.nii.gz`（35,526,876 B）
  与 article `data/train/images/07069189.nii.gz`（→ labeled_197，35,526,876 B）**字节同大小 = 同源**。
  → article 图像与原投影来源图像是同一批 197 例，**D11 复用 `.npy` 在「同源」维度是安全的**（剩余顾虑仅
  provenance 自包含性，见 D11）。
- 注意命名差异：nnU-Net 用 `<case>_0000.nii.gz`，article 用 `<case>.nii.gz`；复用 `.npy` 不受影响（按 case_id
  索引），但若重新生成投影，`image_path` 要用 article 路径。

---

## 6. 实施顺序（D1-D29 已全部决策；按 D28 里程碑）

0. 预放置 resnet18 权重到 Huawei（D21 / G5）。
1. port 工具 + config 到 `tools/roi_classifier/`（D5 / G1 / G9 / G10）→ static 检查（py_compile / `--help` / `--dry-run`）。
2. 派生 manifest：Huawei 上跑 `make_classifier_manifest.py`（D7 + D8 + G4）→ 一致性检查（presence≡source）→ rsync 回本地进 git。
3. orientation audit（D13 / G6）→ **自动 PASS/FAIL 判据**（head cranial / testis caudal / testis lower 跨例一致；不过则 block）。
4. projection 生成 train/val（D11 / D12）→ 检查（rows 计数、禁 test、preview）。
5. 训练 30 models（D16 / D18）→ runtime/调度检查；跨 grid 选每 endpoint 最佳（G3）。
6. test 一次性评估（D3 / G2）→ 最终报告（D23 口径）。

每步后 REVIEW / 检查 gate（D28）。

---

## 7. NO-GO

- orientation 无法验证；
- GT mask 用于构造 per-case ROI input（foreground / bbox / ROI 边界 / classifier 像素）；
- 30-model grid 缺模型（D16）；
- **test 在模型冻结前被触碰**（D3 只允许冻结 best 后的一次性 test 投影/预测/评估，且 test 不得参与模型选择）；
- job 落 node ≥170 或 `whshare-agent-174`；
- imagenet init 在未预放置 resnet18 权重的情况下跑（D21，Huawei 无外网）。

---

## 8. 落地审查 pin 项（G1-G14；2026-06-23 审查新增）

> 决策（D1-D29）完整后、直接落地前的审查发现。**G1 / G2 / G3 / G4 / G6 含需用户拍板的小决策**
> （附建议）；其余标 🔧 为技术规格 pin（只有一种合理做法，记录用）。

### G1. ✅ 已决策（2026-06-23）

**决定（用户）**：保留 `summarize_roi_results.py` 并改造。最终 `tools/roi_classifier/` 清单：

| 脚本 | 处置 |
|---|---|
| `common.py` + `default_config.yaml` | port + 改（路径/常量/config 字段，G10） |
| `audit_roi_orientation.py` | port（D13） |
| `generate_roi_projections.py` | port + 改（加 `--include-test` flag，G2；val token，D6） |
| `train_roi_classifier.py` | port + 改（val token；grid=30，D16） |
| `summarize_roi_results.py` | 保留 + 改造（去 control 依赖；加 G3 跨 grid best 选择；读 G2 test 输出；产 result card，D23） |
| `run_roi_controls.py` | **丢**（D22） |
| `make_classifier_manifest.py` | **new**（D7 + D8，G4） |
| `eval_test.py` | **new**（G2；见下） |

### G2. ✅ 已决策（2026-06-23）

**决定（用户）**：① `--include-test` flag + ② 新建 `eval_test.py` + ③ val 同款指标 + calibration、thr 0.5。

- **test 投影**：`generate_roi_projections.py` 加 `--include-test` gated flag（复用投影逻辑；无 flag 仍 forbid test；仅里程碑⑥ + 显式 flag 放开）。
- **test 预测+评估**：新建 `eval_test.py`（import train 的 `_build_resnet18` / `_metrics` / `_temperature_summary` / `RoiProjectionDataset`）→ 加载 G3 选出的冻结 best checkpoint → 预测 test 投影 → 算指标；`summarize` 读其输出汇总。
- **test 指标**：= val 同款（AUPRC/AUROC/Brier/ECE/FA/FP/sens/spec）+ calibration（temperature）；阈值 0.5（verbatim）。
- test 指标**一次性、冻结 best 后才算**，不参与选择（D3 / §7）。

### G3. ✅ 已决策（2026-06-23）

**决定（用户）**：**(a)** 跨 grid 选择键 = val AUPRC 最高且过 FA≤1 gate；tiebreak：**FA↓ → Brier↓ → 确定性字典序（role, variant, init）**。

- 每 endpoint 选出**一个**最终 classifier（head 一个、testis 一个）。
- 主键与 D19 单模型口径一致（FA 优先）；末位字典序保证可复现。
- 选中模型 → G2 test 评估 + D29 下游 gate。选择逻辑落在 `summarize_roi_results.py` 改造里（G1）。

### G4. ✅ 已决策（2026-06-23）

**决定（用户）**：一致性**违反时 stop + 上报**。技术 pin 如下：

- **路径构造**：`image_path`/`label_path` 从 `split + case_id` 拼 `data/{split}/{images,labels}/<case>.nii.gz`，**不读** case_metadata 的 `/workspace` 路径。
- **presence 扫描**：SimpleITK 读 label，class 9/6 voxel count > 0 → present。
- **join**：`split_manifest.csv`（case_id, source, source_detail, breed_en, split）+ 构造路径 + GT presence → D8 schema 197 行。
- **一致性检查**：GT 派生 presence 逐例必须 == source 派生；**违反 → 打印 case_id + 详情、退出非 0、不写 manifest**（触发人工核查，预期 0 违反）。
- **运行位置**：Huawei（CPU），产出 rsync 回本地进 git。

### G6. ✅ 已决策（2026-06-23）

**决定（用户）**：**全自动审核**（不要人工抽检）。audit 脚本自身出 PASS/FAIL 判据。

ported `audit_roi_orientation.py` 增强：算完质心后**自动判定**并 emit verdict（不再只产 artifacts 等
人看）。判据（全过 = PASS，任一违反 = FAIL/block）：

1. **cranial/caudal（x 轴）**：`flip_cranial_to_left=True` 后 cranial = 左 = 低 x index。
   - HZAU 各例 **head 质心 x ∈ 左半**（cranial 侧）。
   - TB 各例 **testis 质心 x ∈ 右半**（caudal 侧）。
2. **lower/bed（y 轴）**：TB 各例 **testis 质心 y 跨例一致落在某一侧**（bed/ventral 侧）→ 该侧即
   "lower"，坐实 `lower_side_default=bottom` + 为 `testis_caudal_lower_half` ROI 定向。

- PASS → 写 `--orientation-verified` 凭证，放行 projection 生成；FAIL → block，打印违反 case_id + 详情。
- audit json 保留全部 per-case 质心（verdict 可复算/可事后审查），虽全自动但完全可审计。
- 头/睾丸 label id 走 config（G10：9/6），centroid 计算依赖之。

### 🔧 G5. resnet18 权重 SHA（技术 pin）
- URL `https://download.pytorch.org/models/resnet18-f37072fd.pth`（~45MB）；本地 Mac 下载算 SHA256 记录；rsync 到 Huawei `~/.cache/torch/hub/checkpoints/`。job 内 `weights=IMAGENET1K_V1` 走缓存。

### 🔧 G8 / G13. 运行 / 拉取位置（技术 pin）
- presence 扫描 + manifest 生成在 Huawei（CPU）；manifest rsync 回本地进 git。
- 小报告 rsync 拉到本地 `swine-CT-article/runs/roi_presence_classifier/`（gitignored）；大文件按 CLAUDE.md EXCLUDES 不拉，不加 `--delete`。

### 🔧 G9. package 结构（技术 pin）
- `tools/roi_classifier/` 作 package：加 `__init__.py`；保留原脚本 `from .common import` 相对导入 + fallback。

### 🔧 G10. config 必改字段（技术 pin，但 label id 是必设）
- `labels.head_label_id: 9`、`labels.testis_label_id: 6`（原 `null`，**必设**）。
- `model_grid.expected_model_count: 60 → 30`；`split.reuse_previous_6_2_2: true → false`（用 article split）；`roi.roles` 砍 wrong/control roles（对齐 D16）；`project.output_root` → article `runs/roi_presence_classifier/`；`common.py` 的 `SPEC_PATH`/`DEFAULT_OUTPUT_ROOT` → article。

### 🔧 G14. `summarize` 改造范围（技术 pin，依赖 G1）
- 若 G1 保留 summarize：去 `run_roi_controls` 依赖、加 G3 跨 grid best 选择、加 G2 test 报告、产出 result card（D23 口径）。

---

## 附：文件命名

`docs/roi-presence-classifier-spec-draft.md`（不带 `v1-` 前缀，因这是 auxiliary ResNet-18 2D
classifier，非 nnU-Net v1 segmentation）。冻结正式 spec 时 → `docs/roi-presence-classifier-spec.md`。
