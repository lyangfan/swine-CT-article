# ROI/Half-Projection Presence Classifier — 正式实施 Spec（article 自有 split）

Status: **FORMAL**（由 `roi-presence-classifier-spec-draft.md` 冻结；D1-D29 + G1-G6 全部决策已固化）

Date: 2026-06-23

本 spec 是**可直接落地执行**的协议。所有选项已在 draft 阶段拍板，此处以**固定规程**呈现，
实施者照做即可，不再列选项。技术约束（只有一种正确做法）标注 🔧。

---

## 1. 目标与 scope

在 article 仓库自有的固定 split（`data/splits/split_manifest.csv`，seed 42，**train 120 / val 38 /
test 39**）上，训练一个**分类准确**的 head / testis presence classifier（2D lateral projection +
CT-image-only 前景 bbox + 解剖 ROI crop + endpoint-specific ResNet-18 二分类）。

- **接受 shortcut**：本数据集 `presence ≡ source`（§2），AUPRC≈1.0、FA≈0 近乎必然；准确性反映
  source/FOV 信号而非解剖定位。报告以一句话 caveat 说明，**不**做 shortcut 诊断、**不**跑控制组。
- **不改** nnU-Net segmentation 训练/推理。
- classifier 产物**用途 = 下游 segmentation gate 输入**（handoff 实验 C 路线）；但 gating 的实施/评估
  是**独立后续 spec**，不在本 spec 范围。本 spec 不主张 segmentation 改善或 champion。

---

## 2. 核心事实：presence ≡ source（结构性）

来源：`data/README.md §149` + CLAUDE.md 数据节。

- **HZAU 93 例**（阉猪）：head-present / testis-absent → `head_present=1 ⟺ source==HZAU`
- **TB 104 例**（公猪）：head-absent / testis-present → `testis_present=1 ⟺ source==TB`

逐例完全相等，无 within-source 变化。这是数据结构决定的事实，**换 split 不改变**。label class id：
**head = 9、testis = 6**。

---

## 3. 环境与路径

| 项 | 值 |
|---|---|
| article 仓库（本地 canonical） | `/Users/liuyangfan/Documents/work/CT/swine-CT-article` |
| article 仓库（Huawei 镜像） | `/home/share/hzau/home/liuyangfan/swine-CT-article` |
| 输出 root（Huawei） | `<article 镜像>/runs/roi_presence_classifier/` |
| 既有实现（参考，**不**直接用） | `AutoScientists/output/swct06042040/task/tools/roi_half_projection_presence_classifier/` |
| conda env | `swine_ct_autonomous_discovery/envs/nnunetv1`（torch 2.4.1+cuda118 / torchvision 0.19.1 / cuDNN 8600，实测与原 run 一致） |
| module stack | `module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.6.0_cuda11` |
| 默认 base python | anaconda3 base **无 torch**，必须用上面 env + module load |
| 外网 | **Huawei 节点无外网**（DNS 失败）→ resnet18 权重必须预放置（§11） |

**图像/label 路径**（F1 已核查）：article 仓库在 Huawei 已物化
`data/{train,val,test}/{images,labels}/`，计数全对（images+labels 各 120/38/39），每个文件是软连接 →
canonical 存储 `/home/share/hzau/whcs-share37/liuyangfan/nnunet_medsam_semisup/data/labeled_197/`。
HZAU 命名 `07069186.nii.gz`、TB 命名 `136021_122090.nii.gz`。`case_metadata` 里的 `/workspace/...`
（hzau_gpu 路径）在 Huawei 不存在，**弃用**。

---

## 4. 数据、split、标签、派生 manifest

### 4.1 split（冻结，C1）

用 article `data/splits/split_manifest.csv`（seed 42，120/38/39；HZAU 56/18/19、TB 64/20/20），
**不得改动**。split 取值为 `train / val / test`（注意是 `val`，不是 `validation`）。

### 4.2 派生 classifier manifest

原 split manifest 缺 classifier 要用的列 → 生成派生 manifest
**`data/manifests/classifier_split_manifest.csv`**（git-tracked，纯小文件）。脚本：
**`tools/roi_classifier/make_classifier_manifest.py`**（new）。

**schema**（9 列）：

```
case_id, split, source, source_detail, breed_en, image_path, label_path, head_present, testis_present
```

- `split` ∈ {train, val, test}；覆盖全部 197（含 test，供 §9 一次性评估）。
- `image_path` / `label_path`：**相对仓库根**，由 `split + case_id` 拼接：
  `data/{split}/images/<case>.nii.gz`、`data/{split}/labels/<case>.nii.gz`。🔧 **不读** `case_metadata`
  的 `/workspace` 路径。工具用 `--repo-root` 解析（默认 article Huawei 镜像根）。
- `source` / `source_detail` / `breed_en`：从 `split_manifest.csv` join。

### 4.3 presence 标签（GT 扫描）

`head_present` / `testis_present` 由 **GT label 体素扫描**派生（D7）：SimpleITK 读 label，
class 9 (head) / class 6 (testis) voxel count > 0 → 1，否则 0。

**一致性检查**（强制）：GT 派生 presence 必须**逐例 == source 派生**（head_present==source==HZAU、
testis_present==source==TB）。**违反 → 打印 case_id + 详情、退出非 0、不写 manifest**，触发人工核查
（预期 0 违反；presence≡source 是前提）。

🔧 运行位置：Huawei（label 只在 Huawei），CPU 步骤；产出 rsync 回本地进 git。

---

## 5. Orientation audit（自动 PASS/FAIL）

脚本：`tools/roi_classifier/audit_roi_orientation.py`（port + 增强）。在 article 仓库**重跑**全 197 例
（用 §4.2 派生 manifest + article 图像/label 路径；head/testis label id 走 config = 9/6）。

**自动判据**（全过 = PASS，任一违反 = FAIL/block）：

1. **cranial/caudal（x 轴）**：投影 x 轴 = cranial→caudal，`flip_cranial_to_left=True` 后 cranial = 左 = 低 x index。
   - HZAU 各例 **head 质心 x ∈ 左半**（cranial 侧）。
   - TB 各例 **testis 质心 x ∈ 右半**（caudal 侧）。
2. **lower/bed（y 轴）**：TB 各例 **testis 质心 y 跨例一致落在某一侧**（bed/ventral 侧）→ 该侧即 "lower"，
   坐实 `lower_side_default=bottom` 并为 `testis_caudal_lower_half` ROI 定向。

- PASS → 写 `--orientation-verified` 凭证，放行 §6 projection 生成；FAIL → block + 打印违反 case_id。
- audit json 保留全部 per-case 质心（verdict 可复算/事后审查），**全自动但完全可审计**，不做人工抽检。

---

## 6. ROI 投影生成

脚本：`tools/roi_classifier/generate_roi_projections.py`（port + 改）。

**全部重新生成**（D11；不复用 swct06042040 旧 `.npy`）：train+val 158 + test 39 = 197 例。
原 157 例旧 `.npy` 不引用、不拷贝。

**参数（verbatim，D12）**：foreground `HU > -800`、CT bed 计入、2D `binary_fill_holes`、2D 连通域清理
`min_component_area_ratio=0.001`、bbox padding `0.05`、ROI 50/50 cranial-caudal、resize-with-padding
`224×224`、`padding_value=0`、bilinear、单通道复制为 3 通道。

🔧 投影方向：SimpleITK 读出 z,y,x，沿 **axis 2（x = 解剖 L-R）** 投影（image 属性，与 split 无关）。

**ROI roles（仅 correct，D16 砍掉 wrong/control）**：

- head：`head_cranial_half`
- testis：`testis_caudal_half`、`testis_caudal_lower_half`

**variants（5）**：`muscle_only_mean`、`muscle_only_p90`、`foreground_thickness`、`bone_only_mip`、`multi_channel_compact`。

**test 投影（G2）**：原 `generate` 硬禁 test → 加 **`--include-test` gated flag**。默认（无 flag）仍 forbid
test；仅在里程碑⑥ + 显式 `--include-test` 时生成 test 投影。val token 适配（D6）。

🔧 生成可多 worker 并行（~30min 量级）；产出 **2955 `.npy`**（3 correct roles × 5 variants × 197 cases；
投影按 endpoint/role/variant 共享，不乘 init），约 **1.7 GB**（每个 588 KiB）。

---

## 7. 训练（30-model grid）

脚本：`tools/roi_classifier/train_roi_classifier.py`（port + 改：val token；grid=30）。

**grid（D16）**：3 correct ROI roles × 5 variants × 2 init（imagenet, random）= **30 models**
（head 1×5×2=10；testis 2×5×2=20）。砍掉 3 个 wrong/control roles。

**超参（verbatim，D18）**：ResNet-18、endpoint-specific binary、AdamW（imagenet lr=3e-4 / random lr=1e-3、
wd=1e-4）、cosine + 5ep warmup、max 100ep、early stop patience 15、batch 16、BCEWithLogitsLoss + capped
`pos_weight`（cap 5.0；article split 下 pos_weight≈1.14/0.88，不触发 cap）、AMP、deterministic cuDNN。

**seed**：`20260520`（单 seed，D4/D17）。

**per-model best（D19）**：best = val AUPRC 最高 subject to **FA≤1 safety gate**（FA=false-absent）。退化下
原逻辑仍合理（FA 优先 + 首达峰值胜出）。**额外报告**选中模型 Brier / ECE（calibration，不纳入选择）。

🔧 `common.py` 常量：`ROI_ROLES_BY_ENDPOINT`（head 1 role、testis 2 roles）、`EXPECTED_MODEL_COUNT=30`、
`FIRST_ROUND_SPLITS={"train","val"}`、reserved=`"test"`、`WRONG_OR_CONTROL_ROI_ROLES` 清空（或移除引用）。

---

## 8. 跨 grid 选"the classifier"（G3）

脚本逻辑落在 `summarize_roi_results.py` 改造里（G1）。

**每 endpoint 选一个**最终 classifier：val AUPRC 最高且过 FA≤1 gate；**tiebreak：FA↓ → Brier↓ →
确定性字典序（role, variant, init）**。选中模型 → §9 test 评估 + 下游 gate（独立 spec）。

---

## 9. Test 一次性评估（D3 / G2）

**test-sealed 用于选择**；冻结 §8 best model 后，对 test **一次性**预测 + 评估：

- **test 投影**：`generate_roi_projections.py --include-test`（仅此步，gated）。
- **预测+评估**：**`tools/roi_classifier/eval_test.py`**（new；import train 的 `_build_resnet18` /
  `_metrics` / `_temperature_summary` / `RoiProjectionDataset`）→ 加载 §8 冻结 best checkpoint → 预测 test
  投影 → 算指标。
- **指标**：= val 同款（AUPRC / AUROC / Brier / ECE / FA / FP / sens / spec）+ calibration（temperature）；
  **阈值 0.5**（verbatim）。
- **test 不得参与模型选择**（§13 NO-GO）。

`summarize_roi_results.py` 读 `eval_test.py` 输出，汇总最终 result card。

---

## 10. 控制组与结论口径

- **控制组：不跑**（D22）。`run_roi_controls.py` 不 port。source-only / shape-only / correct-vs-wrong ROI
  全部不做。
- **结论口径（D23，中性实用）**：主轴 =「训练得到准确的 head/testis presence classifier，train/val/test
  近乎完美分离」；一句话 caveat：「本数据集 presence≡source，故准确性反映 source/FOV 信号而非解剖定位」。
  不主张 gating / champion（gating 留独立后续 spec，D29）。

---

## 11. resnet18 权重预放置（D21 / G5）

Huawei 无外网 → imagenet init 必须预放置权重：

1. 本地 Mac（有网）下载 `https://download.pytorch.org/models/resnet18-f37072fd.pth`（~45MB）。
2. 算 SHA256，**必须等于** `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
   （官方 resnet18 IMAGENET1K_V1；文件名 `f37072fd` 即此 hash 前 8 位，torchvision 命名约定）—— 证清白、非自定义权重。
3. rsync 到 Huawei `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`。
4. job 内 `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` 走缓存（离线）。

random init 不需要权重；D16 双 init（imagenet + random）均保留。

---

## 12. 产物布局、本地 mirror、GitHub

### 12.1 Huawei 输出 root（D15 / D25）

`<article 镜像>/runs/roi_presence_classifier/`，子结构：

```
data/manifests/  data/projections/<STAMP>/  data/previews/<STAMP>/
models/batch_<STAMP>/  reports/<STAMP>/  jobs/batch_<STAMP>/
```

不加 `article_split/` 子层；stamp（`batch_<STAMP>`）区分每次 run。

### 12.2 本地 mirror（D26）

mirror **小报告**到本地 `swine-CT-article/runs/roi_presence_classifier/`（gitignored 核查副本）：
`training_summary.json`、per-model `metrics_summary.json`/`run_manifest.json`、`training_history.csv`、
train/val/test `predictions.csv`、orientation audit json/md、projection manifest csv。

**不** mirror：`.npy` 投影（~1.7 GB）、`.pt` checkpoint、preview PNG pack。rsync 拉用 CLAUDE.md 通用
EXCLUDES **并额外 `--exclude='*.npy'`**（🔧 CLAUDE.md 通用 EXCLUDES 只含 `.npz`、**不含 `.npy`**；不显式
排除会把 ~1.7 GB 投影 `.npy` 拉回本地，与"不 mirror `.npy`"冲突）；**不加 `--delete`**。

### 12.3 GitHub（D27）

- **进 git**：`docs/roi-presence-classifier-spec*.md`、`tools/roi_classifier/`（脚本+config）、
  `data/manifests/classifier_split_manifest.csv` + 生成脚本。
- **不进 git**（`runs/` 整个 gitignored）：投影 `.npy`、checkpoint `.pt`、preview PNG、mirror 的小报告。
- **`.gitignore` 增项**：`runs/`、`*.npy`（已有 `*.pt`/`*.pth`）。

---

## 13. 调度与 runtime（D20 / D21）

- **GPU job（训练 Stage 5 + test 推理 Stage 6）**：单 GPU 顺序 job，resource `cpu=8;mem=64000;gpu=1`，
  node `agent<170` 且**不用** `whshare-agent-174`，loop dsub，early-runtime 检查（dsub 后 1-2min `djob`，
  FAILED 立刻读 `.err` 修）。**node 约束对所有 GPU job 生效**（训练 + test 推理），非仅训练。
- **调度结构**：CPU 阶段（§4 manifest 生成、§5 orientation audit、§6 projection 生成、§8 选 best + 报告
  汇总）各为/合并为 CPU job；GPU 阶段（§7 训练 + §9 test 推理）为 GPU job；阶段间 REVIEW/检查 gate 串行。
- 提交：`dsub -pn '!whshare-agent-174' -s <job_script>`（或脚本内 `#DSUB -pn !whshare-agent-174`）。
  🔧 `-pn` 谓词**只能排除 174，无法表达 `agent<170`** → 提交后必须用 `djob <job_id>` 读**实际执行节点**
  （`TASK_EXEC_NODES`）；若落在 suffix ≥170 或 =174 → `djob -T <job_id>` 终止 + 修复重提
  （handoff §4.3 同款做法，fold4 曾因节点问题重排）。early-runtime 检查即含此核验。

---

## 14. 脚本 inventory（G1 / G9）

`tools/roi_classifier/`（作 package，加 `__init__.py`；保留 `from .common import` 相对导入 + fallback）：

| 脚本 | 处置 |
|---|---|
| `common.py` + `default_config.yaml` | port + 改（§15） |
| `audit_roi_orientation.py` | port + 增强（§5 自动判据） |
| `generate_roi_projections.py` | port + 改（`--include-test` flag §6；val token） |
| `train_roi_classifier.py` | port + 改（val token；grid=30 §7） |
| `summarize_roi_results.py` | 保留 + 改造（去 control 依赖；加 §8 跨 grid best；读 §9 test 输出；产 result card） |
| `run_roi_controls.py` | **丢**（D22） |
| `make_classifier_manifest.py` | **new**（§4） |
| `eval_test.py` | **new**（§9） |

## 15. config 必改字段（G10）

`default_config.yaml` + `common.py`：

- `labels.head_label_id: 9`、`labels.testis_label_id: 6`（原 `null`，**必设**）。
- `model_grid.expected_model_count: 30`。
- `split.reuse_previous_6_2_2: false`（用 article split）。🔧 注：`split.seed` 字段此时 **vestigial** ——
  split 一律取自 §4.2 派生 manifest（article `split_manifest.csv`，seed 42），**不由 config 重生成**；
  训练 seed 是 `training.seed: 20260520`（不同概念，§7）。
- `roi.roles`：砍 wrong/control roles（对齐 §6：head 1、testis 2 correct roles）。
- `project.output_root` → article `runs/roi_presence_classifier/`。
- `common.py`：`SPEC_PATH` / `DEFAULT_OUTPUT_ROOT` → article 路径；`EXPECTED_CASE_COUNT=197` 保留。

---

## 16. 实施 Stage（D28；每 stage 产物 + gate PASS 才进下一 stage）

> 原则：**test 在 Stage 6（模型冻结后）才被触碰**；Stage 0-5 不得生成任何 test 投影 / 预测 / 指标。
> （Stage 2 manifest 会读 test 的 **GT label** 派生 presence 标签 —— 这是最终评估的 ground truth，不构成
> 对 test 的"触碰"；禁令只针对 test **投影 / 预测 / 指标**。）
> 每 stage 的 gate（自动检查 / REVIEW）通过后才进下一 stage；FAIL → 修复重跑该 stage。

### Stage 0 — resnet18 权重预放置（prep）
- **运行**：本地 Mac（有网）→ rsync 到 Huawei。
- **动作**：下载 `https://download.pytorch.org/models/resnet18-f37072fd.pth`（~45MB）；算 SHA256 记录；
  rsync 到 Huawei `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`。
- **产物**：Huawei 缓存里的权重文件 + 记录的 SHA256。
- **gate**：文件存在；**SHA256 == `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`**
  （官方 resnet18-f37072fd，§11）。

### Stage 1 — port 工具 + config（实施）
- **运行**：本地 Mac（canonical）→ rsync 到 Huawei。
- **动作**：port `common.py`/`default_config.yaml`/`audit_roi_orientation.py`/`generate_roi_projections.py`/
  `train_roi_classifier.py`/`summarize_roi_results.py` 到 `tools/roi_classifier/`（加 `__init__.py`，保留
  `from .common import` 相对导入）；按 §15 改 config 字段；按 §7 改 `common.py` 常量
  （`ROI_ROLES_BY_ENDPOINT` 砍到 correct、`EXPECTED_MODEL_COUNT=30`、`FIRST_ROUND_SPLITS={"train","val"}`、
  reserved=`"test"`）；new `make_classifier_manifest.py`（§4）、`eval_test.py`（§9）；丢 `run_roi_controls.py`。
- **产物**：`tools/roi_classifier/` package（进 git，§12.3）。
- **gate**：static 检查全过 —— `python -m py_compile *.py`；每个入口 `--help` 正常；
  `make_classifier_manifest`/`generate`/`train` `--dry-run` 通过；**`.gitignore` 已含 `runs/` 与 `*.npy`**
  （§12.3；须在 Stage 4 产 `.npy` 前完成 —— CLAUDE.md 通用 rsync EXCLUDES 不含 `*.npy`，否则 rsync+`git add`
  可能把 ~1.7 GB `.npy` 误入仓库）；rsync 到 Huawei。

### Stage 2 — 派生 classifier manifest（数据 prep）
- **运行**：Huawei（CPU）。
- **输入**：article `data/splits/split_manifest.csv` + `data/{split}/labels/*.nii.gz`。
- **动作**：`make_classifier_manifest.py --repo-root <article Huawei 镜像>` → SimpleITK 扫 GT label
  class 9/6 体素派生 presence；join split/source/source_detail/breed_en；拼相对 image_path/label_path
  （`data/{split}/{images,labels}/<case>.nii.gz`）。
- **产物**：`data/manifests/classifier_split_manifest.csv`（**197 行，9 列**）；rsync 回本地进 git。
- **gate**：197 行；schema 正确；**一致性检查** GT-presence == source-presence 逐例（预期 0 违反；
  违反 → stop + 退出非 0 + 不写 manifest）。

### Stage 3 — orientation audit（自动 PASS/FAIL）
- **运行**：Huawei（CPU）。
- **输入**：Stage 2 manifest + 图像/label。
- **动作**：`audit_roi_orientation.py`（全 197；head/testis label id 走 config = 9/6）。
- **产物**：audit json（per-case 质心）+ 自动 verdict + `--orientation-verified` 凭证。
- **gate**：**自动 verdict PASS**（head 质心 x ∈ cranial 半 / testis 质心 x ∈ caudal 半 / testis 质心 y
  跨例一致 = lower 侧）。FAIL → block，排查后再跑。

### Stage 4 — 投影生成 train/val（无 test）
- **运行**：Huawei（CPU，可多 worker 并行）。
- **输入**：Stage 2 manifest + 图像 + Stage 3 `--orientation-verified`。
- **动作**：`generate_roi_projections.py`（**不带** `--include-test`）→ train+val 158 例投影。
- **产物**：`data/projections/<STAMP>/` 下 **2370 `.npy`**（= 158 cases × 3 roles × 5 variants）+
  projection manifest + previews。
- **gate**：projection rows = 2370；split ∈ {train, val}（**0 test**）；preview 存在；
  `uses_gt_input=0`、`test_consumed=0`。

### Stage 5 — 训练 30 models
- **运行**：Huawei（GPU），单 GPU 顺序 job（§13）。
- **输入**：Stage 4 train/val 投影 + Stage 2 manifest（labels）+ Stage 0 权重。
- **动作**：`train_roi_classifier.py` → 3 roles × 5 variants × 2 init = **30 models**（seed 20260520，
  超参 §7；per-model best = val AUPRC under FA≤1 gate）。
- **产物**：`models/batch_<STAMP>/` 下 30 个模型目录（`model_best.pt`/`model_final.pt`、
  `metrics_summary.json`、`training_history.csv`、train/val `predictions.csv`）。
- **gate**：scheduler SUCCEEDED；30 目录齐全；实际 node <170 且非 174（§13，用 `djob` 核验，落 bad node 则
  `djob -T` + 重提）；**0 test 输出**。

### Stage 6 — 选 best → 冻结 → test 评估 → 报告
- **运行**：Huawei（选 best / test 投影 = CPU；**test 推理 = GPU**，与 §13 GPU job 一致）。
- **输入**：Stage 5 的 30 模型 + Stage 2 manifest（含 test）+ 图像。
- **动作（按序；冻结边界在第 1 步后）**：
  1. `summarize_roi_results.py` 跨 grid 选 best（§8：val AUPRC(FA≤1) → FA↓ → Brier↓ → 字典序）→
     **每 endpoint 一个冻结 best**：复制选中 `model_best.pt` → `models/batch_<STAMP>/frozen/<endpoint>.pt`
     + 写 `reports/<STAMP>/best_selection.json`（记录 endpoint / role / variant / init / checkpoint 路径 /
     val 指标）（head 1、testis 1）。
  2. **冻结后**：`generate_roi_projections.py --include-test` → test 39 例投影（**585 `.npy`** = 39×3×5）。
  3. `eval_test.py`（加载冻结 best）→ 预测 test → 指标（val 同款 + calibration，thr 0.5）。
  4. `summarize_roi_results.py` 汇总 → 最终 result card（§10 口径）。
- **产物**：2 个冻结 best 模型 + test 指标 + result card（**含 Stage 3/4 小产物**在内的小报告 mirror 回本地，§12.2）。
- **gate**：每 endpoint 恰好 1 best（`frozen/<endpoint>.pt` + `best_selection.json` 齐全）；test 投影 = 585
  且仅 test split；test 指标一次性、未参与选择；报告含 ≡source caveat（§10）；**test 推理实际 node <170
  且非 174**（§13，与 Stage 5 同口径；用 `djob` 核验，落 bad node 则 `djob -T` + 重提）。

**投影总计**：Stage 4 (2370) + Stage 6 (585) = **2955 `.npy`** ≈ 1.7 GB（与 §6 一致）。

---

## 17. NO-GO

- orientation 自动判据 FAIL（§5）；
- GT mask 用于构造 per-case ROI input（foreground / bbox / ROI 边界 / classifier 像素）；
- 30-model grid 缺模型（§7）；
- **test 在模型冻结前被触碰**（§9 只允许冻结 best 后的一次性 test 投影/预测/评估，test 不得参与选择）；
- job 落 node ≥170 或 `whshare-agent-174`；
- imagenet init 在未预放置 resnet18 权重的情况下跑（§11，Huawei 无外网）。

---

## 18. Downstream（独立后续 spec，不在本 scope）

D29=(b)：classifier 产物计划接 article segmentation 做 gate（handoff 实验 C 路线：pre-argmax probability
transform、article split 上 head-absent FP→0 评估、副作用）。该 gating 实验**单独 spec**；本 spec 只产出
classifier。因 presence≡source，gate 会"有效"但靠 source 信号 —— downstream 报告须写明。
