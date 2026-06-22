# CLAUDE.md

## Huawei Server

```sshconfig
Host paca_share 27.18.114.38
  HostName 27.18.114.38
  Port 22
  User liuyangfan
  IdentityFile ~/.ssh/paca_share_ed25519
  IdentitiesOnly yes
```

- Connect with `ssh paca_share`, or `ssh 27.18.114.38`.
- Verified in this project context: hostname `whshare-ccs-cli-2`, user
  `liuyangfan`, home `/home/share/hzau/home/liuyangfan`.
- Scheduler commands should be run from local Mac as:
  `ssh paca_share 'bash -lc "<command>"'`.
- DSUB commands:
  - submit: `dsub -s <job_script>`
  - query: `djob <job_id>`, `djob -L <job_id>`, or `djob -D`
  - terminate: `djob -T <job_id>`
- **提交作业后必须做 early runtime 检查**:`dsub` 后等 ~1-2min,跑 `djob <job_id>`
  看状态。**若 FAILED,立刻读 `.err` 日志定位原因并修复**(常见:flag 拼写/单双横、
  缺 `module load`、缺 lib、缺执行权限),**修好再重提**;不要在没确认 RUNNING 的情况
  下就长时间等待。长任务也应每隔 ~10min 复查一次状态,避免白等一个早已 FAILED 的作业。
- Do not schedule PACA jobs on `whshare-agent-174`; use
  `dsub -pn '!whshare-agent-174' -s <job_script>` or
  `#DSUB -pn !whshare-agent-174`.
- Huawei CUDA jobs should load the verified stack: GCC 9.3.0, CUDA 11.8,
  cuDNN 8.8.1 CUDA11, NCCL 2.16.5 CUDA11.8.
- Canonical Huawei swine CT AutoScientists root:
  `/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery`.
- For future Huawei nnU-Net v2 swine CT training, keep the native v2 policy:
  `.b2nd` Blosc2 preprocessed data plus nnU-Net v2 CUDA default AMP. Do not add
  `--unpack-data` to `nnUNetv2_train`; that option is not part of the v2 train
  CLI in this environment.
- MedSAM low-label formal v2 root:
  `/home/share/hzau/home/liuyangfan/medsam_lowlabel_v2`.
- Deprecated historical root:
  `/home/share/hzau/home/liuyangfan/nnunet_medsam_semisup`; use only for
  historical evidence or explicit migration.
- Legacy PACA root:
  `/home/share/hzau/home/liuyangfan/paca`; use only when explicitly referenced.

## swine-CT-article 镜像同步（Huawei ↔ 本地 Mac）

- Huawei 端：`/home/share/hzau/home/liuyangfan/swine-CT-article/`
- 本地 Mac 端：`/Users/liuyangfan/Documents/work/CT/swine-CT-article/`

### 工作流：本地优先

- 新内容（README、docs、scripts、manifests、CSV、JSON 等）**先在本地 Mac 实现**，
  再推到 Huawei。
- 本地是这些小文件的 canonical 副本，Huawei 是镜像。
- 当且仅当某个文件**必须依赖 Huawei 端数据/算力生成**（例如 `case_metadata.csv`
  是 join Huawei 端的 phe/inventory 产出的），才允许先在 Huawei 上写，然后
  pull 回本地。这种情况下该文件在本地也是 canonical。

### 大文件只留 Huawei

- 不论方向，CT/label `.nii.gz`、预处理 `.b2nd`/`.pkl`、模型权重
  `.pt/.ckpt/.model/.pth`、softmax `.npz` 等大二进制**永远不进本地**。
- `images/` 和 `labels/` 目录在 Huawei 上是软连接，sync 时整个跳过，不复制符号
  连接，也不 dereference 拷贝实际影像。本地需要查文件清单读
  `manifests/case_metadata.csv`（含每例 `image_path`、`label_path`、size）。

### rsync 命令

通用 exclude 列表（两个方向共用）：

```bash
EXCLUDES=(
  --exclude='images/'
  --exclude='labels/'
  --exclude='*.nii.gz'
  --exclude='*.b2nd'
  --exclude='*.pkl'
  --exclude='*.npz'
  --exclude='*.pt'
  --exclude='*.ckpt'
  --exclude='*.model'
  --exclude='*.pth'
  --exclude='.DS_Store'
)
```

**推：本地 → Huawei**（默认方向，**不加 `--delete`**，避免误删 Huawei 端独有的
产物）：

```bash
mkdir -p /Users/liuyangfan/Documents/work/CT/swine-CT-article
rsync -av "${EXCLUDES[@]}" \
  /Users/liuyangfan/Documents/work/CT/swine-CT-article/ \
  paca_share:/home/share/hzau/home/liuyangfan/swine-CT-article/
```

**拉：Huawei → 本地**（仅在 Huawei 生成了新小文件时使用，**不加 `--delete`**，
避免本地未推完的草稿被覆盖）：

```bash
rsync -av "${EXCLUDES[@]}" \
  paca_share:/home/share/hzau/home/liuyangfan/swine-CT-article/ \
  /Users/liuyangfan/Documents/work/CT/swine-CT-article/
```

只在双方都确认对方没有未保存改动时，才在某一个方向上加 `--delete` 做严格
mirror；默认两方向都不加。

### 注意事项

- 提交前如要 push，先 `rsync -avn`（dry-run）看一遍将要传什么。
- 新增大文件类型时（`.safetensors`、`.h5` 等），追加到 `EXCLUDES`。
- **普通的临时 / 探索性脚本不要留在本目录**；只有用在实验里、需要长期保留和
  复现的**正式脚本**才写进本目录（如 `data/manifests/` 下那几个生成 / patch
  manifest 的脚本）。一次性试探脚本用完即删，或放到本仓库之外。
- 不要把 AutoScientists、PACA、`swine_ct_autonomous_discovery` 的数据 / 输出
  混进本目录；那些 workspace 各自有自己的同步规则。

## 数据集（197 例 labeled swine CT，9 类胴体分割）

详细文档在 `data/README.md`；本节只列**影响实验**的关键事实，深入时再读 README。

- **2 个 source / class presence**（最重要的分层维度）：
  - `HZAU` 93 例：全 Yorkshire、阉猪 → **head-present / testis-absent**
  - `TB` 104 例：Yorkshire / Landrace / Pietrain / Duroc 各 26、公猪 → **head-absent / testis-present**
  - head 只能在 HZAU 上评、testis 只能在 TB 上评，所以任何 split 都必须让两个
    source 在 train/val/test 里按比例出现。
- **品种**：canonical **4 分类**，EB5 已折入 Duroc；HZAU 品种来自屠宰测定表
  （非 CT 目录自带），TB 品种来自 DICOM 目录结构。
- **label class**（label NIfTI 里的整数，共 9 类前景 + background）：
  `0` background / `1` front / `2` middle / `3` end / `4` left_kidney /
  `5` right_kidney / `6` testis / `7` thoracic_cavity /
  `8` abdominal_and_pelvic_cavity / `9` head。其中 `6 testis` 仅 TB 有、
  `9 head` 仅 HZAU 有（条件性 class，见上 source 条）；front/middle/end/kidney×2/
  cavity×2 两 source 都有。定义文件
  `hzau_gpu:/workspace/data/CT/HZAU_veterinary_hospital/label/labels.json`。
- **固定 split**（一次性确定，所有实验共用，不得改动）：6:2:2、seed 42 →
  **train 120 / val 38 / test 39**。TB 按品种分层（每品种 16/5/5），HZAU 纯随机。
  canonical 定义在 `data/splits/split_manifest.csv`；Huawei 上物化成
  `data/{train,val,test}/{images,labels}/` 软连接。**test 冻结**（仅最终评估），
  val 用于模型选择。重新生成跑 `data/splits/make_split.py`（逐字节可复现）。
- **文件分布**：本地只有 `data/manifests/`（`case_metadata.csv` 等）+
  `data/splits/`（split 定义 + 脚本）+ `data/README.md`，都是小文件、canonical；
  影像 `images/`、`labels/` 和 train/val/test 软连接目录**只在 Huawei**。
  本地不存任何 `.nii.gz`，查文件清单直接读 `case_metadata.csv`。

## GitHub

- 仓库：`https://github.com/lyangfan/swine-CT-article`
- SSH remote（本地配置走 socks5 代理，已验证可用）：
  `git@github.com:lyangfan/swine-CT-article.git`
- 默认分支 `main`；本仓库本地即 canonical，Huawei 是数据镜像，GitHub 是
  公开版本控制。
- 提交流程：`git status --short` → `git add <files>` →
  `git commit -m "<message>"` → `git push origin main`。
- **`.phe` 等 PLINK 风格育种数据已 gitignore**（含可识别场名 / 个体耳号 /
  育种值，不公开）。这些文件仍保留在本地与 Huawei，只是不进 GitHub。
  新增敏感数据类型时同样追加到 `.gitignore`。
- 大文件（CT / 权重 / 预处理数组）由 `.gitignore` 拦截，永不进仓库；本地本身
  也不存这些（见上文同步规则）。

## HZAU Server

```sshconfig
Host hzau_gpu
  HostName 211.69.141.179
  Port 2225
  User root
```

- Connect with `ssh hzau_gpu`, or `ssh -p 2225 root@211.69.141.179` if the
  alias is unavailable.
- Verified in this project context: hostname `565d8e27d8a8`, user `root`.

## 启动独立 subagent

当需要"真正独立"的 subagent(全新进程、零父会话上下文,如独立审查 / 复核)时:

- 本 harness 暴露的 `Task*` 工具(`TaskCreate` 等)**只是待办追踪器**,不执行任何东西;
  会话内的 `Task(subagent_type=…, prompt=…)` 在这里调不到。
- **用 Bash 调 `claude` CLI 起子进程**(本机 `/usr/local/bin/claude`,v2.1.185):

  ```bash
  claude -p "$(cat /path/to/self-contained-prompt.md)" \
         --model sonnet --dangerously-skip-permissions
  ```

- `-p`=非交互输出;`--dangerously-skip-permissions`=让 ssh/bash 无人值守跑(不加会卡权限提示);
  `--ephemeral` 在此版本**不支持**(报 unknown option),别加。
- prompt **必须自包含**:子进程是新 session,看不到本会话任何上下文 —— 路径、
  期望值、ssh 别名(`paca_share`)、只读约束全要写进去。
- print 模式跑完才一次性输出;长任务会自动转后台,完成时通知。重 I/O 任务(如
  逐个读 NIfTI)可能几十分钟,可在 prompt 里要求"合并成一次 ssh + 抽样验证"加速。
- 源参考:AutoScientists `source_code_claude/system/reference/AGENT-SETUP.md`。

## 统一训练 / 预测入口

所有网络(5 个)共用同一套入口,只通过 `--network` / `--network-dim` 区分:

### 训练:`framework/train.py`

```bash
python -m framework.train --network <net> --seed <seed> \
    --config configs/<net>.yaml --task-id 601 --fold 0 \
    --output-folder <out>
```

3D 网络(nnunet_v1 / swinunetr / segformer3d / mednext_s / mednext_l)走 `MultiNetworkTrainer`;
2D 网络(nnunet_2d)不走 framework,走 `train_paca_deterministic.py --network 2d`(原生
nnUNetTrainerV2_2D)。

### 预测:`framework/predict.py`(统一入口,5 个网络全走这一个)

```bash
# 3D 网络
python -m framework.predict --network <net> --seed <seed> \
    --checkpoint <ckpt> --input-folder imagesTs --output-folder <pred>

# 2D 网络(多一个 --network-dim)
python -m framework.predict --network nnunet_v1 --network-dim 2d --seed <seed> \
    --checkpoint <ckpt> --input-folder imagesTs --output-folder <pred>
```

全部统一走:`MultiNetworkTrainer.initialize(training=False)` → `load_checkpoint_ram`
(只加载权重,不需要 `restore_model`)→ sliding window(overlap 0.5, TTA off)→
argmax → resample 回原 spacing → nii.gz。2D 预测不能用原生 `nnUNet_predict`
(无法 restore `PACAV1DeterministicTrainer`),必须走 `framework.predict --network-dim 2d`。

## Evaluation 统一规范(HD95 + Dice)

### 先用项目已有的 locked evaluator,不要自己写

项目里已有 **locked evaluator**:`swine_ct_autonomous_discovery/metrics/evaluate_swine_ct.py`
(35KB,带完整的 HD95 + Dice + absent-FP + missed + 百分位统计)。spec 引用它时,
**必须先读它的实现,再决定是自己写还是直接调用**。本次 v1 comparison 绕过了它自己写 HD95,
试错 5 轮才对齐 —— 直接读源码可以省掉全部来回。

### HD95 算法(locked evaluator 的标准方法,不要改)

`hd95_mm(gt_mask, pred_mask, spacing)`:

1. **union-bbox 裁剪(margin=2)**:`crop = bbox(GT ∪ Pred) + 2 voxels`。裁剪是无损的
   (所有表面点都在 union bbox 内),margin=2 防 erosion 边界效应。
2. **表面提取**:`surface = mask & ~binary_erosion(mask, structure=6-connectivity)`。
3. **spacing 缩放**:`surface_points *= spacing`(转 mm,各向异性)。
4. **cKDTree 双向最近邻**:`gt_tree.query(pred_points, k=1)` + `pred_tree.query(gt_points, k=1)`。
   cKDTree 是精确搜索(不是近似),结果和暴力 O(N²) 逐位相同。
5. **P95**:concat 双向距离,取 `np.percentile(distances, 95)`。

**不要用 `distance_transform_edt` 替代** —— 虽然数学上等价,但边界 case 浮点差异导致
和 locked evaluator 不完全对齐,文章里会出分歧。

### Evaluation 的并行与 I/O 规则

- **不要在 run_eval 里用 `fcntl.flock`**。compute node 的共享存储是 NFS,
  `flock` 在 NFS 上会永久 hang(login node `/tmp` 测试通过,DSUB 上 hang —— 这个坑
  浪费了大量调试时间)。并发 eval 保护用 **DSUB job 脚本里的 `flock -n`**(job 级互斥)。
- **每 case flush + print 进度**,不要攒 10 case 才 flush(否则 0 输出看起来像 hang)。
- **cKDTree 的 `workers` 参数**:在 `multiprocessing.Pool` worker 里调
  `cKDTree.query(workers=N)` 会触发 scipy 内部 Pool → 嵌套 multiprocessing 死锁。
  解决:`workers<=1` 时不传 `workers` kwarg(走 scipy 单线程路径),靠 case 级
  `multiprocessing.Pool` 做并行。
- **测试必须在 DSUB job 里跑**,不是只在 login node 跑 —— NFS vs 本地文件系统、
  GPU 显存、CUDA 初始化行为都不同。

### 后续 evaluation 的统一流程

1. **读 locked evaluator** → 确认 HD95 / Dice / absent-FP 的定义。
2. **从 per-case CSV 算统计**(`evaluation/run_stats.py`):
   - Per-class:mean / median / std / **P10(Dice)** / **P90(HD95)** / max。
   - 条件类(head/testis)的 absent-FP:voxels / rate / incidence。
   - 跨 3 seed 聚合 → Wilcoxon signed-rank(per-case mean Dice,3-seed 先平均)→ Holm-Bonferroni。
3. **肾脏等小目标**:用 median + P90,不用 mean(mean 被尾部拉飞,median 才是真相)。
4. **图表**:`evaluation/make_figures.py`(bar / heatmap / box / significance matrix)。
5. **结果推 GitHub + 嵌入 issue**(raw URL,repo 需 public)。
