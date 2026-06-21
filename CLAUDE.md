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
