# Labeled Swine CT Data — `swine-CT-article/data/`

197 例带人工分割标注的猪胴体 CT，用于文章相关的监督训练 / 评估 / 可视化。本目录全部为软连接 + 元信息文档，不复制实际影像数据。

## 目录结构

```
data/
├── images/                       # 197 个 CT 源图像软连接
│   └── {case_id}.nii.gz
├── labels/                       # 197 个人工分割标注软连接
│   └── {case_id}.nii.gz
├── manifests/
│   ├── case_metadata.csv         # 每头猪一行的主元信息表
│   ├── tb_phenotype.phe          # TB 原始 phenotype 表副本（HZAU 上 795 行 × 151 列）
│   ├── tb_dicom_breed.csv        # TB DICOM 目录 ind→breed 映射（846 个 individual）
│   ├── hzau_slaughter_breed.csv  # HZAU 屠宰测定表抽出的 case_id→breed 映射（93 行）
└── README.md                     # 本文件
```

软连接指向：
- `images/` → `/home/hzau/whcs-share37/liuyangfan/nnunet_medsam_semisup/data/labeled_197/images/`
- `labels/` → `/home/hzau/whcs-share37/liuyangfan/nnunet_medsam_semisup/data/labeled_197/labels/`

实际源文件分布在 HZAU GPU 服务器（`hzau_gpu:/workspace/data/CT/`）。

## 数据来源

| source | source_detail | n_cases | breed 记录 |
|---|---|---:|---|
| `HZAU` | `HZAU_veterinary_hospital` | 93 | 92 例来自屠宰测定表 + 1 例群体推断，全 Yorkshire |
| `TB` | `TB_sequence_846_train_80` | 80 | 4 分类完整 |
| `TB` | `TB_sequence_846_manual_test_24` | 24 | 4 分类完整 |
| **合计** | | **197** | |

### HZAU Veterinary Hospital（93 例）

- 数据路径：`hzau_gpu:/workspace/data/CT/HZAU_veterinary_hospital/{nifti,label}/`
- CT 数：196（其中 93 例有人工分割标注）
- 标注协议：兽医院临床数据，9 类胴体分割
- **品种来源**：原始 CT 目录本身不记录品种；92/93 例的品种来自**屠宰测定表**
  `/Volumes/EXTERNAL_USB/project/pig/Huazhong/屠宰测定/CT屠宰测定表.xlsx`
  （本地 USB，read-only；14 个按屠宰日期命名的 sheet，转置布局：行=性状，列=个体）。
  抽出的映射见 `manifests/hzau_slaughter_breed.csv`。
  本数据集 92 例 HZAU 直接匹配屠宰表，全部为 **Yorkshire（大白）**。
- **1 例（`156202411000519`，`156202411xxxxxx_late` 批次）**在屠宰表中没有对应条目，
  按 HZAU 群体归属推断为 Yorkshire，metadata 中 `breed_source=inferred_from_hzau_cohort`。
- 文件名仍有时间批次前缀（仅用于排序，非品种）：
  - `070xxxx_early`：早期批次
  - `156052xxxxxxx_mid`：大批次
  - `156202411xxxxxx_late`：较新批次
- **关键 caveat**（来自 PACA `docs/data_description.md`）：HZAU 已标注猪均为**阉猪**（head-present, testis-absent）。HZAU 未标注的 nifti 中可能存在有头有睾丸的猪，但本数据集只包含已标注的 93 例。

### TB Sequence 846（104 例，来自 846 头实验群体）

- 数据路径：`hzau_gpu:/workspace/data/CT/TB/sequence_846/`
- 总群体规模：846 头，按品种组织 DICOM 目录
- 标注协议：`carcass_9_classes` 9 类胴体分割
- 本数据集仅取其中 104 例已标注样本（train_80 + manual_test_24）
- **关键 caveat**：TB labeled cases 均为**公猪**（testis-present, head-absent / out-of-FOV）

## 品种分布

品种来源（记录在 `breed_source` 列）：
- **TB**：DICOM 目录结构（`dicom_dir`），phe 表中文列作为更细粒度的辅助
- **HZAU**：屠宰测定表（`slaughter_table:<letter>@<date>`），92/93 例均为 Yorkshire

采用 **4 分类 canonical**：Yorkshire / Landrace / Pietrain / Duroc。

| breed_en | breed_zh | n_cases |
|---|---|---:|
| Yorkshire | 大白 | 119 |
| Duroc | 杜洛克 | 26 |
| Landrace | 长白 | 26 |
| Pietrain | 皮特兰 | 26 |

说明：HZAU 93 例全部 Yorkshire，TB 104 例按品种近似均衡采样（Yorkshire/Landrace/Pietrain/Duroc 各 26）。Yorkshire 合计 119 = HZAU 93 + TB 26。

### EB5 归入 Duroc 的策略

- **EB5 是杜洛克合成系**，本数据集将其归入 Duroc，不再单列。
- TB DICOM 目录物理上仍保留 `EB5/` 子目录（13 个个体，见 `tb_dicom_breed.csv` 的
  `breed_from_dicom_dir` 列），但 canonical 分类统一为 Duroc。
- TB phe 表英文列（`品种品系.1`）原本就把 EB5 折叠为 `Duroc`，与本策略一致；
  phe 中文列（`品种品系`）保留更细的 `EB5杜洛克`，作为 `breed_zh_phe` 溯源保留。

### phe 表与 DICOM 目录的其它分歧

除 EB5 之外，phe 中文列还保留更细的品系名（如 `美系杜洛克`、`美系大白`、
`美系长白`、`美系皮特兰`），这些在 canonical 4 分类里都折叠到对应主品种，
细粒度信息只通过 `breed_zh_phe` 字段保留。

## 标签定义

9 类胴体分割（label intensities in label NIfTI）：

| label | class |
|---:|---|
| 0 | background |
| 1 | front |
| 2 | middle |
| 3 | end |
| 4 | left_kidney |
| 5 | right_kidney |
| 6 | testis |
| 7 | thoracic_cavity |
| 8 | abdominal_and_pelvic_cavity |
| 9 | head |

定义文件：`hzau_gpu:/workspace/data/CT/HZAU_veterinary_hospital/label/labels.json`

## Manifest Schema

`manifests/case_metadata.csv` 列定义：

| 列 | 含义 |
|---|---|
| `case_id` | 唯一 ID，TB 为 `{individualID}_{experimentID}`，HZAU 为单个编号 |
| `source` | `HZAU` 或 `TB` |
| `source_detail` | 进一步分类（见上文表格） |
| `hzau_batch` | 仅 HZAU 有，时间批次前缀 |
| `individual_id` | 个体 ID（TB = case_id 的下划线前半） |
| `experiment_id` | 实验 ID（TB = case_id 的下划线后半） |
| `breed_en` | canonical 品种英文（4 分类：Yorkshire/Landrace/Pietrain/Duroc） |
| `breed_zh` | canonical 品种中文 |
| `breed_zh_phe` | phe 表中文品系（更细，仅 TB，可能为空） |
| `breed_en_phe` | phe 表英文品种（Duroc 含 EB5，仅 TB） |
| `breed_source` | 品种来源：`dicom_dir`（TB）/ `slaughter_table:<letter>@<date>`（HZAU 92 例）/ `inferred_from_hzau_cohort`（HZAU 1 例群体推断） |
| `sex` | 性别（仅 TB phe 有） |
| `birth_farm` / `current_farm` / `test_farm` | 繁育场 / 当前场 / 测定场（仅 TB phe 有） |
| `birth_parity` / `litter_size` / `inbreeding_coef` | 出生胎次 / 同窝仔猪数 / 近交系数 |
| `image_path` / `label_path` | HZAU 上的原始绝对路径 |
| `image_size` / `label_size` | 文件字节数 |

`tb_phenotype.phe` 是原始 PLINK 风格 phenotype 表，包含 151 列完整育种 / 胴体 / EBV 性状，可直接用 `pandas.read_csv(..., sep="\t")` 读取。

## 复现 / 校验

```bash
# 在 Huawei paca_share 上
DST=/home/share/hzau/home/liuyangfan/swine-CT-article/data

# 1. 校验软连接完整性
for cid in $(awk -F, 'NR>1{print $1}' $DST/manifests/case_metadata.csv); do
  test -L $DST/images/$cid.nii.gz && test -L $DST/labels/$cid.nii.gz || echo "MISSING: $cid"
done

# 2. case_metadata.csv 的来源（生成脚本已是一次性产物，已清理；产物即 canonical）
#    拼装逻辑（留作记录，不在本目录保留脚本）：
#    - 基础行：labeled_197 source inventory + TB phe join（TB 品种取自 DICOM 目录）
#    - HZAU 品种：用屠宰测定表 CT屠宰测定表.xlsx 的"编号/品种"行 patch（92 例），
#      剩余 1 例按 HZAU 群体推断为 Yorkshire；来源凭证见 hzau_slaughter_breed.csv
#    - EB5 → Duroc canonical collapse

# 3. tb_phenotype.phe 与 HZAU 原表比对
ssh paca_share 'md5sum /home/share/hzau/home/liuyangfan/swine-CT-article/data/manifests/tb_phenotype.phe'
ssh hzau_gpu    'md5sum /workspace/data/CT/TB/sequence_846/all_original_phenotype.phe'
# expected: af46ce0241b67dddf0a20968a273d854
```
