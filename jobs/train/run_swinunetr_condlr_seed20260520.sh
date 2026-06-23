#!/usr/bin/env bash
#DSUB -n train_swinunetr_condlr_seed20260520
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=16;mem=64000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/train/train_swinunetr_condlr_seed20260520.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/train/train_swinunetr_condlr_seed20260520.err
set -euo pipefail

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_results

source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.8.1_cuda11 libs/nccl/2.16.5-1_cuda11.8 libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONHASHSEED=20260520

OUT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison/swinunetr__condlr_seed20260520
mkdir -p "${OUT}"
cd "/home/share/hzau/home/liuyangfan/swine-CT-article"
echo "=== train swinunetr condlr seed=20260520 axis=2 -> ${OUT} ==="
"${NNUNETV1_PYTHON}" -m framework.train \
    --network swinunetr \
    --seed 20260520 \
    --config configs/swinunetr.yaml \
    --task-id 601 --fold 0 \
    --lr-mirror-mode conditional \
    --conditional-mirror-axis 2 \
    --output-folder "${OUT}"
echo "=== train_swinunetr_condlr_seed20260520 DONE (exit $?) ==="
