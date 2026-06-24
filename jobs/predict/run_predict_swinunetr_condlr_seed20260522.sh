#!/usr/bin/env bash
#DSUB -n predict_swinunetr_condlr_seed20260522
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=8;mem=48000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/predict/predict_swinunetr_condlr_seed20260522.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/predict/predict_swinunetr_condlr_seed20260522.err
set -euo pipefail

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_results
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.8.1_cuda11 libs/nccl/2.16.5-1_cuda11.8 libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

CKPT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison/swinunetr__condlr_seed20260522/fold_0/model_final_checkpoint.model
INPUT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_raw_data/Task601_Article622_Carcass9Class/imagesTs
OUT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison_predictions/swinunetr__condlr_seed20260522
mkdir -p "${OUT}"
cd "/home/share/hzau/home/liuyangfan/swine-CT-article"
echo "=== predict swinunetr condlr seed=20260522 ==="
"${NNUNETV1_PYTHON}" -m framework.predict \
    --network swinunetr --seed 20260522 \
    --checkpoint "${CKPT}" \
    --input-folder "${INPUT}" \
    --output-folder "${OUT}" \
    --step-size 0.5
echo "=== predict_swinunetr_condlr_seed20260522 DONE (exit $?) ==="
