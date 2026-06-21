#!/usr/bin/env bash
#DSUB -n predict_nnunet_2d_seed20260522
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=8;mem=48000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/predict/predict_nnunet_2d_seed20260522.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/predict/predict_nnunet_2d_seed20260522.err
set -euo pipefail

# 2D nnUNet — native nnUNet_predict (same locked sliding-window protocol, 2D)
export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

# train_paca_deterministic.py writes to runs/<run>/fold_0/; nnUNet_predict expects
# $RESULTS_FOLDER/2d/Task<id>/<trainer>__<plans>/fold_0/. Bridge with a symlink.
RUN_DIR=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison_2d_root/seed20260522/runs/nnunet_2d__seed20260522
LAYOUT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison_2d_nnunet_layout/seed20260522
TASK=Task601_Article622_Carcass9Class
mkdir -p "${LAYOUT}/2d/${TASK}/nnUNetTrainerV2__nnUNetPlansv2.1"
ln -sfn "${RUN_DIR}/fold_0" "${LAYOUT}/2d/${TASK}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0"
export RESULTS_FOLDER="${LAYOUT}"

INPUT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_raw_data/Task601_Article622_Carcass9Class/imagesTs
OUT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison_predictions/nnunet_2d__seed20260522
mkdir -p "${OUT}"
echo "=== predict nnunet_2d seed=20260522 (native nnUNet_predict, TTA off, overlap 0.5) ==="
nnUNet_predict -i "${INPUT}" -o "${OUT}" -t 601 -tr nnUNetTrainerV2 -m 2d -f 0 \
    --disable_tta --step_size 0.5
echo "=== predict_nnunet_2d_seed20260522 DONE (exit $?) ==="
