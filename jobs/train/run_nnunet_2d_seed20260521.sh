#!/usr/bin/env bash
#DSUB -n train_nnunet_2d_seed20260521
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=16;mem=64000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/train/train_nnunet_2d_seed20260521.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/train/train_nnunet_2d_seed20260521.err
set -euo pipefail

# 2D nnUNet reference — native nnUNetTrainerV2_2D via train_paca_deterministic.py
# (NOT the MultiNetworkTrainer; determinism patches apply globally before init).
export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_results

source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONHASHSEED=20260521

PREPROCESSED=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
RAW_BASE=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
RESULTS=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison
ROOT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison_2d_root
mkdir -p "${ROOT}"

cd "/home/share/hzau/home/liuyangfan/swine-CT-article"
echo "=== train nnunet_2d seed=20260521 -> ${RESULTS}/nnunet_2d__seed20260521 ==="
"${NNUNETV1_PYTHON}" \
    /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/scripts/nnunetv1_compat/train_paca_deterministic.py \
    --root "${ROOT}/seed20260521" \
    --run-name nnunet_2d__seed20260521 \
    --base-seed 20260521 \
    --task-id 601 \
    --network 2d \
    --plans-identifier nnUNetPlansv2.1 \
    --fold 0 \
    --epochs 500 \
    --train-batches 250 \
    --val-batches 50 \
    --raw-base "${RAW_BASE}/nnUNet_raw_data" \
    --preprocessed "${PREPROCESSED}" \
    --results-folder "${RESULTS}/nnunet_2d__seed20260521" \
    --fp16 --unpack-data \
    --checkpoint-policy final-best
echo "=== train_nnunet_2d_seed20260521 DONE (exit $?) ==="
