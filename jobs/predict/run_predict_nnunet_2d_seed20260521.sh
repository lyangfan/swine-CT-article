#!/usr/bin/env bash
#DSUB -n predict_nnunet_2d_seed20260521
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=8;mem=48000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/predict/predict_nnunet_2d_seed20260521.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/predict/predict_nnunet_2d_seed20260521.err
set -euo pipefail

# 2D nnUNet — uses the framework's unified predict.py (same locked sliding-window
# protocol, 2D). The 2D model was trained by train_paca_deterministic.py
# (PACAV1DeterministicTrainer = nnUNetTrainerV2_2D); native nnUNet_predict can't
# restore that trainer class, so we load weights via framework.predict's
# load_checkpoint_ram into a Generic_UNet built from the 2D plans (nnunet_v1 spec
# with --network-dim 2d).
export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_results
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

CKPT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison_2d_root/seed20260521/runs/nnunet_2d__seed20260521/fold_0/model_final_checkpoint.model
INPUT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_raw_data/Task601_Article622_Carcass9Class/imagesTs
OUT=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison_predictions/nnunet_2d__seed20260521
mkdir -p "${OUT}"
cd "/home/share/hzau/home/liuyangfan/swine-CT-article"
echo "=== predict nnunet_2d seed=20260521 (framework.predict, 2D, TTA off, overlap 0.5) ==="
"${NNUNETV1_PYTHON}" -m framework.predict \
    --network nnunet_v1 --network-dim 2d --seed 20260521 \
    --checkpoint "${CKPT}" \
    --input-folder "${INPUT}" \
    --output-folder "${OUT}" \
    --step-size 0.5
echo "=== predict_nnunet_2d_seed20260521 DONE (exit $?) ==="
