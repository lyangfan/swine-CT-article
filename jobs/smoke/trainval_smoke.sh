#!/usr/bin/env bash
#DSUB -n trainval_smoke
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=16;mem=48000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/smoke/trainval_smoke.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/smoke/trainval_smoke.err
set -euo pipefail
export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_results
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
cd /home/share/hzau/home/liuyangfan/swine-CT-article
for NET in nnunet_v1 mednext_s swinunetr segformer3d; do
  echo "===== ${NET} ====="
  "${NNUNETV1_PYTHON}" -m framework.smoke_trainval --network "${NET}" --n-train-iters 3 --n-val-iters 3
done
echo "=== ALL TRAINVAL SMOKE DONE ==="
