#!/usr/bin/env bash
#DSUB -n smoke_condlr_telemetry
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=4;mem=16000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/tools/smoke_telemetry.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/tools/smoke_telemetry.err
set -euo pipefail

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
export nnUNet_results=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_results
export PYTHONPATH=/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/scripts/nnunetv1_compat:${PYTHONPATH:-}

source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.8.1_cuda11 libs/nccl/2.16.5-1_cuda11.8 libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16

cd /home/share/hzau/home/liuyangfan/swine-CT-article

echo "=== Telemetry Smoke Test ==="
echo "Date: $(date)"
echo "Python: $(which python)"

python /home/share/hzau/home/liuyangfan/swine-CT-article/tools/smoke_telemetry_test.py

echo "=== Smoke DONE (exit $?) ==="
