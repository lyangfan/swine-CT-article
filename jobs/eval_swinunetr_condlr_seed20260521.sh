#!/usr/bin/env bash
#DSUB -n eval_swinunetr_condlr_seed20260521
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=4;mem=16000;gpu=0"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/eval_swinunetr_condlr_seed20260521.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/eval_swinunetr_condlr_seed20260521.err
set -euo pipefail

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

cd /home/share/hzau/home/liuyangfan/swine-CT-article
mkdir -p evaluation/results
echo "=== eval swinunetr condlr seed=20260521 ==="
${NNUNETV1_PYTHON} -m evaluation.run_eval \
    --predictions /home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/v1_comparison_predictions/swinunetr__condlr_seed20260521 \
    --gt-folder /home/share/hzau/home/liuyangfan/swine-CT-article/data/test/labels \
    --case-metadata /home/share/hzau/home/liuyangfan/swine-CT-article/data/manifests/case_metadata.csv \
    --network swinunetr_condlr \
    --seed 20260521 \
    --output-csv /home/share/hzau/home/liuyangfan/swine-CT-article/evaluation/results/swinunetr_condlr_seed20260521_per_case.csv
echo "=== DONE (exit $?) ==="
