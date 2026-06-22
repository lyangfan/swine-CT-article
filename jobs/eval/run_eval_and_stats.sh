#!/usr/bin/env bash
#DSUB -n eval_and_stats
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=16;mem=64000"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/eval/eval_and_stats.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/eval/eval_and_stats.err
set -euo pipefail

# Stage 6: locked evaluator (Dice + HD95 per class, conditional masking) on every
# (network, seed) prediction folder, then 3-seed aggregation + Wilcoxon + Holm.

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_results
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

ARTICLE=/home/share/hzau/home/liuyangfan/swine-CT-article
PRED_ROOT=${ARTICLE}/data/nnunetv1/v1_comparison_predictions
GT_FOLDER=${ARTICLE}/data/nnunetv1/nnUNet_raw_data/Task601_Article622_Carcass9Class/labelsTs
META=${ARTICLE}/data/manifests/case_metadata.csv
RESULTS=${ARTICLE}/evaluation/results
PER_CASE=${RESULTS}/per_case.csv
mkdir -p "${RESULTS}"

# Non-blocking flock: if another eval job is already running, exit immediately
# (it will produce the output). This prevents the duplicate-eval race that
# previously had two jobs clobber each other's per_case.csv.
exec 9>"${RESULTS}/.eval.flock"
flock -n 9 || { echo "[eval] another eval_and_stats is already running — exiting to avoid a race"; exit 0; }
rm -f "${PER_CASE}"   # safe now: we hold the exclusive lock

cd "${ARTICLE}"
NETWORKS="nnunet_v1 swinunetr mednext_s segformer3d nnunet_2d"
SEEDS="20260520 20260521 20260522"
for net in ${NETWORKS}; do
  for seed in ${SEEDS}; do
    PRED=${PRED_ROOT}/${net}__seed${seed}
    if [ ! -d "${PRED}" ]; then
      echo "[skip] ${net} seed=${seed}: predictions missing (${PRED})"
      continue
    fi
    n=$(ls "${PRED}"/*.nii.gz 2>/dev/null | wc -l)
    if [ "${n}" -lt 39 ]; then
      echo "[warn] ${net} seed=${seed}: only ${n}/39 predictions, evaluating anyway"
    fi
    echo "=== eval ${net} seed=${seed} (${n} preds) ==="
    "${NNUNETV1_PYTHON}" -m evaluation.run_eval \
        --predictions "${PRED}" \
        --gt-folder "${GT_FOLDER}" \
        --case-metadata "${META}" \
        --network "${net}" --seed "${seed}" \
        --output-csv "${PER_CASE}" --num-workers 8
  done
done

echo "=== stats (3-seed aggregation + Wilcoxon + Holm-Bonferroni) ==="
"${NNUNETV1_PYTHON}" -m evaluation.run_stats \
    --input "${PER_CASE}" \
    --out-dir "${RESULTS}"

echo "=== eval_and_stats DONE (exit $?) ==="
echo "Results: ${RESULTS}/summary.md  ${RESULTS}/per_case.csv"
