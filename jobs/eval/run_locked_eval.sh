#!/usr/bin/env bash
#DSUB -n locked_eval_all
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=64;mem=128000"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/eval/locked_eval_all.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/eval/locked_eval_all.err
set -euo pipefail

# Stage 6 (migrated): run the PROJECT'S locked evaluator (evaluate_swine_ct.py)
# on all (network, seed) prediction sets. Produces case_metrics.csv with the
# FULL metric set (Dice, HD95, IoU, Precision, Recall, Specificity, FPR,
# missed, absent-FP, TP/FP/FN/TN) + class/cohort/fold summary aggregations.

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

ARTICLE=/home/share/hzau/home/liuyangfan/swine-CT-article
EVAL_SCRIPT=/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/metrics/evaluate_swine_ct.py
LABEL_MAP=${ARTICLE}/evaluation/label_mapping.yaml
RESULTS=${ARTICLE}/evaluation/results_locked
CASES_DIR=${ARTICLE}/evaluation/cases

mkdir -p "${RESULTS}" "${CASES_DIR}"
cd "${ARTICLE}"

NETWORKS="nnunet_v1 swinunetr mednext_s segformer3d nnunet_2d"
SEEDS="20260520 20260521 20260522"
NUM_WORKERS=16
HD95_WORKERS=4

for net in ${NETWORKS}; do
  for seed in ${SEEDS}; do
    PRED=${ARTICLE}/data/nnunetv1/v1_comparison_predictions/${net}__seed${seed}
    if [ ! -d "${PRED}" ]; then
      echo "[skip] ${net} seed=${seed}: predictions missing"
      continue
    fi
    n=$(ls "${PRED}"/*.nii.gz 2>/dev/null | wc -l)
    if [ "${n}" -lt 39 ]; then
      echo "[warn] ${net} seed=${seed}: only ${n}/39 predictions"
    fi
    CASES_CSV=${CASES_DIR}/${net}__seed${seed}.csv
    METHOD="${net}_seed${seed}"
    OUTDIR=${RESULTS}/${net}__seed${seed}

    echo "=== build cases_csv: ${net} seed=${seed} ==="
    "${NNUNETV1_PYTHON}" -m evaluation.build_cases_csv \
        --network "${net}" --seed "${seed}" \
        --output-csv "${CASES_CSV}"

    echo "=== locked eval: ${net} seed=${seed} (${n} preds) ==="
    "${NNUNETV1_PYTHON}" "${EVAL_SCRIPT}" \
        --cases-csv "${CASES_CSV}" \
        --label-mapping "${LABEL_MAP}" \
        --method "${METHOD}" \
        --cohort "v1_comparison" \
        --output-dir "${OUTDIR}" \
        --num-workers ${NUM_WORKERS} \
        --hd95-workers ${HD95_WORKERS} \
        --schedule size-desc \
        --force

    echo "=== kidney swap eval: ${net} seed=${seed} ==="
    "${NNUNETV1_PYTHON}" -m evaluation.kidney_swap_eval \
        --predictions "${PRED}" \
        --gt-folder "${ARTICLE}/data/nnunetv1/nnUNet_raw_data/Task601_Article622_Carcass9Class/labelsTs" \
        --network "${net}" --seed "${seed}" \
        --output-csv "${RESULTS}/kidney_swap.csv" \
        --num-workers ${NUM_WORKERS}
  done
done

echo "=== ALL EVAL DONE (locked evaluator + kidney swap) ==="
