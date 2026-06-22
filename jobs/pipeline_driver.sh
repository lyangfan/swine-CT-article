#!/usr/bin/env bash
# Pipeline driver: auto-submits predict jobs for completed training, then eval.
# Schedule via cron on the Huawei login node (every ~30 min). Idempotent: skips
# work already done. Logs to pipeline_driver.log.
#
# Checks, per (network, seed):
#   1. training done? (model_final_checkpoint.model exists)
#   2. predictions done? (39 nii.gz in the predict output folder)
#   3. if trained + not predicted → submit predict job
# When all 15 prediction folders have 39 nii.gz → submit eval+stats job once.

set -uo pipefail
ARTICLE=/home/share/hzau/home/liuyangfan/swine-CT-article
LOG=${ARTICLE}/jobs/pipeline_driver.log
PRED_ROOT=${ARTICLE}/data/nnunetv1/v1_comparison_predictions
CKPT_BASE_3D=${ARTICLE}/data/nnunetv1/v1_comparison
CKPT_BASE_2D=${ARTICLE}/data/nnunetv1/v1_comparison_2d_root
EVAL_SENTINEL=${ARTICLE}/jobs/eval/_eval_submitted
EVAL_DONE=${ARTICLE}/evaluation/results/summary.md

NETWORKS_3D="nnunet_v1 swinunetr mednext_s segformer3d"
NETWORKS_2D="nnunet_2d"
SEEDS="20260520 20260521 20260522"
N_TEST=39

echo "=== $(date '+%Y-%m-%d %H:%M:%S') pipeline driver tick ===" >> "${LOG}"

submit_predict() {
  local net=$1 seed=$2 script
  script=${ARTICLE}/jobs/predict/run_predict_${net}_seed${seed}.sh
  if [ ! -x "${script}" ]; then echo "[skip] ${net} seed=${seed}: script missing/not executable" >> "${LOG}"; return; fi
  # NOTE: nested DSUB (dsub from inside a DSUB compute-node job) does NOT work on
  # this cluster. We only LOG the needed action here; a login-node process (or the
  # operator's session) reads these [need-submit] lines and submits from the
  # login node where dsub is available.
  touch "${ARTICLE}/jobs/_needs_submit/${net}__seed${seed}"
  echo "[need-submit] ${net} seed=${seed} (marker written; submit from login node)" >> "${LOG}"
}

n_pred_done=0
n_total=0
for seed in ${SEEDS}; do
  for net in ${NETWORKS_3D}; do
    n_total=$((n_total+1))
    ckpt=${CKPT_BASE_3D}/${net}__seed${seed}/fold_0/model_final_checkpoint.model
    pred_dir=${PRED_ROOT}/${net}__seed${seed}
    if [ ! -f "${ckpt}" ]; then
      echo "[wait] ${net} seed=${seed}: training not done yet" >> "${LOG}"; continue
    fi
    n=$(ls "${pred_dir}"/*.nii.gz 2>/dev/null | wc -l)
    if [ "${n}" -ge ${N_TEST} ]; then
      n_pred_done=$((n_pred_done+1))
      echo "[done] ${net} seed=${seed}: ${n}/${N_TEST} predictions" >> "${LOG}"; continue
    fi
    # trained but not predicted → submit (idempotent: if a predict job already ran
    # and produced some output, we still resubmit only if < N_TEST — acceptable)
    echo "[act] ${net} seed=${seed}: trained, ${n}/${N_TEST} preds → submit predict" >> "${LOG}"
    submit_predict "${net}" "${seed}"
  done
  for net in ${NETWORKS_2D}; do
    n_total=$((n_total+1))
    ckpt=${CKPT_BASE_2D}/seed${seed}/runs/nnunet_2d__seed${seed}/fold_0/model_final_checkpoint.model
    pred_dir=${PRED_ROOT}/${net}__seed${seed}
    if [ ! -f "${ckpt}" ]; then
      echo "[wait] ${net} seed=${seed}: training not done yet" >> "${LOG}"; continue
    fi
    n=$(ls "${pred_dir}"/*.nii.gz 2>/dev/null | wc -l)
    if [ "${n}" -ge ${N_TEST} ]; then
      n_pred_done=$((n_pred_done+1))
      echo "[done] ${net} seed=${seed}: ${n}/${N_TEST} predictions" >> "${LOG}"; continue
    fi
    echo "[act] ${net} seed=${seed}: trained, ${n}/${N_TEST} preds → submit predict" >> "${LOG}"
    submit_predict "${net}" "${seed}"
  done
done

echo "[summary] predictions done: ${n_pred_done}/${n_total}" >> "${LOG}"

# When all predictions done + eval not yet submitted → submit eval+stats
if [ "${n_pred_done}" -eq "${n_total}" ] && [ ! -f "${EVAL_SENTINEL}" ]; then
  echo "[act] all ${n_total} predictions done → submit eval+stats" >> "${LOG}"
  touch "${EVAL_SENTINEL}"
  bash -lc "cd ${ARTICLE} && dsub -s jobs/eval/run_eval_and_stats.sh" >> "${LOG}" 2>&1
fi
if [ -f "${EVAL_DONE}" ]; then
  echo "[complete] eval results ready: ${EVAL_DONE}" >> "${LOG}"
fi
