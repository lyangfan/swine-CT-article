#!/usr/bin/env bash
# Login-node orchestrator: autonomously submits predict jobs for completed
# training, then eval+stats. Run with nohup on the Huawei LOGIN node (where dsub
# works). Nested DSUB from compute nodes does NOT work, so this must run on the
# login node. Idempotent. Exits when eval summary exists.
#
# Start: nohup bash login_node_orchestrator.sh > orchestrator.log 2>&1 &
ARTICLE=/home/share/hzau/home/liuyangfan/swine-CT-article
PRED_ROOT=${ARTICLE}/data/nnunetv1/v1_comparison_predictions
CKPT_BASE_3D=${ARTICLE}/data/nnunetv1/v1_comparison
CKPT_BASE_2D=${ARTICLE}/data/nnunetv1/v1_comparison_2d_root
EVAL_SENTINEL=${ARTICLE}/jobs/eval/_eval_submitted
EVAL_DONE=${ARTICLE}/evaluation/results/summary.md
N_TEST=39
MAX_TICKS=40   # 40 × 20 min ≈ 13 h

NETWORKS_3D="nnunet_v1 swinunetr mednext_s segformer3d"
NETWORKS_2D="nnunet_2d"
SEEDS="20260520 20260521 20260522"

cd "${ARTICLE}"
for tick in $(seq 1 ${MAX_TICKS}); do
  echo "=== orchestrator tick ${tick} @ $(date '+%H:%M') ==="
  [ -f "${EVAL_DONE}" ] && { echo "[complete] eval done — exiting"; break; }

  n_pred_done=0; n_total=0
  for seed in ${SEEDS}; do
    for net in ${NETWORKS_3D} ${NETWORKS_2D}; do
      n_total=$((n_total+1))
      if [ "${net}" = "nnunet_2d" ]; then
        ckpt=${CKPT_BASE_2D}/seed${seed}/runs/nnunet_2d__seed${seed}/fold_0/model_final_checkpoint.model
      else
        ckpt=${CKPT_BASE_3D}/${net}__seed${seed}/fold_0/model_final_checkpoint.model
      fi
      pred_dir=${PRED_ROOT}/${net}__seed${seed}
      [ -f "${ckpt}" ] || { echo "[wait] ${net} seed=${seed}"; continue; }
      n=$(ls "${pred_dir}"/*.nii.gz 2>/dev/null | wc -l)
      if [ "${n}" -ge ${N_TEST} ]; then
        n_pred_done=$((n_pred_done+1)); continue
      fi
      # trained but not fully predicted → submit predict (idempotent: re-running
      # just overwrites identical output). Check no predict job already RUNNING
      # for this net/seed to avoid pile-up (best-effort via job name match).
      running=$(bash -lc "djob -D 2>/dev/null | grep \"predict_${net}_seed${seed}\" | grep RUNNING | wc -l")
      if [ "${running}" -eq 0 ]; then
        jid=$(bash -lc "dsub -s jobs/predict/run_predict_${net}_seed${seed}.sh 2>/dev/null" | awk '{if($1 ~ /^[0-9]+$/) print $1; else if(/submit job successfully/) print "OK"}')
        echo "[submit predict] ${net} seed=${seed} -> ${jid}"
      else
        echo "[running] ${net} seed=${seed} predict already in progress (${n}/${N_TEST})"
      fi
    done
  done
  echo "[summary] predictions done: ${n_pred_done}/${n_total}"

  # all predictions done + eval not yet submitted → submit eval
  if [ "${n_pred_done}" -eq "${n_total}" ] && [ ! -f "${EVAL_SENTINEL}" ]; then
    echo "[act] all ${n_total} predictions done → submit eval+stats"
    touch "${EVAL_SENTINEL}"
    bash -lc "dsub -s jobs/eval/run_eval_and_stats.sh" 2>&1 | tail -2
  fi
  [ -f "${EVAL_DONE}" ] && { echo "[complete] eval done — exiting"; break; }
  sleep 1200   # 20 min
done
echo "=== orchestrator finished @ $(date '+%H:%M') ==="
