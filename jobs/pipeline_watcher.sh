#!/usr/bin/env bash
#DSUB -n pipeline_watcher
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=1;mem=4000"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/pipeline_watcher.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/pipeline_watcher.err
set -uo pipefail

# Autonomous watcher: every 30 min, run the pipeline driver (which submits predict
# jobs for completed training, then eval+stats when all 15 predictions are ready).
# Exits when the eval summary exists, or after ~14 h (safety). Zero-token: runs
# entirely on Huawei. Uses dsub from inside (needs the batch CLI on PATH).

ARTICLE=/home/share/hzau/home/liuyangfan/swine-CT-article
DRIVER=${ARTICLE}/jobs/pipeline_driver.sh
EVAL_DONE=${ARTICLE}/evaluation/results/summary.md
MAX_TICKS=28   # 28 × 30 min = 14 h

source /opt/batch/cli/envs/profile.env >/dev/null 2>&1 || true
export PATH=/opt/batch/cli/bin:${PATH}

for i in $(seq 1 ${MAX_TICKS}); do
  echo "=== watcher tick ${i}/${MAX_TICKS} @ $(date '+%H:%M') ==="
  bash "${DRIVER}" || echo "[warn] driver returned non-zero"
  if [ -f "${EVAL_DONE}" ]; then
    echo "[complete] eval summary ready — watcher exiting"
    break
  fi
  sleep 1800   # 30 min
done
echo "=== watcher finished @ $(date '+%H:%M') ==="
tail -5 "${ARTICLE}/jobs/pipeline_driver.log" 2>/dev/null
