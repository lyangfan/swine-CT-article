#!/usr/bin/env bash
#DSUB -n patch_smoke_test
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=8;mem=32000;gpu=1"
#DSUB -pn !whshare-agent-174
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/smoke/patch_smoke_test.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/smoke/patch_smoke_test.err

set -euo pipefail

source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1

cd /home/share/hzau/home/liuyangfan/swine-CT-article/source
echo "=== patch [64,160,160] forward smoke test (non-nnUNet nets) ==="
"${NNUNETV1_PYTHON}" smoke_test_patch.py
echo "=== DONE ==="
