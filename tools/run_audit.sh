#!/usr/bin/env bash
#DSUB -n audit_lr_axis
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=4;mem=16000;gpu=0"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/tools/audit_output/audit.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/tools/audit_output/audit.err
set -euo pipefail

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh

cd /home/share/hzau/home/liuyangfan/swine-CT-article
mkdir -p tools/audit_output

echo "=== LR Axis Audit ==="
echo "Python: $(which python)"
echo "nibabel: $(python -c 'import nibabel; print(nibabel.__version__)')"

python tools/audit_lr_axis.py \
  --labels-dir /home/share/hzau/home/liuyangfan/swine-CT-article/data/train/labels \
  --split-manifest /home/share/hzau/home/liuyangfan/swine-CT-article/data/splits/split_manifest.csv \
  --output-dir /home/share/hzau/home/liuyangfan/swine-CT-article/tools/audit_output

echo "=== Audit DONE (exit $?) ==="
