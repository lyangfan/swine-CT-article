#!/bin/bash
#DSUB -n roi_classifier_train_30_models
#DSUB -A root.hzau
#DSUB -R "cpu=8;mem=64000;gpu=1"
#DSUB -pn !whshare-agent-174
#DSUB -e train_stage5.err
#DSUB -o train_stage5.out

# Stage 5 — Train 30-model ROI classifier grid
# spec §7 + §13 + §16 Stage 5
#
# Grid: 3 correct ROI roles × 5 variants × 2 init = 30 models
# Per-model best = val AUPRC (highest) subject to FA≤1 safety gate

set -euo pipefail

echo "===== STAGE 5: ROI CLASSIFIER TRAINING ====="
echo "Start: $(date -u +%Y%m%dT%H%M%SZ)"
echo "Node: $(hostname)"
echo "Submit host: $(hostname)"

# Environment (CLAUDE.md verified stack)
module load compilers/gcc/9.3.0
module load compilers/cuda/11.8.0
module load libs/cudnn/8.6.0_cuda11

# Activate conda env
source activate /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/envs/nnunetv1

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "cuDNN: $(python -c 'import torch; print(torch.backends.cudnn.version())')"

REPO_ROOT=/home/share/hzau/home/liuyangfan/swine-CT-article
cd "$REPO_ROOT"

MANIFEST="data/manifests/classifier_split_manifest.csv"
PROJ_MANIFEST="runs/roi_presence_classifier/data/manifests/roi_projection_manifest_20260623T173717Z.csv"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

echo "Manifest: $MANIFEST"
echo "Projection manifest: $PROJ_MANIFEST"
echo "Stamp: $STAMP"

# Run training (30 sequential models)
python tools/roi_classifier/train_roi_classifier.py \
    --manifest "$MANIFEST" \
    --projection-manifest "$PROJ_MANIFEST" \
    --stamp "$STAMP" \
    --amp \
    --batch-size 16 \
    --num-workers 4 \
    --device cuda \
    --resource-request "cpu=8;mem=64000;gpu=1" \
    --node-constraint "!whshare-agent-174" \
    --job-name "roi_classifier_train_30_models" \
    --job-id "${DSUB_JOB_ID:-unknown}"

echo "===== TRAINING COMPLETE ====="
echo "End: $(date -u +%Y%m%dT%H%M%SZ)"
echo "Stamp: $STAMP"
