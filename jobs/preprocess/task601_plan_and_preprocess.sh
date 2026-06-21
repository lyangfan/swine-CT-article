#!/usr/bin/env bash
#DSUB -n task601_plan_and_preprocess
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=32;mem=192000"
#DSUB -pn !whshare-agent-174
#DSUB -oo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/preprocess/task601_plan_and_preprocess.out
#DSUB -eo /home/share/hzau/home/liuyangfan/swine-CT-article/jobs/preprocess/task601_plan_and_preprocess.err

set -euo pipefail

# --- Task601 data roots (override the setup script's v1_ratio_nnunet default) ---
export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1
export nnUNet_preprocessed=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER=/home/share/hzau/home/liuyangfan/swine-CT-article/data/nnunetv1/nnUNet_results

# --- unified nnunetv1 env (CUDA stack + compat + libgomp hack) ---
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
# some compute nodes lack system libopenblas.so.0 (torch import fails); load it explicitly
module load libs/openblas/0.3.18_kgcc9.3.1

export TASK_ID=601
export TL=32   # preprocessing threads (high for speed)
export TF=32   # fingerprint threads (high for speed)

echo "=== env ==="
swct_nnunetv1_preflight

echo "=== plan_and_preprocess -t ${TASK_ID} -tl ${TL} -tf ${TF} ==="
# Full run. crop+fingerprint already cached (overwrite=False) so they're skipped;
# this run does the resample/normalize preprocessing -> nnUNet_preprocessed stage npz.
nnUNet_plan_and_preprocess -t ${TASK_ID} -tl ${TL} -tf ${TF}

echo "=== extract patch_size + target_spacing from plan ==="
"${NNUNETV1_PYTHON}" - <<'PY' || echo "(extract non-fatal, plan files exist)"
import pickle, glob, os
base = os.environ["nnUNet_preprocessed"]
cands = glob.glob(os.path.join(base, "Task601_*", "nnUNetPlansv2.1_plans_3D.pkl"))
print("PLAN_FILE=" + (cands[0] if cands else "NOT FOUND"))
if cands:
    plans = pickle.load(open(cands[0], "rb"))
    print("transpose_forward=", plans.get("transpose_forward"))
    pps = plans.get("plans_per_stage")
    print("plans_per_stage type:", type(pps).__name__)
    # v1: list of stage dicts; print stage0 patch/batch defensively
    try:
        s0 = pps[0]
        print("stage0 keys:", list(s0.keys()) if isinstance(s0, dict) else s0)
        if isinstance(s0, dict):
            print("patch_size=", s0.get("patch_size"), "batch_size=", s0.get("batch_size"),
                  "current_spacing=", s0.get("current_spacing"))
    except Exception as e:
        print("parse note:", e)
PY

echo "=== DONE ==="
