#!/usr/bin/env python3
"""Generate per-(network, seed) prediction job scripts.

3D networks (nnunet_v1 / swinunetr / mednext_s / segformer3d) use the unified
framework.predict sliding-window entry. 2D nnUNet uses the native nnUNet_predict
(same sliding-window semantics, 2D). All obey the locked prediction protocol
(spec §5.4): overlap 0.5, TTA off, no ensemble/postproc, argmax → nii.gz in the
original spacing.

Usage:
  python generate_predict_jobs.py             # all 15 scripts
  python generate_predict_jobs.py --network nnunet_v1 --seed 20260520
"""
import argparse
import stat
from pathlib import Path

ARTICLE = "/home/share/hzau/home/liuyangfan/swine-CT-article"
SCRIPT_DIR = Path(__file__).resolve().parent

NETWORKS_3D = ["nnunet_v1", "swinunetr", "mednext_s", "segformer3d"]
SEEDS = [20260520, 20260521, 20260522]


def _3d_predict(network: str, seed: int) -> str:
    name = f"predict_{network}_seed{seed}"
    return f"""#!/usr/bin/env bash
#DSUB -n {name}
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=8;mem=48000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo {ARTICLE}/jobs/predict/{name}.out
#DSUB -eo {ARTICLE}/jobs/predict/{name}.err
set -euo pipefail

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base={ARTICLE}/data/nnunetv1
export nnUNet_preprocessed={ARTICLE}/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER={ARTICLE}/data/nnunetv1/nnUNet_results
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

CKPT={ARTICLE}/data/nnunetv1/v1_comparison/{network}__seed{seed}/fold_0/model_final_checkpoint.model
INPUT={ARTICLE}/data/nnunetv1/nnUNet_raw_data/Task601_Article622_Carcass9Class/imagesTs
OUT={ARTICLE}/data/nnunetv1/v1_comparison_predictions/{network}__seed{seed}
mkdir -p "${{OUT}}"
cd "{ARTICLE}"
echo "=== predict {network} seed={seed} ==="
"${{NNUNETV1_PYTHON}}" -m framework.predict \\
    --network {network} --seed {seed} \\
    --checkpoint "${{CKPT}}" \\
    --input-folder "${{INPUT}}" \\
    --output-folder "${{OUT}}" \\
    --step-size 0.5
echo "=== {name} DONE (exit $?) ==="
"""


def _2d_predict(seed: int) -> str:
    name = f"predict_nnunet_2d_seed{seed}"
    return f"""#!/usr/bin/env bash
#DSUB -n {name}
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=8;mem=48000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo {ARTICLE}/jobs/predict/{name}.out
#DSUB -eo {ARTICLE}/jobs/predict/{name}.err
set -euo pipefail

# 2D nnUNet — native nnUNet_predict (same locked sliding-window protocol, 2D)
export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base={ARTICLE}/data/nnunetv1
export nnUNet_preprocessed={ARTICLE}/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER={ARTICLE}/data/nnunetv1/v1_comparison_2d_root/seed{seed}/runs/nnunet_2d__seed{seed}
source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

INPUT={ARTICLE}/data/nnunetv1/nnUNet_raw_data/Task601_Article622_Carcass9Class/imagesTs
OUT={ARTICLE}/data/nnunetv1/v1_comparison_predictions/nnunet_2d__seed{seed}
mkdir -p "${{OUT}}"
echo "=== predict nnunet_2d seed={seed} (native nnUNet_predict, TTA off, overlap 0.5) ==="
nnUNet_predict -i "${{INPUT}}" -o "${{OUT}}" -t 601 -tr nnUNetTrainerV2 -m 2d -f 0 \\
    --disable_tta --step_size 0.5
echo "=== {name} DONE (exit $?) ==="
"""


def write_script(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-2d", action="store_true")
    args = ap.parse_args()

    nets = [args.network] if args.network else NETWORKS_3D
    seeds = [args.seed] if args.seed else SEEDS
    written = []
    for net in nets:
        for seed in seeds:
            name = f"run_predict_{net}_seed{seed}.sh"
            write_script(SCRIPT_DIR / name, _3d_predict(net, seed))
            written.append(name)
    if not args.no_2d and not args.network:
        for seed in SEEDS:
            name = f"run_predict_nnunet_2d_seed{seed}.sh"
            write_script(SCRIPT_DIR / name, _2d_predict(seed))
            written.append(name)
    print(f"generated {len(written)} predict scripts:")
    for n in written:
        print(f"  jobs/predict/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
