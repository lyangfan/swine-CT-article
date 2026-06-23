#!/usr/bin/env python3
"""Generate per-(network, seed) DSUB training job scripts from a template.

DSUB parses ``#DSUB`` directives literally (no bash expansion), so each job
needs its own script with the network + seed baked into BOTH the directives
(job name, stdout/err paths) and the runtime env. This generator emits them all.

Grid:
  3D: nnunet_v1 / swinunetr / mednext_s / segformer3d  ×  seeds 20260520/21/22  = 12
  2D: nnunet_2d (via train_paca_deterministic.py)     ×  seeds 20260520/21/22  =  3

Conditional LR-mirror variant (--variant condlr):
  3D: swinunetr / nnunet_v1  ×  seeds 20260520/21/22  =  6
  2D: nnunet_2d (via framework/train.py --network-dim 2d)  ×  seeds 20260520/21/22  =  3

Usage:
  python generate_train_jobs.py                              # write all 15 baseline scripts
  python generate_train_jobs.py --variant condlr             # write 6 condlr scripts
  python generate_train_jobs.py --variant condlr --lr-axis-manifest <path>
  python generate_train_jobs.py --network nnunet_v1 --seed 20260520  # one script
"""
import argparse
import json
import os
import stat
from pathlib import Path

ARTICLE = "/home/share/hzau/home/liuyangfan/swine-CT-article"
SCRIPT_DIR = Path(__file__).resolve().parent

NETWORKS_3D = ["nnunet_v1", "swinunetr", "mednext_s", "segformer3d"]
NETWORKS_2D = ["nnunet_2d"]
# For conditional variant: only the two networks in the ablation scope
CONDLR_NETWORKS_3D = ["swinunetr"]  # only SwinUNETR (spec §2: 2 networks total)
CONDLR_NETWORKS_2D = ["nnunet_2d"]
SEEDS = [20260520, 20260521, 20260522]


def _load_axis_from_manifest(manifest_path: str | None, network_dim: str) -> int:
    """Read confirmed_lr_axis from the audit manifest."""
    if manifest_path is None:
        # Default from Stage 0 audit
        return 0  # both 3D and 2D confirmed as axis 0
    with open(manifest_path) as f:
        manifest = json.load(f)
    key = "confirmed_lr_axis_3d" if network_dim == "3d_fullres" else "confirmed_lr_axis_2d"
    axis = manifest.get(key)
    if axis is None:
        raise KeyError(f"{key} not found in {manifest_path}")
    return int(axis)


def _3d_script(network: str, seed: int) -> str:
    name = f"train_{network}_seed{seed}"
    return f"""#!/usr/bin/env bash
#DSUB -n {name}
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=16;mem=64000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo {ARTICLE}/jobs/train/{name}.out
#DSUB -eo {ARTICLE}/jobs/train/{name}.err
set -euo pipefail

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base={ARTICLE}/data/nnunetv1
export nnUNet_preprocessed={ARTICLE}/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER={ARTICLE}/data/nnunetv1/nnUNet_results

source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.8.1_cuda11 libs/nccl/2.16.5-1_cuda11.8 libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONHASHSEED={seed}

OUT={ARTICLE}/data/nnunetv1/v1_comparison/{network}__seed{seed}
mkdir -p "${{OUT}}"
cd "{ARTICLE}"
echo "=== train {network} seed={seed} -> ${{OUT}} ==="
"${{NNUNETV1_PYTHON}}" -m framework.train \\
    --network {network} \\
    --seed {seed} \\
    --config configs/{network}.yaml \\
    --task-id 601 --fold 0 \\
    --output-folder "${{OUT}}"
echo "=== {name} DONE (exit $?) ==="
"""


def _3d_condlr_script(network: str, seed: int, axis: int) -> str:
    name = f"train_{network}_condlr_seed{seed}"
    return f"""#!/usr/bin/env bash
#DSUB -n {name}
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=16;mem=64000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo {ARTICLE}/jobs/train/{name}.out
#DSUB -eo {ARTICLE}/jobs/train/{name}.err
set -euo pipefail

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base={ARTICLE}/data/nnunetv1
export nnUNet_preprocessed={ARTICLE}/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER={ARTICLE}/data/nnunetv1/nnUNet_results

source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.8.1_cuda11 libs/nccl/2.16.5-1_cuda11.8 libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONHASHSEED={seed}

OUT={ARTICLE}/data/nnunetv1/v1_comparison/{network}__condlr_seed{seed}
mkdir -p "${{OUT}}"
cd "{ARTICLE}"
echo "=== train {network} condlr seed={seed} axis={axis} -> ${{OUT}} ==="
"${{NNUNETV1_PYTHON}}" -m framework.train \\
    --network {network} \\
    --seed {seed} \\
    --config configs/{network}.yaml \\
    --task-id 601 --fold 0 \\
    --lr-mirror-mode conditional \\
    --conditional-mirror-axis {axis} \\
    --output-folder "${{OUT}}"
echo "=== {name} DONE (exit $?) ==="
"""


def _2d_script(seed: int) -> str:
    name = f"train_nnunet_2d_seed{seed}"
    return f"""#!/usr/bin/env bash
#DSUB -n {name}
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=16;mem=64000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo {ARTICLE}/jobs/train/{name}.out
#DSUB -eo {ARTICLE}/jobs/train/{name}.err
set -euo pipefail

# 2D nnUNet reference — native nnUNetTrainerV2_2D via train_paca_deterministic.py
# (NOT the MultiNetworkTrainer; determinism patches apply globally before init).
export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base={ARTICLE}/data/nnunetv1
export nnUNet_preprocessed={ARTICLE}/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER={ARTICLE}/data/nnunetv1/nnUNet_results

source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.8.1_cuda11 libs/nccl/2.16.5-1_cuda11.8 libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONHASHSEED={seed}

PREPROCESSED={ARTICLE}/data/nnunetv1/nnUNet_preprocessed
RAW_BASE={ARTICLE}/data/nnunetv1
RESULTS={ARTICLE}/data/nnunetv1/v1_comparison
ROOT={ARTICLE}/data/nnunetv1/v1_comparison_2d_root
mkdir -p "${{ROOT}}"

cd "{ARTICLE}"
echo "=== train nnunet_2d seed={seed} -> ${{RESULTS}}/nnunet_2d__seed{seed} ==="
"${{NNUNETV1_PYTHON}}" \\
    /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/scripts/nnunetv1_compat/train_paca_deterministic.py \\
    --root "${{ROOT}}/seed{seed}" \\
    --run-name nnunet_2d__seed{seed} \\
    --base-seed {seed} \\
    --task-id 601 \\
    --network 2d \\
    --plans-identifier nnUNetPlansv2.1 \\
    --fold 0 \\
    --epochs 500 \\
    --train-batches 250 \\
    --val-batches 50 \\
    --raw-base "${{RAW_BASE}}/nnUNet_raw_data" \\
    --preprocessed "${{PREPROCESSED}}" \\
    --results-folder "${{RESULTS}}/nnunet_2d__seed{seed}" \\
    --fp16 --unpack-data \\
    --checkpoint-policy final-best
echo "=== {name} DONE (exit $?) ==="
"""


def _2d_condlr_script(seed: int, axis: int) -> str:
    """2D conditional LR-mirror: uses framework/train.py (not train_paca_deterministic.py)
    because the conditional mirror patch must be installed via MultiNetworkTrainer."""
    name = f"train_nnunet_2d_condlr_seed{seed}"
    return f"""#!/usr/bin/env bash
#DSUB -n {name}
#DSUB -N 1
#DSUB -A root.hzau
#DSUB -R "cpu=16;mem=64000;gpu=1"
#DSUB -pn '!whshare-agent-174'
#DSUB -oo {ARTICLE}/jobs/train/{name}.out
#DSUB -eo {ARTICLE}/jobs/train/{name}.err
set -euo pipefail

export SWCT_RESPECT_EXISTING_NNUNETV1_PATHS=1
export nnUNet_raw_data_base={ARTICLE}/data/nnunetv1
export nnUNet_preprocessed={ARTICLE}/data/nnunetv1/nnUNet_preprocessed
export RESULTS_FOLDER={ARTICLE}/data/nnunetv1/nnUNet_results

source /home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/runs/swct06042040_codex_hv1_kidney_phase2_v3/task/tools/adopted/setup_nnunetv1_env.sh
module load compilers/gcc/9.3.0 compilers/cuda/11.8.0 libs/cudnn/8.8.1_cuda11 libs/nccl/2.16.5-1_cuda11.8 libs/openblas/0.3.18_kgcc9.3.1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONHASHSEED={seed}

OUT={ARTICLE}/data/nnunetv1/v1_comparison/nnunet_2d__condlr_seed{seed}
mkdir -p "${{OUT}}"
cd "{ARTICLE}"
echo "=== train nnunet_2d condlr seed={seed} axis={axis} -> ${{OUT}} ==="
"${{NNUNETV1_PYTHON}}" -m framework.train \\
    --network nnunet_v1 \\
    --network-dim 2d \\
    --seed {seed} \\
    --config configs/nnunet_2d.yaml \\
    --task-id 601 --fold 0 \\
    --lr-mirror-mode conditional \\
    --conditional-mirror-axis {axis} \\
    --output-folder "${{OUT}}"
echo "=== {name} DONE (exit $?) ==="
"""


def write_script(path: Path, content: str) -> None:
    path.write_text(content)
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default=None, help="generate only this network")
    ap.add_argument("--seed", type=int, default=None, help="generate only this seed")
    ap.add_argument("--no-2d", action="store_true", help="skip 2D scripts")
    ap.add_argument("--variant", default="baseline", choices=["baseline", "condlr"],
                    help="baseline (default) or condlr (conditional LR mirror)")
    ap.add_argument("--lr-axis-manifest", default=None,
                    help="path to lr_axis_audit_manifest.json (reads confirmed axis for condlr)")
    args = ap.parse_args()

    seeds = [args.seed] if args.seed else SEEDS
    written = []

    if args.variant == "condlr":
        # Conditional LR-mirror ablation: SwinUNETR + nnU-Net 2D (spec §2)
        axis_3d = _load_axis_from_manifest(args.lr_axis_manifest, "3d_fullres")
        axis_2d = _load_axis_from_manifest(args.lr_axis_manifest, "2d")

        nets_3d = [args.network] if args.network else CONDLR_NETWORKS_3D
        for net in nets_3d:
            if net not in CONDLR_NETWORKS_3D:
                print(f"WARNING: {net} not in condlr scope (spec §2), skipping")
                continue
            for seed in seeds:
                name = f"run_{net}_condlr_seed{seed}.sh"
                write_script(SCRIPT_DIR / name, _3d_condlr_script(net, seed, axis_3d))
                written.append(name)

        if not args.no_2d and not args.network:
            for seed in seeds:
                name = f"run_nnunet_2d_condlr_seed{seed}.sh"
                write_script(SCRIPT_DIR / name, _2d_condlr_script(seed, axis_2d))
                written.append(name)
    else:
        # Baseline: original 15-job grid
        nets_3d = [args.network] if args.network else NETWORKS_3D
        for net in nets_3d:
            for seed in seeds:
                name = f"run_{net}_seed{seed}.sh"
                write_script(SCRIPT_DIR / name, _3d_script(net, seed))
                written.append(name)

        if not args.no_2d and not args.network:
            for seed in SEEDS:
                name = f"run_nnunet_2d_seed{seed}.sh"
                write_script(SCRIPT_DIR / name, _2d_script(seed))
                written.append(name)

    print(f"generated {len(written)} job scripts in {SCRIPT_DIR}:")
    for n in written:
        print(f"  jobs/train/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
