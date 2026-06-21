#!/usr/bin/env python3
"""Config-driven training entry for the v1 input-consistency comparison.

Constructs a :class:`framework.base_trainer.MultiNetworkTrainer` for the named
registered network + base seed, installs the v1 determinism patches, and runs
500 ep × 250 iters = 125 000 iterations on the locked Task601 pipeline.

Usage (inside the nnunetv1 env, paths already exported):

    python -m framework.train \
        --network swinunetr --seed 20260520 \
        --config configs/swinunetr.yaml \
        --output-folder <RESULTS>/swinunetr__seed20260520

This module is 3D-only. The 2D nnUNet reference trains via the existing
``train_paca_deterministic.py --network 2d`` wrapper (native nnUNetTrainerV2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _ensure_framework_on_path() -> None:
    """Make `framework.*` importable when run as a script from any CWD."""
    here = Path(__file__).resolve().parent
    root = here.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    _ensure_framework_on_path()
    parser = argparse.ArgumentParser(description="v1 input-consistency training entry (3D)")
    parser.add_argument("--network", required=True, help="registered network name (e.g. nnunet_v1, swinunetr, mednext_s, segformer3d)")
    parser.add_argument("--seed", type=int, required=True, help="base seed (20260520 / 20260521 / 20260522)")
    parser.add_argument("--config", required=True, help="configs/<network>.yaml (informational; hyper-params are pinned in the trainer)")
    parser.add_argument("--task-id", type=int, default=601)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--network-dim", default="3d_fullres", choices=["3d_fullres"])
    parser.add_argument("--plans-identifier", default="nnUNetPlansv2.1")
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--grad-accum", type=int, default=1, help="physical-batch multiplier for effective batch 2 (default 1)")
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    output_folder = Path(args.output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    record_dir = output_folder.parent / "records"
    record_dir.mkdir(parents=True, exist_ok=True)

    # --- determinism: PYTHONHASHSEED + seeds + v1 worker-seed patches ---
    # install_v1_determinism_patches must run BEFORE initialize() (it patches
    # get_moreDA_augmentation, which initialize() calls).
    from determinism import (  # type: ignore  (on PYTHONPATH = NNUNETV1_COMPAT_ROOT)
        ensure_pythonhashseed_or_reexec,
        install_v1_determinism_patches,
        seed_everything,
        sha256_file,
    )
    ensure_pythonhashseed_or_reexec(args.seed)
    fold_seed = args.seed + args.fold
    seed_everything(fold_seed)
    patch_info = install_v1_determinism_patches(
        args.seed, args.fold, record_dir, f"{args.network}__seed{args.seed}"
    )

    from framework.registry import get_network_spec
    import framework.nets  # noqa: F401 — registers all plugins
    from framework.base_trainer import MultiNetworkTrainer
    from nnunet.run.default_configuration import get_default_configuration
    from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

    task_name = convert_id_to_task_name(args.task_id)
    plans_file, _, dataset_directory, batch_dice, stage, _ = get_default_configuration(
        args.network_dim, task_name, "nnUNetTrainerV2", args.plans_identifier
    )
    spec = get_network_spec(args.network)

    trainer = MultiNetworkTrainer(
        plans_file,
        args.fold,
        output_folder=str(output_folder),
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=True,
        deterministic=True,
        fp16=True,
        network_spec=spec,
        base_seed=args.seed,
        grad_accum_steps=args.grad_accum,
    )

    config = {"network": args.network, "seed": args.seed, "fold": args.fold, "config": args.config,
              "task_id": args.task_id, "stage": stage, "batch_dice": batch_dice,
              "grad_accum_steps": args.grad_accum, "fold_seed": fold_seed,
              "plans_file": plans_file, "dataset_directory": dataset_directory,
              "spec": {"forward_protocol": spec.forward_protocol, "family": spec.family,
                       "conv_op": spec.conv_op.__name__, "needs_wrap": spec.needs_wrap,
                       "ds_scales_override": spec.ds_scales_override},
              "trainer_script": str(Path(__file__).resolve()),
              "trainer_script_sha256": sha256_file(Path(__file__).resolve())}
    (record_dir / f"env_{args.network}__seed{args.seed}.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    started = time.time()
    try:
        trainer.initialize(training=True)
        trainer.run_training()
    except Exception as exc:
        failure = {"network": args.network, "seed": args.seed, "status": "failed",
                   "error": repr(exc), "traceback": traceback.format_exc()}
        (record_dir / f"failed_{args.network}__seed{args.seed}.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[train] FAILED {args.network} seed={args.seed}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    final_ckpt = output_folder / "model_final_checkpoint.model"
    done = {"network": args.network, "seed": args.seed, "status": "completed",
            "elapsed_seconds": time.time() - started,
            "final_checkpoint": str(final_ckpt),
            "final_checkpoint_exists": final_ckpt.exists(),
            "epoch": int(trainer.epoch),
            "max_num_epochs": int(trainer.max_num_epochs),
            "num_batches_per_epoch": int(trainer.num_batches_per_epoch),
            "global_step": int(trainer._global_step)}
    (record_dir / f"done_{args.network}__seed{args.seed}.json").write_text(
        json.dumps(done, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[train] DONE {args.network} seed={args.seed} in {done['elapsed_seconds']:.0f}s; "
          f"final_ckpt={final_ckpt} exists={final_ckpt.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
