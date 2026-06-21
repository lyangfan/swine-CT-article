#!/usr/bin/env python3
"""Quick train+val iteration smoke (catches run_online_evaluation bugs without
running 500 epochs). Runs a few train iters (with backprop) and a few val iters
(with online evaluation) for the named network, on the real Task601 pipeline.

Requires preprocessed data already unpacked (run once via a full training job's
initialize, or unpack_dataset manually).

Usage:
    python -m framework.smoke_trainval --network nnunet_v1
    python -m framework.smoke_trainval --network swinunetr
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--n-train-iters", type=int, default=3)
    ap.add_argument("--n-val-iters", type=int, default=3)
    args = ap.parse_args()

    from determinism import (  # type: ignore
        install_v1_determinism_patches, seed_everything,
    )
    seed_everything(args.seed + 0)
    record_dir = Path("/tmp/smoke_trainval_records")
    record_dir.mkdir(parents=True, exist_ok=True)
    install_v1_determinism_patches(args.seed, 0, record_dir, f"smoke_{args.network}")

    from framework.registry import get_network_spec
    import framework.nets  # noqa
    from framework.base_trainer import MultiNetworkTrainer
    from nnunet.run.default_configuration import get_default_configuration
    from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

    task_name = convert_id_to_task_name(601)
    plans_file, _, dataset_directory, batch_dice, stage, _ = get_default_configuration(
        "3d_fullres", task_name, "nnUNetTrainerV2", "nnUNetPlansv2.1"
    )
    spec = get_network_spec(args.network)
    out = f"/tmp/smoke_trainval_{args.network}"
    trainer = MultiNetworkTrainer(
        plans_file, 0, output_folder=out, dataset_directory=dataset_directory,
        batch_dice=batch_dice, stage=stage, unpack_data=True, deterministic=True,
        fp16=True, network_spec=spec, base_seed=args.seed,
    )
    trainer.initialize(training=True)
    print(f"[{args.network}] initialized; protocol={spec.forward_protocol} "
          f"family={spec.family} opt={type(trainer.optimizer).__name__}", flush=True)

    # train iters (with backprop) — exercises run_iteration forward/backward path
    trainer.network.do_ds = (spec.forward_protocol == "deep_supervision")
    trainer.network.train()
    # AMP grad scaler is normally init'd by run_training(); init it here since we
    # call run_iteration directly.
    trainer._maybe_init_amp()
    for i in range(args.n_train_iters):
        loss = trainer.run_iteration(trainer.tr_gen, do_backprop=True, run_online_evaluation=False)
        print(f"  train iter {i}: loss={loss:.4f}", flush=True)

    # val iters (online evaluation) — exercises run_online_evaluation for the protocol
    trainer.network.eval()
    for i in range(args.n_val_iters):
        with __import__("torch").no_grad():
            loss = trainer.run_iteration(trainer.val_gen, do_backprop=False, run_online_evaluation=True)
        print(f"  val iter {i}: loss={loss:.4f} (online_eval OK)", flush=True)

    # finish_online_evaluation aggregates the collected tp/fp/fp — catches shape bugs there too
    from nnunet.training.network_training.network_trainer import NetworkTrainer
    NetworkTrainer.finish_online_evaluation(trainer)
    print(f"[{args.network}] finish_online_evaluation OK; "
          f"all_val_eval_metrics={getattr(trainer, 'all_val_eval_metrics', None)}", flush=True)
    print(f"[{args.network}] PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print(f"FAIL: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
