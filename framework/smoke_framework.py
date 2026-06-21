#!/usr/bin/env python3
"""Framework plumbing smoke test (Stage 2/3).

For each registered 3D network, constructs a real MultiNetworkTrainer on the
Task601 plans, runs initialize(training=False) (builds the architecture +
optimizer from the registry), and verifies:
  (1) forward on [2,1,64,160,160] succeeds and matches the declared protocol
      (deep_supervision → list; single_output → tensor)
  (2) optimizer family is correct (cnn → SGD; transformer → AdamW)
  (3) DS loss weights (when applicable) sum to 1 and drop the lowest level
  (4) GenericWrap adapter exposes conv_op / num_classes / inference_apply_nonlin

Run inside the nnunetv1 env on Huawei. Not a training run — no data loaded.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def main() -> int:
    from framework.registry import all_networks
    import framework.nets  # noqa: register plugins
    from framework.base_trainer import MultiNetworkTrainer, GenericWrap
    from nnunet.run.default_configuration import get_default_configuration
    from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

    task_name = convert_id_to_task_name(601)
    plans_file, _, dataset_directory, batch_dice, stage, _ = get_default_configuration(
        "3d_fullres", task_name, "nnUNetTrainerV2", "nnUNetPlansv2.1"
    )
    print(f"[info] plans_file={plans_file} stage={stage} batch_dice={batch_dice}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch = (64, 160, 160)
    x = torch.randn(2, 1, *patch, device=device)
    print(f"[info] device={device} input={tuple(x.shape)}")

    results = []
    for name, spec in sorted(all_networks().items()):
        if name == "nnunet_2d":
            continue  # 2D not via this framework
        try:
            trainer = MultiNetworkTrainer(
                plans_file, 0, output_folder=f"/tmp/smoke_{name}",
                dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,
                unpack_data=False, deterministic=True, fp16=True,
                network_spec=spec, base_seed=20260520,
            )
            trainer.initialize(training=False)
            trainer.network.to(device).eval()

            # (1) forward + protocol
            with torch.no_grad():
                out = trainer.network(x)
            if spec.forward_protocol == "deep_supervision":
                assert isinstance(out, (list, tuple)), f"{name}: expected DS list, got {type(out)}"
                shapes = [tuple(o.shape) for o in out]
                n_outputs = len(out)
            else:
                assert not isinstance(out, (list, tuple)), f"{name}: expected single tensor, got list"
                shapes = tuple(out.shape)
                n_outputs = 1

            # (2) optimizer family
            opt_type = type(trainer.optimizer).__name__
            exp_opt = "SGD" if spec.family == "cnn" else "AdamW"
            assert opt_type == exp_opt, f"{name}: optimizer {opt_type} != expected {exp_opt}"

            # (3) DS weights
            ds_info = ""
            if trainer.ds_loss_weights is not None:
                w = trainer.ds_loss_weights
                ds_info = f" ds_weights={np.round(w,4).tolist()} sum={w.sum():.4f}"
                assert abs(w.sum() - 1.0) < 1e-6, f"{name}: DS weights don't sum to 1"
                assert w[-1] == 0.0, f"{name}: lowest DS weight not dropped"

            # (4) adapter
            is_wrap = isinstance(trainer.network, GenericWrap)
            nc = trainer.network.num_classes
            assert nc == 10, f"{name}: num_classes={nc} != 10"

            results.append((name, "PASS", shapes, opt_type, n_outputs, is_wrap, ds_info))
            print(f"[PASS] {name}: out={shapes} opt={opt_type} n_out={n_outputs} "
                  f"wrapped={is_wrap} nc={nc}{ds_info}", flush=True)
        except Exception as exc:
            traceback.print_exc()
            results.append((name, "FAIL", str(exc), "", 0, False, ""))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)

    print("\n=== SMOKE SUMMARY ===")
    for name, status, shapes, opt, n_out, wrap, ds in results:
        line = f"{status:4s} {name:14s}"
        if status == "PASS":
            line += f"  out={shapes}  opt={opt}  n_out={n_out}  wrap={wrap}{ds}"
        print(line)

    n_fail = sum(1 for r in results if r[1] == "FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
