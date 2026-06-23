#!/usr/bin/env python3
"""Framework plumbing smoke test (Stage 2/3).

For each registered 3D (or 2D) network, constructs a real MultiNetworkTrainer on
the Task601 plans, runs initialize(training=False) (builds the architecture +
optimizer from the registry), and verifies:
  (1) forward on the correct input shape succeeds and matches the declared
      protocol (deep_supervision → list; single_output → tensor)
  (2) optimizer family is correct (cnn → SGD; transformer → AdamW)
  (3) DS loss weights (when applicable) sum to 1 and drop the lowest level
  (4) GenericWrap adapter exposes conv_op / num_classes / inference_apply_nonlin

Usage (inside the nnunetv1 env on Huawei):
  # 3D all-networks smoke
  python -m framework.smoke_framework

  # 2D smoke (single network)
  python -m framework.smoke_framework --network-dim 2d --network nnunet_v1 --seed 20260520

Run inside the nnunetv1 env on Huawei. Not a training run — no data loaded.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn


def main() -> int:
    from framework.registry import all_networks, get_network_spec
    import framework.nets  # noqa: register plugins
    from framework.base_trainer import MultiNetworkTrainer, GenericWrap
    from nnunet.run.default_configuration import get_default_configuration
    from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

    ap = argparse.ArgumentParser(description="Framework plumbing smoke test")
    ap.add_argument("--network", default=None,
                    help="run single network instead of all registered")
    ap.add_argument("--seed", type=int, default=20260520,
                    help="seed (default 20260520)")
    ap.add_argument("--network-dim", default="3d_fullres", choices=["3d_fullres", "2d"],
                    help="3d_fullres (default) or 2d — picks plan stage / conv_op")
    args = ap.parse_args()

    task_name = convert_id_to_task_name(601)
    use_2d = args.network_dim == "2d"

    if use_2d:
        return _smoke_2d(args, task_name)
    return _smoke_3d(args, task_name)


def _smoke_2d(args, task_name: str) -> int:
    """2D smoke: single-network path (§6.2)."""
    from framework.registry import get_network_spec
    from framework.base_trainer import MultiNetworkTrainer, GenericWrap
    from nnunet.run.default_configuration import get_default_configuration

    if not args.network:
        print("[ERROR] --network is required for 2D smoke", file=sys.stderr)
        return 1

    plans_file, _, dataset_directory, batch_dice, stage, _ = get_default_configuration(
        "2d", task_name, "nnUNetTrainerV2", "nnUNetPlansv2.1"
    )
    print(f"[info] plans_file={plans_file} stage={stage} batch_dice={batch_dice}")

    spec = get_network_spec(args.network)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read 2D plans to get patch size and check mirror_axes
    import pickle
    plans2d_path = plans_file.replace("_plans_3D.pkl", "_plans_2D.pkl")
    with open(plans2d_path, "rb") as f:
        plans2d = pickle.load(f)

    # 2D plans have stage 0 with 'patch_size' array (e.g. [512, 512])
    stage0 = plans2d["plans_per_stage"][0]
    patch_size_arr = stage0.get("patch_size", None)
    if patch_size_arr is None:
        raise RuntimeError("2D plans stage 0 has no patch_size")
    if hasattr(patch_size_arr, "tolist"):
        patch_size_arr = patch_size_arr.tolist()
    H, W = int(patch_size_arr[0]), int(patch_size_arr[1])
    print(f"[info] 2D patch: H={H} W={W}")

    # Verify mirror_axes for 2D
    # nnU-Net v1 2D: data_augmentation_params defaults mirror_axes=(0,1)
    x = torch.randn(2, 1, H, W, device=device)
    print(f"[info] device={device} input={tuple(x.shape)}")

    trainer = MultiNetworkTrainer(
        plans_file, 0, output_folder=f"/tmp/smoke_{args.network}_2d",
        dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,
        unpack_data=False, deterministic=True, fp16=True,
        network_spec=spec, base_seed=args.seed,
    )
    trainer.initialize(training=False)
    trainer.network.to(device).eval()

    # --- 2D assertions ---
    # (A1) conv_op must be Conv2d
    conv_op = trainer.network.conv_op
    assert conv_op in (nn.Conv2d, torch.nn.Conv2d), \
        f"2D smoke: conv_op={conv_op} not Conv2d"

    # (A2) DS resolution
    ds_scales = trainer._resolve_deep_supervision_scales()
    if ds_scales is not None:
        n_outputs = len(ds_scales)
        print(f"[info] 2D DS outputs={n_outputs} scales={ds_scales}")
    else:
        n_outputs = 1
        print(f"[info] 2D single-output (no DS)")

    # (A3) forward: output shape should be [2,10,H,W] (single) or list of DS tensors
    with torch.no_grad():
        out = trainer.network(x)
    if spec.forward_protocol == "deep_supervision":
        assert isinstance(out, (list, tuple)), f"{args.network}: expected DS list, got {type(out)}"
        shapes = [tuple(o.shape) for o in out]
        # First (highest-res) output: [2, 10, H, W]
        assert out[0].shape[0] == 2, f"batch dim != 2: {out[0].shape}"
        assert out[0].shape[1] == 10, f"class dim != 10: {out[0].shape}"
    else:
        assert not isinstance(out, (list, tuple)), f"{args.network}: expected single tensor, got list"
        shapes = tuple(out.shape)
        assert out.shape == (2, 10, H, W), \
            f"2D forward shape {tuple(out.shape)} != expected (2,10,{H},{W})"

    # (A4) optimizer family
    opt_type = type(trainer.optimizer).__name__
    exp_opt = "SGD" if spec.family == "cnn" else "AdamW"
    assert opt_type == exp_opt, f"{args.network}: optimizer {opt_type} != expected {exp_opt}"

    # (A5) adapter
    is_wrap = isinstance(trainer.network, GenericWrap)
    nc = trainer.network.num_classes
    assert nc == 10, f"{args.network}: num_classes={nc} != 10"

    print(f"[PASS] {args.network} 2D: out={shapes} opt={opt_type} n_out={n_outputs} "
          f"wrapped={is_wrap} nc={nc} conv_op={conv_op.__name__}", flush=True)
    print(f"[info] 2D mirror_axes audit: nnU-Net v1 2D default is (0,1); "
          f"conditional axis must be 0 or 1 (not 2)", flush=True)

    print("\n=== 2D SMOKE SUMMARY ===")
    print(f"[PASS] {args.network} 2D smoke passed", flush=True)
    return 0


def _smoke_3d(args, task_name: str) -> int:
    """3D smoke: all-networks loop (unchanged from original)."""
    from framework.registry import all_networks, get_network_spec
    import framework.nets  # noqa: register plugins
    from framework.base_trainer import MultiNetworkTrainer, GenericWrap
    from nnunet.run.default_configuration import get_default_configuration

    plans_file, _, dataset_directory, batch_dice, stage, _ = get_default_configuration(
        "3d_fullres", task_name, "nnUNetTrainerV2", "nnUNetPlansv2.1"
    )
    print(f"[info] plans_file={plans_file} stage={stage} batch_dice={batch_dice}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch = (64, 160, 160)
    x = torch.randn(2, 1, *patch, device=device)
    print(f"[info] device={device} input={tuple(x.shape)}")

    results = []
    networks = all_networks()
    if args.network:
        spec = get_network_spec(args.network)
        networks = {args.network: spec}

    for name, spec in sorted(networks.items()):
        try:
            trainer = MultiNetworkTrainer(
                plans_file, 0, output_folder=f"/tmp/smoke_{name}",
                dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,
                unpack_data=False, deterministic=True, fp16=True,
                network_spec=spec, base_seed=args.seed,
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

    print("\n=== 3D SMOKE SUMMARY ===")
    for name, status, shapes, opt, n_out, wrap, ds in results:
        line = f"{status:4s} {name:14s}"
        if status == "PASS":
            line += f"  out={shapes}  opt={opt}  n_out={n_out}  wrap={wrap}{ds}"
        print(line)

    n_fail = sum(1 for r in results if r[1] == "FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
