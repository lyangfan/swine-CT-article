#!/usr/bin/env python3
"""Debug telemetry smoke — check conditional mirror wiring."""
import sys, json, traceback
sys.path.insert(0, "/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/scripts/nnunetv1_compat")
from determinism import seed_everything
seed_everything(20260520)

from framework.registry import get_network_spec
import framework.nets
from framework.base_trainer import MultiNetworkTrainer
from nnunet.run.default_configuration import get_default_configuration
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from pathlib import Path

task_name = convert_id_to_task_name(601)
plans_file, _, dataset_directory, batch_dice, stage, _ = get_default_configuration(
    "3d_fullres", task_name, "nnUNetTrainerV2", "nnUNetPlansv2.1"
)
spec = get_network_spec("nnunet_v1")
out = "/tmp/smoke_condlr2"
trainer = MultiNetworkTrainer(
    plans_file, 0, output_folder=out, dataset_directory=dataset_directory,
    batch_dice=batch_dice, stage=stage, unpack_data=True, deterministic=True,
    fp16=True, network_spec=spec, base_seed=20260520,
    lr_mirror_mode="conditional", conditional_mirror_axis=0,
)
print("mirror_patch_info before init:", trainer._mirror_patch_info)
trainer.initialize(training=True)
print("mirror_patch_info after init:", type(trainer._mirror_patch_info))
if trainer._mirror_patch_info:
    print("keys:", list(trainer._mirror_patch_info.keys()))
    print("telemetry_path:", trainer._mirror_patch_info.get("telemetry_path"))
else:
    print("_mirror_patch_info is None!")
    sys.exit(1)

# Check tr_gen transform names
from framework.transforms.conditional_mirror import _collect_transform_names
names = _collect_transform_names(trainer.tr_gen)
mirror_count = sum(1 for n in names if n == "MirrorTransform")
cond_count = sum(1 for n in names if n == "ConditionalMirrorTransform")
print("transform names (%d total): %s ..." % (len(names), names[:5]))
print("MirrorTransform count: %d" % mirror_count)
print("ConditionalMirrorTransform count: %d" % cond_count)

# Check witness
from pathlib import Path
witness_path = Path(out).parent / "records" / "augmentation_witness.json"
print("witness_path:", witness_path)
print("witness exists:", witness_path.exists())
if witness_path.exists():
    w = json.loads(witness_path.read_text())
    print("remaining_original:", w.get("remaining_original_MirrorTransform_count"))

# Run a few iters
print("\n=== Run 5 train iters ===")
trainer.network.do_ds = (spec.forward_protocol == "deep_supervision")
trainer.network.train()
trainer._maybe_init_amp()
for i in range(5):
    loss = trainer.run_iteration(trainer.tr_gen, do_backprop=True, run_online_evaluation=False)
    print("  iter %d: loss=%.4f" % (i, loss))

# Read telemetry
from framework.transforms.conditional_mirror import read_telemetry_verdict
telemetry = read_telemetry_verdict(Path(out).parent / "records")
for k, v in telemetry.items():
    print("  %s: %s" % (k, v))
print("\n=== DONE ===")
