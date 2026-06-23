import sys, json
from pathlib import Path
sys.path.insert(0, "/home/share/hzau/home/liuyangfan/swine_ct_autonomous_discovery/scripts/nnunetv1_compat")
from determinism import seed_everything
seed_everything(20260520)
from framework.registry import get_network_spec
import framework.nets
from framework.base_trainer import MultiNetworkTrainer
from nnunet.run.default_configuration import get_default_configuration
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

task_name = convert_id_to_task_name(601)
plans_file, _, dataset_directory, batch_dice, stage, _ = get_default_configuration(
    "3d_fullres", task_name, "nnUNetTrainerV2", "nnUNetPlansv2.1")
spec = get_network_spec("nnunet_v1")
out = "/tmp/smoke_final"
trainer = MultiNetworkTrainer(
    plans_file, 0, output_folder=out, dataset_directory=dataset_directory,
    batch_dice=batch_dice, stage=stage, unpack_data=True, deterministic=True,
    fp16=True, network_spec=spec, base_seed=20260520,
    lr_mirror_mode="conditional", conditional_mirror_axis=0,
)
trainer.initialize(training=True)

# Check witness immediately after initialize
witness_path = Path(out).parent / "records" / "augmentation_witness.json"
print("witness_path:", witness_path)
print("witness exists:", witness_path.exists())
if witness_path.exists():
    w = json.loads(witness_path.read_text())
    remaining = w.get("remaining_original_MirrorTransform_count", -1)
    replaced = w.get("original_mirror_transform_replaced", False)
    n_transforms = len(w.get("transform_names", []))
    print("remaining_original_MirrorTransform_count:", remaining)
    print("original_mirror_transform_replaced:", replaced)
    print("n_transforms:", n_transforms)
    assert remaining == 0, "FAIL: remaining=%d" % remaining
    assert replaced, "FAIL: not replaced"
    print("PASS: Witness OK")

# Run a few iters
print("\nRunning 10 train iters...")
trainer.network.do_ds = (spec.forward_protocol == "deep_supervision")
trainer.network.train()
trainer._maybe_init_amp()
for i in range(10):
    loss = trainer.run_iteration(trainer.tr_gen, do_backprop=True, run_online_evaluation=False)

# Check telemetry
from framework.transforms.conditional_mirror import read_telemetry_verdict
telemetry = read_telemetry_verdict(Path(out).parent / "records")
print("\nTelemetry:")
for k in sorted(telemetry.keys()):
    print("  %s: %s" % (k, telemetry[k]))
print("verdict:", telemetry.get("verdict"))
assert telemetry.get("verdict") == "PASS", "FAIL"
assert telemetry.get("protected_lr_mirror_count") == 0, "FAIL: protected LR mirror count != 0"
assert telemetry.get("protected_sample_count", 0) > 0, "FAIL: no protected samples"
print("PASS: Telemetry smoke OK")
