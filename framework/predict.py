#!/usr/bin/env python3
"""Unified sliding-window prediction for the v1 input-consistency comparison.

Constructs the :class:`MultiNetworkTrainer` for the named network, loads the
final checkpoint, and predicts the test set with the LOCKED prediction protocol
(spec §5.4): sliding window, overlap 0.5, TTA off, no ensemble, no
post-processing, argmax segmentation resampled back to the original spacing.

Output: one ``<case>.nii.gz`` per test case (argmax, int16), original geometry.

Usage (inside the nnunetv1 env):

    python -m framework.predict \
        --network swinunetr --seed 20260520 \
        --checkpoint <RESULTS>/.../model_final_checkpoint.model \
        --input-folder <Task601_raw>/imagesTs \
        --output-folder <PRED>/swinunetr__seed20260520
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_framework_on_path() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    _ensure_framework_on_path()
    parser = argparse.ArgumentParser(description="Unified sliding-window prediction")
    parser.add_argument("--network", required=True)
    parser.add_argument("--seed", type=int, required=True, help="base seed (records which checkpoint family)")
    parser.add_argument("--checkpoint", required=True, help="model_final_checkpoint.model path")
    parser.add_argument("--input-folder", required=True, help="imagesTs folder (one _0000.nii.gz per case)")
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--task-id", type=int, default=601)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--plans-identifier", default="nnUNetPlansv2.1")
    parser.add_argument("--step-size", type=float, default=0.5, help="sliding-window overlap (0.5 = 50%)")
    parser.add_argument("--num-threads-preprocessing", type=int, default=2)
    parser.add_argument("--num-threads-nifti-save", type=int, default=1)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    args = parser.parse_args()

    from framework.registry import get_network_spec
    import framework.nets  # noqa: F401
    from framework.base_trainer import MultiNetworkTrainer
    from nnunet.run.default_configuration import get_default_configuration
    from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
    from nnunet.inference.predict import preprocess_multithreaded
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
    from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, isfile
    import numpy as np
    import torch

    output_folder = Path(args.output_folder).resolve()
    maybe_mkdir_p(str(output_folder))

    task_name = convert_id_to_task_name(args.task_id)
    plans_file, _, dataset_directory, batch_dice, stage, _ = get_default_configuration(
        "3d_fullres", task_name, "nnUNetTrainerV2", args.plans_identifier
    )
    spec = get_network_spec(args.network)

    trainer = MultiNetworkTrainer(
        plans_file, args.fold, output_folder=str(output_folder),
        dataset_directory=dataset_directory, batch_dice=batch_dice, stage=stage,
        unpack_data=False, deterministic=True, fp16=args.mixed_precision,
        network_spec=spec, base_seed=args.seed,
    )
    # build architecture (no training data / augmentation needed)
    trainer.initialize(training=False)

    # load trained weights into the architecture
    print(f"[predict] loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    trainer.load_checkpoint_ram(checkpoint, train=False)
    # prediction must run with do_ds=False (already enforced by the V2 wrapper,
    # but set explicitly for safety on the adapter)
    trainer.network.do_ds = False
    trainer.network.eval()

    # --- enumerate test cases ---
    input_folder = Path(args.input_folder)
    case_files = sorted(input_folder.glob("*_0000.nii.gz"))
    if not case_files:
        raise RuntimeError(f"no *_0000.nii.gz under {input_folder}")
    list_of_lists = [[str(f)] for f in case_files]
    output_filenames = [str(output_folder / f.name.replace("_0000.nii.gz", ".nii.gz")) for f in case_files]

    # export kwargs from plans (same defaults as predict_cases)
    seg_export_params = trainer.plans.get(
        "segmentation_export_params",
        {"force_separate_z": None, "interpolation_order": 1, "interpolation_order_z": 0},
    )
    interpolation_order = seg_export_params["interpolation_order"]
    interpolation_order_z = seg_export_params["interpolation_order_z"]
    force_separate_z = seg_export_params["force_separate_z"]

    print(f"[predict] {len(list_of_lists)} test cases; overlap={args.step_size}; TTA=off")
    n_done = 0
    for preprocessed in preprocess_multithreaded(
        trainer, list_of_lists, output_filenames, args.num_threads_preprocessing
    ):
        output_filename, (d, dct) = preprocessed
        if isinstance(d, str):
            data = np.load(d)
            Path(d).unlink()
            d = data
        print(f"[predict] {Path(output_filename).name}")
        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d,
            do_mirroring=False,           # TTA off (spec §5.4)
            mirror_axes=trainer.data_aug_params.get("mirror_axes", ()),
            use_sliding_window=True,
            step_size=args.step_size,     # overlap 0.5
            use_gaussian=True,
            all_in_gpu=False,
            mixed_precision=args.mixed_precision,
        )[1]

        transpose_backward = trainer.plans.get("transpose_backward")
        if transpose_backward is not None:
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

        save_segmentation_nifti_from_softmax(
            softmax, output_filename, dct,
            interpolation_order, None, None, None, None, None,
            force_separate_z, interpolation_order_z,
        )
        n_done += 1

    print(f"[predict] DONE: {n_done}/{len(list_of_lists)} predictions -> {output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
