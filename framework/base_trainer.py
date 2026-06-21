"""Multi-network base trainer for the v1 input-consistency comparison.

Subclasses nnU-Net v1 ``nnUNetTrainerV2`` and locks the v1 data pipeline
(moreDA augmentation + force-foreground sampling), determinism patches, and
gradient-accumulation for ALL networks. The only network-dependent logic is
delegated to a :class:`~framework.registry.NetworkSpec`:

  - how to build the network (:meth:`initialize_network`)
  - whether the forward returns a deep-supervision list or a single tensor
    (:meth:`run_iteration`, :meth:`run_online_evaluation`)
  - which optimizer family to use (:meth:`initialize_optimizer_and_scheduler`,
    :meth:`maybe_update_lr`)

External networks (MONAI SwinUNETR, vendored MedNeXt / SegFormer3D) are wrapped in
:class:`GenericWrap` so nnU-Net's SegmentationNetwork interface (sliding-window
prediction, ``do_ds`` toggling) works unchanged.

Fairness contract (spec §5): identical data/split/patch/augmentation/sampling,
identical training budget (500 ep × 250 iters = 125 000), identical DC_and_CE
loss family, identical sliding-window prediction (overlap 0.5, TTA off), final
checkpoint only.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.data_augmentation.data_augmentation_moreDA import (
    get_moreDA_augmentation,
)
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

from .registry import NetworkSpec


# ---------------------------------------------------------------------------
# budget (spec §5.2): 500 ep × 250 iters = 125 000
# ---------------------------------------------------------------------------
MAX_NUM_EPOCHS = 500
NUM_BATCHES_PER_EPOCH = 250
NUM_VAL_BATCHES_PER_EPOCH = 50  # nnU-Net v1 default val frequency

# optimizer hyper-params, pinned by spec §6
SGD_LR = 1e-2
SGD_MOMENTUM = 0.99
SGD_WD = 3e-4
ADAMW_LR = 4e-4
ADAMW_WD = 1e-5
# warmup-cosine (transformer family, spec §6.2/§6.4)
WARMUP_RATIO = 0.05
MIN_LR_RATIO = 0.01
MIN_LR_FLOOR = 1e-6


class GenericWrap(SegmentationNetwork):
    """Adapter wrapping an arbitrary segmentation network into nnU-Net's
    SegmentationNetwork interface.

    Provides ``conv_op``, ``num_classes``, ``input_shape_must_be_divisible_by``,
    ``inference_apply_nonlin`` and a ``do_ds`` property that propagates to the
    wrapped net when it supports deep supervision. Inherits sliding-window
    prediction (``predict_3D``, ``predict_sliding_window_return_logits``).
    """

    def __init__(self, net, conv_op, num_classes, input_shape_must_be_divisible_by, do_ds):
        super().__init__()
        self.net = net
        self.conv_op = conv_op
        self.num_classes = num_classes
        self.input_shape_must_be_divisible_by = input_shape_must_be_divisible_by
        self.inference_apply_nonlin = softmax_helper
        self._do_ds = do_ds
        if hasattr(net, "do_ds"):
            net.do_ds = do_ds

    @property
    def do_ds(self):
        return self._do_ds

    @do_ds.setter
    def do_ds(self, value):
        self._do_ds = value
        # propagate to the wrapped architecture (MedNeXt / Generic_UNet-style)
        if hasattr(self.net, "do_ds"):
            self.net.do_ds = value

    def forward(self, x):
        return self.net(x)


class MultiNetworkTrainer(nnUNetTrainerV2):
    """Network-agnostic nnU-Net v1 trainer.

    Extra keyword-only args vs the parent:
      network_spec     : NetworkSpec for the architecture being trained
      base_seed        : one of the 3 experiment seeds (20260520 / 21 / 22)
      grad_accum_steps : physical-batch multiplier to reach effective batch 2
                         (default 1 since the plan batch_size is already 2)
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
        *,
        network_spec: NetworkSpec,
        base_seed: int,
        grad_accum_steps: int = 1,
    ):
        super().__init__(
            plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
            unpack_data, deterministic, fp16,
        )
        self.network_spec = network_spec
        self.base_seed = int(base_seed)
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        # budget
        self.max_num_epochs = MAX_NUM_EPOCHS
        self.num_batches_per_epoch = NUM_BATCHES_PER_EPOCH
        self.num_val_batches_per_epoch = NUM_VAL_BATCHES_PER_EPOCH

        # checkpoint policy (spec Q16): final only — keep best too (harmless), drop intermediates
        self.save_every = 10 ** 9
        self.save_latest_only = True
        self.save_intermediate_checkpoints = False

        # initial LR depends on family (spec §6)
        self.initial_lr = SGD_LR if self.network_spec.family == "cnn" else ADAMW_LR
        self.weight_decay = SGD_WD if self.network_spec.family == "cnn" else ADAMW_WD

        # step-based LR bookkeeping (transformer warmup-cosine)
        self._global_step = 0
        self._total_steps = MAX_NUM_EPOCHS * NUM_BATCHES_PER_EPOCH

        # grad-accum counter
        self._accum_counter = 0

        # picklable init args for save_checkpoint (network_spec → name string)
        self.init_args = (
            plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
            unpack_data, deterministic, fp16,
            network_spec.name, int(base_seed), int(grad_accum_steps),
        )

    # ------------------------------------------------------------------ #
    # deep-supervision scales / loss weights
    # ------------------------------------------------------------------ #
    def _compute_ds_loss_weights(self, n_outputs: int) -> np.ndarray:
        """nnU-Net v1 default DS weights: 1/2**i, drop the lowest-res output, renormalise."""
        weights = np.array([1 / (2 ** i) for i in range(n_outputs)])
        mask = np.array(
            [True] + [True if i < n_outputs - 1 else False for i in range(1, n_outputs)]
        )
        weights[~mask] = 0
        return weights / weights.sum()

    def _resolve_deep_supervision_scales(self):
        """Return the DS scales for this network (None for single-output)."""
        if self.network_spec.forward_protocol == "single_output":
            return None
        if self.network_spec.ds_scales_override is not None:
            return self.network_spec.ds_scales_override
        # plan-derived (Generic_UNet): set by setup_DA_params -> self.deep_supervision_scales
        return self.deep_supervision_scales

    # ------------------------------------------------------------------ #
    # initialize — same v1 pipeline, DS scales/loss branched on protocol
    # ------------------------------------------------------------------ #
    def initialize(self, training=True, force_load_plans=False):
        if self.was_initialized:
            self.print_to_log_file("self.was_initialized is True, not running self.initialize again")
            self.was_initialized = True
            return

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()
        self.process_plans(self.plans)
        self.setup_DA_params()  # sets self.deep_supervision_scales from plans

        # --- resolve DS scales + loss for THIS network ---
        ds_scales = self._resolve_deep_supervision_scales()
        # overwrite self.deep_supervision_scales so get_moreDA_augmentation sees the right value
        self.deep_supervision_scales = ds_scales

        if self.network_spec.forward_protocol == "deep_supervision":
            n_outputs = len(ds_scales)
            self.ds_loss_weights = self._compute_ds_loss_weights(n_outputs)
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
        else:
            # single-output: keep self.loss = DC_and_CE_loss (set by grandparent __init__)
            self.ds_loss_weights = None
            self.print_to_log_file(
                f"[{self.network_spec.name}] single-output protocol: "
                "DC_and_CE_loss without MultipleOutputLoss2 wrapping"
            )

        self.folder_with_preprocessed_data = join(
            self.dataset_directory,
            self.plans["data_identifier"] + "_stage%d" % self.stage,
        )

        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()  # load_dataset + do_split
            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file("INFO: Not unpacking data — training may be slow.")

            self.tr_gen, self.val_gen = get_moreDA_augmentation(
                self.dl_tr,
                self.dl_val,
                self.data_aug_params["patch_size_for_spatialtransform"],
                self.data_aug_params,
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                use_nondetMultiThreadedAugmenter=False,
            )
            self.print_to_log_file(
                "TRAINING KEYS:\n %s" % str(self.dataset_tr.keys()),
                also_print_to_console=False,
            )
            self.print_to_log_file(
                "VALIDATION KEYS:\n %s" % str(self.dataset_val.keys()),
                also_print_to_console=False,
            )

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()

        assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel)), (
            f"network must be a SegmentationNetwork after initialize_network, got {type(self.network)}"
        )
        self.was_initialized = True

    # ------------------------------------------------------------------ #
    # initialize_network — build from spec, wrap external nets in GenericWrap
    # ------------------------------------------------------------------ #
    def initialize_network(self):
        net = self.network_spec.build_fn(self)
        if isinstance(net, SegmentationNetwork):
            # Generic_UNet is already a SegmentationNetwork
            self.network = net
        else:
            do_ds = self.network_spec.forward_protocol == "deep_supervision"
            self.network = GenericWrap(
                net,
                conv_op=self.network_spec.conv_op,
                num_classes=self.num_classes,
                input_shape_must_be_divisible_by=self.network_spec.input_shape_must_be_divisible_by,
                do_ds=do_ds,
            )
        if torch.cuda.is_available():
            self.network.cuda()
        # inference_apply_nonlin for sliding-window prediction
        self.network.inference_apply_nonlin = softmax_helper

    # ------------------------------------------------------------------ #
    # optimizer + scheduler — family-specific (spec §6)
    # ------------------------------------------------------------------ #
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        if self.network_spec.family == "cnn":
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                self.initial_lr,  # 1e-2
                weight_decay=self.weight_decay,  # 3e-4
                momentum=SGD_MOMENTUM,
                nesterov=True,
            )
        else:  # transformer
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(),
                self.initial_lr,  # 4e-4
                weight_decay=self.weight_decay,  # 1e-5
            )
        self.lr_scheduler = None  # we schedule manually (poly or warmup_cosine)

    # ------------------------------------------------------------------ #
    # LR schedule
    # ------------------------------------------------------------------ #
    def _warmup_cosine_lr(self, step: int) -> float:
        total = max(1, self._total_steps)
        warmup = max(1, int(WARMUP_RATIO * total))
        min_lr = max(self.initial_lr * MIN_LR_RATIO, MIN_LR_FLOOR)
        if step < warmup:
            # linear warmup from min_lr to initial_lr
            return min_lr + (self.initial_lr - min_lr) * (step / warmup)
        # cosine decay from initial_lr to min_lr over the remaining steps
        import math
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        return min_lr + 0.5 * (self.initial_lr - min_lr) * (1 + math.cos(math.pi * progress))

    def maybe_update_lr(self, epoch=None):
        """CNN: poly per-epoch (nnU-Net v1 default). Transformer: warmup-cosine per-step."""
        if self.network_spec.family == "cnn":
            from nnunet.training.learning_rate.poly_lr import poly_lr
            ep = self.epoch + 1 if epoch is None else epoch
            self.optimizer.param_groups[0]["lr"] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
            self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]["lr"], decimals=6))
        else:
            lr = self._warmup_cosine_lr(self._global_step)
            self.optimizer.param_groups[0]["lr"] = lr

    def _step_lr_update(self):
        """Per-iteration LR update for transformer family (warmup-cosine is step-based)."""
        if self.network_spec.family == "transformer":
            self.optimizer.param_groups[0]["lr"] = self._warmup_cosine_lr(self._global_step)

    # ------------------------------------------------------------------ #
    # run_iteration — grad-accum + AMP + DS/single forward branching
    # ------------------------------------------------------------------ #
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["target"]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        do_accum = do_backprop and self.grad_accum_steps > 1
        if do_accum and self._accum_counter == 0:
            self.optimizer.zero_grad()
        if do_backprop and not do_accum:
            self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)
            if do_backprop:
                scale = 1.0 / self.grad_accum_steps if do_accum else 1.0
                self.amp_grad_scaler.scale(l * scale).backward()
                if not do_accum:
                    self.amp_grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)
            if do_backprop:
                scale = 1.0 / self.grad_accum_steps if do_accum else 1.0
                (l * scale).backward()
                if not do_accum:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                    self.optimizer.step()

        if do_accum:
            self._accum_counter += 1
            if self._accum_counter >= self.grad_accum_steps:
                if self.fp16:
                    self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                if self.fp16:
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
                else:
                    self.optimizer.step()
                self._accum_counter = 0

        # step-based LR (transformer) advances once per optimizer step-equivalent
        if do_backprop:
            self._global_step += 1
            self._step_lr_update()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return l.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # online evaluation — DS list vs single tensor
    # ------------------------------------------------------------------ #
    def run_online_evaluation(self, output, target):
        if self.network_spec.forward_protocol == "deep_supervision":
            # nnUNetTrainerV2.run_online_evaluation already extracts output[0] /
            # target[0] from the DS lists before calling the grandparent, so just
            # delegate to it.
            return super().run_online_evaluation(output, target)
        # single-output: output / target are already single tensors — skip the
        # parent's [0] indexing (which would grab the batch dim) and call the
        # grandparent (nnUNetTrainer.run_online_evaluation) directly.
        from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
        return nnUNetTrainer.run_online_evaluation(self, output, target)

    # ------------------------------------------------------------------ #
    # on_epoch_end — skip the nnU-Net-specific momentum/He reset (only valid
    # for Generic_UNet + SGD; would break external architectures / AdamW)
    # ------------------------------------------------------------------ #
    def on_epoch_end(self):
        # Call the training base (NetworkTrainer) for checkpoint / eval / LR
        # bookkeeping, but DROP nnUNetTrainerV2's He-reinit-on-zero-dice guard
        # (InitWeights_He is only correct for Generic_UNet + SGD; it would silently
        # corrupt external architectures and is irrelevant for AdamW). Match
        # nnUNetTrainerV2's explicit return so patience-based early stopping never
        # fires — the budget is fixed at exactly max_num_epochs.
        from nnunet.training.network_training.network_trainer import NetworkTrainer
        NetworkTrainer.on_epoch_end(self)
        return self.epoch < self.max_num_epochs
