"""MedNeXt-S plugin (vendored architecture, MICCAI 2023).

Spec §6.3:
  num_input_channels=1, num_classes=10
  model_id="S"  → n_channels=32, block_counts=[2,2,2,2,2,2,2,2,2]
  kernel_size=3, exp_r=4, norm_type="group"
  deep_supervision=True, do_res=True, do_res_up_down=True (paper defaults)

MedNeXt's decoder has 5 resolution levels → 5 DS outputs (isotropic ÷2 per
level). This is INDEPENDENT of the Task601 plan's 6-level anisotropic pooling,
so we override deep_supervision_scales to 5 isotropic entries. DS loss weights
still use the nnU-Net v1 formula (1/2**i, drop lowest, renormalise). CNN family
→ SGD + poly (MedNeXt paper uses SGD).
"""
import torch.nn as nn

from .MedNextV1 import MedNeXt
from ...registry import NetworkSpec, register

# MedNeXt-S: 4 downsample stages → bottleneck at ÷16 → 5 DS output levels
# (full, ÷2, ÷4, ÷8, ÷16). Isotropic because MedNeXt pools all axes uniformly.
_MEDNEXT_DS_SCALES = [
    [1, 1, 1],
    [0.5, 0.5, 0.5],
    [0.25, 0.25, 0.25],
    [0.125, 0.125, 0.125],
    [0.0625, 0.0625, 0.0625],
]


def build(trainer):
    return MedNeXt(
        in_channels=trainer.num_input_channels,
        n_channels=32,
        n_classes=trainer.num_classes,
        exp_r=4,
        kernel_size=3,
        deep_supervision=True,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        norm_type="group",
        dim="3d",
    )


register(NetworkSpec(
    name="mednext_s",
    build_fn=build,
    forward_protocol="deep_supervision",
    family="cnn",
    conv_op=nn.Conv3d,
    num_classes=10,
    needs_wrap=True,
    input_shape_must_be_divisible_by=(16, 16, 16),
    ds_scales_override=_MEDNEXT_DS_SCALES,
    architecture="MedNeXt-S (n_channels=32, k=3, exp_r=4, 4 downsample stages, DS=5)",
))


def build_l(trainer):
    """MedNeXt-L — verbatim upstream create_mednextv1_large config."""
    return MedNeXt(
        in_channels=trainer.num_input_channels,
        n_channels=32,
        n_classes=trainer.num_classes,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        kernel_size=3,
        deep_supervision=True,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        norm_type="group",
        dim="3d",
        checkpoint_style="outside_block",
    )


register(NetworkSpec(
    name="mednext_l",
    build_fn=build_l,
    forward_protocol="deep_supervision",
    family="cnn",
    conv_op=nn.Conv3d,
    num_classes=10,
    needs_wrap=True,
    input_shape_must_be_divisible_by=(16, 16, 16),
    ds_scales_override=_MEDNEXT_DS_SCALES,
    architecture="MedNeXt-L (n_channels=32, k=3, exp_r=[3,4,8,8,8,8,8,4,3], 45 blocks, checkpoint=outside_block, DS=5)",
))
