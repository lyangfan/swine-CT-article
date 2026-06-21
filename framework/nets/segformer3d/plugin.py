"""SegFormer3D-aniso plugin (vendored architecture, CVPR-W 2024).

Spec §6.4: all repo defaults — sr_ratios=[4,2,1,1], embed_dims=[32,64,160,256],
patch_kernel/stride/padding, num_heads=[1,2,5,8], depths=[2,2,2,2],
mlp_ratios=[4,4,4,4], decoder_head_embedding_dim=256, decoder_dropout=0.0.
Single output (SegFormer3D has no DS heads). Transformer family → AdamW +
warmup_cosine (same schedule as SwinUNETR).
"""
import torch.nn as nn

from .segformer3d_aniso import SegFormer3D
from ...registry import NetworkSpec, register

# First PatchEmbedding strides by (4,2,2): deepest encoder channel divisor.
INPUT_DIVISIBLE_BY = (4, 2, 2)


def build(trainer):
    return SegFormer3D(
        in_channels=trainer.num_input_channels,
        num_classes=trainer.num_classes,
        sr_ratios=[4, 2, 1, 1],
        embed_dims=[32, 64, 160, 256],
        patch_kernel_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        num_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        mlp_ratios=[4, 4, 4, 4],
        decoder_head_embedding_dim=256,
        decoder_dropout=0.0,
    )


register(NetworkSpec(
    name="segformer3d",
    build_fn=build,
    forward_protocol="single_output",
    family="transformer",
    conv_op=nn.Conv3d,
    num_classes=10,
    needs_wrap=True,
    input_shape_must_be_divisible_by=INPUT_DIVISIBLE_BY,
    architecture="SegFormer3D-aniso (embed_dims=[32,64,160,256], sr_ratios=[4,2,1,1])",
))
