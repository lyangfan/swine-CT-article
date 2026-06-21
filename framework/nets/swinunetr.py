"""MONAI SwinUNETR (V2) plugin.

Spec §6.2: feature_size=48, depths=[2,2,2,2], num_heads=[3,6,12,24],
window_size=7, norm_name="instance", use_v2=True, all drop rates 0.
Single output (no DS heads in the base SwinUNETR forward) → forward_protocol
"single_output". Transformer family → AdamW + warmup_cosine.
"""
import torch.nn as nn

from ..registry import NetworkSpec, register

WINDOW_SIZE = 7
# SwinUNETR downsamples ×4 (÷2 each encoder stage): input must be divisible by 16
INPUT_DIVISIBLE_BY = (2 ** 4, 2 ** 4, 2 ** 4)


def build(trainer):
    from monai.networks.nets import SwinUNETR

    return SwinUNETR(
        in_channels=trainer.num_input_channels,
        out_channels=trainer.num_classes,
        spatial_dims=3,
        feature_size=48,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=WINDOW_SIZE,
        norm_name="instance",
        use_v2=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
    )


register(NetworkSpec(
    name="swinunetr",
    build_fn=build,
    forward_protocol="single_output",
    family="transformer",
    conv_op=nn.Conv3d,
    num_classes=10,
    needs_wrap=True,
    input_shape_must_be_divisible_by=INPUT_DIVISIBLE_BY,
    architecture="MONAI SwinUNETR-V2 (feature_size=48, window=7, 4 stages)",
))
