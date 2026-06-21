"""nnU-Net v1 (Generic_UNet) plugin.

Architecture is fully auto-configured from the Task601 plans (base_features,
num_pool, conv_kernels, pool_kernels). Generic_UNet already subclasses
SegmentationNetwork, so no GenericWrap is needed. Native multi-scale deep
supervision (forward returns a list). CNN family → SGD + poly.
"""
import torch.nn as nn
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper

from ..registry import NetworkSpec, register


def build(trainer):
    if trainer.threeD:
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
    else:
        conv_op = nn.Conv2d
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d

    norm_op_kwargs = {"eps": 1e-5, "affine": True}
    dropout_op_kwargs = {"p": 0, "inplace": True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}

    net = Generic_UNet(
        trainer.num_input_channels,
        trainer.base_num_features,
        trainer.num_classes,
        len(trainer.net_num_pool_op_kernel_sizes),
        trainer.conv_per_stage,
        2,
        conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
        net_nonlin, net_nonlin_kwargs,
        True, False, lambda x: x, InitWeights_He(1e-2),
        trainer.net_num_pool_op_kernel_sizes,
        trainer.net_conv_kernel_sizes,
        False, True, True,
    )
    net.inference_apply_nonlin = softmax_helper
    return net  # already a SegmentationNetwork


register(NetworkSpec(
    name="nnunet_v1",
    build_fn=build,
    forward_protocol="deep_supervision",
    family="cnn",
    conv_op=nn.Conv3d,
    num_classes=10,
    needs_wrap=False,  # Generic_UNet IS a SegmentationNetwork
    architecture="Generic_UNet (nnU-Net v1 3d_fullres, from Task601 plans)",
))
