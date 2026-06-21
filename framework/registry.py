"""Network registry for the v1 input-consistency comparison.

Each registered network declares:
  - build_fn(trainer) -> nn.Module  : construct the architecture from trainer plans
  - forward_protocol                : "deep_supervision" (list output) | "single_output"
  - family                          : "cnn" (SGD+poly) | "transformer" (AdamW+warmup_cosine)
  - conv_op                         : nn.Conv3d / nn.Conv2d (for the SegmentationNetwork adapter)
  - needs_wrap                      : True if build_fn returns a bare nn.Module (adapter needed);
                                      False if it already subclasses SegmentationNetwork (Generic_UNet)
  - ds_scales_override              : for DS networks whose decoder depth != plan pool depth
                                      (e.g. MedNeXt 5-level isotropic), override the plan-derived scales

Adding a network = add a plugin in framework/nets/ + a configs/*.yaml. Nothing here changes.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Type

import torch.nn as nn


@dataclass
class NetworkSpec:
    name: str
    build_fn: Callable
    forward_protocol: str  # "deep_supervision" | "single_output"
    family: str            # "cnn" | "transformer"
    conv_op: Type[nn.Module]
    num_classes: int = 10
    needs_wrap: bool = True
    input_shape_must_be_divisible_by: Optional[tuple] = None
    # For DS networks whose native decoder depth differs from the plan pool depth.
    # When set, the base trainer uses these scales instead of the plan-derived ones.
    ds_scales_override: Optional[List[list]] = None
    # Architecture label recorded in env/audit JSON (not used for logic).
    architecture: str = ""


_NETWORKS: dict = {}


def register(spec: "NetworkSpec") -> "NetworkSpec":
    if spec.name in _NETWORKS:
        raise ValueError(f"network '{spec.name}' already registered")
    _NETWORKS[spec.name] = spec
    return spec


def get_network_spec(name: str) -> "NetworkSpec":
    if name not in _NETWORKS:
        raise KeyError(
            f"network '{name}' not registered; known: {sorted(_NETWORKS)}"
        )
    return _NETWORKS[name]


def all_networks() -> dict:
    return dict(_NETWORKS)
