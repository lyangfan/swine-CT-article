"""Vendored SegFormer3D-aniso (CVPR-W 2024) — architecture only.

Source: https://github.com/OSUPCVLab/SegFormer3D (MIT license, 1 file vendored).
The anisotropic fix (explicit D/H/W threading, removal of ``@torch.jit.script
cube_root``) is applied so non-cubic patches like [64,160,160] work. Attention /
PatchEmbedding convs / SR convs / all-MLP decoder are UNCHANGED vs upstream.
"""
from .segformer3d_aniso import SegFormer3D  # noqa: F401
