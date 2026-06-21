"""Vendored MedNeXt (MICCAI 2023) — architecture only.

Source: https://github.com/MIC-DIKU/MedNeXt (MIT license, vendored 3 files).
Only import path adjusted: ``MedNextV1.py`` uses ``from .blocks import *``
instead of the upstream absolute ``nnunet_mednext...`` path. Architecture code
is UNCHANGED.

Files:
  - blocks.py      : MedNeXtBlock / Down / Up / conv primitives
  - MedNextV1.py   : top-level MedNeXt encoder-decoder + deep supervision
  - plugin.py      : NetworkSpec registration for MedNeXt-S
"""
from .MedNextV1 import MedNeXt  # noqa: F401
