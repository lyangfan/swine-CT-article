"""Network plugins. Importing this package registers every architecture in
:mod:`framework.registry` as a side effect."""
from . import nnunet       # noqa: F401  (registers nnunet_v1)
from . import swinunetr    # noqa: F401  (registers swinunetr)
from .mednext import plugin as _mednext_plugin       # noqa: F401  (registers mednext_s)
from .segformer3d import plugin as _segformer_plugin  # noqa: F401  (registers segformer3d)
