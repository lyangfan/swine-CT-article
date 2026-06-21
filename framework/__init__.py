"""v1 input-consistency multi-network framework.

Importing :mod:`framework` registers every network plugin in
:mod:`framework.nets` so ``registry.get_network_spec(name)`` resolves them.
"""
from .registry import NetworkSpec, register, get_network_spec, all_networks  # noqa: F401
from . import nets  # noqa: F401  (side-effect: registers all plugins)
