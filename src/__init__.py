"""SLBR package.

Subpackages are resolved lazily instead of being imported eagerly. The legacy
``src.models``/``src.utils`` modules depend on APIs that were removed from
modern scikit-image/SciPy; importing them here made every ``src.*`` import
fail, including the dependency-light ``src.networks.slbr_next`` package.

Explicit imports (``import src.models``, ``from src.utils.misc import ...``)
continue to resolve exactly as before.
"""

from __future__ import absolute_import

import importlib

_SUBMODULES = ("networks", "utils", "models")


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
