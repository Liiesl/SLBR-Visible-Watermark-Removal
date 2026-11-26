"""Network implementations.

The 2021-era ``methods`` and ``discriminator`` modules are imported on a
best-effort basis: they depend on APIs removed from modern scikit-image/SciPy.
Guarding them keeps dependency-light subpackages such as
``src.networks.slbr_next`` importable, while environments that still have the
legacy dependency stack keep the historical star exports.
"""

try:  # pragma: no cover - depends on the installed legacy dependency stack
    from .methods import *
except ImportError:
    pass

try:  # pragma: no cover - depends on the installed legacy dependency stack
    from .discriminator import *
except ImportError:
    pass
