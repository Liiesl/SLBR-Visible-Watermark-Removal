"""Model package.

The 2021-era SLBR models depend on APIs removed from modern scikit-image.
They are imported on a best-effort basis so dependency-light subpackages such
as ``src.models.slbr_next`` remain importable. Calling the legacy factories
after a failed import raises a clear error instead of failing package import.
"""

try:  # pragma: no cover - depends on the installed legacy dependency stack
    from .BasicModel import BasicModel
except ImportError:  # pragma: no cover
    BasicModel = None

try:  # pragma: no cover - depends on the installed legacy dependency stack
    from .SLBR import SLBR
except ImportError:  # pragma: no cover
    SLBR = None


def _require_legacy(name):
    raise ImportError(
        f"{name} is unavailable because the legacy SLBR dependency stack "
        "could not be imported (scikit-image/scipy API changes)."
    )


def basic(**kwargs):
    if BasicModel is None:
        _require_legacy("BasicModel")
    return BasicModel(**kwargs)


def slbr(**kwargs):
    if SLBR is None:
        _require_legacy("SLBR")
    return SLBR(**kwargs)
