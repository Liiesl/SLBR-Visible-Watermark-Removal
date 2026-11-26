"""Import-isolation checks for the dependency-light SLBR-Next package.

``src/__init__.py`` used to import ``src.models`` eagerly, which pulls the
2021-era scikit-image/SciPy APIs and made every ``src.*`` import fail on a
modern stack. The new package must import without touching those modules.
"""

import sys

from torch import nn


def test_new_package_imports_without_legacy_models():
    from src.networks import slbr_next

    assert "src.models" not in sys.modules
    assert slbr_next.__all__ == [
        "LayerNorm2d",
        "SimpleGate",
        "SCA",
        "NAFConvNeXtBlock",
    ]


def test_public_blocks_are_modules():
    from src.networks.slbr_next import (
        LayerNorm2d,
        NAFConvNeXtBlock,
        SCA,
        SimpleGate,
    )

    for block in (LayerNorm2d, SimpleGate, SCA, NAFConvNeXtBlock):
        assert issubclass(block, nn.Module)
