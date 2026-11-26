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
        "HaarDWT",
        "HaarIWT",
        "split_haar_bands",
        "merge_haar_bands",
        "EncoderOutput",
        "HaarDownsample",
        "SLBRNextEncoder",
        "MDTA",
        "GatedSkipFusion",
        "GatedHighFrequency",
        "HaarFrequencyBranch",
        "DisentangledTrunk",
        "BottleneckOutput",
        "SLBRNextBottleneck",
        "MaskDecoderOutput",
        "MaskedCrossAttention",
        "PartScoreHead",
        "MaskDecoder",
        "FiLMConditioning",
        "ResampleBlock",
        "BackgroundDecoderOutput",
        "BackgroundDecoder",
        "BackgroundOutput",
        "RefinementUNet",
        "composite_v1",
        "PAD_MULTIPLE",
        "pad_to_multiple",
        "SLBRNextV1Config",
        "SLBRNextV1",
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


def test_top_level_model_is_public():
    from src.networks.slbr_next import SLBRNextV1, SLBRNextV1Config

    assert issubclass(SLBRNextV1, nn.Module)
    assert SLBRNextV1Config().convnext_pretrained is True
