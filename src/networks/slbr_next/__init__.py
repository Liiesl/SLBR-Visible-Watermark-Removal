"""SLBR-Next: blocks, stages, and the assembled v1 model.

The top-level :class:`SLBRNextV1` implements the v1 contract documented in
``src/networks/slbr_next/model.py``. The stage classes remain public so tests,
distillation hooks and future versioned contracts can reuse them.
"""

from .bg_decoder import (
    BackgroundDecoder,
    BackgroundDecoderOutput,
    FiLMConditioning,
    ResampleBlock,
)
from .blocks import LayerNorm2d, NAFConvNeXtBlock, SCA, SimpleGate
from .bottleneck import (
    MDTA,
    BottleneckOutput,
    DisentangledTrunk,
    GatedHighFrequency,
    GatedSkipFusion,
    HaarFrequencyBranch,
    SLBRNextBottleneck,
)
from .encoder import EncoderOutput, HaarDownsample, SLBRNextEncoder
from .haar import HaarDWT, HaarIWT, merge_haar_bands, split_haar_bands
from .mask_decoder import (
    MaskDecoder,
    MaskDecoderOutput,
    MaskedCrossAttention,
    PartScoreHead,
)
from .model import (
    PAD_MULTIPLE,
    SLBRNextV1,
    SLBRNextV1Config,
    pad_to_multiple,
)
from .refinement import BackgroundOutput, RefinementUNet, composite_v1

__all__ = [
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
