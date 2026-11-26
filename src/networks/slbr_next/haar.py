"""Orthonormal Haar DWT/IWT for SLBR-Next (spec ``slbr-next.md`` §3.3).

The analysis transform is a grouped ``Conv2d`` (stride 2) with four fixed
2x2 filters per input channel; the synthesis transform is the matching grouped
``ConvTranspose2d``. The filters are registered buffers, never parameters:

.. code-block:: text

    LL = 0.5 * [[ 1,  1], [ 1,  1]]
    LH = 0.5 * [[ 1, -1], [ 1, -1]]
    HL = 0.5 * [[ 1,  1], [-1, -1]]
    HH = 0.5 * [[ 1, -1], [-1,  1]]

All four filters are orthonormal, so ``HaarIWT(HaarDWT(x)) == x`` for even
spatial sizes (FP32 maximum error is well below ``1e-5``). The transforms are
ONNX-native convolutions and require no FFT.

Channel order
-------------
A grouped convolution interleaves the four bands **per input channel**::

    [LL_0, LH_0, HL_0, HH_0, LL_1, LH_1, HL_1, HH_1, ...]

Do not split with ``chunk(4, dim=1)``; use :func:`split_haar_bands` and
:func:`merge_haar_bands`, which implement the interleaved layout exactly.

The module is stateless apart from the fixed filters, so it can be shared by
the encoder stem and the H/8 frequency branch (A3) without any trainable copy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "HaarDWT",
    "HaarIWT",
    "split_haar_bands",
    "merge_haar_bands",
]


def _haar_filters(channels: int) -> Tensor:
    """Build the fixed depthwise filter bank ``[4 * channels, 1, 2, 2]``."""
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")

    filters = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],    # LL
            [[0.5, -0.5], [0.5, -0.5]],  # LH
            [[0.5, 0.5], [-0.5, -0.5]],  # HL
            [[0.5, -0.5], [-0.5, 0.5]],  # HH
        ],
        dtype=torch.float32,
    )
    # One filter quad per input channel, in interleaved [LL, LH, HL, HH] order.
    return filters.unsqueeze(1).repeat(channels, 1, 1, 1)


def _check_even(x: Tensor, name: str) -> None:
    if x.dim() != 4:
        raise ValueError(
            f"{name} expects a 4D [B, C, H, W] tensor, got shape {tuple(x.shape)}"
        )
    height, width = x.shape[2], x.shape[3]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            f"{name} requires even spatial sizes, got H={height}, W={width}. "
            "Pad to a multiple of 16 before the encoder (spec §1.1)."
        )


class HaarDWT(nn.Module):
    """Orthonormal 2D Haar analysis transform (grouped ``Conv2d``).

    Args:
        channels: Number of input channels ``C``. The output has ``4 * C``
            channels in the interleaved per-channel order
            ``[LL_0, LH_0, HL_0, HH_0, LL_1, ...]``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        # Buffers follow ``.to(dtype/device)`` but are never trained.
        self.register_buffer(
            "weight", _haar_filters(self.channels), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        _check_even(x, "HaarDWT")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"HaarDWT expected {self.channels} input channels, "
                f"got {x.shape[1]}"
            )

        weight = self.weight
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        return F.conv2d(x, weight, stride=2, groups=self.channels)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, stride=2, trainable=False"


class HaarIWT(nn.Module):
    """Orthonormal 2D Haar synthesis transform (grouped ``ConvTranspose2d``).

    Args:
        channels: Number of output channels ``C``; the input must carry
            ``4 * C`` interleaved coefficients in ``[LL, LH, HL, HH]`` order.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.register_buffer(
            "weight", _haar_filters(self.channels), persistent=False
        )

    def forward(self, coeffs: Tensor) -> Tensor:
        if coeffs.dim() != 4:
            raise ValueError(
                "HaarIWT expects a 4D [B, 4 * C, H, W] tensor, "
                f"got shape {tuple(coeffs.shape)}"
            )
        if coeffs.shape[1] != 4 * self.channels:
            raise ValueError(
                f"HaarIWT expected {4 * self.channels} coefficient channels, "
                f"got {coeffs.shape[1]}"
            )

        weight = self.weight
        if weight.dtype != coeffs.dtype:
            weight = weight.to(dtype=coeffs.dtype)
        return F.conv_transpose2d(
            coeffs, weight, stride=2, groups=self.channels
        )

    def extra_repr(self) -> str:
        return f"channels={self.channels}, stride=2, trainable=False"


def split_haar_bands(
    coeffs: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split interleaved DWT coefficients into ``(LL, LH, HL, HH)``.

    This is the correct alternative to ``chunk(4, dim=1)`` for the grouped
    convolution layout. Each returned tensor has ``C`` channels.

    Args:
        coeffs: ``[B, 4 * C, H, W]`` interleaved coefficient tensor.

    Returns:
        Four ``[B, C, H, W]`` tensors in ``LL, LH, HL, HH`` order.
    """
    if coeffs.dim() != 4:
        raise ValueError(
            "split_haar_bands expects a 4D tensor, "
            f"got shape {tuple(coeffs.shape)}"
        )
    if coeffs.shape[1] % 4 != 0:
        raise ValueError(
            "split_haar_bands expects 4 * C channels, "
            f"got {coeffs.shape[1]}"
        )

    batch, channels4, height, width = coeffs.shape
    reshaped = coeffs.reshape(batch, channels4 // 4, 4, height, width)
    ll, lh, hl, hh = reshaped.unbind(dim=2)
    return ll, lh, hl, hh


def merge_haar_bands(
    ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor
) -> Tensor:
    """Interleave ``(LL, LH, HL, HH)`` bands for :class:`HaarIWT`.

    Args:
        ll, lh, hl, hh: Four ``[B, C, H, W]`` tensors.

    Returns:
        ``[B, 4 * C, H, W]`` coefficients in the grouped-convolution layout.
    """
    if not (ll.dim() == lh.dim() == hl.dim() == hh.dim() == 4):
        raise ValueError("merge_haar_bands expects four 4D tensors")
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError(
            "merge_haar_bands expects four tensors with identical shapes, got "
            f"{tuple(ll.shape)}, {tuple(lh.shape)}, {tuple(hl.shape)}, "
            f"{tuple(hh.shape)}"
        )

    batch, channels, height, width = ll.shape
    stacked = torch.stack((ll, lh, hl, hh), dim=2)
    return stacked.reshape(batch, 4 * channels, height, width)
