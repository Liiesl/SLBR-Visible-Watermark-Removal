"""Background decoder for SLBR-Next (spec ``slbr-next.md`` §3.6, §3.8).

The decoder consumes the background trunk of
:class:`~.bottleneck.BottleneckOutput` (``bg_h8``/``bg_h16``), the encoder
skips at ``H/4``/``H/2``/``H`` and the mask decoder features. Mask
information enters through a FiLM/SPADE-style conditioning block built from
``cat([mask_feat, mask_prob])``; there is no InstanceNorm.

The ``H/8`` background feature is decoded with the usual
bilinear -> 3x3 depthwise -> 1x1 pattern (spec §3.6) and the full-resolution
output is a residual over the input::

    coarse_bg = clamp(input + head(h), 0, 1)

With the default widths and an input of ``[B, 3, H, W]``:

=============  =============================  ==========================
field          shape                          notes
=============  =============================  ==========================
``coarse_bg``  ``[B, 3, H, W]``               residual over ``input``
``skip_h4``    ``[B, 128, H/4, W/4]``        refinement skip
``skip_h2``    ``[B, 64, H/2, W/2]``         refinement skip
``skip_h``     ``[B, 32, H, W]``             refinement skip
=============  =============================  ==========================

``H`` and ``W`` must be multiples of 16 to match the encoder contract, and
``H``/``W`` may differ. Every interpolation targets ``size=<reference>.shape[2:]``
with ``align_corners=False``; no interpolation reverses the spatial axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import LayerNorm2d, NAFConvNeXtBlock, SimpleGate

__all__ = [
    "FiLMConditioning",
    "ResampleBlock",
    "BackgroundDecoderOutput",
    "BackgroundDecoder",
]


class FiLMConditioning(nn.Module):
    """FiLM/SPADE-style conditioning (spec §3.6).

    ::text

        scale, shift = conv(cat([mask_feat, mask_prob]))
        h = scale * LayerNorm2d(x) + shift
        x = x + SimpleGate(conv1x1(h))

    ``SimpleGate`` halves the ``2C`` gate channels back to ``C``. The update
    is residual, so a zero-initialized gate is the identity. There is no
    InstanceNorm anywhere in the block.

    Args:
        channels: Feature width of the conditioned tensor.
        cond_channels: Channels of the conditioning tensor, i.e.
            ``mask_feat_channels + mask_prob_channels``.
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        channels: int,
        cond_channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if cond_channels <= 0:
            raise ValueError(
                f"cond_channels must be positive, got {cond_channels}"
            )

        self.channels = int(channels)
        self.cond_channels = int(cond_channels)

        self.norm = LayerNorm2d(channels, eps=eps)
        self.film = nn.Conv2d(cond_channels, 2 * channels, kernel_size=1)
        self.gate = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.simple_gate = SimpleGate()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        if x.dim() != 4 or cond.dim() != 4:
            raise ValueError(
                "FiLMConditioning expects 4D tensors, got "
                f"{tuple(x.shape)} and {tuple(cond.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"FiLMConditioning expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )
        if cond.shape[1] != self.cond_channels:
            raise ValueError(
                f"FiLMConditioning expected {self.cond_channels} "
                f"conditioning channels, got {cond.shape[1]}"
            )
        if cond.shape[0] != x.shape[0]:
            raise ValueError(
                "FiLMConditioning batch mismatch: "
                f"{x.shape[0]} vs {cond.shape[0]}"
            )

        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(
                cond,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        scale, shift = self.film(cond).chunk(2, dim=1)
        y = scale * self.norm(x) + shift
        return x + self.simple_gate(self.gate(y))

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, cond_channels={self.cond_channels}"
        )


class ResampleBlock(nn.Module):
    """Bilinear resample -> 3x3 depthwise conv -> 1x1 conv (spec §3.6).

    The target size is always a reference tensor's ``shape[2:]``. This single
    block is used for both upsampling and downsampling in the decoder and the
    refinement U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )
        if out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive, got {out_channels}"
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, size: Sequence[int]) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "ResampleBlock expects a 4D [B, C, H, W] tensor, "
                f"got shape {tuple(x.shape)}"
            )
        size = tuple(int(value) for value in size)
        if len(size) != 2 or any(value <= 0 for value in size):
            raise ValueError(
                f"size must be two positive integers, got {size}"
            )
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return self.proj(self.dwconv(x))

    def extra_repr(self) -> str:
        return f"{self.in_channels} -> {self.out_channels}"


@dataclass(frozen=True)
class BackgroundDecoderOutput:
    """Frozen background decoder interface consumed by the refinement stage.

    ``coarse_bg`` is the full-resolution residual background. ``skip_h4``,
    ``skip_h2`` and ``skip_h`` are decoder features at ``H/4``, ``H/2`` and
    ``H``; they feed the Stage 2 refinement U-Net (spec §3.7).
    """

    coarse_bg: Tensor
    skip_h4: Tensor
    skip_h2: Tensor
    skip_h: Tensor


class _DecoderLevel(nn.Module):
    """One decoder level: resample, encoder-skip fusion, NAF block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        *,
        blocks: int,
        dw_kernel_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.resample = ResampleBlock(in_channels, out_channels)
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, 1)
        self.blocks = nn.Sequential(
            *[
                NAFConvNeXtBlock(
                    out_channels,
                    dw_kernel_size=dw_kernel_size,
                    eps=eps,
                )
                for _ in range(blocks)
            ]
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.resample(x, size=skip.shape[2:])
        return self.blocks(x + self.skip_proj(skip))

    def extra_repr(self) -> str:
        return f"skip_channels={self.skip_proj.in_channels}"


class BackgroundDecoder(nn.Module):
    """MBE v2 background decoder (spec §3.6, §3.8).

    Args:
        bg_h8_channels: Channels of ``BottleneckOutput.bg_h8`` (256).
        bg_h16_channels: Channels of ``BottleneckOutput.bg_h16`` (512).
        mask_feat_channels: Channels of the mask decoder feature that
            conditions the background (256 at H/8 in the reference design).
        mask_prob_channels: Channels of the mask probability (1).
        width_h8: Decoder width after fusing ``bg_h8`` and ``bg_h16`` (256).
        widths: Widths at ``(H/4, H/2, H)``; defaults to (128, 64, 32) from
            the reference shape flow. The last width feeds the 3-channel head.
        skip_h4_channels: Encoder ``x_h4`` channels (128).
        skip_h2_channels: Encoder ``x_h2`` channels (64).
        skip_h_channels: Encoder ``x_h`` channels (32).
        blocks_per_level: NAF blocks per decoder level.
        dw_kernel_sizes: Depthwise kernels at ``(H/4, H/2, H)``. The default
            uses 5x5 below full resolution and 3x3 at ``H`` (spec §5.4).
        norm_eps: LayerNorm epsilon.
        clamp_output: Clamp ``coarse_bg`` to ``[0, 1]``. Keep enabled for RGB
            inputs in ``[0, 1]``.
    """

    def __init__(
        self,
        *,
        bg_h8_channels: int = 256,
        bg_h16_channels: int = 512,
        mask_feat_channels: int = 256,
        mask_prob_channels: int = 1,
        width_h8: int = 256,
        widths: Sequence[int] = (128, 64, 32),
        skip_h4_channels: int = 128,
        skip_h2_channels: int = 64,
        skip_h_channels: int = 32,
        blocks_per_level: int = 1,
        dw_kernel_sizes: Sequence[int] = (5, 5, 3),
        norm_eps: float = 1e-6,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()

        for name, value in (
            ("bg_h8_channels", bg_h8_channels),
            ("bg_h16_channels", bg_h16_channels),
            ("mask_feat_channels", mask_feat_channels),
            ("mask_prob_channels", mask_prob_channels),
            ("width_h8", width_h8),
            ("skip_h4_channels", skip_h4_channels),
            ("skip_h2_channels", skip_h2_channels),
            ("skip_h_channels", skip_h_channels),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        widths = tuple(int(width) for width in widths)
        if len(widths) != 3:
            raise ValueError(
                "widths must contain three entries for (H/4, H/2, H), "
                f"got {widths}"
            )
        if any(width <= 0 for width in widths):
            raise ValueError(f"all widths must be positive, got {widths}")

        kernels = tuple(int(kernel) for kernel in dw_kernel_sizes)
        if len(kernels) != 3:
            raise ValueError(
                "dw_kernel_sizes must contain three entries for "
                f"(H/4, H/2, H), got {kernels}"
            )
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError(
                "dw_kernel_sizes must be positive odd integers, "
                f"got {kernels}"
            )
        if blocks_per_level < 1:
            raise ValueError(
                "blocks_per_level must be at least 1, "
                f"got {blocks_per_level}"
            )

        self.bg_h8_channels = int(bg_h8_channels)
        self.bg_h16_channels = int(bg_h16_channels)
        self.mask_feat_channels = int(mask_feat_channels)
        self.mask_prob_channels = int(mask_prob_channels)
        self.width_h8 = int(width_h8)
        self.widths = widths
        self.skip_h4_channels = int(skip_h4_channels)
        self.skip_h2_channels = int(skip_h2_channels)
        self.skip_h_channels = int(skip_h_channels)
        self.blocks_per_level = int(blocks_per_level)
        self.dw_kernel_sizes = kernels
        self.norm_eps = float(norm_eps)
        self.clamp_output = bool(clamp_output)

        # H/16 context is merged into H/8 before conditioning.
        self.bg8_proj = nn.Conv2d(bg_h8_channels, width_h8, 1)
        self.bg16_proj = nn.Conv2d(bg_h16_channels, width_h8, 1)

        # FiLM/SPADE conditioning from [mask_feat, mask_prob].
        self.film = FiLMConditioning(
            width_h8,
            mask_feat_channels + mask_prob_channels,
            eps=norm_eps,
        )

        # H/8 -> H/4 -> H/2 -> H, each fused with the encoder skip.
        self.levels = nn.ModuleList(
            [
                _DecoderLevel(
                    in_channels=width_h8,
                    out_channels=widths[0],
                    skip_channels=skip_h4_channels,
                    blocks=blocks_per_level,
                    dw_kernel_size=kernels[0],
                    eps=norm_eps,
                ),
                _DecoderLevel(
                    in_channels=widths[0],
                    out_channels=widths[1],
                    skip_channels=skip_h2_channels,
                    blocks=blocks_per_level,
                    dw_kernel_size=kernels[1],
                    eps=norm_eps,
                ),
                _DecoderLevel(
                    in_channels=widths[1],
                    out_channels=widths[2],
                    skip_channels=skip_h_channels,
                    blocks=blocks_per_level,
                    dw_kernel_size=kernels[2],
                    eps=norm_eps,
                ),
            ]
        )
        self.head = nn.Conv2d(widths[2], 3, kernel_size=1)

    def forward(
        self,
        x: Tensor,
        bg_h8: Tensor,
        bg_h16: Tensor,
        mask_feat: Tensor,
        mask_prob: Tensor,
        *,
        skip_h4: Tensor,
        skip_h2: Tensor,
        skip_h: Tensor,
    ) -> BackgroundDecoderOutput:
        """Decode a background residual at the input resolution.

        Args:
            x: Raw input ``[B, 3, H, W]`` in ``[0, 1]``.
            bg_h8: ``BottleneckOutput.bg_h8`` at ``H/8``.
            bg_h16: ``BottleneckOutput.bg_h16`` at ``H/16``.
            mask_feat: Mask decoder feature; resized to ``H/8`` if needed.
            mask_prob: Mask probability at any resolution; resized to
                ``mask_feat`` before concatenation.
            skip_h4: Encoder ``x_h4`` skip at ``H/4``.
            skip_h2: Encoder ``x_h2`` skip at ``H/2``.
            skip_h: Encoder ``x_h`` skip at ``H``.
        """
        if x.dim() != 4:
            raise ValueError(
                "BackgroundDecoder expects a 4D [B, 3, H, W] input, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 3:
            raise ValueError(
                f"BackgroundDecoder expects 3 input channels, got {x.shape[1]}"
            )

        height, width = int(x.shape[2]), int(x.shape[3])
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "BackgroundDecoder requires H and W to be multiples of 16, "
                f"got H={height}, W={width}"
            )

        expected_h8 = (height // 8, width // 8)
        expected_h16 = (height // 16, width // 16)
        expected_h4 = (height // 4, width // 4)
        expected_h2 = (height // 2, width // 2)

        for name, tensor, channels, spatial in (
            ("bg_h8", bg_h8, self.bg_h8_channels, expected_h8),
            ("bg_h16", bg_h16, self.bg_h16_channels, expected_h16),
            ("skip_h4", skip_h4, self.skip_h4_channels, expected_h4),
            ("skip_h2", skip_h2, self.skip_h2_channels, expected_h2),
            ("skip_h", skip_h, self.skip_h_channels, (height, width)),
        ):
            if tensor.dim() != 4:
                raise ValueError(
                    f"{name} must be 4D, got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[0] != x.shape[0]:
                raise ValueError(
                    f"{name} batch mismatch: {tensor.shape[0]} vs "
                    f"{x.shape[0]}"
                )
            if tensor.shape[1] != channels:
                raise ValueError(
                    f"{name} must have {channels} channels, "
                    f"got {tensor.shape[1]}"
                )
            if tuple(tensor.shape[2:]) != spatial:
                raise ValueError(
                    f"{name} must be spatially {spatial} for an input of "
                    f"H={height}, W={width}, got {tuple(tensor.shape[2:])}"
                )

        for name, tensor, channels in (
            ("mask_feat", mask_feat, self.mask_feat_channels),
            ("mask_prob", mask_prob, self.mask_prob_channels),
        ):
            if tensor.dim() != 4:
                raise ValueError(
                    f"{name} must be 4D, got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[0] != x.shape[0]:
                raise ValueError(
                    f"{name} batch mismatch: {tensor.shape[0]} vs "
                    f"{x.shape[0]}"
                )
            if tensor.shape[1] != channels:
                raise ValueError(
                    f"{name} must have {channels} channels, "
                    f"got {tensor.shape[1]}"
                )

        if mask_prob.shape[2:] != mask_feat.shape[2:]:
            mask_prob = F.interpolate(
                mask_prob,
                size=mask_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        cond = torch.cat((mask_feat, mask_prob), dim=1)

        # H/16 context -> H/8.
        context = self.bg16_proj(bg_h16)
        context = F.interpolate(
            context,
            size=bg_h8.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        h = self.bg8_proj(bg_h8) + context

        # FiLM conditioning is the only place mask information enters.
        h = self.film(h, cond)

        h4 = self.levels[0](h, skip_h4)
        h2 = self.levels[1](h4, skip_h2)
        h_full = self.levels[2](h2, skip_h)

        coarse_bg = x + self.head(h_full)
        if self.clamp_output:
            coarse_bg = coarse_bg.clamp(min=0.0, max=1.0)

        return BackgroundDecoderOutput(
            coarse_bg=coarse_bg,
            skip_h4=h4,
            skip_h2=h2,
            skip_h=h_full,
        )

    def extra_repr(self) -> str:
        return (
            f"widths=({self.width_h8}, {self.widths[0]}, "
            f"{self.widths[1]}, {self.widths[2]}), "
            f"mask={self.mask_feat_channels}+{self.mask_prob_channels}, "
            f"clamp={self.clamp_output}"
        )
