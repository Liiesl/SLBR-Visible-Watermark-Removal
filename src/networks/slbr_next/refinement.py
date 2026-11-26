"""Stage 2 background refinement for SLBR-Next (spec ``slbr-next.md`` §3.7).

The refinement U-Net sees the **raw** input, the coarse background and the
final mask in one concatenation::

    refine_in = cat([input, coarse_bg, mask_final], dim=1)   # 7 channels

This replaces ``coarse_composite``: the raw input is already present once and
is not added a second time. A slim three-scale NAF U-Net (``H``, ``H/2``,
``H/4``) with one MDTA block at ``H/8`` predicts a bounded residual::

    refine_bg = clamp(input + tanh(head(features)), 0, 1)

The background decoder supplies skips at ``H``, ``H/2`` and ``H/4``. Every
interpolation targets ``size=<reference>.shape[2:]`` with
``align_corners=False``; there is no BatchNorm or InstanceNorm.

The v1 composite stays outside the network::

    composite_v1(refine_bg, input, mask_final)
        = refine_bg * mask_final + input * (1 - mask_final)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .bg_decoder import ResampleBlock
from .blocks import LayerNorm2d, NAFConvNeXtBlock
from .bottleneck import MDTA

__all__ = ["BackgroundOutput", "RefinementUNet", "composite_v1"]


@dataclass(frozen=True)
class BackgroundOutput:
    """Frozen Stage 2 output interface for A6.

    ``coarse_bg`` is passed through from the background decoder and
    ``refine_bg`` is the refined full-resolution background. Both are
    ``[B, 3, H, W]`` and, with the default ``clamp_output=True``, lie in
    ``[0, 1]``. The final v1 composite is
    :func:`composite_v1` outside the model.
    """

    coarse_bg: Tensor
    refine_bg: Tensor


def composite_v1(
    refine_bg: Tensor,
    input: Tensor,
    mask_final: Tensor,
) -> Tensor:
    """v1 composite (spec §3.7).

    Args:
        refine_bg: Refined background ``[B, 3, H, W]``.
        input: Raw input ``[B, 3, H, W]``.
        mask_final: Final mask ``[B, 1, H, W]`` or ``[B, 3, H, W]``.

    Returns:
        ``refine_bg * mask_final + input * (1 - mask_final)``.
    """
    if refine_bg.shape != input.shape:
        raise ValueError(
            "refine_bg and input must have identical shapes, got "
            f"{tuple(refine_bg.shape)} and {tuple(input.shape)}"
        )
    if mask_final.dim() != 4:
        raise ValueError(
            "mask_final must be 4D, got shape " f"{tuple(mask_final.shape)}"
        )
    if mask_final.shape[0] != input.shape[0]:
        raise ValueError(
            f"mask_final batch mismatch: {mask_final.shape[0]} vs "
            f"{input.shape[0]}"
        )
    if mask_final.shape[1] not in (1, input.shape[1]):
        raise ValueError(
            "mask_final must have 1 channel or match the input channels, "
            f"got {mask_final.shape[1]}"
        )
    if mask_final.shape[2:] != input.shape[2:]:
        raise ValueError(
            "mask_final must match the input spatial size, got "
            f"{tuple(mask_final.shape[2:])} and {tuple(input.shape[2:])}"
        )
    return refine_bg * mask_final + input * (1.0 - mask_final)


class RefinementUNet(nn.Module):
    """Slim three-scale NAF U-Net for Stage 2 (spec §3.7, §5.4).

    Args:
        in_channels: Channels of ``cat([input, coarse_bg, mask_final])``.
            The default 7 assumes 3 + 3 + 1; the mask part may have more
            channels, in which case this is ``6 + mask_channels``.
        widths: Widths at ``(H, H/2, H/4)``. The default ``(24, 48, 96)``
            follows the slim T4 profile in §5.4. The ``H/8`` bottleneck
            reuses ``widths[2]``.
        bg_skip_channels: Channels of the background decoder skips at
            ``(H, H/2, H/4)``; defaults to ``(32, 64, 128)``.
        blocks_per_scale: NAF blocks per encoder and decoder scale.
        attention: ``"mdta"`` for one MDTA block at ``H/8`` (reference
            design), or ``"naf"`` for the T4 fallback NAF block.
        mdta_heads: MDTA heads; must divide ``widths[2]``.
        dw_kernel_sizes: Depthwise kernels at ``(H, H/2, H/4)``. The default
            uses 3x3 at full resolution and 5x5 below it (§5.4).
        norm_eps: LayerNorm epsilon.
        clamp_output: Clamp ``refine_bg`` to ``[0, 1]``. Keep enabled for
            RGB inputs in ``[0, 1]``.
    """

    def __init__(
        self,
        *,
        in_channels: int = 7,
        widths: Sequence[int] = (24, 48, 96),
        bg_skip_channels: Sequence[int] = (32, 64, 128),
        blocks_per_scale: int = 1,
        attention: str = "mdta",
        mdta_heads: int = 4,
        dw_kernel_sizes: Sequence[int] = (3, 5, 5),
        norm_eps: float = 1e-6,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()

        if in_channels < 7:
            raise ValueError(
                "in_channels must fit cat([input(3), coarse_bg(3), "
                f"mask_final]), got {in_channels}"
            )

        widths = tuple(int(width) for width in widths)
        if len(widths) != 3:
            raise ValueError(
                "widths must contain three entries for (H, H/2, H/4), "
                f"got {widths}"
            )
        if any(width <= 0 for width in widths):
            raise ValueError(f"all widths must be positive, got {widths}")

        bg_skips = tuple(int(channels) for channels in bg_skip_channels)
        if len(bg_skips) != 3:
            raise ValueError(
                "bg_skip_channels must contain three entries for "
                f"(H, H/2, H/4), got {bg_skips}"
            )
        if any(channels <= 0 for channels in bg_skips):
            raise ValueError(
                f"all bg_skip_channels must be positive, got {bg_skips}"
            )

        kernels = tuple(int(kernel) for kernel in dw_kernel_sizes)
        if len(kernels) != 3:
            raise ValueError(
                "dw_kernel_sizes must contain three entries for "
                f"(H, H/2, H/4), got {kernels}"
            )
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError(
                "dw_kernel_sizes must be positive odd integers, "
                f"got {kernels}"
            )

        if blocks_per_scale < 1:
            raise ValueError(
                "blocks_per_scale must be at least 1, "
                f"got {blocks_per_scale}"
            )
        if mdta_heads <= 0:
            raise ValueError(
                f"mdta_heads must be positive, got {mdta_heads}"
            )

        attention = attention.lower()
        if attention not in {"mdta", "naf"}:
            raise ValueError(
                "attention must be 'mdta' or 'naf', " f"got {attention!r}"
            )
        if attention == "mdta" and widths[2] % mdta_heads != 0:
            raise ValueError(
                f"widths[2] ({widths[2]}) must be divisible by mdta_heads "
                f"({mdta_heads})"
            )

        self.in_channels = int(in_channels)
        self.mask_channels = int(in_channels) - 6
        self.widths = widths
        self.bg_skip_channels = bg_skips
        self.blocks_per_scale = int(blocks_per_scale)
        self.attention = attention
        self.mdta_heads = int(mdta_heads)
        self.dw_kernel_sizes = kernels
        self.norm_eps = float(norm_eps)
        self.clamp_output = bool(clamp_output)

        width_h, width_h2, width_h4 = widths

        # H: raw input + coarse background + mask, fused with the decoder skip.
        self.stem = nn.Conv2d(in_channels, width_h, kernel_size=3, padding=1)
        self.bg_h_proj = nn.Conv2d(bg_skips[0], width_h, 1)
        self.enc_h = self._make_blocks(
            width_h, kernels[0], blocks_per_scale, norm_eps
        )

        # H -> H/2.
        self.down_h2 = ResampleBlock(width_h, width_h2)
        self.bg_h2_proj = nn.Conv2d(bg_skips[1], width_h2, 1)
        self.enc_h2 = self._make_blocks(
            width_h2, kernels[1], blocks_per_scale, norm_eps
        )

        # H/2 -> H/4.
        self.down_h4 = ResampleBlock(width_h2, width_h4)
        self.bg_h4_proj = nn.Conv2d(bg_skips[2], width_h4, 1)
        self.enc_h4 = self._make_blocks(
            width_h4, kernels[2], blocks_per_scale, norm_eps
        )

        # H/4 -> H/8 bottleneck, one MDTA (or NAF) block.
        self.down_h8 = ResampleBlock(width_h4, width_h4)
        self.bottleneck = self._make_blocks(
            width_h4, kernels[2], blocks_per_scale, norm_eps
        )
        if attention == "mdta":
            self.mdta_norm: nn.Module | None = LayerNorm2d(
                width_h4, eps=norm_eps
            )
            self.mdta: nn.Module | None = MDTA(width_h4, mdta_heads)
            self.naf_bottleneck: nn.Module | None = None
        else:
            self.mdta_norm = None
            self.mdta = None
            self.naf_bottleneck = NAFConvNeXtBlock(
                width_h4,
                dw_kernel_size=kernels[2],
                eps=norm_eps,
            )

        # H/8 -> H/4 -> H/2 -> H, with the U-Net skips.
        self.up_h4 = ResampleBlock(width_h4, width_h4)
        self.dec_h4 = self._make_blocks(
            width_h4, kernels[2], blocks_per_scale, norm_eps
        )
        self.up_h2 = ResampleBlock(width_h4, width_h2)
        self.dec_h2 = self._make_blocks(
            width_h2, kernels[1], blocks_per_scale, norm_eps
        )
        self.up_h = ResampleBlock(width_h2, width_h)
        self.dec_h = self._make_blocks(
            width_h, kernels[0], blocks_per_scale, norm_eps
        )

        self.head = nn.Conv2d(width_h, 3, kernel_size=1)

    @staticmethod
    def _make_blocks(
        channels: int,
        dw_kernel_size: int,
        blocks: int,
        eps: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                NAFConvNeXtBlock(
                    channels,
                    dw_kernel_size=dw_kernel_size,
                    eps=eps,
                )
                for _ in range(blocks)
            ]
        )

    def forward(
        self,
        x: Tensor,
        coarse_bg: Tensor,
        mask_final: Tensor,
        *,
        skip_h4: Tensor,
        skip_h2: Tensor,
        skip_h: Tensor,
    ) -> BackgroundOutput:
        """Refine the coarse background at the input resolution.

        Args:
            x: Raw input ``[B, 3, H, W]`` in ``[0, 1]``.
            coarse_bg: Coarse background ``[B, 3, H, W]`` from the decoder.
            mask_final: Final mask ``[B, mask_channels, H, W]``.
            skip_h4: Background decoder skip at ``H/4``.
            skip_h2: Background decoder skip at ``H/2``.
            skip_h: Background decoder skip at ``H``.

        Returns:
            :class:`BackgroundOutput` with ``coarse_bg`` passed through and
            the clamped ``refine_bg``.
        """
        if x.dim() != 4:
            raise ValueError(
                "RefinementUNet expects a 4D [B, 3, H, W] input, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 3:
            raise ValueError(
                f"RefinementUNet expects 3 input channels, got {x.shape[1]}"
            )

        height, width = int(x.shape[2]), int(x.shape[3])
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "RefinementUNet requires H and W to be multiples of 8, "
                f"got H={height}, W={width}"
            )

        if coarse_bg.shape != x.shape:
            raise ValueError(
                "coarse_bg must match the input shape exactly, got "
                f"{tuple(coarse_bg.shape)} and {tuple(x.shape)}"
            )
        if mask_final.dim() != 4:
            raise ValueError(
                "mask_final must be 4D, got shape "
                f"{tuple(mask_final.shape)}"
            )
        if mask_final.shape[0] != x.shape[0]:
            raise ValueError(
                f"mask_final batch mismatch: {mask_final.shape[0]} vs "
                f"{x.shape[0]}"
            )
        if mask_final.shape[1] != self.mask_channels:
            raise ValueError(
                f"mask_final must have {self.mask_channels} channels, "
                f"got {mask_final.shape[1]}"
            )
        if mask_final.shape[2:] != x.shape[2:]:
            raise ValueError(
                "mask_final must match the input spatial size, got "
                f"{tuple(mask_final.shape[2:])} and {tuple(x.shape[2:])}"
            )

        expected = {
            "skip_h4": (height // 4, width // 4),
            "skip_h2": (height // 2, width // 2),
            "skip_h": (height, width),
        }
        for name, tensor in (
            ("skip_h4", skip_h4),
            ("skip_h2", skip_h2),
            ("skip_h", skip_h),
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
            if tuple(tensor.shape[2:]) != expected[name]:
                raise ValueError(
                    f"{name} must be spatially {expected[name]} for an "
                    f"input of H={height}, W={width}, got "
                    f"{tuple(tensor.shape[2:])}"
                )
        if skip_h.shape[1] != self.bg_skip_channels[0]:
            raise ValueError(
                f"skip_h must have {self.bg_skip_channels[0]} channels, "
                f"got {skip_h.shape[1]}"
            )
        if skip_h2.shape[1] != self.bg_skip_channels[1]:
            raise ValueError(
                f"skip_h2 must have {self.bg_skip_channels[1]} channels, "
                f"got {skip_h2.shape[1]}"
            )
        if skip_h4.shape[1] != self.bg_skip_channels[2]:
            raise ValueError(
                f"skip_h4 must have {self.bg_skip_channels[2]} channels, "
                f"got {skip_h4.shape[1]}"
            )

        # The raw input appears exactly once, next to coarse_bg and the mask.
        refine_in = torch.cat((x, coarse_bg, mask_final), dim=1)

        h = self.stem(refine_in)
        h = self.enc_h(h + self.bg_h_proj(skip_h))
        enc_h = h

        h = self.down_h2(h, size=skip_h2.shape[2:])
        h = self.enc_h2(h + self.bg_h2_proj(skip_h2))
        enc_h2 = h

        h = self.down_h4(h, size=skip_h4.shape[2:])
        h = self.enc_h4(h + self.bg_h4_proj(skip_h4))
        enc_h4 = h

        h = self.down_h8(
            h,
            size=(x.shape[2] // 8, x.shape[3] // 8),
        )
        h = self.bottleneck(h)
        if self.mdta is not None:
            h = h + self.mdta(self.mdta_norm(h))
        else:
            h = self.naf_bottleneck(h)

        h = self.up_h4(h, size=enc_h4.shape[2:])
        h = self.dec_h4(h + enc_h4)

        h = self.up_h2(h, size=enc_h2.shape[2:])
        h = self.dec_h2(h + enc_h2)

        h = self.up_h(h, size=enc_h.shape[2:])
        h = self.dec_h(h + enc_h)

        refine_bg = x + torch.tanh(self.head(h))
        if self.clamp_output:
            refine_bg = refine_bg.clamp(min=0.0, max=1.0)

        return BackgroundOutput(coarse_bg=coarse_bg, refine_bg=refine_bg)

    def extra_repr(self) -> str:
        return (
            f"widths={self.widths}, "
            f"bg_skips={self.bg_skip_channels}, "
            f"attention={self.attention!r}, "
            f"clamp={self.clamp_output}"
        )
