"""Disentangled bottleneck for SLBR-Next (spec ``slbr-next.md`` §3.4, §3.8).

The bottleneck consumes :class:`~.encoder.EncoderOutput` and produces the
mask/background feature pairs that A4/A5 decode::

    @dataclass(frozen=True)
    class BottleneckOutput:
        mask_h8: Tensor
        mask_h16: Tensor
        bg_h8: Tensor
        bg_h16: Tensor

Flow
----

1. A shared ``H/8`` stem (1x1 projection + NAF block) embeds
   ``EncoderOutput.x_h8``.
2. The Haar frequency branch runs **once on the shared stem, before the
   mask/background split**. The Haar decomposition is a shared low/high
   frequency prior; running it per trunk would double the analysis/synthesis
   cost and let the two trunks learn different frequency bases. The
   disentanglement starts after this shared prior.
3. The features split into two independent trunks (separate parameters, no
   shared up-conv). Each trunk fuses ``EncoderOutput.x_h8`` through a gated
   skip, applies NAF blocks and one MDTA block at ``H/8``, downsamples to
   ``H/16``, fuses ``EncoderOutput.x_h16`` through a second gated skip, applies
   NAF blocks, and merges the ``H/16`` context back into ``H/8``.

The frequency branch itself is ``DWT -> split_haar_bands -> [LL: MDTA or NAF;
LH/HL/HH: gated depthwise conv] -> merge_haar_bands -> IWT``. The bands are
split and merged with A2's helpers, never with ``chunk(4, dim=1)``.

Invariants (spec §1.3)
----------------------

* Channels-first ``[B, C, H, W]``; ``H`` and ``W`` may differ.
* No BatchNorm or InstanceNorm anywhere.
* No shape reversals; every interpolation targets ``size=x.shape[2:]`` of the
  reference tensor, with ``align_corners=False``.
* The shared H/8 stem runs in FP32 with autocast disabled (A2/A4 pattern):
  its 1x1 stem/project/mixer convolutions receive the summed mask,
  background and frequency gradients and overflowed FP16 weight-gradient
  accumulation in the 2xT4 Stage 2 smoke. The output is cast back to the
  activation dtype.
* MDTA keeps ``F.normalize``, ``softmax`` and the attention matmul in FP32 and
  casts the result back to the activation dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import LayerNorm2d, NAFConvNeXtBlock
from .encoder import EncoderOutput
from .haar import HaarDWT, HaarIWT, merge_haar_bands, split_haar_bands

__all__ = [
    "MDTA",
    "GatedSkipFusion",
    "GatedHighFrequency",
    "HaarFrequencyBranch",
    "DisentangledTrunk",
    "BottleneckOutput",
    "SLBRNextBottleneck",
]


@dataclass(frozen=True)
class BottleneckOutput:
    """Frozen bottleneck interface consumed by A4/A5/A6.

    With the default widths and an input of ``[B, 3, H, W]``:

    =============  =========================
    field          shape
    =============  =========================
    ``mask_h8``    ``[B, 256, H/8, W/8]``
    ``mask_h16``   ``[B, 512, H/16, W/16]``
    ``bg_h8``      ``[B, 256, H/8, W/8]``
    ``bg_h16``     ``[B, 512, H/16, W/16]``
    =============  =========================
    """

    mask_h8: Tensor
    mask_h16: Tensor
    bg_h8: Tensor
    bg_h16: Tensor


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer, spec §3.4).

    ``q``/``k`` are L2-normalized along the **spatial** axis (all ``H * W``
    positions of a head), exactly as in Restormer; they are not normalized
    along channels. ``F.normalize``, ``softmax`` and the two attention matmuls
    run in FP32 for numerical safety, then the result is cast back to the
    activation dtype before the output projection.

    The spatial dimension stays symbolic (``-1``) in the reshapes so dynamic
    ONNX export, ``B > 1`` and non-square ``H != W`` all work.

    Args:
        dim: Input/output channel count.
        num_heads: Number of attention heads; must divide ``dim``.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        self.channels = int(dim)
        self.num_heads = int(num_heads)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dw = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            padding=1,
            groups=dim * 3,
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "MDTA expects a 4D [B, C, H, W] tensor, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"MDTA expected {self.channels} input channels, "
                f"got {x.shape[1]}"
            )

        batch, channels, height, width = x.shape
        heads = self.num_heads
        head_dim = channels // heads

        qkv = self.qkv_dw(self.qkv(x))
        query, key, value = qkv.chunk(3, dim=1)

        # [B, C, H, W] -> [B, heads, C / heads, H * W]; keep the spatial
        # dimension symbolic for dynamic ONNX export.
        query = query.reshape(batch, heads, head_dim, -1)
        key = key.reshape(batch, heads, head_dim, -1)
        value = value.reshape(batch, heads, head_dim, -1)

        # Full FP32 attention; normalize over H * W, not over channels.
        query_f = F.normalize(query.float(), dim=-1)
        key_f = F.normalize(key.float(), dim=-1)
        attn = (query_f @ key_f.transpose(-2, -1)) * self.temperature.float()
        attn = attn.softmax(dim=-1)

        out = (attn @ value.float()).to(x.dtype)
        out = out.reshape(batch, channels, height, width)
        return self.proj(out)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, num_heads={self.num_heads}"


class GatedSkipFusion(nn.Module):
    """Sigmoid-gated residual fusion of an encoder skip.

    ::

        fused = x + sigmoid(gate(proj(skip))) * proj(skip)

    The gate is computed from the projected skip, so the trunk decides how
    much of the encoder feature to admit. If the spatial sizes differ the
    skip is resized to ``x.shape[2:]`` with bilinear interpolation (never a
    reversed ``(H, W)``).

    Args:
        skip_channels: Channels of the encoder skip tensor.
        channels: Channels of the trunk tensor.
    """

    def __init__(self, skip_channels: int, channels: int) -> None:
        super().__init__()
        if skip_channels <= 0:
            raise ValueError(
                f"skip_channels must be positive, got {skip_channels}"
            )
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.skip_channels = int(skip_channels)
        self.channels = int(channels)
        self.skip_proj = nn.Conv2d(skip_channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if x.dim() != 4 or skip.dim() != 4:
            raise ValueError(
                "GatedSkipFusion expects 4D tensors, got "
                f"{tuple(x.shape)} and {tuple(skip.shape)}"
            )
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        projected = self.skip_proj(skip)
        return x + torch.sigmoid(self.gate(projected)) * projected

    def extra_repr(self) -> str:
        return f"skip_channels={self.skip_channels}, channels={self.channels}"


class GatedHighFrequency(nn.Module):
    """Gated depthwise conv for the ``LH/HL/HH`` Haar bands.

    The three bands are concatenated in ``[LH, HL, HH]`` order and processed by
    one grouped depthwise 3x3 convolution plus a grouped 1x1 sigmoid gate
    (``groups=3`` so each group gates exactly one band). This is equivalent to
    three band-specific depthwise convs but uses a single launch.

    Splitting the result back uses ``Tensor.split(channels, dim=1)``: after
    the concatenation the bands are **contiguous**, unlike the interleaved
    DWT coefficient layout.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer, "
                f"got {kernel_size}"
            )

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        hidden = 3 * channels
        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden,
        )
        # groups=3 keeps [LH, HL, HH] as three independent channel groups.
        self.gate = nn.Conv2d(hidden, hidden, kernel_size=1, groups=3)

    def forward(
        self, lh: Tensor, hl: Tensor, hh: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not (lh.shape == hl.shape == hh.shape):
            raise ValueError(
                "GatedHighFrequency expects three bands with identical "
                f"shapes, got {tuple(lh.shape)}, {tuple(hl.shape)}, "
                f"{tuple(hh.shape)}"
            )
        if lh.shape[1] != self.channels:
            raise ValueError(
                f"GatedHighFrequency expected {self.channels} channels, "
                f"got {lh.shape[1]}"
            )

        y = self.dwconv(torch.cat((lh, hl, hh), dim=1))
        y = y * torch.sigmoid(self.gate(y))
        lh, hl, hh = y.split(self.channels, dim=1)
        return lh, hl, hh

    def extra_repr(self) -> str:
        return f"channels={self.channels}, kernel_size={self.kernel_size}"


class HaarFrequencyBranch(nn.Module):
    """Haar frequency branch at ``H/8`` without FFT (spec §3.4).

    ``DWT`` produces ``LL, LH, HL, HH`` at ``H/16``. ``LL`` is contextualized
    by MDTA (default) or a NAF block; ``LH/HL/HH`` go through gated depthwise
    convolutions. ``IWT`` reconstructs ``H/8`` with the same fixed Haar
    filters. The input and output channel counts are equal.

    Args:
        channels: Input/output channel count.
        freq_width: Internal width after the 1x1 pre-projection. Lowering it
            reduces the branch cost without touching the trunk width.
        ll_mode: ``"mdta"`` (global color/structure context) or ``"naf"``.
        heads: MDTA heads; only used when ``ll_mode="mdta"``.
        kernel_size: Depthwise kernel for the NAF block and the high-frequency
            convs; must be odd.
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        channels: int,
        *,
        freq_width: int = 128,
        ll_mode: str = "mdta",
        heads: int = 2,
        kernel_size: int = 3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if freq_width <= 0:
            raise ValueError(
                f"freq_width must be positive, got {freq_width}"
            )
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer, "
                f"got {kernel_size}"
            )
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if freq_width % heads != 0:
            raise ValueError(
                f"freq_width ({freq_width}) must be divisible by heads "
                f"({heads})"
            )

        ll_mode = ll_mode.lower()
        if ll_mode not in {"mdta", "naf"}:
            raise ValueError(
                "ll_mode must be 'mdta' or 'naf', " f"got {ll_mode!r}"
            )

        self.channels = int(channels)
        self.freq_width = int(freq_width)
        self.ll_mode = ll_mode
        self.heads = int(heads)
        self.kernel_size = int(kernel_size)

        self.pre = nn.Conv2d(channels, freq_width, kernel_size=1)
        self.dwt = HaarDWT(freq_width)

        if ll_mode == "mdta":
            self.ll_norm = LayerNorm2d(freq_width, eps=eps)
            self.ll_attn = MDTA(freq_width, heads)
            self.ll_block = None
        else:
            self.ll_norm = None
            self.ll_attn = None
            self.ll_block = NAFConvNeXtBlock(
                freq_width,
                dw_kernel_size=kernel_size,
                eps=eps,
            )

        self.high_frequency = GatedHighFrequency(
            freq_width, kernel_size=kernel_size
        )
        self.iwt = HaarIWT(freq_width)
        self.post = nn.Conv2d(freq_width, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "HaarFrequencyBranch expects a 4D [B, C, H, W] tensor, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"HaarFrequencyBranch expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )

        y = self.pre(x)
        ll, lh, hl, hh = split_haar_bands(self.dwt(y))

        if self.ll_mode == "mdta":
            ll = ll + self.ll_attn(self.ll_norm(ll))
        else:
            ll = self.ll_block(ll)

        lh, hl, hh = self.high_frequency(lh, hl, hh)
        y = self.iwt(merge_haar_bands(ll, lh, hl, hh))
        return self.post(y)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, freq_width={self.freq_width}, "
            f"ll_mode={self.ll_mode!r}, heads={self.heads}"
        )


class DisentangledTrunk(nn.Module):
    """One mask or background trunk after the ``H/8`` split.

    The trunk fuses the encoder skip at ``H/8``, applies NAF blocks and one
    MDTA block, travels to ``H/16`` (fusing the ``H/16`` encoder skip), then
    merges the ``H/16`` context back into ``H/8``. Mask and background use
    separate instances, so no up-conv weights or skip projections are shared.

    Args:
        in_channels: Channels of the shared bottleneck stem.
        channels_h8: Trunk width at ``H/8``.
        channels_h16: Trunk width at ``H/16``.
        skip_h8_channels: Channels of ``EncoderOutput.x_h8``.
        skip_h16_channels: Channels of ``EncoderOutput.x_h16``.
        blocks_h8: NAF blocks at ``H/8`` before the ``H/16`` excursion.
        blocks_h16: NAF blocks at ``H/16``.
        blocks_h8_post: NAF blocks at ``H/8`` after the ``H/16`` merge.
        mdta_heads: MDTA heads at ``H/8``.
        dw_kernel_size: NAF depthwise kernel size.
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        in_channels: int,
        channels_h8: int,
        channels_h16: int,
        skip_h8_channels: int,
        skip_h16_channels: int,
        *,
        blocks_h8: int = 1,
        blocks_h16: int = 1,
        blocks_h8_post: int = 1,
        mdta_heads: int = 4,
        dw_kernel_size: int = 5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("channels_h8", channels_h8),
            ("channels_h16", channels_h16),
            ("skip_h8_channels", skip_h8_channels),
            ("skip_h16_channels", skip_h16_channels),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in (
            ("blocks_h8", blocks_h8),
            ("blocks_h16", blocks_h16),
            ("blocks_h8_post", blocks_h8_post),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if dw_kernel_size < 1 or dw_kernel_size % 2 == 0:
            raise ValueError(
                "dw_kernel_size must be a positive odd integer, "
                f"got {dw_kernel_size}"
            )

        self.in_channels = int(in_channels)
        self.channels_h8 = int(channels_h8)
        self.channels_h16 = int(channels_h16)

        if in_channels == channels_h8:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Conv2d(in_channels, channels_h8, 1)

        self.skip_h8 = GatedSkipFusion(skip_h8_channels, channels_h8)
        self.blocks_h8 = nn.Sequential(
            *[
                NAFConvNeXtBlock(
                    channels_h8,
                    dw_kernel_size=dw_kernel_size,
                    eps=eps,
                )
                for _ in range(blocks_h8)
            ]
        )
        self.mdta_norm = LayerNorm2d(channels_h8, eps=eps)
        self.mdta = MDTA(channels_h8, mdta_heads)

        self.down_proj = nn.Conv2d(channels_h8, channels_h16, 1)
        self.skip_h16 = GatedSkipFusion(skip_h16_channels, channels_h16)
        self.blocks_h16 = nn.Sequential(
            *[
                NAFConvNeXtBlock(
                    channels_h16,
                    dw_kernel_size=dw_kernel_size,
                    eps=eps,
                )
                for _ in range(blocks_h16)
            ]
        )

        # Bilinear -> 3x3 depthwise -> 1x1 (spec §3.3).
        self.up_dwconv = nn.Conv2d(
            channels_h16,
            channels_h16,
            kernel_size=3,
            padding=1,
            groups=channels_h16,
        )
        self.up_proj = nn.Conv2d(channels_h16, channels_h8, 1)
        self.blocks_h8_post = nn.Sequential(
            *[
                NAFConvNeXtBlock(
                    channels_h8,
                    dw_kernel_size=dw_kernel_size,
                    eps=eps,
                )
                for _ in range(blocks_h8_post)
            ]
        )

    def forward(
        self, x: Tensor, skip_h8: Tensor, skip_h16: Tensor
    ) -> tuple[Tensor, Tensor]:
        x = self.input_proj(x)
        x = self.skip_h8(x, skip_h8)
        x = self.blocks_h8(x)
        x = x + self.mdta(self.mdta_norm(x))

        h16 = F.interpolate(
            x,
            size=skip_h16.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        h16 = self.down_proj(h16)
        h16 = self.skip_h16(h16, skip_h16)
        h16 = self.blocks_h16(h16)

        up = F.interpolate(
            h16,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        up = self.up_dwconv(up)
        up = self.up_proj(up)

        x = x + up
        x = self.blocks_h8_post(x)
        return x, h16

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"channels_h8={self.channels_h8}, "
            f"channels_h16={self.channels_h16}"
        )


class SLBRNextBottleneck(nn.Module):
    """Disentangled mask/background bottleneck (spec §3.4, §3.8).

    The shared ``H/8`` stem and the Haar frequency branch run once; the mask
    and background trunks then process ``H/8``/``H/16`` independently with
    gated skips from ``EncoderOutput.x_h8`` and ``EncoderOutput.x_h16``.

    Args:
        encoder_h8_channels: Channels of ``EncoderOutput.x_h8`` (256 by
            default, matching the H/8 encoder width).
        encoder_h16_channels: Channels of ``EncoderOutput.x_h16`` (512).
        mask_width: Mask trunk width at ``H/8`` (256, spec §3.8).
        bg_width: Background trunk width at ``H/8`` (256, spec §3.8).
        mask_width_h16: Mask trunk width at ``H/16`` (512).
        bg_width_h16: Background trunk width at ``H/16`` (512).
        shared_width: Width of the shared H/8 stem before the split. Defaults
            to ``max(mask_width, bg_width)``.
        stem_blocks: NAF blocks in the shared H/8 stem.
        blocks_h8: NAF blocks per trunk at ``H/8`` (pre excursion).
        blocks_h16: NAF blocks per trunk at ``H/16``.
        blocks_h8_post: NAF blocks per trunk at ``H/8`` (post merge).
        mdta_heads: MDTA heads per trunk at ``H/8``; must divide both trunk
            widths at ``H/8``.
        frequency_branch: Enable the Haar frequency branch.
        freq_width: Internal width of the frequency branch.
        freq_ll: ``"mdta"`` or ``"naf"`` for the ``LL`` band.
        freq_heads: MDTA heads in the frequency branch.
        freq_kernel_size: Depthwise kernel in the frequency branch.
        dw_kernel_size: NAF depthwise kernel in the stem and trunks.
        norm_eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        *,
        encoder_h8_channels: int = 256,
        encoder_h16_channels: int = 512,
        mask_width: int = 256,
        bg_width: int = 256,
        mask_width_h16: int = 512,
        bg_width_h16: int = 512,
        shared_width: int | None = None,
        stem_blocks: int = 1,
        blocks_h8: int = 1,
        blocks_h16: int = 1,
        blocks_h8_post: int = 1,
        mdta_heads: int = 4,
        frequency_branch: bool = True,
        freq_width: int = 128,
        freq_ll: str = "mdta",
        freq_heads: int = 2,
        freq_kernel_size: int = 3,
        dw_kernel_size: int = 5,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        for name, value in (
            ("encoder_h8_channels", encoder_h8_channels),
            ("encoder_h16_channels", encoder_h16_channels),
            ("mask_width", mask_width),
            ("bg_width", bg_width),
            ("mask_width_h16", mask_width_h16),
            ("bg_width_h16", bg_width_h16),
            ("freq_width", freq_width),
            ("freq_heads", freq_heads),
            ("mdta_heads", mdta_heads),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if shared_width is not None and shared_width <= 0:
            raise ValueError(
                f"shared_width must be positive, got {shared_width}"
            )
        for name, value in (
            ("stem_blocks", stem_blocks),
            ("blocks_h8", blocks_h8),
            ("blocks_h16", blocks_h16),
            ("blocks_h8_post", blocks_h8_post),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if dw_kernel_size < 1 or dw_kernel_size % 2 == 0:
            raise ValueError(
                "dw_kernel_size must be a positive odd integer, "
                f"got {dw_kernel_size}"
            )
        if freq_kernel_size < 1 or freq_kernel_size % 2 == 0:
            raise ValueError(
                "freq_kernel_size must be a positive odd integer, "
                f"got {freq_kernel_size}"
            )
        if mask_width % mdta_heads != 0 or bg_width % mdta_heads != 0:
            raise ValueError(
                "mdta_heads must divide both mask_width and bg_width, got "
                f"mdta_heads={mdta_heads}, mask_width={mask_width}, "
                f"bg_width={bg_width}"
            )
        if freq_width % freq_heads != 0:
            raise ValueError(
                f"freq_width ({freq_width}) must be divisible by freq_heads "
                f"({freq_heads})"
            )
        if freq_ll.lower() not in {"mdta", "naf"}:
            raise ValueError(
                "freq_ll must be 'mdta' or 'naf', " f"got {freq_ll!r}"
            )

        self.encoder_h8_channels = int(encoder_h8_channels)
        self.encoder_h16_channels = int(encoder_h16_channels)
        self.mask_width = int(mask_width)
        self.bg_width = int(bg_width)
        self.mask_width_h16 = int(mask_width_h16)
        self.bg_width_h16 = int(bg_width_h16)
        self.shared_width = int(
            shared_width if shared_width is not None else max(mask_width, bg_width)
        )
        self.stem_blocks = int(stem_blocks)
        self.blocks_h8 = int(blocks_h8)
        self.blocks_h16 = int(blocks_h16)
        self.blocks_h8_post = int(blocks_h8_post)
        self.mdta_heads = int(mdta_heads)
        self.frequency_branch = bool(frequency_branch)
        self.freq_width = int(freq_width)
        self.freq_ll = freq_ll.lower()
        self.freq_heads = int(freq_heads)
        self.freq_kernel_size = int(freq_kernel_size)
        self.dw_kernel_size = int(dw_kernel_size)
        self.norm_eps = float(norm_eps)

        # Shared H/8 stem; the frequency branch runs once, before the split.
        self.h8_stem = nn.Sequential(
            nn.Conv2d(encoder_h8_channels, self.shared_width, 1),
            *[
                NAFConvNeXtBlock(
                    self.shared_width,
                    dw_kernel_size=dw_kernel_size,
                    eps=norm_eps,
                )
                for _ in range(stem_blocks)
            ],
        )
        if self.frequency_branch:
            self.freq_branch: nn.Module = HaarFrequencyBranch(
                self.shared_width,
                freq_width=freq_width,
                ll_mode=freq_ll,
                heads=freq_heads,
                kernel_size=freq_kernel_size,
                eps=norm_eps,
            )
        else:
            self.freq_branch = nn.Identity()

        trunk_kwargs = {
            "in_channels": self.shared_width,
            "skip_h8_channels": encoder_h8_channels,
            "skip_h16_channels": encoder_h16_channels,
            "blocks_h8": blocks_h8,
            "blocks_h16": blocks_h16,
            "blocks_h8_post": blocks_h8_post,
            "mdta_heads": mdta_heads,
            "dw_kernel_size": dw_kernel_size,
            "eps": norm_eps,
        }
        self.mask_trunk = DisentangledTrunk(
            channels_h8=mask_width,
            channels_h16=mask_width_h16,
            **trunk_kwargs,
        )
        self.bg_trunk = DisentangledTrunk(
            channels_h8=bg_width,
            channels_h16=bg_width_h16,
            **trunk_kwargs,
        )

    def forward(self, features: EncoderOutput) -> BottleneckOutput:
        """Run the bottleneck on an :class:`EncoderOutput`.

        Use :meth:`forward_features` when only the two H/8/H/16 tensors are
        available.
        """
        if not isinstance(features, EncoderOutput):
            raise TypeError(
                "SLBRNextBottleneck.forward expects an EncoderOutput; use "
                "forward_features(x_h8, x_h16) for plain tensors"
            )
        return self.forward_features(features.x_h8, features.x_h16)

    def _forward_shared_stem(self, x: Tensor) -> Tensor:
        """Run the shared H/8 stem in FP32 with autocast disabled.

        The 2xT4 Stage 2 smoke reports non-finite unscaled gradients first in
        ``h8_stem.0.bias`` and the NAF block's ``project_conv``/``mixer``
        1x1 convolutions while the loss is finite. This stem is shared by the
        mask trunk, background trunk and frequency branch, so its backward
        receives the sum of all three paths. With GradScaler's initial
        ``65536`` scale, the FP16 weight/bias-gradient accumulation over the
        ``B * H/8 * W/8`` samples overflows.

        Follow the A2/A4 pattern: run the small H/8 stem inside
        ``autocast(enabled=False)`` with FP32 tensors, then cast the result
        back to the activation dtype. AMP stores FP32 master weights, so the
        guarded path applies to real FP16 training; an explicitly half model
        keeps its native path and is unchanged.
        """
        parameter = next(self.h8_stem.parameters(), None)
        if parameter is None or parameter.dtype != torch.float32:
            return self.h8_stem(x)

        activation_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            stem = self.h8_stem(x.float())
        return stem.to(dtype=activation_dtype)

    def forward_features(
        self, x_h8: Tensor, x_h16: Tensor
    ) -> BottleneckOutput:
        if x_h8.dim() != 4 or x_h16.dim() != 4:
            raise ValueError(
                "SLBRNextBottleneck expects 4D encoder features, got "
                f"{tuple(x_h8.shape)} and {tuple(x_h16.shape)}"
            )
        if x_h8.shape[1] != self.encoder_h8_channels:
            raise ValueError(
                f"Expected {self.encoder_h8_channels} channels at H/8, "
                f"got {x_h8.shape[1]}"
            )
        if x_h16.shape[1] != self.encoder_h16_channels:
            raise ValueError(
                f"Expected {self.encoder_h16_channels} channels at H/16, "
                f"got {x_h16.shape[1]}"
            )
        if self.frequency_branch and (
            x_h8.shape[2] % 2 != 0 or x_h8.shape[3] % 2 != 0
        ):
            raise ValueError(
                "The Haar frequency branch requires even H/8 and W/8 "
                f"(input multiple of 16), got {tuple(x_h8.shape[2:])}"
            )

        shared = self._forward_shared_stem(x_h8)
        if self.frequency_branch:
            shared = shared + self.freq_branch(shared)

        mask_h8, mask_h16 = self.mask_trunk(shared, x_h8, x_h16)
        bg_h8, bg_h16 = self.bg_trunk(shared, x_h8, x_h16)

        return BottleneckOutput(
            mask_h8=mask_h8,
            mask_h16=mask_h16,
            bg_h8=bg_h8,
            bg_h16=bg_h16,
        )

    def extra_repr(self) -> str:
        return (
            f"encoder=({self.encoder_h8_channels}, "
            f"{self.encoder_h16_channels}), "
            f"widths=({self.mask_width}, {self.bg_width}) -> "
            f"({self.mask_width_h16}, {self.bg_width_h16}), "
            f"freq={self.frequency_branch}"
        )
