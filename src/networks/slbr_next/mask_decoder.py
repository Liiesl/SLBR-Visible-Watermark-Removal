"""K-query mask decoder for SLBR-Next (spec ``slbr-next.md`` §3.5, §3.8).

The decoder consumes the mask-trunk feature at ``H/8`` from A3 and optional
encoder skips at ``H/4``, ``H/2`` and ``H``::

    mask_h8 [B, C8, H/8, W/8]
      -> NAF stem
      -> masked cross-attention @ H/8   (K + 1 queries, coarse-mask bias)
      -> bilinear + DWConv + 1x1, optional H/4 skip
      -> masked cross-attention @ H/4   (K + 1 queries, coarse-mask bias)
      -> mask logits @ H/4
      -> bilinear + DWConv + 1x1, optional H/2 skip
      -> NAF blocks -> part scores -> mask logits @ H/2
      -> bilinear + DWConv + 1x1, optional H skip
      -> NAF blocks -> part scores -> mask logits @ H

Queries
-------
``K = 8`` learnable part queries followed by one watermark-free/background
query. The query dimension equals the mask-trunk width at ``H/8`` (the first
entry of ``widths``). The attention blocks update both the queries and the
spatial mask features, and the updated features propagate upward.

Logits
------
Part scores ``z_k`` are raw pixel-query similarities, softmaxed over the
``K + 1`` queries to get part probabilities. The binary foreground logit is::

    mask_logits = logsumexp(z_1 .. z_K) - z_bg

so ``sigmoid(mask_logits) == 1 - p_bg`` exactly. Raw part logits are returned
in :attr:`MaskDecoderOutput.part_logits` for the auxiliary part-diversity
loss; post-softmax probabilities must never reach ``BCEWithLogits``.

Invariants (spec §1.3)
----------------------
* Channels-first ``[B, C, H, W]``; ``H`` and ``W`` may differ.
* No BatchNorm or InstanceNorm anywhere.
* No shape reversals; every interpolation receives an explicit
  ``size=reference.shape[2:]`` (or the exact doubled H/8 size when the
  optional reference skip is omitted), with ``align_corners=False``.
* Every attention projection and decoder-upsample 1x1 projection,
  normalization, softmax and logsumexp subtraction runs in FP32 with
  autocast disabled and is cast back to the activation dtype. This keeps
  FP16 projection weight gradients finite while preserving the output
  dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import LayerNorm2d, NAFConvNeXtBlock

__all__ = [
    "MaskDecoderOutput",
    "MaskedCrossAttention",
    "PartScoreHead",
    "MaskDecoder",
]

# Clamp for ``log(coarse_probability)`` in the attention bias. The bias is
# added before the softmax, so a fully-zero coarse mask simply becomes a
# constant (uniform) shift rather than ``-inf``.
_BIAS_EPS = 1e-4


@dataclass(frozen=True)
class MaskDecoderOutput:
    """Frozen mask-decoder interface consumed by A5/A6.

    With an input of ``[B, 3, H, W]`` (padded to a multiple of 16) and the
    default widths:

    =================  =========================
    field              shape
    =================  =========================
    ``mask_logits``    ``[B, 1, H, W]``
    ``mask_logits_h2`` ``[B, 1, H/2, W/2]``
    ``mask_logits_h4`` ``[B, 1, H/4, W/4]``
    ``mask_feat``      ``[B, 32, H, W]``
    ``part_logits``    ``[B, K + 1, H, W]``
    =================  =========================

    ``part_logits`` holds raw (pre-softmax) pixel-query similarities at full
    resolution, with the ``K`` part entries first and the background entry
    last. They are the input to the auxiliary part-diversity loss and must
    not be passed through ``BCEWithLogits``.

    ``mask_logits`` is the foreground logit
    ``logsumexp(part_logits[:, :K]) - part_logits[:, K]``, so
    ``sigmoid(mask_logits) == 1 - softmax(part_logits)[:, K]``.
    """

    mask_logits: Tensor      # B, 1, H, W
    mask_logits_h2: Tensor   # B, 1, H/2, W/2
    mask_logits_h4: Tensor   # B, 1, H/4, W/4
    mask_feat: Tensor        # B, C, H, W
    part_logits: Tensor      # raw K+1 part logits; full H, W


def _foreground_logits(part_logits: Tensor) -> Tensor:
    """``logsumexp(z_1..z_K) - z_bg`` from raw ``[B, K+1, H, W]`` scores.

    Evaluated in FP32 so the result is finite for FP16 activations, then cast
    back to the activation dtype. This preserves
    ``sigmoid(mask_logits) == 1 - p_bg`` without ever touching post-softmax
    probabilities.
    """
    if part_logits.dim() != 4:
        raise ValueError(
            "part_logits must be a 4D [B, K + 1, H, W] tensor, "
            f"got shape {tuple(part_logits.shape)}"
        )
    foreground = part_logits[:, :-1].float()
    background = part_logits[:, -1:].float()
    logits = torch.logsumexp(foreground, dim=1, keepdim=True) - background
    return logits.to(part_logits.dtype)


def _fp32_autocast(device: torch.device) -> torch.autocast:
    """Disable autocast so enclosed projections execute in FP32."""
    return torch.autocast(device_type=device.type, enabled=False)


def _linear_fp32(x: Tensor, linear: nn.Linear) -> Tensor:
    """FP32 linear that bypasses autocast and FP16 parameter storage.

    Under autocast training the master weights are already FP32, so the
    ``.float()`` calls are no-ops. Under an explicit ``.half()`` model they
    still keep the projection (and its weight gradient) in FP32.
    """
    bias = None if linear.bias is None else linear.bias.float()
    return F.linear(x.float(), linear.weight.float(), bias)


def _conv2d_fp32(x: Tensor, conv: nn.Conv2d) -> Tensor:
    """FP32 convolution that bypasses autocast and FP16 parameter storage.

    ``kv_proj`` in particular previously ran as an autocast FP16 convolution;
    its FP16 weight-gradient accumulation was the first tensor to go
    non-finite in the 2xT4 smoke.
    """
    bias = None if conv.bias is None else conv.bias.float()
    return F.conv2d(
        x.float(),
        conv.weight.float(),
        bias,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )


def _layer_norm_fp32(x: Tensor, norm: nn.LayerNorm) -> Tensor:
    """FP32 LayerNorm over the last dimension, ignoring autocast."""
    weight = None if norm.weight is None else norm.weight.float()
    bias = None if norm.bias is None else norm.bias.float()
    return F.layer_norm(
        x.float(), norm.normalized_shape, weight, bias, norm.eps
    )


def _layer_norm_2d_fp32(x: Tensor, norm: LayerNorm2d) -> Tensor:
    """FP32 :class:`LayerNorm2d` that returns channels-first FP32."""
    y = x.float().permute(0, 2, 3, 1)
    y = _layer_norm_fp32(y, norm.norm)
    return y.permute(0, 3, 1, 2)


def _pixel_query_scores(
    features: Tensor,
    queries: Tensor,
    pixel_norm: nn.Module,
    pixel_proj: nn.Module,
    query_norm: nn.Module,
    query_key: nn.Module,
) -> Tensor:
    """LayerNormed pixel-query similarity ``[B, Q, H * W]`` in FP32.

    Both the pixel embedding and the query key are normalized per pixel by
    LayerNorm, so the scaled dot product stays ``O(1)``. Every projection and
    matmul is forced to FP32; callers must run this inside
    :func:`_fp32_autocast` and cast the result back to the activation dtype.
    """
    pixel = _layer_norm_2d_fp32(
        _conv2d_fp32(features, pixel_proj), pixel_norm
    )
    pixel = pixel.flatten(2)
    query = _linear_fp32(_layer_norm_fp32(queries, query_norm), query_key)

    # The explicit 1/sqrt(D) scale keeps the logits well conditioned.
    scores = torch.matmul(query, pixel)
    scores = scores * (pixel.shape[1] ** -0.5)
    return scores


class MaskedCrossAttention(nn.Module):
    """Masked query<->pixel cross-attention at one decoder scale.

    ``K + 1`` learnable queries cross-attend to the spatial mask features
    (query-to-pixel, softmax over all pixels). A coarse mask probability
    predicted from the current features biases the attention logits: the
    ``K`` part queries are pulled toward foreground pixels and the background
    query toward background pixels. The outputs update the queries.

    The reverse direction (pixel-to-query) uses a softmax over the ``K + 1``
    queries and updates the spatial features with the query values. The same
    similarity computation produces the raw part scores ``z_k`` returned to
    the caller.

    All attention math -- LayerNorm, q/k/v and output projections, the
    temperature scaling, the coarse-mask bias, both softmaxes and the
    attention matmuls -- runs in FP32 with autocast explicitly disabled, then
    the feature and part-logit outputs are cast back to the activation dtype.
    This keeps the FP16 weight-gradient path finite without changing the
    module interface.

    Args:
        channels: Mask feature width at this scale.
        query_dim: Query width; equals the mask-trunk width at ``H/8``.
        num_queries: ``K + 1`` (parts plus background).
        heads: Attention heads; must divide ``query_dim``.
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        channels: int,
        query_dim: int,
        num_queries: int,
        heads: int,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if query_dim <= 0:
            raise ValueError(f"query_dim must be positive, got {query_dim}")
        if num_queries < 2:
            raise ValueError(
                "num_queries must include at least one part and one "
                f"background query, got {num_queries}"
            )
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if query_dim % heads != 0:
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by heads "
                f"({heads})"
            )

        self.channels = int(channels)
        self.query_dim = int(query_dim)
        self.num_queries = int(num_queries)
        self.heads = int(heads)
        self.head_dim = self.query_dim // self.heads
        self.eps = float(eps)

        # Query -> pixel attention.
        self.query_norm = nn.LayerNorm(self.query_dim, eps=eps)
        self.query_proj = nn.Linear(self.query_dim, self.query_dim)
        self.feature_norm = LayerNorm2d(self.channels, eps=eps)
        self.kv_proj = nn.Conv2d(
            self.channels, 2 * self.query_dim, kernel_size=1
        )
        self.out_proj = nn.Linear(self.query_dim, self.query_dim)
        self.temperature = nn.Parameter(torch.ones(self.heads, 1, 1))

        # Coarse mask prior used as an additive attention bias.
        self.coarse_norm = LayerNorm2d(self.channels, eps=eps)
        self.coarse_head = nn.Conv2d(
            self.channels, 1, kernel_size=3, padding=1
        )

        # Pixel -> query attention and part scores.
        self.pixel_norm = LayerNorm2d(self.query_dim, eps=eps)
        self.pixel_proj = nn.Conv2d(
            self.channels, self.query_dim, kernel_size=1
        )
        self.score_query_norm = nn.LayerNorm(self.query_dim, eps=eps)
        self.score_query_key = nn.Linear(self.query_dim, self.query_dim)
        self.query_value = nn.Linear(self.query_dim, self.channels)
        self.spatial_proj = nn.Conv2d(
            self.channels, self.channels, kernel_size=1
        )

    def coarse_mask_bias(self, coarse_prob: Tensor) -> Tensor:
        """Additive softmax bias ``[B, 1, Q, H * W]`` from ``p_foreground``."""
        prob = coarse_prob.flatten(2).float()
        prob = prob.clamp(min=_BIAS_EPS, max=1.0 - _BIAS_EPS)

        foreground = torch.log(prob)
        background = torch.log1p(-prob)
        bias = torch.cat(
            (
                foreground.expand(-1, self.num_queries - 1, -1),
                background,
            ),
            dim=1,
        )
        return bias.unsqueeze(1)

    def forward(
        self, features: Tensor, queries: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run masked cross-attention.

        Args:
            features: ``[B, channels, h, w]`` mask features at this scale.
            queries: ``[B, K + 1, query_dim]`` current queries.

        Returns:
            ``(features, queries, part_logits)`` where ``features`` and
            ``queries`` are updated and ``part_logits`` is the raw
            ``[B, K + 1, h, w]`` pixel-query similarity.
        """
        if features.dim() != 4:
            raise ValueError(
                "MaskedCrossAttention expects 4D features, got "
                f"{tuple(features.shape)}"
            )
        if features.shape[1] != self.channels:
            raise ValueError(
                f"MaskedCrossAttention expected {self.channels} feature "
                f"channels, got {features.shape[1]}"
            )
        if queries.dim() != 3:
            raise ValueError(
                "MaskedCrossAttention expects 3D queries "
                f"[B, Q, D], got {tuple(queries.shape)}"
            )
        if queries.shape[1] != self.num_queries:
            raise ValueError(
                f"MaskedCrossAttention expected {self.num_queries} queries, "
                f"got {queries.shape[1]}"
            )
        if queries.shape[2] != self.query_dim:
            raise ValueError(
                f"MaskedCrossAttention expected query dim "
                f"{self.query_dim}, got {queries.shape[2]}"
            )

        batch, _, height, width = features.shape
        activation_dtype = features.dtype

        # Every projection, normalization, temperature/bias term, softmax and
        # matmul below runs in FP32 with autocast disabled. ``kv_proj`` is the
        # historical failure point: its weight gradient was accumulated in
        # FP16 under autocast and went non-finite even when the loss was
        # finite. The feature and part-logit outputs are cast back to the
        # activation dtype so the decoder's memory/layout contract is
        # unchanged.
        with _fp32_autocast(features.device):
            # --- Query -> pixel: masked softmax attention ---------------
            coarse_prob = torch.sigmoid(
                _conv2d_fp32(
                    _layer_norm_2d_fp32(features, self.coarse_norm),
                    self.coarse_head,
                )
            )

            query = _linear_fp32(
                _layer_norm_fp32(queries, self.query_norm),
                self.query_proj,
            )
            query = query.reshape(
                batch, self.num_queries, self.heads, self.head_dim
            ).permute(0, 2, 1, 3)

            key, value = _conv2d_fp32(
                _layer_norm_2d_fp32(features, self.feature_norm),
                self.kv_proj,
            ).chunk(2, dim=1)
            key = key.reshape(batch, self.heads, -1, self.head_dim)
            value = value.reshape(batch, self.heads, -1, self.head_dim)

            logits = torch.matmul(query, key.transpose(-2, -1))
            logits = logits * self.temperature.float()
            logits = logits + self.coarse_mask_bias(coarse_prob)
            attention = logits.softmax(dim=-1)

            attended = torch.matmul(attention, value)
            attended = attended.permute(0, 2, 1, 3).reshape(
                batch, self.num_queries, self.query_dim
            )
            queries = queries.float() + _linear_fp32(
                attended, self.out_proj
            )

            # --- Pixel -> query: part scores and spatial update ----------
            part_logits = _pixel_query_scores(
                features,
                queries,
                self.pixel_norm,
                self.pixel_proj,
                self.score_query_norm,
                self.score_query_key,
            )
            probabilities = part_logits.softmax(dim=1)

            values = _linear_fp32(
                _layer_norm_fp32(queries, self.score_query_norm),
                self.query_value,
            )
            update = torch.matmul(values.transpose(1, 2), probabilities)
            update = update.reshape(
                batch, self.channels, height, width
            )
            update = _conv2d_fp32(update, self.spatial_proj)
            features = features.float() + update

            part_logits = part_logits.reshape(
                batch, self.num_queries, height, width
            )

        return (
            features.to(activation_dtype),
            queries,
            part_logits.to(activation_dtype),
        )

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, query_dim={self.query_dim}, "
            f"queries={self.num_queries}, heads={self.heads}"
        )


class PartScoreHead(nn.Module):
    """Raw pixel-query part scores at one decoder scale.

    A lightweight head for scales without a full cross-attention block
    (``H/2`` and ``H``). It returns the pre-softmax ``[B, K + 1, h, w]``
    scores used by the logsumexp foreground logit and the part-diversity
    auxiliary loss. As in :class:`MaskedCrossAttention`, the projections,
    normalization and score matmul run in explicit FP32 and the scores are
    cast back to the activation dtype.

    Args:
        channels: Feature width at this scale.
        query_dim: Query width.
        eps: LayerNorm epsilon.
    """

    def __init__(
        self, channels: int, query_dim: int, *, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if query_dim <= 0:
            raise ValueError(f"query_dim must be positive, got {query_dim}")

        self.channels = int(channels)
        self.query_dim = int(query_dim)
        self.pixel_norm = LayerNorm2d(self.query_dim, eps=eps)
        self.pixel_proj = nn.Conv2d(
            self.channels, self.query_dim, kernel_size=1
        )
        self.query_norm = nn.LayerNorm(self.query_dim, eps=eps)
        self.query_key = nn.Linear(self.query_dim, self.query_dim)

    def forward(self, features: Tensor, queries: Tensor) -> Tensor:
        activation_dtype = features.dtype
        with _fp32_autocast(features.device):
            scores = _pixel_query_scores(
                features,
                queries,
                self.pixel_norm,
                self.pixel_proj,
                self.query_norm,
                self.query_key,
            )
        return scores.reshape(
            features.shape[0], queries.shape[1], *features.shape[2:]
        ).to(activation_dtype)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, query_dim={self.query_dim}"


class _DecoderUpsample(nn.Module):
    """Bilinear -> 3x3 DWConv -> 1x1, with an optional gated skip.

    The 1x1 ``project`` runs in FP32 with autocast disabled. In the T5
    one-stage FP16 gate its weight/bias gradients were non-finite while the
    loss stayed finite: the projection was an autocast FP16 convolution, so
    its scaled parameter gradients accumulated in FP16. This is the same
    failure class as the attention ``kv_proj`` guard. The DWConv and the
    gated-skip path stay in the activation dtype.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.skip_channels = int(skip_channels)

        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_norm = LayerNorm2d(out_channels, eps=eps)
        self.skip_proj = nn.Conv2d(
            skip_channels, out_channels, kernel_size=1
        )
        self.skip_gate = nn.Conv2d(
            out_channels, out_channels, kernel_size=1
        )

    def forward(
        self, x: Tensor, size: tuple[int, int], skip: Tensor | None = None
    ) -> Tensor:
        x = F.interpolate(
            x, size=size, mode="bilinear", align_corners=False
        )
        x = self.dwconv(x)
        if self.project.weight.dtype != torch.float32:
            # Explicitly half/bf16 models cannot mix FP32 weights with a
            # half-precision activation under autocast-disabled execution.
            x = self.project(x)
        else:
            activation_dtype = x.dtype
            with _fp32_autocast(x.device):
                x = self.project(x.float())
            x = x.to(dtype=activation_dtype)

        if skip is not None:
            projected = self.skip_proj(skip)
            gate = torch.sigmoid(
                self.skip_gate(self.skip_norm(projected))
            )
            x = x + gate * projected
        return x

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels} -> {self.out_channels}, "
            f"skip={self.skip_channels}"
        )


class MaskDecoder(nn.Module):
    """K-query mask decoder (spec §3.5).

    Args:
        in_channels: Channels of the ``H/8`` mask-trunk feature from A3.
        widths: Four decoder widths ``(H/8, H/4, H/2, H)``. The first entry is
            the mask-trunk width and therefore the query dimension; defaults
            to ``(256, 128, 64, 32)`` (spec §3.8).
        skip_channels: Encoder skip widths at ``(H/4, H/2, H)``. Defaults to
            ``(128, 64, 32)``, matching the default encoder. The skips are
            optional at call time; when omitted the decoder derives the exact
            doubled ``H/8`` sizes and simply skips the fusion.
        num_parts: Number of foreground part queries ``K`` (default 8); one
            background query is added.
        heads: Query-attention heads; must divide ``widths[0]``.
        blocks_per_level: NAF blocks after each level's attention/upsample,
            in ``(H/8, H/4, H/2, H)`` order.
        dw_kernel_sizes: Depthwise kernel per level. Defaults to
            ``(5, 5, 5, 3)``: 3x3 at full resolution for T4 (§5.4).
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        in_channels: int = 256,
        *,
        widths: Sequence[int] = (256, 128, 64, 32),
        skip_channels: Sequence[int] = (128, 64, 32),
        num_parts: int = 8,
        heads: int = 4,
        blocks_per_level: Sequence[int] = (1, 1, 1, 1),
        dw_kernel_sizes: Sequence[int] = (5, 5, 5, 3),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )

        widths = tuple(widths)
        if len(widths) != 4:
            raise ValueError(
                "widths must contain four entries for (H/8, H/4, H/2, H), "
                f"got {widths}"
            )
        if any(width <= 0 for width in widths):
            raise ValueError(f"all widths must be positive, got {widths}")

        skip_channels = tuple(skip_channels)
        if len(skip_channels) != 3:
            raise ValueError(
                "skip_channels must contain three entries for (H/4, H/2, H), "
                f"got {skip_channels}"
            )
        if any(channels <= 0 for channels in skip_channels):
            raise ValueError(
                f"all skip_channels must be positive, got {skip_channels}"
            )

        if num_parts < 1:
            raise ValueError(f"num_parts must be at least 1, got {num_parts}")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if widths[0] % heads != 0:
            raise ValueError(
                f"widths[0] ({widths[0]}) must be divisible by heads "
                f"({heads})"
            )

        blocks_per_level = tuple(blocks_per_level)
        if len(blocks_per_level) != 4:
            raise ValueError(
                "blocks_per_level must contain four entries, got "
                f"{blocks_per_level}"
            )
        if any(blocks < 0 for blocks in blocks_per_level):
            raise ValueError(
                "blocks_per_level entries must be non-negative, got "
                f"{blocks_per_level}"
            )

        dw_kernel_sizes = tuple(dw_kernel_sizes)
        if len(dw_kernel_sizes) != 4:
            raise ValueError(
                "dw_kernel_sizes must contain four entries, got "
                f"{dw_kernel_sizes}"
            )
        if any(
            kernel < 1 or kernel % 2 == 0 for kernel in dw_kernel_sizes
        ):
            raise ValueError(
                "dw_kernel_sizes must be positive odd integers, got "
                f"{dw_kernel_sizes}"
            )

        self.in_channels = int(in_channels)
        self.widths = widths
        self.skip_channels = skip_channels
        self.num_parts = int(num_parts)
        self.num_queries = int(num_parts) + 1
        self.query_dim = int(widths[0])
        self.heads = int(heads)
        self.blocks_per_level = blocks_per_level
        self.dw_kernel_sizes = dw_kernel_sizes
        self.eps = float(eps)

        if in_channels == widths[0]:
            self.input_proj: nn.Module = nn.Identity()
        else:
            self.input_proj = nn.Conv2d(
                in_channels, widths[0], kernel_size=1
            )

        self.stem = NAFConvNeXtBlock(
            widths[0],
            dw_kernel_size=dw_kernel_sizes[0],
            eps=eps,
        )

        # K part queries followed by one watermark-free/background query.
        self.queries = nn.Parameter(
            torch.empty(1, self.num_queries, self.query_dim)
        )
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Masked cross-attention at H/8 and H/4 (spec §3.5).
        self.attention_h8 = MaskedCrossAttention(
            widths[0],
            self.query_dim,
            self.num_queries,
            heads,
            eps=eps,
        )
        self.attention_h4 = MaskedCrossAttention(
            widths[1],
            self.query_dim,
            self.num_queries,
            heads,
            eps=eps,
        )

        self.blocks_h8 = self._make_blocks(
            widths[0], blocks_per_level[0], dw_kernel_sizes[0], eps
        )
        self.blocks_h4 = self._make_blocks(
            widths[1], blocks_per_level[1], dw_kernel_sizes[1], eps
        )
        self.blocks_h2 = self._make_blocks(
            widths[2], blocks_per_level[2], dw_kernel_sizes[2], eps
        )
        self.blocks_h = self._make_blocks(
            widths[3], blocks_per_level[3], dw_kernel_sizes[3], eps
        )

        self.up_h8_h4 = _DecoderUpsample(
            widths[0], widths[1], skip_channels[0], eps=eps
        )
        self.up_h4_h2 = _DecoderUpsample(
            widths[1], widths[2], skip_channels[1], eps=eps
        )
        self.up_h2_h = _DecoderUpsample(
            widths[2], widths[3], skip_channels[2], eps=eps
        )

        self.score_h2 = PartScoreHead(widths[2], self.query_dim, eps=eps)
        self.score_h = PartScoreHead(widths[3], self.query_dim, eps=eps)

    @staticmethod
    def _make_blocks(
        channels: int,
        count: int,
        kernel_size: int,
        eps: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                NAFConvNeXtBlock(
                    channels,
                    dw_kernel_size=kernel_size,
                    eps=eps,
                )
                for _ in range(count)
            ]
        )

    def _check_skip(
        self,
        skip: Tensor | None,
        name: str,
        channels: int,
        expected_size: tuple[int, int],
    ) -> None:
        if skip is None:
            return
        if skip.dim() != 4:
            raise ValueError(
                f"{name} must be a 4D [B, C, H, W] tensor, "
                f"got shape {tuple(skip.shape)}"
            )
        if skip.shape[1] != channels:
            raise ValueError(
                f"{name} expected {channels} channels, got {skip.shape[1]}"
            )
        if tuple(skip.shape[2:]) != expected_size:
            raise ValueError(
                f"{name} expected spatial size {expected_size} "
                f"(2x/4x/8x the H/8 input), got {tuple(skip.shape[2:])}"
            )

    def forward(
        self,
        mask_h8: Tensor,
        *,
        skip_h4: Tensor | None = None,
        skip_h2: Tensor | None = None,
        skip_h: Tensor | None = None,
    ) -> MaskDecoderOutput:
        """Decode mask logits from the ``H/8`` mask trunk feature.

        Args:
            mask_h8: ``[B, in_channels, H/8, W/8]`` from A3.
            skip_h4: Optional encoder feature ``[B, C4, H/4, W/4]``.
            skip_h2: Optional encoder feature ``[B, C2, H/2, W/2]``.
            skip_h: Optional encoder feature ``[B, C, H, W]``.

        The optional skips supply both the fusion features and the exact
        interpolation targets. When omitted, the targets are the exact
        ``2x``, ``4x`` and ``8x`` multiples of the ``H/8`` size, which is
        valid for inputs padded to a multiple of 16 (spec §1.1).
        """
        if mask_h8.dim() != 4:
            raise ValueError(
                "MaskDecoder expects a 4D [B, C, H/8, W/8] tensor, "
                f"got shape {tuple(mask_h8.shape)}"
            )
        if mask_h8.shape[1] != self.in_channels:
            raise ValueError(
                f"MaskDecoder expected {self.in_channels} input channels, "
                f"got {mask_h8.shape[1]}"
            )

        batch, _, h8, w8 = mask_h8.shape
        h4, w4 = 2 * h8, 2 * w8
        h2, w2 = 4 * h8, 4 * w8
        h, w = 8 * h8, 8 * w8

        self._check_skip(
            skip_h4, "skip_h4", self.skip_channels[0], (h4, w4)
        )
        self._check_skip(
            skip_h2, "skip_h2", self.skip_channels[1], (h2, w2)
        )
        self._check_skip(skip_h, "skip_h", self.skip_channels[2], (h, w))

        size_h4 = skip_h4.shape[2:] if skip_h4 is not None else (h4, w4)
        size_h2 = skip_h2.shape[2:] if skip_h2 is not None else (h2, w2)
        size_h = skip_h.shape[2:] if skip_h is not None else (h, w)

        features = self.input_proj(mask_h8)
        features = self.stem(features)
        queries = self.queries.expand(batch, -1, -1)

        # H/8: masked cross-attention updates queries and features.
        features, queries, _ = self.attention_h8(features, queries)
        features = self.blocks_h8(features)

        # H/4: updated H/8 features propagate upward, then attention again.
        features = self.up_h8_h4(features, size_h4, skip_h4)
        features, queries, part_h4 = self.attention_h4(
            features, queries
        )
        features = self.blocks_h4(features)
        mask_logits_h4 = _foreground_logits(part_h4)

        # H/2: part scores against the twice-updated queries.
        features = self.up_h4_h2(features, size_h2, skip_h2)
        features = self.blocks_h2(features)
        mask_logits_h2 = _foreground_logits(
            self.score_h2(features, queries)
        )

        # H: final mask feature and full-resolution part/mask logits.
        features = self.up_h2_h(features, size_h, skip_h)
        features = self.blocks_h(features)
        part_logits = self.score_h(features, queries)
        mask_logits = _foreground_logits(part_logits)

        return MaskDecoderOutput(
            mask_logits=mask_logits,
            mask_logits_h2=mask_logits_h2,
            mask_logits_h4=mask_logits_h4,
            mask_feat=features,
            part_logits=part_logits,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, widths={self.widths}, "
            f"queries={self.num_queries} x {self.query_dim}, "
            f"skips={self.skip_channels}"
        )
