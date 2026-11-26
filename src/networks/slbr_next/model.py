"""Assembled SLBR-Next v1 model (spec ``slbr-next.md`` §1, §3.8, §6.1-§6.2).

This module wires the independently tested stages into the shipping v1
contract::

    encoder -> disentangled bottleneck -> mask decoder
            -> FiLM background decoder -> refinement U-Net

Input and output contract
-------------------------

``forward(x)`` accepts any floating point ``[B, 3, H, W]`` tensor with
``H > 0`` and ``W > 0``. No square assumption is made. ``H`` and ``W`` are
padded on the right/bottom to a multiple of :data:`PAD_MULTIPLE` (16), the
network runs on the padded image, and every returned tensor is cropped back
to the original geometry. Reflection padding is used when the pad is smaller
than the input dimension; replicate padding is used when the pad is greater
than or equal to that dimension (``reflect`` is undefined there, and a tiny
input must not crash). Zero padding is never used.

The v1 return value is exactly::

    ([refine_bg, coarse_bg], mask_list, [None])

* ``refine_bg`` and ``coarse_bg`` are ``[B, 3, H, W]``, finite, and, with the
  default ``clamp_output=True``, in ``[0, 1]``.
* ``imwatermark`` is a one-element list containing ``None`` (SLBR's legacy
  placeholder; v1 has no alpha head).
* ``mask_list`` is ordered **finest to coarsest** and never duplicates the
  final mask::

      mask_list[0] -> foreground probability [B, 1, H, W]
      mask_list[1] -> foreground probability [B, 1, ceil(H/2), ceil(W/2)]
      mask_list[2] -> foreground probability [B, 1, ceil(H/4), ceil(W/4)]

  For odd original sizes the half/quarter maps use ``ceil`` so that every
  original pixel is covered by the padded pyramid. The masks are sigmoid
  probabilities in ``[0, 1]``; the pre-sigmoid logits are available from
  :meth:`SLBRNextV1.forward_logits` in the same order for distillation and
  numerical checks.

The v1 composite is deliberately outside ``forward`` and is implemented by
:meth:`SLBRNextV1.composite` (also exposed as ``forward_composite``)::

    final = refine_bg * mask_final + input * (1 - mask_final)

where ``mask_final = mask_list[0]``. This matches spec §1.2 and §3.7.

Configuration
-------------

All widths and block counts live in :class:`SLBRNextV1Config`. The defaults
assemble the reference widths from spec §3.8 and target the 12-15M parameter
budget on T4. Pass a modified config, or keyword overrides, to tune widths::

    model = SLBRNextV1(encoder_widths=(32, 64, 128, 256, 384))
    model = SLBRNextV1(config=SLBRNextV1Config(refinement_widths=(32, 64, 128)))

``convnext_pretrained=True`` is the production default and downloads the
FCMAE ConvNeXt V2 weights in the encoder. Tests and offline fixtures may set
it to ``False``; it does not change the parameter count.

Hard invariants (spec §1.1, §1.3, §3.1)
---------------------------------------

* No BatchNorm or InstanceNorm anywhere in the assembled model (the pretrained
  ConvNeXt V2 branch uses LayerNorm/GRN).
* Every interpolation inside the stages targets a reference ``shape[2:]`` with
  ``align_corners=False``; there are no ``(H, W)`` reversals.
* No ``torch.fft`` and no custom CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .bg_decoder import BackgroundDecoder, BackgroundDecoderOutput
from .bottleneck import BottleneckOutput, SLBRNextBottleneck
from .encoder import EncoderOutput, SLBRNextEncoder
from .mask_decoder import MaskDecoder, MaskDecoderOutput
from .refinement import BackgroundOutput, RefinementUNet, composite_v1

__all__ = [
    "PAD_MULTIPLE",
    "pad_to_multiple",
    "SLBRNextV1Config",
    "SLBRNextV1",
]

#: Spatial padding multiple required by the four-stage encoding pyramid
#: (stem H/4 + ConvNeXt V2 H/16 + Haar frequency branch at H/8).
PAD_MULTIPLE = 16


def _ceil_div(value: int, divisor: int) -> int:
    """Integer ceiling division for positive values."""
    return (value + divisor - 1) // divisor


def pad_to_multiple(
    x: Tensor, multiple: int = PAD_MULTIPLE
) -> tuple[Tensor, tuple[int, int]]:
    """Pad the right/bottom of ``x`` to a multiple of ``multiple``.

    Reflection padding is used when the required pad is smaller than the
    corresponding input dimension; replicate padding is used when the pad is
    greater than or equal to that dimension (including inputs smaller than
    ``multiple``), because ``reflect`` requires ``pad < input``. No zero
    padding is ever used. The top-left origin is preserved, so the output can
    be cropped with ``[..., :H, :W]``.

    Args:
        x: 4D ``[B, C, H, W]`` tensor.
        multiple: Positive spatial multiple. Defaults to :data:`PAD_MULTIPLE`.

    Returns:
        ``(padded, (H, W))`` where ``padded`` has spatial sizes divisible by
        ``multiple`` and ``(H, W)`` is the original spatial size.
    """
    if x.dim() != 4:
        raise ValueError(
            "pad_to_multiple expects a 4D [B, C, H, W] tensor, "
            f"got shape {tuple(x.shape)}"
        )
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")

    height, width = int(x.shape[2]), int(x.shape[3])
    if height <= 0 or width <= 0:
        raise ValueError(
            "pad_to_multiple requires positive spatial sizes, got "
            f"H={height}, W={width}"
        )

    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)

    padded = x
    if pad_w:
        mode = "reflect" if pad_w < width else "replicate"
        padded = F.pad(padded, (0, pad_w, 0, 0), mode=mode)
    if pad_h:
        mode = "reflect" if pad_h < height else "replicate"
        padded = F.pad(padded, (0, 0, 0, pad_h), mode=mode)
    return padded, (height, width)


@dataclass(frozen=True)
class SLBRNextV1Config:
    """Widths and block counts for the assembled v1 model.

    The defaults follow the reference shape flow in spec §3.8 and the T4
    profile in §5.4. Every width is configurable so the overall budget can be
    tuned without editing A1-A5 modules.

    Attributes are grouped by stage. ``encoder_widths`` is
    ``(H, H/2, H/4, H/8, H/16)``; ``mask_widths`` is
    ``(H/8, H/4, H/2, H)``; ``background_widths`` is ``(H/4, H/2, H)``;
    ``refinement_widths`` is ``(H, H/2, H/4)``.
    """

    # --- Encoder (A3) --------------------------------------------------
    encoder_widths: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512)
    encoder_stem_blocks: int = 2
    encoder_stem_dw_kernel_sizes: tuple[int, int, int] = (3, 5, 5)
    encoder_downsample: str = "haar"
    convnext_model: str = "convnextv2_atto"
    convnext_pretrained: bool = True
    convnext_out_indices: tuple[int, int, int] = (0, 1, 2)

    # --- Disentangled bottleneck (A3) ----------------------------------
    bottleneck_mask_width: int = 256
    bottleneck_bg_width: int = 256
    bottleneck_mask_width_h16: int = 512
    bottleneck_bg_width_h16: int = 512
    bottleneck_shared_width: int | None = None
    bottleneck_stem_blocks: int = 1
    bottleneck_blocks_h8: int = 1
    bottleneck_blocks_h16: int = 1
    bottleneck_blocks_h8_post: int = 1
    bottleneck_mdta_heads: int = 4
    bottleneck_dw_kernel_size: int = 5
    frequency_branch: bool = True
    frequency_width: int = 128
    frequency_ll_mode: str = "mdta"
    frequency_heads: int = 2
    frequency_kernel_size: int = 3

    # --- K-query mask decoder (A4) -------------------------------------
    mask_widths: tuple[int, int, int, int] = (256, 128, 64, 32)
    mask_blocks_per_level: tuple[int, int, int, int] = (1, 1, 1, 1)
    mask_num_parts: int = 8
    mask_heads: int = 4
    mask_dw_kernel_sizes: tuple[int, int, int, int] = (5, 5, 5, 3)

    # --- FiLM background decoder (A5) ----------------------------------
    background_width_h8: int = 256
    background_widths: tuple[int, int, int] = (128, 64, 32)
    background_blocks_per_level: int = 1
    background_dw_kernel_sizes: tuple[int, int, int] = (5, 5, 3)

    # --- Refinement U-Net (A5) -----------------------------------------
    refinement_widths: tuple[int, int, int] = (24, 48, 96)
    refinement_blocks_per_scale: int = 1
    refinement_attention: str = "mdta"
    refinement_mdta_heads: int = 4
    refinement_dw_kernel_sizes: tuple[int, int, int] = (3, 5, 5)

    # --- Shared --------------------------------------------------------
    norm_eps: float = 1e-6
    clamp_output: bool = True


def _normalise_widths(
    name: str, values: Sequence[int], expected: int
) -> tuple[int, ...]:
    values = tuple(int(value) for value in values)
    if len(values) != expected:
        raise ValueError(
            f"{name} must contain {expected} entries, got {values}"
        )
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} entries must be positive, got {values}")
    return values


class SLBRNextV1(nn.Module):
    """End-to-end SLBR-Next v1 model (encoder -> decoders -> refinement).

    See the module docstring for the exact return contract, mask ordering,
    padding behaviour and composite formula.

    Args:
        config: Optional :class:`SLBRNextV1Config`. Defaults to the reference
            configuration.
        **overrides: Optional ``SLBRNextV1Config`` field overrides, applied
            with :func:`dataclasses.replace`. Unknown names raise ``TypeError``.

    The five named children are ``encoder``, ``bottleneck``, ``mask_decoder``,
    ``background_decoder`` and ``refinement``.
    """

    def __init__(
        self,
        config: SLBRNextV1Config | None = None,
        **overrides: object,
    ) -> None:
        super().__init__()

        if config is None:
            config = SLBRNextV1Config()
        elif not isinstance(config, SLBRNextV1Config):
            raise TypeError(
                "config must be an SLBRNextV1Config, "
                f"got {type(config).__name__}"
            )
        if overrides:
            field_names = {field.name for field in fields(SLBRNextV1Config)}
            unknown = sorted(set(overrides) - field_names)
            if unknown:
                raise TypeError(
                    "Unknown SLBRNextV1Config field(s): "
                    + ", ".join(unknown)
                )
            config = replace(config, **overrides)
        self.config = config

        encoder_widths = _normalise_widths(
            "encoder_widths", config.encoder_widths, 5
        )
        mask_widths = _normalise_widths(
            "mask_widths", config.mask_widths, 4
        )
        background_widths = _normalise_widths(
            "background_widths", config.background_widths, 3
        )
        refinement_widths = _normalise_widths(
            "refinement_widths", config.refinement_widths, 3
        )
        mask_kernels = _normalise_widths(
            "mask_dw_kernel_sizes", config.mask_dw_kernel_sizes, 4
        )
        background_kernels = _normalise_widths(
            "background_dw_kernel_sizes",
            config.background_dw_kernel_sizes,
            3,
        )
        refinement_kernels = _normalise_widths(
            "refinement_dw_kernel_sizes",
            config.refinement_dw_kernel_sizes,
            3,
        )
        for name, kernels in (
            ("mask_dw_kernel_sizes", mask_kernels),
            ("background_dw_kernel_sizes", background_kernels),
            ("refinement_dw_kernel_sizes", refinement_kernels),
        ):
            if any(kernel % 2 == 0 for kernel in kernels):
                raise ValueError(
                    f"{name} must be odd integers, got {kernels}"
                )

        self.encoder = SLBRNextEncoder(
            widths=encoder_widths,
            stem_blocks=config.encoder_stem_blocks,
            stem_dw_kernel_sizes=config.encoder_stem_dw_kernel_sizes,
            downsample=config.encoder_downsample,
            convnext_model=config.convnext_model,
            convnext_pretrained=config.convnext_pretrained,
            convnext_out_indices=config.convnext_out_indices,
            norm_eps=config.norm_eps,
        )

        self.bottleneck = SLBRNextBottleneck(
            encoder_h8_channels=encoder_widths[3],
            encoder_h16_channels=encoder_widths[4],
            mask_width=config.bottleneck_mask_width,
            bg_width=config.bottleneck_bg_width,
            mask_width_h16=config.bottleneck_mask_width_h16,
            bg_width_h16=config.bottleneck_bg_width_h16,
            shared_width=config.bottleneck_shared_width,
            stem_blocks=config.bottleneck_stem_blocks,
            blocks_h8=config.bottleneck_blocks_h8,
            blocks_h16=config.bottleneck_blocks_h16,
            blocks_h8_post=config.bottleneck_blocks_h8_post,
            mdta_heads=config.bottleneck_mdta_heads,
            frequency_branch=config.frequency_branch,
            freq_width=config.frequency_width,
            freq_ll=config.frequency_ll_mode,
            freq_heads=config.frequency_heads,
            freq_kernel_size=config.frequency_kernel_size,
            dw_kernel_size=config.bottleneck_dw_kernel_size,
            norm_eps=config.norm_eps,
        )

        self.mask_decoder = MaskDecoder(
            in_channels=config.bottleneck_mask_width,
            widths=mask_widths,
            skip_channels=(
                encoder_widths[2],
                encoder_widths[1],
                encoder_widths[0],
            ),
            num_parts=config.mask_num_parts,
            heads=config.mask_heads,
            blocks_per_level=config.mask_blocks_per_level,
            dw_kernel_sizes=mask_kernels,
            eps=config.norm_eps,
        )

        self.background_decoder = BackgroundDecoder(
            bg_h8_channels=config.bottleneck_bg_width,
            bg_h16_channels=config.bottleneck_bg_width_h16,
            mask_feat_channels=mask_widths[3],
            mask_prob_channels=1,
            width_h8=config.background_width_h8,
            widths=background_widths,
            skip_h4_channels=encoder_widths[2],
            skip_h2_channels=encoder_widths[1],
            skip_h_channels=encoder_widths[0],
            blocks_per_level=config.background_blocks_per_level,
            dw_kernel_sizes=background_kernels,
            norm_eps=config.norm_eps,
            clamp_output=config.clamp_output,
        )

        # ``BackgroundDecoder`` produces ``skip_h4``/``skip_h2``/``skip_h``
        # with widths ``(H/4, H/2, H)``; ``RefinementUNet`` consumes them as
        # ``(H, H/2, H/4)``.
        self.refinement = RefinementUNet(
            in_channels=7,
            widths=refinement_widths,
            bg_skip_channels=(
                background_widths[2],
                background_widths[1],
                background_widths[0],
            ),
            blocks_per_scale=config.refinement_blocks_per_scale,
            attention=config.refinement_attention,
            mdta_heads=config.refinement_mdta_heads,
            dw_kernel_sizes=refinement_kernels,
            norm_eps=config.norm_eps,
            clamp_output=config.clamp_output,
        )

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @staticmethod
    def _check_input(x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(
                "SLBRNextV1 expects a torch.Tensor, "
                f"got {type(x).__name__}"
            )
        if x.dim() != 4:
            raise ValueError(
                "SLBRNextV1 expects a 4D [B, 3, H, W] tensor, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 3:
            raise ValueError(
                f"SLBRNextV1 expects 3 RGB channels, got {x.shape[1]}"
            )
        if x.shape[2] <= 0 or x.shape[3] <= 0:
            raise ValueError(
                "SLBRNextV1 requires positive spatial sizes, got "
                f"H={x.shape[2]}, W={x.shape[3]}"
            )
        if not x.is_floating_point():
            raise TypeError(
                "SLBRNextV1 expects a floating point input, "
                f"got {x.dtype}"
            )

    # ------------------------------------------------------------------
    # Stage assembly
    # ------------------------------------------------------------------

    def _mask_stage(
        self, padded: Tensor
    ) -> tuple[EncoderOutput, BottleneckOutput, MaskDecoderOutput]:
        """Run encoder -> bottleneck -> mask decoder on padded input."""
        features = self.encoder(padded)
        bottleneck = self.bottleneck.forward_features(
            features.x_h8, features.x_h16
        )
        masks = self.mask_decoder(
            bottleneck.mask_h8,
            skip_h4=features.x_h4,
            skip_h2=features.x_h2,
            skip_h=features.x_h,
        )
        return features, bottleneck, masks

    def _forward_padded(
        self, padded: Tensor
    ) -> tuple[MaskDecoderOutput, BackgroundDecoderOutput, BackgroundOutput]:
        """Run the full pipeline on a tensor already padded to /16.

        Returns the mask decoder output plus the background decoder output
        (which carries the refinement skips) and the refinement output.
        """
        features, bottleneck, masks = self._mask_stage(padded)
        mask_final = torch.sigmoid(masks.mask_logits)

        background = self.background_decoder(
            padded,
            bottleneck.bg_h8,
            bottleneck.bg_h16,
            masks.mask_feat,
            mask_final,
            skip_h4=features.x_h4,
            skip_h2=features.x_h2,
            skip_h=features.x_h,
        )
        refined = self.refinement(
            padded,
            background.coarse_bg,
            mask_final,
            skip_h4=background.skip_h4,
            skip_h2=background.skip_h2,
            skip_h=background.skip_h,
        )
        return masks, background, refined

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def forward_logits(self, x: Tensor) -> list[Tensor]:
        """Return cropped mask logits in ``mask_list`` order.

        This runs the encoder, bottleneck and mask decoder only (no background
        decoder or refinement). It is the hook surface for Stage 1 mask
        distillation: ``[logits_H, logits_ceil_H/2, logits_ceil_H/4]``. The
        logits are finite and are the direct input to ``sigmoid`` for
        :meth:`forward`.
        """
        self._check_input(x)
        padded, (height, width) = pad_to_multiple(x, PAD_MULTIPLE)
        _, _, masks = self._mask_stage(padded)

        return [
            masks.mask_logits[..., :height, :width],
            masks.mask_logits_h2[
                ...,
                : _ceil_div(height, 2),
                : _ceil_div(width, 2),
            ],
            masks.mask_logits_h4[
                ...,
                : _ceil_div(height, 4),
                : _ceil_div(width, 4),
            ],
        ]

    def forward(
        self, x: Tensor
    ) -> tuple[list[Tensor], list[Tensor], list[None]]:
        """Run the v1 model.

        Args:
            x: ``[B, 3, H, W]`` RGB tensor in ``[0, 1]``. ``H``/``W`` are
                arbitrary positive sizes and may differ.

        Returns:
            ``([refine_bg, coarse_bg], mask_list, [None])`` exactly as
            documented in the module docstring.
        """
        self._check_input(x)
        padded, (height, width) = pad_to_multiple(x, PAD_MULTIPLE)
        masks, background, refined = self._forward_padded(padded)

        mask_full = torch.sigmoid(masks.mask_logits)[..., :height, :width]
        mask_h2 = torch.sigmoid(masks.mask_logits_h2)[
            ...,
            : _ceil_div(height, 2),
            : _ceil_div(width, 2),
        ]
        mask_h4 = torch.sigmoid(masks.mask_logits_h4)[
            ...,
            : _ceil_div(height, 4),
            : _ceil_div(width, 4),
        ]

        image_output = [
            refined.refine_bg[..., :height, :width],
            background.coarse_bg[..., :height, :width],
        ]
        return image_output, [mask_full, mask_h2, mask_h4], [None]

    def composite(self, x: Tensor) -> Tensor:
        """Return the v1 composite ``refine_bg * M + x * (1 - M)``.

        The mask is ``mask_list[0]`` from :meth:`forward`; the formula is
        exactly :func:`~.refinement.composite_v1`. This is the computation the
        ONNX v1 export performs after the single-image output.
        """
        (refine_bg, _), mask_list, _ = self.forward(x)
        return composite_v1(refine_bg, x, mask_list[0])

    def forward_composite(self, x: Tensor) -> Tensor:
        """Documented alias of :meth:`composite` for symmetry with ``forward``."""
        return self.composite(x)

    def extra_repr(self) -> str:
        config = self.config
        return (
            f"encoder={tuple(config.encoder_widths)}, "
            f"bottleneck=({config.bottleneck_mask_width}, "
            f"{config.bottleneck_bg_width}), "
            f"mask={tuple(config.mask_widths)}, "
            f"background={tuple(config.background_widths)}, "
            f"refinement={tuple(config.refinement_widths)}, "
            f"pretrained={config.convnext_pretrained}"
        )
