"""Dual-resolution encoder for SLBR-Next (spec ``slbr-next.md`` §3.3, §3.8).

The encoder has two paths that meet at ``H/4``:

* a from-scratch high-resolution stem ``3 -> 32 @ H -> 64 @ H/2 -> 128 @ H/4``
  with two :class:`NAFConvNeXtBlock` per level and configurable downsampling
  (stride-2 convolution or orthonormal Haar DWT + 1x1 projection);
* a pretrained ``timm`` ConvNeXt V2 atto branch (FCMAE weights) whose stages
  at ``H/4``, ``H/8`` and ``H/16`` are projected to the encoder widths with
  1x1 convolutions. The branch receives ImageNet-normalized input internally;
  the stem sees the raw ``[0, 1]`` image.

The semantic branch runs in FP32 with autocast explicitly disabled (spec
§5.4). This covers its GRN layers and the stem/downsample/depthwise
convolutions whose FP16 parameter gradients went non-finite in the 2xT4
training gate. The Haar downsampling 1x1 projections use the same guard: the
T5 one-stage gate reported ``down1.project`` weight gradients as non-finite
while the loss was finite. The branch only operates at ``H/4``-``H/16``, and
the Haar projection only at ``H/2``-``H/4``, so their FP32 cost is a small
fraction of the full-resolution stem.

The public interface is frozen for A3-A6::

    @dataclass(frozen=True)
    class EncoderOutput:
        x_h: Tensor    # stem width 32, H
        x_h2: Tensor   # 64, H/2
        x_h4: Tensor   # 128, H/4
        x_h8: Tensor   # 256, H/8
        x_h16: Tensor  # 512, H/16

Widths are constructor-configurable so A6 can tune the parameter budget.
``H`` and ``W`` may differ but must both be multiples of 16 (spec §1.1: pad
externally, preferably with reflect mode, then crop the output). There are no
BatchNorm/InstanceNorm layers and no square-only reshapes in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .blocks import NAFConvNeXtBlock
from .haar import HaarDWT

__all__ = ["EncoderOutput", "HaarDownsample", "SLBRNextEncoder"]

# ImageNet statistics used only inside the pretrained ConvNeXt V2 branch.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EncoderOutput:
    """Feature pyramid returned by :class:`SLBRNextEncoder`.

    All tensors are channels-first and preserve the input aspect ratio. With
    the default widths and an input of ``[B, 3, H, W]``:

    ==========  =========================
    field       shape
    ==========  =========================
    ``x_h``     ``[B, 32, H, W]``
    ``x_h2``    ``[B, 64, H/2, W/2]``
    ``x_h4``    ``[B, 128, H/4, W/4]``
    ``x_h8``    ``[B, 256, H/8, W/8]``
    ``x_h16``   ``[B, 512, H/16, W/16]``
    ==========  =========================
    """

    x_h: Tensor    # stem width 32, H
    x_h2: Tensor   # 64, H/2
    x_h4: Tensor   # 128, H/4
    x_h8: Tensor   # 256, H/8
    x_h16: Tensor  # 512, H/16


class _ConvDownsample(nn.Module):
    """2x2 stride-2 convolution downsampling (spec §3.3)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def extra_repr(self) -> str:
        return (
            f"{self.conv.in_channels} -> {self.conv.out_channels}, "
            "kernel=2, stride=2"
        )


class HaarDownsample(nn.Module):
    """Haar DWT downsampling followed by a 1x1 ``4C -> C_out`` projection.

    The DWT keeps all four bands; the projection learns which frequencies to
    retain at the new scale (spec §3.3).

    The 1x1 projection runs in FP32 with autocast disabled. In the T5
    one-stage FP16 gate its weight/bias gradients were the first non-finite
    tensors even though the loss was finite: the projection was executing as
    an autocast FP16 convolution, so its scaled weight gradient accumulated in
    FP16. This is the same failure class as the mask-decoder attention and
    bottleneck guards. The DWT itself is stateless and stays on the input
    path; the projected output is cast back to the activation dtype so the
    encoder contract is unchanged.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dwt = HaarDWT(in_channels)
        self.project = nn.Conv2d(
            4 * in_channels, out_channels, kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        bands = self.dwt(x)
        if self.project.weight.dtype != torch.float32:
            # Explicitly half/bf16 models cannot mix FP32 weights with a
            # half-precision activation under autocast-disabled execution.
            return self.project(bands)
        activation_dtype = bands.dtype
        with torch.autocast(device_type=bands.device.type, enabled=False):
            projected = self.project(bands.float())
        return projected.to(dtype=activation_dtype)

    def extra_repr(self) -> str:
        return (
            f"{self.dwt.channels} -> {self.project.out_channels} "
            f"(DWT 4x -> 1x1 {self.project.in_channels} -> "
            f"{self.project.out_channels})"
        )


def _feature_channels(
    model: nn.Module, out_indices: Sequence[int]
) -> list[int]:
    """Read the selected feature widths from a ``timm`` feature extractor.

    ``timm`` versions differ on whether ``feature_info.channels()`` already
    filters by ``out_indices``; handle both layouts.
    """
    feature_info = getattr(model, "feature_info", None)
    if feature_info is None:
        raise RuntimeError(
            "The timm feature extractor does not expose feature_info; "
            "pass a ConvNeXt V2 model that supports features_only=True."
        )

    channels_getter = getattr(feature_info, "channels", None)
    channels = channels_getter() if callable(channels_getter) else feature_info
    channels = [int(channel) for channel in channels]

    if len(channels) > len(out_indices):
        try:
            channels = [channels[index] for index in out_indices]
        except IndexError as exc:
            raise RuntimeError(
                f"timm out_indices {tuple(out_indices)} exceed the available "
                f"feature stages {channels}"
            ) from exc

    if len(channels) != len(out_indices):
        raise RuntimeError(
            f"Expected {len(out_indices)} feature stages from timm, "
            f"got {len(channels)} ({channels})"
        )
    return channels


class SLBRNextEncoder(nn.Module):
    """High-resolution stem plus pretrained ConvNeXt V2 semantic branch.

    Args:
        widths: Five output widths for ``(H, H/2, H/4, H/8, H/16)``.
            Defaults to ``(32, 64, 128, 256, 512)`` (spec §3.8). A6 may widen
            or slim these to hit the overall parameter target.
        stem_blocks: Number of NAF blocks per stem level (default 2, §3.3).
        stem_dw_kernel_sizes: Depthwise kernel size for the stem blocks at
            ``(H, H/2, H/4)``. Defaults to ``(3, 5, 5)``: 3x3 at full
            resolution for T4 throughput, 5x5 below (§5.4).
        downsample: ``"haar"`` (default) for the orthonormal DWT + 1x1
            projection, or ``"conv"`` for a 2x2 stride-2 convolution.
        convnext_model: ``timm`` model name. ``convnextv2_atto`` is the
            default; ``convnextv2_nano``/``tiny`` also work.
        convnext_pretrained: Request FCMAE-pretrained weights. The branch is
            trainable after loading; freeze it explicitly if desired.
        convnext_out_indices: Three ``timm`` feature stages at ``H/4``,
            ``H/8`` and ``H/16``. Defaults to ``(0, 1, 2)``.
        norm_eps: LayerNorm epsilon for the stem blocks.
    """

    def __init__(
        self,
        widths: Sequence[int] = (32, 64, 128, 256, 512),
        *,
        stem_blocks: int = 2,
        stem_dw_kernel_sizes: Sequence[int] = (3, 5, 5),
        downsample: str = "haar",
        convnext_model: str = "convnextv2_atto",
        convnext_pretrained: bool = True,
        convnext_out_indices: Sequence[int] = (0, 1, 2),
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        widths = tuple(widths)
        if len(widths) != 5:
            raise ValueError(
                "widths must contain five entries for "
                f"(H, H/2, H/4, H/8, H/16), got {widths}"
            )
        if any(width <= 0 for width in widths):
            raise ValueError(f"all widths must be positive, got {widths}")

        kernels = tuple(stem_dw_kernel_sizes)
        if len(kernels) != 3:
            raise ValueError(
                "stem_dw_kernel_sizes must contain three entries for "
                f"(H, H/2, H/4), got {kernels}"
            )
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError(
                "stem_dw_kernel_sizes must be positive odd integers, "
                f"got {kernels}"
            )

        if stem_blocks < 1:
            raise ValueError(
                f"stem_blocks must be at least 1, got {stem_blocks}"
            )

        downsample = downsample.lower()
        if downsample not in {"haar", "conv"}:
            raise ValueError(
                "downsample must be 'haar' or 'conv', "
                f"got {downsample!r}"
            )

        out_indices = tuple(int(index) for index in convnext_out_indices)
        if (
            len(out_indices) != 3
            or any(index < 0 for index in out_indices)
            or tuple(sorted(set(out_indices))) != out_indices
        ):
            raise ValueError(
                "convnext_out_indices must be three strictly increasing "
                f"non-negative indices, got {out_indices}"
            )

        self.widths = widths
        self.stem_blocks = int(stem_blocks)
        self.stem_dw_kernel_sizes = kernels
        self.downsample = downsample
        self.convnext_model = convnext_model
        self.convnext_pretrained = bool(convnext_pretrained)
        self.convnext_out_indices = out_indices

        # --- High-resolution stem -------------------------------------
        self.stem_conv = nn.Conv2d(3, widths[0], kernel_size=3, padding=1)
        self.stem_stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        NAFConvNeXtBlock(
                            channels,
                            dw_kernel_size=kernel,
                            eps=norm_eps,
                        )
                        for _ in range(self.stem_blocks)
                    ]
                )
                for channels, kernel in zip(widths[:3], kernels)
            ]
        )
        self.down1 = self._make_downsample(
            downsample, widths[0], widths[1]
        )
        self.down2 = self._make_downsample(
            downsample, widths[1], widths[2]
        )

        # --- Pretrained semantic branch -------------------------------
        self.semantic_branch, branch_channels = self._build_branch()

        # --- Fusion with 1x1 projections ------------------------------
        # H/4 fuses the high-resolution stem with the first atto stage;
        # H/8 and H/16 are the projected lower atto stages (spec §3.8).
        self.fuse_h4 = nn.Conv2d(
            widths[2] + branch_channels[0], widths[2], kernel_size=1
        )
        self.fuse_h8 = nn.Conv2d(
            branch_channels[1], widths[3], kernel_size=1
        )
        self.fuse_h16 = nn.Conv2d(
            branch_channels[2], widths[4], kernel_size=1
        )

        # Internal ImageNet normalization for the pretrained branch only.
        self.register_buffer(
            "image_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _make_downsample(
        mode: str, in_channels: int, out_channels: int
    ) -> nn.Module:
        if mode == "haar":
            return HaarDownsample(in_channels, out_channels)
        return _ConvDownsample(in_channels, out_channels)

    def _build_branch(self) -> tuple[nn.Module, list[int]]:
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "SLBRNextEncoder requires timm for the ConvNeXt V2 branch. "
                "Install it with `pip install timm`."
            ) from exc

        model = timm.create_model(
            self.convnext_model,
            pretrained=self.convnext_pretrained,
            features_only=True,
            out_indices=self.convnext_out_indices,
        )
        channels = _feature_channels(model, self.convnext_out_indices)
        return model, channels

    def forward(self, x: Tensor) -> EncoderOutput:
        if x.dim() != 4:
            raise ValueError(
                "SLBRNextEncoder expects a 4D [B, 3, H, W] tensor, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 3:
            raise ValueError(
                f"SLBRNextEncoder expects 3 input channels, got {x.shape[1]}"
            )

        height, width = x.shape[2], x.shape[3]
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "SLBRNextEncoder requires H and W to be multiples of 16 "
                f"(spec §1.1), got H={height}, W={width}. Pad externally "
                "with reflect/replicate mode, then crop the output."
            )

        # The pretrained branch sees ImageNet-normalized input; the stem sees
        # the raw [0, 1] image. The branch and its 1x1 fusion projections run
        # in FP32 with autocast disabled: the T6 2xT4 log shows the branch
        # stem/downsample/depthwise convolution weights and biases are the
        # first non-finite FP16 gradients, while the high-resolution stem is
        # clean. FP32 covers GRN (spec §5.4) and removes FP16 convolution
        # backward from the branch's small H/4-H/16 feature pyramid.
        branch_param = next(self.semantic_branch.parameters(), None)
        branch_dtype = (
            branch_param.dtype if branch_param is not None else x.dtype
        )
        run_branch_fp32 = branch_dtype == torch.float32
        mean = self.image_mean.to(dtype=branch_dtype, device=x.device)
        std = self.image_std.to(dtype=branch_dtype, device=x.device)
        branch_input = (x.to(dtype=branch_dtype) - mean) / std

        x_h = self.stem_stages[0](self.stem_conv(x))
        x_h2 = self.stem_stages[1](self.down1(x_h))
        stem_h4 = self.stem_stages[2](self.down2(x_h2))

        if run_branch_fp32:
            with torch.autocast(device_type=x.device.type, enabled=False):
                branch_features = self.semantic_branch(branch_input)
                branch_features = self._check_branch_features(branch_features)

                x_h4 = self.fuse_h4(
                    torch.cat((stem_h4.float(), branch_features[0]), dim=1)
                )
                x_h8 = self.fuse_h8(branch_features[1])
                x_h16 = self.fuse_h16(branch_features[2])
            x_h4 = x_h4.to(dtype=x.dtype)
            x_h8 = x_h8.to(dtype=x.dtype)
            x_h16 = x_h16.to(dtype=x.dtype)
        else:
            branch_features = self._check_branch_features(
                self.semantic_branch(branch_input)
            )
            x_h4 = self.fuse_h4(
                torch.cat((stem_h4, branch_features[0]), dim=1)
            )
            x_h8 = self.fuse_h8(branch_features[1])
            x_h16 = self.fuse_h16(branch_features[2])

        return EncoderOutput(
            x_h=x_h,
            x_h2=x_h2,
            x_h4=x_h4,
            x_h8=x_h8,
            x_h16=x_h16,
        )

    @staticmethod
    def _check_branch_features(
        branch_features: Tensor | list[Tensor] | tuple[Tensor, ...],
    ) -> list[Tensor]:
        """Normalize the timm feature container to a 3-stage list."""
        if isinstance(branch_features, dict):
            branch_features = list(branch_features.values())
        features = list(branch_features)
        if len(features) != 3:
            raise RuntimeError(
                "The timm feature branch must return three stages at "
                f"H/4, H/8 and H/16, got {len(features)}"
            )
        return features

    def extra_repr(self) -> str:
        return (
            f"widths={self.widths}, stem_blocks={self.stem_blocks}, "
            f"downsample={self.downsample!r}, "
            f"convnext={self.convnext_model!r}, "
            f"pretrained={self.convnext_pretrained}"
        )
