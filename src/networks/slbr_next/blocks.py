"""Core building blocks for SLBR-Next (spec ``slbr-next.md`` §3.2).

Design constraints:

* Channels-first ``[B, C, H, W]`` tensors; ``H`` and ``W`` may differ.
* No BatchNorm, no InstanceNorm, no shape reversals, no FFT.
* ONNX-native operators only: Conv (including depthwise), LayerNorm,
  AdaptiveAvgPool, Mul, Add and Sigmoid.
* Channels-last friendly: the ``LayerNorm2d`` permutation is a zero-copy view
  when the input uses ``torch.channels_last`` memory format.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["LayerNorm2d", "SimpleGate", "SCA", "NAFConvNeXtBlock"]


class LayerNorm2d(nn.Module):
    """LayerNorm over channels at every spatial position.

    The normalization is per pixel, not over ``(H, W)``, so it is invariant to
    batch size, tiling and image shape (spec §3.2 and §6.2).

    Args:
        channels: Number of input channels.
        eps: LayerNorm epsilon.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                "LayerNorm2d expects a 4D [B, C, H, W] tensor, "
                f"got shape {tuple(x.shape)}"
            )
        # [B, C, H, W] -> [B, H, W, C] -> normalize over C -> back.
        # Both permutes are views in channels_last memory format.
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def extra_repr(self) -> str:
        return (
            f"channels={self.norm.normalized_shape[0]}, "
            f"eps={self.norm.eps}"
        )


class SimpleGate(nn.Module):
    """Split the channels in half and multiply them.

    Parameter-free activation used by NAFNet; it has no dead-unit failure mode
    and maps to plain ``Split``/``Mul`` in ONNX.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError(
                "SimpleGate expects an even number of channels, "
                f"got {x.shape[1]}"
            )
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    """Simplified channel attention (global average -> 1x1 conv -> sigmoid)."""

    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.conv(self.pool(x)))


class NAFConvNeXtBlock(nn.Module):
    """NAFNet-style block with a ConvNeXt-style residual layout.

    Spatial mixer::

        LayerNorm2d
          -> 1x1 Conv (C -> 2C)
          -> kxk Depthwise Conv
          -> SimpleGate
          -> SCA
          -> 1x1 Conv (C -> C)
          -> residual add

    Optional channel mixer (exactly as in spec §3.2)::

        LayerNorm2d
          -> 1x1 Conv (C -> 2C)
          -> SimpleGate
          -> 1x1 Conv (C -> C)
          -> residual add

    Args:
        channels: Input/output channel count.
        dw_kernel_size: Depthwise kernel size; must be odd. Use 3 for the
            full-resolution refinement stage on T4, 5 elsewhere (spec §5.4).
        channel_mixer: Append the channel-mixer residual branch.
        eps: LayerNorm epsilon.
        sca_bias: Whether the SCA 1x1 convolution has a bias.
    """

    def __init__(
        self,
        channels: int,
        dw_kernel_size: int = 5,
        channel_mixer: bool = True,
        eps: float = 1e-6,
        sca_bias: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if dw_kernel_size < 1 or dw_kernel_size % 2 == 0:
            raise ValueError(
                "dw_kernel_size must be a positive odd integer, "
                f"got {dw_kernel_size}"
            )

        self.channels = channels
        self.dw_kernel_size = dw_kernel_size
        self.channel_mixer = channel_mixer

        hidden = 2 * channels

        self.norm1 = LayerNorm2d(channels, eps=eps)
        self.expand_conv = nn.Conv2d(channels, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=dw_kernel_size,
            padding=dw_kernel_size // 2,
            groups=hidden,
        )
        self.gate = SimpleGate()
        self.sca = SCA(channels, bias=sca_bias)
        self.project_conv = nn.Conv2d(channels, channels, kernel_size=1)

        if channel_mixer:
            self.norm2 = LayerNorm2d(channels, eps=eps)
            self.mixer_expand = nn.Conv2d(channels, hidden, kernel_size=1)
            self.mixer_gate = SimpleGate()
            self.mixer_project = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.expand_conv(y)
        y = self.dwconv(y)
        y = self.gate(y)
        y = self.sca(y)
        y = self.project_conv(y)
        x = x + y

        if self.channel_mixer:
            z = self.norm2(x)
            z = self.mixer_expand(z)
            z = self.mixer_gate(z)
            z = self.mixer_project(z)
            x = x + z

        return x

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, "
            f"dw_kernel_size={self.dw_kernel_size}, "
            f"channel_mixer={self.channel_mixer}"
        )
