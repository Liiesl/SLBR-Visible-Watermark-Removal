"""Tests for the SLBR-Next core blocks (spec §3.2, §5.4; issue #3)."""

import pytest
import torch
from torch import nn

from src.networks.slbr_next import (
    LayerNorm2d,
    NAFConvNeXtBlock,
    SCA,
    SimpleGate,
)


def _block(channels=16, **kwargs):
    torch.manual_seed(0)
    return NAFConvNeXtBlock(channels, **kwargs)


# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(256, 256), (256, 512)])
def test_forward_backward(size):
    block = _block(32)
    x = torch.randn(1, 32, *size, requires_grad=True)

    out = block(x)

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    loss = out.square().mean() + out.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


@pytest.mark.parametrize("size", [(37, 83), (64, 96)])
def test_arbitrary_non_square_shape(size):
    block = _block(12).eval()
    x = torch.randn(2, 12, *size)

    with torch.no_grad():
        out = block(x)

    assert out.shape == (2, 12, *size)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Dtype / memory format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    block = _block(12).to(dtype=dtype).eval()
    x = torch.randn(2, 12, 32, 48, dtype=dtype)

    with torch.no_grad():
        out = block(x)

    assert out.shape == x.shape
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP16 forward/backward is CUDA-only; CPU CI skips it gracefully.",
)
def test_fp16_forward_backward_cuda():
    block = NAFConvNeXtBlock(16).cuda().half()
    x = torch.randn(
        1, 16, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True
    )

    out = block(x)

    assert out.shape == x.shape
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()

    out.float().square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()


def test_channels_last_memory_format():
    block = _block(16).eval()
    x = torch.randn(2, 16, 64, 96)
    x_channels_last = x.contiguous(memory_format=torch.channels_last)
    assert x_channels_last.is_contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        out = block(x_channels_last)
        out_contiguous = block(x)
        normed = block.norm1(x_channels_last)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # LayerNorm2d keeps the NHWC layout: both permutes are zero-copy views.
    assert normed.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(
        out, out_contiguous, rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# Residual behaviour
# ---------------------------------------------------------------------------


def test_residual_is_identity_when_projections_are_zero():
    block = _block(8, dw_kernel_size=3).eval()
    with torch.no_grad():
        block.project_conv.weight.zero_()
        block.project_conv.bias.zero_()
        block.mixer_project.weight.zero_()
        block.mixer_project.bias.zero_()

    x = torch.randn(2, 8, 23, 37)
    out = block(x)

    torch.testing.assert_close(out, x, rtol=0.0, atol=0.0)


def test_block_is_not_identity_at_init():
    block = _block(8).eval()
    x = torch.randn(2, 8, 23, 37)

    with torch.no_grad():
        out = block(x)

    assert not torch.allclose(out, x)


def test_spatial_residual_without_channel_mixer():
    # Only the spatial branch is active, so zeroing it must yield the input.
    block = _block(8, channel_mixer=False).eval()
    with torch.no_grad():
        block.project_conv.weight.zero_()
        block.project_conv.bias.zero_()

    x = torch.randn(2, 8, 16, 24)
    torch.testing.assert_close(block(x), x, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", [3, 5])
def test_depthwise_kernel_is_configurable(kernel):
    block = _block(8, dw_kernel_size=kernel, channel_mixer=False)

    assert block.dwconv.kernel_size == (kernel, kernel)
    assert block.dwconv.padding == (kernel // 2, kernel // 2)
    assert block.dwconv.groups == 16

    with torch.no_grad():
        out = block(torch.randn(2, 8, 32, 48))

    assert out.shape == (2, 8, 32, 48)


def test_even_depthwise_kernel_is_rejected():
    with pytest.raises(ValueError):
        NAFConvNeXtBlock(8, dw_kernel_size=4)


def test_channel_mixer_toggle():
    with_mixer = _block(16, channel_mixer=True)
    without_mixer = _block(16, channel_mixer=False)

    assert hasattr(with_mixer, "norm2")
    assert hasattr(with_mixer, "mixer_expand")
    assert hasattr(with_mixer, "mixer_gate")
    assert hasattr(with_mixer, "mixer_project")
    assert not hasattr(without_mixer, "norm2")

    channels, kernel = 16, 5
    spatial_only = 4 * channels**2 + 2 * channels * kernel**2 + 7 * channels
    channel_only = 3 * channels**2 + 5 * channels

    spatial_params = sum(p.numel() for p in without_mixer.parameters())
    full_params = sum(p.numel() for p in with_mixer.parameters())

    assert spatial_params == spatial_only
    assert full_params == spatial_only + channel_only


# ---------------------------------------------------------------------------
# Parameter-count report (see tests/test_slbr_next/param_counts.md)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [32, 64, 128])
@pytest.mark.parametrize("channel_mixer", [False, True])
def test_parameter_count_report(channels, channel_mixer):
    block = NAFConvNeXtBlock(
        channels, dw_kernel_size=5, channel_mixer=channel_mixer
    )
    total = sum(p.numel() for p in block.parameters())

    # LN1 (2C) + expand (2C^2 + 2C) + DW (2C*k^2 + 2C)
    # + SCA (C^2, no bias) + project (C^2 + C)
    # + optional LN2 (2C) + mixer expand (2C^2 + 2C)
    # + mixer project (C^2 + C).
    expected = 4 * channels**2 + 2 * channels * 5**2 + 7 * channels
    if channel_mixer:
        expected += 3 * channels**2 + 5 * channels

    print(
        f"NAFConvNeXtBlock(channels={channels}, dw_kernel_size=5, "
        f"channel_mixer={channel_mixer}): {total:,} parameters"
    )
    assert total == expected


# ---------------------------------------------------------------------------
# LayerNorm2d / SimpleGate / SCA
# ---------------------------------------------------------------------------


def test_layer_norm2d_normalizes_channels_per_pixel():
    layer_norm = LayerNorm2d(8, eps=1e-6).eval()
    x = torch.randn(3, 8, 11, 13) * 2.5 - 0.8

    with torch.no_grad():
        out = layer_norm(x)

    assert out.shape == x.shape
    mean = out.mean(dim=1)
    var = out.var(dim=1, unbiased=False)
    torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=0, atol=1e-5)
    torch.testing.assert_close(var, torch.ones_like(var), rtol=0, atol=1e-4)


def test_layer_norm2d_is_batch_and_tile_invariant():
    layer_norm = LayerNorm2d(4, eps=1e-6).eval()
    x = torch.randn(1, 4, 16, 16)

    with torch.no_grad():
        full = layer_norm(x)
        tiled = torch.cat(
            [
                torch.cat(
                    [layer_norm(x[..., :8, :8]), layer_norm(x[..., :8, 8:])],
                    dim=-1,
                ),
                torch.cat(
                    [layer_norm(x[..., 8:, :8]), layer_norm(x[..., 8:, 8:])],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        batched = layer_norm(x.repeat(3, 1, 1, 1))

    torch.testing.assert_close(tiled, full, rtol=0, atol=1e-6)
    for index in range(3):
        torch.testing.assert_close(
            batched[index : index + 1], full, rtol=0, atol=1e-6
        )


def test_layer_norm2d_rejects_non_4d_input():
    with pytest.raises(ValueError):
        LayerNorm2d(4)(torch.randn(8, 4))


def test_simple_gate_multiplies_halves():
    gate = SimpleGate()
    x = torch.randn(2, 8, 5, 7)

    out = gate(x)

    assert out.shape == (2, 4, 5, 7)
    torch.testing.assert_close(out, x[:, :4] * x[:, 4:])


def test_simple_gate_requires_even_channels():
    with pytest.raises(ValueError):
        SimpleGate()(torch.randn(2, 7, 5, 5))


def test_sca_gates_channels_globally():
    sca = SCA(6).eval()
    x = torch.randn(2, 6, 9, 11)

    with torch.no_grad():
        out = sca(x)
        gate = torch.sigmoid(sca.conv(sca.pool(x)))

    assert out.shape == x.shape
    assert gate.min() > 0.0
    assert gate.max() < 1.0
    torch.testing.assert_close(out, x * gate)


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


def test_no_batch_or_instance_norm():
    block = NAFConvNeXtBlock(16)
    forbidden = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    offenders = [
        type(module).__name__
        for module in block.modules()
        if isinstance(module, forbidden)
    ]
    assert offenders == []


def test_only_onnx_native_modules_are_used():
    allowed = (
        NAFConvNeXtBlock,
        LayerNorm2d,
        SimpleGate,
        SCA,
        nn.Conv2d,
        nn.LayerNorm,
        nn.AdaptiveAvgPool2d,
        nn.Sigmoid,
    )
    block = NAFConvNeXtBlock(8)

    unexpected = [
        type(module).__name__
        for module in block.modules()
        if not isinstance(module, allowed)
    ]
    assert unexpected == []
