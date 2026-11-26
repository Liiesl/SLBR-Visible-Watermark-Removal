"""Tests for the SLBR-Next dual-resolution encoder (issue #4).

Covers spec §3.3 (encoder, Haar DWT), §3.8 (shape flow), §1.1 (input
contract) and §1.3 (spatial invariants).

No network access is required by default. The ConvNeXt V2 loading path is
tested by mocking ``timm.create_model`` and asserting that
``pretrained=True`` is requested. To run the real FCMAE download test::

    SLBR_NEXT_TEST_PRETRAINED=1 pytest tests/test_slbr_next/test_encoder.py

FP16 is CUDA-only and is skipped gracefully on the CPU CI runner.
"""

import os
from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from torch import nn

from src.networks.slbr_next.encoder import (
    EncoderOutput,
    HaarDownsample,
    SLBRNextEncoder,
)
from src.networks.slbr_next.haar import (
    HaarDWT,
    HaarIWT,
    merge_haar_bands,
    split_haar_bands,
)


# ---------------------------------------------------------------------------
# Fixtures and fake timm branch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encoder():
    """Default-width encoder with randomly initialized (not downloaded) atto."""
    torch.manual_seed(0)
    return SLBRNextEncoder(convnext_pretrained=False).eval()


class _FakeFeatureInfo:
    def __init__(self, channels):
        self._channels = tuple(channels)

    def channels(self):
        return list(self._channels)


class _FakeConvNext(nn.Module):
    """Minimal features_only ConvNeXt V2 stand-in at H/4, H/8 and H/16."""

    def __init__(self, channels=(4, 8, 16)):
        super().__init__()
        self.channels = tuple(channels)
        self.feature_info = _FakeFeatureInfo(channels)
        self.last_input = None
        self.last_outputs = None
        self.recorded = None

    def forward(self, x):
        self.last_input = x
        height, width = x.shape[2], x.shape[3]
        outputs = [
            x.new_full(
                (x.shape[0], channels, height // reduction, width // reduction),
                float(channels),
            )
            for channels, reduction in zip(self.channels, (4, 8, 16))
        ]
        self.last_outputs = outputs
        return outputs


def _fake_timm(monkeypatch, channels=(4, 8, 16)):
    import timm

    fake = _FakeConvNext(channels)

    def create_model(name, **kwargs):
        fake.recorded = {"name": name, **kwargs}
        return fake

    monkeypatch.setattr(timm, "create_model", create_model)
    return fake


def _small_encoder(monkeypatch, **overrides):
    """Fast encoder with a fake semantic branch and narrow widths."""
    fake = _fake_timm(monkeypatch)
    kwargs = {
        "widths": (8, 16, 32, 64, 128),
        "stem_blocks": 1,
        "stem_dw_kernel_sizes": (3, 5, 5),
        "downsample": "haar",
        "convnext_pretrained": False,
    }
    kwargs.update(overrides)
    return SLBRNextEncoder(**kwargs), fake


# ---------------------------------------------------------------------------
# Haar DWT / IWT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 3, 64, 96), (2, 5, 32, 32)])
def test_haar_iwt_roundtrip_fp32(shape):
    torch.manual_seed(0)
    channels = shape[1]
    x = torch.randn(*shape)
    dwt = HaarDWT(channels)
    iwt = HaarIWT(channels)

    coeffs = dwt(x)

    assert coeffs.shape == (
        shape[0],
        4 * channels,
        shape[2] // 2,
        shape[3] // 2,
    )
    y = iwt(coeffs)

    assert y.shape == x.shape
    error = (y - x).abs().max().item()
    assert error < 1e-5
    torch.testing.assert_close(y, x, rtol=0.0, atol=1e-5)


def test_haar_channel_order_is_interleaved():
    # Two one-hot inputs in the same 2x2 block so every band is distinguishable.
    x = torch.zeros(1, 2, 4, 4)
    x[0, 0, 0, 1] = 1.0  # channel 0, offset (0, 1)
    x[0, 1, 1, 0] = 1.0  # channel 1, offset (1, 0)

    coeffs = HaarDWT(2)(x)

    expected = torch.zeros(1, 8, 2, 2)
    # Channel 0: [LL, LH, HL, HH] -> [0.5, -0.5, 0.5, -0.5]
    expected[0, 0, 0, 0] = 0.5
    expected[0, 1, 0, 0] = -0.5
    expected[0, 2, 0, 0] = 0.5
    expected[0, 3, 0, 0] = -0.5
    # Channel 1: [LL, LH, HL, HH] -> [0.5, 0.5, -0.5, -0.5]
    expected[0, 4, 0, 0] = 0.5
    expected[0, 5, 0, 0] = 0.5
    expected[0, 6, 0, 0] = -0.5
    expected[0, 7, 0, 0] = -0.5

    torch.testing.assert_close(coeffs, expected, rtol=0.0, atol=0.0)


def test_haar_split_merge_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 8, 12)
    coeffs = HaarDWT(3)(x)

    ll, lh, hl, hh = split_haar_bands(coeffs)
    merged = merge_haar_bands(ll, lh, hl, hh)

    assert ll.shape == (1, 3, 4, 6)
    torch.testing.assert_close(merged, coeffs, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        HaarIWT(3)(merged), x, rtol=0.0, atol=1e-5
    )


def test_haar_filters_are_fixed_buffers():
    dwt = HaarDWT(4)
    iwt = HaarIWT(4)

    assert list(dwt.parameters()) == []
    assert list(iwt.parameters()) == []
    assert dwt.weight.shape == (16, 1, 2, 2)
    assert iwt.weight.shape == (16, 1, 2, 2)
    assert dwt.weight.requires_grad is False
    assert iwt.weight.requires_grad is False
    assert [name for name, _ in dwt.named_buffers()] == ["weight"]


def test_haar_rejects_odd_spatial_size():
    with pytest.raises(ValueError):
        HaarDWT(2)(torch.randn(1, 2, 7, 8))
    with pytest.raises(ValueError):
        HaarDWT(2)(torch.randn(1, 2, 8, 9))


def test_haar_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        HaarDWT(3)(torch.randn(1, 2, 8, 8))
    with pytest.raises(ValueError):
        HaarIWT(3)(torch.randn(1, 8, 4, 4))


# ---------------------------------------------------------------------------
# Encoder forward / interface
# ---------------------------------------------------------------------------


def test_encoder_output_interface_is_frozen():
    assert [field.name for field in fields(EncoderOutput)] == [
        "x_h",
        "x_h2",
        "x_h4",
        "x_h8",
        "x_h16",
    ]

    output = EncoderOutput(*(torch.zeros(1) for _ in range(5)))
    with pytest.raises(FrozenInstanceError):
        output.x_h = torch.ones(1)


@pytest.mark.parametrize(
    ("size", "batch"),
    [
        ((256, 256), 1),
        ((256, 512), 1),
        ((256, 256), 2),
    ],
)
def test_forward_shapes(encoder, size, batch):
    height, width = size
    x = torch.rand(batch, 3, height, width)

    with torch.no_grad():
        out = encoder(x)

    assert out.x_h.shape == (batch, 32, height, width)
    assert out.x_h2.shape == (batch, 64, height // 2, width // 2)
    assert out.x_h4.shape == (batch, 128, height // 4, width // 4)
    assert out.x_h8.shape == (batch, 256, height // 8, width // 8)
    assert out.x_h16.shape == (batch, 512, height // 16, width // 16)
    for feature in (out.x_h, out.x_h2, out.x_h4, out.x_h8, out.x_h16):
        assert feature.dtype == torch.float32
        assert torch.isfinite(feature).all()


def test_fusion_combines_stem_and_semantic_branch(monkeypatch):
    encoder, fake = _small_encoder(monkeypatch)
    captured = {}

    def capture(module, inputs, output):
        captured["stem_h4"] = output

    handle = encoder.stem_stages[2].register_forward_hook(capture)
    x = torch.randn(1, 3, 64, 96)
    try:
        with torch.no_grad():
            out = encoder(x)

            branch_h4, branch_h8, branch_h16 = fake.last_outputs
            expected_h4 = encoder.fuse_h4(
                torch.cat((captured["stem_h4"], branch_h4), dim=1)
            )
            branch_only_h4 = encoder.fuse_h4(
                torch.cat(
                    (torch.zeros_like(captured["stem_h4"]), branch_h4), dim=1
                )
            )
    finally:
        handle.remove()

    assert captured["stem_h4"].shape == (1, 32, 16, 24)
    assert out.x_h.shape == (1, 8, 64, 96)
    assert out.x_h2.shape == (1, 16, 32, 48)
    assert out.x_h4.shape == (1, 32, 16, 24)
    assert out.x_h8.shape == (1, 64, 8, 12)
    assert out.x_h16.shape == (1, 128, 4, 6)

    torch.testing.assert_close(out.x_h4, expected_h4, rtol=0, atol=1e-6)
    torch.testing.assert_close(
        out.x_h8, encoder.fuse_h8(branch_h8), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        out.x_h16, encoder.fuse_h16(branch_h16), rtol=0, atol=1e-6
    )
    # The H/4 fusion must include the high-resolution stem, not the branch only.
    assert not torch.allclose(out.x_h4, branch_only_h4)


def test_semantic_branch_input_is_imagenet_normalized(monkeypatch):
    encoder, fake = _small_encoder(monkeypatch, downsample="conv")
    x = torch.full((1, 3, 64, 96), 0.5)

    with torch.no_grad():
        encoder(x)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    expected = (x - mean) / std

    assert fake.last_input is not None
    torch.testing.assert_close(
        fake.last_input, expected, rtol=1e-6, atol=1e-6
    )


# ---------------------------------------------------------------------------
# Downsampling configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["haar", "conv"])
def test_downsample_configuration(monkeypatch, mode):
    encoder, _ = _small_encoder(monkeypatch, downsample=mode)

    if mode == "haar":
        assert isinstance(encoder.down1, HaarDownsample)
        assert isinstance(encoder.down2, HaarDownsample)
        assert isinstance(encoder.down1.dwt, HaarDWT)
        assert encoder.down1.dwt.channels == 8
        assert encoder.down1.project.in_channels == 32
        assert encoder.down1.project.out_channels == 16
        assert encoder.down1.project.kernel_size == (1, 1)
        assert encoder.down2.dwt.channels == 16
        assert encoder.down2.project.in_channels == 64
        assert encoder.down2.project.out_channels == 32
    else:
        assert isinstance(encoder.down1.conv, nn.Conv2d)
        assert encoder.down1.conv.kernel_size == (2, 2)
        assert encoder.down1.conv.stride == (2, 2)
        assert encoder.down2.conv.in_channels == 16
        assert encoder.down2.conv.out_channels == 32

    with torch.no_grad():
        out = encoder(torch.randn(1, 3, 64, 96))

    assert out.x_h2.shape == (1, 16, 32, 48)
    assert out.x_h4.shape == (1, 32, 16, 24)


# ---------------------------------------------------------------------------
# Pretrained branch loading
# ---------------------------------------------------------------------------


def test_pretrained_weights_are_requested(monkeypatch):
    fake = _fake_timm(monkeypatch)

    encoder = SLBRNextEncoder(
        widths=(8, 16, 32, 64, 128),
        stem_blocks=1,
        convnext_pretrained=True,
    )

    assert fake.recorded["name"] == "convnextv2_atto"
    assert fake.recorded["pretrained"] is True
    assert fake.recorded["features_only"] is True
    assert fake.recorded["out_indices"] == (0, 1, 2)
    assert encoder.semantic_branch is fake


def test_pretrained_default_is_true_and_model_is_configurable(monkeypatch):
    fake = _fake_timm(monkeypatch)

    SLBRNextEncoder(
        widths=(8, 16, 32, 64, 128),
        stem_blocks=1,
        convnext_model="convnextv2_nano",
        downsample="conv",
    )

    assert fake.recorded["name"] == "convnextv2_nano"
    assert fake.recorded["pretrained"] is True


@pytest.mark.skipif(
    os.environ.get("SLBR_NEXT_TEST_PRETRAINED") != "1",
    reason=(
        "Set SLBR_NEXT_TEST_PRETRAINED=1 to download the FCMAE weights; "
        "the default CI path never touches the network."
    ),
)
def test_real_pretrained_convnextv2_atto_loads():
    encoder = SLBRNextEncoder(convnext_pretrained=True).eval()

    with torch.no_grad():
        out = encoder(torch.zeros(1, 3, 64, 96))

    assert out.x_h16.shape == (1, 512, 4, 6)
    assert torch.isfinite(out.x_h16).all()


# ---------------------------------------------------------------------------
# Contract guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"widths": (32, 64, 128, 256)},
        {"widths": (32, 64, 128, 256, 0)},
        {"stem_blocks": 0},
        {"stem_dw_kernel_sizes": (4, 5, 5)},
        {"stem_dw_kernel_sizes": (3, 5)},
        {"downsample": "pixel_shuffle"},
        {"convnext_out_indices": (0, 1)},
        {"convnext_out_indices": (0, 0, 1)},
    ],
)
def test_invalid_configuration_rejected(overrides):
    with pytest.raises(ValueError):
        SLBRNextEncoder(convnext_pretrained=False, **overrides)


def test_input_validation(monkeypatch):
    encoder, _ = _small_encoder(monkeypatch)

    with pytest.raises(ValueError):
        encoder(torch.randn(1, 3, 64, 80))  # 80 is not a multiple of 16
    with pytest.raises(ValueError):
        encoder(torch.randn(1, 1, 64, 96))  # not RGB
    with pytest.raises(ValueError):
        encoder(torch.randn(3, 64, 96))  # not 4D


def test_no_batch_or_instance_norm(encoder):
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
        for module in encoder.modules()
        if isinstance(module, forbidden)
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# FP16 (CUDA only; skipped on CPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP16 encoder forward is CUDA-only; CPU CI skips it gracefully.",
)
def test_fp16_forward_cuda():
    model = (
        SLBRNextEncoder(
            widths=(16, 32, 64, 128, 256),
            stem_blocks=1,
            convnext_pretrained=False,
        )
        .cuda()
        .half()
        .eval()
    )
    x = torch.randn(1, 3, 64, 96, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        out = model(x)

    assert out.x_h.dtype == torch.float16
    assert out.x_h16.dtype == torch.float16
    assert out.x_h.shape == (1, 16, 64, 96)
    assert torch.isfinite(out.x_h.float()).all()
    assert torch.isfinite(out.x_h16.float()).all()
