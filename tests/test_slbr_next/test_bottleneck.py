"""Tests for the SLBR-Next disentangled bottleneck (issue #5).

Covers spec §3.4 (MDTA, Haar frequency branch), §1.3 (spatial/numerical
invariants), §3.8 (shape flow) and §6.2 (channel interleaving, FP32 attention,
no BatchNorm/InstanceNorm).

FP16 coverage is CUDA-only and skips gracefully on the CPU CI runner.
"""

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.networks.slbr_next.bottleneck import (
    BottleneckOutput,
    GatedSkipFusion,
    HaarFrequencyBranch,
    MDTA,
    SLBRNextBottleneck,
)
from src.networks.slbr_next.encoder import EncoderOutput
from src.networks.slbr_next.haar import split_haar_bands


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_bottleneck():
    """Fast bottleneck with narrow, deliberately asymmetric trunk widths."""
    torch.manual_seed(0)
    return SLBRNextBottleneck(
        encoder_h8_channels=16,
        encoder_h16_channels=32,
        mask_width=12,
        bg_width=10,
        mask_width_h16=24,
        bg_width_h16=20,
        shared_width=16,
        stem_blocks=1,
        blocks_h8=1,
        blocks_h16=1,
        blocks_h8_post=1,
        mdta_heads=2,
        freq_width=8,
        freq_ll="mdta",
        freq_heads=2,
        freq_kernel_size=3,
        dw_kernel_size=3,
    ).eval()


def _features(size, batch=1, h8_channels=16, h16_channels=32):
    height, width = size
    x_h8 = torch.randn(batch, h8_channels, height // 8, width // 8)
    x_h16 = torch.randn(batch, h16_channels, height // 16, width // 16)
    return x_h8, x_h16


def _forward(bottleneck, size, batch=1):
    x_h8, x_h16 = _features(size, batch)
    with torch.no_grad():
        return bottleneck.forward_features(x_h8, x_h16)


# ---------------------------------------------------------------------------
# Frozen output interface
# ---------------------------------------------------------------------------


def test_bottleneck_output_interface_is_frozen():
    assert [field.name for field in fields(BottleneckOutput)] == [
        "mask_h8",
        "mask_h16",
        "bg_h8",
        "bg_h16",
    ]

    output = BottleneckOutput(*(torch.zeros(1) for _ in range(4)))
    with pytest.raises(FrozenInstanceError):
        output.mask_h8 = torch.ones(1)


# ---------------------------------------------------------------------------
# Forward shapes (256x256, 256x512, B=1/B=2, 512x256)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "batch"),
    [
        ((256, 256), 1),
        ((256, 512), 1),
        ((256, 512), 2),
        ((512, 256), 1),
    ],
)
def test_forward_shapes(tiny_bottleneck, size, batch):
    out = _forward(tiny_bottleneck, size, batch)
    height, width = size

    assert out.mask_h8.shape == (batch, 12, height // 8, width // 8)
    assert out.mask_h16.shape == (batch, 24, height // 16, width // 16)
    assert out.bg_h8.shape == (batch, 10, height // 8, width // 8)
    assert out.bg_h16.shape == (batch, 20, height // 16, width // 16)

    for tensor in (out.mask_h8, out.mask_h16, out.bg_h8, out.bg_h16):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_default_widths_forward_256():
    torch.manual_seed(0)
    bottleneck = SLBRNextBottleneck().eval()

    assert bottleneck.mask_width == 256
    assert bottleneck.bg_width == 256
    assert bottleneck.mask_width_h16 == 512
    assert bottleneck.bg_width_h16 == 512

    x_h8 = torch.randn(1, 256, 32, 32)
    x_h16 = torch.randn(1, 512, 16, 16)
    features = EncoderOutput(
        x_h=torch.empty(0),
        x_h2=torch.empty(0),
        x_h4=torch.empty(0),
        x_h8=x_h8,
        x_h16=x_h16,
    )

    with torch.no_grad():
        out = bottleneck(features)

    assert out.mask_h8.shape == (1, 256, 32, 32)
    assert out.mask_h16.shape == (1, 512, 16, 16)
    assert out.bg_h8.shape == (1, 256, 32, 32)
    assert out.bg_h16.shape == (1, 512, 16, 16)
    for tensor in (out.mask_h8, out.mask_h16, out.bg_h8, out.bg_h16):
        assert torch.isfinite(tensor).all()

    total = sum(parameter.numel() for parameter in bottleneck.parameters())
    print(f"SLBRNextBottleneck(default): {total:,} parameters")
    assert total > 0


def test_forward_rejects_raw_tensors_and_bad_shapes(tiny_bottleneck):
    with pytest.raises(TypeError):
        tiny_bottleneck(torch.randn(1, 16, 32, 32))

    with pytest.raises(ValueError):
        tiny_bottleneck.forward_features(
            torch.randn(1, 16, 31, 32), torch.randn(1, 32, 15, 16)
        )
    with pytest.raises(ValueError):
        tiny_bottleneck.forward_features(
            torch.randn(1, 8, 32, 32), torch.randn(1, 32, 16, 16)
        )
    with pytest.raises(ValueError):
        tiny_bottleneck.forward_features(
            torch.randn(1, 16, 32, 32), torch.randn(1, 16, 16, 16)
        )
    with pytest.raises(ValueError):
        tiny_bottleneck.forward_features(
            torch.randn(16, 32, 32), torch.randn(1, 32, 16, 16)
        )


# ---------------------------------------------------------------------------
# MDTA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "batch"),
    [((16, 32), 2), ((7, 11), 1), ((32, 16), 2)],
)
def test_mdta_fp32_finite_non_square(size, batch):
    torch.manual_seed(0)
    mdta = MDTA(12, 3).eval()
    x = torch.randn(batch, 12, *size)

    with torch.no_grad():
        out = mdta(x)

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_mdta_matches_reference_formula():
    torch.manual_seed(0)
    mdta = MDTA(8, 2).eval()
    with torch.no_grad():
        mdta.temperature.copy_(
            torch.tensor([0.7, 1.3]).view(2, 1, 1)
        )

    x = torch.randn(2, 8, 5, 7)

    with torch.no_grad():
        out = mdta(x)

        qkv = mdta.qkv_dw(mdta.qkv(x))
        query, key, value = qkv.chunk(3, dim=1)
        query = query.reshape(2, 2, 4, -1)
        key = key.reshape(2, 2, 4, -1)
        value = value.reshape(2, 2, 4, -1)

        # L2-normalize over the spatial axis (H * W), as in Restormer.
        query_f = F.normalize(query.float(), dim=-1)
        key_f = F.normalize(key.float(), dim=-1)
        attn = (query_f @ key_f.transpose(-2, -1)) * (
            mdta.temperature.float()
        )
        attn = attn.softmax(dim=-1)
        expected = (attn @ value.float()).to(x.dtype)
        expected = mdta.proj(expected.reshape(2, 8, 5, 7))

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(out).all()


def test_mdta_backward_is_finite():
    torch.manual_seed(0)
    mdta = MDTA(16, 4)
    x = torch.randn(2, 16, 8, 12, requires_grad=True)

    out = mdta(x)
    out.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in mdta.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all()


def test_mdta_configuration_validation():
    with pytest.raises(ValueError):
        MDTA(10, 3)
    with pytest.raises(ValueError):
        MDTA(0, 1)
    with pytest.raises(ValueError):
        MDTA(8, 0)
    with pytest.raises(ValueError):
        MDTA(8, 2)(torch.randn(1, 4, 8, 8))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP16 MDTA is CUDA-only; the CPU CI runner skips it gracefully.",
)
def test_mdta_fp16_cuda():
    mdta = MDTA(32, 4).cuda().half().eval()
    x = torch.randn(
        2, 32, 512, 512, device="cuda", dtype=torch.float16
    )

    with torch.no_grad():
        out = mdta(x)

    assert out.shape == x.shape
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()


# ---------------------------------------------------------------------------
# Haar frequency branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ll_mode", ["mdta", "naf"])
def test_frequency_branch_forward_shapes(ll_mode):
    torch.manual_seed(0)
    branch = HaarFrequencyBranch(
        12,
        freq_width=8,
        ll_mode=ll_mode,
        heads=2,
        kernel_size=3,
    ).eval()
    x = torch.randn(2, 12, 24, 40)

    with torch.no_grad():
        out = branch(x)
        coeffs = branch.dwt(branch.pre(x))

    # DWT halves the spatial size and interleaves four bands per channel.
    assert coeffs.shape == (2, 4 * 8, 12, 20)
    ll, lh, hl, hh = split_haar_bands(coeffs)
    for band in (ll, lh, hl, hh):
        assert band.shape == (2, 8, 12, 20)

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_frequency_branch_dwt_channel_interleaving():
    # One-hot inputs in the same 2x2 block make every band distinguishable.
    branch = HaarFrequencyBranch(
        4,
        freq_width=2,
        ll_mode="naf",
        heads=1,
    ).eval()
    x = torch.zeros(1, 2, 4, 4)
    x[0, 0, 0, 1] = 1.0  # channel 0, offset (0, 1)
    x[0, 1, 1, 0] = 1.0  # channel 1, offset (1, 0)

    with torch.no_grad():
        coeffs = branch.dwt(x)

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

    assert coeffs.shape == expected.shape
    torch.testing.assert_close(coeffs, expected, rtol=0.0, atol=0.0)


class _IdentityHighFrequency(nn.Module):
    def forward(self, lh, hl, hh):
        return lh, hl, hh


def test_frequency_branch_uses_split_merge_interleaving():
    # If the branch split coefficients with ``chunk(4, dim=1)`` instead of the
    # A2 helpers, this identity-through-the-middle test would mix the bands and
    # fail the Haar perfect-reconstruction check.
    torch.manual_seed(0)
    branch = HaarFrequencyBranch(
        4,
        freq_width=4,
        ll_mode="naf",
        heads=2,
        kernel_size=3,
    ).eval()

    with torch.no_grad():
        for conv in (branch.pre, branch.post):
            conv.weight.copy_(torch.eye(4).view(4, 4, 1, 1))
            conv.bias.zero_()
        branch.ll_block = nn.Identity()
        branch.high_frequency = _IdentityHighFrequency()

    x = torch.randn(2, 4, 8, 12)
    with torch.no_grad():
        out = branch(x)

    assert out.shape == x.shape
    torch.testing.assert_close(out, x, rtol=0.0, atol=1e-5)


def test_frequency_branch_can_be_disabled():
    torch.manual_seed(0)
    bottleneck = SLBRNextBottleneck(
        encoder_h8_channels=8,
        encoder_h16_channels=16,
        mask_width=8,
        bg_width=8,
        mask_width_h16=16,
        bg_width_h16=16,
        shared_width=8,
        mdta_heads=2,
        frequency_branch=False,
        freq_width=4,
        freq_heads=2,
        dw_kernel_size=3,
    ).eval()

    assert isinstance(bottleneck.freq_branch, nn.Identity)
    x_h8 = torch.randn(1, 8, 8, 12)
    x_h16 = torch.randn(1, 16, 4, 6)
    with torch.no_grad():
        out = bottleneck.forward_features(x_h8, x_h16)

    assert out.mask_h8.shape == (1, 8, 8, 12)
    assert out.bg_h16.shape == (1, 16, 4, 6)


# ---------------------------------------------------------------------------
# Gated skip fusion
# ---------------------------------------------------------------------------


def test_gated_skip_fusion_math():
    torch.manual_seed(0)
    fusion = GatedSkipFusion(6, 4).eval()
    x = torch.randn(2, 4, 8, 10)
    skip = torch.randn(2, 6, 8, 10)

    with torch.no_grad():
        out = fusion(x, skip)
        projected = fusion.skip_proj(skip)
        gate = torch.sigmoid(fusion.gate(projected))

    assert out.shape == x.shape
    torch.testing.assert_close(out, x + gate * projected)
    assert bool((gate > 0).all())
    assert bool((gate < 1).all())


def test_gated_skip_fusion_zero_projection_is_identity():
    fusion = GatedSkipFusion(4, 4)
    with torch.no_grad():
        fusion.skip_proj.weight.zero_()
        fusion.skip_proj.bias.zero_()

    x = torch.randn(2, 4, 8, 8)
    skip = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        out = fusion(x, skip)

    torch.testing.assert_close(out, x, rtol=0.0, atol=0.0)


def test_gated_skip_fusion_resizes_to_trunk_size():
    fusion = GatedSkipFusion(3, 5).eval()
    x = torch.randn(1, 5, 4, 6)
    skip = torch.randn(1, 3, 16, 24)

    with torch.no_grad():
        out = fusion(x, skip)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_encoder_skips_reach_both_trunks(tiny_bottleneck):
    torch.manual_seed(0)
    x_h8 = torch.randn(1, 16, 32, 32)
    x_h16 = torch.randn(1, 32, 16, 16)

    with torch.no_grad():
        base = tiny_bottleneck.forward_features(x_h8, x_h16)

        changed_h8 = x_h8.clone()
        changed_h8[0, 0, 0, 0] += 2.0
        from_h8 = tiny_bottleneck.forward_features(changed_h8, x_h16)

        changed_h16 = x_h16.clone()
        changed_h16[0, 0, 0, 0] += 2.0
        from_h16 = tiny_bottleneck.forward_features(x_h8, changed_h16)

    assert not torch.allclose(base.mask_h8, from_h8.mask_h8)
    assert not torch.allclose(base.bg_h8, from_h8.bg_h8)
    assert not torch.allclose(base.mask_h16, from_h16.mask_h16)
    assert not torch.allclose(base.bg_h16, from_h16.bg_h16)


def test_frequency_branch_runs_once_before_the_split(tiny_bottleneck):
    order = []
    handles = [
        tiny_bottleneck.freq_branch.register_forward_hook(
            lambda *_: order.append("freq")
        ),
        tiny_bottleneck.mask_trunk.register_forward_hook(
            lambda *_: order.append("mask")
        ),
        tiny_bottleneck.bg_trunk.register_forward_hook(
            lambda *_: order.append("bg")
        ),
    ]
    try:
        with torch.no_grad():
            tiny_bottleneck.forward_features(
                torch.randn(1, 16, 32, 48),
                torch.randn(1, 32, 16, 24),
            )
    finally:
        for handle in handles:
            handle.remove()

    assert order.count("freq") == 1
    assert order[0] == "freq"
    assert set(order[1:]) == {"mask", "bg"}


# ---------------------------------------------------------------------------
# Spatial invariants and architecture guards
# ---------------------------------------------------------------------------


def _autocast_cpu_enabled():
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:  # pragma: no cover - older torch signature
        return torch.is_autocast_cpu_enabled()


def test_h8_stem_fp32_under_bf16_autocast_keeps_grads_finite(monkeypatch):
    """Regression guard for the 2xT4 Stage 2 FP16 backward overflow.

    The smoke reports non-finite unscaled gradients first in
    ``bottleneck.h8_stem.0.bias`` and the NAF block's
    ``project_conv``/``mixer_expand`` 1x1 convolutions while the loss is
    finite. The shared stem receives the summed mask-trunk, background-trunk
    and frequency-branch gradient, so its FP16 weight/bias-gradient
    accumulation over ``B * H/8 * W/8`` samples overflows under GradScaler's
    initial 65536 scale.

    CPU BF16 has FP32's exponent range and cannot reproduce an FP16 range
    overflow by itself; as in the A2/A4 regressions, the conv spy is the
    discriminator. Pre-fix the stem executes inside the autocast region
    (``autocast_enabled=True``); post-fix every stem convolution must see
    FP32 input/weight with autocast disabled. The large synthetic loss is
    scaled by the same 65536 factor and still requires finite gradients for
    the reported parameters.
    """
    torch.manual_seed(0)
    bottleneck = SLBRNextBottleneck(
        encoder_h8_channels=256,
        encoder_h16_channels=64,
        mask_width=32,
        bg_width=32,
        mask_width_h16=64,
        bg_width_h16=64,
        shared_width=256,
        stem_blocks=1,
        blocks_h8=1,
        blocks_h16=1,
        blocks_h8_post=1,
        mdta_heads=2,
        freq_width=32,
        freq_heads=2,
        freq_kernel_size=3,
        dw_kernel_size=3,
    ).train()

    x_h8 = (torch.randn(2, 256, 32, 32) * 8.0).requires_grad_(True)
    x_h16 = (torch.randn(2, 64, 16, 16) * 8.0).requires_grad_(True)

    calls = []
    stem_active = {"value": False}
    real_conv2d = F.conv2d

    def checked_conv2d(input, weight, *args, **kwargs):
        if stem_active["value"]:
            calls.append(
                (
                    input.dtype,
                    weight.dtype,
                    _autocast_cpu_enabled(),
                )
            )
        return real_conv2d(input, weight, *args, **kwargs)

    monkeypatch.setattr(F, "conv2d", checked_conv2d)
    handles = [
        bottleneck.h8_stem.register_forward_pre_hook(
            lambda *_: stem_active.__setitem__("value", True)
        ),
        bottleneck.h8_stem.register_forward_hook(
            lambda *_: stem_active.__setitem__("value", False)
        ),
    ]
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = bottleneck.forward_features(x_h8, x_h16)
            loss = 65536.0 * (
                output.mask_h8.float().square().mean()
                + output.mask_h16.float().square().mean()
                + output.bg_h8.float().square().mean()
                + output.bg_h16.float().square().mean()
            )
        assert torch.isfinite(loss)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    assert calls
    for input_dtype, weight_dtype, autocast_enabled in calls:
        assert input_dtype == torch.float32
        assert weight_dtype == torch.float32
        assert autocast_enabled is False

    parameters = dict(bottleneck.named_parameters())
    for name in (
        "h8_stem.0.bias",
        "h8_stem.1.project_conv.weight",
        "h8_stem.1.project_conv.bias",
        "h8_stem.1.mixer_expand.weight",
    ):
        parameter = parameters[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    for name, parameter in bottleneck.h8_stem.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    for tensor in (x_h8, x_h16):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_remaining_bottleneck_paths_fp32_under_bf16_autocast_keep_grads_finite(
    monkeypatch,
):
    """Regression guard for the step-2 frequency/trunk FP16 overflow.

    After the shared-stem guard, the 2xT4 smoke reports non-finite unscaled
    gradients in ``bottleneck.freq_branch.pre.bias``, ``post.bias`` and
    ``mask_trunk.blocks_h8.0.project_conv.bias`` at step 2. The residual
    frequency branch and both trunk stages still executed inside FP16
    autocast, so their 1x1 projections and NAF blocks accumulated scaled
    gradients beyond the FP16 range.

    The whole low-resolution bottleneck now runs in FP32 with autocast
    disabled. The conv spy fails on the step-1-only code, where the
    frequency/trunk convolutions still see autocast enabled, and the large
    65536x synthetic loss asserts finite gradients for every remaining
    frequency and trunk parameter.
    """
    torch.manual_seed(0)
    bottleneck = SLBRNextBottleneck(
        encoder_h8_channels=256,
        encoder_h16_channels=128,
        mask_width=128,
        bg_width=128,
        mask_width_h16=64,
        bg_width_h16=64,
        shared_width=256,
        stem_blocks=1,
        blocks_h8=1,
        blocks_h16=1,
        blocks_h8_post=1,
        mdta_heads=4,
        freq_width=128,
        freq_heads=2,
        freq_kernel_size=3,
        dw_kernel_size=5,
    ).train()

    x_h8 = (torch.randn(2, 256, 32, 32) * 8.0).requires_grad_(True)
    x_h16 = (torch.randn(2, 128, 16, 16) * 8.0).requires_grad_(True)

    calls = []
    real_conv2d = F.conv2d

    def checked_conv2d(input, weight, *args, **kwargs):
        calls.append(
            (input.dtype, weight.dtype, _autocast_cpu_enabled())
        )
        return real_conv2d(input, weight, *args, **kwargs)

    monkeypatch.setattr(F, "conv2d", checked_conv2d)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = bottleneck.forward_features(x_h8, x_h16)
        loss = 65536.0 * (
            output.mask_h8.float().square().mean()
            + output.mask_h16.float().square().mean()
            + output.bg_h8.float().square().mean()
            + output.bg_h16.float().square().mean()
        )
    assert torch.isfinite(loss)
    loss.backward()

    for tensor in (
        output.mask_h8,
        output.mask_h16,
        output.bg_h8,
        output.bg_h16,
    ):
        assert tensor.dtype == x_h8.dtype

    assert calls
    for input_dtype, weight_dtype, autocast_enabled in calls:
        assert input_dtype == torch.float32
        assert weight_dtype == torch.float32
        assert autocast_enabled is False

    parameters = dict(bottleneck.named_parameters())
    for name in (
        "freq_branch.pre.bias",
        "freq_branch.post.bias",
        "mask_trunk.blocks_h8.0.project_conv.bias",
    ):
        parameter = parameters[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    for prefix in ("freq_branch.", "mask_trunk.", "bg_trunk."):
        names = [name for name in parameters if name.startswith(prefix)]
        assert names, prefix
        for name in names:
            parameter = parameters[name]
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_interpolation_targets_reference_sizes(monkeypatch, tiny_bottleneck):
    recorded = []
    real_interpolate = F.interpolate

    def recording_interpolate(x, size=None, **kwargs):
        recorded.append(tuple(size) if size is not None else None)
        return real_interpolate(x, size=size, **kwargs)

    monkeypatch.setattr(
        "src.networks.slbr_next.bottleneck.F.interpolate",
        recording_interpolate,
    )

    with torch.no_grad():
        tiny_bottleneck.forward_features(
            torch.randn(1, 16, 32, 64),
            torch.randn(1, 32, 16, 32),
        )

    # Two trunks, each with one H/8 -> H/16 downsample and one
    # H/16 -> H/8 upsample. No reversed (W, H) and no scale_factor.
    assert len(recorded) == 4
    assert None not in recorded
    assert set(recorded) == {(16, 32), (32, 64)}


def test_mask_and_bg_trunks_do_not_share_parameters(tiny_bottleneck):
    assert tiny_bottleneck.mask_trunk is not tiny_bottleneck.bg_trunk

    mask_ids = {
        id(parameter)
        for parameter in tiny_bottleneck.mask_trunk.parameters()
    }
    bg_ids = {
        id(parameter) for parameter in tiny_bottleneck.bg_trunk.parameters()
    }
    assert mask_ids.isdisjoint(bg_ids)


def test_no_batch_or_instance_norm(tiny_bottleneck):
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
        for module in tiny_bottleneck.modules()
        if isinstance(module, forbidden)
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"encoder_h8_channels": 0},
        {"encoder_h16_channels": 0},
        {"mask_width": 0},
        {"bg_width": -2},
        {"mask_width_h16": 0},
        {"bg_width_h16": 0},
        {"shared_width": 0},
        {"stem_blocks": -1},
        {"blocks_h8": -1},
        {"blocks_h16": -1},
        {"blocks_h8_post": -1},
        {"mdta_heads": 5},
        {"mdta_heads": 0},
        {"freq_width": 0},
        {"freq_heads": 0},
        {"freq_width": 6, "freq_heads": 4},
        {"freq_ll": "fft"},
        {"freq_kernel_size": 4},
        {"dw_kernel_size": 4},
    ],
)
def test_invalid_configuration_rejected(overrides):
    kwargs = {
        "encoder_h8_channels": 8,
        "encoder_h16_channels": 16,
        "mask_width": 8,
        "bg_width": 8,
        "mask_width_h16": 16,
        "bg_width_h16": 16,
        "shared_width": 8,
        "mdta_heads": 2,
        "freq_width": 8,
        "freq_heads": 2,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        SLBRNextBottleneck(**kwargs)
