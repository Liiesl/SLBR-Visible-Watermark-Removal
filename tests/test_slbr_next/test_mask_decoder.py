"""Tests for the SLBR-Next K-query mask decoder (issue #6).

Covers spec §3.5 (K + 1 queries, masked cross-attention at H/8 and H/4,
logsumexp logits, multi-scale masks), §3.8 (shape flow) and §1.3
(spatial/numerical invariants).

FP16 coverage is CUDA-only and skips gracefully on the CPU CI runner.
"""

from dataclasses import FrozenInstanceError, fields
from math import log

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.networks.slbr_next.mask_decoder import (
    MaskDecoder,
    MaskDecoderOutput,
    MaskedCrossAttention,
    PartScoreHead,
)


TINY_WIDTHS = (16, 12, 8, 6)
TINY_SKIPS = (12, 8, 6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_decoder(**overrides):
    """Fast decoder with narrow, deliberately non-power-of-two widths."""
    torch.manual_seed(0)
    kwargs = {
        "in_channels": 16,
        "widths": TINY_WIDTHS,
        "skip_channels": TINY_SKIPS,
        "heads": 2,
        "blocks_per_level": (1, 1, 1, 1),
        "dw_kernel_sizes": (3, 3, 3, 3),
    }
    kwargs.update(overrides)
    return MaskDecoder(**kwargs).eval()


def _inputs(size, batch=1, in_channels=16, skip_channels=TINY_SKIPS):
    height, width = size
    return (
        torch.randn(batch, in_channels, height // 8, width // 8),
        torch.randn(batch, skip_channels[0], height // 4, width // 4),
        torch.randn(batch, skip_channels[1], height // 2, width // 2),
        torch.randn(batch, skip_channels[2], height, width),
    )


def _forward(decoder, size, batch=1, with_skips=True):
    mask_h8, skip_h4, skip_h2, skip_h = _inputs(
        size,
        batch,
        in_channels=decoder.in_channels,
        skip_channels=decoder.skip_channels,
    )
    with torch.no_grad():
        if with_skips:
            return decoder(
                mask_h8,
                skip_h4=skip_h4,
                skip_h2=skip_h2,
                skip_h=skip_h,
            )
        return decoder(mask_h8)


# ---------------------------------------------------------------------------
# Frozen output interface
# ---------------------------------------------------------------------------


def test_mask_decoder_output_interface_is_frozen():
    assert [field.name for field in fields(MaskDecoderOutput)] == [
        "mask_logits",
        "mask_logits_h2",
        "mask_logits_h4",
        "mask_feat",
        "part_logits",
    ]

    output = MaskDecoderOutput(*(torch.zeros(1) for _ in range(5)))
    with pytest.raises(FrozenInstanceError):
        output.mask_feat = torch.ones(1)


# ---------------------------------------------------------------------------
# Forward shapes: 256x256, 256x512, 512x256, B=1/B=2
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
def test_forward_shapes_with_skips(size, batch):
    decoder = _tiny_decoder()
    out = _forward(decoder, size, batch)

    height, width = size
    assert out.mask_logits.shape == (batch, 1, height, width)
    assert out.mask_logits_h2.shape == (
        batch,
        1,
        height // 2,
        width // 2,
    )
    assert out.mask_logits_h4.shape == (
        batch,
        1,
        height // 4,
        width // 4,
    )
    assert out.mask_feat.shape == (batch, TINY_WIDTHS[3], height, width)
    assert out.part_logits.shape == (
        batch,
        decoder.num_queries,
        height,
        width,
    )

    for tensor in (
        out.mask_logits,
        out.mask_logits_h2,
        out.mask_logits_h4,
        out.mask_feat,
        out.part_logits,
    ):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_forward_without_skips_derives_exact_sizes():
    decoder = _tiny_decoder()
    out = _forward(decoder, (256, 512), with_skips=False)

    assert out.mask_logits.shape == (1, 1, 256, 512)
    assert out.mask_logits_h2.shape == (1, 1, 128, 256)
    assert out.mask_logits_h4.shape == (1, 1, 64, 128)
    assert out.mask_feat.shape == (1, TINY_WIDTHS[3], 256, 512)
    assert out.part_logits.shape == (1, 9, 256, 512)
    for tensor in (out.mask_logits, out.part_logits):
        assert torch.isfinite(tensor).all()


def test_widths_are_configurable():
    decoder = _tiny_decoder(
        widths=(16, 10, 8, 4), skip_channels=(10, 8, 4)
    )
    out = _forward(decoder, (256, 256))

    assert out.mask_feat.shape == (1, 4, 256, 256)
    assert out.mask_logits.shape == (1, 1, 256, 256)


def test_default_widths_forward_and_parameter_count():
    torch.manual_seed(0)
    decoder = MaskDecoder().eval()

    assert decoder.widths == (256, 128, 64, 32)
    assert decoder.query_dim == 256
    assert decoder.num_parts == 8
    assert decoder.queries.shape == (1, 9, 256)

    mask_h8 = torch.randn(1, 256, 8, 8)
    skip_h4 = torch.randn(1, 128, 16, 16)
    skip_h2 = torch.randn(1, 64, 32, 32)
    skip_h = torch.randn(1, 32, 64, 64)
    with torch.no_grad():
        out = decoder(
            mask_h8,
            skip_h4=skip_h4,
            skip_h2=skip_h2,
            skip_h=skip_h,
        )

    assert out.mask_logits.shape == (1, 1, 64, 64)
    assert out.mask_logits_h2.shape == (1, 1, 32, 32)
    assert out.mask_logits_h4.shape == (1, 1, 16, 16)
    assert out.mask_feat.shape == (1, 32, 64, 64)
    assert out.part_logits.shape == (1, 9, 64, 64)
    for tensor in (out.mask_logits, out.part_logits):
        assert torch.isfinite(tensor).all()

    total = sum(parameter.numel() for parameter in decoder.parameters())
    print(f"MaskDecoder(default): {total:,} parameters")
    assert total > 0


# ---------------------------------------------------------------------------
# Queries and raw part logits
# ---------------------------------------------------------------------------


def test_query_and_part_shapes():
    decoder = _tiny_decoder()
    assert decoder.num_parts == 8
    assert decoder.num_queries == 9
    assert decoder.query_dim == TINY_WIDTHS[0]
    assert decoder.queries.shape == (1, 9, TINY_WIDTHS[0])
    assert decoder.queries.requires_grad

    out = _forward(decoder, (256, 256))
    assert out.part_logits.shape == (1, 9, 256, 256)

    small = _tiny_decoder(num_parts=3)
    small_out = _forward(small, (256, 256))
    assert small.queries.shape == (1, 4, TINY_WIDTHS[0])
    assert small_out.part_logits.shape == (1, 4, 256, 256)


def test_sigmoid_mask_logits_equals_one_minus_background_probability():
    decoder = _tiny_decoder()
    out = _forward(decoder, (256, 512))

    part_logits = out.part_logits.float()
    probabilities = torch.softmax(part_logits, dim=1)

    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        rtol=0.0,
        atol=1e-5,
    )
    torch.testing.assert_close(
        torch.sigmoid(out.mask_logits.float()),
        1.0 - probabilities[:, -1:],
        rtol=1e-5,
        atol=1e-5,
    )

    expected = (
        torch.logsumexp(part_logits[:, :-1], dim=1, keepdim=True)
        - part_logits[:, -1:]
    )
    torch.testing.assert_close(
        out.mask_logits.float(), expected, rtol=0.0, atol=1e-6
    )
    assert torch.isfinite(out.mask_logits).all()
    assert torch.isfinite(out.mask_logits_h2).all()
    assert torch.isfinite(out.mask_logits_h4).all()


def test_part_diversity_auxiliary_is_finite_and_differentiable():
    torch.manual_seed(0)
    decoder = _tiny_decoder()
    mask_h8, skip_h4, skip_h2, skip_h = _inputs((128, 128), batch=2)

    out = decoder(
        mask_h8, skip_h4=skip_h4, skip_h2=skip_h2, skip_h=skip_h
    )

    # Raw K part logits -> part probabilities. The diversity auxiliary uses
    # a low-entropy term plus a pairwise-overlap penalty, as in the ICCV
    # paper referenced by spec §3.5.
    part_logits = out.part_logits[:, : decoder.num_parts]
    flat = part_logits.flatten(2).float()
    probabilities = torch.softmax(flat, dim=1)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=1).mean()
    normalized = F.normalize(flat, dim=-1)
    gram = torch.matmul(normalized, normalized.transpose(1, 2))
    eye = torch.eye(decoder.num_parts, dtype=flat.dtype)
    overlap = (gram - eye).abs().mean()
    auxiliary = entropy + overlap

    assert torch.isfinite(auxiliary)
    auxiliary.backward()
    assert decoder.queries.grad is not None
    assert torch.isfinite(decoder.queries.grad).all()
    assert decoder.queries.grad.abs().sum() > 0


def test_mask_logits_backward_reaches_all_parameters():
    torch.manual_seed(0)
    decoder = _tiny_decoder()
    mask_h8, skip_h4, skip_h2, skip_h = _inputs((128, 128), batch=1)

    out = decoder(
        mask_h8, skip_h4=skip_h4, skip_h2=skip_h2, skip_h=skip_h
    )
    loss = (
        out.mask_logits.mean()
        + out.mask_logits_h2.mean()
        + out.mask_logits_h4.mean()
    )
    loss.backward()

    for name, parameter in decoder.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP16 mask decoder is CUDA-only; the CPU CI runner skips it.",
)
def test_forward_fp16_cuda():
    decoder = _tiny_decoder().cuda().half().eval()
    mask_h8, skip_h4, skip_h2, skip_h = (
        tensor.cuda().half() for tensor in _inputs((256, 256), batch=2)
    )

    with torch.no_grad():
        out = decoder(
            mask_h8, skip_h4=skip_h4, skip_h2=skip_h2, skip_h=skip_h
        )

    assert out.mask_logits.dtype == torch.float16
    assert out.mask_logits_h2.dtype == torch.float16
    assert out.mask_logits_h4.dtype == torch.float16
    assert out.part_logits.dtype == torch.float16
    for tensor in (
        out.mask_logits,
        out.mask_logits_h2,
        out.mask_logits_h4,
        out.mask_feat,
        out.part_logits,
    ):
        assert torch.isfinite(tensor.float()).all()


# ---------------------------------------------------------------------------
# Attention mechanics
# ---------------------------------------------------------------------------


def test_masked_cross_attention_updates_both_streams():
    torch.manual_seed(0)
    attention = MaskedCrossAttention(
        channels=12, query_dim=16, num_queries=5, heads=4
    ).eval()
    features = torch.randn(2, 12, 8, 10)
    queries = torch.randn(2, 5, 16)

    with torch.no_grad():
        updated_features, updated_queries, part_logits = attention(
            features, queries
        )

    assert updated_features.shape == features.shape
    assert updated_queries.shape == queries.shape
    assert part_logits.shape == (2, 5, 8, 10)
    assert torch.isfinite(updated_features).all()
    assert torch.isfinite(updated_queries).all()
    assert torch.isfinite(part_logits).all()
    assert not torch.allclose(updated_features, features)
    assert not torch.allclose(updated_queries, queries)

    probabilities = torch.softmax(part_logits.flatten(2), dim=1)
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(2, 8, 10),
        rtol=0.0,
        atol=1e-5,
    )


def test_coarse_mask_bias_targets_foreground_and_background_queries():
    attention = MaskedCrossAttention(
        channels=8, query_dim=8, num_queries=5, heads=2
    )
    foreground_probability = torch.full((1, 1, 2, 3), 0.25)
    bias = attention.coarse_mask_bias(foreground_probability)

    assert bias.shape == (1, 1, 5, 6)
    foreground_bias = torch.full_like(bias[:, :, :4], log(0.25))
    background_bias = torch.full_like(bias[:, :, 4:], log(0.75))
    torch.testing.assert_close(bias[:, :, :4], foreground_bias)
    torch.testing.assert_close(bias[:, :, 4:], background_bias)

    # Saturated probabilities must stay finite before the softmax.
    saturated = attention.coarse_mask_bias(torch.zeros(1, 1, 2, 3))
    assert torch.isfinite(saturated).all()


def test_part_score_head_shapes():
    head = PartScoreHead(channels=10, query_dim=16).eval()
    features = torch.randn(2, 10, 8, 12)
    queries = torch.randn(2, 5, 16)

    with torch.no_grad():
        scores = head(features, queries)

    assert scores.shape == (2, 5, 8, 12)
    assert torch.isfinite(scores).all()


def test_cross_attention_updates_queries():
    decoder = _tiny_decoder()
    mask_h8, skip_h4, skip_h2, skip_h = _inputs((256, 256))
    captured = []
    handle = decoder.attention_h8.register_forward_hook(
        lambda _module, _inputs_, output: captured.append(
            output[1].detach().clone()
        )
    )
    try:
        with torch.no_grad():
            decoder(
                mask_h8,
                skip_h4=skip_h4,
                skip_h2=skip_h2,
                skip_h=skip_h,
            )
    finally:
        handle.remove()

    assert len(captured) == 1
    assert captured[0].shape == (1, 9, TINY_WIDTHS[0])
    assert (captured[0] - decoder.queries).abs().max() > 1e-6


def test_updated_h8_features_flow_up_to_the_mask():
    decoder = _tiny_decoder()
    mask_h8, skip_h4, skip_h2, skip_h = _inputs((256, 256))
    captured = []
    handle = decoder.attention_h4.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            first = decoder(
                mask_h8,
                skip_h4=skip_h4,
                skip_h2=skip_h2,
                skip_h=skip_h,
            )
            decoder.attention_h8.spatial_proj.weight.zero_()
            decoder.attention_h8.spatial_proj.bias.zero_()
            second = decoder(
                mask_h8,
                skip_h4=skip_h4,
                skip_h2=skip_h2,
                skip_h=skip_h,
            )
    finally:
        handle.remove()

    assert len(captured) == 2
    assert (captured[0] - captured[1]).abs().max() > 1e-6
    assert (first.mask_logits - second.mask_logits).abs().max() > 1e-6


# ---------------------------------------------------------------------------
# Spatial invariants and architecture guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_skips", [False, True])
def test_interpolation_targets_are_exact_and_not_reversed(
    monkeypatch, with_skips
):
    decoder = _tiny_decoder()
    recorded = []
    real_interpolate = F.interpolate

    def recording_interpolate(x, size=None, **kwargs):
        recorded.append(
            (
                tuple(size) if size is not None else None,
                kwargs.get("mode"),
                kwargs.get("align_corners"),
                kwargs.get("scale_factor"),
            )
        )
        return real_interpolate(x, size=size, **kwargs)

    monkeypatch.setattr(
        "src.networks.slbr_next.mask_decoder.F.interpolate",
        recording_interpolate,
    )

    _forward(decoder, (256, 512), with_skips=with_skips)

    assert [entry[0] for entry in recorded] == [
        (64, 128),
        (128, 256),
        (256, 512),
    ]
    for size, mode, align_corners, scale_factor in recorded:
        assert size is not None
        assert mode == "bilinear"
        assert align_corners is False
        assert scale_factor is None


def test_no_batch_or_instance_norm():
    decoder = _tiny_decoder()
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
        for module in decoder.modules()
        if isinstance(module, forbidden)
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# Constructor and input guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"in_channels": 0},
        {"widths": (16, 12, 8)},
        {"widths": (16, 0, 8, 6)},
        {"widths": (-16, 12, 8, 6)},
        {"skip_channels": (12, 8)},
        {"skip_channels": (12, 0, 6)},
        {"num_parts": 0},
        {"heads": 0},
        {"heads": 3},
        {"blocks_per_level": (1, 1, 1)},
        {"blocks_per_level": (-1, 1, 1, 1)},
        {"dw_kernel_sizes": (3, 3, 3)},
        {"dw_kernel_sizes": (3, 3, 3, 4)},
    ],
)
def test_invalid_configuration_rejected(overrides):
    kwargs = {
        "in_channels": 16,
        "widths": TINY_WIDTHS,
        "skip_channels": TINY_SKIPS,
        "num_parts": 8,
        "heads": 2,
        "blocks_per_level": (1, 1, 1, 1),
        "dw_kernel_sizes": (3, 3, 3, 3),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        MaskDecoder(**kwargs)


def test_forward_rejects_bad_inputs_and_skips():
    decoder = _tiny_decoder()
    mask_h8, skip_h4, skip_h2, skip_h = _inputs((256, 256))

    with pytest.raises(ValueError):
        decoder(torch.randn(1, 16, 32))
    with pytest.raises(ValueError):
        decoder(torch.randn(1, 15, 32, 32))

    with pytest.raises(ValueError):
        decoder(mask_h8, skip_h4=torch.randn(1, 11, 64, 64))
    with pytest.raises(ValueError):
        decoder(mask_h8, skip_h4=torch.randn(1, 12, 33, 32))
    with pytest.raises(ValueError):
        decoder(mask_h8, skip_h2=torch.randn(1, 8, 64, 64))
    with pytest.raises(ValueError):
        decoder(mask_h8, skip_h=torch.randn(1, 6, 128, 64))
    with pytest.raises(ValueError):
        decoder(
            mask_h8,
            skip_h4=skip_h4,
            skip_h2=skip_h2,
            skip_h=torch.randn(1, 6, 256, 256, 1),
        )
