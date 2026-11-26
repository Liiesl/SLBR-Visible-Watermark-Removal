"""Tests for the SLBR-Next background decoder and refinement (issue #7).

Covers spec §3.6 (FiLM background decoder), §3.7 (Stage 2 refinement and v1
composite), §3.8 (shape flow), §5.4 (slim refinement, MDTA at H/8) and §1.3
(no BatchNorm/InstanceNorm, no shape reversals).

Shapes are exercised at 256x256 and 256x512 with B=1/B=2 where practical.
FP16 coverage is CUDA-only and skips gracefully on the CPU CI runner.
"""

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from torch import nn

from src.networks.slbr_next.bg_decoder import (
    BackgroundDecoder,
    BackgroundDecoderOutput,
    FiLMConditioning,
)
from src.networks.slbr_next.blocks import LayerNorm2d, SimpleGate
from src.networks.slbr_next.bottleneck import MDTA
from src.networks.slbr_next.refinement import (
    BackgroundOutput,
    RefinementUNet,
    composite_v1,
)

# Tiny widths keep the 256x512 forward tests fast on the CPU runner.
BG_H8_CHANNELS = 16
BG_H16_CHANNELS = 32
MASK_FEAT_CHANNELS = 12
BG_SKIP_CHANNELS = (4, 6, 8)  # H, H/2, H/4
REFINE_WIDTHS = (8, 12, 16)
REFINE_MDTA_HEADS = 2


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _tiny_decoder(*, clamp_output: bool = True) -> BackgroundDecoder:
    torch.manual_seed(0)
    return BackgroundDecoder(
        bg_h8_channels=BG_H8_CHANNELS,
        bg_h16_channels=BG_H16_CHANNELS,
        mask_feat_channels=MASK_FEAT_CHANNELS,
        mask_prob_channels=1,
        width_h8=8,
        widths=(8, 6, 4),
        skip_h4_channels=BG_SKIP_CHANNELS[2],
        skip_h2_channels=BG_SKIP_CHANNELS[1],
        skip_h_channels=BG_SKIP_CHANNELS[0],
        blocks_per_level=1,
        dw_kernel_sizes=(3, 3, 3),
        clamp_output=clamp_output,
    ).eval()


def _tiny_refiner(
    *,
    attention: str = "mdta",
    clamp_output: bool = True,
) -> RefinementUNet:
    torch.manual_seed(0)
    return RefinementUNet(
        widths=REFINE_WIDTHS,
        bg_skip_channels=BG_SKIP_CHANNELS,
        blocks_per_scale=1,
        attention=attention,
        mdta_heads=REFINE_MDTA_HEADS,
        dw_kernel_sizes=(3, 3, 3),
        clamp_output=clamp_output,
    ).eval()


@pytest.fixture(scope="module")
def tiny_models():
    return _tiny_decoder(), _tiny_refiner()


def _make_inputs(size, batch=1, mask_prob_at_h8=True):
    height, width = size
    mask_prob_size = (
        (height // 8, width // 8) if mask_prob_at_h8 else (height, width)
    )
    return {
        "x": torch.rand(batch, 3, height, width),
        "bg_h8": torch.randn(
            batch, BG_H8_CHANNELS, height // 8, width // 8
        ),
        "bg_h16": torch.randn(
            batch, BG_H16_CHANNELS, height // 16, width // 16
        ),
        "mask_feat": torch.randn(
            batch, MASK_FEAT_CHANNELS, height // 8, width // 8
        ),
        "mask_prob": torch.rand(
            batch, 1, *mask_prob_size
        ),
        "mask_final": torch.rand(batch, 1, height, width),
        "skip_h4": torch.randn(
            batch, BG_SKIP_CHANNELS[2], height // 4, width // 4
        ),
        "skip_h2": torch.randn(
            batch, BG_SKIP_CHANNELS[1], height // 2, width // 2
        ),
        "skip_h": torch.randn(
            batch, BG_SKIP_CHANNELS[0], height, width
        ),
    }


def _run_decoder(decoder, inputs, **overrides):
    kwargs = dict(inputs)
    kwargs.update(overrides)
    return decoder(
        kwargs["x"],
        kwargs["bg_h8"],
        kwargs["bg_h16"],
        kwargs["mask_feat"],
        kwargs["mask_prob"],
        skip_h4=kwargs["skip_h4"],
        skip_h2=kwargs["skip_h2"],
        skip_h=kwargs["skip_h"],
    )


def _run_pipeline(decoder, refiner, inputs):
    decoded = _run_decoder(decoder, inputs)
    output = refiner(
        inputs["x"],
        decoded.coarse_bg,
        inputs["mask_final"],
        skip_h4=decoded.skip_h4,
        skip_h2=decoded.skip_h2,
        skip_h=decoded.skip_h,
    )
    return decoded, output


# ---------------------------------------------------------------------------
# Frozen output interfaces
# ---------------------------------------------------------------------------


def test_background_output_interface_is_frozen():
    assert [field.name for field in fields(BackgroundOutput)] == [
        "coarse_bg",
        "refine_bg",
    ]

    output = BackgroundOutput(torch.zeros(1), torch.ones(1))
    with pytest.raises(FrozenInstanceError):
        output.refine_bg = torch.zeros(1)


def test_background_decoder_output_interface_is_frozen():
    assert [field.name for field in fields(BackgroundDecoderOutput)] == [
        "coarse_bg",
        "skip_h4",
        "skip_h2",
        "skip_h",
    ]

    output = BackgroundDecoderOutput(
        *(torch.zeros(1) for _ in range(4))
    )
    with pytest.raises(FrozenInstanceError):
        output.coarse_bg = torch.ones(1)


# ---------------------------------------------------------------------------
# Forward shapes (256x256, 256x512, B=1/B=2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "batch"),
    [
        ((256, 256), 1),
        ((256, 256), 2),
        ((256, 512), 1),
    ],
)
def test_forward_shapes(tiny_models, size, batch):
    decoder, refiner = tiny_models
    inputs = _make_inputs(size, batch)
    height, width = size

    with torch.no_grad():
        decoded, output = _run_pipeline(decoder, refiner, inputs)

    assert decoded.coarse_bg.shape == (batch, 3, height, width)
    assert decoded.skip_h4.shape == (
        batch,
        BG_SKIP_CHANNELS[2],
        height // 4,
        width // 4,
    )
    assert decoded.skip_h2.shape == (
        batch,
        BG_SKIP_CHANNELS[1],
        height // 2,
        width // 2,
    )
    assert decoded.skip_h.shape == (
        batch,
        BG_SKIP_CHANNELS[0],
        height,
        width,
    )

    assert output.coarse_bg.shape == (batch, 3, height, width)
    assert output.refine_bg.shape == (batch, 3, height, width)
    assert output.coarse_bg.dtype == torch.float32
    assert output.refine_bg.dtype == torch.float32
    for tensor in (
        decoded.coarse_bg,
        decoded.skip_h4,
        decoded.skip_h2,
        decoded.skip_h,
        output.refine_bg,
    ):
        assert torch.isfinite(tensor).all()


def test_default_widths_forward_256():
    torch.manual_seed(0)
    decoder = BackgroundDecoder().eval()
    refiner = RefinementUNet().eval()

    assert decoder.width_h8 == 256
    assert decoder.widths == (128, 64, 32)
    assert refiner.widths == (24, 48, 96)

    inputs = {
        "x": torch.rand(1, 3, 256, 256),
        "bg_h8": torch.randn(1, 256, 32, 32),
        "bg_h16": torch.randn(1, 512, 16, 16),
        "mask_feat": torch.randn(1, 256, 32, 32),
        "mask_prob": torch.rand(1, 1, 32, 32),
        "mask_final": torch.rand(1, 1, 256, 256),
        "skip_h4": torch.randn(1, 128, 64, 64),
        "skip_h2": torch.randn(1, 64, 128, 128),
        "skip_h": torch.randn(1, 32, 256, 256),
    }

    with torch.no_grad():
        decoded, output = _run_pipeline(decoder, refiner, inputs)

    assert decoded.coarse_bg.shape == (1, 3, 256, 256)
    assert decoded.skip_h4.shape == (1, 128, 64, 64)
    assert decoded.skip_h2.shape == (1, 64, 128, 128)
    assert decoded.skip_h.shape == (1, 32, 256, 256)
    assert output.refine_bg.shape == (1, 3, 256, 256)
    assert torch.isfinite(output.refine_bg).all()


def test_default_parameter_budget():
    decoder = BackgroundDecoder()
    refiner = RefinementUNet()

    decoder_params = sum(
        parameter.numel() for parameter in decoder.parameters()
    )
    refiner_params = sum(
        parameter.numel() for parameter in refiner.parameters()
    )
    print(f"BackgroundDecoder(default): {decoder_params:,} parameters")
    print(f"RefinementUNet(default): {refiner_params:,} parameters")

    assert decoder_params > 0
    assert refiner_params > 0
    # Slim decoders; the whole A5 stage must leave room in the 12-15M budget.
    assert decoder_params < 1_000_000
    assert refiner_params < 600_000


# ---------------------------------------------------------------------------
# Residual clamping
# ---------------------------------------------------------------------------


def test_coarse_bg_is_clamped_residual(tiny_models):
    decoder, _ = tiny_models
    inputs = _make_inputs((64, 64))

    with torch.no_grad():
        out = _run_decoder(decoder, inputs)
        expected = (
            inputs["x"] + decoder.head(out.skip_h)
        ).clamp(0.0, 1.0)

    torch.testing.assert_close(out.coarse_bg, expected)
    assert out.coarse_bg.min() >= 0.0
    assert out.coarse_bg.max() <= 1.0


def test_coarse_bg_clamp_can_be_disabled():
    inputs = _make_inputs((64, 64))
    clamped = _tiny_decoder(clamp_output=True)
    unclamped = _tiny_decoder(clamp_output=False)

    for decoder in (clamped, unclamped):
        with torch.no_grad():
            decoder.head.weight.zero_()
            decoder.head.bias.fill_(10.0)

    with torch.no_grad():
        clamped_out = _run_decoder(clamped, inputs)
        unclamped_out = _run_decoder(unclamped, inputs)

    torch.testing.assert_close(
        unclamped_out.coarse_bg, inputs["x"] + 10.0
    )
    assert unclamped_out.coarse_bg.min() > 1.0
    assert clamped_out.coarse_bg.min() >= 0.0
    assert clamped_out.coarse_bg.max() <= 1.0
    torch.testing.assert_close(
        clamped_out.coarse_bg, torch.ones_like(inputs["x"])
    )


def test_refine_bg_is_clamped_residual(tiny_models):
    decoder, _ = tiny_models
    refiner = _tiny_refiner()
    inputs = _make_inputs((64, 64))

    with torch.no_grad():
        decoded = _run_decoder(decoder, inputs)

        refiner.head.weight.zero_()
        refiner.head.bias.zero_()
        zeroed = refiner(
            inputs["x"],
            decoded.coarse_bg,
            inputs["mask_final"],
            skip_h4=decoded.skip_h4,
            skip_h2=decoded.skip_h2,
            skip_h=decoded.skip_h,
        )
        torch.testing.assert_close(zeroed.refine_bg, inputs["x"])

        refiner.head.bias.fill_(10.0)
        saturated = refiner(
            inputs["x"],
            decoded.coarse_bg,
            inputs["mask_final"],
            skip_h4=decoded.skip_h4,
            skip_h2=decoded.skip_h2,
            skip_h=decoded.skip_h,
        )

    torch.testing.assert_close(
        saturated.refine_bg, torch.ones_like(inputs["x"])
    )
    assert saturated.refine_bg.min() >= 0.0
    assert saturated.refine_bg.max() <= 1.0


# ---------------------------------------------------------------------------
# v1 composite
# ---------------------------------------------------------------------------


def test_composite_v1_matches_reference_formula():
    torch.manual_seed(0)
    refine_bg = torch.rand(2, 3, 16, 20)
    image = torch.rand(2, 3, 16, 20)
    mask = (torch.rand(2, 1, 16, 20) > 0.5).to(image.dtype)

    out = composite_v1(refine_bg, image, mask)
    expected = refine_bg * mask + image * (1.0 - mask)

    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out * mask, refine_bg * mask)
    torch.testing.assert_close(out * (1.0 - mask), image * (1.0 - mask))


def test_composite_v1_accepts_three_channel_mask_and_validates():
    refine_bg = torch.rand(1, 3, 8, 8)
    image = torch.rand(1, 3, 8, 8)
    mask = torch.rand(1, 3, 8, 8)

    out = composite_v1(refine_bg, image, mask)
    torch.testing.assert_close(
        out, refine_bg * mask + image * (1.0 - mask)
    )

    with pytest.raises(ValueError):
        composite_v1(refine_bg, image, torch.rand(1, 2, 8, 8))
    with pytest.raises(ValueError):
        composite_v1(refine_bg, image, torch.rand(1, 1, 4, 4))
    with pytest.raises(ValueError):
        composite_v1(refine_bg, torch.rand(1, 3, 4, 4), mask)


# ---------------------------------------------------------------------------
# Refinement input contract
# ---------------------------------------------------------------------------


def test_refinement_consumes_raw_input_once(tiny_models):
    decoder, refiner = tiny_models
    inputs = _make_inputs((64, 64))
    captured = []

    with torch.no_grad():
        decoded = _run_decoder(decoder, inputs)

        handle = refiner.stem.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach())
        )
        try:
            output = refiner(
                inputs["x"],
                decoded.coarse_bg,
                inputs["mask_final"],
                skip_h4=decoded.skip_h4,
                skip_h2=decoded.skip_h2,
                skip_h=decoded.skip_h,
            )
        finally:
            handle.remove()

    assert refiner.stem.in_channels == 7
    assert len(captured) == 1
    stem_input = captured[0]
    assert stem_input.shape[1] == 7
    # [input(3), coarse_bg(3), mask_final(1)] and the raw input appears once.
    torch.testing.assert_close(
        stem_input[:, :3], inputs["x"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        stem_input[:, 3:6], decoded.coarse_bg, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        stem_input[:, 6:], inputs["mask_final"], rtol=0.0, atol=0.0
    )
    assert output.refine_bg.shape == inputs["x"].shape


# ---------------------------------------------------------------------------
# FiLM conditioning
# ---------------------------------------------------------------------------


def test_film_conditioning_math_and_channels(tiny_models):
    decoder, _ = tiny_models
    assert isinstance(decoder.film, FiLMConditioning)
    assert decoder.film.film.in_channels == MASK_FEAT_CHANNELS + 1
    assert decoder.film.film.out_channels == 2 * decoder.width_h8

    torch.manual_seed(0)
    block = FiLMConditioning(4, 3).eval()
    assert isinstance(block.norm, LayerNorm2d)
    assert isinstance(block.simple_gate, SimpleGate)

    x = torch.randn(2, 4, 7, 9)
    cond = torch.randn(2, 3, 7, 9)
    with torch.no_grad():
        out = block(x, cond)
        scale, shift = block.film(cond).chunk(2, dim=1)
        expected = x + block.simple_gate(
            block.gate(scale * block.norm(x) + shift)
        )

    assert out.shape == x.shape
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_film_resizes_conditioning_and_changes_the_background():
    decoder = _tiny_decoder(clamp_output=False)
    inputs = _make_inputs((64, 64), mask_prob_at_h8=False)
    changed = dict(inputs)
    changed["mask_feat"] = inputs["mask_feat"] + 0.5

    with torch.no_grad():
        base = _run_decoder(decoder, inputs)
        modulated = _run_decoder(decoder, changed)

    assert base.coarse_bg.shape == inputs["x"].shape
    assert (
        base.coarse_bg - modulated.coarse_bg
    ).abs().max() > 1e-5
    assert (
        base.skip_h - modulated.skip_h
    ).abs().max() > 1e-5


def test_background_decoder_uses_encoder_skips():
    decoder = _tiny_decoder(clamp_output=False)
    inputs = _make_inputs((64, 64))
    changed = dict(inputs)
    changed["skip_h4"] = inputs["skip_h4"] + 3.0

    with torch.no_grad():
        base = _run_decoder(decoder, inputs)
        shifted = _run_decoder(decoder, changed)

    assert (base.skip_h4 - shifted.skip_h4).abs().max() > 1e-5
    assert (base.coarse_bg - shifted.coarse_bg).abs().max() > 1e-5


def test_refinement_uses_background_decoder_skips(tiny_models):
    decoder, _ = tiny_models
    refiner = _tiny_refiner(clamp_output=False)
    inputs = _make_inputs((64, 64))

    with torch.no_grad():
        decoded = _run_decoder(decoder, inputs)
        base = refiner(
            inputs["x"],
            decoded.coarse_bg,
            inputs["mask_final"],
            skip_h4=decoded.skip_h4,
            skip_h2=decoded.skip_h2,
            skip_h=decoded.skip_h,
        )
        shifted = refiner(
            inputs["x"],
            decoded.coarse_bg,
            inputs["mask_final"],
            skip_h4=decoded.skip_h4,
            skip_h2=decoded.skip_h2 + 3.0,
            skip_h=decoded.skip_h,
        )

    assert (base.refine_bg - shifted.refine_bg).abs().max() > 1e-5


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


def test_refinement_has_one_mdta_at_h8(tiny_models):
    _, refiner = tiny_models
    assert isinstance(refiner.mdta, MDTA)
    assert refiner.mdta.channels == REFINE_WIDTHS[2]
    assert refiner.mdta.num_heads == REFINE_MDTA_HEADS
    assert refiner.naf_bottleneck is None


def test_refinement_naf_fallback_forward():
    decoder = _tiny_decoder()
    refiner = _tiny_refiner(attention="naf")
    inputs = _make_inputs((64, 64))

    assert refiner.mdta is None
    assert refiner.mdta_norm is None
    assert isinstance(refiner.naf_bottleneck, nn.Module)

    with torch.no_grad():
        _, output = _run_pipeline(decoder, refiner, inputs)

    assert output.refine_bg.shape == inputs["x"].shape
    assert torch.isfinite(output.refine_bg).all()


@pytest.mark.parametrize("which", ["decoder", "refiner"])
def test_no_batch_or_instance_norm(which, tiny_models):
    decoder, refiner = tiny_models
    model = decoder if which == "decoder" else refiner
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
        for module in model.modules()
        if isinstance(module, forbidden)
    ]
    assert offenders == []


def test_interpolation_targets_reference_sizes(tiny_models, monkeypatch):
    decoder, refiner = tiny_models
    inputs = _make_inputs((256, 512))
    recorded = []
    real_interpolate = torch.nn.functional.interpolate

    def recording_interpolate(x, size=None, **kwargs):
        recorded.append(tuple(size) if size is not None else None)
        return real_interpolate(x, size=size, **kwargs)

    monkeypatch.setattr(
        "src.networks.slbr_next.bg_decoder.F.interpolate",
        recording_interpolate,
    )

    with torch.no_grad():
        decoded = _run_decoder(decoder, inputs)

    expected = [(32, 64), (64, 128), (128, 256), (256, 512)]
    assert None not in recorded
    assert len(recorded) == len(expected)
    assert recorded == expected

    # ``ResampleBlock`` lives in ``bg_decoder`` and uses that module's ``F``
    # object, so the patch above already records the refinement calls too.
    recorded.clear()

    with torch.no_grad():
        refiner(
            inputs["x"],
            decoded.coarse_bg,
            inputs["mask_final"],
            skip_h4=decoded.skip_h4,
            skip_h2=decoded.skip_h2,
            skip_h=decoded.skip_h,
        )

    assert None not in recorded
    assert len(recorded) == 6
    assert set(recorded) == {
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
    }
    assert all(height <= width for height, width in recorded)


# ---------------------------------------------------------------------------
# Constructor and forward validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"widths": (8, 4)},
        {"widths": (8, 0, 4)},
        {"dw_kernel_sizes": (3, 4, 3)},
        {"blocks_per_level": 0},
        {"bg_h8_channels": 0},
        {"mask_prob_channels": 0},
    ],
)
def test_background_decoder_rejects_invalid_configuration(overrides):
    kwargs = {
        "bg_h8_channels": 8,
        "bg_h16_channels": 16,
        "mask_feat_channels": 8,
        "width_h8": 8,
        "widths": (8, 4, 2),
        "skip_h4_channels": 8,
        "skip_h2_channels": 4,
        "skip_h_channels": 2,
        "dw_kernel_sizes": (3, 3, 3),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        BackgroundDecoder(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"in_channels": 6},
        {"widths": (8, 16)},
        {"widths": (8, 16, 0)},
        {"bg_skip_channels": (8, 16)},
        {"bg_skip_channels": (0, 16, 32)},
        {"dw_kernel_sizes": (3, 4, 5)},
        {"blocks_per_scale": 0},
        {"mdta_heads": 0},
        {"mdta_heads": 5},
        {"attention": "fft"},
    ],
)
def test_refinement_rejects_invalid_configuration(overrides):
    kwargs = {
        "widths": (8, 16, 32),
        "bg_skip_channels": (8, 16, 32),
        "mdta_heads": 4,
        "dw_kernel_sizes": (3, 3, 3),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        RefinementUNet(**kwargs)


def test_forward_validation(tiny_models):
    decoder, refiner = tiny_models
    inputs = _make_inputs((64, 64))

    with pytest.raises(ValueError):
        _run_decoder(decoder, inputs, x=torch.rand(1, 1, 64, 64))
    with pytest.raises(ValueError):
        _run_decoder(
            decoder,
            inputs,
            bg_h8=torch.randn(1, BG_H8_CHANNELS, 4, 4),
        )
    with pytest.raises(ValueError):
        _run_decoder(
            decoder,
            inputs,
            skip_h2=torch.randn(1, BG_SKIP_CHANNELS[1] + 1, 32, 32),
        )

    with torch.no_grad():
        decoded = _run_decoder(decoder, inputs)

    with pytest.raises(ValueError):
        refiner(
            inputs["x"],
            decoded.coarse_bg,
            torch.rand(1, 2, 64, 64),
            skip_h4=decoded.skip_h4,
            skip_h2=decoded.skip_h2,
            skip_h=decoded.skip_h,
        )
    with pytest.raises(ValueError):
        refiner(
            inputs["x"],
            decoded.coarse_bg,
            inputs["mask_final"],
            skip_h4=decoded.skip_h4,
            skip_h2=torch.randn(1, BG_SKIP_CHANNELS[1] + 1, 32, 32),
            skip_h=decoded.skip_h,
        )


# ---------------------------------------------------------------------------
# FP16 (CUDA-only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP16 refinement is CUDA-only; the CPU CI runner skips it.",
)
def test_fp16_cuda():
    decoder = _tiny_decoder().cuda().half()
    refiner = _tiny_refiner().cuda().half()
    inputs = {
        name: tensor.cuda().half()
        for name, tensor in _make_inputs((64, 64)).items()
    }

    with torch.no_grad():
        decoded, output = _run_pipeline(decoder, refiner, inputs)

    assert decoded.coarse_bg.dtype == torch.float16
    assert output.refine_bg.dtype == torch.float16
    assert output.refine_bg.shape == inputs["x"].shape
    assert torch.isfinite(output.refine_bg.float()).all()
