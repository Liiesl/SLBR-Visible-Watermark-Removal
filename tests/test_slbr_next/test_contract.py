"""Contract and invariant tests for the assembled SLBR-Next v1 model (issue #8).

Covers spec §1 (contracts), §3.8 (shape flow), §6.1 (functional tests) and
§6.2 (numerical tests):

* arbitrary sizes, including non-square and non-multiples of 16;
* exact pad-to-/16 + crop-back behaviour;
* the v1 return contract ``([refine_bg, coarse_bg], mask_list, [None])`` and
  the documented finest-to-coarsest mask ordering (no duplicate final mask);
* images finite and in ``[0, 1]``, masks finite and in ``[0, 1]``, logits
  finite;
* the v1 composite matches ``refine_bg * M + x * (1 - M)`` exactly;
* 12-15M parameters, printed per module and in total;
* FLOPs at 256x256 through ``torch.utils.flop_counter.FlopCounterMode``;
* no BatchNorm/InstanceNorm;
* an unconditional FP32 contract test and a CUDA/MPS-gated FP16 contract test.

The FP16 test is skipped on the hosted Linux CPU runner because CPU FP16
convolution is unsupported there. It is CUDA/MPS-gated for local and GPU CI
runners; GitHub's macOS 15 paravirtual Metal device reports that it does not
support Metal 2.0, so MPS is unavailable on that hosted runner (documented in
``model_report.md``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from src.networks.slbr_next import (
    SLBRNextV1,
    SLBRNextV1Config,
    pad_to_multiple,
)


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _crop_scaled(tensor: torch.Tensor, height: int, width: int, divisor: int):
    """Crop a padded pyramid tensor back to the original geometry."""
    return tensor[
        ...,
        : _ceil_div(height, divisor),
        : _ceil_div(width, divisor),
    ]


def _fp16_device() -> str | None:
    """Return ``"cuda"``, ``"mps"`` or ``None`` (skip) for FP16 tests."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return None


FP16_DEVICE = _fp16_device()


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    # ``pretrained=False`` keeps CI offline; it does not change parameter count.
    return SLBRNextV1(convnext_pretrained=False).eval()


# ---------------------------------------------------------------------------
# Forward contract and shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size",
    [
        (256, 256),
        (384, 384),
        (256, 512),
        (512, 512),
        (512, 256),
    ],
)
def test_forward_contract_shapes(model, size):
    height, width = size
    x = torch.rand(1, 3, height, width)

    with torch.no_grad():
        image_output, mask_list, imwatermark = model(x)

    assert isinstance(image_output, list) and len(image_output) == 2
    refine_bg, coarse_bg = image_output
    for tensor in (refine_bg, coarse_bg):
        assert tensor.shape == (1, 3, height, width)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    assert isinstance(mask_list, list) and len(mask_list) == 3
    expected_shapes = [
        (1, 1, height, width),
        (1, 1, _ceil_div(height, 2), _ceil_div(width, 2)),
        (1, 1, _ceil_div(height, 4), _ceil_div(width, 4)),
    ]
    for mask, expected_shape in zip(mask_list, expected_shapes):
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == expected_shape
        assert mask.dtype == torch.float32
        assert torch.isfinite(mask).all()
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    # Documented ordering: H, H/2, H/4. The final mask is not duplicated.
    assert mask_list[0].shape[2] > mask_list[1].shape[2]
    assert mask_list[1].shape[2] > mask_list[2].shape[2]
    assert mask_list[0] is not mask_list[1]

    assert isinstance(imwatermark, list)
    assert imwatermark == [None]


def test_forward_contract_batch_two_non_square(model):
    x = torch.rand(2, 3, 128, 192)

    with torch.no_grad():
        (refine_bg, coarse_bg), mask_list, imwatermark = model(x)

    assert refine_bg.shape == coarse_bg.shape == (2, 3, 128, 192)
    assert mask_list[0].shape == (2, 1, 128, 192)
    assert mask_list[1].shape == (2, 1, 64, 96)
    assert mask_list[2].shape == (2, 1, 32, 48)
    for tensor in (refine_bg, coarse_bg, *mask_list):
        assert torch.isfinite(tensor).all()
    assert imwatermark == [None]


# ---------------------------------------------------------------------------
# Padding and crop-back
# ---------------------------------------------------------------------------


def test_padding_uses_reflect_then_replicate():
    # Small input: both required pads are >= the input dimension, so both
    # axes must use replicate (reflect would be undefined).
    small = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    padded, original = pad_to_multiple(small, 8)
    assert original == (3, 4)
    assert padded.shape == (1, 1, 8, 8)
    expected = F.pad(small, (0, 4, 0, 5), mode="replicate")
    assert torch.equal(padded, expected)

    # Normal case: pad < input dimension on both axes, so reflect is used.
    normal = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
    padded, original = pad_to_multiple(normal, 16)
    assert original == (10, 10)
    assert padded.shape == (1, 1, 16, 16)
    expected = F.pad(normal, (0, 6, 0, 6), mode="reflect")
    assert torch.equal(padded, expected)

    # Already aligned: no copy, no padding.
    aligned = torch.rand(1, 3, 16, 32)
    padded, original = pad_to_multiple(aligned)
    assert original == (16, 32)
    assert padded is aligned


@pytest.mark.parametrize("size", [(250, 333), (257, 255)])
def test_pad_and_crop_are_exact(model, size):
    height, width = size
    torch.manual_seed(11)
    x = torch.rand(1, 3, height, width)
    padded, original = pad_to_multiple(x)
    assert original == (height, width)
    assert padded.shape[2] % 16 == 0
    assert padded.shape[3] % 16 == 0

    with torch.no_grad():
        direct_images, direct_masks, _ = model(x)
        padded_images, padded_masks, _ = model(padded)

    # Running the model on the already-padded tensor and cropping must match
    # running it on the original tensor bit-for-bit, for images and masks.
    for direct, full in zip(direct_images, padded_images):
        expected = _crop_scaled(full, height, width, 1)
        assert direct.shape == expected.shape == (1, 3, height, width)
        assert torch.equal(direct, expected)

    for direct, full, divisor in zip(
        direct_masks, padded_masks, (1, 2, 4)
    ):
        expected = _crop_scaled(full, height, width, divisor)
        assert direct.shape == expected.shape
        assert torch.equal(direct, expected)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def test_composite_matches_reference_formula(model):
    torch.manual_seed(5)
    x = torch.rand(1, 3, 70, 93)

    with torch.no_grad():
        (refine_bg, _), mask_list, _ = model(x)
        composite = model.composite(x)
        alias = model.forward_composite(x)

    expected = refine_bg * mask_list[0] + x * (1.0 - mask_list[0])

    assert composite.shape == alias.shape == x.shape
    torch.testing.assert_close(composite, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(alias, expected, rtol=0.0, atol=0.0)
    assert torch.isfinite(composite).all()
    assert composite.min() >= 0.0
    assert composite.max() <= 1.0


# ---------------------------------------------------------------------------
# Mask logits and ordering
# ---------------------------------------------------------------------------


def test_mask_logits_are_finite_and_match_mask_list(model):
    torch.manual_seed(7)
    x = torch.rand(1, 3, 64, 96)

    with torch.no_grad():
        logits = model.forward_logits(x)
        _, mask_list, _ = model(x)

    assert isinstance(logits, list) and len(logits) == 3
    for logit, mask, divisor in zip(logits, mask_list, (1, 2, 4)):
        expected_shape = (
            1,
            1,
            _ceil_div(64, divisor),
            _ceil_div(96, divisor),
        )
        assert logit.shape == mask.shape == expected_shape
        assert torch.isfinite(logit).all()
        # ``mask_list`` is exactly ``sigmoid`` of the published logits.
        torch.testing.assert_close(
            torch.sigmoid(logit), mask, rtol=0.0, atol=0.0
        )


# ---------------------------------------------------------------------------
# Parameter budget and FLOPs
# ---------------------------------------------------------------------------


def test_parameter_budget(model):
    per_module: dict[str, int] = {}
    for name, child in model.named_children():
        per_module[name] = sum(
            parameter.numel() for parameter in child.parameters()
        )
        print(f"{name:>20s}: {per_module[name]:,} parameters")

    total = sum(parameter.numel() for parameter in model.parameters())
    printed_total = sum(per_module.values())
    print(f"{'SLBRNextV1 total':>20s}: {total:,} parameters")

    assert printed_total == total
    assert 12_000_000 <= total <= 15_000_000

    # The default config must be the reference configuration; the budget is
    # a property of widths/block counts, not of pretrained weights.
    default = SLBRNextV1Config()
    assert default.convnext_pretrained is True
    assert model.config.encoder_widths == default.encoder_widths
    assert model.config.mask_widths == default.mask_widths
    assert model.config.background_widths == default.background_widths
    assert model.config.refinement_widths == default.refinement_widths


def test_flops_at_256(model):
    torch.manual_seed(3)
    x = torch.rand(1, 3, 256, 256)
    counter = FlopCounterMode(display=False)

    # FlopCounterMode's ModuleTracker asserts on parameters without a grad_fn,
    # so it must run with autograd enabled (no ``torch.no_grad()`` here). No
    # backward pass is performed and the graph is released immediately.
    with counter:
        output = model(x)
    del output

    total_flops = counter.get_total_flops()
    print(
        "SLBRNextV1 FLOPs @ 256x256: "
        f"{total_flops:,} ({total_flops / 1e9:.2f} GFLOPs)"
    )
    assert total_flops > 0


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


def test_no_batch_or_instance_norm(model):
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
        f"{name}: {type(module).__name__}"
        for name, module in model.named_modules()
        if isinstance(module, forbidden)
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# Dtype contracts
# ---------------------------------------------------------------------------


def test_fp32_contract(model):
    """Unconditional FP32 contract test; this must run on every CI runner."""
    torch.manual_seed(13)
    x = torch.rand(2, 3, 96, 80, dtype=torch.float32)

    with torch.no_grad():
        (refine_bg, coarse_bg), mask_list, imwatermark = model(x)

    assert refine_bg.dtype == coarse_bg.dtype == torch.float32
    assert all(mask.dtype == torch.float32 for mask in mask_list)
    assert refine_bg.shape == coarse_bg.shape == (2, 3, 96, 80)
    for tensor in (refine_bg, coarse_bg, *mask_list):
        assert torch.isfinite(tensor).all()
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
    assert imwatermark == [None]


@pytest.mark.skipif(
    FP16_DEVICE is None,
    reason=(
        "FP16 contract requires CUDA or MPS; the hosted CPU runner cannot "
        "run FP16 convolution, so it skips this test."
    ),
)
def test_fp16_contract():
    """Full v1 contract in FP16 on CUDA or MPS (skipped elsewhere)."""
    device = FP16_DEVICE
    assert device is not None

    torch.manual_seed(17)
    model = SLBRNextV1(convnext_pretrained=False).eval().half().to(device)
    x = torch.rand(1, 3, 64, 96, device=device).half()

    with torch.no_grad():
        (refine_bg, coarse_bg), mask_list, imwatermark = model(x)
        composite = model.composite(x)

    assert refine_bg.dtype == torch.float16
    assert coarse_bg.dtype == torch.float16
    assert all(mask.dtype == torch.float16 for mask in mask_list)
    assert refine_bg.shape == coarse_bg.shape == (1, 3, 64, 96)
    assert mask_list[0].shape == (1, 1, 64, 96)
    assert mask_list[1].shape == (1, 1, 32, 48)
    assert mask_list[2].shape == (1, 1, 16, 24)
    assert composite.dtype == torch.float16
    assert composite.shape == (1, 3, 64, 96)

    for tensor in (refine_bg, coarse_bg, *mask_list, composite):
        assert torch.isfinite(tensor.float()).all()
        assert tensor.float().min() >= 0.0
        assert tensor.float().max() <= 1.0
    assert imwatermark == [None]
