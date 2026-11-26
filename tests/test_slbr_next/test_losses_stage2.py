"""Tests for the Stage 2 losses (spec §5.1-§5.2, §5.4; issue #11)."""

import math

import pytest
import torch

BATCH = 2
HEIGHT = 24
WIDTH = 40

LOSS_KEYS = {
    "total",
    "reconstruction",
    "charbonnier_refine",
    "charbonnier_coarse",
    "perceptual",
    "fft",
    "mask",
    "mask_focal",
    "mask_dice",
    "mask_sobel",
    "alpha",
    "alpha_l1",
    "alpha_grad",
}


@pytest.fixture(scope="module")
def losses():
    """Import lazily so test collection does not pull ``src.models``.

    ``test_imports.py`` asserts that importing ``src.networks.slbr_next``
    leaves ``src.models`` out of ``sys.modules``. Pytest imports every test
    module during collection, so this module's import of the training package
    must happen at test time, after the isolation check.
    """
    from src.models.slbr_next import losses_stage2

    return losses_stage2


class DummyPerceptual(torch.nn.Module):
    """Deterministic stand-in for LPIPS/DISTS (no downloads in CI)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        self.calls += 1
        self.inputs.append((pred.detach().clone(), target.detach().clone()))
        return (pred - target).abs().mean()


def _inputs(
    height: int = HEIGHT,
    width: int = WIDTH,
    batch: int = BATCH,
    seed: int = 0,
):
    generator = torch.Generator().manual_seed(seed)
    clean = torch.rand(batch, 3, height, width, generator=generator)
    refine = torch.rand(batch, 3, height, width, generator=generator)
    coarse = torch.rand(batch, 3, height, width, generator=generator)
    mask = (
        torch.rand(batch, 1, height, width, generator=generator) > 0.55
    ).float()
    logits = torch.randn(batch, 1, height, width, generator=generator)
    alpha_pred = torch.rand(batch, 1, height, width, generator=generator)
    alpha_gt = torch.rand(batch, 1, height, width, generator=generator)

    refine.requires_grad_(True)
    coarse.requires_grad_(True)
    logits.requires_grad_(True)
    alpha_pred.requires_grad_(True)

    return clean, refine, coarse, mask, logits, alpha_pred, alpha_gt


def _call(criterion, tensors, **kwargs) -> dict[str, torch.Tensor]:
    clean, refine, coarse, mask, logits, _, _ = tensors
    return criterion(
        refine,
        coarse,
        logits,
        clean,
        mask,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Synthetic composites
# ---------------------------------------------------------------------------


def _blend_inputs(batch: int = 3, height: int = 16, width: int = 24):
    generator = torch.Generator().manual_seed(11)
    background = torch.rand(
        batch, 3, height, width, generator=generator
    )
    watermark = torch.rand(
        batch, 3, height, width, generator=generator
    )
    return background, watermark


def test_linear_composite_returns_true_alpha(losses):
    background, watermark = _blend_inputs()
    alpha = torch.tensor([0.1, 0.5, 0.9]).view(3, 1, 1, 1)

    composite = losses.build_synthetic_composite(
        background, watermark, modes="linear", alpha=alpha
    )

    assert isinstance(composite, losses.SyntheticComposite)
    assert composite.modes == ("linear",) * 3
    assert composite.alpha is not None
    assert composite.alpha.shape == (3, 1, 16, 24)
    assert composite.alpha_valid.all()
    torch.testing.assert_close(
        composite.image, (1.0 - alpha) * background + alpha * watermark
    )
    torch.testing.assert_close(
        composite.alpha, alpha.expand(3, 1, 16, 24)
    )


@pytest.mark.parametrize("mode", ["screen", "multiply"])
def test_nonlinear_composites_return_no_alpha(mode, losses):
    background, watermark = _blend_inputs()

    composite = losses.build_synthetic_composite(
        background,
        watermark,
        modes=mode,
        alpha=torch.rand(3, 1, 1, 1),
    )
    assert composite.alpha is None
    assert not bool(composite.alpha_valid.any())

    if mode == "screen":
        expected = 1.0 - (1.0 - background) * (1.0 - watermark)
    else:
        expected = background * watermark
    torch.testing.assert_close(composite.image, expected)

    # Alpha is ignored for nonlinear families.
    other = losses.build_synthetic_composite(
        background,
        watermark,
        modes=mode,
        alpha=torch.zeros(3, 1, 1, 1),
    )
    torch.testing.assert_close(other.image, composite.image)


def test_per_sample_modes_and_alpha(losses):
    background, watermark = _blend_inputs(batch=4)
    modes = ("linear", "screen", "multiply", "linear")
    alpha = torch.tensor([0.2, 0.4, 0.6, 0.8]).view(4, 1, 1, 1)

    composite = losses.build_synthetic_composite(
        background, watermark, modes=modes, alpha=alpha
    )

    assert composite.modes == modes
    assert composite.alpha_valid.tolist() == [True, False, False, True]
    assert composite.alpha is not None
    assert torch.count_nonzero(composite.alpha[1]).item() == 0
    assert torch.count_nonzero(composite.alpha[2]).item() == 0

    torch.testing.assert_close(
        composite.image[0],
        (1.0 - alpha[0]) * background[0] + alpha[0] * watermark[0],
    )
    torch.testing.assert_close(
        composite.image[1],
        1.0 - (1.0 - background[1]) * (1.0 - watermark[1]),
    )
    torch.testing.assert_close(
        composite.image[2], background[2] * watermark[2]
    )
    torch.testing.assert_close(
        composite.image[3],
        (1.0 - alpha[3]) * background[3] + alpha[3] * watermark[3],
    )
    torch.testing.assert_close(
        composite.alpha[0], alpha[0].expand(1, 16, 24)
    )


def test_random_alpha_is_reproducible_and_bounded(losses):
    background, watermark = _blend_inputs()
    kwargs = dict(
        modes="linear",
        alpha=None,
        alpha_range=(0.2, 0.6),
    )
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    first = losses.build_synthetic_composite(
        background, watermark, generator=generator_a, **kwargs
    )
    second = losses.build_synthetic_composite(
        background, watermark, generator=generator_b, **kwargs
    )

    assert first.alpha is not None
    torch.testing.assert_close(first.alpha, second.alpha)
    assert float(first.alpha.min()) >= 0.2 - 1e-6
    assert float(first.alpha.max()) <= 0.6 + 1e-6


def test_composite_rejects_bad_modes_and_shapes(losses):
    background, watermark = _blend_inputs()

    assert losses.BLEND_MODES == ("linear", "screen", "multiply")
    with pytest.raises(ValueError):
        losses.build_synthetic_composite(
            background, watermark, modes="overlay"
        )
    with pytest.raises(ValueError):
        losses.build_synthetic_composite(
            background, watermark, modes=("linear",)
        )
    with pytest.raises(ValueError):
        losses.build_synthetic_composite(
            background, watermark[:, :, :, :5]
        )


# ---------------------------------------------------------------------------
# Component sanity
# ---------------------------------------------------------------------------


def test_all_components_are_finite_with_finite_gradients(losses):
    tensors = _inputs()
    refine = tensors[1]
    coarse = tensors[2]
    logits = tensors[4]
    alpha_pred = tensors[5]
    criterion = losses.Stage2Loss(perceptual=DummyPerceptual())
    criterion.set_progress(1, 2)

    result = _call(
        criterion,
        tensors,
        alpha_pred=alpha_pred,
        alpha_gt=tensors[6],
        alpha_valid=torch.tensor([True, True]),
    )

    assert set(result) == LOSS_KEYS
    for key, value in result.items():
        assert value.shape == (), key
        assert value.dtype == torch.float32, key
        assert torch.isfinite(value).all(), key

    result["total"].backward()

    for tensor in (refine, coarse, logits, alpha_pred):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_fixed_batch_matches_documented_weights(losses):
    tensors = _inputs(seed=5)
    alpha_pred = tensors[5]
    alpha_gt = tensors[6]
    criterion = losses.Stage2Loss(perceptual=DummyPerceptual())
    criterion.set_progress(1, 1)

    result = _call(
        criterion,
        tensors,
        alpha_pred=alpha_pred,
        alpha_gt=alpha_gt,
        alpha_valid=torch.tensor([True, True]),
    )

    expected_total = (
        1.0 * result["reconstruction"]
        + 0.1 * result["perceptual"]
        + 0.05 * result["fft"]
        + 1.0 * result["mask"]
        + 0.5 * result["alpha"]
    )
    torch.testing.assert_close(result["total"], expected_total)
    torch.testing.assert_close(
        result["reconstruction"],
        0.5
        * (result["charbonnier_refine"] + result["charbonnier_coarse"]),
    )
    torch.testing.assert_close(
        result["mask"],
        result["mask_focal"]
        + result["mask_dice"]
        + result["mask_sobel"],
    )
    assert result["perceptual"].item() > 0
    assert result["fft"].item() > 0


def test_perfect_prediction_hits_charbonnier_floor(losses):
    clean, _, _, mask, logits, alpha_pred, alpha_gt = _inputs(seed=7)
    tensors = (
        clean,
        clean.clone(),
        clean.clone(),
        mask,
        logits,
        alpha_pred,
        alpha_gt,
    )
    criterion = losses.Stage2Loss(perceptual=None)

    result = _call(criterion, tensors)

    # sqrt(0 + eps^2) = eps inside and outside. The 5 px dilation usually
    # covers the whole small test mask, so the anchor adds 0 to 0.1 eps.
    floor = criterion.charbonnier_eps
    for key in ("charbonnier_refine", "reconstruction"):
        value = result[key].item()
        assert floor - 1e-7 <= value <= 1.1 * floor + 1e-7, key


# ---------------------------------------------------------------------------
# Alpha gating
# ---------------------------------------------------------------------------


def test_alpha_loss_only_for_valid_linear_samples(losses):
    tensors = _inputs(seed=3)
    alpha_pred = tensors[5]
    alpha_gt = tensors[6]
    criterion = losses.Stage2Loss(perceptual=None)

    linear = _call(
        criterion,
        tensors,
        alpha_pred=alpha_pred,
        alpha_gt=alpha_gt,
        alpha_valid=torch.tensor([True, True]),
    )
    mixed = _call(
        criterion,
        tensors,
        alpha_pred=alpha_pred,
        alpha_gt=alpha_gt,
        alpha_valid=torch.tensor([True, False]),
    )
    none_linear = _call(
        criterion,
        tensors,
        alpha_pred=alpha_pred * 0.5,
        alpha_gt=torch.ones_like(alpha_gt),
        alpha_valid=torch.zeros(BATCH, dtype=torch.bool),
    )

    per_sample_l1 = (
        (alpha_pred.detach() - alpha_gt).abs().flatten(1).mean(dim=1)
    )
    torch.testing.assert_close(linear["alpha_l1"], per_sample_l1.mean())
    torch.testing.assert_close(mixed["alpha_l1"], per_sample_l1[0])
    assert linear["alpha_grad"].item() > 0.0
    assert none_linear["alpha"].item() == 0.0
    assert none_linear["alpha_l1"].item() == 0.0
    assert none_linear["alpha_grad"].item() == 0.0

    torch.testing.assert_close(
        none_linear["total"], _call(criterion, tensors)["total"]
    )


def test_no_linear_samples_keep_a_zero_alpha_gradient(losses):
    tensors = _inputs(seed=4)
    alpha_pred = tensors[5]
    criterion = losses.Stage2Loss(perceptual=None)

    result = _call(
        criterion,
        tensors,
        alpha_pred=alpha_pred,
        alpha_gt=tensors[6],
        alpha_valid=torch.zeros(BATCH, dtype=torch.bool),
    )
    result["total"].backward()

    assert alpha_pred.grad is not None
    assert torch.count_nonzero(alpha_pred.grad).item() == 0


def test_alpha_gt_requires_validity_mask(losses):
    tensors = _inputs(seed=6)
    criterion = losses.Stage2Loss(perceptual=None)

    with pytest.raises(ValueError):
        _call(
            criterion,
            tensors,
            alpha_pred=tensors[5],
            alpha_gt=tensors[6],
        )


# ---------------------------------------------------------------------------
# Perceptual schedule
# ---------------------------------------------------------------------------


def test_perceptual_activates_late_and_uses_refine_bg_only(losses):
    tensors = _inputs(seed=8)
    clean = tensors[0]
    refine = tensors[1]
    dummy = DummyPerceptual()
    criterion = losses.Stage2Loss(perceptual=dummy)

    criterion.set_progress(0, 100)
    assert not criterion.perceptual_active
    before = _call(criterion, tensors)
    assert dummy.calls == 0
    assert before["perceptual"].item() == 0.0

    criterion.set_progress(34, 100)
    assert not criterion.perceptual_active
    near = _call(criterion, tensors)
    assert dummy.calls == 0
    assert near["perceptual"].item() == 0.0

    criterion.set_progress(35, 100)
    assert criterion.perceptual_active
    after = _call(criterion, tensors)
    assert dummy.calls == 1
    assert after["perceptual"].item() > 0.0

    pred, target = dummy.inputs[-1]
    torch.testing.assert_close(pred, refine.detach())
    torch.testing.assert_close(target, clean)


def test_perceptual_can_be_disabled_and_progress_is_clamped(losses):
    tensors = _inputs(seed=9)
    criterion = losses.Stage2Loss(perceptual=None)
    criterion.set_progress(500, 100)

    assert criterion.progress == pytest.approx(1.0)
    assert not criterion.perceptual_active
    result = _call(criterion, tensors)
    assert result["perceptual"].item() == 0.0
    assert torch.isfinite(result["total"]).all()

    with pytest.raises(ValueError):
        criterion.set_progress(1, 0)


def test_auto_perceptual_backend_is_lazy(losses):
    criterion = losses.Stage2Loss(perceptual="auto")

    assert criterion.perceptual is None
    assert not criterion.perceptual_active


# ---------------------------------------------------------------------------
# FFT precision
# ---------------------------------------------------------------------------


def test_fft_path_is_fp32_under_autocast(monkeypatch, losses):
    tensors = _inputs(seed=10)
    seen_dtypes: list[torch.dtype] = []
    original_rfft2 = torch.fft.rfft2

    def spy(input, *args, **kwargs):
        seen_dtypes.append(input.dtype)
        return original_rfft2(input, *args, **kwargs)

    monkeypatch.setattr(torch.fft, "rfft2", spy)
    criterion = losses.Stage2Loss(
        perceptual=None, fft_scales=(1.0, 0.5)
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = _call(criterion, tensors)

    assert seen_dtypes
    assert all(dtype == torch.float32 for dtype in seen_dtypes)
    assert len(seen_dtypes) == 2 * 2  # two scales, magnitude/real-imag FFT
    assert torch.isfinite(result["fft"]).all()
    assert result["fft"].item() > 0.0


# ---------------------------------------------------------------------------
# Non-square geometry
# ---------------------------------------------------------------------------


def test_non_square_mask_levels_and_sobel(losses):
    height, width = 37, 83
    generator = torch.Generator().manual_seed(13)
    clean = torch.rand(BATCH, 3, height, width, generator=generator)
    refine = torch.rand(
        BATCH, 3, height, width, generator=generator
    ).requires_grad_(True)
    coarse = torch.rand(
        BATCH, 3, height, width, generator=generator
    ).requires_grad_(True)
    mask = (
        torch.rand(BATCH, 1, height, width, generator=generator) > 0.6
    ).float()
    levels = [
        torch.randn(
            BATCH, 1, height, width, generator=generator
        ).requires_grad_(True),
        torch.randn(
            BATCH,
            1,
            math.ceil(height / 2),
            math.ceil(width / 2),
            generator=generator,
        ).requires_grad_(True),
        torch.randn(
            BATCH,
            1,
            math.ceil(height / 4),
            math.ceil(width / 4),
            generator=generator,
        ).requires_grad_(True),
    ]
    criterion = losses.Stage2Loss(perceptual=None)

    result = criterion(refine, coarse, levels, clean, mask)

    for key in ("mask", "mask_focal", "mask_dice", "mask_sobel", "fft"):
        assert torch.isfinite(result[key]).all(), key

    result["total"].backward()
    for level in levels:
        assert level.grad is not None
        assert torch.isfinite(level.grad).all()
    assert refine.grad is not None
    assert coarse.grad is not None
    assert torch.isfinite(refine.grad).all()
    assert torch.isfinite(coarse.grad).all()
