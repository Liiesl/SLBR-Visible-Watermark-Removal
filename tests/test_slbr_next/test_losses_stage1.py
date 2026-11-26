"""Tests for the Stage 1 losses (spec §4.2-§4.4; issue #10).

The core tests run on CPU in a few seconds. ``src.models.slbr_next`` is
imported lazily in the fixtures so that ``test_imports.py`` keeps its
import-isolation guarantee: pytest imports every test module during
collection, and a module-level training-package import would put
``src.models`` into ``sys.modules`` before that assertion runs. CUDA/FP16
checks skip when no GPU is present.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

BATCH = 2
HEIGHT = 32
WIDTH = 48

LOSS_KEYS = {
    "total",
    "mask",
    "mask_bce",
    "mask_dice",
    "mask_bce_h",
    "mask_bce_h2",
    "mask_bce_h4",
    "mask_dice_h",
    "mask_dice_h2",
    "mask_dice_h4",
    "cwd",
    "cwd_h8",
    "cwd_h4",
    "cwd_h",
    "charbonnier",
    "charbonnier_refine",
    "charbonnier_coarse",
    "fft",
}


@pytest.fixture(scope="module")
def stage1():
    """Import the training package only when this module's tests run."""
    from src.models.slbr_next import losses_stage1

    return losses_stage1


@pytest.fixture(scope="module")
def teacher_target_cls():
    from src.models.slbr_next.teacher import TeacherTarget

    return TeacherTarget


# ---------------------------------------------------------------------------
# Synthetic batch helpers
# ---------------------------------------------------------------------------


def _level_sizes(height: int, width: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        ((height + divisor - 1) // divisor, (width + divisor - 1) // divisor)
        for divisor in (1, 2, 4)
    )


def _make_target(
    TeacherTarget,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    batch: int = BATCH,
    seed: int = 0,
    teacher_h8_channels: int = 5,
    teacher_h4_channels: int = 7,
    device: str = "cpu",
):
    generator = torch.Generator().manual_seed(seed)

    def randn(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.randn(
            batch, channels, h, w, generator=generator
        ).to(device)

    def rand(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.rand(batch, channels, h, w, generator=generator).to(
            device
        )

    sizes = _level_sizes(height, width)
    h8 = ((height + 7) // 8, (width + 7) // 8)
    h4 = sizes[2]
    return TeacherTarget(
        mask_logits=tuple(randn(1, h, w) for h, w in sizes),
        primary_logits=tuple(randn(1, h, w) for h, w in sizes),
        refine_bg=rand(3, height, width),
        coarse_bg=rand(3, height, width),
        bottleneck_h8=randn(teacher_h8_channels, *h8),
        bg_decoder_h4=randn(teacher_h4_channels, *h4),
        height=height,
        width=width,
    )


def _make_output(
    stage1,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    batch: int = BATCH,
    seed: int = 1,
    student_h8_channels: int = 3,
    student_h4_channels: int = 4,
    student_h_channels: int = 2,
    device: str = "cpu",
    requires_grad: bool = True,
    include_full_h: bool = True,
):
    generator = torch.Generator().manual_seed(seed)

    def randn(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.randn(
            batch, channels, h, w, generator=generator
        ).to(device)

    def rand(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.rand(batch, channels, h, w, generator=generator).to(
            device
        )

    sizes = _level_sizes(height, width)
    h8 = ((height + 7) // 8, (width + 7) // 8)
    h4 = sizes[2]
    refine = rand(3, height, width)
    coarse = rand(3, height, width)
    mask_logits = tuple(randn(1, h, w) for h, w in sizes)
    feature_h8 = randn(student_h8_channels, *h8)
    feature_h4 = randn(student_h4_channels, *h4)
    feature_h = randn(student_h_channels, height, width)

    tensors = [
        refine,
        coarse,
        *mask_logits,
        feature_h8,
        feature_h4,
    ]
    if include_full_h:
        tensors.append(feature_h)
    if requires_grad:
        for tensor in tensors:
            tensor.requires_grad_(True)

    return stage1.StudentStage1Output(
        refine_bg=refine,
        coarse_bg=coarse,
        mask_logits=mask_logits,
        feature_h8=feature_h8,
        feature_h4=feature_h4,
        feature_h=feature_h if include_full_h else None,
    )


def _criterion(stage1, **overrides):
    values = dict(
        student_h8_channels=3,
        student_h4_channels=4,
        student_h_channels=2,
        teacher_h8_channels=5,
        teacher_h4_channels=7,
        teacher_h_channels=3,
    )
    values.update(overrides)
    return stage1.Stage1Loss(**values)


# ---------------------------------------------------------------------------
# Independent re-implementations used to check the documented formulas
# ---------------------------------------------------------------------------


def _manual_soft_dice(prediction, target, eps: float) -> torch.Tensor:
    dims = tuple(range(1, prediction.dim()))
    intersection = (prediction * target).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def _manual_cwd(student, teacher, temperature: float) -> torch.Tensor:
    if student.shape[-2:] != teacher.shape[-2:]:
        student = F.interpolate(
            student,
            size=teacher.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    student = F.softmax(student.flatten(2) / temperature, dim=-1)
    teacher = F.softmax(teacher.flatten(2) / temperature, dim=-1)
    student = F.normalize(student, p=2, dim=-1)
    teacher = F.normalize(teacher, p=2, dim=-1)
    return (1.0 - (student * teacher).sum(dim=-1)).mean()


def _manual_charbonnier(
    prediction,
    target,
    mask_logits,
    *,
    radius: int,
    unmasked_weight: float,
    eps: float,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.detach().float()
    mask = torch.sigmoid(mask_logits.detach().float())
    if mask.shape[-2:] != prediction.shape[-2:]:
        mask = F.interpolate(
            mask,
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    dilated = F.max_pool2d(
        mask,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ).clamp(0.0, 1.0)
    error = torch.sqrt((prediction - target).square() + eps * eps)

    def weighted_mean(values, weights):
        weights = weights.expand_as(values).flatten(1)
        values = values.flatten(1)
        return (values * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1e-8)

    inside = weighted_mean(error, dilated)
    outside = weighted_mean(error, 1.0 - dilated)
    return (inside + unmasked_weight * outside).mean()


# ---------------------------------------------------------------------------
# Component sanity and gradients
# ---------------------------------------------------------------------------


def test_all_components_are_finite_with_finite_gradients(
    stage1, teacher_target_cls
):
    target = _make_target(teacher_target_cls, seed=2)
    output = _make_output(stage1, seed=3)
    full_h = torch.randn(
        BATCH, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(4)
    )
    full_h.requires_grad_(True)
    criterion = _criterion(stage1, full_h_every=2)

    result = criterion(output, target, teacher_full_h=full_h, step=0)

    assert set(result) == LOSS_KEYS
    for key, value in result.items():
        assert value.shape == (), key
        assert value.dtype == torch.float32, key
        assert torch.isfinite(value).all(), key

    assert result["mask"].item() > 0.0
    assert result["cwd"].item() > 0.0
    assert result["cwd_h"].item() > 0.0
    assert result["charbonnier"].item() > 0.0
    assert result["fft"].item() > 0.0
    assert 0.0 <= result["cwd_h8"].item() <= 2.0
    assert 0.0 <= result["cwd_h4"].item() <= 2.0

    result["total"].backward()

    for tensor in (
        output.refine_bg,
        output.coarse_bg,
        *output.mask_logits,
        output.feature_h8,
        output.feature_h4,
        output.feature_h,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()

    # The teacher is a target, never a differentiable branch.
    assert full_h.grad is None
    for parameter in criterion.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_fixed_batch_matches_documented_total(stage1, teacher_target_cls):
    target = _make_target(teacher_target_cls, seed=5)
    output = _make_output(stage1, seed=6)
    full_h = torch.randn(
        BATCH, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(7)
    )
    criterion = _criterion(stage1, full_h_every=3)
    result = criterion(output, target, teacher_full_h=full_h, step=3)

    expected_total = (
        1.0 * result["mask"]
        + 0.1 * result["cwd"]
        + 1.0 * result["charbonnier"]
        + 0.05 * result["fft"]
    )
    torch.testing.assert_close(result["total"], expected_total)
    torch.testing.assert_close(
        result["charbonnier"],
        0.5
        * (
            result["charbonnier_refine"]
            + result["charbonnier_coarse"]
        ),
    )
    torch.testing.assert_close(
        result["cwd"],
        result["cwd_h8"] + result["cwd_h4"] + result["cwd_h"],
    )


# ---------------------------------------------------------------------------
# Mask distillation
# ---------------------------------------------------------------------------


def test_mask_geometric_weights_count_final_h_once(
    stage1, teacher_target_cls
):
    target = _make_target(teacher_target_cls, seed=8)
    output = _make_output(stage1, seed=9)
    criterion = _criterion(
        stage1,
        mask_weight=1.0,
        cwd_weight=0.0,
        charbonnier_weight=0.0,
        fft_weight=0.0,
    )

    result = criterion(output, target)

    assert criterion.mask_temperature == pytest.approx(2.0)
    assert criterion.mask_scale_weights == (1.0, 0.5, 0.25)

    temperature = criterion.mask_temperature
    expected_bce: dict[str, torch.Tensor] = {}
    expected_dice: dict[str, torch.Tensor] = {}
    for index, name in enumerate(("h", "h2", "h4")):
        student = output.mask_logits[index].detach()
        teacher = target.mask_logits[index]

        bce = (
            F.binary_cross_entropy_with_logits(
                student / temperature,
                torch.sigmoid(teacher / temperature),
            )
            * temperature
            * temperature
        )
        dice = _manual_soft_dice(
            torch.sigmoid(student),
            torch.sigmoid(teacher),
            criterion.mask_dice_eps,
        )
        expected_bce[name] = bce
        expected_dice[name] = dice

        torch.testing.assert_close(
            result[f"mask_bce_{name}"], bce, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            result[f"mask_dice_{name}"], dice, rtol=1e-5, atol=1e-6
        )

    expected_bce_total = (
        1.0 * expected_bce["h"]
        + 0.5 * expected_bce["h2"]
        + 0.25 * expected_bce["h4"]
    )
    expected_dice_total = (
        1.0 * expected_dice["h"]
        + 0.5 * expected_dice["h2"]
        + 0.25 * expected_dice["h4"]
    )
    expected_mask = expected_bce_total + expected_dice_total

    torch.testing.assert_close(
        result["mask_bce"], expected_bce_total, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        result["mask_dice"], expected_dice_total, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        result["mask"], expected_mask, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        result["total"], expected_mask, rtol=1e-5, atol=1e-6
    )

    # A duplicate H target would add a second full-weight H term.
    duplicated = expected_mask + expected_bce["h"] + expected_dice["h"]
    assert not math.isclose(
        result["mask"].item(),
        duplicated.item(),
        rel_tol=1e-6,
        abs_tol=1e-7,
    )


# ---------------------------------------------------------------------------
# Feature distillation
# ---------------------------------------------------------------------------


def test_cwd_uses_temporary_adapters_and_spatial_cosine(
    stage1, teacher_target_cls
):
    target = _make_target(teacher_target_cls, seed=10)
    output = _make_output(stage1, seed=11)
    criterion = _criterion(stage1)

    assert isinstance(criterion.cwd_h8_adapter, nn.Conv2d)
    assert criterion.cwd_h8_adapter.in_channels == 3
    assert criterion.cwd_h8_adapter.out_channels == 5
    assert criterion.cwd_h8_adapter.kernel_size == (1, 1)
    assert isinstance(criterion.cwd_h4_adapter, nn.Conv2d)
    assert criterion.cwd_h4_adapter.in_channels == 4
    assert criterion.cwd_h4_adapter.out_channels == 7
    assert criterion.cwd_h_adapter is None

    result = criterion(output, target)

    adapted_h8 = F.conv2d(
        output.feature_h8.detach().float(),
        criterion.cwd_h8_adapter.weight.detach().float(),
    )
    adapted_h4 = F.conv2d(
        output.feature_h4.detach().float(),
        criterion.cwd_h4_adapter.weight.detach().float(),
    )
    torch.testing.assert_close(
        result["cwd_h8"],
        _manual_cwd(
            adapted_h8,
            target.bottleneck_h8,
            criterion.cwd_temperature,
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["cwd_h4"],
        _manual_cwd(
            adapted_h4,
            target.bg_decoder_h4,
            criterion.cwd_temperature,
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result["cwd_h"].item() == 0.0
    torch.testing.assert_close(
        result["cwd"], result["cwd_h8"] + result["cwd_h4"]
    )

    # Taps may be at another spatial size; CWD resizes to the teacher tap.
    resized = replace(
        output,
        feature_h8=torch.randn(
            BATCH, 3, 2, 3, generator=torch.Generator().manual_seed(12)
        ),
    )
    resized_result = criterion(resized, target)
    adapted_small = F.conv2d(
        resized.feature_h8.detach().float(),
        criterion.cwd_h8_adapter.weight.detach().float(),
    )
    torch.testing.assert_close(
        resized_result["cwd_h8"],
        _manual_cwd(
            adapted_small,
            target.bottleneck_h8,
            criterion.cwd_temperature,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    with pytest.raises(ValueError, match="CWD adapter expects"):
        criterion(
            replace(
                output,
                feature_h8=torch.randn(BATCH, 2, 4, 6),
            ),
            target,
        )


def test_full_h_cwd_is_scheduled_every_n_steps(
    stage1, teacher_target_cls
):
    target = _make_target(teacher_target_cls, seed=13)
    output = _make_output(stage1, seed=14)
    full_h = torch.randn(
        BATCH, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(15)
    )
    criterion = _criterion(stage1, full_h_every=2)

    assert isinstance(criterion.cwd_h_adapter, nn.Conv2d)
    assert criterion.cwd_h_adapter.in_channels == 2
    assert criterion.cwd_h_adapter.out_channels == 3

    for step, active in ((0, True), (1, False), (2, True), (3, False)):
        result = criterion(
            output, target, teacher_full_h=full_h, step=step
        )
        if active:
            assert result["cwd_h"].item() > 0.0, step
        else:
            assert result["cwd_h"].item() == 0.0, step
        assert torch.isfinite(result["cwd"]).all()

    # The stored step is used when no explicit step is passed.
    criterion.set_step(2)
    assert (
        criterion(output, target, teacher_full_h=full_h)["cwd_h"].item()
        > 0.0
    )

    # Cached targets have no full-H tap, and full_h_every=0 disables it.
    assert (
        criterion(output, target, step=0)["cwd_h"].item() == 0.0
    )
    disabled = _criterion(stage1, full_h_every=0)
    assert disabled.cwd_h_adapter is None
    assert (
        disabled(output, target, teacher_full_h=full_h, step=0)[
            "cwd_h"
        ].item()
        == 0.0
    )

    with pytest.raises(ValueError, match="teacher feature_h"):
        criterion(
            output,
            target,
            teacher_full_h=torch.randn(BATCH, 99, HEIGHT, WIDTH),
            step=0,
        )


# ---------------------------------------------------------------------------
# Output distillation
# ---------------------------------------------------------------------------


def test_charbonnier_matches_dilated_teacher_mask(stage1, teacher_target_cls):
    target = _make_target(teacher_target_cls, seed=16)
    output = _make_output(stage1, seed=17)
    criterion = _criterion(
        stage1,
        mask_weight=0.0,
        cwd_weight=0.0,
        fft_weight=0.0,
    )

    result = criterion(output, target)

    assert criterion.dilation_radius == 5
    expected_refine = _manual_charbonnier(
        output.refine_bg.detach(),
        target.refine_bg,
        target.mask_logits[0],
        radius=criterion.dilation_radius,
        unmasked_weight=criterion.unmasked_weight,
        eps=criterion.charbonnier_eps,
    )
    expected_coarse = _manual_charbonnier(
        output.coarse_bg.detach(),
        target.coarse_bg,
        target.mask_logits[0],
        radius=criterion.dilation_radius,
        unmasked_weight=criterion.unmasked_weight,
        eps=criterion.charbonnier_eps,
    )
    torch.testing.assert_close(
        result["charbonnier_refine"], expected_refine, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        result["charbonnier_coarse"], expected_coarse, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        result["charbonnier"],
        0.5 * (expected_refine + expected_coarse),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["total"], result["charbonnier"], rtol=1e-6, atol=1e-7
    )

    # A perfect prediction hits the Charbonnier floor: eps inside the
    # dilated mask plus 0.1 * eps outside.
    mask_logits = torch.full((BATCH, 1, HEIGHT, WIDTH), -10.0)
    mask_logits[
        :,
        :,
        HEIGHT // 2 - 2 : HEIGHT // 2 + 2,
        WIDTH // 2 - 2 : WIDTH // 2 + 2,
    ] = 10.0
    local_target = replace(
        target,
        mask_logits=(
            mask_logits,
            target.mask_logits[1],
            target.mask_logits[2],
        ),
    )
    perfect = replace(
        output,
        refine_bg=local_target.refine_bg.clone(),
        coarse_bg=local_target.coarse_bg.clone(),
    )
    perfect_result = criterion(perfect, local_target)
    floor = 1.1 * criterion.charbonnier_eps
    assert perfect_result["charbonnier"].item() == pytest.approx(
        floor, rel=1e-5
    )


def test_fft_is_fp32_under_autocast(monkeypatch, stage1, teacher_target_cls):
    target = _make_target(teacher_target_cls, seed=18)
    output = _make_output(stage1, seed=19)
    seen_dtypes: list[torch.dtype] = []
    original_rfft2 = torch.fft.rfft2

    def spy(input, *args, **kwargs):
        seen_dtypes.append(input.dtype)
        return original_rfft2(input, *args, **kwargs)

    monkeypatch.setattr(torch.fft, "rfft2", spy)
    criterion = _criterion(stage1, fft_scales=(1.0, 0.5))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = criterion(output, target)

    assert seen_dtypes
    assert all(dtype == torch.float32 for dtype in seen_dtypes)
    assert len(seen_dtypes) == 2 * 2  # two scales, prediction + target
    assert result["fft"].dtype == torch.float32
    assert result["fft"].item() > 0.0


def test_fft_only_backprops_into_refine(stage1, teacher_target_cls):
    target = _make_target(teacher_target_cls, seed=20)
    output = _make_output(stage1, seed=21)
    criterion = _criterion(
        stage1,
        mask_weight=0.0,
        cwd_weight=0.0,
        charbonnier_weight=0.0,
        fft_weight=1.0,
    )

    result = criterion(output, target)
    assert result["fft"].item() > 0.0
    result["total"].backward()

    assert output.refine_bg.grad is not None
    assert torch.isfinite(output.refine_bg.grad).all()
    assert torch.count_nonzero(output.refine_bg.grad).item() > 0

    for tensor in (
        output.coarse_bg,
        *output.mask_logits,
        output.feature_h8,
        output.feature_h4,
        output.feature_h,
    ):
        if tensor.grad is not None:
            assert torch.count_nonzero(tensor.grad).item() == 0


# ---------------------------------------------------------------------------
# Cached targets, geometry, validation
# ---------------------------------------------------------------------------


def test_cached_teacher_payload_is_accepted(stage1, teacher_target_cls):
    target = _make_target(teacher_target_cls, seed=22)
    output = _make_output(stage1, seed=23)
    criterion = _criterion(stage1, full_h_every=2)
    full_h = torch.randn(
        BATCH, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(24)
    )

    direct = criterion(output, target, teacher_full_h=full_h, step=0)
    cached = criterion(
        output, target.to_payload(), teacher_full_h=full_h, step=0
    )

    assert set(cached) == LOSS_KEYS
    for key in direct:
        torch.testing.assert_close(cached[key], direct[key])


def test_non_square_odd_geometry(stage1, teacher_target_cls):
    height, width = 30, 46
    target = _make_target(
        teacher_target_cls, height=height, width=width, seed=25
    )
    output = _make_output(stage1, height=height, width=width, seed=26)
    criterion = _criterion(stage1, full_h_every=1)

    result = criterion(output, target, step=0)

    assert set(result) == LOSS_KEYS
    for key, value in result.items():
        assert value.shape == (), key
        assert torch.isfinite(value).all(), key
    result["total"].backward()
    assert torch.isfinite(output.refine_bg.grad).all()
    assert torch.isfinite(output.coarse_bg.grad).all()


def test_rejects_bad_configuration_and_shapes(stage1, teacher_target_cls):
    with pytest.raises(ValueError):
        stage1.Stage1Loss(mask_scale_weights=(1.0, 0.5))
    with pytest.raises(ValueError):
        stage1.Stage1Loss(full_h_every=-1)
    with pytest.raises(ValueError):
        stage1.Stage1Loss(fft_scales=())
    with pytest.raises(ValueError):
        stage1.Stage1Loss(mask_temperature=0.0)

    target = _make_target(teacher_target_cls, seed=27)
    output = _make_output(stage1, seed=28)

    with pytest.raises(ValueError, match="exactly three levels"):
        stage1.StudentStage1Output(
            refine_bg=output.refine_bg,
            coarse_bg=output.coarse_bg,
            mask_logits=(
                output.mask_logits[0],
                output.mask_logits[1],
                output.mask_logits[2],
                output.mask_logits[0],
            ),
        )

    criterion = _criterion(stage1)
    with pytest.raises(TypeError, match="StudentStage1Output"):
        criterion(
            (
                [output.refine_bg, output.coarse_bg],
                [torch.sigmoid(level) for level in output.mask_logits],
                [None],
            ),
            target,
        )

    # Student logits matching the student resolution but not the teacher's
    # resolution are rejected instead of silently resized.
    other_target = _make_target(
        teacher_target_cls, height=34, width=WIDTH, seed=29
    )
    with pytest.raises(ValueError, match="spatial mismatch"):
        criterion(output, other_target)


# ---------------------------------------------------------------------------
# Student feature extraction helper
# ---------------------------------------------------------------------------


class _FakeMaskDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        logits = self.conv(x)
        return SimpleNamespace(
            mask_logits=logits,
            mask_logits_h2=F.interpolate(
                logits,
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            ),
            mask_logits_h4=F.interpolate(
                logits,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            ),
        )


class _FakeBgTrunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 1)

    def forward(self, x):
        bg_h8 = F.avg_pool2d(self.conv(x), kernel_size=8)
        return bg_h8, bg_h8


class _FakeBottleneck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bg_trunk = _FakeBgTrunk()

    def forward_features(self, x):
        return self.bg_trunk(x)

    def forward(self, x):
        return self.forward_features(x)


class _FakeBackgroundDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(5, 7, 1)

    def forward(self, bg_h8):
        h4 = F.interpolate(
            self.conv(bg_h8),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        h2 = F.interpolate(
            h4, scale_factor=2, mode="bilinear", align_corners=False
        )
        h = F.interpolate(
            h2, scale_factor=2, mode="bilinear", align_corners=False
        )
        return SimpleNamespace(
            coarse_bg=torch.sigmoid(h),
            skip_h4=h4,
            skip_h2=h2,
            skip_h=h,
        )


class _FakeStudent(nn.Module):
    """Mimics the v1 call pattern: ``forward_features`` -> ``bg_trunk``."""

    def __init__(self) -> None:
        super().__init__()
        self.mask_decoder = _FakeMaskDecoder()
        self.bottleneck = _FakeBottleneck()
        self.background_decoder = _FakeBackgroundDecoder()
        self.head = nn.Conv2d(7, 3, 1)

    def forward(self, x):
        masks = self.mask_decoder(x)
        bg_h8, _ = self.bottleneck.forward_features(x)
        background = self.background_decoder(bg_h8)
        refine = torch.sigmoid(self.head(background.skip_h))
        coarse = torch.sigmoid(self.head(background.skip_h) + 0.1)
        return (
            [refine, coarse],
            [
                torch.sigmoid(masks.mask_logits),
                torch.sigmoid(masks.mask_logits_h2),
                torch.sigmoid(masks.mask_logits_h4),
            ],
            [None],
        )


def test_run_student_stage1_captures_logits_and_features(stage1):
    model = _FakeStudent()
    x = torch.rand(
        BATCH, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(30)
    )

    output = stage1.run_student_stage1(model, x)

    assert output.refine_bg.shape == (BATCH, 3, HEIGHT, WIDTH)
    assert output.coarse_bg.shape == (BATCH, 3, HEIGHT, WIDTH)
    assert [tuple(level.shape) for level in output.mask_logits] == [
        (BATCH, 1, HEIGHT, WIDTH),
        (BATCH, 1, HEIGHT // 2, WIDTH // 2),
        (BATCH, 1, HEIGHT // 4, WIDTH // 4),
    ]
    assert output.feature_h8.shape == (BATCH, 5, HEIGHT // 8, WIDTH // 8)
    assert output.feature_h4.shape == (BATCH, 7, HEIGHT // 4, WIDTH // 4)
    assert output.feature_h is None
    assert output.refine_bg.requires_grad

    for module in (
        model.mask_decoder,
        model.bottleneck.bg_trunk,
        model.background_decoder,
    ):
        assert not module._forward_hooks

    full = stage1.run_student_stage1(model, x, include_full_h=True)
    assert full.feature_h.shape == (BATCH, 7, HEIGHT, WIDTH)

    # Repeated calls are safe: hooks were removed after the first call.
    again = stage1.run_student_stage1(model, x)
    assert again.feature_h is None

    with pytest.raises(AttributeError):
        stage1.run_student_stage1(nn.Conv2d(3, 3, 1), x)


# ---------------------------------------------------------------------------
# CUDA/FP16 (skipped on the CPU runner)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fp16_autocast_is_finite(stage1, teacher_target_cls):
    target = _make_target(teacher_target_cls, seed=31, device="cuda")
    output = _make_output(stage1, seed=32, device="cuda")
    full_h = torch.randn(
        BATCH, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(33)
    ).to("cuda")
    criterion = _criterion(stage1, full_h_every=1).to("cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = criterion(output, target, teacher_full_h=full_h, step=0)

    for key, value in result.items():
        assert value.dtype == torch.float32, key
        assert torch.isfinite(value).all(), key

    result["total"].backward()
    for tensor in (
        output.refine_bg,
        output.coarse_bg,
        *output.mask_logits,
        output.feature_h8,
        output.feature_h4,
        output.feature_h,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    for parameter in criterion.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
