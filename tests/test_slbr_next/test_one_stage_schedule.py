"""Tests for the optional one-stage supervised schedule (issue #13, spec §5.5).

The CPU tests use tiny tensors and a tiny model so they finish in a few
seconds. ``src.models.slbr_next`` is imported lazily inside fixtures/tests:
``test_imports.py`` asserts that ``src.models`` is not pulled in while
importing ``src.networks.slbr_next``, and pytest imports every test module
during collection.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

BATCH = 2
HEIGHT = 16
WIDTH = 16
H8_CHANNELS = 4


# ---------------------------------------------------------------------------
# Lazy imports and small fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def modules():
    from src.models.slbr_next import losses_stage1, losses_stage2, teacher

    return SimpleNamespace(
        losses_stage1=losses_stage1,
        losses_stage2=losses_stage2,
        teacher=teacher,
    )


@pytest.fixture(scope="module")
def trainer_module():
    from src.models.slbr_next import trainer

    return trainer


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
    h8_channels: int = H8_CHANNELS,
):
    generator = torch.Generator().manual_seed(seed)
    sizes = _level_sizes(height, width)
    h8 = ((height + 7) // 8, (width + 7) // 8)
    h4 = sizes[2]

    def randn(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.randn(
            batch, channels, h, w, generator=generator
        )

    def rand(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.rand(batch, channels, h, w, generator=generator)

    return TeacherTarget(
        mask_logits=tuple(randn(1, h, w) for h, w in sizes),
        primary_logits=tuple(randn(1, h, w) for h, w in sizes),
        refine_bg=rand(3, height, width),
        coarse_bg=rand(3, height, width),
        bottleneck_h8=randn(h8_channels, *h8),
        bg_decoder_h4=randn(h8_channels, *h4),
        height=height,
        width=width,
    )


def _make_output(
    StudentStage1Output,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    batch: int = BATCH,
    seed: int = 1,
    h8_channels: int = H8_CHANNELS,
    requires_grad: bool = True,
):
    generator = torch.Generator().manual_seed(seed)
    sizes = _level_sizes(height, width)
    h8 = ((height + 7) // 8, (width + 7) // 8)
    h4 = sizes[2]

    def randn(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.randn(
            batch, channels, h, w, generator=generator
        )

    def rand(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.rand(batch, channels, h, w, generator=generator)

    refine = rand(3, height, width)
    coarse = rand(3, height, width)
    mask_logits = tuple(randn(1, h, w) for h, w in sizes)
    feature_h8 = randn(h8_channels, *h8)
    feature_h4 = randn(h8_channels, *h4)
    tensors = [refine, coarse, *mask_logits, feature_h8, feature_h4]
    if requires_grad:
        for tensor in tensors:
            tensor.requires_grad_(True)

    return StudentStage1Output(
        refine_bg=refine,
        coarse_bg=coarse,
        mask_logits=mask_logits,
        feature_h8=feature_h8,
        feature_h4=feature_h4,
        feature_h=None,
    )


class _PoisonBatch(dict):
    """A mapping that fails loudly if a forbidden key is read."""

    def __init__(self, *args, poisoned=(), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.poisoned = set(poisoned)

    def __getitem__(self, key):
        if key in self.poisoned:
            raise AssertionError(f"forbidden batch key was read: {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.poisoned:
            raise AssertionError(f"forbidden batch key was read: {key!r}")
        return super().get(key, default)


class CountingTeacher:
    """Callable teacher double recording calls and raw inputs."""

    def __init__(self, target) -> None:
        self.target = target
        self.calls = 0
        self.inputs: list[torch.Tensor] = []

    def __call__(self, images: torch.Tensor):
        self.calls += 1
        self.inputs.append(images.detach().clone())
        return self.target


class _ModuleTeacher(nn.Module):
    """An nn.Module teacher used to check frozen/eval/no-norm behavior."""

    def __init__(self, target) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.target = target
        self.inputs: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor):
        self.inputs.append(images.detach().clone())
        return self.target


def _make_batch(
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    batch: int = BATCH,
    seed: int = 2,
    teacher_target=None,
    poison_teacher: bool = False,
):
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand(batch, 3, height, width, generator=generator)
    clean = torch.rand(batch, 3, height, width, generator=generator)
    mask = (
        torch.rand(batch, 1, height, width, generator=generator) > 0.7
    ).float()
    data = {"image": image, "clean": clean, "mask": mask}
    if teacher_target is not None:
        data["teacher_target"] = teacher_target
    if poison_teacher:
        data["teacher_target"] = object()
        return _PoisonBatch(data, poisoned=("teacher_target",))
    return data


def _one_stage(
    OneStageLoss,
    Stage2Loss,
    *,
    teacher=None,
    decay: int = 7,
    enable_teacher: bool = True,
    student_h8_channels: int = H8_CHANNELS,
    teacher_h8_channels: int = H8_CHANNELS,
):
    return OneStageLoss(
        supervised=Stage2Loss(perceptual=None),
        teacher=teacher,
        enable_teacher=enable_teacher,
        teacher_decay_epochs=decay,
        student_h8_channels=student_h8_channels,
        teacher_h8_channels=teacher_h8_channels,
    )


# ---------------------------------------------------------------------------
# Decay schedule and hard detachment
# ---------------------------------------------------------------------------


def test_teacher_weight_decays_linearly_to_exact_zero(modules):
    one_stage = modules.losses_stage1
    stage2 = modules.losses_stage2

    criterion = _one_stage(one_stage.OneStageLoss, stage2.Stage2Loss)
    assert criterion.epoch == 0
    assert criterion.teacher_weight == 1.0
    assert criterion.teacher_active
    assert not criterion.teacher_detached

    for epoch in range(7):
        criterion.set_epoch(epoch)
        assert criterion.teacher_weight == pytest.approx(1.0 - epoch / 7.0)
        assert criterion.teacher_active
        assert not criterion.teacher_detached

    criterion.set_epoch(7)
    assert criterion.teacher_weight == 0.0
    assert not criterion.teacher_active
    assert criterion.teacher_detached

    criterion.set_epoch(9)
    assert criterion.teacher_weight == 0.0
    assert not criterion.teacher_active

    with pytest.raises(ValueError, match="non-negative"):
        criterion.set_epoch(-1)

    # The window is configurable (spec §5.5 allows 5-10 epochs).
    short = _one_stage(
        one_stage.OneStageLoss, stage2.Stage2Loss, decay=5
    )
    short.set_epoch(4)
    assert short.teacher_weight == pytest.approx(0.2)
    short.set_epoch(5)
    assert short.teacher_weight == 0.0

    disabled = _one_stage(
        one_stage.OneStageLoss,
        stage2.Stage2Loss,
        enable_teacher=False,
    )
    assert disabled.teacher_weight == 0.0
    assert not disabled.teacher_active
    assert disabled.teacher_detached

    # The trainer's progress hook reaches the supervised late-perceptual
    # schedule unchanged.
    criterion.set_progress(1, 2)
    assert criterion.supervised.progress == pytest.approx(0.5)

    with pytest.raises(ValueError, match="positive"):
        _one_stage(
            one_stage.OneStageLoss,
            stage2.Stage2Loss,
            decay=0,
        )


def test_teacher_forward_is_skipped_after_decay(modules):
    one_stage = modules.losses_stage1
    stage2 = modules.losses_stage2
    teacher = modules.teacher

    target = _make_target(teacher.TeacherTarget)
    counter = CountingTeacher(target)
    criterion = _one_stage(
        one_stage.OneStageLoss, stage2.Stage2Loss, teacher=counter
    )
    outputs = _make_output(one_stage.StudentStage1Output)
    batch = _make_batch(poison_teacher=True)

    criterion.set_epoch(7)
    assert criterion.teacher_detached
    result = criterion(outputs, batch)

    assert counter.calls == 0
    assert float(result["teacher"].detach()) == 0.0
    assert float(result["teacher_mask"].detach()) == 0.0
    assert float(result["teacher_cwd_h8"].detach()) == 0.0
    assert float(result["teacher_weight"]) == 0.0
    assert torch.equal(result["loss"], result["supervised"])

    # One epoch before the window ends, the teacher is still used and the
    # auxiliary contributes a positive regularizer.
    criterion.set_epoch(6)
    active_result = criterion(outputs, _make_batch())
    assert counter.calls == 1
    assert float(active_result["teacher_weight"]) == pytest.approx(1.0 / 7.0)
    assert float(active_result["teacher"].detach()) > 0.0


def test_batch_teacher_target_is_not_read_after_decay(modules):
    one_stage = modules.losses_stage1
    stage2 = modules.losses_stage2

    criterion = _one_stage(one_stage.OneStageLoss, stage2.Stage2Loss)
    criterion.set_epoch(7)
    outputs = _make_output(one_stage.StudentStage1Output)
    batch = _make_batch(poison_teacher=True)

    result = criterion(outputs, batch)
    assert float(result["teacher"].detach()) == 0.0


# ---------------------------------------------------------------------------
# Combined loss, gradients, and no output imitation
# ---------------------------------------------------------------------------


def test_one_stage_combines_supervised_and_teacher_terms(modules):
    one_stage = modules.losses_stage1
    stage2 = modules.losses_stage2
    teacher = modules.teacher

    target = _make_target(teacher.TeacherTarget)
    counter = CountingTeacher(target)
    criterion = _one_stage(
        one_stage.OneStageLoss, stage2.Stage2Loss, teacher=counter
    )
    outputs = _make_output(one_stage.StudentStage1Output)
    batch = _make_batch()

    result = criterion(outputs, batch)
    assert counter.calls == 1
    assert criterion.teacher_active
    assert float(result["teacher_weight"]) == 1.0

    for key in (
        "loss",
        "total",
        "supervised",
        "teacher",
        "teacher_mask",
        "teacher_mask_bce",
        "teacher_mask_dice",
        "teacher_cwd_h8",
        "reconstruction",
        "perceptual",
        "fft",
        "mask",
        "alpha",
    ):
        assert key in result, key
        assert result[key].numel() == 1
        assert torch.isfinite(result[key]).all()

    assert float(result["teacher_mask_bce"].detach()) > 0.0
    assert float(result["teacher_mask_dice"].detach()) > 0.0
    assert float(result["teacher_cwd_h8"].detach()) >= 0.0

    expected = result["supervised"] + result["teacher_weight"] * (
        result["teacher_mask"] + 0.1 * result["teacher_cwd_h8"]
    )
    torch.testing.assert_close(result["loss"], expected)
    torch.testing.assert_close(result["total"], result["loss"])
    assert torch.equal(result["loss"], result["supervised"] + result["teacher"])

    result["loss"].backward()
    grad_tensors = [
        outputs.refine_bg,
        outputs.coarse_bg,
        *outputs.mask_logits,
        outputs.feature_h8,
    ]
    for tensor in grad_tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.abs().sum()) > 0.0


def test_one_stage_has_no_teacher_output_imitation(modules):
    one_stage = modules.losses_stage1
    stage2 = modules.losses_stage2
    teacher = modules.teacher

    target = _make_target(teacher.TeacherTarget)
    criterion = _one_stage(one_stage.OneStageLoss, stage2.Stage2Loss)
    outputs = _make_output(one_stage.StudentStage1Output)
    batch = _make_batch(teacher_target=target)
    result = criterion(outputs, batch)

    # Mutating both teacher image outputs must not move the loss: only mask
    # logits and the H/8 feature tap are consumed.
    poisoned_target = replace(
        target,
        refine_bg=torch.full_like(target.refine_bg, 0.123),
        coarse_bg=torch.full_like(target.coarse_bg, 0.987),
    )
    poisoned_batch = dict(batch)
    poisoned_batch["teacher_target"] = poisoned_target
    poisoned = criterion(outputs, poisoned_batch)

    torch.testing.assert_close(result["loss"], poisoned["loss"], rtol=0, atol=0)
    for key in result:
        assert "imitation" not in key
        assert "teacher_charbonnier" not in key
        assert "teacher_fft" not in key
        assert "teacher_output" not in key
    # The teacher target carries those tensors, but the loss must not add
    # component keys for them.
    assert "charbonnier_teacher" not in result
    assert "fft_teacher" not in result


# ---------------------------------------------------------------------------
# Frozen teacher, no normalization, and validation isolation
# ---------------------------------------------------------------------------


def test_module_teacher_is_frozen_eval_and_receives_raw_rgb(modules):
    one_stage = modules.losses_stage1
    stage2 = modules.losses_stage2
    teacher = modules.teacher

    target = _make_target(teacher.TeacherTarget)
    module_teacher = _ModuleTeacher(target)
    criterion = _one_stage(
        one_stage.OneStageLoss, stage2.Stage2Loss, teacher=module_teacher
    )
    outputs = _make_output(one_stage.StudentStage1Output)
    batch = _make_batch()

    assert module_teacher.training is False
    assert all(
        not parameter.requires_grad
        for parameter in module_teacher.parameters()
    )
    assert not any(
        parameter is module_teacher.scale
        for parameter in criterion.parameters()
    )

    # criterion.train() must not put an unregistered teacher back in train
    # mode or add it to the optimizer parameter list.
    criterion.train()
    assert module_teacher.training is False

    result = criterion(outputs, batch)
    assert len(module_teacher.inputs) == 1
    assert torch.equal(module_teacher.inputs[0], batch["image"])
    assert torch.isfinite(result["teacher"]).all()


def test_validation_uses_only_supervised_terms(modules):
    one_stage = modules.losses_stage1
    stage2 = modules.losses_stage2
    teacher = modules.teacher

    target = _make_target(teacher.TeacherTarget)
    counter = CountingTeacher(target)
    criterion = _one_stage(
        one_stage.OneStageLoss, stage2.Stage2Loss, teacher=counter
    )
    criterion.set_epoch(0)
    criterion.eval()
    assert criterion.teacher_weight == 1.0
    assert not criterion.teacher_active
    outputs = _make_output(one_stage.StudentStage1Output)
    batch = _make_batch(poison_teacher=True)

    result = criterion(outputs, batch)

    assert counter.calls == 0
    assert float(result["teacher"].detach()) == 0.0
    assert float(result["teacher_mask"].detach()) == 0.0
    assert float(result["teacher_cwd_h8"].detach()) == 0.0
    assert float(result["teacher_weight"]) == 0.0

    expected = criterion.supervised(
        outputs.refine_bg,
        outputs.coarse_bg,
        outputs.mask_logits,
        batch["clean"],
        batch["mask"],
    )["total"]
    torch.testing.assert_close(result["loss"], expected)
    assert torch.equal(result["loss"], result["supervised"])


# ---------------------------------------------------------------------------
# Config switch and CLI plumbing
# ---------------------------------------------------------------------------


def test_one_stage_supervises_synthetic_composites(modules):
    one_stage = modules.losses_stage1
    stage2 = modules.losses_stage2
    teacher = modules.teacher
    build_synthetic = modules.losses_stage2.build_synthetic_composite

    generator = torch.Generator().manual_seed(11)
    background = torch.rand(BATCH, 3, HEIGHT, WIDTH, generator=generator)
    watermark = torch.rand(BATCH, 3, HEIGHT, WIDTH, generator=generator)
    composite = build_synthetic(
        background,
        watermark,
        modes=("linear", "screen"),
        alpha=0.4,
    )
    assert composite.alpha is not None
    assert composite.alpha_valid.tolist() == [True, False]

    mask = (composite.image - background).abs().mean(dim=1, keepdim=True)
    mask = (mask > 0.05).float()
    alpha_pred = torch.rand(
        BATCH, 1, HEIGHT, WIDTH, generator=generator
    )
    batch = {
        "image": composite.image,
        "clean": background,
        "mask": mask,
        "alpha_pred": alpha_pred,
        "alpha_gt": composite.alpha,
        "alpha_valid": composite.alpha_valid,
        "teacher_target": _make_target(teacher.TeacherTarget),
    }

    criterion = _one_stage(one_stage.OneStageLoss, stage2.Stage2Loss)
    outputs = _make_output(one_stage.StudentStage1Output)
    result = criterion(outputs, batch)

    assert torch.isfinite(result["loss"]).all()
    assert float(result["alpha"].detach()) > 0.0
    assert float(result["alpha_l1"].detach()) > 0.0
    # Nonlinear rows carry no alpha target and must not be supervised.
    assert float(result["alpha_grad"].detach()) >= 0.0
    result["loss"].backward()
    assert outputs.refine_bg.grad is not None


def test_trainer_config_schedule_switch(trainer_module):
    config = trainer_module.TrainerConfig()
    assert config.schedule == "two_stage"
    assert config.one_stage is False
    assert config.teacher_decay_epochs == 7

    one = trainer_module.TrainerConfig(
        schedule="one_stage", teacher_decay_epochs=5
    )
    assert one.one_stage is True
    assert one.schedule == "one_stage"
    assert one.teacher_decay_epochs == 5

    restored = trainer_module.TrainerConfig.from_json(one.to_json())
    assert restored == one

    with pytest.raises(ValueError, match="schedule"):
        trainer_module.TrainerConfig(schedule="one-stage")
    with pytest.raises(ValueError, match="teacher_decay_epochs"):
        trainer_module.TrainerConfig(teacher_decay_epochs=0)


def test_cli_parses_both_schedules(trainer_module):
    import train_slbr_next

    parser = train_slbr_next.build_parser()
    args = parser.parse_args(
        [
            "--schedule",
            "one_stage",
            "--teacher-decay-epochs",
            "5",
            "--perceptual",
            "none",
            "--subset-size",
            "512",
        ]
    )
    assert args.schedule == "one_stage"
    assert args.teacher_decay_epochs == 5
    assert args.perceptual == "none"
    assert args.subset_size == 512

    config = train_slbr_next._apply_config_overrides(
        trainer_module.TrainerConfig(), args
    )
    assert config.schedule == "one_stage"
    assert config.one_stage is True
    assert config.teacher_decay_epochs == 5

    default_args = parser.parse_args([])
    default_config = train_slbr_next._apply_config_overrides(
        trainer_module.TrainerConfig(), default_args
    )
    assert default_config.schedule == "two_stage"
    assert default_config.one_stage is False
    assert default_config.teacher_decay_epochs == 7

    two_stage_args = parser.parse_args(["--schedule", "two_stage"])
    assert two_stage_args.schedule == "two_stage"


def test_build_schedule_loss_factory(modules, trainer_module):
    one_stage = modules.losses_stage1

    one = one_stage.build_schedule_loss(
        SimpleNamespace(
            config={"schedule": "one_stage", "teacher_decay_epochs": 5}
        )
    )
    assert isinstance(one, one_stage.OneStageLoss)
    assert one.teacher_decay_epochs == 5
    assert one.forward_fn is one_stage.run_student_stage1

    two = one_stage.build_schedule_loss(
        SimpleNamespace(config={"schedule": "two_stage"})
    )
    assert isinstance(two, one_stage.TwoStageScheduleLoss)
    assert two.forward_fn is one_stage.run_student_stage1

    # Alternative argument shape used outside the CLI.
    direct = one_stage.build_schedule_loss(
        SimpleNamespace(schedule="one_stage", teacher_decay_epochs=6)
    )
    assert isinstance(direct, one_stage.OneStageLoss)
    assert direct.teacher_decay_epochs == 6

    with pytest.raises(ValueError, match="schedule"):
        one_stage.build_schedule_loss(
            SimpleNamespace(config={"schedule": "bogus"})
        )


class _ScheduleSpyLoss(nn.Module):
    """Records the trainer's optional epoch/progress hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.epochs: list[int] = []
        self.progress: list[tuple[int, int]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(int(epoch))

    def set_progress(self, step: float, total: float) -> None:
        self.progress.append((int(step), int(total)))

    def forward(self, outputs, batch):
        del batch
        return {"loss": outputs.square().mean() * 0.0}


def test_trainer_calls_loss_schedule_hooks(
    trainer_module, tmp_path: Path
):
    model = nn.Conv2d(3, 3, 1)
    loss = _ScheduleSpyLoss()
    batch = {"image": torch.rand(BATCH, 3, 8, 8)}
    config = _micro_config(trainer_module, tmp_path / "hooks", "two_stage")
    trainer = trainer_module.Trainer(
        model=model,
        loss=loss,
        train_loader=[batch],
        val_loader=[batch],
        config=config,
        device="cpu",
    )

    trainer.train()

    # One training epoch: set_epoch(0) before the step and set_epoch(1) for
    # validation; set_progress(0, 1) before the first optimizer step.
    assert loss.epochs == [0, 1]
    assert loss.progress == [(0, 1)]
    assert trainer.global_step == 1


# ---------------------------------------------------------------------------
# run_student_stage1 wrapper support
# ---------------------------------------------------------------------------


class _FakeMaskDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        logits = self.conv(x)
        return SimpleNamespace(
            mask_logits=logits,
            mask_logits_h2=F.avg_pool2d(logits, 2),
            mask_logits_h4=F.avg_pool2d(logits, 4),
        )


class _FakeBgTrunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, H8_CHANNELS, 1)

    def forward(self, x):
        feature = F.avg_pool2d(self.conv(x), 8)
        return feature, feature


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
        self.conv = nn.Conv2d(H8_CHANNELS, H8_CHANNELS, 1)

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
        return SimpleNamespace(skip_h4=h4, skip_h2=h2, skip_h=h)


class _FakeStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask_decoder = _FakeMaskDecoder()
        self.bottleneck = _FakeBottleneck()
        self.background_decoder = _FakeBackgroundDecoder()
        self.head = nn.Conv2d(H8_CHANNELS, 3, 1)

    def forward(self, x):
        masks = self.mask_decoder(x)
        bg_h8, _ = self.bottleneck.forward_features(x)
        background = self.background_decoder(bg_h8)
        refine = torch.sigmoid(self.head(background.skip_h))
        coarse = torch.sigmoid(self.head(background.skip_h))
        return (
            [refine, coarse],
            [
                torch.sigmoid(masks.mask_logits),
                torch.sigmoid(masks.mask_logits_h2),
                torch.sigmoid(masks.mask_logits_h4),
            ],
            [None],
        )


class _Wrapper(nn.Module):
    """Mimics DDP/compile wrapping with a ``.module`` child."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def test_run_student_stage1_unwraps_forward_wrapper(modules):
    stage1 = modules.losses_stage1
    student = _FakeStudent()
    wrapped = _Wrapper(student)
    x = torch.rand(BATCH, 3, HEIGHT, WIDTH)

    output = stage1.run_student_stage1(wrapped, x)

    assert output.refine_bg.shape == (BATCH, 3, HEIGHT, WIDTH)
    assert [tuple(level.shape) for level in output.mask_logits] == [
        (BATCH, 1, HEIGHT, WIDTH),
        (BATCH, 1, HEIGHT // 2, WIDTH // 2),
        (BATCH, 1, HEIGHT // 4, WIDTH // 4),
    ]
    assert output.feature_h8.shape == (
        BATCH,
        H8_CHANNELS,
        HEIGHT // 8,
        WIDTH // 8,
    )
    assert output.feature_h4.shape == (
        BATCH,
        H8_CHANNELS,
        HEIGHT // 4,
        WIDTH // 4,
    )
    for module in (
        student.mask_decoder,
        student.bottleneck.bg_trunk,
        student.background_decoder,
    ):
        assert not module._forward_hooks


# ---------------------------------------------------------------------------
# CPU micro A/B: one optimizer step for each schedule, curves logged
# ---------------------------------------------------------------------------


class _TinyStage1Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.refine = nn.Conv2d(8, 3, 1)
        self.coarse = nn.Conv2d(8, 3, 1)
        self.mask_h = nn.Conv2d(8, 1, 1)
        self.mask_h2 = nn.Conv2d(8, 1, 1)
        self.mask_h4 = nn.Conv2d(8, 1, 1)
        self.h8 = nn.Conv2d(8, H8_CHANNELS, 1)
        self.h4 = nn.Conv2d(8, H8_CHANNELS, 1)

    def forward_stage1(self, x, StudentStage1Output):
        features = self.stem(x)
        height, width = x.shape[-2:]
        pooled_h2 = F.avg_pool2d(features, 2)
        pooled_h4 = F.avg_pool2d(features, 4)
        pooled_h8 = F.adaptive_avg_pool2d(
            features,
            (max(1, height // 8), max(1, width // 8)),
        )
        return StudentStage1Output(
            refine_bg=torch.sigmoid(self.refine(features)),
            coarse_bg=torch.sigmoid(self.coarse(features)),
            mask_logits=(
                self.mask_h(features),
                self.mask_h2(pooled_h2),
                self.mask_h4(pooled_h4),
            ),
            feature_h8=self.h8(pooled_h8),
            feature_h4=self.h4(pooled_h4),
        )

    def forward(self, x, StudentStage1Output):
        output = self.forward_stage1(x, StudentStage1Output)
        return (
            [output.refine_bg, output.coarse_bg],
            [torch.sigmoid(level) for level in output.mask_logits],
            [None],
        )


def _micro_config(trainer_module, output_dir: Path, schedule: str):
    return trainer_module.TrainerConfig(
        epochs=1,
        max_steps=None,
        batch_size_per_gpu=BATCH,
        schedule=schedule,
        teacher_decay_epochs=7,
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        warmup_steps=0,
        min_lr=1e-6,
        grad_clip=1.0,
        ema_decay=0.9,
        amp="fp32",
        channels_last=False,
        cudnn_benchmark=False,
        compile=False,
        ddp=False,
        device="cpu",
        seed=7,
        log_every=1,
        val_every_epochs=1,
        validate_ema=False,
        early_stop_patience=0,
        output_dir=str(output_dir),
        save_best=False,
        save_every_epoch=False,
        save_on_exit=False,
    )


def test_cpu_micro_ab_one_step_each_schedule_logs_curves(
    modules, trainer_module, tmp_path: Path
):
    stage1 = modules.losses_stage1
    stage2 = modules.losses_stage2
    teacher = modules.teacher

    target = _make_target(teacher.TeacherTarget)
    batch = _make_batch()
    two_stage_loss = stage1.TwoStageScheduleLoss(
        criterion=stage1.Stage1Loss(
            student_h8_channels=H8_CHANNELS,
            student_h4_channels=H8_CHANNELS,
            student_h_channels=H8_CHANNELS,
            teacher_h8_channels=H8_CHANNELS,
            teacher_h4_channels=H8_CHANNELS,
            teacher_h_channels=H8_CHANNELS,
        )
    )
    two_stage_batch = dict(batch)
    two_stage_batch["teacher_target"] = target

    counter = CountingTeacher(target)
    one_stage_loss = stage1.OneStageLoss(
        supervised=stage2.Stage2Loss(perceptual=None),
        teacher=counter,
        teacher_decay_epochs=7,
        student_h8_channels=H8_CHANNELS,
        teacher_h8_channels=H8_CHANNELS,
    )

    def forward_fn(model, inputs):
        return model.forward_stage1(inputs, stage1.StudentStage1Output)

    def run_schedule(loss, train_batch, name: str):
        output_dir = tmp_path / name
        model = _TinyStage1Model()
        trainer = trainer_module.Trainer(
            model=model,
            loss=loss,
            train_loader=[train_batch],
            val_loader=[train_batch],
            config=_micro_config(
                trainer_module, output_dir, name.split("_")[0] + "_stage"
            ),
            device="cpu",
            forward_fn=forward_fn,
        )
        history = trainer.train()
        assert len(history) == 1
        record = history[0]
        assert math.isfinite(record["train_loss"])
        assert math.isfinite(record["val_loss"])

        lines = [
            json.loads(line)
            for line in (output_dir / "train_log.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        steps = [line for line in lines if line["event"] == "step"]
        epochs = [line for line in lines if line["event"] == "epoch"]
        assert len(steps) == 1
        assert len(epochs) == 1
        assert math.isfinite(steps[0]["loss"])
        assert steps[0]["components"]
        assert math.isfinite(epochs[0]["val_loss"])
        assert epochs[0]["train_loss"] is not None
        return steps[0], epochs[0]

    two_step, two_epoch = run_schedule(
        two_stage_loss, two_stage_batch, "two_stage"
    )
    assert "mask" in two_step["components"]
    assert "cwd" in two_step["components"]
    assert "charbonnier" in two_step["components"]

    one_step, one_epoch = run_schedule(
        one_stage_loss, batch, "one_stage"
    )
    components = one_step["components"]
    assert "supervised" in components
    assert "teacher" in components
    assert "teacher_cwd_h8" in components
    assert math.isfinite(components["teacher"])
    # Training used the teacher exactly once; validation ran under eval()
    # and must not have called it at all.
    assert counter.calls == 1
    assert one_epoch["teacher_weight"] == 0.0
    assert one_epoch["teacher"] == 0.0
