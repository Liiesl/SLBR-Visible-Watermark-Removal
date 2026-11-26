"""T6 training-gate tests (issue #14, spec §5.4/§6).

Core tests are CPU-only. They cover:

* one real ``SLBRNextV1`` + ``Stage2Loss`` optimizer step with finite grads;
* one ``Stage1Loss`` step with synthetic teacher targets/features;
* the T5 one-stage schedule through the trainer, including the teacher decay
  to exact zero and a poison check that the target is not read afterwards;
* matrix row parsing/selection and the JSON/markdown output contract.

``src.models...`` imports are deliberately inside fixtures/tests.
``tests/test_slbr_next/test_imports.py`` asserts that importing
``src.networks.slbr_next`` does not pull ``src.models`` into ``sys.modules``,
and pytest imports every test module during collection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench import throughput_matrix as matrix
from bench.common import Stage2ProbabilityLoss, synthetic_stage2_batch


@pytest.fixture(scope="module")
def slbr_modules():
    from src.models.slbr_next import losses_stage1, losses_stage2, teacher, trainer

    return SimpleNamespace(
        losses_stage1=losses_stage1,
        losses_stage2=losses_stage2,
        teacher=teacher,
        trainer=trainer,
    )


# ---------------------------------------------------------------------------
# One real model + Stage2Loss training step
# ---------------------------------------------------------------------------


def test_real_model_stage2_step_has_finite_gradients(slbr_modules) -> None:
    from src.networks.slbr_next import SLBRNextV1

    torch.manual_seed(0)
    model = SLBRNextV1(convnext_pretrained=False)
    model.train()
    criterion = Stage2ProbabilityLoss(
        slbr_modules.losses_stage2.Stage2Loss(perceptual=None)
    )
    batch = synthetic_stage2_batch(1, 64, seed=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    outputs = model(batch["image"])
    result = criterion(outputs, batch)
    assert "loss" in result and "total" in result
    loss = result["loss"]
    assert loss.numel() == 1
    assert torch.isfinite(loss).all()
    for key in ("reconstruction", "fft", "mask"):
        assert key in result, key
        assert torch.isfinite(result[key]).all(), key

    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients, "backward produced no gradients"
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    total_abs = sum(float(gradient.abs().sum()) for gradient in gradients)
    assert total_abs > 0.0
    mask_abs = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if name.startswith("mask_decoder") and parameter.grad is not None
    )
    assert mask_abs > 0.0, "the mask loss must backpropagate to mask_decoder"
    optimizer.step()

    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


# ---------------------------------------------------------------------------
# Stage 1 loss with synthetic teacher targets/features
# ---------------------------------------------------------------------------


def _stage1_target(
    teacher_module,
    *,
    batch: int = 2,
    height: int = 32,
    width: int = 32,
    h8_channels: int = 8,
    h4_channels: int = 8,
):
    generator = torch.Generator().manual_seed(11)
    sizes = (
        (height, width),
        ((height + 1) // 2, (width + 1) // 2),
        ((height + 3) // 4, (width + 3) // 4),
    )

    def logits(index: int) -> torch.Tensor:
        h, w = sizes[index]
        return torch.randn(batch, 1, h, w, generator=generator)

    return teacher_module.TeacherTarget(
        mask_logits=(logits(0), logits(1), logits(2)),
        primary_logits=(logits(0), logits(1), logits(2)),
        refine_bg=torch.rand(batch, 3, height, width, generator=generator),
        coarse_bg=torch.rand(batch, 3, height, width, generator=generator),
        bottleneck_h8=torch.randn(
            batch, h8_channels, (height + 7) // 8, (width + 7) // 8,
            generator=generator,
        ),
        bg_decoder_h4=torch.randn(
            batch, h4_channels, (height + 3) // 4, (width + 3) // 4,
            generator=generator,
        ),
        height=height,
        width=width,
    )


def test_stage1_loss_step_has_finite_gradients(slbr_modules) -> None:
    stage1 = slbr_modules.losses_stage1
    batch_size = 2
    height = width = 32
    h8_channels = 8
    h4_channels = 8
    generator = torch.Generator().manual_seed(12)

    def randn(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.randn(batch_size, channels, h, w, generator=generator)

    def rand(channels: int, h: int, w: int) -> torch.Tensor:
        return torch.rand(batch_size, channels, h, w, generator=generator)

    mask_logits = tuple(
        randn(1, h, w)
        for h, w in (
            (height, width),
            ((height + 1) // 2, (width + 1) // 2),
            ((height + 3) // 4, (width + 3) // 4),
        )
    )
    refine = rand(3, height, width)
    coarse = rand(3, height, width)
    feature_h8 = randn(h8_channels, (height + 7) // 8, (width + 7) // 8)
    feature_h4 = randn(h4_channels, (height + 3) // 4, (width + 3) // 4)
    for tensor in (refine, coarse, *mask_logits, feature_h8, feature_h4):
        tensor.requires_grad_(True)

    outputs = stage1.StudentStage1Output(
        refine_bg=refine,
        coarse_bg=coarse,
        mask_logits=mask_logits,
        feature_h8=feature_h8,
        feature_h4=feature_h4,
    )
    target = _stage1_target(
        slbr_modules.teacher,
        batch=batch_size,
        height=height,
        width=width,
        h8_channels=h8_channels,
        h4_channels=h4_channels,
    )
    criterion = stage1.Stage1Loss(
        student_h8_channels=h8_channels,
        student_h4_channels=h4_channels,
        student_h_channels=h8_channels,
        teacher_h8_channels=h8_channels,
        teacher_h4_channels=h4_channels,
        teacher_h_channels=h8_channels,
    )

    result = criterion(outputs, target)
    assert torch.isfinite(result["total"]).all()
    assert float(result["mask"].detach()) > 0.0
    assert float(result["charbonnier"].detach()) > 0.0
    assert float(result["fft"].detach()) > 0.0
    result["total"].backward()

    tracked = [refine, coarse, *mask_logits, feature_h8, feature_h4]
    for tensor in tracked:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.abs().sum()) > 0.0
    adapter_grads = [
        parameter.grad
        for parameter in criterion.parameters()
        if parameter.grad is not None
    ]
    assert adapter_grads
    for gradient in adapter_grads:
        assert torch.isfinite(gradient).all()


# ---------------------------------------------------------------------------
# T5 one-stage schedule through the trainer
# ---------------------------------------------------------------------------


class _TinyScheduleModel(nn.Module):
    """Small hook-free student that still emits the Stage 1 contract."""

    def __init__(self, h8_channels: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.refine = nn.Conv2d(8, 3, 1)
        self.coarse = nn.Conv2d(8, 3, 1)
        self.mask_h = nn.Conv2d(8, 1, 1)
        self.mask_h2 = nn.Conv2d(8, 1, 1)
        self.mask_h4 = nn.Conv2d(8, 1, 1)
        self.h8 = nn.Conv2d(8, h8_channels, 1)
        self.h4 = nn.Conv2d(8, h8_channels, 1)

    def stage1(self, x: torch.Tensor, output_cls):
        features = self.stem(x)
        height, width = x.shape[-2:]
        pooled_h2 = F.avg_pool2d(features, 2)
        pooled_h4 = F.avg_pool2d(features, 4)
        pooled_h8 = F.adaptive_avg_pool2d(
            features, (max(1, height // 8), max(1, width // 8))
        )
        return output_cls(
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


class _PoisonBatch(dict):
    """Mapping that fails loudly if a forbidden key is read."""

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


class _TwoEpochScheduleLoader:
    """One batch per epoch; the second epoch poisons the teacher target."""

    def __init__(self, target) -> None:
        self.target = target
        self.epoch = 0

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        self.epoch += 1
        generator = torch.Generator().manual_seed(20 + self.epoch)
        image = torch.rand(1, 3, 32, 32, generator=generator)
        clean = torch.rand(1, 3, 32, 32, generator=generator)
        mask = (torch.rand(1, 1, 32, 32, generator=generator) > 0.7).float()
        batch = {"image": image, "clean": clean, "mask": mask}
        if self.epoch == 1:
            batch["teacher_target"] = self.target
            yield batch
        else:
            poisoned = _PoisonBatch(batch, poisoned=("teacher_target",))
            poisoned["teacher_target"] = object()
            yield poisoned


def _micro_config(trainer_module, output_dir: Path):
    return trainer_module.TrainerConfig(
        epochs=2,
        max_steps=None,
        batch_size_per_gpu=1,
        schedule="one_stage",
        teacher_decay_epochs=1,
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
        val_every_epochs=0,
        validate_ema=False,
        finite_checks=True,
        fail_on_overflow=True,
        output_dir=str(output_dir),
        save_best=False,
        save_every_epoch=False,
        save_on_exit=False,
    )


def test_one_stage_schedule_reaches_zero_teacher_weight_in_trainer(
    slbr_modules, tmp_path: Path
) -> None:
    stage1 = slbr_modules.losses_stage1
    stage2 = slbr_modules.losses_stage2
    teacher_module = slbr_modules.teacher
    trainer_module = slbr_modules.trainer

    target = _stage1_target(
        teacher_module, batch=1, height=32, width=32, h8_channels=4, h4_channels=4
    )
    loss = stage1.OneStageLoss(
        supervised=stage2.Stage2Loss(perceptual=None),
        teacher=None,
        teacher_decay_epochs=1,
        student_h8_channels=4,
        teacher_h8_channels=4,
    )
    loader = _TwoEpochScheduleLoader(target)
    trainer = trainer_module.Trainer(
        model=_TinyScheduleModel(),
        loss=loss,
        train_loader=loader,
        val_loader=None,
        config=_micro_config(trainer_module, tmp_path),
        device="cpu",
        forward_fn=lambda model, inputs: model.stage1(
            inputs, stage1.StudentStage1Output
        ),
    )

    history = trainer.train()

    assert trainer.epoch == 2
    assert trainer.global_step == 2
    assert trainer.overflow_steps == 0
    assert trainer.skipped_steps == 0
    assert len(history) == 2
    assert all(math.isfinite(entry["train_loss"]) for entry in history)
    assert loss.teacher_detached
    assert loss.teacher_weight == 0.0
    trainer.assert_finite_state(check_grads=False)

    records = [
        json.loads(line)
        for line in (tmp_path / "train_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    steps = [record for record in records if record["event"] == "step"]
    assert len(steps) == 2
    assert steps[0]["components"]["teacher_weight"] == pytest.approx(1.0)
    assert steps[1]["components"]["teacher_weight"] == 0.0
    assert steps[1]["components"]["teacher"] == 0.0
    assert math.isfinite(steps[1]["components"]["supervised"])


# ---------------------------------------------------------------------------
# Matrix specification, parsing and selection
# ---------------------------------------------------------------------------


def _matrix_row(**overrides):
    row = {
        "id": "row",
        "amp": "fp16",
        "channels_last": True,
        "compile": False,
        "batch_size_per_gpu": 6,
        "world_size": 2,
        "size": 256,
        "warmup_steps": 2,
        "measured_steps": 5,
        "samples_per_s": 100.0,
        "peak_vram_mb": [1000.0, 1001.0],
        "loss": 1.0,
        "loss_finite": True,
        "overflow_steps": 0,
        "skipped_steps": 0,
        "oom": False,
        "status": "ok",
        "error": None,
        "compile_s": None,
        "first_step_s": 0.5,
        "compile_fallback": False,
        "step_seconds": [0.1] * 5,
    }
    row.update(overrides)
    return row


def _rank_payload(rank: int, **overrides):
    payload = {
        "rank": rank,
        "world_size": 2,
        "status": "ok",
        "oom": False,
        "samples_per_s": 100.0 - rank,
        "measured_steps": 5,
        "warmup_steps": 2,
        "step_seconds": [0.1] * 5,
        "loss": 1.0,
        "loss_finite": True,
        "overflow_steps": 0,
        "skipped_steps": 0,
        "peak_vram_mb": 1000.0 + rank,
        "compile_s": None,
        "first_step_s": 0.5,
        "compile_fallback": False,
    }
    payload.update(overrides)
    return payload


def test_default_matrix_covers_required_configurations() -> None:
    specs = matrix.default_specs()
    assert len(specs) == 10
    assert len({spec["id"] for spec in specs}) == len(specs)
    assert all(spec["size"] == 256 for spec in specs)
    assert all(spec["world_size"] == 2 for spec in specs)
    combinations = {
        (
            spec["amp"],
            spec["channels_last"],
            spec["compile"],
            spec["batch_size_per_gpu"],
        )
        for spec in specs
    }
    required = {
        ("fp16", True, False, 6),
        ("fp16", True, False, 8),
        ("fp16", True, False, 10),
        ("fp16", True, True, 6),
        ("fp16", True, True, 8),
        ("fp16", True, True, 10),
        ("fp32", True, False, 6),
        ("fp16", False, False, 6),
    }
    assert required <= combinations
    assert ("fp32", False, False, 6) in combinations
    assert ("fp16", False, True, 6) in combinations


def test_matrix_selection_excludes_oom_overflow_and_fallback() -> None:
    rows = [
        _matrix_row(id="stable_slow", samples_per_s=100.0),
        _matrix_row(id="stable_fast", samples_per_s=120.0),
        _matrix_row(id="oom", samples_per_s=999.0, oom=True, status="oom"),
        _matrix_row(id="overflow", samples_per_s=500.0, overflow_steps=1),
        _matrix_row(id="skipped", samples_per_s=400.0, skipped_steps=1),
        _matrix_row(id="nonfinite", samples_per_s=300.0, loss=float("nan")),
        _matrix_row(id="bad_status", samples_per_s=200.0, status="error"),
        _matrix_row(
            id="compile_fallback",
            samples_per_s=1000.0,
            compile=True,
            compile_fallback=True,
        ),
    ]

    best = matrix.select_best(rows)

    assert best is not None
    assert best["id"] == "stable_fast"
    assert best["samples_per_s"] == 120.0
    assert matrix.row_is_stable(rows[0])
    assert not matrix.row_is_stable(rows[2])
    assert not matrix.row_is_stable(rows[3])
    assert not matrix.row_is_stable(rows[4])
    assert not matrix.row_is_stable(rows[5])
    assert not matrix.row_is_stable(rows[6])
    assert not matrix.row_is_stable(rows[7])
    assert matrix.select_best([]) is None


def test_aggregate_rank_results_handles_success_and_failures() -> None:
    spec = matrix.default_specs()[0]
    stable = matrix.aggregate_rank_results(
        spec, [_rank_payload(0), _rank_payload(1)]
    )
    assert stable["status"] == "ok"
    assert stable["oom"] is False
    assert stable["samples_per_s"] == pytest.approx(100.0)
    assert stable["peak_vram_mb"] == [1000.0, 1001.0]
    assert stable["overflow_steps"] == 0
    assert matrix.row_is_stable(stable)

    oom = matrix.aggregate_rank_results(
        spec,
        [_rank_payload(0, status="error", oom=True, error="CUDA out of memory")],
        returncode=42,
    )
    assert oom["status"] == "oom"
    assert oom["oom"] is True

    missing = matrix.aggregate_rank_results(spec, [_rank_payload(0)])
    assert missing["status"] == "error"
    assert "expected 2" in missing["error"]

    text_oom = matrix.aggregate_rank_results(
        spec, [], returncode=-9, output_text="CUDA out of memory"
    )
    assert text_oom["status"] == "oom"

    overflow = matrix.aggregate_rank_results(
        spec,
        [
            _rank_payload(0, overflow_steps=1),
            _rank_payload(1, overflow_steps=0),
        ],
    )
    assert overflow["status"] == "ok"
    assert overflow["overflow_steps"] == 1
    assert not matrix.row_is_stable(overflow)


def test_matrix_markdown_and_json_outputs_are_valid(tmp_path: Path) -> None:
    rows = [
        _matrix_row(id="stable_fast", samples_per_s=120.0),
        _matrix_row(id="stable_slow", samples_per_s=100.0),
        _matrix_row(id="oom", samples_per_s=999.0, oom=True, status="oom"),
    ]
    markdown = matrix.render_markdown(rows)
    assert markdown.startswith("| row |")
    assert "| samples/s |" in markdown
    assert "stable_fast" in markdown
    assert "OOM" in markdown

    paths = matrix.write_outputs(rows, tmp_path)
    payload = json.loads(paths["results_json"].read_text(encoding="utf-8"))
    assert payload["best"]["id"] == "stable_fast"
    assert payload["best_config"]["id"] == "stable_fast"
    assert len(payload["rows"]) == 3
    assert paths["matrix_md"].read_text(encoding="utf-8") == markdown + "\n"

    best = matrix.select_best(rows)
    line = matrix.format_best_line(best)
    assert line.startswith("BEST_CONFIG ")
    decoded = json.loads(line[len("BEST_CONFIG ") :])
    assert decoded["id"] == "stable_fast"
    assert decoded["amp"] == "fp16"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_fp16_stage2_step_is_finite(slbr_modules) -> None:
    from src.networks.slbr_next import SLBRNextV1

    torch.manual_seed(1)
    device = torch.device("cuda")
    model = SLBRNextV1(convnext_pretrained=False).to(device)
    model.train()
    criterion = Stage2ProbabilityLoss(
        slbr_modules.losses_stage2.Stage2Loss(perceptual=None)
    ).to(device)
    batch = {
        key: value.to(device)
        for key, value in synthetic_stage2_batch(1, 64, seed=4).items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = (
        torch.amp.GradScaler("cuda")
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler()
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = criterion(model(batch["image"]), batch)
    assert torch.isfinite(result["loss"]).all()
    scaler.scale(result["loss"]).backward()
    scaler.unscale_(optimizer)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    scaler.step(optimizer)
    scaler.update()
