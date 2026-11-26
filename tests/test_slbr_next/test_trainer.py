"""CPU-first tests for the SLBR-Next trainer (issue #12).

The tests use a tiny convolutional model and a synthetic ``TensorDataset`` so
they run on the hosted CPU runner in a few seconds. CUDA/AMP-specific checks
skip when no GPU is present. No external data, downloads or new dependencies
are used.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ``test_imports.py`` asserts that importing ``src.networks.slbr_next`` does
# not pull the legacy ``src.models`` package into ``sys.modules``.  Pytest
# imports every test module during collection, so a normal
# ``from src.models.slbr_next.trainer import ...`` here would load the legacy
# package before that assertion runs.  The trainer module has no package
# relative imports, so load it directly under a private name instead.
_TRAINER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "models"
    / "slbr_next"
    / "trainer.py"
)


def _load_trainer_module():
    module_name = "slbr_next_trainer_under_test"
    spec = importlib.util.spec_from_file_location(module_name, _TRAINER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise RuntimeError(f"could not load trainer module from {_TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_trainer = _load_trainer_module()
ModelEMA = _trainer.ModelEMA
Trainer = _trainer.Trainer
TrainerConfig = _trainer.TrainerConfig
WarmupCosineLR = _trainer.WarmupCosineLR
resolve_amp_dtype = _trainer.resolve_amp_dtype
seed_everything = _trainer.seed_everything


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """A 4D-conv model so channels_last is meaningful."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv2(self.act(self.conv1(x))))


class TinyLoss(nn.Module):
    """Mapping-returning loss to exercise component logging."""

    def forward(self, outputs: torch.Tensor, batch) -> dict[str, torch.Tensor]:
        target = batch[1] if isinstance(batch, (list, tuple)) else batch
        l1 = F.l1_loss(outputs, target)
        return {"loss": l1, "l1": l1.detach()}


class _InfiniteGradient(torch.autograd.Function):
    """Finite forward value with an Inf backward gradient."""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value * 0.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.full_like(grad_output, float("inf"))


class InfGradientLoss(nn.Module):
    """Returns a finite zero loss but poisons the gradient with Inf."""

    def forward(self, outputs: torch.Tensor, batch) -> torch.Tensor:
        del batch
        return _InfiniteGradient.apply(outputs).sum()


def _loader(
    length: int = 4,
    batch_size: int = 1,
    seed: int = 0,
    size: int = 8,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(length, 3, size, size, generator=generator)
    dataset = TensorDataset(images, images.clone())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _config(output_dir: Path, **overrides) -> TrainerConfig:
    values = dict(
        epochs=1,
        max_steps=None,
        batch_size_per_gpu=1,
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        warmup_steps=1,
        min_lr=1e-5,
        grad_clip=1.0,
        ema_decay=0.9,
        amp="auto",
        channels_last=False,
        cudnn_benchmark=False,
        compile=False,
        ddp=False,
        device="cpu",
        seed=123,
        log_every=1,
        val_every_epochs=0,
        validate_ema=True,
        early_stop_patience=0,
        output_dir=str(output_dir),
        save_best=False,
        save_every_epoch=False,
        save_on_exit=False,
    )
    values.update(overrides)
    return TrainerConfig(**values)


# ---------------------------------------------------------------------------
# Basic training, logging, validation
# ---------------------------------------------------------------------------


def test_two_step_cpu_training_logs_and_checkpoints(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        epochs=1,
        max_steps=2,
        val_every_epochs=1,
        save_on_exit=True,
    )
    trainer = Trainer(
        model=TinyModel(),
        loss=TinyLoss(),
        train_loader=_loader(length=4, batch_size=2),
        val_loader=_loader(length=2, batch_size=1, seed=1),
        config=config,
    )

    history = trainer.train()

    assert trainer.global_step == 2
    assert trainer.epoch == 1
    assert trainer.batch_in_epoch == 0
    assert trainer.last_loss is not None
    assert math.isfinite(trainer.last_loss)
    assert history and history[-1]["event"] == "epoch"
    assert "val_loss" in history[-1]
    assert math.isfinite(history[-1]["val_loss"])

    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "last.pt").is_file()
    records = [
        json.loads(line)
        for line in (tmp_path / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = [record["event"] for record in records]
    assert events.count("run") == 1
    assert events.count("step") == 2
    assert events.count("epoch") == 1
    assert any(record.get("val_loss") is not None for record in records
               if record["event"] == "epoch")


def test_trainer_uses_warmup_lr(tmp_path: Path) -> None:
    config = _config(tmp_path, lr=1e-3, warmup_steps=4, max_steps=1)
    trainer = Trainer(
        model=TinyModel(),
        loss=TinyLoss(),
        train_loader=_loader(length=2, batch_size=1),
        config=config,
    )

    trainer.train(max_steps=1)

    assert trainer.last_lr == pytest.approx(1e-3 / 4)
    assert trainer.global_step == 1


# ---------------------------------------------------------------------------
# Scheduler and EMA
# ---------------------------------------------------------------------------


def test_warmup_cosine_schedule_values_and_state() -> None:
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineLR(
        optimizer, warmup_steps=4, total_steps=10, min_lr=0.01
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.25)
    assert scheduler.lr_for_step(0)[0] == pytest.approx(0.25)
    assert scheduler.lr_for_step(3)[0] == pytest.approx(1.0)
    assert scheduler.lr_for_step(9)[0] == pytest.approx(0.01)

    for _ in range(10):
        scheduler.step()
    assert scheduler.last_step == 10
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    state = scheduler.state_dict()
    other_parameter = nn.Parameter(torch.zeros(1))
    other_optimizer = torch.optim.SGD([other_parameter], lr=1.0)
    other = WarmupCosineLR(
        other_optimizer, warmup_steps=4, total_steps=10, min_lr=0.01
    )
    other.load_state_dict(state)
    assert other.last_step == 10
    assert other_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_ema_update_matches_formula(tmp_path: Path) -> None:
    seed_everything(5)
    model = TinyModel()
    initial = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    config = _config(tmp_path, ema_decay=0.9, max_steps=1, lr=1e-2)
    trainer = Trainer(
        model=model,
        loss=TinyLoss(),
        train_loader=_loader(length=2, batch_size=1, seed=2),
        config=config,
    )

    trainer.train(max_steps=1)

    assert trainer.ema.num_updates == 1
    ema_state = trainer.ema.module.state_dict()
    current = trainer.raw_model.state_dict()
    for name, value in initial.items():
        expected = value * 0.9 + current[name] * 0.1
        assert torch.allclose(ema_state[name], expected, atol=1e-6), name


# ---------------------------------------------------------------------------
# Checkpoint/resume and atomic writes
# ---------------------------------------------------------------------------


def test_checkpoint_resume_continues_exactly(tmp_path: Path) -> None:
    seed_everything(7)
    base = TinyModel()

    interrupted = copy.deepcopy(base)
    first_config = _config(tmp_path / "first", epochs=1, max_steps=None)
    first = Trainer(
        model=interrupted,
        loss=TinyLoss(),
        train_loader=_loader(length=4, batch_size=1, seed=11),
        config=first_config,
    )
    first.train(max_steps=1)
    checkpoint = first.save_checkpoint(tmp_path / "partial.pt")
    assert first.global_step == 1
    assert first.batch_in_epoch == 1

    resumed_config = _config(tmp_path / "resumed", epochs=1, max_steps=None)
    resumed = Trainer(
        model=copy.deepcopy(base),
        loss=TinyLoss(),
        train_loader=_loader(length=4, batch_size=1, seed=11),
        config=resumed_config,
    )
    resumed.resume(checkpoint)
    assert resumed.global_step == 1
    assert resumed.batch_in_epoch == 1
    resumed.train(max_steps=3)

    reference_config = _config(tmp_path / "reference", epochs=1, max_steps=None)
    reference = Trainer(
        model=copy.deepcopy(base),
        loss=TinyLoss(),
        train_loader=_loader(length=4, batch_size=1, seed=11),
        config=reference_config,
    )
    reference.train(max_steps=3)

    assert resumed.global_step == 3
    assert reference.global_step == 3
    assert resumed.epoch == reference.epoch == 0
    assert resumed.batch_in_epoch == reference.batch_in_epoch == 3
    assert resumed.scheduler.last_step == reference.scheduler.last_step == 3
    assert resumed.last_loss == pytest.approx(reference.last_loss, rel=1e-7)

    reference_state = reference.raw_model.state_dict()
    for name, value in resumed.raw_model.state_dict().items():
        assert torch.equal(value, reference_state[name]), name

    reference_ema = reference.ema.module.state_dict()
    for name, value in resumed.ema.module.state_dict().items():
        assert torch.equal(value, reference_ema[name]), name

    resumed_optimizer = resumed.optimizer.state_dict()
    reference_optimizer = reference.optimizer.state_dict()
    assert resumed_optimizer["param_groups"] == reference_optimizer["param_groups"]
    for key, state in reference_optimizer["state"].items():
        other = resumed_optimizer["state"][key]
        if torch.is_tensor(state.get("step")):
            assert torch.equal(state["step"], other["step"])
        for name, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, other[name]), (key, name)


def test_checkpoint_is_atomic_and_complete(tmp_path: Path) -> None:
    config = _config(tmp_path, max_steps=1)
    trainer = Trainer(
        model=TinyModel(),
        loss=TinyLoss(),
        train_loader=_loader(length=2, batch_size=1),
        config=config,
    )
    trainer.train(max_steps=1)

    path = trainer.save_checkpoint(tmp_path / "nested" / "last.pt")

    assert path.is_file()
    assert not Path(str(path) + ".tmp").exists()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "epoch",
        "batch_in_epoch",
        "global_step",
        "config",
        "rng",
    ):
        assert key in checkpoint, key
    assert checkpoint["global_step"] == 1
    assert checkpoint["config"]["batch_size_per_gpu"] == 1


# ---------------------------------------------------------------------------
# Clipping, compile fallback, channels_last, AMP
# ---------------------------------------------------------------------------


def test_gradient_clipping_caps_norm(tmp_path: Path) -> None:
    config = _config(tmp_path, grad_clip=0.1)
    trainer = Trainer(
        model=TinyModel(),
        loss=TinyLoss(),
        train_loader=_loader(length=1),
        config=config,
    )
    for parameter in trainer.model.parameters():
        parameter.grad = torch.full_like(parameter, 3.0)

    before = math.sqrt(
        sum(
            float(parameter.grad.pow(2).sum())
            for parameter in trainer.model.parameters()
        )
    )
    returned = trainer.clip_gradients()
    after = math.sqrt(
        sum(
            float(parameter.grad.pow(2).sum())
            for parameter in trainer.model.parameters()
        )
    )

    assert returned == pytest.approx(before, rel=1e-5)
    assert before > 0.1
    assert after <= 0.1 + 1e-6


def test_compile_falls_back_to_eager(monkeypatch, tmp_path: Path) -> None:
    def unavailable(*args, **kwargs):
        raise RuntimeError("compiler unavailable in this test")

    monkeypatch.setattr(torch, "compile", unavailable)
    config = _config(tmp_path, compile=True)
    trainer = Trainer(
        model=TinyModel(),
        loss=TinyLoss(),
        train_loader=_loader(length=2, batch_size=1),
        config=config,
    )

    trainer.train(max_steps=1)

    assert trainer.compile_fallback is True
    assert trainer.last_loss is not None
    assert math.isfinite(trainer.last_loss)
    # Smoke timing plumbing: the first step and the compile attempt are timed
    # even when compilation falls back before the first compiled forward.
    assert trainer.first_step_seconds is not None
    assert trainer.compile_seconds is not None


def test_overflow_skips_optimizer_and_reports_finite_metrics(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        max_steps=1,
        finite_checks=True,
        fail_on_overflow=False,
        lr=1e-2,
    )
    model = TinyModel()
    initial = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    trainer = Trainer(
        model=model,
        loss=InfGradientLoss(),
        train_loader=_loader(length=1),
        config=config,
    )

    trainer.train(max_steps=1)

    assert trainer.overflow_steps == 1
    assert trainer.skipped_steps == 1
    assert trainer.last_overflow_step == 1
    assert trainer.last_grad_norm is None
    assert trainer.last_loss == 0.0
    assert not trainer.optimizer.state
    for name, value in trainer.raw_model.state_dict().items():
        assert torch.equal(value, initial[name]), name
    trainer.assert_finite_state(check_grads=False)

    records = [
        json.loads(line)
        for line in (tmp_path / "train_log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    step = [record for record in records if record["event"] == "step"][-1]
    assert step["grad_norm"] is None
    assert step["grad_norm_finite"] is False
    assert step["loss_finite"] is True
    assert step["overflow"] is True
    assert step["step_skipped"] is True
    assert step["overflow_steps"] == 1
    assert step["skipped_steps"] == 1
    assert step["nonfinite_grad_params"]
    assert math.isfinite(step["loss"])


def test_fail_on_overflow_blocks_smoke_ok(tmp_path: Path) -> None:
    config = _config(
        tmp_path / "strict",
        max_steps=1,
        finite_checks=True,
        fail_on_overflow=True,
    )
    trainer = Trainer(
        model=TinyModel(),
        loss=InfGradientLoss(),
        train_loader=_loader(length=1),
        config=config,
    )

    with pytest.raises(FloatingPointError, match="optimizer update skipped"):
        trainer.train(max_steps=1)

    assert trainer.overflow_steps == 1
    assert trainer.skipped_steps == 1
    assert not trainer.optimizer.state
    trainer.assert_finite_state(check_grads=False)


def test_smoke_loss_bce_is_autocast_safe(monkeypatch) -> None:
    """The smoke BCE must run in FP32 with autocast disabled (old code raised).

    ``SLBRNextV1.forward`` returns probabilities, so the fix is an explicit
    FP32/no-autocast BCE rather than switching to logits. CPU BF16 autocast
    reproduces the same PyTorch safety guard on the hosted CI runner. The
    wrapper also records the autocast state at the BCE call so the regression
    is explicit even if a future torch version changes the guard.
    """
    calls: dict[str, object] = {}
    original_bce = F.binary_cross_entropy

    def checked_bce(prediction, target, **kwargs):
        try:
            calls["autocast_cpu"] = torch.is_autocast_enabled("cpu")
        except TypeError:  # pragma: no cover - older torch signature
            calls["autocast_cpu"] = torch.is_autocast_cpu_enabled()
        calls["dtype"] = prediction.dtype
        return original_bce(prediction, target, **kwargs)

    monkeypatch.setattr(F, "binary_cross_entropy", checked_bce)

    torch.manual_seed(0)
    refine = torch.rand(2, 3, 16, 16, requires_grad=True)
    coarse = torch.rand(2, 3, 16, 16, requires_grad=True)
    masks = [
        torch.rand(2, 1, 16, 16, requires_grad=True),
        torch.rand(2, 1, 8, 8, requires_grad=True),
        torch.rand(2, 1, 4, 4, requires_grad=True),
    ]
    mask_gt = torch.zeros(2, 1, 16, 16)
    mask_gt[:, :, 4:10, 4:10] = 1.0
    batch = {"clean": torch.rand(2, 3, 16, 16), "mask": mask_gt}

    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = _trainer._smoke_loss(([refine, coarse], masks, [None]), batch)

    assert calls["autocast_cpu"] is False
    assert calls["dtype"] == torch.float32
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert refine.grad is not None
    assert torch.isfinite(refine.grad).all()


def test_channels_last_smoke(tmp_path: Path) -> None:
    config = _config(tmp_path, channels_last=True)
    try:
        trainer = Trainer(
            model=TinyModel(),
            loss=TinyLoss(),
            train_loader=_loader(length=2, batch_size=1),
            config=config,
        )
        trainer.train(max_steps=1)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "channel" in message or "memory format" in message:
            pytest.skip(f"channels_last is unsupported here: {exc}")
        raise

    weight = next(trainer.raw_model.parameters())
    assert weight.dim() == 4
    assert weight.is_contiguous(memory_format=torch.channels_last)


def test_cpu_amp_falls_back_to_fp32(tmp_path: Path) -> None:
    for index, amp in enumerate(("auto", "fp16", "bf16", "fp32")):
        config = _config(tmp_path / str(index), amp=amp)
        trainer = Trainer(
            model=TinyModel(),
            loss=TinyLoss(),
            train_loader=_loader(length=1),
            config=config,
        )
        assert trainer.amp_dtype == torch.float32
        assert trainer.scaler.is_enabled() is False

    assert resolve_amp_dtype("fp16", torch.device("cpu")) == torch.float32
    assert resolve_amp_dtype("bf16", torch.device("cpu")) == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_amp_selection_and_fp16_step(tmp_path: Path) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    assert resolve_amp_dtype("fp16", device) == torch.float16
    capability = torch.cuda.get_device_capability(device)
    if capability[0] >= 8:
        assert resolve_amp_dtype("bf16", device) == torch.bfloat16
        assert resolve_amp_dtype("auto", device) == torch.bfloat16
    else:
        with pytest.raises(ValueError):
            resolve_amp_dtype("bf16", device)
        assert resolve_amp_dtype("auto", device) == torch.float16

    config = _config(tmp_path, amp="fp16", device="cuda")
    trainer = Trainer(
        model=TinyModel(),
        loss=TinyLoss(),
        train_loader=_loader(length=2, batch_size=1),
        config=config,
    )
    assert trainer.amp_dtype == torch.float16
    assert trainer.scaler.is_enabled() is True
    trainer.train(max_steps=1)
    assert trainer.last_loss is not None
    assert math.isfinite(trainer.last_loss)


# ---------------------------------------------------------------------------
# Config and early stop
# ---------------------------------------------------------------------------


def test_config_json_roundtrip_and_defaults() -> None:
    config = TrainerConfig(
        betas=(0.8, 0.95), max_steps=7, output_dir="runs/example"
    )
    payload = json.loads(config.to_json())
    assert payload["betas"] == [0.8, 0.95]
    restored = TrainerConfig.from_dict(payload)
    assert restored == config
    assert restored.betas == (0.8, 0.95)
    assert json.dumps(config.to_dict())

    with pytest.raises(ValueError):
        TrainerConfig.from_dict({"not_a_field": 1})

    defaults = TrainerConfig()
    assert defaults.batch_size_per_gpu == 6
    assert defaults.lr == pytest.approx(3e-4)
    assert defaults.weight_decay == pytest.approx(1e-4)
    assert defaults.betas == (0.9, 0.99)
    assert defaults.warmup_steps == 500
    assert defaults.grad_clip == pytest.approx(1.0)
    assert defaults.ema_decay == pytest.approx(0.999)
    assert defaults.compile is True


def test_early_stop_hook(tmp_path: Path) -> None:
    seen: list[dict] = []

    def hook(entry: dict) -> bool:
        seen.append(entry)
        return True

    config = _config(tmp_path, epochs=5, max_steps=None)
    trainer = Trainer(
        model=TinyModel(),
        loss=TinyLoss(),
        train_loader=_loader(length=2, batch_size=1),
        config=config,
        early_stop_hook=hook,
    )

    trainer.train()

    assert trainer.stopped_early is True
    assert len(seen) == 1
    assert trainer.epoch == 1
    assert trainer.global_step == 2


def test_model_ema_state_roundtrip() -> None:
    model = TinyModel()
    ema = ModelEMA(model, decay=0.5)
    state = ema.state_dict()
    other = ModelEMA(copy.deepcopy(model), decay=0.9)
    other.load_state_dict(state)
    assert other.decay == pytest.approx(0.5)
    assert other.num_updates == 0
    for name, value in ema.module.state_dict().items():
        assert torch.equal(value, other.module.state_dict()[name]), name


def test_cli_source_parses_and_documents_smoke() -> None:
    # Importing the CLI would import src.models (and defeat test_imports.py's
    # isolation assertion), so validate its source syntactically instead.
    cli_path = _TRAINER_PATH.parents[3] / "train_slbr_next.py"
    source = cli_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(cli_path))
    assert "smoke_train" in source
    assert "--smoke" in source
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {"build_parser", "main"} <= functions
