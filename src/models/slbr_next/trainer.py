"""Dependency-light training loop for SLBR-Next (issue #12, spec §5.3-§5.5).

The trainer is deliberately generic: the model and the loss are injected, so
Stage 1 distillation (T2) and Stage 2 fine-tuning (T3) can plug in their own
modules without circular imports. A loss is any callable::

    loss(outputs, batch) -> Tensor | Mapping[str, Tensor]

When a mapping is returned it must contain ``"loss"``; every other scalar
tensor is treated as a metric component and logged. ``batch`` is already on
the training device: tensors are moved recursively and 4D floating point
tensors are converted to ``channels_last`` when ``config.channels_last`` is
set. The model input is taken from ``batch[config.input_key]`` for mapping
batches, the first element for tuple/list batches, or the tensor itself.

Features required by the T4 profile (§5.3, §5.4):

* FP16 autocast + ``GradScaler`` on CUDA/Turing, BF16 only on ``sm_80+``,
  FP32 fallback on CPU (``config.amp = "auto" | "fp16" | "bf16" | "fp32"``).
* One fused ``AdamW`` (``fused=True`` only on CUDA), ``lr=3e-4``,
  ``weight_decay=1e-4``, ``betas=(0.9, 0.99)``.
* ~500 step linear warmup followed by cosine decay to ``1e-6``.
* Gradient clip 1.0 and an EMA with decay 0.999 (evaluate/export the EMA).
* ``channels_last``, ``cudnn.benchmark``, ``torch.compile`` with a lazy
  catch-and-fallback to eager, and DDP from ``torchrun`` env vars with a
  configurable ``static_graph``.
* Atomic checkpoint/resume of model, optimizer, scheduler, scaler, EMA,
  epoch, in-epoch batch position, step, config, history and RNG state, so a
  resumed deterministic CPU run continues exactly where it stopped.
* JSON config and JSONL metrics (step/epoch loss, LR, grad norm, samples/s,
  peak VRAM). Grad-norm/overflow detection runs after ``GradScaler.unscale_``;
  a non-finite step is skipped, logged as ``grad_norm: null`` /
  ``step_skipped: true`` and counted instead of being reported as an Inf norm.
* Optional loss schedule hooks: ``loss.set_epoch(epoch)`` before each epoch /
  validation and ``loss.set_progress(completed_steps, total_steps)`` before
  each training step. ``OneStageLoss`` uses the epoch for its teacher decay
  and ``Stage2Loss`` uses the progress for its late perceptual schedule.
  ``Trainer(..., forward_fn=...)`` lets a loss that needs feature taps use
  ``run_student_stage1`` while forwarding through the DDP/compile wrapper.

Command lines
-------------
Two-T4 (Kaggle) throughput/DDP smoke, synthetic data, no dataset required::

    torchrun --nproc_per_node=2 train_slbr_next.py --smoke --steps 10 \
        --batch-size 6 --size 256 --compile \
        --output-dir /kaggle/working/slbr_next_smoke

First exploratory run on T4 (spec §5.4 recommends no compile for the first
session)::

    torchrun --nproc_per_node=2 train_slbr_next.py --smoke --steps 10 \
        --batch-size 6 --size 256 --no-compile \
        --output-dir /kaggle/working/slbr_next_smoke_eager

Real training (data/loss factories are supplied by T2/T3)::

    torchrun --nproc_per_node=2 train_slbr_next.py \
        --data-factory src.data.slbr_next:build_loaders \
        --loss-factory src.losses.slbr_next:build_loss \
        --epochs 30 --output-dir /kaggle/working/slbr_next

The integration contract for real runs is documented in
``train_slbr_next.py``. Batch size starts at 6 images/GPU; 8 and 10 are
experiments, not assumptions, for a 15.6GB T4.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:  # numpy is a test/runtime dependency, but keep the trainer importable.
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is present in CI
    _np = None

__all__ = [
    "TrainerConfig",
    "Trainer",
    "ModelEMA",
    "WarmupCosineLR",
    "init_distributed",
    "resolve_amp_dtype",
    "seed_everything",
    "smoke_train",
]

LOGGER = logging.getLogger(__name__)

_AMP_CHOICES = ("auto", "fp16", "bf16", "fp32")
_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")
_SCHEDULE_CHOICES = ("two_stage", "one_stage")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """JSON-serializable training configuration.

    Every field is a plain JSON value so the config can be written to
    ``config.json`` and reloaded with :meth:`from_json`. Defaults follow spec
    §5.3/§5.4 and target 2xT4: batch size 6 images/GPU, fused AdamW at
    ``3e-4``, 500 warmup steps, cosine to ``1e-6``, grad clip 1.0 and EMA
    0.999.
    """

    epochs: int = 30
    #: Hard cap on optimizer steps. ``None`` means ``epochs * len(loader)``.
    max_steps: int | None = None
    #: Images per GPU. The trainer does not slice batches; data loaders use it.
    batch_size_per_gpu: int = 6

    # Training schedule (spec §5.5)
    #: ``"two_stage"`` keeps the existing Stage 1 -> Stage 2 recipe;
    #: ``"one_stage"`` selects the supervised + decaying-teacher auxiliary.
    #: The loss factory reads this value; the trainer only logs/validates it.
    schedule: str = "two_stage"
    #: One-stage teacher auxiliary linear decay window in epochs (spec §5.5).
    teacher_decay_epochs: int = 7

    # Optimization (§5.3)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    warmup_steps: int = 500
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    # Precision and layout (§5.3, §5.4)
    amp: str = "auto"
    channels_last: bool = True
    cudnn_benchmark: bool = True

    # Compiler and DDP (§5.4)
    compile: bool = True
    compile_mode: str = "default"
    ddp: bool = True
    ddp_static_graph: bool = False
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True

    # Runtime and logging
    device: str = "auto"
    seed: int = 1
    log_every: int = 50
    val_every_epochs: int = 1
    validate_ema: bool = True
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0

    # Numerical-safety checks (enabled by smoke_train, off by default so real
    # training does not pay a per-step full-model finiteness scan).
    #: Assert finite parameters after every step; with ``fail_on_overflow``,
    #: a detected non-finite unscaled gradient fails the run after logging it.
    finite_checks: bool = False
    fail_on_overflow: bool = False

    # Checkpointing
    output_dir: str = "checkpoints/slbr_next"
    checkpoint_name: str = "last.pt"
    best_checkpoint_name: str = "best.pt"
    save_best: bool = True
    save_every_epoch: bool = True
    save_on_exit: bool = True
    #: Key used to extract the model input from mapping batches.
    input_key: str = "image"

    def __post_init__(self) -> None:
        self.epochs = int(self.epochs)
        self.max_steps = None if self.max_steps is None else int(self.max_steps)
        self.batch_size_per_gpu = int(self.batch_size_per_gpu)
        self.schedule = str(self.schedule).lower()
        self.teacher_decay_epochs = int(self.teacher_decay_epochs)
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        try:
            beta1, beta2 = self.betas
        except (TypeError, ValueError) as exc:
            raise ValueError("betas must contain exactly two values") from exc
        self.betas = (float(beta1), float(beta2))
        self.warmup_steps = int(self.warmup_steps)
        self.min_lr = float(self.min_lr)
        self.grad_clip = float(self.grad_clip)
        self.ema_decay = float(self.ema_decay)
        self.amp = str(self.amp).lower()
        self.compile_mode = str(self.compile_mode)
        self.device = str(self.device)
        self.seed = int(self.seed)
        self.log_every = int(self.log_every)
        self.val_every_epochs = int(self.val_every_epochs)
        self.early_stop_patience = int(self.early_stop_patience)
        self.early_stop_min_delta = float(self.early_stop_min_delta)
        self.output_dir = str(self.output_dir)
        self.checkpoint_name = str(self.checkpoint_name)
        self.best_checkpoint_name = str(self.best_checkpoint_name)
        self.input_key = str(self.input_key)

        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 or None, got {self.max_steps}")
        if self.batch_size_per_gpu < 1:
            raise ValueError("batch_size_per_gpu must be >= 1")
        if self.schedule not in _SCHEDULE_CHOICES:
            raise ValueError(
                f"schedule must be one of {_SCHEDULE_CHOICES}, "
                f"got {self.schedule!r}"
            )
        if self.teacher_decay_epochs < 1:
            raise ValueError(
                "teacher_decay_epochs must be >= 1, got "
                f"{self.teacher_decay_epochs}"
            )
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not (0.0 <= self.betas[0] < 1.0 and 0.0 <= self.betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {self.betas}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.min_lr < 0 or self.min_lr > self.lr:
            raise ValueError("min_lr must be in [0, lr]")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be >= 0 (0 disables clipping)")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if self.amp not in _AMP_CHOICES:
            raise ValueError(f"amp must be one of {_AMP_CHOICES}, got {self.amp!r}")
        if self.compile_mode not in _COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {_COMPILE_MODES}, "
                f"got {self.compile_mode!r}"
            )
        if self.ddp_static_graph and self.ddp_find_unused_parameters:
            raise ValueError(
                "ddp_static_graph=True requires ddp_find_unused_parameters=False"
            )
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.val_every_epochs < 0:
            raise ValueError("val_every_epochs must be >= 0")
        if self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must be >= 0")
        if not self.input_key:
            raise ValueError("input_key must not be empty")
        if not self.checkpoint_name or not self.best_checkpoint_name:
            raise ValueError("checkpoint names must not be empty")

    @property
    def one_stage(self) -> bool:
        """Whether the selected schedule is the one-stage supervised variant."""
        return self.schedule == "one_stage"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (``betas`` becomes a list)."""
        data = asdict(self)
        data["betas"] = [float(self.betas[0]), float(self.betas[1])]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainerConfig:
        """Build a config from a mapping, rejecting unknown fields."""
        if not isinstance(data, Mapping):
            raise TypeError(f"config must be a mapping, got {type(data).__name__}")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError("Unknown TrainerConfig field(s): " + ", ".join(unknown))
        return cls(**dict(data))

    def to_json(self, indent: int = 2) -> str:
        """Return an indented JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> TrainerConfig:
        """Parse a config from JSON text."""
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int, rank: int = 0) -> int:
    """Seed Python, NumPy, torch and CUDA, returning the effective seed."""
    effective = int(seed) + int(rank)
    random.seed(effective)
    if _np is not None:
        _np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return effective


def init_distributed(enabled: bool = True) -> tuple[int, int, int]:
    """Initialise the default process group from torchrun env vars.

    Returns ``(rank, world_size, local_rank)``. With ``WORLD_SIZE <= 1`` this
    is a no-op. Calling it more than once is safe: an existing group is
    reused. If torchrun started more than one process but ``enabled`` is
    false, this raises instead of silently duplicating the run.
    """
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
        return dist.get_rank(), dist.get_world_size(), local_rank

    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    if world_size <= 1:
        return 0, 1, local_rank

    if not enabled:
        raise RuntimeError(
            "torchrun launched WORLD_SIZE=%d but config.ddp is False; "
            "enable DDP or launch a single process" % world_size
        )
    if not dist.is_available():
        raise RuntimeError(
            "torchrun launched multiple processes but torch.distributed "
            "is unavailable in this build"
        )
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), local_rank


def resolve_amp_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Resolve ``config.amp`` for a device (CPU always falls back to FP32).

    CUDA Turing (``sm_75``) uses FP16 for ``"auto"``. ``"bf16"`` is rejected
    below ``sm_80`` because Turing has no native BF16 tensor cores.
    """
    requested = str(requested).lower()
    if requested not in _AMP_CHOICES:
        raise ValueError(f"amp must be one of {_AMP_CHOICES}, got {requested!r}")
    if device.type != "cuda":
        return torch.float32

    capability = torch.cuda.get_device_capability(device)
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        if capability[0] < 8:
            raise ValueError(
                "BF16 autocast requires sm_80+ (Ampere or newer); "
                f"this device is sm_{capability[0]}{capability[1]}. Use amp='fp16'."
            )
        return torch.bfloat16
    return torch.bfloat16 if capability[0] >= 8 else torch.float16


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------


class WarmupCosineLR:
    """Linear warmup followed by cosine decay to ``min_lr``.

    ``last_step`` is the number of optimizer steps already completed. The
    constructor applies the LR for step 0, and :meth:`step` advances to the
    next step. This makes the schedule a plain object that checkpoints
    exactly (no lambda pickling and no scheduler/optimizer ordering warning).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if min_lr < 0:
            raise ValueError("min_lr must be >= 0")
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = 0
        self._apply(self.last_step)

    def _factor_for(self, base_lr: float, step: int) -> float:
        step = max(int(step), 0)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return 1.0
        progress = (step - self.warmup_steps + 1) / float(
            self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        if base_lr <= 0:
            return 0.0
        min_ratio = min(self.min_lr / base_lr, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    def _apply(self, step: int) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * self._factor_for(base_lr, step)

    def lr_for_step(self, step: int) -> list[float]:
        """Return the LR each param group would use at a 0-based step."""
        return [
            base_lr * self._factor_for(base_lr, step) for base_lr in self.base_lrs
        ]

    def factor(self, step: int) -> float:
        """Return the first param group's factor at a 0-based step."""
        if not self.base_lrs:
            return 0.0
        return self._factor_for(self.base_lrs[0], step)

    def step(self) -> list[float]:
        """Advance one optimizer step and apply the next LR."""
        self.last_step += 1
        self._apply(self.last_step)
        return self.get_last_lr()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "base_lrs": list(self.base_lrs),
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self.min_lr = float(state["min_lr"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        if len(self.base_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "scheduler checkpoint has %d param groups, optimizer has %d"
                % (len(self.base_lrs), len(self.optimizer.param_groups))
            )
        self.last_step = int(state["last_step"])
        # The optimizer state already contains the LR for the next step, but
        # applying it again makes a fresh optimizer/scheduler pair consistent.
        self._apply(self.last_step)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class ModelEMA:
    """Exponential moving average of a model's state dict.

    Floating point tensors are averaged; non-floating buffers are copied from
    the live model so normalization statistics stay in sync. The EMA module
    lives on the same device as the model and is never optimized.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(model)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the shadow weights from ``model``."""
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(
                    model_value, alpha=1.0 - self.decay
                )
            else:
                ema_value.copy_(model_value)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "module": self.module.state_dict(),
            "decay": self.decay,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state["module"])
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module):
        """Temporarily load EMA weights into ``model`` for evaluation."""
        backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.module.state_dict())
        try:
            yield model
        finally:
            model.load_state_dict(backup)


# ---------------------------------------------------------------------------
# Compile wrapper
# ---------------------------------------------------------------------------


class _NoOpGradScaler:
    """Stand-in for ``GradScaler`` when AMP is disabled (BF16/FP32/CPU)."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer):
        return optimizer.step()

    def update(self) -> None:
        return None

    def get_scale(self) -> float:
        return 1.0

    def is_enabled(self) -> bool:
        return False

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        del state


class _CompileWithFallback(nn.Module):
    """Lazily compile a module and permanently fall back to eager on error."""

    def __init__(self, module: nn.Module, mode: str = "default") -> None:
        super().__init__()
        self.module = module
        self.mode = mode
        self._compiled: nn.Module | None = None
        self.fallback = False
        #: Wall time of the first compiled forward. Inductor compiles lazily
        #: on that call, so this is where a multi-minute compile shows up.
        self.compile_seconds: float | None = None

    def forward(self, *args: Any, **kwargs: Any):
        if self.fallback:
            return self.module(*args, **kwargs)
        if self._compiled is None:
            created = time.perf_counter()
            try:
                self._compiled = torch.compile(self.module, mode=self.mode)
            except Exception as exc:  # pragma: no cover - depends on torch build
                self.fallback = True
                self.compile_seconds = time.perf_counter() - created
                LOGGER.warning(
                    "torch.compile failed at first use; falling back to eager: %s",
                    exc,
                )
                return self.module(*args, **kwargs)
        started = time.perf_counter()
        try:
            result = self._compiled(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on torch build
            if self.compile_seconds is None:
                self.compile_seconds = time.perf_counter() - started
            self.fallback = True
            LOGGER.warning(
                "compiled forward failed; falling back to eager and retrying: %s",
                exc,
            )
            return self.module(*args, **kwargs)
        if self.compile_seconds is None:
            self.compile_seconds = time.perf_counter() - started
        return result


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Train an injected model with an injected loss.

    Args:
        model: The network. It is moved to the selected device, optionally
            compiled, and wrapped in DDP when torchrun provides >1 process.
        loss: Callable ``loss(outputs, batch) -> Tensor | Mapping``. If it is
            an ``nn.Module`` its training/eval mode is toggled with the model.
            Its parameters are *not* added to the optimizer; freeze auxiliary
            losses (e.g. a VGG perceptual net) before passing them in.
        train_loader: Iterable of batches. Mapping batches must contain
            ``config.input_key``.
        val_loader: Optional validation loader; its loss is reduced across
            ranks. Validation uses EMA weights by default.
        config: :class:`TrainerConfig`; defaults are the T4 profile.
        optimizer: Optional pre-built optimizer. When omitted a fused AdamW is
            built from the config.
        scheduler: Optional scheduler with ``step/state_dict/load_state_dict``.
            It is stepped once per optimizer step, after the update. When
            omitted :class:`WarmupCosineLR` is used.
        device: Optional device override (``"cuda:1"``, ``"cpu"``, ...).
        forward_fn: Optional ``fn(model, inputs) -> outputs`` used instead of
            ``model(inputs)`` in both training and validation. The one-stage
            loss uses this to obtain ``run_student_stage1`` outputs (mask
            logits plus the H/8 feature tap) while still forwarding through
            the DDP/compile wrapper. When omitted, a callable
            ``loss.forward_fn`` is used automatically;
            ``OneStageLoss.forward_fn`` is the ready made value.
        early_stop_hook: Optional ``hook(history_entry) -> bool``. Returning
            ``True`` stops training after the epoch is logged and saved.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[..., Any],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        config: TrainerConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: str | torch.device | None = None,
        forward_fn: Callable[[nn.Module, torch.Tensor], Any] | None = None,
        early_stop_hook: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        if not callable(loss):
            raise TypeError("loss must be callable")
        if forward_fn is None:
            candidate = getattr(loss, "forward_fn", None)
            if callable(candidate):
                forward_fn = candidate
        if forward_fn is not None and not callable(forward_fn):
            raise TypeError("forward_fn must be callable or None")
        if config is None:
            config = TrainerConfig()
        elif not isinstance(config, TrainerConfig):
            raise TypeError("config must be a TrainerConfig")
        self.config = config
        self.loss = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.forward_fn = forward_fn
        self.early_stop_hook = early_stop_hook

        self.rank, self.world_size, self.local_rank = init_distributed(
            enabled=config.ddp
        )
        self.device = self._select_device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"device {self.device} requested but CUDA is unavailable"
            )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)

        seed_everything(config.seed, self.rank)

        self.raw_model = model
        self.raw_model.to(self.device)
        if config.channels_last:
            self.raw_model.to(memory_format=torch.channels_last)

        self.ema = ModelEMA(self.raw_model, decay=config.ema_decay)

        self._compile_wrapper: _CompileWithFallback | None = None
        if config.compile:
            if hasattr(torch, "compile"):
                self._compile_wrapper = _CompileWithFallback(
                    self.raw_model, mode=config.compile_mode
                )
                self.model: nn.Module = self._compile_wrapper
            else:  # pragma: no cover - very old torch
                LOGGER.warning("torch.compile is unavailable; using eager mode")
                self.model = self.raw_model
        else:
            self.model = self.raw_model

        if self.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel

            ddp_kwargs: dict[str, Any] = {
                "static_graph": config.ddp_static_graph,
                "find_unused_parameters": config.ddp_find_unused_parameters,
                "gradient_as_bucket_view": config.ddp_gradient_as_bucket_view,
            }
            if self.device.type == "cuda":
                device_index = (
                    self.device.index
                    if self.device.index is not None
                    else self.local_rank
                )
                ddp_kwargs["device_ids"] = [device_index]
                ddp_kwargs["output_device"] = device_index
            self.model = DistributedDataParallel(self.model, **ddp_kwargs)

        self.amp_dtype = resolve_amp_dtype(config.amp, self.device)
        if self.amp_dtype != torch.float16 and config.amp == "fp16":
            LOGGER.warning(
                "FP16 autocast requested on %s; falling back to %s",
                self.device,
                self.amp_dtype,
            )
        self._autocast_enabled = self.amp_dtype != torch.float32
        scaler_enabled = self.amp_dtype == torch.float16
        self.scaler = self._make_grad_scaler(scaler_enabled)

        self.total_steps = self._compute_total_steps()

        if optimizer is None:
            optimizer = self._build_optimizer()
        self.optimizer = optimizer
        if scheduler is None:
            scheduler = WarmupCosineLR(
                self.optimizer,
                warmup_steps=config.warmup_steps,
                total_steps=self.total_steps,
                min_lr=config.min_lr,
            )
        self.scheduler = scheduler

        self.epoch = 0
        self.batch_in_epoch = 0
        self.global_step = 0
        self.best_metric: float | None = None
        self.bad_epochs = 0
        self.stopped_early = False
        self.overflow_steps = 0
        self.skipped_steps = 0
        self.last_overflow_step: int | None = None
        self.last_nonfinite_grad_params: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.last_loss: float | None = None
        self.last_grad_norm: float | None = None
        self.last_lr: float | None = None
        self.first_step_seconds: float | None = None
        self._epoch_loss_sum = 0.0
        self._epoch_batches = 0
        self._last_components: dict[str, float | None] = {}

        self.output_dir = Path(config.output_dir)
        self._setup_logging()
        LOGGER.info(
            "Trainer ready: device=%s amp=%s ddp=%d/%d compile=%s params=%d",
            self.device,
            self.amp_dtype,
            self.rank,
            self.world_size,
            self.config.compile,
            self._parameter_count(),
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _select_device(self, explicit: str | torch.device | None) -> torch.device:
        if explicit is not None:
            device = torch.device(explicit)
        elif self.config.device != "auto":
            device = torch.device(self.config.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            return torch.device("cpu")
        if device.type == "cuda" and device.index is None and self.world_size > 1:
            device = torch.device("cuda", self.local_rank)
        return device

    def _compute_total_steps(self) -> int:
        try:
            self._loader_len: int | None = len(self.train_loader)  # type: ignore[arg-type]
        except TypeError:
            self._loader_len = None
        if self.config.max_steps is not None:
            return int(self.config.max_steps)
        if self._loader_len is None:
            raise ValueError(
                "train_loader has no length; set TrainerConfig.max_steps"
            )
        if self._loader_len < 1:
            raise ValueError("train_loader must yield at least one batch")
        return int(self.config.epochs * self._loader_len)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("model has no trainable parameters")
        kwargs = {
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "betas": self.config.betas,
        }
        if self.device.type == "cuda":
            try:
                return torch.optim.AdamW(params, fused=True, **kwargs)
            except (TypeError, RuntimeError) as exc:  # pragma: no cover
                LOGGER.warning(
                    "fused AdamW unavailable (%s); using the standard kernel", exc
                )
        return torch.optim.AdamW(params, **kwargs)

    def _make_grad_scaler(self, enabled: bool):
        if not enabled:
            return _NoOpGradScaler()
        if hasattr(torch.amp, "GradScaler"):  # pragma: no cover - CUDA only
            try:
                return torch.amp.GradScaler("cuda")
            except (TypeError, RuntimeError):
                pass
        return torch.cuda.amp.GradScaler()  # pragma: no cover - CUDA only

    def _autocast_context(self):
        if not self._autocast_enabled:
            return contextlib.nullcontext()
        if hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype
            )
        return torch.cuda.amp.autocast(dtype=self.amp_dtype)  # pragma: no cover

    def _parameter_count(self) -> int:
        return sum(p.numel() for p in self.raw_model.parameters())

    @property
    def compile_fallback(self) -> bool:
        """True when a requested ``torch.compile`` fell back to eager."""
        return bool(
            self._compile_wrapper is not None and self._compile_wrapper.fallback
        )

    @property
    def compile_seconds(self) -> float | None:
        """First compiled-forward wall time (includes lazy Inductor compile)."""
        return getattr(self._compile_wrapper, "compile_seconds", None)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        if self.rank != 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.output_dir / "config.json"
        config_path.write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._log(
            {
                "event": "run",
                "world_size": self.world_size,
                "device": str(self.device),
                "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
                "total_steps": self.total_steps,
                "parameters": self._parameter_count(),
                "batch_size_per_gpu": self.config.batch_size_per_gpu,
                "compile": self.config.compile,
                "ddp_static_graph": self.config.ddp_static_graph,
                "schedule": self.config.schedule,
                "teacher_decay_epochs": self.config.teacher_decay_epochs,
            }
        )

    def _log(self, payload: Mapping[str, Any]) -> None:
        if self.rank != 0:
            return
        record = dict(payload)
        record.setdefault("time", round(time.time(), 3))
        line = json.dumps(record, sort_keys=True, default=str)
        print(line, flush=True)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / "train_log.jsonl", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:  # pragma: no cover - disk issues
            LOGGER.warning("could not append to train log: %s", exc)

    # ------------------------------------------------------------------
    # Device movement and loss parsing
    # ------------------------------------------------------------------

    def _move_to_device(self, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.is_floating_point() and value.dim() == 4:
                moved = value.to(self.device, non_blocking=True)
                if self.config.channels_last:
                    moved = moved.contiguous(memory_format=torch.channels_last)
                return moved
            return value.to(self.device, non_blocking=True)
        if isinstance(value, Mapping):
            return {key: self._move_to_device(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        return value

    def _extract_input(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, Mapping):
            if self.config.input_key not in batch:
                raise KeyError(
                    f"batch is missing input key {self.config.input_key!r}; "
                    f"available keys: {sorted(batch)}"
                )
            value = batch[self.config.input_key]
        elif isinstance(batch, (list, tuple)):
            value = batch[0]
        else:
            value = batch
        if not torch.is_tensor(value):
            raise TypeError(
                "could not find a tensor input in the batch; "
                f"got {type(value).__name__}"
            )
        return value

    def _batch_size(self, batch: Any) -> int:
        try:
            value = self._extract_input(batch)
        except (KeyError, TypeError):
            return 1
        return int(value.shape[0]) if value.dim() > 0 else 1

    def _notify_loss_epoch(self) -> None:
        """Call an optional ``loss.set_epoch(epoch)`` schedule hook.

        Losses that only implement the basic ``loss(outputs, batch)``
        interface are unaffected. ``OneStageLoss`` uses this to decay its
        teacher auxiliary; the call is made before each training epoch and at
        the start of validation.
        """
        setter = getattr(self.loss, "set_epoch", None)
        if callable(setter):
            setter(self.epoch)

    def _notify_loss_progress(self) -> None:
        """Call an optional ``loss.set_progress(step, total)`` schedule hook.

        ``Stage2Loss`` uses the fraction for its late perceptual schedule.
        The trainer passes completed optimizer steps, so the first batch sees
        progress 0.
        """
        setter = getattr(self.loss, "set_progress", None)
        if callable(setter):
            setter(self.global_step, self.total_steps)

    def _forward_outputs(self, model: nn.Module, inputs: torch.Tensor) -> Any:
        if self.forward_fn is None:
            return model(inputs)
        return self.forward_fn(model, inputs)

    def _split_loss_result(
        self, result: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(result, Mapping):
            if "loss" not in result:
                raise KeyError("loss mapping must contain a 'loss' entry")
            loss = result["loss"]
            components = {
                str(name): value.detach()
                for name, value in result.items()
                if name != "loss"
                and torch.is_tensor(value)
                and value.numel() == 1
            }
        else:
            loss = result
            components = {}
        if not torch.is_tensor(loss):
            raise TypeError(
                f"loss must return a tensor or a mapping, got {type(loss).__name__}"
            )
        if loss.numel() != 1:
            raise ValueError(f"loss must be a scalar, got shape {tuple(loss.shape)}")
        return loss, components

    @staticmethod
    def _as_float(value: torch.Tensor | float) -> float:
        if torch.is_tensor(value):
            return float(value.detach().float().item())
        return float(value)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def clip_gradients(self) -> float:
        """Clip gradients to ``config.grad_clip`` and return the pre-clip norm.

        A ``grad_clip`` of 0 disables clipping but still returns the norm.
        Must be called after ``scaler.unscale_(optimizer)``.
        """
        params = [p for p in self.model.parameters() if p.grad is not None]
        if not params:
            return 0.0
        if self.config.grad_clip > 0:
            return float(
                torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip)
            )
        total = torch.sqrt(
            sum(
                p.grad.detach().float().pow(2).sum()  # type: ignore[union-attr]
                for p in params
            )
        )
        return float(total)

    def _gradients_are_finite(self) -> bool:
        """Return False if any unscaled gradient is NaN/Inf.

        Must be called after ``GradScaler.unscale_``; scaled gradient norms are
        meaningless and can be large without indicating an overflow.
        """
        checks = [
            torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not checks:
            return True
        # One host sync per step instead of one per parameter.
        return bool(torch.stack(checks).all())

    def _nonfinite_gradient_names(self, limit: int = 5) -> list[str]:
        """Names of the first parameters with non-finite gradients."""
        names: list[str] = []
        for name, parameter in self.raw_model.named_parameters():
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                names.append(name)
                if len(names) >= limit:
                    break
        return names

    def assert_finite_state(self, *, check_grads: bool = True) -> None:
        """Raise ``FloatingPointError`` if model state is non-finite.

        Smoke mode calls this after every step. ``check_grads=True`` asserts
        the unscaled gradients before they are zeroed; an overflow step that
        was skipped should pass with ``check_grads=False`` because its
        parameters must remain unchanged and finite.
        """
        with torch.no_grad():
            for name, parameter in self.raw_model.named_parameters():
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(
                        f"non-finite parameter {name!r} "
                        f"(shape {tuple(parameter.shape)})"
                    )
                if (
                    check_grads
                    and parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ):
                    raise FloatingPointError(
                        f"non-finite gradient {name!r} "
                        f"(shape {tuple(parameter.grad.shape)})"
                    )
            for name, buffer in self.raw_model.named_buffers():
                if buffer.is_floating_point() and not bool(
                    torch.isfinite(buffer).all()
                ):
                    raise FloatingPointError(f"non-finite buffer {name!r}")

    def _train_step(self, batch: Any) -> None:
        batch = self._move_to_device(batch)
        inputs = self._extract_input(batch)
        self.last_lr = float(self.optimizer.param_groups[0]["lr"])
        self._notify_loss_progress()
        started = time.perf_counter()

        with self._autocast_context():
            outputs = self._forward_outputs(self.model, inputs)
            loss_result = self.loss(outputs, batch)
            loss, components = self._split_loss_result(loss_result)
        loss_value = self._as_float(loss)
        loss_finite = math.isfinite(loss_value)

        self.scaler.scale(loss).backward()
        # Unscale before any norm/finiteness check. Reporting the norm of
        # scaled gradients would make ordinary FP16 scaling look like an
        # overflow; an Inf below is a genuine unscaled NaN/Inf.
        self.scaler.unscale_(self.optimizer)
        grads_finite = self._gradients_are_finite()
        amp_found_inf = not grads_finite

        grad_norm: float | None = None
        if grads_finite:
            candidate = self.clip_gradients()
            if math.isfinite(candidate):
                grad_norm = candidate
            else:
                # Finite elements can still overflow the norm accumulator;
                # treat that as an overflow and never log Infinity.
                grads_finite = False
        overflow = (not grads_finite) or (not loss_finite)
        self.last_grad_norm = grad_norm
        self.last_loss = loss_value if loss_finite else None
        self.last_nonfinite_grad_params = (
            self._nonfinite_gradient_names() if not grads_finite else []
        )

        if overflow:
            self.overflow_steps += 1
            self.last_overflow_step = self.global_step + 1
            if self.scaler.is_enabled() and amp_found_inf:
                # unscale_ already recorded found_inf; step() skips the
                # optimizer update and update() decays the scale.
                self.scaler.step(self.optimizer)
                self.scaler.update()
            stepped = False
            self.skipped_steps += 1
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            stepped = True

        if self.config.finite_checks:
            # Gradients are still unscaled and present here. An overflow step
            # is allowed to have non-finite grads only because it was skipped;
            # its parameters are checked below.
            self.assert_finite_state(check_grads=not overflow)
        self.optimizer.zero_grad(set_to_none=True)
        if stepped:
            self.ema.update(self.raw_model)
        self.scheduler.step()

        self.global_step += 1
        if loss_finite:
            self._epoch_loss_sum += loss_value
        self._epoch_batches += 1

        elapsed = time.perf_counter() - started
        if self.global_step == 1 and self.first_step_seconds is None:
            # On the compile path this includes lazy Inductor compilation.
            self.first_step_seconds = elapsed
        if overflow or self.global_step % self.config.log_every == 0:
            # Always surface overflow steps even when log_every > 1.
            component_values = {
                name: self._as_float(value) for name, value in components.items()
            }
            self._last_components = {
                name: (value if math.isfinite(value) else None)
                for name, value in component_values.items()
            }
            samples = self._batch_size(batch) * self.world_size
            reduced_loss = self._reduce_mean(loss_value)
            if not math.isfinite(reduced_loss):
                reduced_loss = None
            self._log(
                {
                    "event": "step",
                    "epoch": self.epoch + 1,
                    "step": self.global_step,
                    "loss": reduced_loss,
                    "loss_finite": loss_finite,
                    "lr": self.last_lr,
                    "grad_norm": self.last_grad_norm,
                    "grad_norm_finite": grad_norm is not None,
                    "overflow": overflow,
                    "step_skipped": not stepped,
                    "overflow_steps": self.overflow_steps,
                    "skipped_steps": self.skipped_steps,
                    "nonfinite_grad_params": self.last_nonfinite_grad_params,
                    "samples_per_s": round(samples / max(elapsed, 1e-9), 2),
                    "vram_mb": self._peak_vram_mb(),
                    "components": self._last_components,
                }
            )

        if self.config.finite_checks:
            self.assert_finite_state(check_grads=False)
            if stepped and not grads_finite:
                raise FloatingPointError(
                    "optimizer stepped with non-finite gradients at step "
                    f"{self.global_step}"
                )
        if overflow and self.config.fail_on_overflow:
            raise FloatingPointError(
                "non-finite unscaled gradient or loss at step "
                f"{self.global_step}; optimizer update skipped "
                f"(overflow_steps={self.overflow_steps}, "
                f"skipped_steps={self.skipped_steps}, "
                f"nonfinite_grad_params={self.last_nonfinite_grad_params})"
            )

    # ------------------------------------------------------------------
    # Validation and epoch handling
    # ------------------------------------------------------------------

    def _all_reduce_sums(self, values: list[float]) -> list[float]:
        if not (dist.is_available() and dist.is_initialized()) or self.world_size <= 1:
            return list(values)
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return [float(value) for value in tensor.tolist()]

    def _reduce_mean(self, value: float) -> float:
        if not (dist.is_available() and dist.is_initialized()) or self.world_size <= 1:
            return float(value)
        totals = self._all_reduce_sums([float(value)])
        return totals[0] / float(self.world_size)

    def _peak_vram_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return round(torch.cuda.max_memory_allocated(self.device) / (1024**2), 1)

    def validate(self) -> dict[str, float]:
        """Run the validation loader and return ``{"val_loss": ...}`` plus components.

        When EMA weights exist and ``config.validate_ema`` is set, evaluation
        temporarily uses the EMA shadow weights, matching the export path.
        """
        if self.val_loader is None:
            return {}
        was_training = self.raw_model.training
        if isinstance(self.loss, nn.Module):
            self.loss.eval()
        self.raw_model.eval()
        self._notify_loss_epoch()
        loss_sum = 0.0
        count = 0
        component_sums: dict[str, float] = {}
        use_ema = self.config.validate_ema and self.ema.num_updates > 0
        ema_context = (
            self.ema.average_parameters(self.raw_model)
            if use_ema
            else contextlib.nullcontext()
        )
        # §5.4: benchmark is a training-shape optimization; evaluation may
        # see variable shapes, where autotuning can be counterproductive.
        benchmark_before = torch.backends.cudnn.benchmark
        if self.device.type == "cuda" and benchmark_before:
            torch.backends.cudnn.benchmark = False
        try:
            with ema_context, torch.no_grad(), self._autocast_context():
                for batch in self.val_loader:
                    batch = self._move_to_device(batch)
                    outputs = self._forward_outputs(
                        self.raw_model, self._extract_input(batch)
                    )
                    result = self.loss(outputs, batch)
                    loss, components = self._split_loss_result(result)
                    batch_size = self._batch_size(batch)
                    loss_sum += self._as_float(loss) * batch_size
                    count += batch_size
                    for name, value in components.items():
                        component_sums[name] = (
                            component_sums.get(name, 0.0)
                            + self._as_float(value) * batch_size
                        )
        finally:
            torch.backends.cudnn.benchmark = benchmark_before
            if was_training:
                self.raw_model.train()
            if isinstance(self.loss, nn.Module):
                self.loss.train()

        if count == 0:
            return {}
        names = sorted(component_sums)
        totals = self._all_reduce_sums(
            [loss_sum, float(count)]
            + [component_sums[name] for name in names]
        )
        total_count = totals[1]
        if total_count <= 0:
            return {}
        metrics: dict[str, float] = {
            "val_loss": totals[0] / total_count,
        }
        for index, name in enumerate(names):
            metrics[name] = totals[2 + index] / total_count
        metrics["val_loss"] = totals[0] / total_count
        return metrics

    def _finish_epoch(self) -> None:
        self.epoch += 1
        train_loss = (
            self._epoch_loss_sum / self._epoch_batches
            if self._epoch_batches
            else float("nan")
        )
        self._epoch_loss_sum = 0.0
        self._epoch_batches = 0
        self.batch_in_epoch = 0

        metrics: dict[str, float] = {}
        if (
            self.val_loader is not None
            and self.config.val_every_epochs > 0
            and self.epoch % self.config.val_every_epochs == 0
        ):
            metrics = self.validate()

        reduced_train_loss = self._reduce_mean(train_loss)
        if not math.isfinite(reduced_train_loss):
            reduced_train_loss = None
        entry: dict[str, Any] = {
            "event": "epoch",
            "epoch": self.epoch,
            "step": self.global_step,
            "train_loss": reduced_train_loss,
            "overflow_steps": self.overflow_steps,
            "skipped_steps": self.skipped_steps,
        }
        entry.update(metrics)
        self.history.append(entry)
        self._log(entry)

        improved = False
        val_loss = metrics.get("val_loss")
        if val_loss is not None:
            if self.best_metric is None or val_loss < (
                self.best_metric - self.config.early_stop_min_delta
            ):
                self.best_metric = val_loss
                self.bad_epochs = 0
                improved = True
            else:
                self.bad_epochs += 1

            if improved and self.config.save_best:
                self.save_checkpoint(
                    self.output_dir / self.config.best_checkpoint_name
                )
            if (
                self.config.early_stop_patience > 0
                and self.bad_epochs >= self.config.early_stop_patience
            ):
                LOGGER.info(
                    "Early stop at epoch %d (no improvement for %d epochs)",
                    self.epoch,
                    self.bad_epochs,
                )
                self.stopped_early = True

        if self.early_stop_hook is not None and self.early_stop_hook(entry):
            self.stopped_early = True

        if self.config.save_every_epoch:
            self.save_checkpoint()

    def _run_epoch(self, limit: int | None) -> bool:
        """Run one epoch, returning True when it completed."""
        self.model.train()
        if isinstance(self.loss, nn.Module):
            self.loss.train()
        self._notify_loss_epoch()
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)
        skip = self.batch_in_epoch
        for index, batch in enumerate(self.train_loader):
            if index < skip:
                continue
            if limit is not None and self.global_step >= limit:
                return False
            self._train_step(batch)
            self.batch_in_epoch = index + 1
        return True

    # ------------------------------------------------------------------
    # Public training entry point
    # ------------------------------------------------------------------

    def train(self, max_steps: int | None = None) -> list[dict[str, Any]]:
        """Train until ``config.epochs``/``config.max_steps`` or ``max_steps``.

        ``max_steps`` is an absolute optimizer-step cap for this call; it can
        stop mid-epoch. The in-epoch position is checkpointed so a later
        ``resume`` skips the batches already consumed.
        """
        if max_steps is not None:
            max_steps = int(max_steps)
            if max_steps < 0:
                raise ValueError("max_steps must be >= 0")
        limit = self.config.max_steps
        if max_steps is not None:
            limit = max_steps if limit is None else min(limit, max_steps)

        while self.epoch < self.config.epochs:
            if limit is not None and self.global_step >= limit:
                break
            completed = self._run_epoch(limit)
            if completed:
                self._finish_epoch()
                if self.stopped_early:
                    break
            else:
                break

        if self.config.save_on_exit:
            self.save_checkpoint()
        return self.history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": _np.random.get_state() if _np is not None else None,
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if _np is not None and state.get("numpy") is not None:
            _np.random.set_state(state["numpy"])
        if torch.cuda.is_available() and state.get("cuda"):
            try:
                torch.cuda.set_rng_state_all(state["cuda"])
            except RuntimeError as exc:  # pragma: no cover - device count change
                LOGGER.warning("could not restore CUDA RNG state: %s", exc)

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "config": self.config.to_dict(),
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict(),
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "global_step": self.global_step,
            "total_steps": self.total_steps,
            "loader_len": self._loader_len,
            "epoch_loss_sum": self._epoch_loss_sum,
            "epoch_batches": self._epoch_batches,
            "best_metric": self.best_metric,
            "bad_epochs": self.bad_epochs,
            "overflow_steps": self.overflow_steps,
            "skipped_steps": self.skipped_steps,
            "last_overflow_step": self.last_overflow_step,
            "history": self.history,
            "world_size": self.world_size,
            "parameters": self._parameter_count(),
            "rng": self._rng_state(),
        }

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Atomically save a checkpoint (rank 0 writes, all ranks return).

        With DDP every rank contributes its RNG state, so a resumed run
        restores rank-local RNG streams instead of rank 0's stream on every
        process. All ranks must call this method.
        """
        target = (
            Path(path)
            if path is not None
            else self.output_dir / self.config.checkpoint_name
        )
        local_rng = self._rng_state()
        rng_by_rank: list[dict[str, Any]] | None = None
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            gathered: list[Any] = [None] * self.world_size
            dist.all_gather_object(gathered, local_rng)
            rng_by_rank = gathered  # type: ignore[assignment]
        if self.rank == 0:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            payload = self._checkpoint_payload()
            payload["rng"] = local_rng
            if rng_by_rank is not None:
                payload["rng_by_rank"] = rng_by_rank
            with open(temporary, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            dist.barrier()
        return target

    @staticmethod
    def _load_torch(path: Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - torch < 1.13
            return torch.load(path, map_location="cpu")

    def resume(self, path: str | Path) -> None:
        """Restore full training state and continue exactly.

        Restores model, optimizer, scheduler, scaler, EMA, epoch, in-epoch
        batch position, step counters, history, best metric and RNG state.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = self._load_torch(checkpoint_path)
        if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
            raise ValueError(
                f"{checkpoint_path} is not an SLBR-Next trainer checkpoint"
            )

        self.raw_model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("ema") is not None:
            self.ema.load_state_dict(checkpoint["ema"])

        self.epoch = int(checkpoint.get("epoch", 0))
        self.batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.total_steps = int(checkpoint.get("total_steps", self.total_steps))
        self._epoch_loss_sum = float(checkpoint.get("epoch_loss_sum", 0.0))
        self._epoch_batches = int(checkpoint.get("epoch_batches", 0))
        self.best_metric = checkpoint.get("best_metric", None)
        self.bad_epochs = int(checkpoint.get("bad_epochs", 0))
        self.overflow_steps = int(checkpoint.get("overflow_steps", 0))
        self.skipped_steps = int(checkpoint.get("skipped_steps", 0))
        last_overflow_step = checkpoint.get("last_overflow_step", None)
        self.last_overflow_step = (
            None if last_overflow_step is None else int(last_overflow_step)
        )
        self.history = list(checkpoint.get("history", []))
        self.stopped_early = False

        if self.batch_in_epoch < 0:
            raise ValueError("checkpoint batch_in_epoch is negative")
        checkpoint_loader_len = checkpoint.get("loader_len", None)
        if (
            checkpoint_loader_len is not None
            and self._loader_len is not None
            and int(checkpoint_loader_len) != self._loader_len
        ):
            LOGGER.warning(
                "checkpoint loader_len=%s differs from current loader_len=%s; "
                "in-epoch resume may be offset",
                checkpoint_loader_len,
                self._loader_len,
            )
        if (
            self._loader_len is not None
            and self.batch_in_epoch > self._loader_len
        ):
            raise ValueError(
                "checkpoint batch_in_epoch=%d exceeds loader length %d"
                % (self.batch_in_epoch, self._loader_len)
            )
        checkpoint_world_size = int(checkpoint.get("world_size", self.world_size))
        if checkpoint_world_size != self.world_size:
            LOGGER.warning(
                "resuming with world_size %d (checkpoint had %d); "
                "optimizer/data order will differ",
                self.world_size,
                checkpoint_world_size,
            )
        scheduler_step = getattr(self.scheduler, "last_step", None)
        if scheduler_step is not None and int(scheduler_step) != self.global_step:
            LOGGER.warning(
                "scheduler last_step=%s differs from global_step=%d",
                scheduler_step,
                self.global_step,
            )
        rng_by_rank = checkpoint.get("rng_by_rank")
        if isinstance(rng_by_rank, (list, tuple)) and len(rng_by_rank) == self.world_size:
            self._restore_rng(rng_by_rank[self.rank] or {})
        else:
            self._restore_rng(checkpoint.get("rng") or {})
        self._log(
            {
                "event": "resume",
                "checkpoint": str(checkpoint_path),
                "epoch": self.epoch,
                "batch_in_epoch": self.batch_in_epoch,
                "step": self.global_step,
            }
        )


# ---------------------------------------------------------------------------
# Synthetic smoke helper (2xT4)
# ---------------------------------------------------------------------------


class _SyntheticWatermarkDataset(Dataset):
    """Random images with a rectangular mask; enough to exercise every head."""

    def __init__(self, length: int, size: int) -> None:
        self.length = int(length)
        self.size = int(size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        size = self.size
        image = torch.rand(3, size, size)
        mask = torch.zeros(1, size, size)
        height = max(1, size // 4)
        width = max(1, size // 3)
        top = int(torch.randint(0, size - height + 1, (1,)).item())
        left = int(torch.randint(0, size - width + 1, (1,)).item())
        mask[:, top : top + height, left : left + width] = 1.0
        return {"image": image, "clean": image.clone(), "mask": mask}


def _bce_probabilities_fp32(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Run probability BCE in FP32 with autocast disabled.

    ``SLBRNextV1.forward`` returns sigmoid probabilities, not logits, so the
    v1 smoke loss cannot switch to ``binary_cross_entropy_with_logits``
    without reconstructing logits. ``F.binary_cross_entropy`` is unsafe under
    AMP autocast (PyTorch raises even when the input is FP32); disable
    autocast for this op and cast both sides to FP32. The probability-BCE
    semantics and the existing ``1e-4`` clamp are unchanged.
    """
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        return F.binary_cross_entropy(
            prediction.float().clamp(1e-4, 1.0 - 1e-4),
            target.float(),
        )


def _smoke_loss(outputs: Any, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    (refine_bg, coarse_bg), mask_list, _ = outputs
    target = batch["clean"]
    reconstruction = F.l1_loss(refine_bg, target)
    loss = reconstruction + 0.5 * F.l1_loss(coarse_bg, target)
    mask_gt = batch["mask"]
    for mask in mask_list:
        resized = F.interpolate(mask_gt, size=mask.shape[-2:], mode="area")
        loss = loss + 0.1 * _bce_probabilities_fp32(mask, resized)
    return {"loss": loss, "l1": reconstruction.detach()}


def smoke_train(
    config: TrainerConfig | None = None,
    *,
    steps: int = 10,
    size: int = 256,
    batch_size: int | None = None,
    workers: int = 0,
    device: str | None = None,
    convnext_pretrained: bool = False,
) -> dict[str, Any]:
    """Run a synthetic 2-GPU smoke with the real v1 model.

    This is the T6 acceptance check: it exercises DDP, FP16/AMP, channels_last,
    fused AdamW, EMA, checkpoint save/load and (optionally) ``torch.compile``
    without needing the CLWD/LVW datasets. On 2xT4::

        torchrun --nproc_per_node=2 train_slbr_next.py --smoke --steps 10 \\
            --batch-size 6 --size 256 --compile \\
            --output-dir /kaggle/working/slbr_next_smoke

    The run prints ``SMOKE_TIMING {...}`` before ``SMOKE_OK {...}``. On the
    compile path ``first_step_s`` includes lazy Inductor compilation and
    ``compile_s`` isolates the first compiled forward; there is no compile
    timeout because interrupting Inductor cannot be done safely, but the
    catch-and-fallback path remains active. ``steady_state_steps_per_s``
    excludes that first step. The §5.4 ~15% wall-clock fallback is a
    long-run operator decision, not an automatic smoke behavior: compare
    ``steady_state_steps_per_s`` against ``--no-compile`` over a longer run
    and drop compile if it does not win; short smokes are dominated by
    ``compile_s`` by design.

    The process group is destroyed in ``finally`` on success and failure when
    this function created it; an externally created group is left alone.

    Smoke enables per-step finite checks and ``fail_on_overflow``: a skipped
    non-finite gradient/loss is logged with ``grad_norm: null``,
    ``overflow: true`` and ``step_skipped: true``, then raises and prints
    ``SMOKE_FAILED``. ``SMOKE_OK`` therefore means no unscaled Inf/NaN was
    observed and all parameters remain finite.

    Returns a JSON-serializable report. Synchronous units: ``steps`` are
    optimizer steps per rank, ``batch_size`` is images per GPU.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if size < 32:
        raise ValueError("size must be >= 32")
    if config is None:
        config = TrainerConfig()
    if batch_size is None:
        batch_size = config.batch_size_per_gpu
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    config = replace(
        config,
        epochs=1,
        max_steps=steps,
        batch_size_per_gpu=batch_size,
        warmup_steps=min(config.warmup_steps, steps),
        log_every=1,
        val_every_epochs=0,
        save_best=False,
        save_every_epoch=False,
        save_on_exit=False,
        finite_checks=True,
        fail_on_overflow=True,
    )

    smoke_started = time.perf_counter()
    already_initialized = dist.is_available() and dist.is_initialized()
    rank, world_size, local_rank = init_distributed(enabled=config.ddp)
    owns_process_group = (
        not already_initialized
        and dist.is_available()
        and dist.is_initialized()
    )
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        seed_everything(config.seed, rank)
        setup_done = time.perf_counter()

        from src.networks.slbr_next import SLBRNextV1

        model = SLBRNextV1(convnext_pretrained=convnext_pretrained)
        model_done = time.perf_counter()

        dataset = _SyntheticWatermarkDataset(
            length=steps * batch_size * world_size, size=size
        )
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        data_done = time.perf_counter()

        trainer = Trainer(
            model=model,
            loss=_smoke_loss,
            train_loader=loader,
            val_loader=None,
            config=config,
            device=device,
        )
        trainer_done = time.perf_counter()

        trainer.train()
        train_done = time.perf_counter()
        checkpoint = trainer.save_checkpoint()
        # Verify the resume path without allocating a second model.
        trainer.resume(checkpoint)
        # Final gate: SMOKE_OK must never describe non-finite parameters.
        trainer.assert_finite_state(check_grads=False)
        checkpoint_done = time.perf_counter()

        train_seconds = max(train_done - trainer_done, 1e-9)
        timing: dict[str, Any] = {
            "setup_s": round(setup_done - smoke_started, 3),
            "model_build_s": round(model_done - setup_done, 3),
            "data_s": round(data_done - model_done, 3),
            "trainer_init_s": round(trainer_done - data_done, 3),
            "train_s": round(train_seconds, 3),
            "first_step_s": (
                None
                if trainer.first_step_seconds is None
                else round(trainer.first_step_seconds, 3)
            ),
            "compile_s": (
                None
                if trainer.compile_seconds is None
                else round(trainer.compile_seconds, 3)
            ),
            "checkpoint_resume_s": round(checkpoint_done - train_done, 3),
            "total_s": round(checkpoint_done - smoke_started, 3),
            "steps": trainer.global_step,
            "steps_per_s": round(trainer.global_step / train_seconds, 4),
        }
        if (
            timing["first_step_s"] is not None
            and trainer.global_step > 1
            and timing["train_s"] > timing["first_step_s"]
        ):
            timing["steady_state_steps_per_s"] = round(
                (trainer.global_step - 1)
                / (timing["train_s"] - timing["first_step_s"]),
                4,
            )

        report: dict[str, Any] = {
            "ok": True,
            "steps": trainer.global_step,
            "epoch": trainer.epoch,
            "batch_size_per_gpu": batch_size,
            "size": size,
            "world_size": world_size,
            "amp_dtype": str(trainer.amp_dtype).replace("torch.", ""),
            "channels_last": config.channels_last,
            "compile": config.compile,
            "compile_fallback": trainer.compile_fallback,
            "parameters": trainer._parameter_count(),
            "final_loss": trainer.last_loss,
            "peak_vram_mb": trainer._peak_vram_mb(),
            "overflow_steps": trainer.overflow_steps,
            "skipped_steps": trainer.skipped_steps,
            "last_overflow_step": trainer.last_overflow_step,
            "nonfinite_grad_params": trainer.last_nonfinite_grad_params,
            "numerically_clean": trainer.overflow_steps == 0,
            "timing": timing,
            "checkpoint": str(checkpoint),
        }
        if rank == 0:
            print(
                "SMOKE_TIMING " + json.dumps(timing, sort_keys=True),
                flush=True,
            )
            print("SMOKE_OK " + json.dumps(report, sort_keys=True), flush=True)
        return report
    except Exception as exc:
        if rank == 0:
            failure: dict[str, Any] = {
                "elapsed_s": round(time.perf_counter() - smoke_started, 3),
                "error": str(exc),
            }
            failed_trainer = locals().get("trainer")
            if failed_trainer is not None:
                failure.update(
                    {
                        "first_step_s": (
                            None
                            if failed_trainer.first_step_seconds is None
                            else round(failed_trainer.first_step_seconds, 3)
                        ),
                        "compile_s": (
                            None
                            if failed_trainer.compile_seconds is None
                            else round(failed_trainer.compile_seconds, 3)
                        ),
                        "overflow_steps": failed_trainer.overflow_steps,
                        "skipped_steps": failed_trainer.skipped_steps,
                        "nonfinite_grad_params": (
                            failed_trainer.last_nonfinite_grad_params
                        ),
                    }
                )
            print(
                "SMOKE_FAILED "
                + json.dumps(failure, sort_keys=True),
                flush=True,
            )
        raise
    finally:
        if (
            owns_process_group
            and dist.is_available()
            and dist.is_initialized()
        ):
            try:
                dist.destroy_process_group()
            except Exception as exc:  # pragma: no cover - teardown best effort
                LOGGER.warning("destroy_process_group() failed: %s", exc)
