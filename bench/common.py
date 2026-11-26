"""Shared synthetic-data and loss adapters for the T6 gate tooling.

Everything here is deliberately dependency-light: ``torch`` plus the standard
library. The training package (``src.models.slbr_next``) is imported by the
callers, never by this module, so the CPU test suite can import the bench
helpers without defeating ``tests/test_slbr_next/test_imports.py``.

Why the loss adapter exists
---------------------------
``SLBRNextV1.forward`` returns **probabilities** in ``mask_list`` while
:class:`~src.models.slbr_next.losses_stage2.Stage2Loss` expects **pre-sigmoid
logits**. The adapter converts only that boundary:

``logit(clamp(probability, 1e-4, 1 - 1e-4))``

The clamp is the same numerical guard used by the legacy smoke loss and keeps
the conversion finite in FP16. The criterion itself is T3's unchanged
``Stage2Loss``; the benchmark/smoke runs with ``perceptual=None`` so no
LPIPS/DISTS download or dataset is needed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch
from torch import nn


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True for a CUDA out-of-memory failure.

    ``torch.cuda.OutOfMemoryError`` exists on recent PyTorch versions, but
    ``torch.compile``/Inductor and some CUDA paths wrap the same message in a
    plain ``RuntimeError``. Both are classified by message as a fallback.
    """
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def mask_probabilities_to_logits(
    masks: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    eps: float = 1e-4,
) -> list[torch.Tensor]:
    """Convert model mask probabilities to finite logits.

    Args:
        masks: A single ``[N, 1, H, W]`` probability tensor or the v1
            ``mask_list`` (finest to coarsest).
        eps: Clamp bound applied before ``logit``.

    Returns:
        A list of FP32 logits, one per input level. The loss casts to FP32
        internally, so doing it here keeps the conversion stable under AMP.
    """
    if isinstance(masks, torch.Tensor):
        levels = [masks]
    else:
        levels = list(masks)
    if not levels:
        raise ValueError("masks must contain at least one level")
    logits: list[torch.Tensor] = []
    for level in levels:
        if not isinstance(level, torch.Tensor):
            raise TypeError(
                "every mask level must be a torch.Tensor, got "
                f"{type(level).__name__}"
            )
        probabilities = level.float().clamp(float(eps), 1.0 - float(eps))
        logits.append(torch.logit(probabilities))
    return logits


class Stage2ProbabilityLoss(nn.Module):
    """Trainer-compatible adapter around T3's ``Stage2Loss``.

    The wrapped criterion is T3's real loss. This module only adapts the v1
    output tuple (probabilities) to the Stage 2 input contract (logits) and
    exposes the trainer's required ``"loss"`` key.

    Args:
        criterion: A ``Stage2Loss`` instance, normally built with
            ``perceptual=None`` for offline throughput/smoke runs.
        clean_key: Batch key for the clean target.
        mask_key: Batch key for the GT mask.
        confidence_key: Optional pseudo-GT confidence key.
    """

    def __init__(
        self,
        criterion: nn.Module,
        *,
        clean_key: str = "clean",
        mask_key: str = "mask",
        confidence_key: str = "confidence",
    ) -> None:
        super().__init__()
        if not callable(criterion):
            raise TypeError("criterion must be callable")
        self.criterion = criterion
        self.clean_key = str(clean_key)
        self.mask_key = str(mask_key)
        self.confidence_key = str(confidence_key)

    def forward(
        self, outputs: Any, batch: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "Stage2ProbabilityLoss expects a mapping batch, got "
                f"{type(batch).__name__}"
            )
        try:
            image_outputs, masks, _ = outputs
            refine_bg, coarse_bg = image_outputs
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "expected the v1 output ([refine_bg, coarse_bg], mask_list, "
                "[None])"
            ) from exc
        for key in (self.clean_key, self.mask_key):
            if key not in batch:
                raise KeyError(f"batch is missing {key!r}")
        confidence = batch.get(self.confidence_key)
        result = self.criterion(
            refine_bg,
            coarse_bg,
            mask_probabilities_to_logits(masks),
            batch[self.clean_key],
            batch[self.mask_key],
            confidence=confidence,
        )
        if not isinstance(result, Mapping) or "total" not in result:
            raise TypeError("Stage2Loss must return a mapping with 'total'")
        payload = dict(result)
        payload["loss"] = result["total"]
        return payload

    def extra_repr(self) -> str:
        return (
            f"clean_key={self.clean_key!r}, mask_key={self.mask_key!r}, "
            f"confidence_key={self.confidence_key!r}"
        )


def _rectangular_mask(
    batch_size: int,
    size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """One random rectangular watermark region per sample."""
    mask = torch.zeros(batch_size, 1, size, size)
    minimum = max(1, size // 8)
    maximum = max(minimum + 1, size // 3 + 1)
    for index in range(batch_size):
        height = int(
            torch.randint(minimum, maximum, (1,), generator=generator).item()
        )
        width = int(
            torch.randint(minimum, maximum, (1,), generator=generator).item()
        )
        top = int(
            torch.randint(0, max(1, size - height + 1), (1,), generator=generator).item()
        )
        left = int(
            torch.randint(0, max(1, size - width + 1), (1,), generator=generator).item()
        )
        mask[index, :, top : top + height, left : left + width] = 1.0
    return mask


def synthetic_stage2_batch(
    batch_size: int,
    size: int,
    *,
    seed: int = 0,
    alpha: float = 0.4,
) -> dict[str, torch.Tensor]:
    """Build one synthetic ``image``/``clean``/``mask`` batch on CPU.

    The composite is a linear blend, so ``clean`` is a valid reconstruction
    target. Masks are rectangular regions with a soft-ish random coverage;
    the exact content is irrelevant for a throughput/smoke gate.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if size < 16:
        raise ValueError("size must be >= 16")
    generator = torch.Generator().manual_seed(int(seed))
    clean = torch.rand(batch_size, 3, size, size, generator=generator)
    watermark = torch.rand(batch_size, 3, size, size, generator=generator)
    alpha_map = torch.full((batch_size, 1, 1, 1), float(alpha))
    image = (1.0 - alpha_map) * clean + alpha_map * watermark
    mask = _rectangular_mask(batch_size, size, generator)
    return {"image": image, "clean": clean, "mask": mask}


TargetBuilder = Callable[
    [int, int, int, Mapping[str, torch.Tensor]], Mapping[str, Any]
]


class SyntheticStepLoader:
    """Deterministic, re-iterable synthetic loader for short training runs.

    ``Trainer`` only requires ``len(loader)`` and iteration, so this avoids a
    DataLoader/sampler while keeping the synthetic subset reproducible. Each
    step's data is seeded by ``seed + rank * 100003 + step``; a resume that
    restarts an epoch therefore sees the same batches.

    Args:
        steps: Batches yielded per epoch.
        batch_size: Images per rank/GPU.
        size: Square image side.
        seed: Base seed.
        rank: Optional rank offset for data diversity.
        target_builder: Optional callable
            ``(batch_size, size, seed, batch) -> mapping`` merged into each
            batch. Used by the Stage 1/one-stage smoke rows to attach a
            synthetic ``TeacherTarget``.
    """

    def __init__(
        self,
        steps: int,
        batch_size: int,
        size: int,
        *,
        seed: int = 0,
        rank: int = 0,
        target_builder: TargetBuilder | None = None,
    ) -> None:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.size = int(size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.target_builder = target_builder

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for step in range(self.steps):
            batch_seed = self.seed + self.rank * 100_003 + step
            batch: dict[str, Any] = synthetic_stage2_batch(
                self.batch_size, self.size, seed=batch_seed
            )
            if self.target_builder is not None:
                extra = self.target_builder(
                    self.batch_size, self.size, batch_seed, batch
                )
                if not isinstance(extra, Mapping):
                    raise TypeError(
                        "target_builder must return a mapping, got "
                        f"{type(extra).__name__}"
                    )
                batch.update(extra)
            yield batch

    def materialize(self) -> list[dict[str, Any]]:
        """Return all batches as a list, outside any timed region."""
        return list(self)


def mean_finite(values: list[float | None]) -> float | None:
    """Mean of a list of finite floats, or ``None`` if any entry is invalid."""
    if not values:
        return None
    numbers: list[float] = []
    for value in values:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        numbers.append(number)
    return sum(numbers) / len(numbers)
