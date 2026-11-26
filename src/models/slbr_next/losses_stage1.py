"""Stage 1 distillation losses for SLBR-Next (spec §4.2-§4.4, §5.4).

This module consumes the frozen targets produced by
:class:`~.teacher.TeacherTarget` (or a cached payload written by
:class:`~.cache.TeacherTargetCache`) and the student outputs captured by
:func:`run_student_stage1`.

Student outputs and forward hooks
---------------------------------
``SLBRNextV1.forward`` returns mask *probabilities*, not logits, and it does
not expose the H/8 and H/4 features needed for Channel-Wise Distillation
(CWD). :func:`run_student_stage1` runs one normal forward pass and captures,
with temporary forward hooks:

* ``model.mask_decoder`` -> the pre-sigmoid ``mask_logits``,
  ``mask_logits_h2`` and ``mask_logits_h4``. The captured tensors are cropped
  to the spatial sizes of the returned ``mask_list`` so that padded inputs
  behave like the public contract.
* ``model.bottleneck.bg_trunk`` -> the background trunk output at ``H/8``
  (``bg_h8``). The hook is attached to ``bg_trunk`` because
  ``SLBRNextBottleneck.forward_features`` calls it through ``__call__``;
  hooking ``bottleneck`` itself would never fire.
* ``model.background_decoder`` -> ``skip_h4`` (``H/4``) and, when requested,
  ``skip_h`` (``H``) from :class:`BackgroundDecoderOutput`.

The helper never changes ``SLBRNextV1.forward`` or the A1-A5 modules. It is
eager-friendly; ``torch.compile`` may graph-break on the hooks, so Stage 1
should run it eagerly (or compile only after the features have been wired
through another mechanism).

Loss components
---------------
Mask distillation (spec §4.2), for each teacher level ``d = 1, 2, 4``::

    bce_d = BCEWithLogits(student_logits_d / T,
                          sigmoid(teacher_logits_d / T)) * T^2
    dice_d = SoftDice(sigmoid(student_logits_d),
                      sigmoid(teacher_logits_d))

    mask = sum_d w_d * (bce_d + dice_d),  w = (1.0, 0.5, 0.25)

``TeacherTarget.mask_logits`` already contains exactly one target per scale
(finest first); the final H mask is used once. ``mask_list[0]`` is never
added a second time, and the teacher's optional ``primary_logits`` are
deliberately not mixed in.

Feature distillation (spec §4.3) uses CWD at ``H/8`` and ``H/4``: a temporary
1x1 adapter maps the student channels to the teacher channels, then a spatial
softmax followed by a per-channel cosine similarity is compared. Raw MSE on
BatchNorm/InstanceNorm features is never used. The adapters belong to the
criterion, not to the student, so dropping the criterion after Stage 1 removes
them; **they are trainable**, so include ``criterion.parameters()`` in the
Stage-1 optimizer and do not load them into the student checkpoint. Full-H
CWD is optional and scheduled every ``full_h_every`` steps; cached
``TeacherTarget`` objects have no full-H feature tap, so callers that want the
term must pass the teacher ``H`` feature explicitly as ``teacher_full_h``
(online teacher extraction) and provide a student ``feature_h``.

Output distillation (spec §4.4) applies Charbonnier inside the teacher mask
dilated by a 5 px radius plus a ``0.1x`` unmasked anchor to **both**
``refine_bg`` and ``coarse_bg``. The multi-scale FFT magnitude and real/imag
L1 term is applied to ``refine_bg`` only, always in FP32 with autocast
disabled, because cuFFT does not support BF16.

The returned mapping contains 0-dimensional FP32 tensors on the student
device::

    total, mask, mask_bce, mask_dice,
    mask_bce_h, mask_bce_h2, mask_bce_h4,
    mask_dice_h, mask_dice_h2, mask_dice_h4,
    cwd, cwd_h8, cwd_h4, cwd_h,
    charbonnier, charbonnier_refine, charbonnier_coarse,
    fft

``total = mask_weight * mask + cwd_weight * cwd
+ charbonnier_weight * charbonnier + fft_weight * fft``.

Numerical and shape notes
-------------------------
* No BatchNorm/InstanceNorm is assumed anywhere; cosine similarity is
  scale-insensitive and the adapters are 1x1 convolutions.
* ``H`` and ``W`` may differ. The loss itself is not square-only, but Stage 1
  *training* remains 256x256 because that is the teacher's native geometry
  (spec §4.5). Mask logits must already match their teacher levels; CWD and
  output terms resize only when a documented tap has a different spatial
  size.
* All teacher tensors are detached. Mask/FFT/CWD arithmetic runs in FP32 with
  autocast explicitly disabled for the FFT and for the logit losses.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .teacher import TeacherTarget

__all__ = [
    "StudentStage1Output",
    "Stage1Loss",
    "run_student_stage1",
]

#: Mask level names in ``TeacherTarget.mask_logits`` order (finest first).
_MASK_LEVEL_NAMES = ("h", "h2", "h4")

#: Divisors for the same order, used only for validation messages.
_MASK_LEVEL_DIVISORS = (1, 2, 4)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _autocast_disabled(device: torch.device):
    """Disable autocast for CPU/CUDA, which is where Stage 1 runs."""
    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def _zero(reference: Tensor) -> Tensor:
    """A 0-dimensional FP32 zero that is safe under autocast."""
    return reference.float().sum() * 0.0


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(value).__name__}"
        )
    return value


def _require_4d(name: str, value: object) -> Tensor:
    tensor = _require_tensor(name, value)
    if tensor.dim() != 4:
        raise ValueError(
            f"{name} must be 4D [N, C, H, W], got shape "
            f"{tuple(tensor.shape)}"
        )
    return tensor


def _resize_to(tensor: Tensor, size: tuple[int, int]) -> Tensor:
    """Bilinear resize with the repository's ``align_corners=False`` rule."""
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    return F.interpolate(
        tensor, size=tuple(size), mode="bilinear", align_corners=False
    )


def _soft_dice(prediction: Tensor, target: Tensor, eps: float) -> Tensor:
    """Per-sample SoftDice averaged over the batch."""
    dims = tuple(range(1, prediction.dim()))
    intersection = (prediction * target).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
    return (
        1.0
        - (2.0 * intersection + eps) / (denominator + eps)
    ).mean()


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    """Per-sample weighted mean over ``[N, C, H, W]`` and ``[N, 1, H, W]``."""
    weights = weights.expand_as(values)
    values = values.flatten(1)
    weights = weights.flatten(1)
    numerator = (values * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1e-8)
    return numerator / denominator


def _conv1x1_fp32(adapter: nn.Conv2d, x: Tensor) -> Tensor:
    """FP32 1x1 convolution that also tolerates a criterion on another device.

    During normal Stage 1 training the criterion is moved to the training
    device with the model. The explicit ``.to(device=...)`` calls keep the
    adapter usable if a caller forgets that move; gradients still flow back to
    the adapter parameters.
    """
    weight = adapter.weight.to(device=x.device, dtype=torch.float32)
    bias = (
        None
        if adapter.bias is None
        else adapter.bias.to(device=x.device, dtype=torch.float32)
    )
    return F.conv2d(x.float(), weight, bias, stride=1, padding=0)


def _cwd(
    student: Tensor, teacher: Tensor, temperature: float
) -> Tensor:
    """Channel-Wise Distillation with a spatial softmax + cosine similarity.

    Both operands are ``[N, C, H, W]`` in FP32 with the same channels and
    spatial size. The spatial softmax and the cosine are computed per sample
    and per channel, then averaged over channels and samples.
    """
    student_flat = student.flatten(2) / temperature
    teacher_flat = teacher.flatten(2) / temperature
    student_flat = F.softmax(student_flat, dim=-1)
    teacher_flat = F.softmax(teacher_flat, dim=-1)
    student_flat = F.normalize(student_flat, p=2, dim=-1)
    teacher_flat = F.normalize(teacher_flat, p=2, dim=-1)
    cosine = (student_flat * teacher_flat).sum(dim=-1)
    return (1.0 - cosine).mean()


def _coerce_target(target: object) -> TeacherTarget:
    """Accept a live/cached :class:`TeacherTarget` or its payload mapping."""
    if isinstance(target, TeacherTarget):
        return target
    if isinstance(target, Mapping):
        try:
            return TeacherTarget.from_payload(target)
        except (KeyError, TypeError, ValueError) as exc:
            raise TypeError(
                "target mapping is not a cached TeacherTarget payload"
            ) from exc
    raise TypeError(
        "target must be a TeacherTarget or its cached payload mapping, got "
        f"{type(target).__name__}"
    )


# ---------------------------------------------------------------------------
# Student outputs and feature extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentStage1Output:
    """One student forward with logits and the Stage 1 feature taps.

    Attributes:
        refine_bg: Refined background ``[N, 3, H, W]``.
        coarse_bg: Coarse background ``[N, 3, H, W]``.
        mask_logits: Pre-sigmoid mask logits at ``H``, ``H/2`` and ``H/4``,
            finest first. These must already match the teacher levels.
        feature_h8: Background trunk feature at ``H/8`` (256 channels for the
            reference student), or ``None`` to skip the H/8 CWD term.
        feature_h4: Background-decoder feature at ``H/4`` (128 channels for
            the reference student), or ``None``.
        feature_h: Optional full-``H`` background-decoder feature (32
            channels for the reference student). It is only used when
            :class:`Stage1Loss` has a full-H schedule enabled.
    """

    refine_bg: Tensor
    coarse_bg: Tensor
    mask_logits: tuple[Tensor, Tensor, Tensor]
    feature_h8: Tensor | None = None
    feature_h4: Tensor | None = None
    feature_h: Tensor | None = None

    def __post_init__(self) -> None:
        refine = _require_4d("refine_bg", self.refine_bg)
        coarse = _require_4d("coarse_bg", self.coarse_bg)
        if refine.shape != coarse.shape:
            raise ValueError(
                "refine_bg and coarse_bg must have identical shapes, got "
                f"{tuple(refine.shape)} and {tuple(coarse.shape)}"
            )
        if refine.shape[1] != 3:
            raise ValueError(
                "refine_bg/coarse_bg must have 3 RGB channels, got "
                f"{refine.shape[1]}"
            )

        logits = tuple(self.mask_logits)
        if len(logits) != len(_MASK_LEVEL_NAMES):
            raise ValueError(
                "mask_logits must contain exactly three levels (H, H/2, H/4), "
                f"got {len(logits)}; do not add a duplicate final mask"
            )
        for name, divisor, tensor in zip(
            _MASK_LEVEL_NAMES, _MASK_LEVEL_DIVISORS, logits
        ):
            tensor = _require_4d(f"mask_logits[{name}]", tensor)
            if tensor.shape[1] != 1:
                raise ValueError(
                    f"mask_logits[{name}] must have 1 channel, got "
                    f"{tensor.shape[1]}"
                )
            if tensor.shape[0] != refine.shape[0]:
                raise ValueError(
                    f"mask_logits[{name}] batch mismatch: "
                    f"{tensor.shape[0]} vs {refine.shape[0]}"
                )
            expected_h = _ceil_div(int(refine.shape[2]), divisor)
            expected_w = _ceil_div(int(refine.shape[3]), divisor)
            if tuple(tensor.shape[2:]) != (expected_h, expected_w):
                raise ValueError(
                    f"mask_logits[{name}] must be "
                    f"{(expected_h, expected_w)} for H={refine.shape[2]}, "
                    f"W={refine.shape[3]}, got {tuple(tensor.shape[2:])}"
                )
        object.__setattr__(self, "mask_logits", logits)

        batch = int(refine.shape[0])
        for name in ("feature_h8", "feature_h4", "feature_h"):
            tensor = getattr(self, name)
            if tensor is None:
                continue
            tensor = _require_4d(name, tensor)
            if tensor.shape[0] != batch:
                raise ValueError(
                    f"{name} batch mismatch: {tensor.shape[0]} vs {batch}"
                )


def run_student_stage1(
    model: nn.Module,
    x: Tensor,
    *,
    include_full_h: bool = False,
) -> StudentStage1Output:
    """Run one student forward and capture Stage 1 logits/features.

    The model is expected to expose the assembled v1 children
    ``mask_decoder``, ``bottleneck.bg_trunk`` and ``background_decoder``. The
    hooks are temporary and always removed, including when the forward raises.

    Args:
        model: An :class:`~src.networks.slbr_next.model.SLBRNextV1` (or a
            shaped-compatible module).
        x: ``[N, 3, H, W]`` input, normally 256x256 for Stage 1.
        include_full_h: Also capture the background decoder's ``skip_h``
            feature for the optional full-H CWD schedule.

    Returns:
        A :class:`StudentStage1Output` whose tensors keep their autograd graph.

    Raises:
        AttributeError: If the expected children/hook points are absent.
        RuntimeError: If a hook does not fire exactly once.
        ValueError: If the model's output layout is not the v1 contract.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"model must be an nn.Module, got {type(model).__name__}"
        )
    x = _require_4d("student input", x)

    mask_module = getattr(model, "mask_decoder", None)
    bottleneck = getattr(model, "bottleneck", None)
    bg_trunk = (
        None if bottleneck is None else getattr(bottleneck, "bg_trunk", None)
    )
    background_module = getattr(model, "background_decoder", None)

    missing = [
        name
        for name, module in (
            ("mask_decoder", mask_module),
            ("bottleneck.bg_trunk", bg_trunk),
            ("background_decoder", background_module),
        )
        if not isinstance(module, nn.Module)
    ]
    if missing:
        raise AttributeError(
            "run_student_stage1 needs the SLBR-Next v1 children: "
            + ", ".join(missing)
            + " is missing"
        )

    captured: dict[str, object] = {}
    handles = []

    def make_hook(label: str):
        def hook(module, inputs, output) -> None:
            del module, inputs
            if label in captured:
                raise RuntimeError(
                    f"student hook {label} fired more than once; the module "
                    "is shared or the model was called recursively"
                )
            captured[label] = output

        return hook

    for label, module in (
        ("mask_decoder", mask_module),
        ("bg_trunk", bg_trunk),
        ("background_decoder", background_module),
    ):
        handles.append(module.register_forward_hook(make_hook(label)))

    try:
        outputs = model(x)
    finally:
        for handle in handles:
            handle.remove()

    for label in ("mask_decoder", "bg_trunk", "background_decoder"):
        if label not in captured:
            raise RuntimeError(
                f"student hook {label} did not fire; check the v1 wiring"
            )

    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise ValueError(
            "student forward must return ([refine_bg, coarse_bg], mask_list, "
            f"[None]); got {type(outputs).__name__}"
        )
    image_outputs, mask_list = outputs[0], outputs[1]
    if not isinstance(image_outputs, (tuple, list)) or len(image_outputs) < 2:
        raise ValueError(
            "student image outputs must be [refine_bg, coarse_bg], got "
            f"{type(image_outputs).__name__}"
        )
    if not isinstance(mask_list, (tuple, list)) or len(mask_list) < 3:
        count = (
            len(mask_list)
            if isinstance(mask_list, (tuple, list))
            else "?"
        )
        raise ValueError(
            "student mask_list must contain at least the three H/H2/H4 "
            f"probabilities, got {count}"
        )
    refine_bg = _require_4d("student refine_bg", image_outputs[0])
    coarse_bg = _require_4d("student coarse_bg", image_outputs[1])

    mask_output = captured["mask_decoder"]
    mask_triplet = tuple(
        getattr(mask_output, name, None)
        for name in ("mask_logits", "mask_logits_h2", "mask_logits_h4")
    )
    if any(tensor is None for tensor in mask_triplet):
        raise ValueError(
            "mask_decoder output must expose mask_logits, mask_logits_h2 and "
            "mask_logits_h4"
        )

    # Crop padded logits to the public, unpadded mask_list sizes.
    cropped_logits = []
    for index, (logits, probability) in enumerate(
        zip(mask_triplet, mask_list[:3])
    ):
        logits = _require_4d(f"mask_logits[{index}]", logits)
        probability = _require_4d(
            f"mask_list[{index}]", probability
        )
        if probability.shape[0] != logits.shape[0]:
            raise ValueError(
                f"mask_list[{index}] batch mismatch: "
                f"{probability.shape[0]} vs {logits.shape[0]}"
            )
        height = int(probability.shape[2])
        width = int(probability.shape[3])
        if tuple(logits.shape[-2:]) != (height, width):
            logits = logits[..., :height, :width]
        cropped_logits.append(logits)

    bg_trunk_output = captured["bg_trunk"]
    if isinstance(bg_trunk_output, (tuple, list)):
        if not bg_trunk_output:
            raise RuntimeError("bottleneck.bg_trunk returned an empty tuple")
        bg_h8 = bg_trunk_output[0]
    else:
        bg_h8 = bg_trunk_output
    feature_h8 = _require_4d("student feature_h8", bg_h8)

    background_output = captured["background_decoder"]
    feature_h4 = _require_4d(
        "student feature_h4",
        getattr(background_output, "skip_h4", None),
    )
    feature_h = None
    if include_full_h:
        feature_h = _require_4d(
            "student feature_h",
            getattr(background_output, "skip_h", None),
        )

    return StudentStage1Output(
        refine_bg=refine_bg,
        coarse_bg=coarse_bg,
        mask_logits=tuple(cropped_logits),
        feature_h8=feature_h8,
        feature_h4=feature_h4,
        feature_h=feature_h,
    )


# ---------------------------------------------------------------------------
# Stage 1 criterion
# ---------------------------------------------------------------------------


class Stage1Loss(nn.Module):
    """Stage 1 mask + CWD + output distillation criterion (spec §4.2-§4.4).

    Args:
        mask_temperature: Distillation temperature ``T`` for the logit BCE
            (default 2.0).
        mask_scale_weights: Direct multipliers for the H, H/2 and H/4 mask
            levels (default ``1.0, 0.5, 0.25``). The level terms are summed
            as-is, not renormalized.
        mask_dice_eps: SoftDice smoothing constant (default 1.0).
        mask_weight: Weight of the complete mask term in ``total``.
        cwd_weight: Weight of the complete CWD term in ``total``.
        cwd_temperature: Spatial-softmax temperature used by CWD.
        student_h8_channels: Channels of ``feature_h8``.
        student_h4_channels: Channels of ``feature_h4``.
        student_h_channels: Channels of ``feature_h`` (full-H schedule only).
        teacher_h8_channels: Channels of ``TeacherTarget.bottleneck_h8``.
        teacher_h4_channels: Channels of ``TeacherTarget.bg_decoder_h4``.
        teacher_h_channels: Channels of the optional ``teacher_full_h``.
        cwd_h8_weight: Multiplier for the H/8 CWD tap inside ``cwd``.
        cwd_h4_weight: Multiplier for the H/4 CWD tap inside ``cwd``.
        cwd_h_weight: Multiplier for the full-H CWD tap inside ``cwd``.
        full_h_every: Apply full-H CWD every N steps. ``0`` disables it and
            no full-H adapter is created.
        charbonnier_weight: Weight of the combined Charbonnier term.
        charbonnier_eps: Charbonnier epsilon (default 1e-3, so ``eps^2`` is
            the spec's ``1e-6``).
        dilation_radius: Teacher-mask dilation radius in pixels (default 5,
            i.e. the 11x11 max-pool from spec §4.4).
        unmasked_weight: Outside-mask anchor multiplier (default 0.1).
        fft_weight: Weight of the refine-only FFT term. Stage 2 starts at
            0.05; Stage 1 may use the same value.
        fft_scales: Relative resolutions for the multi-scale FFT term.
    """

    def __init__(
        self,
        *,
        mask_temperature: float = 2.0,
        mask_scale_weights: tuple[float, float, float] = (1.0, 0.5, 0.25),
        mask_dice_eps: float = 1.0,
        mask_weight: float = 1.0,
        cwd_weight: float = 0.1,
        cwd_temperature: float = 1.0,
        student_h8_channels: int = 256,
        student_h4_channels: int = 128,
        student_h_channels: int = 32,
        teacher_h8_channels: int = 256,
        teacher_h4_channels: int = 128,
        teacher_h_channels: int = 32,
        cwd_h8_weight: float = 1.0,
        cwd_h4_weight: float = 1.0,
        cwd_h_weight: float = 1.0,
        full_h_every: int = 0,
        charbonnier_weight: float = 1.0,
        charbonnier_eps: float = 1e-3,
        dilation_radius: int = 5,
        unmasked_weight: float = 0.1,
        fft_weight: float = 0.05,
        fft_scales: tuple[float, ...] = (1.0, 0.5),
    ) -> None:
        super().__init__()

        if float(mask_temperature) <= 0.0:
            raise ValueError(
                f"mask_temperature must be positive, got {mask_temperature}"
            )
        if float(cwd_temperature) <= 0.0:
            raise ValueError(
                f"cwd_temperature must be positive, got {cwd_temperature}"
            )
        if float(mask_dice_eps) <= 0.0:
            raise ValueError(
                f"mask_dice_eps must be positive, got {mask_dice_eps}"
            )
        if float(charbonnier_eps) <= 0.0:
            raise ValueError(
                f"charbonnier_eps must be positive, got {charbonnier_eps}"
            )

        weights = tuple(float(weight) for weight in mask_scale_weights)
        if len(weights) != len(_MASK_LEVEL_NAMES):
            raise ValueError(
                "mask_scale_weights must contain three entries for "
                f"(H, H/2, H/4), got {weights}"
            )
        if any(weight < 0.0 for weight in weights):
            raise ValueError(
                f"mask_scale_weights must be non-negative, got {weights}"
            )
        if sum(weights) <= 0.0:
            raise ValueError("at least one mask_scale_weight must be positive")

        for name, value in (
            ("mask_weight", mask_weight),
            ("cwd_weight", cwd_weight),
            ("cwd_h8_weight", cwd_h8_weight),
            ("cwd_h4_weight", cwd_h4_weight),
            ("cwd_h_weight", cwd_h_weight),
            ("charbonnier_weight", charbonnier_weight),
            ("unmasked_weight", unmasked_weight),
            ("fft_weight", fft_weight),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        channel_args = {
            "student_h8_channels": student_h8_channels,
            "student_h4_channels": student_h4_channels,
            "student_h_channels": student_h_channels,
            "teacher_h8_channels": teacher_h8_channels,
            "teacher_h4_channels": teacher_h4_channels,
            "teacher_h_channels": teacher_h_channels,
        }
        for name, value in channel_args.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if int(dilation_radius) < 0:
            raise ValueError(
                f"dilation_radius must be non-negative, got {dilation_radius}"
            )
        if int(full_h_every) < 0:
            raise ValueError(
                f"full_h_every must be non-negative, got {full_h_every}"
            )
        fft_scales = tuple(float(scale) for scale in fft_scales)
        if not fft_scales or any(scale <= 0.0 for scale in fft_scales):
            raise ValueError(
                f"fft_scales must be positive, got {fft_scales}"
            )

        self.mask_temperature = float(mask_temperature)
        self.mask_scale_weights = weights
        self.mask_dice_eps = float(mask_dice_eps)
        self.mask_weight = float(mask_weight)
        self.cwd_weight = float(cwd_weight)
        self.cwd_temperature = float(cwd_temperature)
        self.cwd_h8_weight = float(cwd_h8_weight)
        self.cwd_h4_weight = float(cwd_h4_weight)
        self.cwd_h_weight = float(cwd_h_weight)
        self.full_h_every = int(full_h_every)
        self.charbonnier_weight = float(charbonnier_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        self.dilation_radius = int(dilation_radius)
        self.unmasked_weight = float(unmasked_weight)
        self.fft_weight = float(fft_weight)
        self.fft_scales = fft_scales
        self._step: int | None = None

        # Stage-1-only 1x1 adapters. They live in the criterion so the
        # student checkpoint never contains them; discard the criterion (or
        # just its adapters) after distillation.
        self.cwd_h8_adapter = nn.Conv2d(
            int(student_h8_channels),
            int(teacher_h8_channels),
            kernel_size=1,
            bias=False,
        )
        self.cwd_h4_adapter = nn.Conv2d(
            int(student_h4_channels),
            int(teacher_h4_channels),
            kernel_size=1,
            bias=False,
        )
        if self.full_h_every > 0:
            self.cwd_h_adapter = nn.Conv2d(
                int(student_h_channels),
                int(teacher_h_channels),
                kernel_size=1,
                bias=False,
            )
        else:
            self.cwd_h_adapter = None

    # ------------------------------------------------------------------
    # Full-H schedule
    # ------------------------------------------------------------------

    def set_step(self, step: int) -> None:
        """Record the optimizer step used by the full-H CWD schedule.

        A ``step`` passed directly to :meth:`forward` takes precedence over
        this value. The default internal step is 0, so the full-H term (when
        enabled) is active on the first batch.
        """
        step = int(step)
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        self._step = step

    @property
    def step(self) -> int | None:
        """The stored full-H schedule step, or ``None`` if never set."""
        return self._step

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: StudentStage1Output,
        target: TeacherTarget | Mapping[str, object],
        *,
        teacher_full_h: Tensor | None = None,
        step: int | None = None,
    ) -> dict[str, Tensor]:
        """Compute every Stage 1 loss component.

        Args:
            outputs: One :func:`run_student_stage1` result (or an equivalent
                hand-built :class:`StudentStage1Output`).
            target: A live/cached :class:`TeacherTarget`, or a payload
                mapping accepted by :meth:`TeacherTarget.from_payload`.
            teacher_full_h: Optional teacher ``[N, C, H, W]`` feature for the
                full-H CWD term. Cached targets do not contain this tap.
            step: Optimizer step for the full-H schedule. When omitted,
                :meth:`set_step` (or 0) is used.

        Returns:
            The documented dict of 0-dimensional FP32 tensors, including
            ``"total"``.
        """
        if not isinstance(outputs, StudentStage1Output):
            raise TypeError(
                "outputs must be a StudentStage1Output from "
                "run_student_stage1 (SLBRNextV1.forward returns probabilities "
                "and cannot be used directly for logit distillation), got "
                f"{type(outputs).__name__}"
            )
        target = _coerce_target(target)

        refine = outputs.refine_bg
        coarse = outputs.coarse_bg
        if refine.shape != coarse.shape:
            raise ValueError(
                "refine_bg and coarse_bg must have identical shapes, got "
                f"{tuple(refine.shape)} and {tuple(coarse.shape)}"
            )
        if target.batch_size != int(refine.shape[0]):
            raise ValueError(
                "teacher target batch mismatch: "
                f"{target.batch_size} vs {refine.shape[0]}"
            )

        if step is None:
            effective_step = 0 if self._step is None else self._step
        else:
            effective_step = int(step)
            if effective_step < 0:
                raise ValueError(
                    f"step must be non-negative, got {effective_step}"
                )

        mask_bce, mask_dice, mask_levels = self._mask_loss(
            outputs.mask_logits, target
        )
        mask = mask_bce + mask_dice

        cwd, cwd_h8, cwd_h4, cwd_h = self._feature_loss(
            outputs, target, teacher_full_h, effective_step
        )

        char_refine = self._charbonnier_masked(
            refine, target.refine_bg, target.mask_logits[0]
        )
        char_coarse = self._charbonnier_masked(
            coarse, target.coarse_bg, target.mask_logits[0]
        )
        charbonnier = 0.5 * (char_refine + char_coarse)

        fft = self._fft_loss(refine, target.refine_bg)

        total = (
            self.mask_weight * mask
            + self.cwd_weight * cwd
            + self.charbonnier_weight * charbonnier
            + self.fft_weight * fft
        )

        result: dict[str, Tensor] = {
            "total": total,
            "mask": mask,
            "mask_bce": mask_bce,
            "mask_dice": mask_dice,
            "cwd": cwd,
            "cwd_h8": cwd_h8,
            "cwd_h4": cwd_h4,
            "cwd_h": cwd_h,
            "charbonnier": charbonnier,
            "charbonnier_refine": char_refine,
            "charbonnier_coarse": char_coarse,
            "fft": fft,
        }
        for name in _MASK_LEVEL_NAMES:
            bce, dice = mask_levels[name]
            result[f"mask_bce_{name}"] = bce
            result[f"mask_dice_{name}"] = dice
        return result

    # ------------------------------------------------------------------
    # Mask distillation
    # ------------------------------------------------------------------

    def _mask_loss(
        self,
        student_logits: tuple[Tensor, Tensor, Tensor],
        target: TeacherTarget,
    ) -> tuple[Tensor, Tensor, dict[str, tuple[Tensor, Tensor]]]:
        levels = tuple(student_logits)
        if len(levels) != len(_MASK_LEVEL_NAMES):
            raise ValueError(
                "student mask_logits must contain exactly three levels, got "
                f"{len(levels)}; do not duplicate the final H mask"
            )

        device = levels[0].device
        temperature = self.mask_temperature
        bce_terms: list[Tensor] = []
        dice_terms: list[Tensor] = []
        per_level: dict[str, tuple[Tensor, Tensor]] = {}

        with _autocast_disabled(device):
            for name, divisor, student, teacher in zip(
                _MASK_LEVEL_NAMES,
                _MASK_LEVEL_DIVISORS,
                levels,
                target.mask_logits,
            ):
                student = _require_4d(
                    f"student mask_logits[{name}]", student
                ).float()
                teacher = _require_4d(
                    f"teacher mask_logits[{name}]", teacher
                ).detach().to(device=device, dtype=torch.float32)
                if student.shape[1] != 1 or teacher.shape[1] != 1:
                    raise ValueError(
                        f"mask level {name} must have 1 channel, got "
                        f"{student.shape[1]} and {teacher.shape[1]}"
                    )
                if student.shape[0] != teacher.shape[0]:
                    raise ValueError(
                        f"mask level {name} batch mismatch: "
                        f"{student.shape[0]} vs {teacher.shape[0]}"
                    )
                if student.shape[-2:] != teacher.shape[-2:]:
                    raise ValueError(
                        f"mask level {name} spatial mismatch: student "
                        f"{tuple(student.shape[-2:])} vs teacher "
                        f"{tuple(teacher.shape[-2:])}; Stage 1 mask logits "
                        "must already match their teacher level "
                        f"(divisor {divisor})"
                    )

                teacher_scaled = torch.sigmoid(teacher / temperature)
                bce = (
                    F.binary_cross_entropy_with_logits(
                        student / temperature, teacher_scaled
                    )
                    * temperature
                    * temperature
                )
                dice = _soft_dice(
                    torch.sigmoid(student),
                    torch.sigmoid(teacher),
                    self.mask_dice_eps,
                )
                bce_terms.append(bce)
                dice_terms.append(dice)
                per_level[name] = (bce, dice)

        bce_total = _zero(levels[0])
        dice_total = _zero(levels[0])
        for weight, bce, dice in zip(
            self.mask_scale_weights, bce_terms, dice_terms
        ):
            bce_total = bce_total + weight * bce
            dice_total = dice_total + weight * dice
        return bce_total, dice_total, per_level

    # ------------------------------------------------------------------
    # Feature distillation (CWD)
    # ------------------------------------------------------------------

    def _feature_loss(
        self,
        outputs: StudentStage1Output,
        target: TeacherTarget,
        teacher_full_h: Tensor | None,
        step: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        zero = _zero(outputs.refine_bg)

        if outputs.feature_h8 is None:
            cwd_h8 = zero
        else:
            cwd_h8 = self._cwd_tap(
                "h8",
                outputs.feature_h8,
                target.bottleneck_h8,
                self.cwd_h8_adapter,
            )
        if outputs.feature_h4 is None:
            cwd_h4 = zero
        else:
            cwd_h4 = self._cwd_tap(
                "h4",
                outputs.feature_h4,
                target.bg_decoder_h4,
                self.cwd_h4_adapter,
            )

        cwd_h = zero
        full_h_active = (
            self.cwd_h_adapter is not None
            and self.full_h_every > 0
            and step % self.full_h_every == 0
            and outputs.feature_h is not None
            and teacher_full_h is not None
        )
        if full_h_active:
            cwd_h = self._cwd_tap(
                "h",
                outputs.feature_h,
                teacher_full_h,
                self.cwd_h_adapter,
            )

        cwd = (
            self.cwd_h8_weight * cwd_h8
            + self.cwd_h4_weight * cwd_h4
            + self.cwd_h_weight * cwd_h
        )
        return cwd, cwd_h8, cwd_h4, cwd_h

    def _cwd_tap(
        self,
        name: str,
        student: Tensor,
        teacher: Tensor,
        adapter: nn.Conv2d | None,
    ) -> Tensor:
        if adapter is None:
            raise RuntimeError(
                f"no CWD adapter configured for the {name} tap"
            )
        student = _require_4d(f"student feature_{name}", student)
        teacher = _require_4d(f"teacher feature_{name}", teacher)
        if student.shape[1] != adapter.in_channels:
            raise ValueError(
                f"student feature_{name} has {student.shape[1]} channels; "
                f"the CWD adapter expects {adapter.in_channels}"
            )

        with _autocast_disabled(student.device):
            adapted = _conv1x1_fp32(adapter, student)
            teacher = teacher.detach().to(
                device=adapted.device, dtype=torch.float32
            )
            if adapted.shape[1] != teacher.shape[1]:
                raise ValueError(
                    f"teacher feature_{name} has {teacher.shape[1]} channels; "
                    f"the CWD adapter produces {adapted.shape[1]}"
                )
            if adapted.shape[0] != teacher.shape[0]:
                raise ValueError(
                    f"feature_{name} batch mismatch: "
                    f"{adapted.shape[0]} vs {teacher.shape[0]}"
                )
            adapted = _resize_to(adapted, teacher.shape[-2:])
            return _cwd(adapted, teacher, self.cwd_temperature)

    # ------------------------------------------------------------------
    # Output distillation
    # ------------------------------------------------------------------

    def _charbonnier_masked(
        self,
        prediction: Tensor,
        target: Tensor,
        teacher_mask_logits: Tensor,
    ) -> Tensor:
        prediction = _require_4d("prediction", prediction).float()
        target = _require_4d("target", target).detach().to(
            device=prediction.device, dtype=torch.float32
        )
        if target.shape[0] != prediction.shape[0]:
            raise ValueError(
                f"target batch mismatch: {target.shape[0]} vs "
                f"{prediction.shape[0]}"
            )
        if target.shape[1] != prediction.shape[1]:
            raise ValueError(
                f"target has {target.shape[1]} channels, prediction has "
                f"{prediction.shape[1]}"
            )
        target = _resize_to(target, prediction.shape[-2:])

        mask = torch.sigmoid(
            teacher_mask_logits.detach().to(
                device=prediction.device, dtype=torch.float32
            )
        )
        if mask.shape[0] != prediction.shape[0] or mask.shape[1] != 1:
            raise ValueError(
                "teacher mask must be [N, 1, H, W], got "
                f"{tuple(mask.shape)}"
            )
        mask = _resize_to(mask, prediction.shape[-2:]).clamp(0.0, 1.0)

        kernel = 2 * self.dilation_radius + 1
        if kernel > 1:
            dilated = F.max_pool2d(
                mask,
                kernel_size=kernel,
                stride=1,
                padding=self.dilation_radius,
            )
        else:
            dilated = mask
        dilated = dilated.clamp(0.0, 1.0)

        error = torch.sqrt(
            (prediction - target).square()
            + self.charbonnier_eps * self.charbonnier_eps
        )
        inside = _weighted_mean(error, dilated)
        outside = _weighted_mean(error, 1.0 - dilated)
        return (inside + self.unmasked_weight * outside).mean()

    def _fft_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Multi-scale FFT L1, forced to FP32 under any autocast context."""
        with _autocast_disabled(prediction.device):
            prediction = prediction.float()
            target = target.detach().to(
                device=prediction.device, dtype=torch.float32
            )
            if target.shape[0] != prediction.shape[0]:
                raise ValueError(
                    f"FFT target batch mismatch: {target.shape[0]} vs "
                    f"{prediction.shape[0]}"
                )
            if target.shape[1] != prediction.shape[1]:
                raise ValueError(
                    f"FFT target has {target.shape[1]} channels, prediction "
                    f"has {prediction.shape[1]}"
                )
            target = _resize_to(target, prediction.shape[-2:])

            height, width = prediction.shape[-2:]
            total = prediction.new_zeros(())
            weight_sum = 0.0

            for index, scale in enumerate(self.fft_scales):
                if scale == 1.0:
                    pred_level = prediction
                    target_level = target
                else:
                    size = (
                        max(1, int(round(height * scale))),
                        max(1, int(round(width * scale))),
                    )
                    pred_level = F.interpolate(
                        prediction, size=size, mode="area"
                    )
                    target_level = F.interpolate(
                        target, size=size, mode="area"
                    )

                pred_fft = torch.fft.rfft2(pred_level, norm="ortho")
                target_fft = torch.fft.rfft2(target_level, norm="ortho")

                magnitude = (
                    pred_fft.abs() - target_fft.abs()
                ).abs().mean()
                real = (pred_fft.real - target_fft.real).abs().mean()
                imag = (pred_fft.imag - target_fft.imag).abs().mean()

                weight = 0.5 ** index
                total = total + weight * (magnitude + real + imag)
                weight_sum += weight

            return total / weight_sum

    def extra_repr(self) -> str:
        return (
            f"mask_weight={self.mask_weight}, "
            f"cwd_weight={self.cwd_weight}, "
            f"charbonnier_weight={self.charbonnier_weight}, "
            f"fft_weight={self.fft_weight}, "
            f"full_h_every={self.full_h_every}"
        )
