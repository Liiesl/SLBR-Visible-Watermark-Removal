"""Frozen SLBR teacher hooks and Stage 1 distillation targets (spec §4.1-§4.3).

This module wraps the 2021 SLBR network without importing it: the caller
passes an already-constructed ``torch.nn.Module`` whose architecture follows
``src/networks/resunet.py`` and ``src/networks/blocks.py``. The wrapper is
deliberately dependency-light (``torch`` plus the standard library) so the
training package can be imported without the legacy scikit-image/scipy stack.

Hook paths
----------
SLBR's ``CoarseDecoder`` has three levels (``i = 0, 1, 2``). For every level
the wrapper hooks the **pre-sigmoid** mask heads:

* primary head: ``coarse_decoder.up_convs_mask[i].primary_mask[0]``, the
  1x1 conv that produces ``[B, 1, H/4]``, ``[B, 1, H/2]`` or ``[B, 1, H]``
  logits;
* self-calibrated head:
  ``coarse_decoder.up_convs_mask[i].self_calibrated.out_conv[2]``, the last
  3x3 conv (index 2) that produces the same three logit resolutions.

``primary_mask[1]`` and the ``sigmoid`` inside ``SelfAttentionSimple.forward``
are intentionally bypassed: distillation consumes true logits, not
``logit(clamp(probability))``. The final teacher mask is the finest
self-calibrated logit, ``sigmoid(mask_logits[0])``.

Resolution ordering
-------------------
SLBR executes its levels **L1 (H/4) -> L2 (H/2) -> L3 (H)**, so hook execution
order is coarsest-to-finest. This wrapper pairs every captured tensor to the
student scale by its actual spatial resolution, never by hook order. The
returned ``TeacherTarget.mask_logits`` and ``primary_logits`` are therefore
ordered **finest-first**::

    mask_logits = (logits_H, logits_H/2, logits_H/4)

which matches ``SLBRNextV1.forward_logits`` and ``forward``'s ``mask_list``.
``mask_logits[0]`` is the one and only final-mask target. SLBR's
``mask_list[0]`` / ``mask_list[6]`` duplicate is **not** reproduced: there is
no fourth, identical target. Primary logits are a separate optional term for
T2; do not silently mix them with the self-calibrated targets at the same
scale.

Feature taps
------------
The two cheap Stage 1 feature-distillation taps are captured at these exact
module outputs (spec §4.3 and §5.4):

* ``shared_decoder.up_im_atts[0]`` -> shared bottleneck **image** feature,
  ``256 x H/8`` for the reference SLBR widths;
* ``coarse_decoder.up_convs_bg[0]`` -> coarse-decoder background feature,
  ``128 x H/4`` for the reference widths.

``shared_decoder.up_im_atts`` is the **image-stream** ECA block, so the hook
fires exactly once. Hooking ``shared_decoder.up_convs[0]`` would be ambiguous:
``SharedBottleNeck`` calls the same up-conv once for the image stream and once
for the mask stream.

Execution contract
------------------
* The teacher is forced to ``eval()`` and every parameter gets
  ``requires_grad_(False)`` in the wrapper constructor.
* ``forward`` runs under ``torch.no_grad()``.
* Inputs are RGB in ``[0, 1]``. **No ImageNet normalization is applied**;
  SLBR's ``norm``/``denorm`` are identity. The wrapper casts the input to
  float32 and moves it to the teacher's device.
* SLBR is square-only (its refinement stage reverses ``(H, W)``), so the
  wrapper requires a square input whose side is a multiple of 16. Stage 1 uses
  ``256 x 256``.
* ``fp16_autocast=True`` enables CUDA FP16 autocast around the teacher. All
  returned tensors, including mask logits, are upcast to FP32. On non-CUDA
  devices the flag is ignored and the teacher runs in FP32.
* Returned tensors are detached copies on ``output_device`` (CPU by default),
  so they can be cached or moved to a training device independently.

Cache contract for T2/T6
------------------------
Targets are serialised by :meth:`TeacherTarget.to_payload`, which contains
only tensors, integers, strings and lists, and reconstructed by
:meth:`TeacherTarget.from_payload`. :mod:`src.models.slbr_next.cache` stores
one payload per **non-empty string key** in atomic shard files. Only the
completed ``shard-XXXXXX.pt`` + ``shard-XXXXXX.json`` pairs are visible on
reopen; keys already present are skipped by ``put(key, target)``, and
``missing(keys)`` returns the keys that still need generation. A crash can at
worst leave an unregistered ``.pt`` orphan or a ``.tmp`` file, both of which
are ignored on resume. See :mod:`src.models.slbr_next.cache` for the on-disk
layout and the full resume semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "TEACHER_TARGET_FORMAT",
    "TEACHER_TARGET_VERSION",
    "TEACHER_SCALE_DIVISORS",
    "TeacherTarget",
    "SLBRTeacher",
]

#: Payload schema written into every cached teacher target.
TEACHER_TARGET_FORMAT = "slbr-next-teacher-target"
TEACHER_TARGET_VERSION = 1

#: Scale divisors in the finest-first order used by every target tuple.
TEACHER_SCALE_DIVISORS: tuple[int, int, int] = (1, 2, 4)

_LEVEL_COUNT = 3


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_tensor(name: str, tensor: object) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
        )
    return tensor


def _require_4d(name: str, tensor: Tensor) -> None:
    if tensor.dim() != 4:
        raise ValueError(
            f"{name} must be 4D [B, C, H, W], got shape "
            f"{tuple(tensor.shape)}"
        )


@dataclass(frozen=True)
class TeacherTarget:
    """Frozen Stage 1 targets produced by :class:`SLBRTeacher`.

    Every tensor keeps the teacher batch dimension. ``mask_logits`` and
    ``primary_logits`` are ordered **finest-first**, matching
    ``SLBRNextV1.forward_logits``::

        mask_logits[0]   self-calibrated logits at H       (final mask)
        mask_logits[1]   self-calibrated logits at H/2
        mask_logits[2]   self-calibrated logits at H/4

    ``mask_logits[0]`` is the only final-mask target. There is deliberately no
    second copy of the H self-calibrated map (SLBR's ``mask_list[0]`` /
    ``mask_list[6]`` duplicate is not reproduced). ``primary_logits`` follow
    the same H, H/2, H/4 order and are an *extra* term for T2, not a
    replacement for the self-calibrated targets.

    Attributes:
        mask_logits: Three pre-sigmoid self-calibrated tensors
            ``[B, 1, H/d, W/d]`` for ``d = 1, 2, 4``.
        primary_logits: Three pre-sigmoid primary-head tensors with the same
            shapes. Both teacher heads are 1-channel logits.
        refine_bg: Teacher refined background ``[B, 3, H, W]``.
        coarse_bg: Teacher coarse background ``[B, 3, H, W]``.
        bottleneck_h8: Shared bottleneck image feature ``[B, C, H/8, W/8]``
            captured at ``shared_decoder.up_im_atts[0]`` (256 channels for
            SLBR).
        bg_decoder_h4: Coarse-decoder background feature
            ``[B, C, H/4, W/4]`` captured at
            ``coarse_decoder.up_convs_bg[0]`` (128 channels for SLBR).
        height: Original input height.
        width: Original input width.

    The target is immutable as a dataclass, but the tensors themselves are
    ordinary tensors; callers should treat them as read-only.
    """

    mask_logits: tuple[Tensor, Tensor, Tensor]
    primary_logits: tuple[Tensor, Tensor, Tensor]
    refine_bg: Tensor
    coarse_bg: Tensor
    bottleneck_h8: Tensor
    bg_decoder_h4: Tensor
    height: int
    width: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mask_logits",
            self._freeze_triplet("mask_logits", self.mask_logits),
        )
        object.__setattr__(
            self,
            "primary_logits",
            self._freeze_triplet(
                "primary_logits", self.primary_logits
            ),
        )

        if (
            not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or self.height <= 0
        ):
            raise ValueError(
                f"height must be a positive int, got {self.height!r}"
            )
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or self.width <= 0
        ):
            raise ValueError(
                f"width must be a positive int, got {self.width!r}"
            )

        refine_bg = _require_tensor("refine_bg", self.refine_bg)
        coarse_bg = _require_tensor("coarse_bg", self.coarse_bg)
        bottleneck_h8 = _require_tensor(
            "bottleneck_h8", self.bottleneck_h8
        )
        bg_decoder_h4 = _require_tensor(
            "bg_decoder_h4", self.bg_decoder_h4
        )
        for name, tensor in (
            ("refine_bg", refine_bg),
            ("coarse_bg", coarse_bg),
            ("bottleneck_h8", bottleneck_h8),
            ("bg_decoder_h4", bg_decoder_h4),
        ):
            _require_4d(name, tensor)

        if refine_bg.shape != coarse_bg.shape:
            raise ValueError(
                "refine_bg and coarse_bg must have identical shapes, got "
                f"{tuple(refine_bg.shape)} and {tuple(coarse_bg.shape)}"
            )
        if refine_bg.shape[1] != 3:
            raise ValueError(
                "refine_bg/coarse_bg must have 3 RGB channels, got "
                f"{refine_bg.shape[1]}"
            )
        expected_full = (self.height, self.width)
        if tuple(refine_bg.shape[2:]) != expected_full:
            raise ValueError(
                "refine_bg/coarse_bg spatial size must match the input, got "
                f"{tuple(refine_bg.shape[2:])} vs {expected_full}"
            )

        batch = int(refine_bg.shape[0])
        if batch <= 0:
            raise ValueError(f"batch size must be positive, got {batch}")

        self._validate_triplet(
            "mask_logits", self.mask_logits, batch, expected_full
        )
        self._validate_triplet(
            "primary_logits", self.primary_logits, batch, expected_full
        )

        self._validate_feature(
            "bottleneck_h8",
            bottleneck_h8,
            batch,
            expected_full,
            divisor=8,
        )
        self._validate_feature(
            "bg_decoder_h4",
            bg_decoder_h4,
            batch,
            expected_full,
            divisor=4,
        )

    @staticmethod
    def _freeze_triplet(
        name: str, values: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(values, (tuple, list)):
            raise TypeError(
                f"{name} must be a tuple or list of three tensors, got "
                f"{type(values).__name__}"
            )
        values = tuple(values)
        if len(values) != _LEVEL_COUNT:
            raise ValueError(
                f"{name} must contain exactly {_LEVEL_COUNT} tensors, got "
                f"{len(values)}"
            )
        return values  # type: ignore[return-value]

    @staticmethod
    def _validate_triplet(
        name: str,
        values: tuple[Tensor, Tensor, Tensor],
        batch: int,
        expected_full: tuple[int, int],
    ) -> None:
        height, width = expected_full
        for tensor, divisor in zip(values, TEACHER_SCALE_DIVISORS):
            _require_tensor(name, tensor)
            _require_4d(name, tensor)
            if tensor.shape[0] != batch:
                raise ValueError(
                    f"{name} batch mismatch: expected {batch}, got "
                    f"{tensor.shape[0]}"
                )
            if tensor.shape[1] != 1:
                raise ValueError(
                    f"{name} must have 1 channel, got {tensor.shape[1]}"
                )
            expected = (
                _ceil_div(height, divisor),
                _ceil_div(width, divisor),
            )
            if tuple(tensor.shape[2:]) != expected:
                raise ValueError(
                    f"{name} at divisor {divisor} must be {expected}, got "
                    f"{tuple(tensor.shape[2:])}"
                )

    @staticmethod
    def _validate_feature(
        name: str,
        tensor: Tensor,
        batch: int,
        expected_full: tuple[int, int],
        divisor: int,
    ) -> None:
        _require_tensor(name, tensor)
        _require_4d(name, tensor)
        if tensor.shape[0] != batch:
            raise ValueError(
                f"{name} batch mismatch: expected {batch}, got "
                f"{tensor.shape[0]}"
            )
        if tensor.shape[1] <= 0:
            raise ValueError(
                f"{name} must have a positive channel count, got "
                f"{tensor.shape[1]}"
            )
        height, width = expected_full
        expected = (_ceil_div(height, divisor), _ceil_div(width, divisor))
        if tuple(tensor.shape[2:]) != expected:
            raise ValueError(
                f"{name} must be {expected}, got {tuple(tensor.shape[2:])}"
            )

    # ------------------------------------------------------------------
    # Convenient views (references, never copies of the final target)
    # ------------------------------------------------------------------

    @property
    def final_mask_logits(self) -> Tensor:
        """The finest self-calibrated logits, ``mask_logits[0]``."""
        return self.mask_logits[0]

    @property
    def final_mask(self) -> Tensor:
        """Teacher's final mask probability, ``sigmoid(mask_logits[0])``."""
        return torch.sigmoid(self.mask_logits[0])

    @property
    def mask_logits_h(self) -> Tensor:
        return self.mask_logits[0]

    @property
    def mask_logits_h2(self) -> Tensor:
        return self.mask_logits[1]

    @property
    def mask_logits_h4(self) -> Tensor:
        return self.mask_logits[2]

    @property
    def primary_logits_h(self) -> Tensor:
        return self.primary_logits[0]

    @property
    def primary_logits_h2(self) -> Tensor:
        return self.primary_logits[1]

    @property
    def primary_logits_h4(self) -> Tensor:
        return self.primary_logits[2]

    @property
    def batch_size(self) -> int:
        return int(self.refine_bg.shape[0])

    @property
    def input_size(self) -> tuple[int, int]:
        """Original ``(height, width)`` of the teacher input."""
        return (self.height, self.width)

    @property
    def feature_h8(self) -> Tensor:
        """Alias of :attr:`bottleneck_h8` (shared bottleneck H/8 tap)."""
        return self.bottleneck_h8

    @property
    def feature_h4(self) -> Tensor:
        """Alias of :attr:`bg_decoder_h4` (coarse-decoder H/4 tap)."""
        return self.bg_decoder_h4

    # ------------------------------------------------------------------
    # Batch splitting, payload round-trip
    # ------------------------------------------------------------------

    def tensors(self) -> tuple[Tensor, ...]:
        """Return every target tensor in a stable order.

        The order is ``mask_logits`` (H, H/2, H/4), ``primary_logits``
        (H, H/2, H/4), ``refine_bg``, ``coarse_bg``, ``bottleneck_h8``,
        ``bg_decoder_h4``. Useful for cache integrity checks and for moving a
        whole target to a device.
        """
        return (
            *self.mask_logits,
            *self.primary_logits,
            self.refine_bg,
            self.coarse_bg,
            self.bottleneck_h8,
            self.bg_decoder_h4,
        )

    def slice(self, index: int) -> "TeacherTarget":
        """Return sample ``index`` as a batch-of-one target."""
        if not 0 <= index < self.batch_size:
            raise IndexError(
                f"sample index {index} out of range for batch "
                f"{self.batch_size}"
            )

        def cut(tensor: Tensor) -> Tensor:
            return tensor[index : index + 1]

        return TeacherTarget(
            mask_logits=tuple(cut(t) for t in self.mask_logits),
            primary_logits=tuple(cut(t) for t in self.primary_logits),
            refine_bg=cut(self.refine_bg),
            coarse_bg=cut(self.coarse_bg),
            bottleneck_h8=cut(self.bottleneck_h8),
            bg_decoder_h4=cut(self.bg_decoder_h4),
            height=self.height,
            width=self.width,
        )

    def split(self) -> tuple["TeacherTarget", ...]:
        """Split a batched target into batch-of-one targets in batch order."""
        return tuple(self.slice(index) for index in range(self.batch_size))

    def to_payload(self) -> dict[str, object]:
        """Return a ``torch.save``-friendly payload.

        Only tensors, integers, strings and lists are used, so
        ``torch.load(..., weights_only=True)`` can read the shard. Tensors are
        not cloned or moved; cache writers currently receive CPU tensors from
        :class:`SLBRTeacher`.
        """
        return {
            "format": TEACHER_TARGET_FORMAT,
            "version": TEACHER_TARGET_VERSION,
            "height": int(self.height),
            "width": int(self.width),
            "scale_divisors": list(TEACHER_SCALE_DIVISORS),
            "mask_logits": list(self.mask_logits),
            "primary_logits": list(self.primary_logits),
            "refine_bg": self.refine_bg,
            "coarse_bg": self.coarse_bg,
            "bottleneck_h8": self.bottleneck_h8,
            "bg_decoder_h4": self.bg_decoder_h4,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "TeacherTarget":
        """Rebuild a target written by :meth:`to_payload`."""
        if not isinstance(payload, dict):
            raise TypeError(
                "teacher target payload must be a dict, got "
                f"{type(payload).__name__}"
            )
        if payload.get("format") != TEACHER_TARGET_FORMAT:
            raise ValueError(
                "unrecognised teacher target payload format: "
                f"{payload.get('format')!r}"
            )
        if payload.get("version") != TEACHER_TARGET_VERSION:
            raise ValueError(
                "unsupported teacher target payload version: "
                f"{payload.get('version')!r}"
            )
        required = (
            "height",
            "width",
            "mask_logits",
            "primary_logits",
            "refine_bg",
            "coarse_bg",
            "bottleneck_h8",
            "bg_decoder_h4",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                "teacher target payload is missing field(s): "
                + ", ".join(missing)
            )
        mask_logits = payload["mask_logits"]
        primary_logits = payload["primary_logits"]
        if not isinstance(mask_logits, (tuple, list)):
            raise TypeError("mask_logits payload must be a list")
        if not isinstance(primary_logits, (tuple, list)):
            raise TypeError("primary_logits payload must be a list")
        return cls(
            mask_logits=tuple(mask_logits),
            primary_logits=tuple(primary_logits),
            refine_bg=payload["refine_bg"],
            coarse_bg=payload["coarse_bg"],
            bottleneck_h8=payload["bottleneck_h8"],
            bg_decoder_h4=payload["bg_decoder_h4"],
            height=payload["height"],
            width=payload["width"],
        )


def _hook_tensor(output: object, label: str) -> Tensor:
    """Extract the tensor produced by a forward hook."""
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError(
                f"hook {label} returned an empty sequence"
            )
        output = output[0]
    if not isinstance(output, Tensor):
        raise TypeError(
            f"hook {label} expected a tensor output, got "
            f"{type(output).__name__}"
        )
    return output


class SLBRTeacher:
    """Read-only SLBR target extractor (spec §4.1-§4.3).

    Args:
        teacher: A constructed SLBR ``nn.Module`` exposing
            ``coarse_decoder.up_convs_mask``, ``coarse_decoder.up_convs_bg``
            and ``shared_decoder.up_im_atts``. The wrapper does not import the
            legacy network itself.
        output_device: Device for returned tensors, or ``None`` to keep them
            on the teacher's device. The default CPU keeps cached targets out
            of GPU memory.
        fp16_autocast: Wrap the teacher call in CUDA FP16 autocast. Ignored on
            CPU. Mask logits and every other returned tensor are upcast to
            FP32 afterwards.
        require_square: Require ``H == W``. SLBR's refinement stage reverses
            the spatial axes, so only square inputs are valid for Stage 1.
            Keep this enabled for the real teacher.

    The constructor calls ``teacher.eval()`` and disables gradients on every
    parameter. Use :meth:`forward` (or call the wrapper) under any outer
    context; the method itself is decorated with ``torch.no_grad()``.

    Example::

        teacher = SLBR(args=args, ...).eval()
        extractor = SLBRTeacher(teacher, fp16_autocast=torch.cuda.is_available())
        target = extractor(watermarked_rgb_256)  # [0, 1], no ImageNet norm
        final_logits = target.mask_logits[0]      # H
    """

    def __init__(
        self,
        teacher: nn.Module,
        *,
        output_device: str | torch.device | None = "cpu",
        fp16_autocast: bool = False,
        require_square: bool = True,
    ) -> None:
        if not isinstance(teacher, nn.Module):
            raise TypeError(
                "teacher must be a torch.nn.Module, got "
                f"{type(teacher).__name__}"
            )
        self.teacher = teacher
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        self.fp16_autocast = bool(fp16_autocast)
        self.require_square = bool(require_square)
        self._output_device = (
            None if output_device is None else torch.device(output_device)
        )
        self._teacher_device = self._first_parameter_device(teacher)

        (
            self._primary_heads,
            self._self_cal_heads,
            self._h8_module,
            self._h4_module,
        ) = self._resolve_modules(teacher)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_parameter_device(module: nn.Module) -> torch.device | None:
        for parameter in module.parameters():
            return parameter.device
        return None

    @staticmethod
    def _resolve_modules(
        teacher: nn.Module,
    ) -> tuple[
        tuple[nn.Module, ...],
        tuple[nn.Module, ...],
        nn.Module,
        nn.Module,
    ]:
        coarse = getattr(teacher, "coarse_decoder", None)
        if coarse is None:
            raise AttributeError(
                "teacher has no 'coarse_decoder'; expected an SLBR network "
                "from src/networks/resunet.py"
            )
        shared = getattr(teacher, "shared_decoder", None)
        if shared is None:
            raise AttributeError(
                "teacher has no 'shared_decoder'; expected an SLBR network "
                "from src/networks/resunet.py"
            )

        mask_levels = getattr(coarse, "up_convs_mask", None)
        if mask_levels is None or len(mask_levels) != _LEVEL_COUNT:
            raise AttributeError(
                "teacher.coarse_decoder.up_convs_mask must contain exactly "
                f"{_LEVEL_COUNT} SMR levels, got "
                f"{None if mask_levels is None else len(mask_levels)}"
            )
        bg_levels = getattr(coarse, "up_convs_bg", None)
        if bg_levels is None or len(bg_levels) < 1:
            raise AttributeError(
                "teacher.coarse_decoder.up_convs_bg must contain at least "
                "one MBE level"
            )
        image_atts = getattr(shared, "up_im_atts", None)
        if image_atts is None or len(image_atts) < 1:
            raise AttributeError(
                "teacher.shared_decoder.up_im_atts must contain at least "
                "one image-stream ECA block"
            )

        primary_heads: list[nn.Module] = []
        self_cal_heads: list[nn.Module] = []
        for level_index, level in enumerate(mask_levels):
            primary_seq = getattr(level, "primary_mask", None)
            if primary_seq is None or len(primary_seq) < 1:
                raise AttributeError(
                    "teacher.coarse_decoder.up_convs_mask"
                    f"[{level_index}].primary_mask is missing"
                )
            primary_head = primary_seq[0]
            if not isinstance(primary_head, nn.Conv2d):
                raise TypeError(
                    "primary mask head must be nn.Conv2d, got "
                    f"{type(primary_head).__name__}"
                )
            if tuple(primary_head.kernel_size) != (1, 1):
                raise ValueError(
                    "primary mask head must be 1x1, got kernel "
                    f"{tuple(primary_head.kernel_size)}"
                )
            if primary_head.out_channels != 1:
                raise ValueError(
                    "primary mask head must output 1 channel, got "
                    f"{primary_head.out_channels}"
                )

            self_cal = getattr(level, "self_calibrated", None)
            out_conv = (
                None
                if self_cal is None
                else getattr(self_cal, "out_conv", None)
            )
            if out_conv is None or len(out_conv) < 3:
                raise AttributeError(
                    "teacher.coarse_decoder.up_convs_mask"
                    f"[{level_index}].self_calibrated.out_conv is missing"
                )
            self_cal_head = out_conv[2]
            if not isinstance(self_cal_head, nn.Conv2d):
                raise TypeError(
                    "self-calibrated mask head must be nn.Conv2d, got "
                    f"{type(self_cal_head).__name__}"
                )
            if tuple(self_cal_head.kernel_size) != (3, 3):
                raise ValueError(
                    "self-calibrated mask head must be 3x3, got kernel "
                    f"{tuple(self_cal_head.kernel_size)}"
                )
            if self_cal_head.out_channels != 1:
                raise ValueError(
                    "self-calibrated mask head must output 1 channel, got "
                    f"{self_cal_head.out_channels}"
                )

            primary_heads.append(primary_head)
            self_cal_heads.append(self_cal_head)

        return (
            tuple(primary_heads),
            tuple(self_cal_heads),
            image_atts[0],
            bg_levels[0],
        )

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _prepare_input(self, x: Tensor) -> Tensor:
        _require_tensor("teacher input", x)
        if x.dim() != 4:
            raise ValueError(
                "teacher input must be 4D [B, 3, H, W], got shape "
                f"{tuple(x.shape)}"
            )
        if x.shape[1] != 3:
            raise ValueError(
                f"teacher input must have 3 RGB channels, got {x.shape[1]}"
            )
        if not x.is_floating_point():
            raise TypeError(
                f"teacher input must be floating point, got {x.dtype}"
            )
        height, width = int(x.shape[2]), int(x.shape[3])
        if height <= 0 or width <= 0:
            raise ValueError(
                f"teacher input must have positive spatial sizes, got "
                f"H={height}, W={width}"
            )
        if self.require_square and height != width:
            raise ValueError(
                "the SLBR teacher is square-only (its refinement stage "
                "reverses H/W), got "
                f"H={height}, W={width}; use 256x256 for Stage 1"
            )
        if height % 16 or width % 16:
            raise ValueError(
                "the SLBR teacher requires spatial sizes divisible by 16 "
                "(H/8 must survive the shared bottleneck's H/16 pool), "
                f"got H={height}, W={width}"
            )

        device = self._teacher_device
        if device is None:
            return x.detach().to(dtype=torch.float32)
        return x.detach().to(device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: Tensor) -> TeacherTarget:
        """Extract every Stage 1 target for a batch of RGB images.

        Args:
            x: ``[B, 3, H, W]`` in ``[0, 1]``, with square ``H == W`` a
                multiple of 16. No ImageNet normalization is applied.

        Returns:
            A :class:`TeacherTarget` with finest-first mask/primary logits,
            teacher outputs and the H/8 + H/4 feature taps.

        Raises:
            ValueError: For non-square/non-divisible inputs or if the hooked
                resolutions do not match the documented H, H/2, H/4 pyramid.
            RuntimeError: If a hook fires more than once (ambiguous feature
                stream) or not at all (wrong teacher architecture).
        """
        x = self._prepare_input(x)
        height, width = int(x.shape[2]), int(x.shape[3])

        primary_captures: list[tuple[int, Tensor]] = []
        self_cal_captures: list[tuple[int, Tensor]] = []
        feature_captures: dict[str, list[Tensor]] = {
            "bottleneck_h8": [],
            "bg_decoder_h4": [],
        }
        counts: dict[str, int] = {}
        handles = []

        def make_hook(
            store: list[tuple[int, Tensor]] | list[Tensor],
            label: str,
            level: int | None = None,
        ):
            counts[label] = 0

            def hook(module, inputs, output):
                counts[label] += 1
                if counts[label] > 1:
                    raise RuntimeError(
                        f"teacher hook {label} fired more than once; the "
                        "hooked module is shared or the teacher was called "
                        "twice in one forward"
                    )
                tensor = _hook_tensor(output, label)
                if level is None:
                    store.append(tensor)  # type: ignore[arg-type]
                else:
                    store.append((level, tensor))  # type: ignore[arg-type]

            return hook

        for level_index, module in enumerate(self._primary_heads):
            label = f"primary_mask[{level_index}][0]"
            handles.append(
                module.register_forward_hook(
                    make_hook(primary_captures, label, level_index)
                )
            )
        for level_index, module in enumerate(self._self_cal_heads):
            label = f"self_calibrated.out_conv[{level_index}][2]"
            handles.append(
                module.register_forward_hook(
                    make_hook(self_cal_captures, label, level_index)
                )
            )
        handles.append(
            self._h8_module.register_forward_hook(
                make_hook(feature_captures["bottleneck_h8"], "up_im_atts[0]")
            )
        )
        handles.append(
            self._h4_module.register_forward_hook(
                make_hook(feature_captures["bg_decoder_h4"], "up_convs_bg[0]")
            )
        )

        # The teacher is frozen by contract; re-assert eval in case a caller
        # switched it to train mode after constructing the wrapper.
        self.teacher.eval()

        try:
            if self.fp16_autocast and self._cuda_enabled():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.teacher(x)
            else:
                outputs = self.teacher(x)
        finally:
            for handle in handles:
                handle.remove()

        missing_hooks = [
            label for label, count in counts.items() if count == 0
        ]
        if missing_hooks:
            raise RuntimeError(
                "teacher hook(s) did not fire: " + ", ".join(missing_hooks)
            )

        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
            raise ValueError(
                "SLBR teacher must return (image_outputs, mask_list, ...); "
                f"got {type(outputs).__name__}"
            )
        image_outputs = outputs[0]
        if not isinstance(image_outputs, (tuple, list)) or len(
            image_outputs
        ) < 2:
            raise ValueError(
                "SLBR teacher must return [refine_bg, coarse_bg]; got "
                f"{type(image_outputs).__name__} with "
                f"{len(image_outputs) if isinstance(image_outputs, (tuple, list)) else '?'} "
                "entries"
            )
        refine_bg = _require_tensor("teacher refine_bg", image_outputs[0])
        coarse_bg = _require_tensor("teacher coarse_bg", image_outputs[1])
        for name, tensor in (
            ("refine_bg", refine_bg),
            ("coarse_bg", coarse_bg),
        ):
            _require_4d(name, tensor)
            if tensor.shape[1] != 3:
                raise ValueError(
                    f"teacher {name} must have 3 RGB channels, got "
                    f"{tensor.shape[1]}"
                )
            if tuple(tensor.shape[2:]) != (height, width):
                raise ValueError(
                    f"teacher {name} spatial size must be "
                    f"{(height, width)}, got {tuple(tensor.shape[2:])}"
                )

        mask_logits = self._pair_by_resolution(
            self_cal_captures, height, width, "self-calibrated"
        )
        primary_logits = self._pair_by_resolution(
            primary_captures, height, width, "primary"
        )

        bottleneck_h8 = self._finalize(
            self._single_feature(
                feature_captures["bottleneck_h8"], "up_im_atts[0]"
            )
        )
        bg_decoder_h4 = self._finalize(
            self._single_feature(
                feature_captures["bg_decoder_h4"], "up_convs_bg[0]"
            )
        )

        return TeacherTarget(
            mask_logits=mask_logits,
            primary_logits=primary_logits,
            refine_bg=self._finalize(refine_bg),
            coarse_bg=self._finalize(coarse_bg),
            bottleneck_h8=bottleneck_h8,
            bg_decoder_h4=bg_decoder_h4,
            height=height,
            width=width,
        )

    __call__ = forward
    extract = forward

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def _cuda_enabled(self) -> bool:
        return (
            self._teacher_device is not None
            and self._teacher_device.type == "cuda"
            and torch.cuda.is_available()
        )

    def _finalize(self, tensor: Tensor) -> Tensor:
        """Detach, upcast to FP32, move to the output device, copy."""
        result = tensor.detach().to(dtype=torch.float32)
        if self._output_device is not None:
            result = result.to(self._output_device)
        return result.clone()

    @staticmethod
    def _single_feature(captures: list[Tensor], label: str) -> Tensor:
        if len(captures) != 1:
            raise RuntimeError(
                f"teacher feature hook {label} must fire exactly once, "
                f"fired {len(captures)} times"
            )
        return captures[0]

    @staticmethod
    def _pair_by_resolution(
        captures: list[tuple[int, Tensor]],
        height: int,
        width: int,
        label: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pair hook outputs to H, H/2, H/4 by spatial size.

        Hook execution order is ignored. SLBR executes L1 (H/4) first and L3
        (H) last, which is the opposite of the student's finest-first order.
        """
        by_divisor: dict[int, tuple[int, Tensor]] = {}
        for level, tensor in captures:
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"{label} hook output must be a tensor, got "
                    f"{type(tensor).__name__}"
                )
            if tensor.dim() != 4:
                raise ValueError(
                    f"{label} hook output must be 4D, got shape "
                    f"{tuple(tensor.shape)}"
                )
            if tensor.shape[1] != 1:
                raise ValueError(
                    f"{label} hook output must have 1 channel, got "
                    f"{tensor.shape[1]}"
                )
            tensor_height = int(tensor.shape[2])
            tensor_width = int(tensor.shape[3])
            if (
                tensor_height <= 0
                or tensor_width <= 0
                or height % tensor_height
                or width % tensor_width
            ):
                raise ValueError(
                    f"{label} hook output spatial size "
                    f"{(tensor_height, tensor_width)} does not divide the "
                    f"input {(height, width)}"
                )
            divisor_h = height // tensor_height
            divisor_w = width // tensor_width
            if divisor_h != divisor_w:
                raise ValueError(
                    f"{label} hook output has inconsistent H/W scale: "
                    f"{(tensor_height, tensor_width)} for input "
                    f"{(height, width)}"
                )
            if divisor_h not in TEACHER_SCALE_DIVISORS:
                raise ValueError(
                    f"{label} hook output at {tensor_height}x{tensor_width} "
                    f"maps to divisor {divisor_h}, expected one of "
                    f"{TEACHER_SCALE_DIVISORS}"
                )
            if divisor_h in by_divisor:
                previous_level = by_divisor[divisor_h][0]
                raise ValueError(
                    f"{label} hook output appeared twice at divisor "
                    f"{divisor_h} (levels {previous_level} and {level}); "
                    "cannot pair scales unambiguously"
                )
            by_divisor[divisor_h] = (level, tensor)

        missing = [
            divisor
            for divisor in TEACHER_SCALE_DIVISORS
            if divisor not in by_divisor
        ]
        if missing:
            raise ValueError(
                f"{label} hooks are missing scale divisor(s) {missing}; "
                f"captured {sorted(by_divisor)}"
            )

        return tuple(
            by_divisor[divisor][1] for divisor in TEACHER_SCALE_DIVISORS
        )  # type: ignore[return-value]

    def extra_repr(self) -> str:
        return (
            f"teacher={type(self.teacher).__name__}, "
            f"fp16_autocast={self.fp16_autocast}, "
            f"output_device={self._output_device}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.extra_repr()})"
