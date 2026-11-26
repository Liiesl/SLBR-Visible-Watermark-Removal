"""Stage 2 losses and synthetic composites for SLBR-Next (spec §5.1-§5.2).

This module is the training-side counterpart of the shipping v1 contract in
``src.networks.slbr_next.model``. It is deliberately independent of the Kaggle
dataset pipeline (K3 owns shards) and only depends on ``torch``.

Synthetic blend families
------------------------

:func:`build_synthetic_composite` turns a clean background ``B`` and a
watermark layer ``W`` (both ``[N, C, H, W]`` in ``[0, 1]``) into a synthetic
watermarked image ``I``. Three families are supported, either for the whole
batch or per sample:

===========  ==========================================  =================
``mode``     formula                                     alpha target
===========  ==========================================  =================
``linear``   ``I = (1 - a) * B + a * W``                 ``a`` (valid)
``screen``   ``I = 1 - (1 - B) * (1 - W)``               none
``multiply`` ``I = B * W``                               none
===========  ==========================================  =================

Alpha semantics
~~~~~~~~~~~~~~~

* Only ``linear`` is a true alpha blend, so only linear samples may supervise
  an alpha head.
* When every sample is nonlinear, :attr:`SyntheticComposite.alpha` is
  ``None``. It is never a fabricated label.
* In a mixed batch, ``alpha`` is a tensor with **zero rows for nonlinear
  samples** and :attr:`SyntheticComposite.alpha_valid` is a boolean ``[N]``
  mask that says which rows carry a real alpha target. Callers must pass that
  mask to :class:`Stage2Loss`; an alpha value is meaningless without it.
* ``alpha`` may be a scalar, a per-sample ``[N]``/``[N, 1]``/``[N, 1, 1]``
  tensor, or a full ``[N, 1, H, W]`` map. ``None`` draws a per-sample value
  uniformly from ``alpha_range``. All construction uses ``torch`` ops, so it
  can run on GPU.

Stage 2 loss
------------

:class:`Stage2Loss` consumes the v1 student outputs
``([refine_bg, coarse_bg], mask_list, [None])`` and a clean target::

    (refine_bg, coarse_bg), mask_list, _ = model(watermarked)
    losses = criterion(
        refine_bg,
        coarse_bg,
        mask_list,          # finest -> coarsest, or just mask_list[0]
        clean,
        gt_mask,
        alpha_pred=alpha_head_features,
        alpha_gt=composite.alpha,
        alpha_valid=composite.alpha_valid,
    )
    losses["total"].backward()

Components and §5.2 starting weights (all configurable):

=========================  ======  ==========================================
component                  weight  detail
=========================  ======  ==========================================
reconstruction             1.0     Charbonnier on ``refine_bg`` and
                                   ``coarse_bg``: inside a 5 px dilated GT
                                   mask plus ``0.1x`` unmasked anchor.
perceptual                 0.1     Late loss on ``refine_bg`` only.
FFT                        0.05    multi-scale magnitude + real/imag L1 on
                                   ``refine_bg`` only, always FP32.
mask                       1.0     Focal BCEWithLogits + SoftDice + Sobel
                                   edge-weighted BCE on pre-sigmoid logits.
alpha                      0.5     L1 + gradient L1, linear samples only.
=========================  ======  ==========================================

Perceptual schedule
~~~~~~~~~~~~~~~~~~~

The perceptual term is **off** at the start of Stage 2 and turns on after
``perceptual_start_ratio`` (default ``0.35``, inside the spec's 30-40%
window). Call :meth:`Stage2Loss.set_progress` once per optimizer step::

    criterion.set_progress(step, total_steps)   # ratio is clamped to [0, 1]
    losses = criterion(...)

Before the threshold the component is exactly zero (and the perceptual module
is never called); Charbonnier, FFT and mask terms carry the early training.
This keeps the expensive LPIPS/DISTS forward out of the layout/colour phase
(spec §5.4). The default perceptual backend is a lazily constructed LPIPS
adapter, so importing this module and running tests never needs a network
download. Inject any callable ``fn(pred, target) -> Tensor`` (receiving images
in ``[0, 1]``) for deterministic tests or a different backend.

All returned values are 0-dimensional tensors on the prediction device. The
dict is::

    total, reconstruction, charbonnier_refine, charbonnier_coarse,
    perceptual, fft, mask, mask_focal, mask_dice, mask_sobel,
    alpha, alpha_l1, alpha_grad

Numerical notes
---------------

* The FFT term is computed under ``torch.autocast(..., enabled=False)`` with
  both operands cast to FP32; cuFFT does not support BF16 (spec §4.4/§5.4).
* Mask losses take **logits**. Never pass post-sigmoid probabilities to the
  focal or Sobel BCE terms.
* ``mask_list`` is ordered finest to coarsest. When the full list is passed,
  levels are weighted geometrically by ``mask_scale_weights`` (default
  ``1.0, 0.5, 0.25``); a single tensor is also accepted.
* There is no BatchNorm/InstanceNorm and no square-only reshape; any positive
  ``H``/``W`` works, including odd non-square sizes.
* ``confidence`` (optional, ``[N, 1, H, W]``) scales the reconstruction terms
  for offline pseudo-GT supervision (spec §5.1). It must never be used for
  mask supervision; mask losses ignore it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "BLEND_MODES",
    "SyntheticComposite",
    "build_synthetic_composite",
    "LPIPSAdapter",
    "Stage2Loss",
]

LINEAR = "linear"
SCREEN = "screen"
MULTIPLY = "multiply"

#: Supported synthetic blend families, in documentation order.
BLEND_MODES = (LINEAR, SCREEN, MULTIPLY)

PerceptualFn = Callable[[Tensor, Tensor], Tensor]


# ---------------------------------------------------------------------------
# Synthetic composites
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticComposite:
    """A synthetic watermarked batch and its alpha supervision.

    Attributes:
        image: ``[N, C, H, W]`` composite.
        alpha: ``[N, 1, H, W]`` true linear alpha, or ``None`` when the batch
            contains no linear sample. Nonlinear rows in a mixed batch are
            zero and are marked invalid in ``alpha_valid``.
        alpha_valid: Boolean ``[N]`` mask; ``True`` only for linear samples.
        modes: Per-sample blend mode names, length ``N``.
    """

    image: Tensor
    alpha: Tensor | None
    alpha_valid: Tensor
    modes: tuple[str, ...]


def _normalise_modes(modes: str | Sequence[str], batch: int) -> tuple[str, ...]:
    if isinstance(modes, str):
        resolved = (modes,) * batch
    else:
        resolved = tuple(modes)
        if len(resolved) != batch:
            raise ValueError(
                "modes must be a single string or one entry per sample; "
                f"got {len(resolved)} entries for batch {batch}"
            )
    for mode in resolved:
        if mode not in BLEND_MODES:
            raise ValueError(
                f"unknown blend mode {mode!r}; expected one of "
                f"{BLEND_MODES}"
            )
    return resolved


def _prepare_alpha(
    alpha: Tensor | float | Sequence[float] | None,
    background: Tensor,
    alpha_range: tuple[float, float],
    generator: torch.Generator | None,
) -> Tensor:
    """Broadcast ``alpha`` to ``[N, 1, H, W]`` on the background device."""
    batch, _, height, width = background.shape

    if alpha is None:
        low, high = float(alpha_range[0]), float(alpha_range[1])
        if low > high:
            raise ValueError(
                f"alpha_range must satisfy low <= high, got {alpha_range}"
            )
        if generator is not None:
            alpha = torch.rand(
                (batch, 1, 1, 1),
                generator=generator,
                device=generator.device,
                dtype=background.dtype,
            ).to(background.device)
        else:
            alpha = torch.rand(
                (batch, 1, 1, 1),
                device=background.device,
                dtype=background.dtype,
            )
        alpha = alpha * (high - low) + low
    elif isinstance(alpha, Tensor):
        alpha = alpha.to(
            device=background.device, dtype=background.dtype
        )
    elif isinstance(alpha, (int, float)):
        alpha = torch.full(
            (batch, 1, 1, 1),
            float(alpha),
            device=background.device,
            dtype=background.dtype,
        )
    elif isinstance(alpha, Sequence):
        alpha = torch.as_tensor(
            list(alpha),
            device=background.device,
            dtype=background.dtype,
        )
    else:
        raise TypeError(
            "alpha must be None, a scalar, a sequence or a tensor, "
            f"got {type(alpha).__name__}"
        )

    if alpha.dim() == 0:
        alpha = alpha.view(1, 1, 1, 1)
    elif alpha.dim() == 1:
        alpha = alpha.view(-1, 1, 1, 1)
    elif alpha.dim() == 2:
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
    elif alpha.dim() == 3:
        alpha = alpha.unsqueeze(1)
    if alpha.dim() != 4:
        raise ValueError(
            "alpha must be broadcastable to [N, 1, H, W], got shape "
            f"{tuple(alpha.shape)}"
        )

    try:
        return torch.broadcast_to(alpha, (batch, 1, height, width))
    except RuntimeError as exc:
        raise ValueError(
            "alpha is not broadcastable to "
            f"[{batch}, 1, {height}, {width}]: got "
            f"{tuple(alpha.shape)}"
        ) from exc


def build_synthetic_composite(
    background: Tensor,
    watermark: Tensor,
    modes: str | Sequence[str] = LINEAR,
    alpha: Tensor | float | Sequence[float] | None = None,
    *,
    alpha_range: tuple[float, float] = (0.05, 0.95),
    generator: torch.Generator | None = None,
) -> SyntheticComposite:
    """Blend a clean background with a watermark layer.

    Args:
        background: Clean image ``[N, C, H, W]`` in ``[0, 1]``.
        watermark: Watermark layer, same shape as ``background``. It is the
            full overlay colour, not a pre-multiplied logo; use ``alpha`` (or
            a zero-valued watermark region) to control coverage.
        modes: One of ``"linear"``, ``"screen"``, ``"multiply"`` for the whole
            batch, or a length-``N`` sequence for per-sample modes.
        alpha: Linear-blend alpha. ``None`` samples per sample from
            ``alpha_range``; otherwise a scalar, sequence or tensor shaped
            ``[N]``, ``[N, 1]``, ``[N, 1, 1]`` or ``[N, 1, H, W]``. Ignored by
            nonlinear samples (their returned alpha rows are zero).
        alpha_range: Inclusive-ish random range used when ``alpha`` is
            ``None``. The default avoids fully invisible or fully opaque
            overlays.
        generator: Optional ``torch.Generator`` for reproducible random alpha.

    Returns:
        A :class:`SyntheticComposite`. ``alpha`` is ``None`` if no sample is
        linear; otherwise it is ``[N, 1, H, W]`` with zero rows for nonlinear
        samples and ``alpha_valid`` marks the usable rows.

    Raises:
        ValueError: On shape, mode or alpha broadcast errors.
        TypeError: If inputs are not floating point tensors.
    """
    if not isinstance(background, Tensor) or not isinstance(
        watermark, Tensor
    ):
        raise TypeError("background and watermark must be torch tensors")
    if background.dim() != 4:
        raise ValueError(
            "background must be 4D [N, C, H, W], got shape "
            f"{tuple(background.shape)}"
        )
    if watermark.shape != background.shape:
        raise ValueError(
            "watermark and background must have identical shapes, got "
            f"{tuple(watermark.shape)} and {tuple(background.shape)}"
        )
    if background.shape[0] <= 0:
        raise ValueError("synthetic composites require a non-empty batch")
    if not background.is_floating_point() or not watermark.is_floating_point():
        raise TypeError("background and watermark must be floating point")

    batch = int(background.shape[0])
    resolved = _normalise_modes(modes, batch)
    alpha_map = _prepare_alpha(
        alpha, background, alpha_range, generator
    )

    linear_image = (1.0 - alpha_map) * background + alpha_map * watermark
    screen_image = 1.0 - (1.0 - background) * (1.0 - watermark)
    multiply_image = background * watermark

    is_linear = torch.tensor(
        [mode == LINEAR for mode in resolved],
        dtype=torch.bool,
        device=background.device,
    ).view(batch, 1, 1, 1)
    is_screen = torch.tensor(
        [mode == SCREEN for mode in resolved],
        dtype=torch.bool,
        device=background.device,
    ).view(batch, 1, 1, 1)

    image = torch.where(
        is_linear,
        linear_image,
        torch.where(is_screen, screen_image, multiply_image),
    )

    has_linear = bool(is_linear.any())
    if has_linear:
        # Nonlinear rows are zeroed so they can never leak as alpha labels.
        alpha_out: Tensor | None = alpha_map * is_linear.to(
            dtype=alpha_map.dtype
        )
    else:
        alpha_out = None

    return SyntheticComposite(
        image=image,
        alpha=alpha_out,
        alpha_valid=is_linear.reshape(batch),
        modes=resolved,
    )


# ---------------------------------------------------------------------------
# Perceptual backend
# ---------------------------------------------------------------------------


class LPIPSAdapter(nn.Module):
    """Lazy LPIPS adapter expecting images in ``[0, 1]``.

    Importing :mod:`lpips` and downloading its weights happens only when this
    class is instantiated, which :class:`Stage2Loss` defers until the
    perceptual schedule first activates. CI never instantiates it; tests
    inject a deterministic callable instead.
    """

    def __init__(self, net: str = "alex", spatial: bool = False) -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The default perceptual backend needs the optional 'lpips' "
                "package. Install it or inject a callable via "
                "Stage2Loss(perceptual=...)."
            ) from exc

        self.model = lpips.LPIPS(net=net, spatial=spatial, verbose=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # LPIPS expects [-1, 1]; the training contract uses [0, 1].
        return self.model(
            pred * 2.0 - 1.0, target * 2.0 - 1.0
        ).mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _charbonnier(x: Tensor, y: Tensor, eps: float) -> Tensor:
    """``sqrt((x - y)^2 + eps^2)``, the spec's Charbonnier form."""
    return torch.sqrt((x - y).square() + eps * eps)


def _resize_to(x: Tensor, size: tuple[int, int]) -> Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(
        x, size=tuple(size), mode="bilinear", align_corners=False
    )


def _expand_alpha(alpha: Tensor, reference: Tensor) -> Tensor:
    """Broadcast an alpha target to the prediction's ``[N, 1, H, W]``."""
    if alpha.dim() == 0:
        alpha = alpha.view(1, 1, 1, 1)
    elif alpha.dim() == 1:
        alpha = alpha.view(-1, 1, 1, 1)
    elif alpha.dim() == 2:
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
    elif alpha.dim() == 3:
        alpha = alpha.unsqueeze(1)
    if alpha.dim() != 4:
        raise ValueError(
            "alpha target must be broadcastable to [N, 1, H, W], got "
            f"shape {tuple(alpha.shape)}"
        )
    try:
        return alpha.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError(
            "alpha target is not broadcastable to "
            f"{tuple(reference.shape)}: got {tuple(alpha.shape)}"
        ) from exc


def _normalise_mask(mask: Tensor, batch: int, name: str) -> Tensor:
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(
            f"{name} must be [N, 1, H, W], got shape {tuple(mask.shape)}"
        )
    if mask.shape[1] != 1:
        raise ValueError(
            f"{name} must have one channel, got {mask.shape[1]}"
        )
    if mask.shape[0] != batch:
        raise ValueError(
            f"{name} batch mismatch: {mask.shape[0]} vs {batch}"
        )
    return mask.float()


# ---------------------------------------------------------------------------
# Stage 2 criterion
# ---------------------------------------------------------------------------


class Stage2Loss(nn.Module):
    """Stage 2 criterion for SLBR-Next (spec §5.2).

    See the module docstring for the blend families, alpha semantics, the
    component list and the perceptual schedule. All images are expected in
    ``[0, 1]``; mask targets are probabilities and mask predictions are
    pre-sigmoid logits.

    Args:
        perceptual: ``"auto"`` (default) for a lazily built
            :class:`LPIPSAdapter`; ``None`` to disable the perceptual term
            permanently; or a callable/module ``fn(pred, target) -> Tensor``.
            Injected callables receive FP32 images in ``[0, 1]`` and are
            responsible for their own device placement.
        perceptual_start_ratio: Fraction of Stage 2 after which the perceptual
            term activates. Defaults to ``0.35``.
        perceptual_net: LPIPS backbone used by the lazy default adapter.
        reconstruction_weight: Weight of the Charbonnier term (default 1.0).
        perceptual_weight: Weight of the perceptual term (default 0.1).
        fft_weight: Weight of the spectral term (default 0.05).
        mask_weight: Weight of the mask term (default 1.0).
        alpha_weight: Weight of the alpha term (default 0.5).
        dilation_radius: GT-mask dilation radius in pixels (default 5, i.e. an
            11x11 max-pool as in spec §4.4).
        unmasked_weight: Outside-mask Charbonnier anchor multiplier (0.1).
        charbonnier_eps: Charbonnier epsilon (default 1e-3, so ``eps^2`` is
            the spec's ``1e-6``).
        fft_scales: Relative resolutions for the multi-scale FFT term.
        focal_gamma: Focal BCE focusing parameter.
        focal_alpha: Focal BCE positive-class balancing factor.
        dice_eps: SoftDice smoothing constant.
        sobel_weight: Strength of the Sobel edge weight (0 disables it).
        mask_scale_weights: Geometric weights for ``mask_list`` levels.
        alpha_grad_weight: Weight of the alpha gradient term inside the alpha
            component.
    """

    def __init__(
        self,
        *,
        perceptual: PerceptualFn | nn.Module | None | str = "auto",
        perceptual_start_ratio: float = 0.35,
        perceptual_net: str = "alex",
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        fft_weight: float = 0.05,
        mask_weight: float = 1.0,
        alpha_weight: float = 0.5,
        dilation_radius: int = 5,
        unmasked_weight: float = 0.1,
        charbonnier_eps: float = 1e-3,
        fft_scales: tuple[float, ...] = (1.0, 0.5),
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        dice_eps: float = 1.0,
        sobel_weight: float = 1.0,
        mask_scale_weights: tuple[float, ...] = (1.0, 0.5, 0.25),
        alpha_grad_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if isinstance(perceptual, str):
            if perceptual != "auto":
                raise ValueError(
                    "perceptual must be 'auto', None, or a callable; got "
                    f"{perceptual!r}"
                )
            self.perceptual: PerceptualFn | nn.Module | None = None
            self._perceptual_auto = True
        else:
            if perceptual is not None and not callable(perceptual):
                raise TypeError(
                    "perceptual must be callable, an nn.Module, None or "
                    f"'auto'; got {type(perceptual).__name__}"
                )
            self.perceptual = perceptual
            self._perceptual_auto = False
        self._perceptual_disabled = perceptual is None

        if not 0.0 <= float(perceptual_start_ratio) <= 1.0:
            raise ValueError(
                "perceptual_start_ratio must lie in [0, 1], got "
                f"{perceptual_start_ratio}"
            )
        self.perceptual_start_ratio = float(perceptual_start_ratio)
        self.perceptual_net = perceptual_net
        self._progress = 0.0

        for name, value in (
            ("reconstruction_weight", reconstruction_weight),
            ("perceptual_weight", perceptual_weight),
            ("fft_weight", fft_weight),
            ("mask_weight", mask_weight),
            ("alpha_weight", alpha_weight),
            ("unmasked_weight", unmasked_weight),
            ("focal_gamma", focal_gamma),
            ("dice_eps", dice_eps),
            ("sobel_weight", sobel_weight),
            ("alpha_grad_weight", alpha_grad_weight),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        if int(dilation_radius) < 0:
            raise ValueError(
                f"dilation_radius must be non-negative, got {dilation_radius}"
            )
        if float(charbonnier_eps) <= 0.0:
            raise ValueError(
                f"charbonnier_eps must be positive, got {charbonnier_eps}"
            )
        if not 0.0 <= float(focal_alpha) <= 1.0:
            raise ValueError(
                f"focal_alpha must lie in [0, 1], got {focal_alpha}"
            )
        if not fft_scales or any(float(scale) <= 0.0 for scale in fft_scales):
            raise ValueError(
                f"fft_scales must be positive, got {fft_scales}"
            )
        if not mask_scale_weights or any(
            float(weight) <= 0.0 for weight in mask_scale_weights
        ):
            raise ValueError(
                "mask_scale_weights must be positive, got "
                f"{mask_scale_weights}"
            )

        self.reconstruction_weight = float(reconstruction_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.fft_weight = float(fft_weight)
        self.mask_weight = float(mask_weight)
        self.alpha_weight = float(alpha_weight)
        self.dilation_radius = int(dilation_radius)
        self.unmasked_weight = float(unmasked_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        self.fft_scales = tuple(float(scale) for scale in fft_scales)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.dice_eps = float(dice_eps)
        self.sobel_weight = float(sobel_weight)
        self.mask_scale_weights = tuple(
            float(weight) for weight in mask_scale_weights
        )
        self.alpha_grad_weight = float(alpha_grad_weight)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("_sobel_x", sobel_x, persistent=False)
        self.register_buffer("_sobel_y", sobel_y, persistent=False)

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def set_progress(self, step: float, total: float) -> None:
        """Record Stage 2 progress for the late perceptual schedule.

        Args:
            step: Completed optimizer steps (or any monotone counter).
            total: Total planned steps; must be positive.

        The stored ratio is ``step / total`` clamped to ``[0, 1]``. The
        perceptual term is active when the ratio reaches
        ``perceptual_start_ratio``.
        """
        total = float(total)
        if total <= 0.0:
            raise ValueError(f"total must be positive, got {total}")
        step = min(max(float(step), 0.0), total)
        self._progress = step / total

    @property
    def progress(self) -> float:
        """Current Stage 2 fraction in ``[0, 1]``."""
        return self._progress

    @property
    def perceptual_active(self) -> bool:
        """Whether the perceptual term is active at the current progress."""
        if self._perceptual_disabled:
            return False
        return self._progress >= self.perceptual_start_ratio

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        refine_bg: Tensor,
        coarse_bg: Tensor,
        mask_logits: Tensor | Sequence[Tensor],
        clean: Tensor,
        mask: Tensor,
        *,
        alpha_pred: Tensor | None = None,
        alpha_gt: Tensor | None = None,
        alpha_valid: Tensor | Sequence[bool] | None = None,
        confidence: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute every Stage 2 loss component.

        Args:
            refine_bg: Refined background ``[N, 3, H, W]`` from the student.
            coarse_bg: Coarse background ``[N, 3, H, W]``.
            mask_logits: Final pre-sigmoid mask logits ``[N, 1, H, W]``, or
                the full ``mask_list`` (finest to coarsest).
            clean: Clean background target ``[N, 3, H, W]``.
            mask: GT mask probabilities ``[N, 1, H, W]``. Resized with
                bilinear interpolation when a mask level has another size.
            alpha_pred: Optional alpha prediction logits or probabilities
                ``[N, 1, H, W]``. Only used where ``alpha_valid`` is true.
            alpha_gt: Optional true linear alpha ``[N, 1, H, W]`` (or a
                broadcastable per-sample form).
            alpha_valid: Boolean ``[N]`` mask marking true linear-alpha
                samples. Required whenever ``alpha_gt`` is supplied.
            confidence: Optional pseudo-GT confidence ``[N, 1, H, W]`` for
                reconstruction only (never for mask supervision).

        Returns:
            A dict of 0-dimensional tensors documented in the module
            docstring, including ``"total"``.
        """
        if not isinstance(refine_bg, Tensor) or not isinstance(
            coarse_bg, Tensor
        ):
            raise TypeError("refine_bg and coarse_bg must be torch tensors")
        if refine_bg.shape != clean.shape:
            raise ValueError(
                "refine_bg and clean must have identical shapes, got "
                f"{tuple(refine_bg.shape)} and {tuple(clean.shape)}"
            )
        if coarse_bg.shape != clean.shape:
            raise ValueError(
                "coarse_bg and clean must have identical shapes, got "
                f"{tuple(coarse_bg.shape)} and {tuple(clean.shape)}"
            )
        if refine_bg.dim() != 4:
            raise ValueError(
                "refine_bg must be [N, 3, H, W], got shape "
                f"{tuple(refine_bg.shape)}"
            )

        batch = int(refine_bg.shape[0])
        refine = refine_bg.float()
        coarse = coarse_bg.float()
        target = clean.detach().float()
        mask_target = _normalise_mask(mask, batch, "mask")
        if confidence is not None:
            confidence = _normalise_mask(
                confidence, batch, "confidence"
            )

        char_refine = self._charbonnier_masked(
            refine, target, mask_target, confidence
        )
        char_coarse = self._charbonnier_masked(
            coarse, target, mask_target, confidence
        )
        reconstruction = 0.5 * (char_refine + char_coarse)

        zero_image = refine.sum() * 0.0
        if self.perceptual_active:
            perceptual = self._perceptual_loss(refine, target)
        else:
            perceptual = zero_image

        fft = self._fft_loss(refine, target)
        mask_focal, mask_dice, mask_sobel = self._mask_loss(
            mask_logits, mask_target
        )
        mask_total = mask_focal + mask_dice + mask_sobel

        alpha_l1, alpha_grad = self._alpha_loss(
            alpha_pred, alpha_gt, alpha_valid
        )
        alpha_total = alpha_l1 + self.alpha_grad_weight * alpha_grad

        total = (
            self.reconstruction_weight * reconstruction
            + self.perceptual_weight * perceptual
            + self.fft_weight * fft
            + self.mask_weight * mask_total
            + self.alpha_weight * alpha_total
        )

        return {
            "total": total,
            "reconstruction": reconstruction,
            "charbonnier_refine": char_refine,
            "charbonnier_coarse": char_coarse,
            "perceptual": perceptual,
            "fft": fft,
            "mask": mask_total,
            "mask_focal": mask_focal,
            "mask_dice": mask_dice,
            "mask_sobel": mask_sobel,
            "alpha": alpha_total,
            "alpha_l1": alpha_l1,
            "alpha_grad": alpha_grad,
        }

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _charbonnier_masked(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        confidence: Tensor | None,
    ) -> Tensor:
        mask = _resize_to(
            mask.to(device=prediction.device), prediction.shape[-2:]
        )
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

        error = _charbonnier(prediction, target, self.charbonnier_eps)
        inside_weight = dilated
        outside_weight = 1.0 - dilated
        if confidence is not None:
            confidence = _resize_to(
                confidence.to(device=prediction.device),
                prediction.shape[-2:],
            ).clamp(0.0, 1.0)
            inside_weight = inside_weight * confidence
            outside_weight = outside_weight * confidence

        inside = self._weighted_mean(error, inside_weight)
        outside = self._weighted_mean(error, outside_weight)
        return (inside + self.unmasked_weight * outside).mean()

    @staticmethod
    def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
        # ``weights`` is [N, 1, H, W] while ``values`` is [N, C, H, W]; expand
        # explicitly so the pixel weight is shared across colour channels.
        weights = weights.expand_as(values)
        values = values.flatten(1)
        weights = weights.flatten(1)
        numerator = (values * weights).sum(dim=1)
        denominator = weights.sum(dim=1).clamp_min(1e-8)
        return numerator / denominator

    # ------------------------------------------------------------------
    # Perceptual
    # ------------------------------------------------------------------

    def _ensure_perceptual(
        self, device: torch.device
    ) -> PerceptualFn | nn.Module | None:
        if self.perceptual is not None:
            return self.perceptual
        if not self._perceptual_auto:
            return None
        module = LPIPSAdapter(net=self.perceptual_net).to(device)
        self.perceptual = module
        return module

    def _perceptual_loss(
        self, prediction: Tensor, target: Tensor
    ) -> Tensor:
        module = self._ensure_perceptual(prediction.device)
        if module is None:
            return prediction.sum() * 0.0
        value = module(prediction.float(), target.detach().float())
        if not isinstance(value, Tensor):
            value = torch.as_tensor(
                value, device=prediction.device, dtype=torch.float32
            )
        return value.mean()

    # ------------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------------

    def _fft_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Multi-scale FFT L1, forced to FP32 under any autocast context."""
        with torch.autocast(
            device_type=prediction.device.type, enabled=False
        ):
            prediction = prediction.float()
            target = target.detach().float()
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

            if weight_sum > 0.0:
                total = total / weight_sum
            return total

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------

    def _mask_loss(
        self,
        mask_logits: Tensor | Sequence[Tensor],
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(mask_logits, Tensor):
            levels = [mask_logits]
        else:
            levels = list(mask_logits)
        if not levels:
            raise ValueError("mask_logits must contain at least one level")
        if len(levels) > len(self.mask_scale_weights):
            raise ValueError(
                f"got {len(levels)} mask levels but only "
                f"{len(self.mask_scale_weights)} mask_scale_weights"
            )

        if not isinstance(levels[0], Tensor):
            raise TypeError("every mask level must be a tensor")

        weights = self.mask_scale_weights[: len(levels)]
        weight_sum = float(sum(weights))
        reference = levels[0]
        focal_total = torch.zeros(
            (), device=reference.device, dtype=torch.float32
        )
        dice_total = torch.zeros(
            (), device=reference.device, dtype=torch.float32
        )
        sobel_total = torch.zeros(
            (), device=reference.device, dtype=torch.float32
        )
        batch = int(mask.shape[0])

        for logits, weight in zip(levels, weights):
            if not isinstance(logits, Tensor):
                raise TypeError("every mask level must be a tensor")
            if logits.dim() != 4 or logits.shape[1] != 1:
                raise ValueError(
                    "mask logits must be [N, 1, H, W], got shape "
                    f"{tuple(logits.shape)}"
                )
            if logits.shape[0] != batch:
                raise ValueError(
                    f"mask logit batch mismatch: {logits.shape[0]} vs "
                    f"{batch}"
                )
            level_target = _resize_to(
                mask, logits.shape[-2:]
            ).to(device=logits.device, dtype=torch.float32)
            focal, dice, sobel = self._mask_level(
                logits.float(), level_target
            )
            share = weight / weight_sum
            focal_total = focal_total + share * focal
            dice_total = dice_total + share * dice
            sobel_total = sobel_total + share * sobel

        return focal_total, dice_total, sobel_total

    def _mask_level(
        self, logits: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        target = target.clamp(0.0, 1.0)

        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        pt = torch.exp(-bce)
        alpha_t = (
            target * self.focal_alpha
            + (1.0 - target) * (1.0 - self.focal_alpha)
        )
        focal = (
            alpha_t * (1.0 - pt).pow(self.focal_gamma) * bce
        ).mean()

        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.dim()))
        intersection = (probabilities * target).sum(dim=dims)
        denominator = (
            probabilities.sum(dim=dims) + target.sum(dim=dims)
        )
        dice = 1.0 - (
            (2.0 * intersection + self.dice_eps)
            / (denominator + self.dice_eps)
        ).mean()

        edges = self._sobel_magnitude(target)
        edge_weights = 1.0 + self.sobel_weight * edges
        sobel = (bce * edge_weights).sum() / edge_weights.sum().clamp_min(
            1e-8
        )
        return focal, dice, sobel

    def _sobel_magnitude(self, mask: Tensor) -> Tensor:
        kernel_x = self._sobel_x.to(
            device=mask.device, dtype=mask.dtype
        )
        kernel_y = self._sobel_y.to(
            device=mask.device, dtype=mask.dtype
        )
        grad_x = F.conv2d(mask, kernel_x, padding=1)
        grad_y = F.conv2d(mask, kernel_y, padding=1)
        magnitude = torch.sqrt(
            grad_x.square() + grad_y.square() + 1e-12
        )
        per_sample = magnitude.flatten(1).amax(dim=1).clamp_min(1e-6)
        per_sample = per_sample.view(-1, 1, 1, 1)
        return (magnitude / per_sample).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Alpha (linear blends only)
    # ------------------------------------------------------------------

    def _alpha_loss(
        self,
        alpha_pred: Tensor | None,
        alpha_gt: Tensor | None,
        alpha_valid: Tensor | Sequence[bool] | None,
    ) -> tuple[Tensor, Tensor]:
        if alpha_pred is None:
            zero = torch.zeros(
                (), dtype=torch.float32, device=self._sobel_x.device
            )
            return zero, zero
        if alpha_pred.dim() == 3:
            alpha_pred = alpha_pred.unsqueeze(1)
        if alpha_pred.dim() != 4 or alpha_pred.shape[1] != 1:
            raise ValueError(
                "alpha_pred must be [N, 1, H, W], got shape "
                f"{tuple(alpha_pred.shape)}"
            )

        zero = alpha_pred.float().sum() * 0.0
        if alpha_gt is None:
            return zero, zero
        if alpha_valid is None:
            raise ValueError(
                "alpha_valid is required with alpha_gt: alpha must only be "
                "supervised on true linear blends (see "
                "build_synthetic_composite)"
            )

        alpha_pred = alpha_pred.float()
        alpha_gt = _expand_alpha(alpha_gt, alpha_pred).float()
        valid = torch.as_tensor(
            alpha_valid, device=alpha_pred.device, dtype=torch.bool
        ).reshape(-1)
        if valid.numel() != alpha_pred.shape[0]:
            raise ValueError(
                "alpha_valid must have one entry per sample, got "
                f"{valid.numel()} for batch {alpha_pred.shape[0]}"
            )
        if not bool(valid.any()):
            return zero, zero

        prediction = alpha_pred[valid]
        target = alpha_gt[valid]

        l1 = (prediction - target).abs().mean()
        gradient_terms: list[Tensor] = []
        if prediction.shape[-2] > 1:
            gradient_terms.append(
                (
                    (prediction[..., 1:, :] - prediction[..., :-1, :])
                    - (target[..., 1:, :] - target[..., :-1, :])
                )
                .abs()
                .mean()
            )
        if prediction.shape[-1] > 1:
            gradient_terms.append(
                (
                    (prediction[..., :, 1:] - prediction[..., :, :-1])
                    - (target[..., :, 1:] - target[..., :, :-1])
                )
                .abs()
                .mean()
            )
        if gradient_terms:
            gradient = torch.stack(gradient_terms).mean()
        else:
            gradient = prediction.sum() * 0.0

        return l1, gradient
