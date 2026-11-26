"""SLBR-Next training entry point (issue #12).

This script is intentionally separate from the legacy 2021 ``train.py``; the
legacy SLBR entry point and its options are untouched.

The trainer is data-agnostic. For real runs, pass factories that import
lazily so T2 (Stage 1 distillation) and T3 (Stage 2 fine-tuning) can supply
their own data pipeline, loss and model without circular imports::

    --data-factory  module:function  -> function(args) returns
                                          (train_loader, val_loader | None)
    --loss-factory  module:function  -> function(args) returns
                                          loss(outputs, batch) -> Tensor | dict
    --model-factory module:function  -> function(args) returns nn.Module
                                          (default: SLBRNextV1)

The data factory is responsible for the batch format. Mapping batches must
contain ``config.input_key`` (default ``"image"``); tuple/list batches use the
first element. With ``torchrun``, build the loaders with
``DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)``;
the trainer does not add a sampler. ``args.subset_size`` is an optional cap
the factory may use for short A/B runs.

Two-T4 Kaggle smoke (synthetic data, no dataset or pretrained download)
----------------------------------------------------------------------
Eager first session (spec §5.4 recommends this before compiling)::

    cd /kaggle/working/SLBR-Visible-Watermark-Removal
    torchrun --nproc_per_node=2 train_slbr_next.py --smoke --steps 10 \
        --batch-size 6 --size 256 --no-compile \
        --output-dir /kaggle/working/slbr_next_smoke_eager

Compiled smoke (the exact T6 acceptance command)::

    cd /kaggle/working/SLBR-Visible-Watermark-Removal
    torchrun --nproc_per_node=2 train_slbr_next.py --smoke --steps 10 \
        --batch-size 6 --size 256 --compile \
        --output-dir /kaggle/working/slbr_next_smoke

The smoke runs the real ``SLBRNextV1`` at 256x256, batch 6/GPU, exercises
DDP, FP16 autocast + GradScaler (T4), fused AdamW, channels_last, EMA,
``torch.compile`` fallback, and an atomic checkpoint save/resume. It prints
``SMOKE_TIMING {...}`` (setup, model build, data, trainer init, first step,
lazy compile, training, checkpoint/resume) followed by one ``SMOKE_OK {...}``
JSON line with throughput and peak VRAM. On the compile path ``first_step_s``
includes lazy Inductor compilation and ``compile_s`` isolates the first
compiled forward; there is no compile timeout, and the existing
catch-and-fallback remains active. ``steady_state_steps_per_s`` excludes the
first step. The §5.4 ~15% wall-clock fallback is a long-run operator decision:
compare that steady-state number against ``--no-compile`` over a longer run;
short smokes are compile-dominated by design. The smoke also enables per-step
finite checks: a skipped AMP overflow is logged with ``grad_norm: null`` and
``step_skipped: true``, then the run fails with ``SMOKE_FAILED`` rather than
printing ``SMOKE_OK``. A clean ``SMOKE_OK`` therefore means parameters stayed
finite and no unscaled Inf/NaN gradient was observed. After it passes, try
``--batch-size 8`` and ``--batch-size 10`` to find the OOM edge.

Real training::

    torchrun --nproc_per_node=2 train_slbr_next.py \
        --data-factory src.data.slbr_next:build_loaders \
        --loss-factory src.losses.slbr_next:build_loss \
        --epochs 30 --output-dir /kaggle/working/slbr_next

One-stage schedule (spec §5.5)
------------------------------
``--schedule one_stage`` selects a supervised run on paired GT plus synthetic
composites with a teacher mask + H/8 regularizer that decays to zero over
``--teacher-decay-epochs`` (default 7). The reference loss factory is::

    --loss-factory src.models.slbr_next.losses_stage1:build_schedule_loss \
    --schedule one_stage --teacher-decay-epochs 7 --no-compile

Add ``--perceptual none`` to a short A/B to skip the optional LPIPS backend
and isolate the schedule comparison; the default ``auto`` enables it after the
Stage 2-style progress threshold.

The data factory must emit ``image``, ``clean`` and ``mask`` (plus optional
``alpha_pred``/``alpha_gt``/``alpha_valid``/``confidence``) and, while the
teacher is active, a ``teacher_target`` TeacherTarget payload. ``two_stage``
uses the existing Stage 1 distillation criterion through the same factory.
Stage 1/one-stage criteria expose ``forward_fn``; the CLI passes it to the
trainer, which builds the mask logits and H/8 feature tap with
``run_student_stage1``. Run this schedule eagerly (``--no-compile``) because
the feature hooks can graph-break ``torch.compile``.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from src.models.slbr_next.trainer import (
    Trainer,
    TrainerConfig,
    init_distributed,
    seed_everything,
    smoke_train,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (``None`` defaults mean "keep config value")."""
    parser = argparse.ArgumentParser(
        description="Train SLBR-Next (SLBR-Next CI / issue #12)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Load TrainerConfig from a JSON file before applying overrides.",
    )
    parser.add_argument("--output-dir", help="Run directory for config/log/checkpoints.")
    parser.add_argument("--resume", type=Path, help="Resume from this checkpoint.")
    parser.add_argument("--device", help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--local-rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Images per GPU (default 6; test 8/10 separately).")
    parser.add_argument("--schedule", choices=("two_stage", "one_stage"),
                        default=None,
                        help="Two-stage recipe (default) or one-stage "
                             "supervised + decaying teacher auxiliary.")
    parser.add_argument("--teacher-decay-epochs", type=int, default=None,
                        help="One-stage teacher auxiliary decay window "
                             "(default 7; spec 5-10).")
    parser.add_argument("--perceptual", choices=("auto", "none"), default=None,
                        help="One-stage supervised perceptual term. 'none' "
                             "disables the optional LPIPS backend.")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Optional paired-sample cap read by the data "
                             "factory for short A/B runs.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--betas", type=float, nargs=2, default=None,
                        metavar=("B1", "B2"))
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--amp", choices=("auto", "fp16", "bf16", "fp32"),
                        default=None)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--compile-mode",
                        choices=("default", "reduce-overhead", "max-autotune"),
                        default=None)
    parser.add_argument("--ddp", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--static-graph", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="DDP static_graph (all parameters must get grads).")
    parser.add_argument("--find-unused-parameters",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-as-bucket-view",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--val-every-epochs", type=int, default=None)
    parser.add_argument("--validate-ema", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--save-every-epoch",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-on-exit", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--best-checkpoint-name", default=None)
    parser.add_argument("--input-key", default=None)

    parser.add_argument("--smoke", action="store_true",
                        help="Run the synthetic 2xT4 smoke and exit.")
    parser.add_argument("--steps", type=int, default=10,
                        help="Optimizer steps per rank for --smoke.")
    parser.add_argument("--size", type=int, default=256,
                        help="Square synthetic image size for --smoke.")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers for --smoke.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download ConvNeXt V2 FCMAE weights.")

    parser.add_argument("--data-factory", help="module:function returning loaders.")
    parser.add_argument("--loss-factory", help="module:function returning a loss.")
    parser.add_argument("--model-factory", help="module:function returning a model.")
    return parser


def _load_factory(spec: str):
    """Import ``"module:attribute"`` (attribute may be dotted)."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            f"factory must be 'module:function', got {spec!r}"
        )
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"factory {spec!r} is not callable")
    return target


def _apply_config_overrides(
    config: TrainerConfig, args: argparse.Namespace
) -> TrainerConfig:
    mapping = {
        "epochs": "epochs",
        "max_steps": "max_steps",
        "batch_size": "batch_size_per_gpu",
        "schedule": "schedule",
        "teacher_decay_epochs": "teacher_decay_epochs",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "warmup_steps": "warmup_steps",
        "min_lr": "min_lr",
        "grad_clip": "grad_clip",
        "ema_decay": "ema_decay",
        "amp": "amp",
        "channels_last": "channels_last",
        "cudnn_benchmark": "cudnn_benchmark",
        "compile": "compile",
        "compile_mode": "compile_mode",
        "ddp": "ddp",
        "static_graph": "ddp_static_graph",
        "find_unused_parameters": "ddp_find_unused_parameters",
        "gradient_as_bucket_view": "ddp_gradient_as_bucket_view",
        "device": "device",
        "seed": "seed",
        "log_every": "log_every",
        "val_every_epochs": "val_every_epochs",
        "validate_ema": "validate_ema",
        "early_stop_patience": "early_stop_patience",
        "early_stop_min_delta": "early_stop_min_delta",
        "save_best": "save_best",
        "save_every_epoch": "save_every_epoch",
        "save_on_exit": "save_on_exit",
        "output_dir": "output_dir",
        "checkpoint_name": "checkpoint_name",
        "best_checkpoint_name": "best_checkpoint_name",
        "input_key": "input_key",
    }
    overrides: dict[str, Any] = {}
    for argument, field_name in mapping.items():
        value = getattr(args, argument, None)
        if value is not None:
            overrides[field_name] = value
    if args.betas is not None:
        overrides["betas"] = tuple(float(value) for value in args.betas)
    return replace(config, **overrides) if overrides else config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config_json is not None:
        config = TrainerConfig.from_json(
            args.config_json.read_text(encoding="utf-8")
        )
    else:
        config = TrainerConfig()
    config = _apply_config_overrides(config, args)

    if args.smoke:
        smoke_train(
            config,
            steps=args.steps,
            size=args.size,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            # The synthetic smoke measures compute/DDP; skip the FCMAE download.
            convnext_pretrained=False,
        )
        return 0

    if not args.data_factory or not args.loss_factory:
        parser.error(
            "--data-factory and --loss-factory are required unless --smoke"
        )

    rank, world_size, local_rank = init_distributed(enabled=config.ddp)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    seed_everything(config.seed, rank)

    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    args.batch_size = config.batch_size_per_gpu
    args.config = config.to_dict()

    if args.model_factory:
        model = _load_factory(args.model_factory)(args)
    else:
        from src.networks.slbr_next import SLBRNextV1

        model = SLBRNextV1(convnext_pretrained=not args.no_pretrained)

    # The one-stage H/8 CWD regularizer needs the student background-trunk
    # width. Read it from the v1 config when available; the default matches
    # the reference SLBRNextV1 (256 channels, same as the SLBR teacher tap).
    model_config = getattr(model, "config", None)
    if model_config is not None:
        student_h8 = getattr(model_config, "bottleneck_bg_width", None)
        if student_h8 is not None:
            args.student_h8_channels = int(student_h8)

    data_factory = _load_factory(args.data_factory)
    loaders = data_factory(args)
    if not isinstance(loaders, (tuple, list)) or len(loaders) != 2:
        raise TypeError(
            "data factory must return (train_loader, val_loader_or_None)"
        )
    train_loader, val_loader = loaders

    loss_factory = _load_factory(args.loss_factory)
    loss = loss_factory(args)
    if not callable(loss):
        raise TypeError("loss factory must return a callable")
    # Stage 1 and one-stage losses consume run_student_stage1 outputs; they
    # expose forward_fn so the trainer can build them without a second API.
    forward_fn = getattr(loss, "forward_fn", None)
    if forward_fn is not None and not callable(forward_fn):
        raise TypeError("loss.forward_fn must be callable or None")

    trainer = Trainer(
        model=model,
        loss=loss,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        forward_fn=forward_fn,
    )
    if args.resume is not None:
        trainer.resume(args.resume)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
