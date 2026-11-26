"""Two-epoch synthetic training gate for SLBR-Next (issue #14, spec §6).

This is the smoke half of the T6 gate. It uses the real ``SLBRNextV1``, T3's
``Stage2Loss`` with ``perceptual=None``, and T4's trainer at 256x256 on
synthetic data. No dataset, teacher checkpoint or network access is required.

The default run does three 2-epoch rows:

* ``stage2_supervised``: the primary smoke (Stage 2 reconstruction + mask +
  FFT losses);
* ``ab_two_stage``: the existing Stage 1 distillation schedule;
* ``ab_one_stage``: T5's supervised + decaying teacher auxiliary.

The last two are a **plumbing/gate A/B only**. They prove that both schedules
run through the same trainer, DDP, AMP and checkpoint path on identical
synthetic data. They are *not* a quality comparison and must not be quoted as
one.

Checkpoint/resume
------------------
Each row trains epoch 1, saves a checkpoint, constructs a fresh trainer/model
and resumes it for epoch 2. The trainer's JSONL log records both trainers and
the resume event. ``SMOKE_2EPOCH_OK`` is printed only after every row has:

* two completed epochs and the expected number of optimizer steps;
* zero FP16 overflow/skipped steps (``finite_checks=True`` and
  ``fail_on_overflow=True``);
* finite parameters, buffers and losses.

One-cell Kaggle command::

    !cd /kaggle/working/SLBR-Visible-Watermark-Removal && \
      python -m torch.distributed.run --nproc_per_node=2 --standalone \
        bench/smoke_2epoch.py \
        --output-dir /kaggle/working/slbr_next_t6_smoke

Output marker: ``SMOKE_2EPOCH_OK {...}`` (and ``SMOKE_2EPOCH_FAILED {...}``
on a failed gate).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.common import (  # noqa: E402  (path bootstrap above)
    Stage2ProbabilityLoss,
    SyntheticStepLoader,
    mean_finite,
)

__all__ = ["main", "run_smoke"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLBR-Next 2-epoch synthetic training gate (issue #14)"
    )
    parser.add_argument(
        "--output-dir", default="/kaggle/working/slbr_next_t6_smoke"
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--amp",
        choices=("fp16",),
        default="fp16",
        help="The gate is defined with FP16 autocast + GradScaler on T4.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model. Off by default: Stage 1 feature hooks can "
        "graph-break torch.compile and compile time dominates a short gate.",
    )
    parser.add_argument(
        "--skip-ab",
        action="store_true",
        help="Run only the primary Stage 2 2-epoch smoke.",
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _curve_summary(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    steps = [record for record in records if record.get("event") == "step"]
    epochs = [record for record in records if record.get("event") == "epoch"]
    return {
        "step_count": len(steps),
        "steps": [
            {
                "epoch": record.get("epoch"),
                "step": record.get("step"),
                "loss": record.get("loss"),
                "overflow": record.get("overflow"),
                "step_skipped": record.get("step_skipped"),
                "components": record.get("components") or {},
            }
            for record in steps
        ],
        "epochs": [
            {
                "epoch": record.get("epoch"),
                "step": record.get("step"),
                "train_loss": record.get("train_loss"),
            }
            for record in epochs
        ],
    }


def _make_target_builder(teacher_target_cls: type) -> Callable[..., dict[str, Any]]:
    """Synthetic TeacherTargets for the A/B plumbing rows.

    Shapes follow the real student taps: H, H/2, H/4 mask logits, a 256-channel
    H/8 bottleneck feature and a 128-channel H/4 background-decoder feature.
    No teacher network is instantiated or downloaded.
    """

    def build(
        batch_size: int,
        size: int,
        seed: int,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        del batch
        generator = torch.Generator().manual_seed(int(seed) + 7)
        height = width = int(size)
        half = ((height + 1) // 2, (width + 1) // 2)
        quarter = ((height + 3) // 4, (width + 3) // 4)
        eighth = ((height + 7) // 8, (width + 7) // 8)
        sizes = ((height, width), half, quarter)

        def logits(index: int) -> torch.Tensor:
            h, w = sizes[index]
            return torch.randn(batch_size, 1, h, w, generator=generator)

        return {
            "teacher_target": teacher_target_cls(
                mask_logits=(logits(0), logits(1), logits(2)),
                primary_logits=(logits(0), logits(1), logits(2)),
                refine_bg=torch.rand(
                    batch_size, 3, height, width, generator=generator
                ),
                coarse_bg=torch.rand(
                    batch_size, 3, height, width, generator=generator
                ),
                bottleneck_h8=torch.randn(
                    batch_size, 256, eighth[0], eighth[1], generator=generator
                ),
                bg_decoder_h4=torch.randn(
                    batch_size, 128, quarter[0], quarter[1], generator=generator
                ),
                height=height,
                width=width,
            )
        }

    return build


def _config(
    trainer_module: Any,
    args: argparse.Namespace,
    output_dir: Path,
    schedule: str,
):
    return trainer_module.TrainerConfig(
        epochs=2,
        max_steps=None,
        batch_size_per_gpu=int(args.batch_size),
        schedule=schedule,
        teacher_decay_epochs=1,
        lr=3e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.99),
        warmup_steps=min(4, int(args.steps_per_epoch)),
        min_lr=1e-6,
        grad_clip=1.0,
        ema_decay=0.999,
        amp="fp16",
        channels_last=True,
        cudnn_benchmark=True,
        compile=bool(args.compile),
        compile_mode="default",
        ddp=True,
        ddp_static_graph=False,
        ddp_find_unused_parameters=False,
        ddp_gradient_as_bucket_view=True,
        device="cuda",
        seed=int(args.seed),
        log_every=1,
        val_every_epochs=0,
        validate_ema=False,
        finite_checks=True,
        fail_on_overflow=True,
        output_dir=str(output_dir),
        save_best=False,
        save_every_epoch=True,
        save_on_exit=False,
    )


def _assert_two_epoch_state(
    trainer: Any, steps_per_epoch: int, *, phase: str
) -> None:
    expected_steps = 2 * steps_per_epoch
    if trainer.epoch != 2:
        raise AssertionError(
            f"{phase}: expected epoch 2, got {trainer.epoch}"
        )
    if trainer.global_step != expected_steps:
        raise AssertionError(
            f"{phase}: expected {expected_steps} steps, got {trainer.global_step}"
        )
    if trainer.overflow_steps != 0 or trainer.skipped_steps != 0:
        raise AssertionError(
            f"{phase}: FP16 overflow gate failed "
            f"(overflow={trainer.overflow_steps}, skipped={trainer.skipped_steps})"
        )
    trainer.assert_finite_state(check_grads=False)
    if trainer.last_loss is None or not math.isfinite(trainer.last_loss):
        raise AssertionError(f"{phase}: final loss is not finite")


def _run_schedule(
    *,
    name: str,
    schedule: str,
    loss_factory: Callable[[], Any],
    target_builder: Callable[..., dict[str, Any]] | None,
    args: argparse.Namespace,
    trainer_module: Any,
    model_cls: type,
    rank: int,
) -> dict[str, Any]:
    steps = int(args.steps_per_epoch)
    run_dir = Path(args.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    # Keep one fresh curve file per invocation. The directory is a dedicated
    # bench output dir; this avoids mixing a previous session into the gate.
    if rank == 0 and log_path.exists():
        log_path.unlink()

    def build_trainer():
        model = model_cls(convnext_pretrained=False)
        loss = loss_factory()
        loss.to("cuda")
        loader = SyntheticStepLoader(
            steps=steps,
            batch_size=int(args.batch_size),
            size=int(args.size),
            seed=int(args.seed),
            rank=rank,
            target_builder=target_builder,
        )
        trainer = trainer_module.Trainer(
            model=model,
            loss=loss,
            train_loader=loader,
            val_loader=None,
            config=_config(trainer_module, args, run_dir, schedule),
            device="cuda",
        )
        return trainer, model, loss

    if dist.is_initialized():
        dist.barrier()

    trainer1, model1, loss1 = build_trainer()
    if trainer1.amp_dtype != torch.float16:
        raise AssertionError(
            f"{name}: strict FP16 gate requested but amp_dtype="
            f"{trainer1.amp_dtype}"
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer1.train(max_steps=steps)
    epoch1_seconds = time.perf_counter() - started
    if trainer1.epoch != 1 or trainer1.global_step != steps:
        raise AssertionError(
            f"{name}: epoch 1 did not stop cleanly "
            f"(epoch={trainer1.epoch}, step={trainer1.global_step})"
        )
    trainer1.assert_finite_state(check_grads=False)
    if trainer1.overflow_steps != 0 or trainer1.skipped_steps != 0:
        raise AssertionError(f"{name}: epoch 1 overflowed")
    epoch1_loss = (
        None if not trainer1.history else trainer1.history[-1].get("train_loss")
    )
    checkpoint = trainer1.save_checkpoint(run_dir / "epoch1.pt")
    if not checkpoint.is_file():
        raise AssertionError(f"{name}: epoch 1 checkpoint was not written")
    checkpoint_epoch = trainer1.epoch
    checkpoint_step = trainer1.global_step
    compile_seconds = trainer1.compile_seconds
    first_step_seconds = trainer1.first_step_seconds
    del trainer1, model1, loss1
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()

    trainer2, model2, loss2 = build_trainer()
    trainer2.resume(checkpoint)
    if (
        trainer2.epoch != checkpoint_epoch
        or trainer2.global_step != checkpoint_step
    ):
        raise AssertionError(
            f"{name}: resume did not restore epoch/step "
            f"({trainer2.epoch}/{trainer2.global_step} vs "
            f"{checkpoint_epoch}/{checkpoint_step})"
        )
    resumed_epoch = trainer2.epoch
    resumed_step = trainer2.global_step
    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer2.train()
    epoch2_seconds = time.perf_counter() - started
    _assert_two_epoch_state(trainer2, steps, phase=name)
    final_checkpoint = trainer2.save_checkpoint(run_dir / "final.pt")

    epoch2_loss = (
        None if not trainer2.history else trainer2.history[-1].get("train_loss")
    )
    epoch_losses = [epoch1_loss, epoch2_loss]
    if mean_finite(epoch_losses) is None:
        raise AssertionError(f"{name}: non-finite epoch losses {epoch_losses}")

    teacher_weights_epoch2: list[float] = []
    if rank == 0:
        curves = _curve_summary(log_path)
        if curves["step_count"] != 2 * steps:
            raise AssertionError(
                f"{name}: expected {2 * steps} logged steps, "
                f"got {curves['step_count']}"
            )
        epoch_numbers = [entry["epoch"] for entry in curves["epochs"]]
        if epoch_numbers != [1, 2]:
            raise AssertionError(
                f"{name}: expected epoch events [1, 2], got {epoch_numbers}"
            )
        curve_losses = [entry["train_loss"] for entry in curves["epochs"]]
        if mean_finite(curve_losses) is None:
            raise AssertionError(
                f"{name}: non-finite epoch curve {curve_losses}"
            )
        for entry in curves["steps"]:
            if entry.get("epoch") != 2:
                continue
            value = (entry.get("components") or {}).get("teacher_weight")
            if value is not None:
                teacher_weights_epoch2.append(float(value))

    report = {
        "name": name,
        "schedule": schedule,
        "epochs": 2,
        "steps_per_epoch": steps,
        "global_step": trainer2.global_step,
        "resumed_epoch": resumed_epoch,
        "resumed_step": resumed_step,
        "epoch1_loss": epoch1_loss,
        "epoch_train_losses": epoch_losses,
        "final_loss": trainer2.last_loss,
        "overflow_steps": trainer2.overflow_steps,
        "skipped_steps": trainer2.skipped_steps,
        "amp_dtype": str(trainer2.amp_dtype).replace("torch.", ""),
        "strict_fp16": trainer2.amp_dtype == torch.float16,
        "channels_last": True,
        "compile": bool(args.compile),
        "compile_s": compile_seconds,
        "first_step_s": first_step_seconds,
        "epoch1_seconds": round(epoch1_seconds, 3),
        "epoch2_seconds": round(epoch2_seconds, 3),
        "peak_vram_mb": trainer2._peak_vram_mb(),
        "checkpoint_epoch1": str(checkpoint),
        "checkpoint_final": str(final_checkpoint),
        "curve_jsonl": str(log_path),
        "teacher_weights_epoch2": teacher_weights_epoch2,
        "teacher_weight_epoch2_zero": (
            all(value == 0.0 for value in teacher_weights_epoch2)
            if teacher_weights_epoch2
            else None
        ),
    }
    if rank == 0:
        print(
            "SMOKE_SCHEDULE " + json.dumps(report, sort_keys=True),
            flush=True,
        )
    del trainer2, model2, loss2
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
    return report


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from src.models.slbr_next import losses_stage1, losses_stage2, teacher
    from src.models.slbr_next import trainer as trainer_module
    from src.networks.slbr_next import SLBRNextV1

    rank, world_size, local_rank = trainer_module.init_distributed(enabled=True)
    if not torch.cuda.is_available():
        raise RuntimeError("the 2-epoch gate requires CUDA (2xT4 target)")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    trainer_module.seed_everything(int(args.seed), rank)
    if args.amp != "fp16":
        raise ValueError("the 2-epoch gate is defined with --amp fp16")
    if args.size < 16 or args.steps_per_epoch < 1 or args.batch_size < 1:
        raise ValueError("size must be >= 16; steps/batch must be >= 1")

    def stage2_loss():
        return Stage2ProbabilityLoss(
            losses_stage2.Stage2Loss(perceptual=None)
        )

    def two_stage_loss():
        return losses_stage1.TwoStageScheduleLoss(
            criterion=losses_stage1.Stage1Loss()
        )

    def one_stage_loss():
        return losses_stage1.OneStageLoss(
            supervised=losses_stage2.Stage2Loss(perceptual=None),
            teacher=None,
            teacher_decay_epochs=1,
            student_h8_channels=256,
            teacher_h8_channels=256,
        )

    target_builder = _make_target_builder(teacher.TeacherTarget)
    schedules: list[tuple[str, str, Callable[[], Any], Any]] = [
        ("stage2_supervised", "two_stage", stage2_loss, None)
    ]
    if not args.skip_ab:
        schedules.extend(
            [
                ("ab_two_stage", "two_stage", two_stage_loss, target_builder),
                ("ab_one_stage", "one_stage", one_stage_loss, target_builder),
            ]
        )

    reports: dict[str, Any] = {}
    for name, schedule, loss_factory, target_builder_fn in schedules:
        reports[name] = _run_schedule(
            name=name,
            schedule=schedule,
            loss_factory=loss_factory,
            target_builder=target_builder_fn,
            args=args,
            trainer_module=trainer_module,
            model_cls=SLBRNextV1,
            rank=rank,
        )

    if dist.is_initialized():
        dist.barrier()
    if not all(report["strict_fp16"] for report in reports.values()):
        raise AssertionError("strict FP16 was not active for every schedule")
    if not all(report["overflow_steps"] == 0 for report in reports.values()):
        raise AssertionError("an overflow step was recorded")
    if rank == 0 and "ab_one_stage" in reports:
        if reports["ab_one_stage"]["teacher_weight_epoch2_zero"] is not True:
            raise AssertionError(
                "the one-stage teacher auxiliary did not reach exact zero "
                "in epoch 2"
            )
    final: dict[str, Any] = {
        "ok": True,
        "kind": "2_epoch_synthetic_training_gate",
        "world_size": world_size,
        "size": int(args.size),
        "batch_size_per_gpu": int(args.batch_size),
        "steps_per_epoch": int(args.steps_per_epoch),
        "strict_fp16": True,
        "checkpoint_resume": True,
        "schedules": reports,
        "plumbing_ab": None,
    }
    if "ab_two_stage" in reports and "ab_one_stage" in reports:
        final["plumbing_ab"] = {
            "label": "plumbing_gate_only_not_a_quality_claim",
            "note": (
                "Synthetic data and random teacher targets: this A/B proves "
                "that both schedules plumb through the trainer/DDP/AMP/"
                "checkpoint path. It says nothing about model quality."
            ),
            "two_stage_final_train_loss": reports["ab_two_stage"][
                "epoch_train_losses"
            ][-1],
            "one_stage_final_train_loss": reports["ab_one_stage"][
                "epoch_train_losses"
            ][-1],
            "one_stage_teacher_weight_epoch2_zero": reports["ab_one_stage"][
                "teacher_weight_epoch2_zero"
            ],
        }

    if rank == 0:
        report_path = Path(args.output_dir) / "smoke_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(final, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        print("SMOKE_2EPOCH_OK " + json.dumps(final, sort_keys=True), flush=True)
    return final


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank = int(os.environ.get("RANK", "0") or "0")
    try:
        run_smoke(args)
        return 0
    except BaseException as exc:  # pragma: no cover - Kaggle failure path
        if rank == 0:
            print(
                "SMOKE_2EPOCH_FAILED "
                + json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True
                ),
                flush=True,
            )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
