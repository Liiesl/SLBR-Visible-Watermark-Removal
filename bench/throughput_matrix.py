"""2xT4 throughput matrix for SLBR-Next (spec §5.4, issue #14).

The matrix exercises the **real** training step: ``SLBRNextV1`` with T3's
``Stage2Loss`` (perceptual disabled), fused AdamW, EMA, AMP, channels_last and
optional ``torch.compile``. Only the data is synthetic, so no dataset or
network access is needed.

Design
------
A row is one ``torchrun --nproc_per_node=2`` process group. The parent runner
spawns a fresh ``torchrun`` per row instead of re-initializing DDP in-process.
That also makes OOM recovery safe: a row that runs out of memory is recorded
and the parent continues with the next row.

One-cell Kaggle command::

    !cd /kaggle/working/SLBR-Visible-Watermark-Removal && \
      python -m bench.throughput_matrix \
        --output-dir /kaggle/working/slbr_next_t6_bench

Output markers::

    BENCH_TABLE
    | row | ...
    BEST_CONFIG {...}
    BENCH_RESULTS {...}

``BEST_CONFIG`` is the fastest row that did not OOM, had zero measured
overflow/skipped steps, a finite loss, and did not silently fall back from
``torch.compile``. The full JSON and markdown are written to
``<output-dir>/results.json`` and ``<output-dir>/matrix.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.common import (  # noqa: E402  (path bootstrap above)
    Stage2ProbabilityLoss,
    SyntheticStepLoader,
    is_cuda_oom,
    mean_finite,
)

__all__ = [
    "aggregate_rank_results",
    "benchmark_specs",
    "best_config_payload",
    "default_specs",
    "format_best_line",
    "render_markdown",
    "row_is_stable",
    "run_matrix",
    "select_best",
    "write_outputs",
]

_TABLE_HEADERS = (
    "row",
    "amp",
    "layout",
    "compile",
    "batch/GPU",
    "samples/s",
    "peak VRAM MB/GPU",
    "loss",
    "overflow",
    "skipped",
    "OOM",
    "compile s",
    "first step s",
    "status",
)


# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------


def _spec(
    identifier: str,
    *,
    amp: str,
    channels_last: bool,
    compile: bool,
    batch_size_per_gpu: int,
    size: int = 256,
    world_size: int = 2,
    warmup_steps: int = 2,
    measured_steps: int = 5,
    seed: int = 1,
    optional: bool = False,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "amp": amp,
        "channels_last": bool(channels_last),
        "compile": bool(compile),
        "batch_size_per_gpu": int(batch_size_per_gpu),
        "world_size": int(world_size),
        "size": int(size),
        "warmup_steps": int(warmup_steps),
        "measured_steps": int(measured_steps),
        "seed": int(seed),
        "optional": bool(optional),
    }


def default_specs(
    *, warmup_steps: int = 2, measured_steps: int = 5, seed: int = 1
) -> list[dict[str, Any]]:
    """Return the §5.4 matrix rows.

    Required rows:
    * FP16 + channels_last + no compile at batch 6/8/10;
    * FP16 + channels_last + compile at batch 6/8/10;
    * FP32 + channels_last + no compile at batch 6;
    * FP16 + contiguous + no compile at batch 6.

    Optional rows (controlled by ``--skip-optional``):
    * FP32 + contiguous + no compile at batch 6;
    * FP16 + contiguous + compile at batch 6.
    """
    common = dict(
        size=256,
        world_size=2,
        warmup_steps=int(warmup_steps),
        measured_steps=int(measured_steps),
        seed=int(seed),
    )
    rows: list[dict[str, Any]] = []
    for batch in (6, 8, 10):
        rows.append(
            _spec(
                f"fp16_cl_nocompile_b{batch}",
                amp="fp16",
                channels_last=True,
                compile=False,
                batch_size_per_gpu=batch,
                **common,
            )
        )
    for batch in (6, 8, 10):
        rows.append(
            _spec(
                f"fp16_cl_compile_b{batch}",
                amp="fp16",
                channels_last=True,
                compile=True,
                batch_size_per_gpu=batch,
                **common,
            )
        )
    rows.append(
        _spec(
            "fp32_cl_nocompile_b6",
            amp="fp32",
            channels_last=True,
            compile=False,
            batch_size_per_gpu=6,
            **common,
        )
    )
    rows.append(
        _spec(
            "fp16_nocl_nocompile_b6",
            amp="fp16",
            channels_last=False,
            compile=False,
            batch_size_per_gpu=6,
            **common,
        )
    )
    rows.append(
        _spec(
            "fp32_nocl_nocompile_b6",
            amp="fp32",
            channels_last=False,
            compile=False,
            batch_size_per_gpu=6,
            optional=True,
            **common,
        )
    )
    rows.append(
        _spec(
            "fp16_nocl_compile_b6",
            amp="fp16",
            channels_last=False,
            compile=True,
            batch_size_per_gpu=6,
            optional=True,
            **common,
        )
    )
    return rows


# Keep the older public name used in tests/docs.
benchmark_specs = default_specs


# ---------------------------------------------------------------------------
# Row parsing, stability and selection
# ---------------------------------------------------------------------------


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _payload_rank(payload: Mapping[str, Any]) -> int:
    value = payload.get("rank", -1)
    if value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _base_row(spec: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": str(spec.get("id", "row")),
        "amp": str(spec.get("amp", "?")),
        "channels_last": bool(spec.get("channels_last", False)),
        "compile": bool(spec.get("compile", False)),
        "batch_size_per_gpu": int(spec.get("batch_size_per_gpu", 0) or 0),
        "world_size": int(spec.get("world_size", 1) or 1),
        "size": int(spec.get("size", 0) or 0),
        "warmup_steps": int(spec.get("warmup_steps", 0) or 0),
        "measured_steps": int(spec.get("measured_steps", 0) or 0),
        "seed": int(spec.get("seed", 0) or 0),
        "samples_per_s": None,
        "peak_vram_mb": None,
        "loss": None,
        "loss_finite": False,
        "overflow_steps": None,
        "skipped_steps": None,
        "oom": False,
        "status": "error",
        "error": None,
        "compile_s": None,
        "first_step_s": None,
        "compile_fallback": False,
        "step_seconds": None,
        "returncode": None,
    }
    return row


def aggregate_rank_results(
    spec: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]],
    *,
    returncode: int | None = 0,
    output_text: str = "",
) -> dict[str, Any]:
    """Combine per-rank worker payloads into one matrix row.

    OOM is sticky: any rank OOM, exit code 42, or an ``out of memory`` marker
    in the process output marks the whole row as OOM. A complete row must have
    exactly one payload for every rank. Overflow/skip counts are the maximum
    observed on any rank (not a sum), so one rank's skipped update is not
    double-counted.
    """
    row = _base_row(spec)
    row["returncode"] = returncode
    payloads = [dict(payload) for payload in payloads if isinstance(payload, Mapping)]
    payloads.sort(key=lambda payload: _payload_rank(payload))

    text = str(output_text or "").lower()
    oom_text = "out of memory" in text or "outofmemoryerror" in text
    oom = bool(any(payload.get("oom") for payload in payloads))
    oom = oom or returncode == 42 or oom_text
    row["oom"] = oom

    if oom:
        row["status"] = "oom"
        row["error"] = "CUDA out of memory"
        return row

    expected = int(spec.get("world_size", 1) or 1)
    ranks = {_payload_rank(payload) for payload in payloads}
    if len(payloads) != expected or ranks != set(range(expected)):
        row["status"] = "error"
        row["error"] = (
            f"expected {expected} rank result(s), got {len(payloads)} "
            f"(ranks={sorted(ranks)})"
        )
        return row

    if returncode not in (0, None):
        row["status"] = "error"
        row["error"] = f"torchrun exited with code {returncode}"
        return row

    failed = [
        payload
        for payload in payloads
        if str(payload.get("status", "ok")).lower() != "ok"
    ]
    if failed:
        row["status"] = "error"
        messages = [
            str(payload.get("error") or "rank failed") for payload in failed
        ]
        row["error"] = "; ".join(messages)[:2000]
        return row

    rank0 = payloads[0]
    try:
        samples_per_s = float(rank0["samples_per_s"])
        measured_steps = int(rank0["measured_steps"])
    except (KeyError, TypeError, ValueError):
        row["status"] = "error"
        row["error"] = "rank 0 result is missing throughput fields"
        return row

    if measured_steps < 1:
        row["status"] = "error"
        row["error"] = "rank 0 measured zero steps"
        return row

    losses = [
        payload.get("loss") for payload in payloads
    ]
    loss = mean_finite(losses)  # type: ignore[arg-type]
    overflow_values = [
        int(payload.get("overflow_steps", 0) or 0) for payload in payloads
    ]
    skipped_values = [
        int(payload.get("skipped_steps", 0) or 0) for payload in payloads
    ]
    peak = [
        payload.get("peak_vram_mb") for payload in payloads
    ]
    compile_times = [
        payload.get("compile_s")
        for payload in payloads
        if payload.get("compile_s") is not None
    ]
    first_step_times = [
        payload.get("first_step_s")
        for payload in payloads
        if payload.get("first_step_s") is not None
    ]
    row.update(
        {
            "samples_per_s": samples_per_s,
            "loss": loss,
            "loss_finite": loss is not None and math.isfinite(loss),
            "overflow_steps": max(overflow_values),
            "skipped_steps": max(skipped_values),
            "peak_vram_mb": peak,
            "compile_s": max(compile_times) if compile_times else None,
            "first_step_s": max(first_step_times) if first_step_times else None,
            "compile_fallback": bool(
                any(payload.get("compile_fallback") for payload in payloads)
            ),
            "measured_steps": measured_steps,
            "step_seconds": rank0.get("step_seconds"),
            "status": "ok",
            "error": None,
        }
    )
    return row


def row_is_stable(row: Mapping[str, Any]) -> bool:
    """Return True when a row is eligible for ``BEST_CONFIG``."""
    try:
        if str(row.get("status", "")).lower() != "ok":
            return False
        if bool(row.get("oom")):
            return False
        if int(row.get("overflow_steps", 0) or 0) != 0:
            return False
        if int(row.get("skipped_steps", 0) or 0) != 0:
            return False
        if int(row.get("measured_steps", 0) or 0) < 1:
            return False
        samples = row.get("samples_per_s")
        loss = row.get("loss")
        if not _finite_number(samples) or not _finite_number(loss):
            return False
        if float(samples) <= 0.0:
            return False
        # A requested compile that fell back is not the configuration the row
        # claims to measure. Keep the row in the table but out of selection.
        if bool(row.get("compile")) and bool(row.get("compile_fallback")):
            return False
        return True
    except (TypeError, ValueError):
        return False


def select_best(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the fastest stable row, or ``None`` when none is eligible."""
    stable = [dict(row) for row in rows if row_is_stable(row)]
    if not stable:
        return None
    return max(stable, key=lambda row: float(row["samples_per_s"]))


def best_config_payload(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Compact JSON payload for the ``BEST_CONFIG`` marker."""
    if row is None:
        return None
    keys = (
        "id",
        "amp",
        "channels_last",
        "compile",
        "batch_size_per_gpu",
        "world_size",
        "size",
        "warmup_steps",
        "measured_steps",
        "samples_per_s",
        "peak_vram_mb",
        "loss",
        "overflow_steps",
        "skipped_steps",
        "oom",
        "compile_s",
        "first_step_s",
        "compile_fallback",
    )
    return {key: row.get(key) for key in keys}


def format_best_line(row: Mapping[str, Any] | None) -> str:
    """Return the exact ``BEST_CONFIG`` line printed by the runner."""
    payload = best_config_payload(row)
    return "BEST_CONFIG " + json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# Markdown / JSON output
# ---------------------------------------------------------------------------


def _format_value(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).replace("|", "\\|")
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}{suffix}"


def _format_peak(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        parts = [
            "?" if item is None else _format_value(item, 0) for item in value
        ]
        return " / ".join(parts)
    return _format_value(value, 0)


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the compact matrix table used by the README and issue."""
    lines = [
        "| " + " | ".join(_TABLE_HEADERS) + " |",
        "|" + "|".join("---" for _ in _TABLE_HEADERS) + "|",
    ]
    for row in rows:
        status = str(row.get("status", "error"))
        if bool(row.get("oom")):
            status = "OOM"
        compile_cell = "yes" if row.get("compile") else "no"
        if row.get("compile_fallback"):
            compile_cell += " (fallback)"
        values = [
            str(row.get("id", "?")),
            str(row.get("amp", "?")),
            "channels_last" if row.get("channels_last") else "contiguous",
            compile_cell,
            _format_value(row.get("batch_size_per_gpu"), 0),
            _format_value(row.get("samples_per_s"), 2),
            _format_peak(row.get("peak_vram_mb")),
            _format_value(row.get("loss"), 4),
            _format_value(row.get("overflow_steps"), 0),
            _format_value(row.get("skipped_steps"), 0),
            "yes" if row.get("oom") else "no",
            _format_value(row.get("compile_s"), 1),
            _format_value(row.get("first_step_s"), 2),
            status,
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_outputs(
    rows: Sequence[Mapping[str, Any]], output_dir: str | Path
) -> dict[str, Path]:
    """Write ``results.json`` and ``matrix.md``; return their paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    best = select_best(rows)
    payload = {
        "generated_at": round(time.time(), 3),
        "world_size": 2,
        "rows": [dict(row) for row in rows],
        "best": None if best is None else dict(best),
        "best_config": best_config_payload(best),
    }
    results_path = directory / "results.json"
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown_path = directory / "matrix.md"
    markdown_path.write_text(render_markdown(rows) + "\n", encoding="utf-8")
    return {"results_json": results_path, "matrix_md": markdown_path}


# ---------------------------------------------------------------------------
# Worker (one row / one torchrun process group)
# ---------------------------------------------------------------------------


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rank_from_env() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0") or "0"),
        int(os.environ.get("WORLD_SIZE", "1") or "1"),
        int(os.environ.get("LOCAL_RANK", "0") or "0"),
    )


def _move_batches(
    batches: list[dict[str, Any]], device: torch.device, *, channels_last: bool
) -> list[dict[str, Any]]:
    moved: list[dict[str, Any]] = []
    for batch in batches:
        entry: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                tensor = value.to(device, non_blocking=True)
                if (
                    channels_last
                    and tensor.is_floating_point()
                    and tensor.dim() == 4
                ):
                    tensor = tensor.contiguous(memory_format=torch.channels_last)
                entry[key] = tensor
            else:
                entry[key] = value
        moved.append(entry)
    return moved


def run_worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run one matrix row under torchrun and write a per-rank JSON result."""
    from src.models.slbr_next.losses_stage2 import Stage2Loss
    from src.models.slbr_next.trainer import (
        Trainer,
        TrainerConfig,
        init_distributed,
        seed_everything,
    )
    from src.networks.slbr_next import SLBRNextV1

    rank, world_size, local_rank = _rank_from_env()
    row_dir = Path(str(spec["output_dir"]))
    row_dir.mkdir(parents=True, exist_ok=True)
    result_path = row_dir / f"rank{rank:02d}.json"

    init_distributed(enabled=True)
    if world_size != int(spec["world_size"]):
        raise RuntimeError(
            f"worker world_size={world_size} but spec requested "
            f"{spec['world_size']}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("the throughput matrix requires CUDA")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    seed_everything(int(spec.get("seed", 1)), rank)

    device = torch.device("cuda", local_rank)
    warmup_steps = int(spec["warmup_steps"])
    measured_steps = int(spec["measured_steps"])
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if measured_steps < 1:
        raise ValueError("measured_steps must be >= 1")
    total_steps = warmup_steps + measured_steps

    model = SLBRNextV1(convnext_pretrained=False)
    criterion = Stage2Loss(perceptual=None)
    loss = Stage2ProbabilityLoss(criterion)
    loss.to(device)
    loader = SyntheticStepLoader(
        steps=total_steps,
        batch_size=int(spec["batch_size_per_gpu"]),
        size=int(spec["size"]),
        seed=int(spec.get("seed", 1)),
        rank=rank,
    )
    batches = loader.materialize()

    config = TrainerConfig(
        epochs=1,
        max_steps=total_steps,
        batch_size_per_gpu=int(spec["batch_size_per_gpu"]),
        lr=3e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.99),
        warmup_steps=min(500, total_steps),
        min_lr=1e-6,
        grad_clip=1.0,
        ema_decay=0.999,
        amp=str(spec["amp"]),
        channels_last=bool(spec["channels_last"]),
        cudnn_benchmark=True,
        compile=bool(spec["compile"]),
        compile_mode="default",
        ddp=True,
        ddp_static_graph=False,
        ddp_find_unused_parameters=False,
        ddp_gradient_as_bucket_view=True,
        device="cuda",
        seed=int(spec.get("seed", 1)),
        log_every=total_steps + 1,
        val_every_epochs=0,
        validate_ema=False,
        finite_checks=False,
        fail_on_overflow=False,
        output_dir=str(row_dir),
        save_best=False,
        save_every_epoch=False,
        save_on_exit=False,
    )
    trainer = Trainer(
        model=model,
        loss=loss,
        train_loader=batches,
        val_loader=None,
        config=config,
        device="cuda",
    )

    # Move the synthetic batches once so timed steps measure compute, not H2D
    # copies of freshly generated random data.
    batches = _move_batches(
        batches, device, channels_last=bool(spec["channels_last"])
    )

    for batch in batches[:warmup_steps]:
        trainer._train_step(batch)
    torch.cuda.synchronize(device)

    overflow_before = trainer.overflow_steps
    skipped_before = trainer.skipped_steps
    step_seconds: list[float] = []
    losses: list[float | None] = []
    for batch in batches[warmup_steps:]:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        trainer._train_step(batch)
        torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)
        losses.append(trainer.last_loss)

    trainer.assert_finite_state(check_grads=False)
    total_measured = sum(step_seconds)
    samples_per_s = (
        float(spec["batch_size_per_gpu"]) * world_size * measured_steps
        / total_measured
        if total_measured > 0
        else None
    )
    loss_mean = mean_finite(losses)
    payload: dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "status": "ok",
        "oom": False,
        "samples_per_s": samples_per_s,
        "measured_steps": measured_steps,
        "warmup_steps": warmup_steps,
        "step_seconds": [round(value, 6) for value in step_seconds],
        "loss": loss_mean,
        "loss_finite": loss_mean is not None,
        # Cumulative over warmup + measured; a stable row must have none.
        "overflow_steps": trainer.overflow_steps,
        "skipped_steps": trainer.skipped_steps,
        "measured_overflow_steps": trainer.overflow_steps - overflow_before,
        "measured_skipped_steps": trainer.skipped_steps - skipped_before,
        "peak_vram_mb": (
            None
            if trainer._peak_vram_mb() is None
            else float(trainer._peak_vram_mb())
        ),
        "compile_s": trainer.compile_seconds,
        "first_step_s": trainer.first_step_seconds,
        "compile_fallback": trainer.compile_fallback,
        "amp_dtype": str(trainer.amp_dtype).replace("torch.", ""),
        "parameters": trainer._parameter_count(),
        "global_step": trainer.global_step,
    }
    _atomic_json(result_path, payload)
    if rank == 0:
        print("BENCH_ROW " + json.dumps(payload, sort_keys=True), flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return payload


def _worker_main(spec_path: str | Path) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    rank, _, _ = _rank_from_env()
    row_dir = Path(str(spec["output_dir"]))
    result_path = row_dir / f"rank{rank:02d}.json"
    try:
        run_worker(spec)
        return 0
    except BaseException as exc:  # pragma: no cover - exercised on Kaggle
        oom = is_cuda_oom(exc)
        failure = {
            "rank": rank,
            "world_size": int(os.environ.get("WORLD_SIZE", "1") or "1"),
            "status": "error",
            "oom": oom,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _atomic_json(result_path, failure)
        print(
            "BENCH_ROW_FAILED " + json.dumps(failure, sort_keys=True),
            flush=True,
        )
        # Do not run collectives here: a peer may already be waiting in DDP.
        # torchrun will terminate the group when this process exits.
        os._exit(42 if oom else 43)


# ---------------------------------------------------------------------------
# Parent runner
# ---------------------------------------------------------------------------


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    process.kill()


def _run_process(
    command: list[str], *, cwd: Path, timeout: float | None
) -> tuple[int, str, bool]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        start_new_session=(os.name == "posix"),
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        return int(process.returncode or 0), output, False
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        output, _ = process.communicate()
        return -9, output, True


def _load_rank_payloads(row_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(row_dir.glob("rank*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _tail(text: str, lines: int = 25) -> str:
    split = str(text or "").splitlines()
    return "\n".join(split[-lines:])


def run_matrix(
    specs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    row_timeout: float | None = 1800.0,
    python_executable: str | None = None,
) -> list[dict[str, Any]]:
    """Run every spec in a fresh torchrun and return the aggregated rows."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    executable = python_executable or sys.executable
    script = str(Path(__file__).resolve())
    rows: list[dict[str, Any]] = []

    for spec in specs:
        spec = dict(spec)
        row_dir = directory / str(spec["id"])
        row_dir.mkdir(parents=True, exist_ok=True)
        for stale in row_dir.glob("rank*.json"):
            stale.unlink()
        spec["output_dir"] = str(row_dir)
        spec_path = row_dir / "spec.json"
        spec_path.write_text(
            json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8"
        )
        command = [
            executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(int(spec["world_size"])),
            "--standalone",
            script,
            "--worker",
            "--spec",
            str(spec_path),
        ]
        print(
            "BENCH_ROW_START "
            + json.dumps(
                {
                    "id": spec["id"],
                    "amp": spec["amp"],
                    "channels_last": spec["channels_last"],
                    "compile": spec["compile"],
                    "batch_size_per_gpu": spec["batch_size_per_gpu"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        started = time.perf_counter()
        try:
            returncode, output, timed_out = _run_process(
                command, cwd=_REPO_ROOT, timeout=row_timeout
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # pragma: no cover - launcher failure
            returncode = -1
            output = f"{type(exc).__name__}: {exc}"
            timed_out = False
        elapsed = time.perf_counter() - started
        log_path = row_dir / "row.log"
        log_path.write_text(output, encoding="utf-8")

        payloads = _load_rank_payloads(row_dir)
        row = aggregate_rank_results(
            spec,
            payloads,
            returncode=returncode,
            output_text=output,
        )
        if timed_out:
            row["status"] = "timeout"
            row["error"] = f"row exceeded {row_timeout}s"
        row["elapsed_s"] = round(elapsed, 2)
        row["log"] = str(log_path)
        _atomic_json(row_dir / "row.json", row)
        rows.append(row)
        print(
            "BENCH_ROW_DONE "
            + json.dumps(
                {
                    "id": row["id"],
                    "status": row["status"],
                    "samples_per_s": row["samples_per_s"],
                    "peak_vram_mb": row["peak_vram_mb"],
                    "oom": row["oom"],
                    "elapsed_s": row["elapsed_s"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not row_is_stable(row):
            print(
                f"[{row['id']}] excluded: status={row['status']} "
                f"error={row.get('error')}",
                flush=True,
            )
            if row["status"] not in {"oom", "timeout"}:
                print(_tail(output), flush=True)

    best = select_best(rows)
    paths = write_outputs(rows, directory)
    print("BENCH_TABLE", flush=True)
    print(render_markdown(rows), flush=True)
    print(format_best_line(best), flush=True)
    print(
        "BENCH_RESULTS "
        + json.dumps(
            {
                "results_json": str(paths["results_json"]),
                "matrix_md": str(paths["matrix_md"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cuda_device_count() -> int:
    """Count visible GPUs without creating a CUDA context in the parent.

    ``torch.cuda.device_count()`` initializes a CUDA context in the notebook
    process, which would permanently hold a few hundred MB on GPU 0 and bias
    the batch-10 OOM edge. Prefer ``nvidia-smi``/``CUDA_VISIBLE_DEVICES`` and
    fall back to torch only when neither is available.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if entries == ["-1"]:
            return 0
        if entries:
            return len(entries)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return sum(
                1 for line in result.stdout.splitlines() if line.startswith("GPU ")
            )
    except (OSError, subprocess.SubprocessError):
        pass
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLBR-Next 2xT4 throughput matrix (issue #14)"
    )
    parser.add_argument("--output-dir", default="bench/results")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5,
                        help="Measured optimizer steps per row.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--row-timeout", type=float, default=1800.0)
    parser.add_argument("--skip-optional", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--spec", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.spec is None:
            raise SystemExit("--worker requires --spec")
        return _worker_main(args.spec)

    device_count = _cuda_device_count()
    if device_count < 2:
        print(
            "BENCH_ERROR the matrix requires 2 CUDA GPUs; "
            f"found {device_count}",
            file=sys.stderr,
        )
        return 2
    specs = default_specs(
        warmup_steps=args.warmup, measured_steps=args.steps, seed=args.seed
    )
    if args.skip_optional:
        specs = [spec for spec in specs if not spec.get("optional")]
    run_matrix(specs, args.output_dir, row_timeout=args.row_timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
