# SLBR-Next T6 bench (issue #14)

This directory contains the T6 training gate:

* `throughput_matrix.py` — the 2×T4 matrix at 256×256. Each row is a real
  training step (`SLBRNextV1` + T3 `Stage2Loss`, perceptual disabled) with
  fused AdamW, EMA, FP16/FP32, `channels_last`, optional `torch.compile`, and
  synthetic batches. No dataset or network access is required.
* `smoke_2epoch.py` — the 2-epoch synthetic smoke with a fresh
  checkpoint/resume after epoch 1, JSONL curves and strict FP16 finite checks.
  It also runs the T5 two-stage vs one-stage **plumbing A/B** (not a quality
  comparison).
* `common.py` — the synthetic loader and the v1-probabilities → Stage2-logits
  adapter. The adapter keeps the mask term differentiable.

No dependencies were added; the scripts use `torch` and the standard library.

## Kaggle one-cell commands

Adjust the repository path if the Kaggle checkout has a different name.

### 1. Throughput matrix (one cell, whole matrix)

```python
!cd /kaggle/working/SLBR-Visible-Watermark-Removal && python -m bench.throughput_matrix --output-dir /kaggle/working/slbr_next_t6_bench
```

The parent runner launches a fresh `torchrun --nproc_per_node=2` for every row,
so a CUDA OOM in one row cannot poison the rest of the matrix. It continues
through the matrix and selects the fastest stable row.

Rows (10 by default; pass `--skip-optional` for 8):

| precision | layout | compile | batch/GPU |
|---|---|---|---|
| FP16 | channels_last | no | 6, 8, 10 |
| FP16 | channels_last | yes | 6, 8, 10 |
| FP32 | channels_last | no | 6 |
| FP16 | contiguous | no | 6 |
| FP32 | contiguous | no | 6 (optional) |
| FP16 | contiguous | yes | 6 (optional) |

Each row warms up for 2 steps, measures 5 steps, and reports:

* total samples/s across both GPUs (`batch/GPU × 2 / mean measured step`);
* peak allocated VRAM per GPU (a ranked list);
* mean measured loss;
* overflow and skipped-step counts (cumulative over warmup + measured);
* OOM flag, compile time, first-step time, compile-fallback flag.

`BEST_CONFIG` requires: `status=ok`, no OOM, zero overflow, zero skipped steps,
positive finite throughput, finite loss, and no silent `torch.compile` fallback.

Expected output markers:

```text
BENCH_ROW_START {...}
BENCH_ROW_DONE {...}
BENCH_TABLE
| row | amp | layout | compile | ...
BEST_CONFIG {...}
BENCH_RESULTS {"matrix_md": "...", "results_json": "..."}
```

Files written under `--output-dir`:

* `results.json` — all rows, the selected best row and its compact config;
* `matrix.md` — the same compact table;
* `<row>/row.log`, `<row>/rankXX.json`, `<row>/row.json` — per-row evidence.

### 2. Two-epoch smoke with checkpoint/resume + A/B

```python
!cd /kaggle/working/SLBR-Visible-Watermark-Removal && python -m torch.distributed.run --nproc_per_node=2 --standalone bench/smoke_2epoch.py --output-dir /kaggle/working/slbr_next_t6_smoke
```

Defaults: 256×256, batch 2/GPU, 8 steps/epoch, FP16 autocast + `GradScaler`,
`channels_last`, eager (no compile), 2 epochs, checkpoint after epoch 1 and a
fresh trainer resumed for epoch 2. `--skip-ab` runs only the primary Stage 2
smoke; `--compile` enables compile but Stage 1 feature hooks can graph-break it.

Expected output markers:

```text
SMOKE_SCHEDULE {...}          # once per schedule
SMOKE_2EPOCH_OK {...}         # final gate; only on success
SMOKE_2EPOCH_FAILED {...}     # failure path
```

Artifacts: `smoke_report.json`, plus `<schedule>/train_log.jsonl`,
`<schedule>/config.json`, `epoch1.pt` and `final.pt`.

The A/B section is labelled in the JSON as
`"plumbing_gate_only_not_a_quality_claim"`. It uses synthetic data and random
teacher targets to prove that T5's schedules both plumb through the trainer,
DDP, AMP and checkpoint path. It is not evidence of model quality.

## Evidence status

The code and CPU tests can be merged without GPUs, but the acceptance matrix
and 2-epoch smoke must be run on Kaggle 2×T4. Issue #14 stays open until those
markers and the JSON/markdown are posted from a real Kaggle run. The exact
commands above are the intended evidence commands.
