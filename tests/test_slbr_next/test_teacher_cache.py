"""Tests for the SLBR teacher hooks and sharded target cache (issue #9).

Coverage:

* hook shapes and finest-first resolution pairing on a synthetic teacher whose
  decoder levels execute in reverse order (H -> H/2 -> H/4), proving that
  pairing is by spatial resolution rather than hook order;
* the final mask appears exactly once in ``TeacherTarget.mask_logits``, and
  the 1x1 primary / 3x3 self-calibrated heads are kept separate;
* eval/no_grad/FP32 behaviour and the square/multiple-of-16 input contract;
* cache shard manifests, exact tensor round-trip, loading by key, resume after
  an abandoned in-memory buffer, and safe handling of orphan/partial shards;
* FP16 CUDA autocast when a GPU is available (skipped on the CPU runner);
* one small real SLBR forward at 64x64 to confirm the documented hook paths.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ``src.models.slbr_next`` is imported lazily in the autouse fixture below.
# Importing it at collection time would put ``src.models`` into ``sys.modules``
# before ``test_imports.py`` runs, breaking its import-isolation assertion even
# though the networks package itself is still isolated.
SLBRTeacher = None
TeacherTarget = None
TeacherTargetCache = None
CacheError = None
CACHE_FORMAT = None
CACHE_VERSION = None


@pytest.fixture(scope="module", autouse=True)
def _slbr_next_training_api():
    """Load the training package only when this module's tests execute."""
    global SLBRTeacher, TeacherTarget
    global TeacherTargetCache, CacheError, CACHE_FORMAT, CACHE_VERSION
    from src.models.slbr_next.cache import (
        CACHE_FORMAT as cache_format,
        CACHE_VERSION as cache_version,
        CacheError as cache_error,
        TeacherTargetCache as target_cache,
    )
    from src.models.slbr_next.teacher import (
        SLBRTeacher as teacher_wrapper,
        TeacherTarget as target_dataclass,
    )

    SLBRTeacher = teacher_wrapper
    TeacherTarget = target_dataclass
    TeacherTargetCache = target_cache
    CacheError = cache_error
    CACHE_FORMAT = cache_format
    CACHE_VERSION = cache_version


# ---------------------------------------------------------------------------
# Synthetic SLBR-shaped teacher
# ---------------------------------------------------------------------------


class _TinySelfCalibrated(nn.Module):
    """Mimics ``SelfAttentionSimple``'s final 3x3 ``out_conv`` head."""

    def __init__(self, channels: int, logit_value: float) -> None:
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, max(channels // 8, 1), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 1), 1, 3, padding=1),
        )
        with torch.no_grad():
            self.out_conv[0].weight.zero_()
            self.out_conv[0].bias.zero_()
            self.out_conv[2].weight.zero_()
            self.out_conv[2].bias.fill_(logit_value)

    def forward(self, xin: Tensor, xout: Tensor, xmask: Tensor):
        logits = self.out_conv(xout)
        return xout, logits.sigmoid()


class _TinySMRBlock(nn.Module):
    """Mimics ``SMRBlock`` with a 1x1 primary and 3x3 self-calibrated head."""

    def __init__(self, logit_value: float) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 1)
        self.primary_mask = nn.Sequential(
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )
        self.self_calibrated = _TinySelfCalibrated(8, logit_value)
        with torch.no_grad():
            self.primary_mask[0].weight.zero_()
            self.primary_mask[0].bias.fill_(logit_value + 10.0)

    def forward(self, x: Tensor, size: tuple[int, int]):
        feats = self.proj(
            F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        )
        primary = self.primary_mask(feats)
        _, self_calibrated = self.self_calibrated(feats, feats, primary)
        return primary, self_calibrated


class _TinyTeacher(nn.Module):
    """Small SLBR-shaped module with the three documented decoder levels.

    Level 0 produces H/4, level 1 H/2 and level 2 H. ``forward`` deliberately
    visits them in reverse order (H -> H/2 -> H/4) so hook-order-dependent
    implementations fail the resolution-pairing assertions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shared_decoder = nn.Module()
        self.shared_decoder.up_im_atts = nn.ModuleList(
            [nn.Conv2d(3, 16, 1)]
        )
        self.coarse_decoder = nn.Module()
        self.coarse_decoder.up_convs_mask = nn.ModuleList(
            [
                _TinySMRBlock(1.0),  # H/4
                _TinySMRBlock(2.0),  # H/2
                _TinySMRBlock(3.0),  # H
            ]
        )
        self.coarse_decoder.up_convs_bg = nn.ModuleList(
            [nn.Conv2d(3, 6, 1) for _ in range(3)]
        )

    def forward(self, x: Tensor):
        height, width = x.shape[2], x.shape[3]
        h8 = F.interpolate(
            x,
            size=(height // 8, width // 8),
            mode="bilinear",
            align_corners=False,
        )
        # This hook fires once, on the image stream only.
        self.shared_decoder.up_im_atts[0](h8)

        primary: dict[int, Tensor] = {}
        self_calibrated: dict[int, Tensor] = {}
        for level, divisor in ((2, 1), (1, 2), (0, 4)):
            size = (height // divisor, width // divisor)
            primary[divisor], self_calibrated[divisor] = (
                self.coarse_decoder.up_convs_mask[level](x, size)
            )

        h4 = F.interpolate(
            x,
            size=(height // 4, width // 4),
            mode="bilinear",
            align_corners=False,
        )
        self.coarse_decoder.up_convs_bg[0](h4)

        refine_bg = x * 0.5 + 0.25
        coarse_bg = x * 0.75
        # Same 7-entry layout as SLBR, including its mask_list[0]/[6]
        # duplicate. The wrapper must not reproduce that duplicate.
        mask_list = [
            self_calibrated[1],
            primary[4],
            self_calibrated[4],
            primary[2],
            self_calibrated[2],
            primary[1],
            self_calibrated[1],
        ]
        return [refine_bg, coarse_bg], mask_list, [None]


# ---------------------------------------------------------------------------
# Hook and target tests
# ---------------------------------------------------------------------------


def test_hook_shapes_resolution_pairing_and_no_duplicate_final():
    torch.manual_seed(0)
    teacher = _TinyTeacher().eval()
    wrapper = SLBRTeacher(teacher)
    x = torch.rand(2, 3, 32, 32)

    target = wrapper(x)

    assert target.height == target.width == 32
    assert target.batch_size == 2

    expected_scales = ((32, 32), (16, 16), (8, 8))
    for logits, primary, expected in zip(
        target.mask_logits, target.primary_logits, expected_scales
    ):
        assert logits.shape == (2, 1, *expected)
        assert primary.shape == (2, 1, *expected)

    # Finest-first mask logits. The synthetic teacher executes H first, so an
    # implementation that used hook order would return 1.0 at index 0 here.
    for logits, value in zip(target.mask_logits, (3.0, 2.0, 1.0)):
        torch.testing.assert_close(
            logits,
            torch.full_like(logits, value),
            rtol=0.0,
            atol=1e-6,
        )
    # Primary logits are a separate head (bias + 10), same scale order.
    for logits, value in zip(target.primary_logits, (13.0, 12.0, 11.0)):
        torch.testing.assert_close(
            logits,
            torch.full_like(logits, value),
            rtol=0.0,
            atol=1e-6,
        )

    # The H target appears exactly once; no mask_list[6] duplicate.
    assert len(target.mask_logits) == 3
    assert len({id(tensor) for tensor in target.mask_logits}) == 3
    assert sum(
        tensor.shape[-2:] == (32, 32) for tensor in target.mask_logits
    ) == 1
    assert target.final_mask_logits is target.mask_logits[0]

    assert target.bottleneck_h8.shape == (2, 16, 4, 4)
    assert target.bg_decoder_h4.shape == (2, 6, 8, 8)
    assert target.refine_bg.shape == (2, 3, 32, 32)
    assert target.coarse_bg.shape == (2, 3, 32, 32)

    # The finest captured logit is exactly the pre-sigmoid input of the
    # teacher's final self-calibrated mask (and the same for primary).
    with torch.no_grad():
        _, masks, _ = teacher(x)
    torch.testing.assert_close(
        torch.sigmoid(target.mask_logits[0]), masks[6], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        torch.sigmoid(target.primary_logits[0]),
        masks[5],
        rtol=0.0,
        atol=0.0,
    )

    # Batch splitting keeps the documented ordering and shapes.
    split = target.split()
    assert len(split) == 2
    for sample in split:
        assert sample.batch_size == 1
        assert [t.shape[-2:] for t in sample.mask_logits] == [
            (32, 32),
            (16, 16),
            (8, 8),
        ]


def test_teacher_runs_eval_no_grad_fp32_on_cpu():
    torch.manual_seed(1)
    teacher = _TinyTeacher()
    assert teacher.training is True
    wrapper = SLBRTeacher(teacher)

    assert teacher.training is False
    assert all(not p.requires_grad for p in teacher.parameters())

    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    target = wrapper(x)

    for tensor in target.tensors():
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert tensor.requires_grad is False
        assert torch.isfinite(tensor).all()


def test_teacher_rejects_non_square_and_misaligned_inputs():
    wrapper = SLBRTeacher(_TinyTeacher())

    with pytest.raises(ValueError, match="square-only"):
        wrapper(torch.rand(1, 3, 32, 40))

    with pytest.raises(ValueError, match="divisible by 16"):
        wrapper(torch.rand(1, 3, 30, 30))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP16 autocast requires CUDA; the hosted CPU runner skips this.",
)
def test_fp16_autocast_teacher_returns_fp32_logits():
    teacher = _TinyTeacher().cuda().eval()
    wrapper = SLBRTeacher(teacher, fp16_autocast=True)
    x = torch.rand(1, 3, 32, 32, device="cuda")

    target = wrapper(x)

    for tensor in target.tensors():
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert torch.isfinite(tensor).all()

    torch.testing.assert_close(
        target.mask_logits[0],
        torch.full_like(target.mask_logits[0], 3.0),
        rtol=0.0,
        atol=1e-3,
    )


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


def _make_target(seed: int, size: int = 16) -> TeacherTarget:
    generator = torch.Generator().manual_seed(seed)

    def logits(divisor: int) -> Tensor:
        side = size // divisor
        return torch.randn((1, 1, side, side), generator=generator)

    return TeacherTarget(
        mask_logits=tuple(logits(d) for d in (1, 2, 4)),
        primary_logits=tuple(logits(d) for d in (1, 2, 4)),
        refine_bg=torch.rand((1, 3, size, size), generator=generator),
        coarse_bg=torch.rand((1, 3, size, size), generator=generator),
        bottleneck_h8=torch.randn(
            (1, 4, size // 8, size // 8), generator=generator
        ),
        bg_decoder_h4=torch.randn(
            (1, 5, size // 4, size // 4), generator=generator
        ),
        height=size,
        width=size,
    )


def _assert_targets_equal(left: TeacherTarget, right: TeacherTarget) -> None:
    assert left.height == right.height
    assert left.width == right.width
    for left_tensor, right_tensor in zip(left.tensors(), right.tensors()):
        assert left_tensor.dtype == right_tensor.dtype
        assert left_tensor.shape == right_tensor.shape
        assert torch.equal(left_tensor, right_tensor)


def test_cache_round_trip_manifest_and_key_lookup(tmp_path):
    root = tmp_path / "cache"
    target_a = _make_target(1)
    target_b = _make_target(2)

    with TeacherTargetCache(root, shard_size=2) as cache:
        assert cache.put("img-000", target_a) is True
        assert cache.put("img-001", target_b) is True
        assert cache.num_shards == 1
        assert cache.keys() == ["img-000", "img-001"]

    shard_manifest_path = root / "shard-000000.json"
    assert shard_manifest_path.is_file()
    shard_manifest = json.loads(shard_manifest_path.read_text("utf-8"))
    assert shard_manifest["format"] == CACHE_FORMAT
    assert shard_manifest["version"] == CACHE_VERSION
    assert shard_manifest["status"] == "complete"
    assert shard_manifest["keys"] == ["img-000", "img-001"]
    assert shard_manifest["count"] == 2
    assert shard_manifest["sizes"] == [[16, 16], [16, 16]]
    assert (root / "shard-000000.pt").is_file()

    global_manifest = json.loads(
        (root / "manifest.json").read_text("utf-8")
    )
    assert global_manifest["format"] == CACHE_FORMAT
    assert global_manifest["shard_size"] == 2
    assert global_manifest["num_shards"] == 1
    assert global_manifest["num_samples"] == 2
    assert global_manifest["shards"][0]["count"] == 2

    reopened = TeacherTargetCache(root, shard_size=2)
    assert reopened.keys() == ["img-000", "img-001"]
    assert reopened.num_shards == 1
    assert reopened.contains("img-000")
    assert not reopened.contains("missing")
    assert reopened.missing(["img-000", "missing"]) == ["missing"]

    _assert_targets_equal(reopened.get("img-000"), target_a)
    _assert_targets_equal(reopened.get("img-001"), target_b)
    assert list(reopened.iter_targets())[0][0] == "img-000"
    assert list(reopened.iter_targets())[1][0] == "img-001"

    with pytest.raises(KeyError):
        reopened.get("missing")

    # Resume semantics: existing keys are skipped, not duplicated.
    assert reopened.put("img-000", target_a) is False
    with pytest.raises(KeyError):
        reopened.put("img-000", target_a, skip_existing=False)
    reopened.close()


def test_cache_resumes_after_abandoned_partial_run(tmp_path):
    root = tmp_path / "resume"
    targets = {
        f"img-{index:03d}": _make_target(index + 10)
        for index in range(5)
    }

    first = TeacherTargetCache(root, shard_size=2)
    assert first.put("img-000", targets["img-000"]) is True
    assert first.put("img-001", targets["img-001"]) is True
    assert first.num_shards == 1
    assert first.put("img-002", targets["img-002"]) is True
    assert first.num_shards == 1  # still buffered
    assert first.abandon() == ["img-002"]
    assert first.num_shards == 1
    assert first.keys() == ["img-000", "img-001"]

    # Simulate a crash during a shard write: a leftover .tmp must be ignored
    # and removed, never published as a shard.
    (root / "shard-000001.pt.tmp").write_bytes(b"incomplete shard")

    second = TeacherTargetCache(root, shard_size=2)
    assert second.keys() == ["img-000", "img-001"]
    assert second.missing(list(targets)) == [
        "img-002",
        "img-003",
        "img-004",
    ]
    for key, target in targets.items():
        second.put(key, target)
    second.close()
    assert second.num_shards == 3

    third = TeacherTargetCache(root, shard_size=2)
    assert third.keys() == list(targets)
    for key, target in targets.items():
        _assert_targets_equal(third.get(key), target)
    manifest = third.manifest()
    assert manifest["num_shards"] == 3
    assert manifest["num_samples"] == 5


def test_cache_ignores_orphan_and_partial_shards(tmp_path):
    root = tmp_path / "orphans"
    target = _make_target(21)

    first = TeacherTargetCache(root, shard_size=8)
    first.put("a", target)
    first.close()
    assert first.num_shards == 1

    # Crash after shard-000001.pt was published but before its JSON manifest:
    # this .pt is an orphan and must never be loaded or overwritten.
    orphan = root / "shard-000001.pt"
    orphan.write_bytes((root / "shard-000000.pt").read_bytes())
    orphan_bytes = orphan.read_bytes()

    # A leftover partial manifest must be ignored as well.
    partial_manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "index": 3,
        "file": "shard-000003.pt",
        "status": "partial",
        "count": 1,
        "keys": ["ghost"],
        "sizes": [[16, 16]],
        "bytes": 0,
    }
    partial_path = root / "shard-000003.json"
    partial_path.write_text(json.dumps(partial_manifest), encoding="utf-8")

    reopened = TeacherTargetCache(root, shard_size=8)
    assert reopened.keys() == ["a"]
    assert reopened.num_shards == 1
    assert reopened.missing(["ghost", "a", "b"]) == ["ghost", "b"]

    # New indices are allocated above every existing .pt/.json file, so the
    # orphan and the partial shard are never overwritten.
    reopened.put("b", target)
    reopened.close()
    assert orphan.read_bytes() == orphan_bytes
    assert partial_path.is_file()
    assert (root / "shard-000004.pt").is_file()

    final = TeacherTargetCache(root, shard_size=8)
    assert final.keys() == ["a", "b"]
    _assert_targets_equal(final.get("b"), target)
    assert final.manifest()["num_samples"] == 2


def test_cache_refuses_non_resume_on_existing_root(tmp_path):
    root = tmp_path / "no-resume"
    with TeacherTargetCache(root, shard_size=4) as cache:
        cache.put("a", _make_target(30))

    with pytest.raises(CacheError, match="resume=False"):
        TeacherTargetCache(root, shard_size=4, resume=False)


# ---------------------------------------------------------------------------
# Real SLBR smoke test
# ---------------------------------------------------------------------------


def test_real_slbr_forward_smoke():
    """One 64x64 forward through the real 2021 SLBR teacher."""
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    try:
        from src.networks.resunet import SLBR
    except ImportError as exc:  # pragma: no cover - legacy dependency mismatch
        pytest.skip(f"legacy SLBR stack unavailable: {exc}")

    args = SimpleNamespace(
        use_refine=True,
        mask_mode="res",
        sim_metric="cos",
        k_center=2,
        project_mode="simple",
        bg_mode="res_mask",
        k_refine=3,
        k_skip_stage=3,
    )

    torch.manual_seed(0)
    teacher = SLBR(
        args=args, shared_depth=1, blocks=3, long_skip=True
    ).eval()
    wrapper = SLBRTeacher(teacher)
    x = torch.rand(1, 3, 64, 64)

    target = wrapper(x)

    assert [t.shape for t in target.mask_logits] == [
        (1, 1, 64, 64),
        (1, 1, 32, 32),
        (1, 1, 16, 16),
    ]
    assert [t.shape for t in target.primary_logits] == [
        (1, 1, 64, 64),
        (1, 1, 32, 32),
        (1, 1, 16, 16),
    ]
    assert target.bottleneck_h8.shape == (1, 256, 8, 8)
    assert target.bg_decoder_h4.shape == (1, 128, 16, 16)
    assert target.refine_bg.shape == (1, 3, 64, 64)
    assert target.coarse_bg.shape == (1, 3, 64, 64)
    for tensor in target.tensors():
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()

    with torch.no_grad():
        _, masks, _ = teacher(x)
    assert len(masks) == 7
    assert masks[0] is masks[6]
    torch.testing.assert_close(
        torch.sigmoid(target.mask_logits[0]), masks[6], rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        torch.sigmoid(target.primary_logits[0]),
        masks[5],
        rtol=0.0,
        atol=1e-6,
    )
