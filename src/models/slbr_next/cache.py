"""Sharded, resumable cache for Stage 1 teacher targets (spec §4.1, §5.4).

The cache writes one ``TeacherTarget`` per caller-supplied **non-empty string
key** and groups samples into atomic shards instead of writing one small file
per image. It uses only the standard library plus ``torch``.

On-disk layout
--------------
::

    <root>/
        manifest.json                 # global index, authoritative for shard list
        shard-000000.pt               # {"format", "version", "index", "keys", "targets"}
        shard-000000.json             # per-shard manifest, authoritative for keys
        shard-000001.pt
        shard-000001.json
        ...

* ``shard-XXXXXX.pt`` is a ``torch.save`` payload whose ``targets`` list is
  made of :meth:`TeacherTarget.to_payload` dictionaries. There is no custom
  class in the payload, so it loads with ``weights_only=True``.
* ``shard-XXXXXX.json`` records ``status = "complete"``, the shard index, the
  ordered keys, per-sample ``(height, width)`` sizes and the ``.pt`` byte
  size. Only a ``complete`` manifest makes a shard visible.
* ``manifest.json`` mirrors the completed shard list for tooling (T6) but the
  per-shard manifests are rescanned on open. It is rebuilt if stale or
  missing.

Crash safety and resume semantics
---------------------------------
Every file is written to ``*.tmp``, flushed and ``fsync``-ed, then published
with :func:`os.replace`. A shard is written in this order:

1. ``shard-XXXXXX.pt.tmp`` -> ``shard-XXXXXX.pt``
2. ``shard-XXXXXX.json.tmp`` -> ``shard-XXXXXX.json``

Therefore a visible JSON implies a complete ``.pt``. A crash can leave:

* a ``.tmp`` file - removed/ignored on the next open;
* a ``.pt`` orphan without its JSON - ignored, and never overwritten because
  new shard indices are allocated above every existing ``.pt``/``.json``;
* a complete shard whose ``manifest.json`` entry was not yet written - found
  by rescanning the per-shard JSON files.

Resume is keyed by sample key, not by position. Open the cache with the same
root and call ``missing(keys)`` to get the keys that still need generation;
``put(key, target)`` with the default ``skip_existing=True`` silently skips
keys that are already present. This also makes it safe to loop over the whole
dataset and let the cache discard completed work. Samples buffered in a
writer that crashes before ``flush()``/``close()`` are lost and regenerated;
they were never part of a visible shard.

Typical use::

    cache = TeacherTargetCache(cache_root, shard_size=32)
    todo = cache.missing(all_keys)
    for key in todo:
        target = extractor(load_crop(key))   # SLBRTeacher
        cache.put(key, target)
        if len(cache) % 500 == 0:
            print(cache.num_shards, "shards")
    cache.close()

    # Later / on another worker:
    cache = TeacherTargetCache(cache_root)
    target = cache.get("train-000123")       # TeacherTarget, exact tensors

The cache is single-writer. It does not lock files; run one generator process
per root (a reader may open it after writing completes).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch

from .teacher import TeacherTarget

__all__ = [
    "CACHE_FORMAT",
    "CACHE_VERSION",
    "CacheError",
    "TeacherTargetCache",
]

#: On-disk schema identifier for both the global and per-shard manifests.
CACHE_FORMAT = "slbr-next-teacher-cache"
CACHE_VERSION = 1

_SHARD_NAME_RE = re.compile(r"^shard-(\d+)\.(pt|json)$")
_ANY_SHARD_NAME_RE = re.compile(r"^shard-(\d+)\.(?:pt|json)(?:\.tmp)?$")


class CacheError(RuntimeError):
    """Raised when a cache is inconsistent or cannot be resumed safely."""


@dataclass(frozen=True)
class _ShardInfo:
    index: int
    file: str
    manifest: str
    keys: tuple[str, ...]
    sizes: tuple[tuple[int, int], ...]
    bytes: int


def _shard_stem(index: int) -> str:
    return f"shard-{index:06d}"


def _validate_key(key: object) -> str:
    if not isinstance(key, str):
        raise TypeError(
            f"cache keys must be strings, got {type(key).__name__}"
        )
    if not key:
        raise ValueError("cache keys must be non-empty strings")
    if "\x00" in key:
        raise ValueError("cache keys must not contain NUL characters")
    return key


def _atomic_torch_save(payload: object, path: Path) -> None:
    """Write ``payload`` to ``path`` atomically via a sibling ``.tmp``."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_dump(payload: object, path: Path) -> None:
    """Write JSON to ``path`` atomically via a sibling ``.tmp``."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_torch(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch without weights_only
        try:
            return torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - corrupt file path
            raise CacheError(
                f"failed to load cache shard {path}: {exc}"
            ) from exc
    except Exception as exc:  # pragma: no cover - corrupt file path
        raise CacheError(f"failed to load cache shard {path}: {exc}") from exc


class TeacherTargetCache:
    """Sharded, resumable ``TeacherTarget`` store.

    Args:
        root: Cache directory. Created if missing.
        shard_size: Maximum number of targets per shard. Larger shards mean
            fewer files and faster sequential reads; smaller shards mean less
            work is lost when a writer is interrupted. A target at 256x256 is
            roughly 5-6 MiB in FP32, so the default 32 targets is ~180 MiB per
            shard.
        resume: When ``True`` (default), scan and reuse completed shards.
            When ``False``, refuse to open a non-empty cache rather than
            silently overwriting it.

    The object is a context manager::

        with TeacherTargetCache(root, shard_size=64) as cache:
            cache.put("train-000001", target)

    Reads (``get``, ``keys``, ``iter_targets``) remain available after
    ``close()``. Writes after ``close()`` raise ``RuntimeError``.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        shard_size: int = 32,
        resume: bool = True,
    ) -> None:
        if (
            isinstance(shard_size, bool)
            or not isinstance(shard_size, int)
            or shard_size <= 0
        ):
            raise ValueError(
                f"shard_size must be a positive int, got {shard_size!r}"
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.resume = bool(resume)

        self._closed = False
        self._buffer: dict[str, TeacherTarget] = {}
        self._shards: list[_ShardInfo] = []
        self._shard_by_index: dict[int, _ShardInfo] = {}
        self._key_to_location: dict[str, tuple[int, int]] = {}
        self._cached_index: int | None = None
        self._cached_payload: dict[str, object] | None = None

        self._clean_temp_files()
        self._scan(resume=self.resume)
        self._write_global_manifest()

    # ------------------------------------------------------------------
    # Opening / scanning
    # ------------------------------------------------------------------

    def _clean_temp_files(self) -> None:
        # ``*.tmp`` files are incomplete by construction; remove leftovers
        # from a previous crash so they cannot be mistaken for data.
        for temporary in self.root.glob("*.tmp"):
            try:
                temporary.unlink()
            except OSError:
                pass

    def _scan(self, *, resume: bool) -> None:
        manifest_paths: list[tuple[int, Path]] = []
        shard_like_paths: list[Path] = []
        for path in self.root.iterdir():
            if not path.is_file():
                continue
            match = _SHARD_NAME_RE.match(path.name)
            if match is not None and match.group(2) == "json":
                manifest_paths.append((int(match.group(1)), path))
            if _ANY_SHARD_NAME_RE.match(path.name):
                shard_like_paths.append(path)

        if not resume and shard_like_paths:
            raise CacheError(
                f"resume=False but {self.root} already contains shard files; "
                "refusing to overwrite an existing cache"
            )

        manifest_paths.sort()
        for expected_index, manifest_path in manifest_paths:
            data = self._read_shard_manifest(manifest_path, expected_index)
            if data.get("status") != "complete":
                continue

            file_name = data.get("file")
            expected_file = _shard_stem(expected_index) + ".pt"
            if file_name != expected_file:
                raise CacheError(
                    f"{manifest_path} names shard file {file_name!r}, "
                    f"expected {expected_file!r}"
                )
            payload_path = self.root / expected_file
            if not payload_path.is_file():
                raise CacheError(
                    f"{manifest_path} is complete but {payload_path} is "
                    "missing"
                )

            keys_data = data.get("keys")
            if not isinstance(keys_data, list) or not all(
                isinstance(key, str) and key for key in keys_data
            ):
                raise CacheError(
                    f"{manifest_path} has an invalid 'keys' list"
                )
            keys = tuple(keys_data)
            count = data.get("count", len(keys))
            if count != len(keys):
                raise CacheError(
                    f"{manifest_path} declares count={count} but lists "
                    f"{len(keys)} keys"
                )

            sizes_data = data.get("sizes", [])
            if not isinstance(sizes_data, list) or len(sizes_data) != len(
                keys
            ):
                raise CacheError(
                    f"{manifest_path} has {len(keys)} keys but "
                    f"{len(sizes_data) if isinstance(sizes_data, list) else '?'} "
                    "sizes"
                )
            sizes: list[tuple[int, int]] = []
            for size in sizes_data:
                if (
                    not isinstance(size, (list, tuple))
                    or len(size) != 2
                    or not all(isinstance(value, int) for value in size)
                ):
                    raise CacheError(
                        f"{manifest_path} has an invalid size entry: "
                        f"{size!r}"
                    )
                sizes.append((int(size[0]), int(size[1])))

            expected_bytes = data.get("bytes")
            actual_bytes = payload_path.stat().st_size
            if expected_bytes is not None and expected_bytes != actual_bytes:
                raise CacheError(
                    f"{payload_path} is {actual_bytes} bytes but its "
                    f"manifest records {expected_bytes}; the shard is "
                    "corrupt"
                )

            info = _ShardInfo(
                index=expected_index,
                file=expected_file,
                manifest=manifest_path.name,
                keys=keys,
                sizes=tuple(sizes),
                bytes=actual_bytes,
            )
            self._shards.append(info)
            self._shard_by_index[expected_index] = info
            for offset, key in enumerate(keys):
                previous = self._key_to_location.get(key)
                if previous is not None:
                    raise CacheError(
                        f"duplicate cache key {key!r} in shards "
                        f"{previous[0]} and {expected_index}"
                    )
                self._key_to_location[key] = (expected_index, offset)

    @staticmethod
    def _read_shard_manifest(
        path: Path, expected_index: int
    ) -> dict[str, object]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheError(
                f"failed to read cache manifest {path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise CacheError(f"{path} must contain a JSON object")
        if data.get("format") != CACHE_FORMAT:
            raise CacheError(
                f"{path} has format {data.get('format')!r}, expected "
                f"{CACHE_FORMAT!r}"
            )
        if data.get("version") != CACHE_VERSION:
            raise CacheError(
                f"{path} has version {data.get('version')!r}, expected "
                f"{CACHE_VERSION}"
            )
        if data.get("index") != expected_index:
            raise CacheError(
                f"{path} declares index {data.get('index')!r}, expected "
                f"{expected_index}"
            )
        return data

    # ------------------------------------------------------------------
    # Manifest writing
    # ------------------------------------------------------------------

    def _write_global_manifest(self) -> None:
        payload = {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "shard_size": self.shard_size,
            "num_shards": len(self._shards),
            "num_samples": sum(len(info.keys) for info in self._shards),
            "shards": [
                {
                    "index": info.index,
                    "file": info.file,
                    "manifest": info.manifest,
                    "count": len(info.keys),
                    "bytes": info.bytes,
                }
                for info in self._shards
            ],
        }
        _atomic_json_dump(payload, self.root / "manifest.json")

    def _next_index(self) -> int:
        next_index = 0
        for path in self.root.iterdir():
            if not path.is_file():
                continue
            match = _ANY_SHARD_NAME_RE.match(path.name)
            if match is not None:
                next_index = max(next_index, int(match.group(1)) + 1)
        return next_index

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def put(
        self,
        key: str,
        target: TeacherTarget,
        *,
        skip_existing: bool = True,
    ) -> bool:
        """Add one target under ``key``.

        Args:
            key: Non-empty sample key.
            target: A :class:`TeacherTarget`.
            skip_existing: When ``True`` (default), an existing key is skipped
                silently and ``False`` is returned. When ``False``, an
                existing key raises ``KeyError``.

        Returns:
            ``True`` if the target was buffered, ``False`` if it was already
            present and skipped.
        """
        self._check_writable()
        _validate_key(key)
        if not isinstance(target, TeacherTarget):
            raise TypeError(
                "target must be a TeacherTarget, got "
                f"{type(target).__name__}"
            )
        if self.contains(key):
            if skip_existing:
                return False
            raise KeyError(f"cache key {key!r} already exists")

        self._buffer[key] = target
        if len(self._buffer) >= self.shard_size:
            self.flush()
        return True

    # ``write`` reads more naturally from a generation loop; keep it as an
    # explicit alias rather than a second implementation.
    def write(
        self,
        key: str,
        target: TeacherTarget,
        *,
        skip_existing: bool = True,
    ) -> bool:
        """Alias of :meth:`put`."""
        return self.put(
            key, target, skip_existing=skip_existing
        )

    def flush(self) -> int:
        """Publish the current buffer as a complete shard.

        Returns the number of samples written. Calling ``flush`` with an
        empty buffer is a no-op. The shard becomes visible only after both
        the ``.pt`` payload and its JSON manifest have been atomically
        replaced.
        """
        self._check_writable()
        if not self._buffer:
            return 0

        index = self._next_index()
        keys = list(self._buffer)
        targets = list(self._buffer.values())
        stem = _shard_stem(index)
        file_name = stem + ".pt"
        manifest_name = stem + ".json"
        payload_path = self.root / file_name

        payload = {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "index": index,
            "keys": keys,
            "targets": [target.to_payload() for target in targets],
        }
        _atomic_torch_save(payload, payload_path)

        # If this JSON write fails, the .pt stays as an unregistered orphan.
        # The buffer is retained, so a retry writes a later shard index and
        # never overwrites the orphan or publishes partial data.
        size = payload_path.stat().st_size
        manifest = {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "index": index,
            "file": file_name,
            "status": "complete",
            "count": len(keys),
            "keys": keys,
            "sizes": [
                [target.height, target.width] for target in targets
            ],
            "bytes": size,
        }
        _atomic_json_dump(manifest, self.root / manifest_name)

        info = _ShardInfo(
            index=index,
            file=file_name,
            manifest=manifest_name,
            keys=tuple(keys),
            sizes=tuple((target.height, target.width) for target in targets),
            bytes=size,
        )
        self._shards.append(info)
        self._shard_by_index[index] = info
        for offset, key in enumerate(keys):
            self._key_to_location[key] = (index, offset)
        self._buffer.clear()
        self._cached_index = None
        self._cached_payload = None
        self._write_global_manifest()
        return len(keys)

    def abandon(self) -> list[str]:
        """Discard unflushed buffered samples without writing a shard.

        This is the explicit way to simulate a writer crash in tests or to
        drop a failed batch. It returns the discarded keys. Completed shards
        are untouched.
        """
        discarded = list(self._buffer)
        self._buffer.clear()
        return discarded

    def close(self) -> None:
        """Flush buffered samples and mark the writer closed.

        Safe to call more than once. Reading remains possible afterwards.
        """
        if self._closed:
            return
        self.flush()
        self._write_global_manifest()
        self._closed = True

    def _check_writable(self) -> None:
        if self._closed:
            raise RuntimeError("teacher target cache is closed")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def contains(self, key: str) -> bool:
        _validate_key(key)
        return key in self._buffer or key in self._key_to_location

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self._buffer or key in self._key_to_location

    def missing(self, keys: Iterable[str]) -> list[str]:
        """Return ``keys`` that are not present, preserving their order."""
        missing: list[str] = []
        for key in keys:
            _validate_key(key)
            if key not in self._buffer and key not in self._key_to_location:
                missing.append(key)
        return missing

    def keys(self) -> list[str]:
        """Return all keys in shard order, then any buffered keys."""
        result = [
            key for info in self._shards for key in info.keys
        ]
        result.extend(self._buffer)
        return result

    def get(self, key: str) -> TeacherTarget:
        """Load one target by sample key.

        Raises:
            KeyError: If the key is not present.
            CacheError: If the shard payload is missing or inconsistent.
        """
        _validate_key(key)
        if key in self._buffer:
            return self._buffer[key]
        location = self._key_to_location.get(key)
        if location is None:
            raise KeyError(f"cache key {key!r} not found")
        index, offset = location
        payload = self._load_shard_payload(index)
        targets = payload.get("targets")
        if not isinstance(targets, list):
            raise CacheError(
                f"shard {index} has no 'targets' list"
            )
        return TeacherTarget.from_payload(targets[offset])

    def load(self, key: str) -> TeacherTarget:
        """Alias of :meth:`get`."""
        return self.get(key)

    __getitem__ = get

    def iter_targets(self) -> Iterator[tuple[str, TeacherTarget]]:
        """Yield ``(key, target)`` in shard order, then buffered samples."""
        for info in self._shards:
            payload = self._load_shard_payload(info.index)
            targets = payload.get("targets")
            if not isinstance(targets, list) or len(targets) != len(
                info.keys
            ):
                raise CacheError(
                    f"shard {info.index} payload does not match its "
                    "manifest key count"
                )
            for key, sample in zip(info.keys, targets):
                yield key, TeacherTarget.from_payload(sample)
        for key, target in self._buffer.items():
            yield key, target

    def __iter__(self) -> Iterator[tuple[str, TeacherTarget]]:
        return self.iter_targets()

    def _load_shard_payload(self, index: int) -> dict[str, object]:
        if self._cached_index == index and self._cached_payload is not None:
            return self._cached_payload
        info = self._shard_by_index.get(index)
        if info is None:
            raise CacheError(f"shard {index} is not registered")
        payload = _load_torch(self.root / info.file)
        if not isinstance(payload, dict):
            raise CacheError(
                f"shard {index} payload must be a dict, got "
                f"{type(payload).__name__}"
            )
        if payload.get("format") != CACHE_FORMAT:
            raise CacheError(
                f"shard {index} has payload format "
                f"{payload.get('format')!r}"
            )
        if payload.get("version") != CACHE_VERSION:
            raise CacheError(
                f"shard {index} has payload version "
                f"{payload.get('version')!r}"
            )
        if payload.get("index") != index:
            raise CacheError(
                f"shard {index} payload declares index "
                f"{payload.get('index')!r}"
            )
        keys = payload.get("keys")
        targets = payload.get("targets")
        if not isinstance(keys, list) or tuple(keys) != info.keys:
            raise CacheError(
                f"shard {index} payload keys do not match its manifest"
            )
        if not isinstance(targets, list) or len(targets) != len(info.keys):
            raise CacheError(
                f"shard {index} payload target count does not match its "
                "manifest"
            )
        self._cached_index = index
        self._cached_payload = payload
        return payload

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def num_shards(self) -> int:
        """Number of completed, published shards."""
        return len(self._shards)

    @property
    def shard_files(self) -> tuple[Path, ...]:
        """Paths of the completed ``.pt`` shards in shard order."""
        return tuple(self.root / info.file for info in self._shards)

    def manifest(self) -> dict[str, object]:
        """Read the current ``manifest.json`` from disk."""
        data = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if not isinstance(data, dict):
            raise CacheError("manifest.json must contain a JSON object")
        return data

    def __len__(self) -> int:
        return len(self._key_to_location) + len(self._buffer)

    def __enter__(self) -> "TeacherTargetCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        # On error, leave buffered samples unwritten (a crash-like exit) so
        # completed shards stay authoritative.
