"""Broken-record tolerance for the distributed scan (`distributed.on_read_error`).

A single corrupt file / unreadable row-group otherwise fails the whole cluster job. With
``on_read_error="skip"`` the failing split is skipped in isolation and the scan continues
over its healthy siblings; the count of skipped splits is recorded so the data loss is
observable rather than silent. The policy travels with each partition manifest, so it
reaches every worker without shipping config. Default ``"error"`` keeps fail-fast.
"""

from __future__ import annotations

import dataclasses
import pickle

import pyarrow as pa
import pytest

from batcher.dist.executors import partition_io, scan_read

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([("x", pa.int64())])


class _FakeSplit:
    """A minimal split whose read either yields one batch or raises (a corrupt split).

    Module-level (not a closure) so a manifest carrying it can be pickled, exactly as
    `_partition_source` writes and `read_partition` reads it.
    """

    def __init__(self, value: int | None, *, broken: bool = False) -> None:
        self.value = value
        self.broken = broken

    def schema(self) -> pa.Schema:
        return _SCHEMA

    def read(self, projection=None) -> list[pa.RecordBatch]:
        if self.broken:
            raise OSError("corrupt page")
        return [pa.record_batch([pa.array([self.value], pa.int64())], schema=_SCHEMA)]

    def row_count(self) -> int | None:
        return 1

    def identity(self) -> str:
        return f"fake:{self.value}:{'broken' if self.broken else 'ok'}"


class _FakeSource:
    """A splittable source of `_FakeSplit`s (enough to skip the eager whole-source path)."""

    def __init__(self, splits: list[_FakeSplit]) -> None:
        self._splits = splits

    def splits(self, target_size: int | None = None) -> list[_FakeSplit]:
        return self._splits

    def schema(self) -> pa.Schema:
        return _SCHEMA


@pytest.fixture(autouse=True)
def _reset_skip_counter():
    scan_read._SKIPPED_SPLITS = 0
    yield
    scan_read._SKIPPED_SPLITS = 0


def _rows(batches: list[pa.RecordBatch]) -> list[int]:
    return pa.Table.from_batches(batches, schema=_SCHEMA).column("x").to_pylist()


# --- the reader skip logic (sequential + prefetch pool) ---------------------------


def test_prefetch_sequential_skips_broken_split():
    splits = [_FakeSplit(1), _FakeSplit(None, broken=True), _FakeSplit(3)]
    # depth=1 forces the sequential path.
    out = list(scan_read._prefetch_split_reads(splits, None, None, 1, skip_errors=True))
    assert sorted(_rows(out)) == [1, 3]
    assert scan_read.skipped_splits() == 1


def test_prefetch_pool_skips_broken_split():
    splits = [_FakeSplit(i) for i in range(10)]
    splits[4] = _FakeSplit(None, broken=True)
    splits[7] = _FakeSplit(None, broken=True)
    # depth>1 with >1 split forces the thread-pool prefetch path.
    out = list(scan_read._prefetch_split_reads(splits, None, None, 4, skip_errors=True))
    assert sorted(_rows(out)) == [0, 1, 2, 3, 5, 6, 8, 9]
    assert scan_read.skipped_splits() == 2


def test_default_error_propagates():
    splits = [_FakeSplit(1), _FakeSplit(None, broken=True)]
    with pytest.raises(OSError, match="corrupt page"):
        list(scan_read._prefetch_split_reads(splits, None, None, 1, skip_errors=False))


def test_read_split_batches_skip_routes_through_isolation():
    splits = [_FakeSplit(1), _FakeSplit(None, broken=True), _FakeSplit(2)]
    out = list(scan_read._read_split_batches(splits, None, None, "skip"))
    assert sorted(_rows(out)) == [1, 2]
    assert scan_read.skipped_splits() == 1
    # And the default policy still fails fast.
    with pytest.raises(OSError):
        list(scan_read._read_split_batches(splits, None, None, "error"))


# --- the manifest round-trip (driver writes policy, worker honors it) -------------


def test_read_partition_manifest_honors_skip(tmp_path):
    manifest = {
        "splits": [_FakeSplit(1), _FakeSplit(None, broken=True), _FakeSplit(5)],
        "projection": None,
        "predicate": None,
        "on_read_error": "skip",
    }
    path = str(tmp_path / "P_part_0.splits")
    with open(path, "wb") as fh:
        pickle.dump(manifest, fh)
    out = partition_io.read_partition(path)
    assert sorted(_rows(out)) == [1, 5]
    assert scan_read.skipped_splits() == 1


def test_read_partition_manifest_defaults_to_error(tmp_path):
    # A manifest with no policy key (older writer) must default to fail-fast.
    manifest = {
        "splits": [_FakeSplit(1), _FakeSplit(None, broken=True)],
        "projection": None,
        "predicate": None,
    }
    path = str(tmp_path / "P_part_0.splits")
    with open(path, "wb") as fh:
        pickle.dump(manifest, fh)
    with pytest.raises(OSError):
        partition_io.read_partition(path)


# --- the driver embeds the configured policy into the manifest --------------------


def test_partition_source_embeds_policy(tmp_path):
    from batcher.config import Config, config_context

    src = _FakeSource([_FakeSplit(1), _FakeSplit(2), _FakeSplit(3)])
    cfg = Config()
    cfg = cfg.replace(distributed=dataclasses.replace(cfg.distributed, on_read_error="skip"))
    with config_context(cfg):
        paths = partition_io._partition_source(src, workers=2, work_dir=str(tmp_path))
    manifests = [p for p in paths if p.endswith(".splits")]
    assert manifests, "expected at least one split-manifest partition"
    for p in manifests:
        with open(p, "rb") as fh:
            assert pickle.load(fh)["on_read_error"] == "skip"


# --- config validation ------------------------------------------------------------


def test_config_rejects_bad_on_read_error():
    from batcher._internal.errors import ConfigError
    from batcher.config import Config

    cfg = Config()
    cfg = cfg.replace(distributed=dataclasses.replace(cfg.distributed, on_read_error="nonsense"))
    with pytest.raises(ConfigError, match="on_read_error"):
        cfg.validate()


def test_config_accepts_valid_on_read_error():
    from batcher.config import Config

    for policy in ("error", "skip"):
        cfg = Config()
        cfg = cfg.replace(distributed=dataclasses.replace(cfg.distributed, on_read_error=policy))
        assert cfg.validate().distributed.on_read_error == policy
