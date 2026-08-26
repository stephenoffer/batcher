"""The shared result cache's one correctness rule, and its containment of failure.

A process cache can only ever be cold. A *shared* cache can be wrong: it outlives the run
that filled it, so an entry whose inputs have since changed would serve the previous run's
rows and nothing would raise. `shareable_key` is the whole defence, and these cases pin
it — that it declines in-memory data, declines a source that cannot version itself, and
changes when a file's content changes.

The rest pins the other half of the contract: every failure a store can produce must
surface as a miss, because a cache that is down has to make queries slower rather than
make them fail.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.carbonite.cache_shared import SharedResultCache, entry_digest, shareable_key
from batcher.carbonite.cache_shared.base import SharedCacheError

pytestmark = pytest.mark.unit


class _DictCache:
    """An in-process `SharedCache`, standing in for Redis or RocksDB.

    The backends are thin translations of `get`/`put`/`delete` onto a driver; everything
    above them — serialization, the key rule, error containment, the counters — is
    driver-independent, and testing it against a dict is what keeps that logic covered on
    a CI machine with neither driver installed.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.ttls: list[int | None] = []

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    def put(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        self.store[key] = value
        self.ttls.append(ttl_seconds)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def close(self) -> None:
        pass


class _BrokenCache(_DictCache):
    """A store that fails every operation, the way an unreachable server does."""

    def get(self, key: str) -> bytes | None:
        raise SharedCacheError("connection refused")

    def put(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        raise SharedCacheError("connection refused")

    def delete(self, key: str) -> None:
        raise SharedCacheError("connection refused")


def _table() -> pa.Table:
    return pa.table({"i": pa.array([1, 2, 3], pa.int32()), "s": pa.array(["a", None, "c"])})


# --- the key rule -------------------------------------------------------------------


def test_in_memory_data_is_never_shareable():
    import batcher as bt

    # In-memory batches are keyed by schema and row count, which two different relations
    # collide on. There is no version to invalidate and no identity to trust.
    ds = bt.from_pydict({"x": [1, 2, 3]})
    assert shareable_key("sig", list(ds._sources), "") is None


def test_a_file_source_is_shareable_and_its_key_tracks_the_content(tmp_path):
    import batcher as bt

    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": [1, 2, 3]}), path)
    before = shareable_key("sig", list(bt.read.parquet(path)._sources), "")
    assert before is not None

    pq.write_table(pa.table({"x": [1, 2, 3, 4, 5]}), path)
    after = shareable_key("sig", list(bt.read.parquet(path)._sources), "")
    assert after is not None
    # The bug this guards is the whole reason a shared cache needs a version token: the
    # path is unchanged, so an identity-only key would serve the three-row result forever.
    assert after != before


def test_the_key_separates_the_plan_and_the_tenant(tmp_path):
    import batcher as bt

    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": [1, 2, 3]}), path)
    sources = list(bt.read.parquet(path)._sources)
    base = shareable_key("sig-a", sources, "")
    assert shareable_key("sig-b", sources, "") != base, "a different plan is a different key"
    assert shareable_key("sig-a", sources, "t=acme") != base, "a tenant must not share rows"


def test_a_source_that_cannot_version_itself_declines():
    class _Unversioned:
        stable_stats_identity = True

        def identity(self) -> str:
            return "custom:table"

    # No `stats_version`: unversioned is indistinguishable from unchanged, so the only
    # safe answer is to recompute.
    assert shareable_key("sig", [_Unversioned()], "") is None


def test_a_source_whose_version_call_fails_declines():
    class _Angry:
        stable_stats_identity = True

        def identity(self) -> str:
            return "custom:table"

        def stats_version(self) -> str:
            raise RuntimeError("catalog unreachable")

    assert shareable_key("sig", [_Angry()], "") is None


def test_entry_digest_is_stable_and_separating():
    assert entry_digest("a", "b") == entry_digest("a", "b")
    assert entry_digest("a", "b") != entry_digest("b", "a")
    # The separator must not be forgeable from the parts, or ("ab", "c") and ("a", "bc")
    # would name the same entry.
    assert entry_digest("ab", "c") != entry_digest("a", "bc")
    assert len(entry_digest("a")) == 32


# --- the store: serialization, containment, measurement -----------------------------


def test_a_result_round_trips_through_the_store_exactly():
    cache = SharedResultCache(_DictCache())
    original = _table()
    cache.put("k", original)
    got = cache.get("k")
    assert got is not None
    # Exact on schema as well as values: a shared entry crossing a process boundary that
    # widened an int32 to int64 would be a wrong answer under invariant #7, not a
    # formatting difference.
    assert got.equals(original)


def test_an_empty_result_round_trips_with_its_schema():
    cache = SharedResultCache(_DictCache())
    empty = pa.table({"x": pa.array([], pa.int32())})
    cache.put("k", empty)
    got = cache.get("k")
    assert got is not None and got.num_rows == 0
    assert got.schema.equals(empty.schema)


def test_a_miss_is_counted_and_returns_none():
    cache = SharedResultCache(_DictCache())
    assert cache.get("absent") is None
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hits"] == 0


def test_an_unreachable_store_is_a_miss_not_an_error():
    cache = SharedResultCache(_BrokenCache())
    assert cache.get("k") is None  # must not raise
    cache.put("k", _table())  # must not raise
    stats = cache.stats()
    assert stats["errors"] == 2
    assert stats["writes"] == 0
    # A failed read is an error, not a miss: the two call for different responses, and
    # counting a dead server as a cold cache is how an outage stays invisible.
    assert stats["misses"] == 0


def test_a_corrupt_entry_is_dropped_rather_than_re_read():
    backend = _DictCache()
    backend.store["k"] = b"not an arrow ipc stream"
    cache = SharedResultCache(backend)
    assert cache.get("k") is None
    assert "k" not in backend.store, "an undecodable entry must not be left to be re-fetched"
    assert cache.stats()["errors"] == 1


def test_the_ttl_is_applied_to_every_write():
    backend = _DictCache()
    SharedResultCache(backend, ttl_seconds=60).put("k", _table())
    assert backend.ttls == [60]


def test_no_ttl_is_passed_through_as_none():
    backend = _DictCache()
    SharedResultCache(backend).put("k", _table())
    assert backend.ttls == [None]


def test_stats_report_the_bytes_actually_moved():
    cache = SharedResultCache(_DictCache())
    cache.put("k", _table())
    cache.get("k")
    stats = cache.stats()
    assert stats["bytes_written"] > 0
    assert stats["bytes_read"] == stats["bytes_written"]
    assert stats["hit_rate"] == 1.0
