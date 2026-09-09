"""A decoded shard kept on the device between queries — and given back the moment it costs.

The read is the larger half of a device query and the one a repeat need not pay. Measured on a
T4 against a 10 M-row shard of TPC-H `lineitem` projected to six columns: **0.12 s to read it
onto the device and 0.10 s to run q1's filter and eight-way aggregate over it**. A GPU worker
persists between tasks (`gpu_worker_reuse`), the split-to-worker assignment is deterministic,
and Parquet is immutable — so the second query over the same shard, columns and predicate can
read nothing at all.

That is the device counterpart of `dist/executors/scan_read.py`'s worker scan cache, keyed the
same way and for the same reason: split identity, projection and pushed predicate each change
the rows that come back, so all three are in the key.

What is new is what happens when it costs. Device memory is the resource a GPU query actually
runs out of, and a cache holding a neighbour's shard is the difference between fitting and not.
So it is dropped whole on a device out-of-memory, in the process that overflowed, *before* the
driver's subdivision ladder — which answers an overflow by re-reading each piece from storage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from batcher.core.gpu_plan import DfBackend
from batcher.dist.gpu import device_read

pytestmark = pytest.mark.unit


class _Split:
    """A split that identifies itself, which is what makes a shard cacheable."""

    def __init__(self, name: str):
        self.name = name

    def identity(self):
        return self.name


class _Frame:
    """A stand-in device frame: knows its size and whether it was shallow-copied."""

    def __init__(self, nbytes: int, tag: str = ""):
        self._nbytes = nbytes
        self.tag = tag
        self.copies = 0

    def memory_usage(self, deep: bool = False):
        class _Sum:
            def __init__(self, n):
                self._n = n

            def sum(self):
                return self._n

        return _Sum(self._nbytes)

    def copy(self, deep: bool = True):
        self.copies += 1
        out = _Frame(self._nbytes, self.tag)
        out.origin = self
        return out


def _descriptor(*names: str, projection=None, predicate=None) -> dict:
    d: dict = {"splits": [_Split(n) for n in names]}
    if projection is not None:
        d["projection"] = projection
    if predicate is not None:
        d["predicate"] = predicate
    return d


def test_the_budget_is_priced_once_per_worker(monkeypatch):
    """Sizing the cache reads the device's inventory, this process's index, the live telemetry
    and its own resident bytes — the same NVML sequence `prepare_device_memory` measures at
    **414 ms**. Paid per shard it cost more than the read the cache exists to avoid."""
    device_read.reset_device_frame_cache()
    prices = []

    def _price(headroom, tenants=1):
        prices.append((headroom, tenants))
        return 1_000_000_000

    monkeypatch.setattr("batcher.carbonite.accel.visible_device_usable_bytes", _price)
    monkeypatch.setattr("batcher.dist.gpu.resources.task_device_tenants", lambda: 1)
    for _ in range(20):
        device_read._cache_budget_bytes()
    assert len(prices) == 1
    device_read.reset_device_frame_cache()


def test_a_differently_packed_stage_is_priced_again(monkeypatch):
    """A stage packed four to a device gets a different budget than one that had the board."""
    device_read.reset_device_frame_cache()
    prices = []
    monkeypatch.setattr(
        "batcher.carbonite.accel.visible_device_usable_bytes",
        lambda headroom, tenants=1: prices.append(tenants) or 1_000_000_000,
    )
    tenancy = {"n": 1}
    monkeypatch.setattr("batcher.dist.gpu.resources.task_device_tenants", lambda: tenancy["n"])
    device_read._cache_budget_bytes()
    tenancy["n"] = 4
    device_read._cache_budget_bytes()
    assert prices == [1, 4]
    device_read.reset_device_frame_cache()


@pytest.fixture
def cache(monkeypatch):
    """A cache with a known budget and a counted reader."""
    device_read.clear_device_frame_cache()
    reads: list[dict] = []

    def _read(descriptor, be):
        reads.append(descriptor)
        return _Frame(100, tag=str(len(reads)))

    monkeypatch.setattr(device_read, "read_descriptor_on_device", _read)
    monkeypatch.setattr(device_read, "_cache_budget_bytes", lambda: 250)
    yield reads
    device_read.clear_device_frame_cache()


# The backend is only ever passed through to the reader, which these tests replace, so
# pandas is the cheapest thing that satisfies `DfBackend`'s constructor.
BE = DfBackend(pd)


# --- the basic contract ------------------------------------------------------


def test_a_second_read_of_the_same_shard_reads_nothing(cache):
    d = _descriptor("a")
    device_read.cached_device_frame(d, BE)
    device_read.cached_device_frame(_descriptor("a"), BE)
    assert len(cache) == 1
    assert device_read.device_frame_cache_stats()["hits"] == 1


def test_a_different_projection_is_a_different_shard(cache):
    device_read.cached_device_frame(_descriptor("a", projection=["x"]), BE)
    device_read.cached_device_frame(_descriptor("a", projection=["x", "y"]), BE)
    assert len(cache) == 2


def test_a_different_predicate_is_a_different_shard(cache):
    """A pushed predicate prunes row-groups, so it changes the rows the entry holds."""
    device_read.cached_device_frame(_descriptor("a", predicate={"e": "col", "name": "x"}), BE)
    device_read.cached_device_frame(_descriptor("a"), BE)
    assert len(cache) == 2


def test_split_order_does_not_change_the_key(cache):
    device_read.cached_device_frame(_descriptor("a", "b"), BE)
    device_read.cached_device_frame(_descriptor("b", "a"), BE)
    assert len(cache) == 1


def test_a_shard_whose_splits_cannot_identify_themselves_is_not_cached(cache):
    class _Anonymous:
        def identity(self):
            raise RuntimeError("no identity")

    d = {"splits": [_Anonymous()]}
    device_read.cached_device_frame(d, BE)
    device_read.cached_device_frame(d, BE)
    assert len(cache) == 2


def test_a_shard_with_no_splits_is_not_cached(cache):
    device_read.cached_device_frame({"batches": []}, BE)
    device_read.cached_device_frame({"batches": []}, BE)
    assert len(cache) == 2


# --- what is handed back -----------------------------------------------------


def test_the_caller_never_gets_the_cached_frame_itself(cache):
    """`aggregate` and `sort` both add private columns to the frame they are given. Handing out
    the cached object would grow the entry every time it was used."""
    first = device_read.cached_device_frame(_descriptor("a"), BE)
    second = device_read.cached_device_frame(_descriptor("a"), BE)
    assert first is not second


def test_the_copy_is_shallow(cache):
    """A deep copy would allocate the shard again on the device, which is the cost being saved."""
    depths = []

    class _Recording(_Frame):
        def copy(self, deep: bool = True):
            depths.append(deep)
            return _Frame(self._nbytes)

    descriptor = _descriptor("k")
    device_read._FRAME_CACHE[device_read._frame_key(descriptor)] = (100, _Recording(100))
    device_read._FRAME_CACHE_BYTES = 100
    device_read.cached_device_frame(descriptor, BE)
    assert depths == [False]


# --- the budget --------------------------------------------------------------


def test_the_cache_stays_under_its_budget(cache):
    for name in "abcdef":
        device_read.cached_device_frame(_descriptor(name), BE)
    assert device_read.device_frame_cache_stats()["bytes"] <= 250


def test_eviction_is_least_recently_used(cache):
    for name in "abc":  # 300 bytes against a 250-byte budget: `a` goes
        device_read.cached_device_frame(_descriptor(name), BE)
    reads_before = len(cache)
    device_read.cached_device_frame(_descriptor("c"), BE)
    assert len(cache) == reads_before, "the most recent entry must still be warm"
    device_read.cached_device_frame(_descriptor("a"), BE)
    assert len(cache) == reads_before + 1, "the least recent entry must have been evicted"


def test_a_shard_larger_than_the_whole_budget_is_served_uncached(cache, monkeypatch):
    """Holding it would evict everything else to keep one entry the next shard evicts again."""
    monkeypatch.setattr(device_read, "read_descriptor_on_device", lambda d, be: _Frame(10_000))
    device_read.cached_device_frame(_descriptor("big"), BE)
    assert device_read.device_frame_cache_stats()["entries"] == 0


def test_a_zero_budget_turns_the_cache_off(cache, monkeypatch):
    monkeypatch.setattr(device_read, "_cache_budget_bytes", lambda: 0)
    device_read.cached_device_frame(_descriptor("a"), BE)
    device_read.cached_device_frame(_descriptor("a"), BE)
    assert len(cache) == 2
    assert device_read.device_frame_cache_stats()["entries"] == 0


# --- giving it back ----------------------------------------------------------


def test_clearing_reports_the_bytes_released(cache):
    device_read.cached_device_frame(_descriptor("a"), BE)
    assert device_read.clear_device_frame_cache() == 100
    held = device_read.device_frame_cache_stats()
    assert (held["entries"], held["bytes"]) == (0, 0)


def test_hits_and_misses_are_counted(cache):
    """Lifetime counters, so they are read as a delta rather than as an absolute."""
    before = device_read.device_frame_cache_stats()
    device_read.cached_device_frame(_descriptor("counted"), BE)
    device_read.cached_device_frame(_descriptor("counted"), BE)
    after = device_read.device_frame_cache_stats()
    assert after["misses"] - before["misses"] == 1
    assert after["hits"] - before["hits"] == 1


def test_clearing_an_empty_cache_reports_nothing_to_release(cache):
    """The caller retries only when there was something to give back, so this is load-bearing."""
    assert device_read.clear_device_frame_cache() == 0


def test_a_declined_device_read_is_not_cached(cache, monkeypatch):
    """`None` means "use the host reader", and there is nothing to hold."""
    monkeypatch.setattr(device_read, "read_descriptor_on_device", lambda d, be: None)
    assert device_read.cached_device_frame(_descriptor("a"), BE) is None
    assert device_read.device_frame_cache_stats()["entries"] == 0


# --- the type memory a hit must not skip -------------------------------------


class _SchemaSplit(_Split):
    """A split that also reports a schema, which is where the date columns are named."""

    def __init__(self, name: str, schema):
        super().__init__(name)
        self._schema = schema

    def schema(self):
        return self._schema


def test_a_cache_hit_still_registers_the_shards_date_columns(cache, monkeypatch):
    """The defect a hit introduces if this is skipped, and the reason it is worth a test.

    The read is where a backend learns which columns are calendar days — a device frame never
    goes through `from_arrow`. A hit skips the read, so without this the shard's dates come back
    out of `to_arrow` as **timestamps**: the right values in the wrong column. Measured on
    TPC-H q3 at sf10, `o_orderdate` arrived as `timestamp[s]` against the engine's
    `date32[day]`, the schema contract refused the whole device result, and the CPU engine
    answered a query the device had already computed.
    """
    import pyarrow as pa

    schema = pa.schema([pa.field("d", pa.date32()), pa.field("v", pa.int64())])
    descriptor = {"splits": [_SchemaSplit("dated", schema)]}
    remembered = []

    class _Recording:
        """`DfBackend` uses `__slots__`, so the spy is a stand-in rather than a patch."""

        is_gpu = True

        def remember_dates(self, schema):
            remembered.append(list(schema.names))

    be = _Recording()
    device_read.cached_device_frame(descriptor, be)  # miss: the real reader registers them
    remembered.clear()
    device_read.cached_device_frame(descriptor, be)  # hit: this must register them too
    assert remembered == [["d", "v"]], "a cache hit skipped the date registration"


def test_registering_dates_survives_a_split_that_will_not_describe_itself(cache):
    """A shard with no schema must cost the registration, not the query."""
    device_read.remember_source_dates({"splits": [_Split("x")]}, BE)
    device_read.remember_source_dates({"batches": []}, BE)
