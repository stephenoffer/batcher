"""Out-of-core breakers: source pushdown, bounded scratch disk, and typed empty results.

Out-of-core is where reading a column you do not need costs the most — it is decoded,
chunked, hash-partitioned, compressed, written to disk, and read back — and where holding a
bucket after its reader is done doubles peak scratch. Neither shows up in a result
comparison, so each is asserted against what the run actually asked the source and the
spill store for.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.dataset.frame import Dataset
from batcher.carbonite.spill.store import TieredSpillStore
from batcher.dist import spill_breakers as br
from batcher.dist.global_window import stream_spilling_global_window
from batcher.dist.spill import execute_spilling_aggregate, spill_collect
from batcher.plan.logical import Scan
from batcher.plan.schema import SchemaRef


class _RecordingSource:
    """A source that reports the projection of every read it is asked for."""

    bounded = True

    def __init__(self, rows: int = 400, columns: int = 5) -> None:
        self.projections: list[list[str] | None] = []
        self._data = {f"c{i}": [(i * 1000 + r) % 97 for r in range(rows)] for i in range(columns)}
        self._data["k"] = [f"g{r % 8}" for r in range(rows)]
        self._table = pa.table(self._data)

    def schema(self) -> pa.Schema:
        return self._table.schema

    def row_count(self) -> int | None:
        return self._table.num_rows

    def identity(self) -> str:
        return "recording"

    def splits(self, target_size: int | None = None):
        return []

    def read(self, projection=None):
        return list(self.iter_batches(projection))

    def iter_batches(self, projection=None):
        self.projections.append(list(projection) if projection is not None else None)
        table = self._table.select(projection) if projection is not None else self._table
        yield from table.to_batches(max_chunksize=64)


def _ds(source: _RecordingSource) -> Dataset:
    return Dataset(Scan(0, SchemaRef.from_arrow(source.schema())), [source])


def _narrowed(source: _RecordingSource) -> set[str]:
    reads = [p for p in source.projections if p is not None]
    assert reads, f"every read took all columns: {source.projections}"
    return set(reads[0])


# --------------------------------------------------------------------------
# Source pushdown through each partition phase.
# --------------------------------------------------------------------------
def test_a_spilling_aggregate_reads_only_the_columns_its_plan_touches():
    """The `projection` parameter on `_iter_spill_morsels` existed and no partition phase
    ever passed one, so every out-of-core run decoded, partitioned, compressed, wrote, and
    re-read columns the plan never referenced."""
    source = _RecordingSource()
    plan = _ds(source).group_by("k").agg(total=bt.col("c0").sum())._plan
    out = execute_spilling_aggregate(plan, [source], num_partitions=4)
    assert out.num_rows == 8
    assert _narrowed(source) == {"k", "c0"}


def test_a_spilling_sort_reads_only_the_columns_its_plan_touches():
    source = _RecordingSource()
    plan = _ds(source).select("k", "c1").sort("c1")._plan
    out = br.execute_spilling_sort(plan, [source], num_partitions=4)
    assert out.num_rows == 400
    assert _narrowed(source) == {"k", "c1"}


def test_a_spilling_window_reads_only_the_columns_its_plan_touches():
    source = _RecordingSource()
    plan = (
        _ds(source)
        .select("k", "c2")
        .with_columns(r=bt.col("c2").sum().over(partition_by="k"))
        ._plan
    )
    batches = list(br.stream_spilling_window(plan, [source], num_partitions=4))
    assert sum(b.num_rows for b in batches) == 400
    assert _narrowed(source) == {"k", "c2"}


def test_the_pushdown_does_not_change_the_out_of_core_result():
    """The projection is a cost decision, so spilled must still equal in-memory."""
    spilled = execute_spilling_aggregate(
        _ds(_RecordingSource()).group_by("k").agg(total=bt.col("c0").sum())._plan,
        [_RecordingSource()],
        num_partitions=4,
    )
    in_memory = _ds(_RecordingSource()).group_by("k").agg(total=bt.col("c0").sum()).collect()
    assert spilled.sort_by("k").to_pydict() == in_memory.sort_by("k").to_pydict()


# --------------------------------------------------------------------------
# Scratch disk is released as buckets are consumed.
# --------------------------------------------------------------------------
class _CountingStore(TieredSpillStore):
    """A spill store that records every release, so peak-scratch behavior is observable."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.released: list[str] = []

    def release(self, handle) -> None:
        self.released.append(handle.path)
        super().release(handle)


@pytest.fixture
def counting_store(monkeypatch):
    made: list[_CountingStore] = []

    def factory(work_dir):
        store = _CountingStore(work_dir)
        made.append(store)
        return store

    import batcher.dist.spill.buckets as buckets_mod

    # One binding, because there is now one caller: every breaker takes its store from
    # `buckets.spill_scratch`. Each of the five used to build its own, so this fixture had to
    # patch `_make_store` in five modules and stay in step with the list.
    monkeypatch.setattr(buckets_mod, "_make_store", factory)
    return made


def test_a_sort_releases_each_bucket_as_it_is_emitted(counting_store):
    """Buckets are read once, in key order. Holding them all until teardown made peak
    scratch the whole spilled input instead of the buckets still outstanding — and the
    staged copy of the *entire* mapped input was held on top of that."""
    source = _RecordingSource(rows=400)
    plan = _ds(source).select("k", "c1").sort("c1")._plan
    out = br.execute_spilling_sort(plan, [source], num_partitions=4)
    assert out.num_rows == 400

    store = counting_store[0]
    assert any("stage" in path for path in store.released), "the staged input was never freed"
    assert sum("bucket" in path for path in store.released) >= 1
    # Everything the run created was handed back before teardown.
    assert not store._local_paths and not store._remote_paths


def test_a_partitioned_window_releases_each_bucket_as_it_is_consumed(counting_store):
    source = _RecordingSource(rows=200)
    plan = (
        _ds(source)
        .select("k", "c2")
        .with_columns(r=bt.col("c2").sum().over(partition_by="k"))
        ._plan
    )
    assert sum(b.num_rows for b in br.stream_spilling_window(plan, [source], 4)) == 200
    assert counting_store[0].released


def test_a_global_window_stream_releases_its_buckets(counting_store):
    source = _RecordingSource(rows=200)
    plan = _ds(source).select("c3").with_columns(r=bt.col("c3").sum().over(order_by="c3"))._plan
    batches = list(stream_spilling_global_window(plan, [source], 4))
    assert sum(b.num_rows for b in batches) == 200
    assert counting_store[0].released


# --------------------------------------------------------------------------
# An empty out-of-core result must carry the same schema as the in-memory one.
# --------------------------------------------------------------------------
def test_an_empty_spilling_sort_carries_the_in_memory_schema():
    """`pa.table({f.name: [] for f in source.schema()})` gave every column a null type and
    the *source's* column set, so an empty spilled sort disagreed with the in-memory one on
    both the types and the names."""
    source = _RecordingSource(rows=50)
    ds = _ds(source).select("k", "c1").filter(bt.col("c1") < -1).sort("c1")
    spilled = br.execute_spilling_sort(ds._plan, [source], num_partitions=4)
    in_memory = ds.collect()
    assert spilled.num_rows == 0
    assert spilled.schema == in_memory.schema


def test_an_empty_spilling_join_carries_the_in_memory_schema():
    left = _RecordingSource(rows=40)
    right = _RecordingSource(rows=40)
    ds = _ds(left).filter(bt.col("c0") < -1).join(_ds(right), on="k")
    plan = ds._plan
    spilled = br.execute_spilling_join(plan, [left, right], num_partitions=4)
    assert spilled.num_rows == 0
    assert spilled.schema == ds.collect().schema


def test_spill_collect_still_declines_a_plan_shape_it_cannot_spill():
    source = _RecordingSource(rows=20)
    assert spill_collect(_ds(source).filter(bt.col("c0") > 0)._plan, [source], 4) is None
