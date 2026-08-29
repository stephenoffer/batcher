"""`collect(spill=True)` over a `map_batches` pipeline must answer, not raise.

`map_batches` runs a Python callable over whole Arrow batches and deliberately does not lower
to the engine IR — `MapBatches.to_ir()` raises by design, because there is nothing to lower.
The disk-shuffle spill executor therefore cannot run a plan carrying one, and it did not
decline such a plan: it entered the executor and raised ``NotImplementedError: map_batches is
executed in Python, not lowered to the engine IR`` from inside `to_ir()`.

So `collect(spill=True)` was a hard failure for every batch-inference pipeline with a breaker
over it — `map_batches(model).group_by(...).agg(...)`, `.sort(...)`, `.distinct()`, a window —
which is precisely the workload that asks for bounded memory. The message was an internal one
about a wire contract, so it read as an engine bug rather than as "this shape has no spilling
path", which is what it meant.

Asking before dispatching gives these shapes the fallback every other unspillable shape
already had. It does **not** give them a bounded-memory path: that wants the map prefix staged
to disk and the breaker run over the staging, the way `dist.executor._stage_map_prefix` does
for the distributed route. `iter_batches()` is the bounded way to run one today, and the last
test here pins that it is, so the gap is recorded by something that fails when it closes
rather than by a comment.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.integration

_N = 2000


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    return pa.table(
        {
            "k": pa.array([i % 11 for i in range(_N)], pa.int64()),
            "t": pa.array([(i * 7) % 101 for i in range(_N)], pa.int64()),
        }
    )


def _double(batch: pa.RecordBatch) -> pa.RecordBatch:
    """A UDF the engine cannot see into — the whole point of `map_batches`."""
    import pyarrow.compute as pc

    return batch.set_column(1, "t", pc.multiply(batch.column("t"), 2))


#: A map pipeline under each kind of breaker. Every one of them raised before.
_SHAPES = {
    "aggregate": lambda ds: ds.map_batches(_double).group_by("k").agg(n=bt.col("t").sum()),
    "sort": lambda ds: ds.map_batches(_double).sort("t"),
    "distinct": lambda ds: ds.map_batches(_double).select("k").distinct(),
    "limit_over_sort": lambda ds: ds.map_batches(_double).sort("t").limit(5),
    "window": lambda ds: ds.map_batches(_double).window(
        partition_by=["k"], order_by=["t"], functions={"r": "row_number"}
    ),
    "join": lambda ds: ds.map_batches(_double).join(
        bt.from_pydict({"k": list(range(11)), "w": [f"w{i}" for i in range(11)]}), on="k"
    ),
}


def _multiset(table: pa.Table) -> list:
    data = table.to_pydict()
    names = sorted(table.column_names)
    return sorted(
        (tuple(data[n][i] for n in names) for i in range(table.num_rows)),
        key=lambda row: tuple(repr(v) for v in row),
    )


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_spilling_a_map_pipeline_returns_the_in_memory_answer(rows, shape):
    build = _SHAPES[shape]
    expected = build(bt.from_arrow(rows)).collect()
    spilled = build(bt.from_arrow(rows)).collect(spill=True, num_partitions=3)
    assert spilled.schema == expected.schema
    assert _multiset(spilled) == _multiset(expected)


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_same_pipeline_streams(rows, shape):
    """`iter_batches()` is the bounded-memory route these shapes actually have.

    Not every one of them takes it — `sort` over a map prefix declines there too — so this
    asserts the weaker, true thing: streaming either answers with the in-memory answer or
    declines in a way the caller sees, and never returns a *different* answer. When the map
    prefix gains a staged spill path, this is the test that says which shapes already had one.
    """
    build = _SHAPES[shape]
    expected = build(bt.from_arrow(rows)).collect()
    try:
        batches = list(build(bt.from_arrow(rows)).iter_batches())
    except NotImplementedError:
        pytest.skip(f"{shape} has no streaming path either")
    streamed = pa.Table.from_batches(batches) if batches else expected.slice(0, 0)
    assert _multiset(streamed) == _multiset(expected)


def test_a_pipeline_without_a_udf_still_takes_the_spill_path(rows):
    """The control.

    The fix is "ask whether the plan has a UDF before dispatching to the spill executor". A
    version of it that answered "yes" for every plan would make every test above pass while
    silently retiring the out-of-core aggregate for everyone.
    """
    from batcher.dist.spill import spill_collect

    plan = bt.from_arrow(rows).group_by("k").agg(n=bt.col("t").sum())
    assert spill_collect(plan._plan, plan._sources, 3) is not None
