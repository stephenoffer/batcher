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


#: The same three breakers over an ordinary expression rather than a UDF. They are the control
#: for the predicate guard: a guard that declined everything would satisfy every assertion
#: about the UDF plans while quietly retiring the out-of-core path for every query.
_PLAIN = {
    "sort": lambda ds: ds.with_columns(t=bt.col("t") * 2).sort("t"),
    "join": lambda ds: ds.with_columns(t=bt.col("t") * 2).join(
        bt.from_pydict({"k": list(range(11)), "w": [f"w{i}" for i in range(11)]}), on="k"
    ),
    "window": lambda ds: ds.with_columns(t=bt.col("t") * 2).window(
        partition_by=["k"], order_by=["t"], functions={"r": "row_number"}
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

    It raised for three of the six until the three `supports_spilling_*` predicates learned to
    decline a plan carrying a UDF: they answered "yes" for a `map_batches(...).sort(...)` and
    the streaming sort then died serializing it, which is the same defect as the one above one
    layer down. This arm used to skip on `NotImplementedError` for exactly that reason; the
    skip is gone because nothing reaches it, and a skip nothing reaches is a case nobody runs.
    """
    build = _SHAPES[shape]
    expected = build(bt.from_arrow(rows)).collect()
    batches = list(build(bt.from_arrow(rows)).iter_batches())
    streamed = pa.Table.from_batches(batches) if batches else expected.slice(0, 0)
    assert _multiset(streamed) == _multiset(expected)


@pytest.mark.parametrize(
    ("name", "predicate", "shape"),
    [
        ("sort", "supports_spilling_sort", "sort"),
        ("join", "supports_spilling_join", "join"),
        ("window", "supports_spilling_window", "window"),
    ],
)
def test_the_spill_predicates_decline_a_udf_plan(rows, name, predicate, shape):
    """Each breaker's own predicate must say no, rather than the executor discovering it.

    A predicate that answers *whether a path applies* must never raise and must never claim a
    plan the path cannot run — `test_spill_predicates_never_raise` states the first half, and
    this is the second. All three said yes to a plan carrying a `map_batches`, whose `to_ir()`
    raises by design, so every caller that trusted them died inside `json.dumps`.
    """
    import importlib

    module = importlib.import_module(f"batcher.dist.spill_breakers.{name}")
    check = getattr(module, predicate)
    assert check(_SHAPES[shape](bt.from_arrow(rows))._plan) is False
    # The control: the same breaker over an expression instead of a UDF must still be claimed,
    # or the guard has simply retired the out-of-core path for everyone.
    assert check(_PLAIN[shape](bt.from_arrow(rows))._plan) is True


def test_a_pipeline_without_a_udf_still_takes_the_spill_path(rows):
    """The control.

    The fix is "ask whether the plan has a UDF before dispatching to the spill executor". A
    version of it that answered "yes" for every plan would make every test above pass while
    silently retiring the out-of-core aggregate for everyone.
    """
    from batcher.dist.spill import spill_collect

    plan = bt.from_arrow(rows).group_by("k").agg(n=bt.col("t").sum())
    assert spill_collect(plan._plan, plan._sources, 3) is not None
