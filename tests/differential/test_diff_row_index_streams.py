"""`with_row_index` must stream, and number the rows `collect()` numbers.

`with_row_index`, `tail` and `with_random` all lower to `RowId`, and it materialized: the
streaming router had no branch for it. A row index is a *position*, so `RowId` is not
partition-independent and the router's peeling loop correctly refuses it — run per partition,
each one restarts the counter at zero.

Streaming is the case where that difficulty does not arise. Batches arrive in the input's own
row order, so a running counter numbers exactly the rows `collect()` numbers. That is why the
distributed path needs `preserve_order` and a driver-side assembly and this needs neither.

Three properties, and only the first is obvious:

* the values match;
* the **schema** matches, including that the index field is non-nullable — the engine emits it
  that way, and a streamed batch that disagrees on nullability alone will not `concat` with a
  collected one;
* the alias comes **first**, which is what `RowId.available_columns()` promises and what a
  positional consumer (a headerless write, an Arrow schema read in order) depends on.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

_N = 3000


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    return pa.table(
        {
            "a": pa.array(range(_N), pa.int64()),
            "g": pa.array([i % 9 for i in range(_N)], pa.int64()),
        }
    )


#: Inputs `is_streamable` admits — row-wise operators over a scan. The premise the running
#: counter rests on is that nothing below reorders or re-batches the input, so a shape that
#: *is* admitted and *does* reorder would be the defect this pins.
_INPUTS = {
    "scan": lambda ds: ds,
    "filter": lambda ds: ds.filter(bt.col("a") % 3 == 0),
    "project": lambda ds: ds.select("a"),
    "filter_then_project": lambda ds: ds.filter(bt.col("g") > 2).select("a", "g"),
}


@pytest.mark.parametrize("shape", sorted(_INPUTS))
@pytest.mark.parametrize("offset", [0, 100])
def test_a_streamed_row_index_equals_the_collected_one(rows, shape, offset):
    ds = _INPUTS[shape](bt.from_arrow(rows)).with_row_index("idx", offset=offset)
    batches = list(ds.iter_batches())
    streamed = pa.Table.from_batches(batches) if batches else ds.collect().slice(0, 0)
    assert_tables_equal(streamed, ds.collect(), ordered=True)


def test_the_index_column_is_first_and_non_nullable(rows):
    """Both are contracts a values-only comparison would miss."""
    ds = bt.from_arrow(rows).with_row_index("idx")
    for batch in ds.iter_batches():
        assert batch.schema.names[0] == "idx"
        assert batch.schema.field("idx").nullable is False
        assert batch.schema.equals(ds.collect().schema)


def test_it_actually_streams_rather_than_materializing(rows):
    """The point of the change. Without this the tests above pass on the materializing path.

    `_collect` is the router's fall-through: reaching it means the whole result was built in
    memory and re-chunked, which is what `iter_batches()` exists not to do.
    """
    import batcher.api.terminal.core as core

    reached = []
    original = core._collect
    core._collect = lambda *a, **k: (reached.append(1), original(*a, **k))[1]
    try:
        list(bt.from_arrow(rows).with_row_index("idx").iter_batches())
    finally:
        core._collect = original
    assert not reached, "with_row_index materialized instead of streaming"


def test_tail_comes_along_because_it_lowers_to_row_id(rows):
    """`tail` is a `RowId` plus a filter, so it streams for free — and is worth pinning
    separately because nothing in the branch mentions it."""
    ds = bt.from_arrow(rows).tail(10)
    batches = list(ds.iter_batches())
    streamed = pa.Table.from_batches(batches) if batches else ds.collect().slice(0, 0)
    assert_tables_equal(streamed, ds.collect(), ordered=True)
