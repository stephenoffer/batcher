"""A sort or global window keyed on a temporal or decimal column must still split.

Both bounded-memory paths — the out-of-core sort and the streamed global window — split by
range-partitioning the leading key into ordered buckets. Whether a key type can be cut that
way is `range_partitionable`'s answer, and it used to be "no" for `Date`, `Timestamp`,
`Time`, `Duration` and `Decimal`: every one of them has an order-preserving numeric backing
the sampler and the Rust router read perfectly well, and the predicate simply had not been
told. The distributed sort had found that out and widened the test at its own call site, so
``ORDER BY <timestamp>`` fanned across a cluster while the same key declined to spill on one
machine.

Declining costs memory rather than correctness, which is why it went unnoticed: the
materializing kernel runs instead and returns the right answer until the relation stops
fitting. That makes ``ORDER BY l_shipdate`` — the shape these paths exist for — the one they
refused.

`tests/unit/test_range_partition_key_types.py` holds the predicate to what the partitioner
actually does, on no data. This holds the *engine* to it: the bounded-memory answer is the
in-memory answer, for each key type, with the budget low enough to force the split.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.integration

_N = 4000


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    base = datetime.date(2020, 1, 1)
    epoch = datetime.datetime(2020, 1, 1)
    return pa.table(
        {
            # `rid` is the row identity both paths carry, so the comparison can be ordered.
            "rid": pa.array(range(_N), pa.int64()),
            "d": pa.array(
                [base + datetime.timedelta(days=(i * 13) % 900) for i in range(_N)], pa.date32()
            ),
            "ts": pa.array(
                [epoch + datetime.timedelta(seconds=(i * 97) % 100_000) for i in range(_N)],
                pa.timestamp("us"),
            ),
            "tm": pa.array(
                [datetime.time((i * 7) % 24, (i * 11) % 60, i % 60) for i in range(_N)],
                pa.time64("us"),
            ),
            "dur": pa.array(
                [datetime.timedelta(seconds=(i * 31) % 5000) for i in range(_N)],
                pa.duration("us"),
            ),
            "dec": pa.array([float((i * 7) % 500) for i in range(_N)], pa.float64()).cast(
                pa.decimal128(12, 2)
            ),
            "v": pa.array([float(i % 71) for i in range(_N)], pa.float64()),
        }
    )


#: Every key type with a numeric backing the range partitioner reads. `rid` is not among them
#: on purpose: an int64 key already worked, so a suite of those alone would pass while the
#: temporal ones were still declined.
_KEYS = ["d", "ts", "tm", "dur", "dec"]


def _ordered(table: pa.Table) -> dict:
    """`table` in `rid` order, outside the engine — a sort result is a sequence, not a set."""
    return table.sort_by([("rid", "ascending")]).to_pydict()


@pytest.mark.parametrize("key", _KEYS)
def test_the_out_of_core_sort_engages_and_agrees(rows, key, monkeypatch):
    """`ORDER BY <temporal>` under a spill budget equals the in-memory sort.

    The `sorted_by` metadata a `bt.from_arrow` source does not carry means nothing can
    eliminate the sort, so this really does exercise the operator.
    """
    from batcher.dist.spill_breakers.sort import supports_spilling_sort

    ds = bt.from_arrow(rows)
    plan = ds.sort(key)._plan
    assert supports_spilling_sort(plan), f"{key} declined the ordered range partition"

    expected = ds.sort(key).collect()
    spilled = ds.sort(key).collect(spill=True)
    assert _ordered(spilled) == _ordered(expected)
    # The sort's own promise, checked on the key rather than through `rid`: the two paths
    # agree on the *sequence*, not merely on the multiset of rows.
    assert spilled.column(key).to_pylist() == expected.column(key).to_pylist()


@pytest.mark.parametrize("key", _KEYS)
def test_the_streamed_global_window_engages_and_agrees(rows, key):
    """A global `ORDER BY <temporal>` window streams in ordered buckets and agrees.

    A global window has one partition over every row, so the ordered-bucket split is its only
    bounded-memory path — declining the key type left it with none.
    """
    from batcher.dist.global_window import supports_ordered_bucket_offsets

    ds = bt.from_arrow(rows)
    windowed = ds.window(order_by=[key], functions={"r": "row_number", "s": ("sum", bt.col("v"))})
    assert supports_ordered_bucket_offsets(windowed._plan), f"{key} declined ordered buckets"

    expected = _ordered(windowed.collect())
    streamed = _ordered(pa.Table.from_batches(list(windowed.iter_batches())))
    assert streamed == expected
