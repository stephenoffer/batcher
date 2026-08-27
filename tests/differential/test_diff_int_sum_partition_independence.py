"""An integer `SUM` decides overflow on a total, not on the order rows arrive in.

`bc-runtime` accumulates an `Int64` sum in an `i64` and checks every add, so a wrap can never
be returned as an answer. What it checked, though, was the **running** total, and a running
total depends on the order the rows arrive in. The multiset ``{2**62, 2**62, -2**62, -2**62}``
sums to 0, which fits an `int64` with room to spare, but it exceeds one partway whenever the
two positives land first. So the same four rows either summed or raised depending only on how
they were batched::

    one batch  [M, M, -M, -M]     -> raised
    two        [M, -M] [M, -M]    -> 0
    two        [M, M] [-M, -M]    -> raised

Batching is a scheduling decision, so a query could succeed single-node and fail distributed
on identical data. The engine now retries the accumulation in `i128` and narrows once, so a
partition's success is a property of its data.

**What that does and does not promise**, because the difference is the whole contract:

* Within a partition, row order no longer decides anything. Any order of a multiset whose true
  sum fits an `int64` now sums.
* Across partitions, a partition whose *own* true sum exceeds `int64` still raises, because a
  partial's state is an `int64` column and there is nothing for it to hold. `[M, M]` as one
  batch is exactly that case and is asserted below rather than skipped.

So batching still decides in that narrower case. Closing it needs a wider intermediate schema,
which is a wire-contract change. `SUM` also still returns `int64` and still raises when the
true total needs more, which is the one place this deliberately differs from DuckDB (which
promotes to `HUGEINT`).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_M = 2**62
_SCHEMA = pa.schema([("x", pa.int64()), ("k", pa.int64())])


def _sum(chunks: list[list[int]]) -> int | None:
    batches = [pa.record_batch({"x": c, "k": [0] * len(c)}, schema=_SCHEMA) for c in chunks]
    got = bt.from_batches(batches).agg(s=bt.col("x").sum()).collect()
    assert got.schema.field("s").type == pa.int64(), "SUM of int64 stays int64"
    return got.to_pydict()["s"][0]


#: Every order of the multiset, as a single batch. All sum to 0; they differ only in when the
#: running total would transiently leave `int64`.
_ORDERS = [
    [_M, _M, -_M, -_M],
    [_M, -_M, _M, -_M],
    [-_M, -_M, _M, _M],
    [_M, -_M, -_M, _M],
    [-_M, _M, -_M, _M],
    [-_M, _M, _M, -_M],
]


@pytest.mark.parametrize("order", _ORDERS)
def test_row_order_within_a_partition_never_decides(order):
    assert _sum([order]) == 0, f"order={order}"


#: Splits in which **every chunk's own true sum fits an int64**, which is the case the
#: mergeable algebra promises. Some chunks still overflow partway through.
_FITTING_SPLITS = [
    [[_M, -_M], [_M, -_M]],
    [[_M, _M, -_M], [-_M]],
    [[_M], [_M], [-_M], [-_M]],
    [[_M, _M, -_M, -_M]],
    [[_M, -_M, _M], [-_M]],
]


@pytest.mark.parametrize("chunks", _FITTING_SPLITS)
def test_a_split_whose_partitions_each_fit_merges_to_the_true_total(chunks):
    assert _sum(chunks) == 0, f"chunks={chunks}"


def test_a_partition_that_cannot_hold_its_own_sum_still_raises():
    # `[M, M]` has a true sum of 2**63. A partial's state is an int64 column, so there is
    # nothing for it to hold -- this is the residual the module docstring describes, and it
    # is asserted so that closing it later is a deliberate change rather than a surprise.
    with pytest.raises(Exception, match=r"(?i)overflow"):
        _sum([[_M, _M], [-_M, -_M]])


def test_a_sum_that_truly_exceeds_int64_still_raises():
    # 4 * 2**62 == 2**64. No order and no split makes this fit, so it must raise rather than
    # wrap: the retry widens the accumulator, never the result type.
    with pytest.raises(Exception, match=r"(?i)overflow"):
        _sum([[_M, _M], [_M, _M]])


def test_an_ordinary_int_sum_is_unaffected():
    # The negative control: the fast path still answers, and nulls still behave.
    assert _sum([[1, 2, 3], [4, 5]]) == 15
    got = (
        bt.from_batches([pa.record_batch({"x": [None, None], "k": [0, 0]}, schema=_SCHEMA)])
        .agg(s=bt.col("x").sum())
        .collect()
    )
    assert got.to_pydict()["s"] == [None], "an all-null sum is NULL, not 0"
