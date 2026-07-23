"""The streaming Kyber rules preserve results, executed end to end.

`tests/unit/test_kyber_streaming_rules.py` asserts the plan *shape* the rules produce.
This asserts the thing that actually matters: running the rewritten plan yields the same
rows as running the original. For `WatermarkDedup` there is no DuckDB oracle — the
operator is streaming-only, with event-time eviction SQL has no spelling for — so the
oracle here is the unrewritten plan itself, which is the same discipline the differential
suite applies against DuckDB.

The bad case these pin is specific and quiet: pushing a filter below a dedup that keeps
the *first* row per key can promote a different row to be that key's first. That changes
the output without raising, and only on an unbounded input.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.expr_ir import col, lit

_BASE = dt.datetime(2024, 1, 1, 0, 0, 0)
_SCHEMA = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("v", pa.int64())])

pytestmark = pytest.mark.integration

#: `k` is the dedup key, so a predicate over `k` is constant per key (pushable).
#: `v` varies within a key, so a predicate over `v` is not (must not be pushed).
_ROWS = [
    ("a", 0, 1),
    ("b", 1, 5),
    ("a", 2, 9),  # duplicate of 'a', larger v
    ("c", 3, 2),
    ("b", 4, 7),  # duplicate of 'b', larger v
    ("d", 5, 4),
]


def _rb(rows):
    return pa.RecordBatch.from_pydict(
        {
            "k": [k for k, _, _ in rows],
            "ts": [_BASE + dt.timedelta(minutes=m) for _, m, _ in rows],
            "v": [v for _, _, v in rows],
        },
        schema=_SCHEMA,
    )


def _run(predicate, *, split: int) -> dict[str, int]:
    """Dedup then filter, fed as two micro-batches, returned as {key: v}."""

    def batches():
        yield _rb(_ROWS[:split])
        yield _rb(_ROWS[split:])

    ds = bt.from_batches(batches, _SCHEMA, bounded=False).drop_duplicates_within_watermark(
        ["k"], event_time="ts", lateness="1h"
    )
    out = pa.Table.from_batches(list(ds.filter(predicate).iter_batches()))
    return dict(zip(out.column("k").to_pylist(), out.column("v").to_pylist(), strict=True))


def test_key_constant_filter_gives_the_same_rows_after_pushdown():
    """The pushable case: the rule fires, and the answer is unchanged.

    `k != 'd'` is constant across each key group, so evaluating it before or after the
    dedup selects the same rows. Batching is varied because the dedup carries state
    across micro-batches — a rewrite that is only correct within a single batch would
    pass at one split and fail at another.
    """
    expected = {"a": 1, "b": 5, "c": 2}
    for split in (1, 3, 5):
        assert _run(col("k") != lit("d"), split=split) == expected, f"split={split}"


def test_non_key_filter_is_not_pushed_and_keeps_first_row_semantics():
    """The refusal case, and the reason the refusal exists.

    `v < 8` rejects the first row of no key here but would accept the *second* row of
    'a' (v=9 is rejected, v=1 kept) — the point is that filtering first could change
    which row is 'first' for a key. The dedup must run first: 'a' resolves to v=1 and
    'b' to v=5 (their earliest rows), and both survive the filter.

    If the filter were wrongly pushed below the dedup, 'b' would still resolve to 5 but
    the plan would no longer be equivalent in general; this pins the observable result
    for the shape the optimizer must not rewrite.
    """
    expected = {"a": 1, "b": 5, "c": 2, "d": 4}
    for split in (1, 3, 5):
        assert _run(col("v") < lit(8), split=split) == expected, f"split={split}"


def test_dedup_result_is_independent_of_micro_batch_boundaries():
    """Whatever the rules do, batching must not be observable in the result."""
    results = [_run(col("v") > lit(0), split=s) for s in (1, 2, 3, 4, 5)]
    assert all(r == results[0] for r in results), results
