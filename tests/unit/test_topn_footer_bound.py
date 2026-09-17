"""The footer-proved top-N bound: the proof, and every shape it must decline.

`kyber.learned_tuning.topn_footer` rewrites the *first* run of ``ORDER BY x LIMIT k`` into a
filter at a value the Parquet row-group statistics prove at least `k` rows reach. These tests
pin the proof on hand-built statistics, where the right threshold can be worked out by
inspection, and the shapes it declines. The end-to-end agreement with DuckDB, across every
scheduling, is `tests/differential/test_diff_topn_footer_bound.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.stats import RowGroupBounds
from batcher.kyber.learned_tuning.topn_footer import (
    TopNScanKey,
    footer_topn_seed,
    topn_scan_key,
)

pytestmark = pytest.mark.unit


def _group(lo, hi, rows=100, nulls=0):
    return RowGroupBounds("f", 0, rows, {"x": lo}, {"x": hi}, {"x": nulls})


def _plan(arrow_type=None, *, descending=True, k=10, nulls_first=False):
    arrow_type = arrow_type or pa.int64()
    table = pa.table({"x": pa.array([], arrow_type), "p": pa.array([], pa.int64())})
    return bt.from_arrow(table).sort("x", descending=descending, nulls_first=nulls_first).limit(k)


def _threshold(seed):
    predicate = seed.plan.to_ir()["input"]["input"]["predicate"]
    return predicate["op"], predicate["right"]["value"]


def test_descending_bound_is_the_min_of_the_group_that_reaches_k():
    plan = _plan(k=150)._plan
    key = topn_scan_key(plan)
    # Largest mins first: 900 (100 rows), 800 (100 rows) -> k=150 is reached inside the 800
    # group, so every row >= 800 is at least 150 rows deep and the bound is 800.
    bounds = [_group(0, 99), _group(900, 999), _group(100, 199), _group(800, 899)] + [
        _group(i, i + 99) for i in range(200, 800, 100)
    ]
    seed = footer_topn_seed(plan, key, bounds)
    assert seed is not None
    assert _threshold(seed) == ("ge", {"int": 800})


def test_ascending_bound_is_the_max_of_the_group_that_reaches_k():
    plan = _plan(descending=False, k=100)._plan
    key = topn_scan_key(plan)
    bounds = [_group(i, i + 99) for i in range(0, 1000, 100)]
    seed = footer_topn_seed(plan, key, bounds)
    assert seed is not None
    assert _threshold(seed) == ("le", {"int": 99})


def test_nulls_do_not_count_toward_k():
    """A group of mostly nulls contributes only its non-null rows to the proof."""
    plan = _plan(k=50)._plan
    key = topn_scan_key(plan)
    bounds = [_group(900, 999, nulls=90)] + [_group(i, i + 99) for i in range(0, 900, 100)]
    seed = footer_topn_seed(plan, key, bounds)
    # 10 non-null rows at >= 900 do not reach 50, so the bound falls to the next group.
    assert _threshold(seed) == ("ge", {"int": 800})


def test_a_group_with_no_statistics_is_never_counted_and_never_pruned():
    plan = _plan(k=50)._plan
    key = TopNScanKey(0, "x", True, 50)
    unknown = RowGroupBounds("f", 1, 10_000, {}, {}, {})
    bounds = [unknown] + [_group(i, i + 99) for i in range(0, 1000, 100)]
    # The unknown group is 10,000 of 11,000 rows and cannot be pruned, so the bound would
    # save nothing: declined on cost.
    assert footer_topn_seed(plan, key, bounds) is None


def test_statistics_that_cannot_prove_k_rows_decline():
    plan = _plan(k=1_000)._plan
    key = topn_scan_key(plan)
    assert footer_topn_seed(plan, key, [_group(0, 99), _group(100, 199)]) is None


def test_a_bound_that_prunes_little_is_declined():
    """Overlapping groups: every group's max clears every group's min, so nothing is skipped."""
    plan = _plan(k=10)._plan
    key = topn_scan_key(plan)
    bounds = [_group(i, 10_000) for i in range(10)]
    assert footer_topn_seed(plan, key, bounds) is None


@pytest.mark.parametrize(
    "arrow_type, seeded",
    [
        (pa.int64(), True),
        (pa.int32(), True),
        (pa.date32(), True),
        (pa.timestamp("ms"), True),
        (pa.timestamp("us", tz="UTC"), True),
        (pa.float64(), False),  # NaN is absent from a float footer's max
        (pa.string(), False),  # a writer may truncate string statistics
        (pa.timestamp("ns"), False),  # no exact Python literal to carry the bound
        (pa.uint64(), False),  # exceeds what the engine accepts at its boundary
    ],
)
def test_key_types(arrow_type, seeded):
    assert (topn_scan_key(_plan(arrow_type)._plan) is not None) is seeded


def test_nulls_first_is_declined():
    assert topn_scan_key(_plan(nulls_first=True)._plan) is None


def test_a_filter_below_the_sort_is_declined():
    """Row-group counts no longer count the rows that reach the sort."""
    table = pa.table({"x": pa.array([], pa.int64()), "v": pa.array([], pa.int64())})
    plan = bt.from_arrow(table).filter(bt.col("v") > 0).sort("x", descending=True).limit(3)
    assert topn_scan_key(plan._plan) is None


def test_a_rename_is_traced_to_the_source_column():
    table = pa.table({"x": pa.array([], pa.int64()), "p": pa.array([], pa.int64())})
    plan = (
        bt.from_arrow(table)
        .select(bt.col("x").alias("key"), "p")
        .sort("key", descending=True)
        .limit(3)
    )
    key = topn_scan_key(plan._plan)
    assert key is not None and key.column == "x"


def test_a_computed_key_is_declined():
    table = pa.table({"x": pa.array([], pa.int64())})
    plan = bt.from_arrow(table).select((bt.col("x") + 1).alias("y")).sort("y").limit(3)
    assert topn_scan_key(plan._plan) is None
