"""Per-partition top-N (QUALIFY) over enough rows to run the parallel window, vs DuckDB.

`test_diff_qualify_topn.py` pins the semantics on eight rows, which never reach the parallel
window: it engages from 32,768 rows (`RuntimeTuning::window_parallel_row_threshold`). Above it a
fused `rank <= k` returns only each hash bucket's surviving rows, ordered by input row once,
rather than a rank for every row and a mask (`bc_runtime::window::window_with_rank_limit`). So
the rows kept, and every column carried alongside them, are checked here at a size that takes
that path: nullable and string partition keys, a float order key holding NaN and -0.0, ties,
both directions, several `k`, a payload column, a partition holding most rows (which falls back
to the serial kernel), and the same query read through `iter_batches`.

`row_number` breaks ties by input position, which DuckDB does not promise to match, so its
cases compare the partition and order values only; `rank` and `dense_rank` keep every tied row,
so their cases compare the payload and the rank too.

The `row_number` cases order by `w`, a column without NaN, because DuckDB is the one that is
wrong there. On DuckDB 1.5.5 a `row_number() ... <= 2` over an order key holding NaN returned the
wrong rows for about 25 of 5,000 partitions, both with and without -0.0 or null keys and at
20,000 rows as well as 200,000; one partition whose two smallest values are -90 and -77 came
back as -26 and NaN. `k = 1` and `rank`/`dense_rank` were unaffected. So NaN ordering under
`row_number` is checked against the smallest values computed here instead, in
`test_row_number_over_nan_keeps_the_smallest_values`.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_ROWS = 200_000


@pytest.fixture(scope="module")
def table() -> pa.Table:
    rng = np.random.default_rng(17)
    keys = rng.integers(0, 5_000, _ROWS)
    key_valid = rng.random(_ROWS) > 0.01
    v = np.round(rng.normal(0, 50, _ROWS), 0)  # rounded, so ties are common
    v[rng.choice(_ROWS, 500, replace=False)] = np.nan
    v[rng.choice(_ROWS, 500, replace=False)] = -0.0
    w = np.where(np.isnan(v), 7.0, v)  # the same values with NaN replaced; -0.0 stays
    dominant = np.where(rng.random(_ROWS) < 0.9, 0, keys)
    return pa.table(
        {
            "k": pa.array(keys, mask=~key_valid),
            "s": pa.array([f"g{x % 700}" for x in keys]),
            "big": pa.array(dominant),
            "v": pa.array(v),
            "w": pa.array(w),
            "payload": pa.array(np.arange(_ROWS)),
        }
    )


@pytest.fixture
def x(duck, table):
    duck.register("x", table)
    return bt.from_arrow(table)


def _query(fn: str, key: str, direction: str, k: int, columns: str, order: str = "v") -> str:
    return (
        f"SELECT {columns} FROM (SELECT *, {fn}() OVER "
        f"(PARTITION BY {key} ORDER BY {order} {direction}) AS r FROM x) s WHERE r <= {k}"
    )


@pytest.mark.parametrize("key", ["k", "s", "big"])
@pytest.mark.parametrize("direction", ["ASC", "DESC"])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_row_number_keeps_the_same_values(duck, x, key, direction, k):
    sql = _query("row_number", key, direction, k, f"{key}, w", order="w")
    assert_same(bt.sql(sql, x=x).collect(), duck.sql(sql))


@pytest.mark.parametrize("descending", [False, True])
def test_row_number_over_nan_keeps_the_smallest_values(x, table, descending):
    """NaN sorts above every number, so it is kept last ascending and first descending."""
    k = 2
    direction = "DESC" if descending else "ASC"
    got = bt.sql(_query("row_number", "k", direction, k, "k, v"), x=x).collect()

    def rank_key(value: float) -> tuple[bool, float]:
        # NaN becomes a flag and a fixed value, so two NaNs compare equal and sort last (or
        # first, descending) without a NaN ever sitting inside a tuple comparison.
        is_nan = value != value
        number = 0.0 if is_nan else value
        return (not is_nan, -number) if descending else (is_nan, number)

    by_key: dict[int, list[float]] = {}
    for key, value in zip(
        table.column("k").to_pylist(), table.column("v").to_pylist(), strict=True
    ):
        by_key.setdefault(key, []).append(value)
    expected = {key: sorted(values, key=rank_key)[:k] for key, values in by_key.items()}
    kept: dict[int, list[float]] = {}
    for key, value in zip(got.column("k").to_pylist(), got.column("v").to_pylist(), strict=True):
        kept.setdefault(key, []).append(value)
    assert kept.keys() == expected.keys()
    nan_kept = 0
    for key, values in expected.items():
        # Compare as sorted keys, where -0.0 == 0.0 and every NaN is equal to every other.
        assert sorted(map(rank_key, kept[key])) == sorted(map(rank_key, values)), key
        nan_kept += sum(v != v for v in kept[key])
    if descending:
        assert nan_kept > 0, "a descending top-2 must keep some NaN, or this checks nothing"


@pytest.mark.parametrize("fn", ["rank", "dense_rank"])
@pytest.mark.parametrize("key", ["k", "s", "big"])
@pytest.mark.parametrize("direction", ["ASC", "DESC"])
@pytest.mark.parametrize("k", [1, 3])
def test_rank_keeps_the_same_rows_and_payload(duck, x, fn, key, direction, k):
    sql = _query(fn, key, direction, k, f"{key}, v, payload, r")
    assert_same(bt.sql(sql, x=x).collect(), duck.sql(sql))


def test_iter_batches_returns_the_same_rows(duck, x):
    sql = _query("rank", "k", "DESC", 2, "k, v, payload, r")
    batches = list(bt.sql(sql, x=x).iter_batches())
    assert batches, "the query returns rows, so iter_batches must yield at least one batch"
    assert_same(pa.Table.from_batches(batches), duck.sql(sql))


def test_a_limit_above_every_partition_keeps_every_row(duck, x):
    """Positive control: with `k` past every partition's size, nothing is dropped."""
    sql = _query("rank", "k", "ASC", _ROWS, "k, v, payload, r")
    got = bt.sql(sql, x=x).collect()
    assert got.num_rows == _ROWS
    assert_same(got, duck.sql(sql))
