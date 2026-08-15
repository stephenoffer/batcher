"""An equi-join on an all-null key column matches nothing, on every execution path.

SQL says `NULL = NULL` is unknown, so an equi-join finds no match for a null key. Batcher's
materializing path agreed with DuckDB about that; its **streaming** path did not, and
returned the full cartesian product instead — `collect()` gave 0 rows and `iter_batches()`
gave 16 for the same query over the same 4x4 input.

The cause is a property of Arrow rather than of the join. A column whose values are *all*
null arrives as arrow's `Null` type, which encodes nullity in the type itself and carries
no validity buffer, so `null_count()` reports **0** for a column in which every value is
null. The streaming probe skipped building its null mask on exactly that signal, and null
then matched null. `logical_null_count()` is arrow's answer for it.

Only an *entirely* null column is affected: `[1, None, 2]` is an `int64` with a validity
buffer and was always correct, which is why the ordinary null-key tests never caught this.
Both shapes are pinned below so the fix cannot be narrowed to the wrong one.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal, duck_materialize

pytestmark = pytest.mark.differential

# Entirely null -> arrow `null` type, the shape that broke. Mixed -> `int64`, the control.
_ALL_NULL = {"k": pa.array([None, None, None, None], type=pa.null()), "v": [1, 2, 3, 4]}
_MIXED = {"k": pa.array([1, None, 2], type=pa.int64()), "v": [1, 2, 3]}

_HOWS = ["inner", "left", "semi", "anti"]


def _tables(data):
    return pa.table(data)


@pytest.mark.parametrize("how", _HOWS)
@pytest.mark.parametrize("data", [_ALL_NULL, _MIXED], ids=["all_null_key", "mixed_key"])
def test_streaming_and_materializing_paths_agree(data, how):
    """`collect()` is the oracle; `iter_batches()` must produce the same rows."""
    left = bt.from_arrow(_tables(data))
    joined = left.join(left, on="k", how=how)
    materialized = joined.collect()
    streamed = list(joined.iter_batches())
    streamed_table = (
        pa.Table.from_batches(streamed, schema=materialized.schema)
        if streamed
        else materialized.slice(0, 0)
    )
    assert_tables_equal(streamed_table, materialized)


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("data", [_ALL_NULL, _MIXED], ids=["all_null_key", "mixed_key"])
def test_the_join_matches_duckdb(duck, data, how):
    table = _tables(data)
    duck_materialize(duck, "l", table)
    duck_materialize(duck, "r", table)
    got = bt.from_arrow(table).join(bt.from_arrow(table), on="k", how=how).collect()
    sql = f"SELECT l.k, l.v, r.v AS v_right FROM l {how.upper()} JOIN r ON l.k = r.k"
    assert_same(got.rename_columns(["k", "v", "v_right"]), duck.sql(sql))


def test_an_all_null_join_key_yields_no_matches():
    """The property in one line: nothing equals NULL, so an inner join finds nothing."""
    left = bt.from_arrow(_tables(_ALL_NULL))
    joined = left.join(left, on="k", how="inner")
    assert joined.collect().num_rows == 0
    assert sum(b.num_rows for b in joined.iter_batches()) == 0
