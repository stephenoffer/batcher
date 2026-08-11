"""`NULLS FIRST` in a window `ORDER BY`, against DuckDB.

The clause was read off the parse tree only for its `desc` flag, so `nulls_first` was
dropped on the floor and every `NULLS FIRST` window returned the `NULLS LAST` answer.

That is not a tie-ordering nicety. Where the nulls sit decides each row's *rank*, and under
a running frame it decides which rows are inside the frame at all -- so
`sum(v) OVER (ORDER BY v NULLS FIRST)` returned `[3.0, 3.0, 1.0, 3.0]` where DuckDB returns
`[3.0, None, 1.0, None]`. Plausible numbers, no error, from a clause the query spelled out.

Nothing below the translator was missing: `SortKeySpec` has carried `nulls_first` all along
and `bc_ir`'s order key has the field, so the fix is plumbing rather than a new capability.
The DataFrame surface gained the third tuple element that was already representable
underneath, and it is asserted here too -- a SQL-only fix would leave `ds.window()` unable
to say the thing its own plan node can express.

Every case is compared row-by-row in a fixed `ORDER BY k`, because `assert_same` is
order-independent and the ranking functions differ only in which row got which number.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_SPECS = [
    "v",
    "v DESC",
    "v ASC NULLS FIRST",
    "v ASC NULLS LAST",
    "v DESC NULLS FIRST",
    "v DESC NULLS LAST",
]
_FUNCS = [
    "rank()",
    "dense_rank()",
    "row_number()",
    "sum(v)",
    "count(v)",
    "lag(v)",
    "first_value(v)",
]


def _table() -> pa.Table:
    """Nulls at both ends and a duplicate value, so ranks and peers both bite."""
    return pa.table(
        {
            "k": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "v": pa.array([2.0, None, 1.0, None, 2.0, 3.0]),
        }
    )


@pytest.mark.parametrize("spec", _SPECS)
@pytest.mark.parametrize("func", _FUNCS)
def test_every_null_placement_matches_duckdb(duck, func, spec):
    """42 combinations: the clause has to survive for each function and each direction."""
    table = _table()
    sql = f"SELECT k, {func} OVER (ORDER BY {spec}) AS r FROM t ORDER BY k"
    duck.register("t", table)
    got = bt.sql(sql, t=table).to_pydict()
    want = [row[1] for row in duck.sql(sql).fetchall()]
    assert got["r"] == want, f"{func} OVER (ORDER BY {spec})"


def test_nulls_first_and_nulls_last_disagree():
    """The bug in one assertion: the two spellings used to return the same thing."""
    table = _table()
    first = bt.sql(
        "SELECT rank() OVER (ORDER BY v NULLS FIRST) AS r FROM t ORDER BY k", t=table
    ).to_pydict()["r"]
    last = bt.sql(
        "SELECT rank() OVER (ORDER BY v NULLS LAST) AS r FROM t ORDER BY k", t=table
    ).to_pydict()["r"]
    assert first != last


def test_a_running_frame_sees_different_rows(duck):
    """Not just ranks: null placement moves which rows fall inside a running frame."""
    table = _table()
    sql = "SELECT k, sum(v) OVER (ORDER BY v NULLS FIRST) AS s FROM t ORDER BY k"
    duck.register("t", table)
    assert bt.sql(sql, t=table).to_pydict()["s"] == [row[1] for row in duck.sql(sql).fetchall()]


def test_it_holds_with_a_partition_and_an_explicit_frame(duck):
    """The clause must survive the paths that group windows by their (partition, order)."""
    table = pa.table(
        {
            "k": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "g": pa.array(["a", "a", "a", "b", "b", "b"]),
            "v": pa.array([2.0, None, 1.0, None, 3.0, 2.0]),
        }
    )
    sql = (
        "SELECT k, sum(v) OVER ("
        "  PARTITION BY g ORDER BY v DESC NULLS FIRST"
        "  ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM t ORDER BY k"
    )
    duck.register("t", table)
    assert bt.sql(sql, t=table).to_pydict()["s"] == [row[1] for row in duck.sql(sql).fetchall()]


def test_two_windows_differing_only_in_null_placement_stay_separate(duck):
    """Windows are grouped by their (partition, order) spec, so the flag must be in the key.

    If `nulls_first` were dropped from the grouping key the two would collapse into one
    operator and both columns would get whichever spec was seen first -- silently, since
    the query still returns two columns of plausible ranks.
    """
    table = _table()
    sql = (
        "SELECT k,"
        " rank() OVER (ORDER BY v NULLS FIRST) AS a,"
        " rank() OVER (ORDER BY v NULLS LAST) AS b"
        " FROM t ORDER BY k"
    )
    duck.register("t", table)
    got = bt.sql(sql, t=table).to_pydict()
    rows = duck.sql(sql).fetchall()
    assert got["a"] == [r[1] for r in rows]
    assert got["b"] == [r[2] for r in rows]
    assert got["a"] != got["b"]


def test_the_dataframe_surface_can_say_it_too():
    """`ds.window()` gained the third tuple element; the two-element form is unchanged."""
    ds = bt.from_arrow(_table())
    default = ds.window(order_by=[("v", False)], functions={"r": "rank"}).to_pydict()["r"]
    explicit_last = ds.window(order_by=[("v", False, False)], functions={"r": "rank"}).to_pydict()
    nulls_first = ds.window(order_by=[("v", False, True)], functions={"r": "rank"}).to_pydict()
    assert default == explicit_last["r"]
    assert nulls_first["r"] != default
    assert ds.window(order_by=["v"], functions={"r": "rank"}).to_pydict()["r"] == default


def test_it_survives_a_partitioned_collect(duck):
    """Ordering metadata has to reach every partition, not just the first."""
    table = _table()
    sql = "SELECT k, rank() OVER (ORDER BY v NULLS FIRST) AS r FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).repartition(3).collect(), duck.sql(sql))
