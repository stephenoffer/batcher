"""Pure non-equi (theta) inner joins vs DuckDB.

``A JOIN B ON A.x < B.y`` has no equality conjunct, so there are no hash keys to
co-partition on. It is exactly a cross join followed by the predicate — the definition
of a nested-loop join — and is lowered to that rather than rejected, which is what these
tests pin. Previously every query below raised ``NotImplementedError``.

The edge cases matter more than the happy path here: NULLs must not match (SQL
three-valued logic, so ``NULL < 5`` is UNKNOWN and the row is dropped), an empty side
must yield no rows rather than the cross product, and a predicate matching nothing must
still produce the right *columns*.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def ab(duck):
    a = pa.table({"x": [1, 2, 3, 4], "lab": ["a", "b", "c", "d"]})
    b = pa.table({"y": [2, 3, 10], "tag": ["p", "q", "r"]})
    duck.register("a", a)
    duck.register("b", b)
    return a, b


@pytest.mark.differential
@pytest.mark.parametrize(
    "cond",
    [
        "a.x < b.y",
        "a.x <= b.y",
        "a.x > b.y",
        "a.x <> b.y",
        "a.x < b.y AND a.x > 1",
        "a.x + 1 < b.y",
    ],
)
def test_pure_theta_inner_join(duck, ab, cond):
    """Every comparison shape must match DuckDB row for row."""
    a, b = ab
    query = f"SELECT x, y, lab, tag FROM a JOIN b ON {cond}"
    out = bt.sql(query, a=a, b=b).collect()
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_theta_join_nulls_never_match(duck):
    """`NULL < y` is UNKNOWN, not true — those rows must be dropped, not emitted."""
    a = pa.table({"x": [1, None, 3]})
    b = pa.table({"y": [2, None, 5]})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT x, y FROM a JOIN b ON a.x < b.y"
    out = bt.sql(query, a=a, b=b).collect()
    assert_same(out, duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("empty_side", ["a", "b"])
def test_theta_join_empty_input(duck, empty_side):
    """An empty side yields no rows — the cross product must not leak through."""
    a = pa.table({"x": pa.array([] if empty_side == "a" else [1, 2], pa.int64())})
    b = pa.table({"y": pa.array([] if empty_side == "b" else [5, 6], pa.int64())})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT x, y FROM a JOIN b ON a.x < b.y"
    out = bt.sql(query, a=a, b=b).collect()
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_theta_join_matching_nothing_keeps_schema(duck, ab):
    """A predicate no pair satisfies returns zero rows with the right columns."""
    a, b = ab
    query = "SELECT x, y FROM a JOIN b ON a.x > b.y + 100"
    out = bt.sql(query, a=a, b=b).collect()
    assert out.num_rows == 0
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_theta_join_aggregate_on_top(duck, ab):
    """A band/range join feeding an aggregate — the shape this actually gets used for."""
    a, b = ab
    query = "SELECT b.tag, count(*) AS n FROM a JOIN b ON a.x < b.y GROUP BY b.tag"
    out = bt.sql(query, a=a, b=b).collect()
    assert_same(out, duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("how", ["LEFT", "RIGHT"])
@pytest.mark.parametrize("cond", ["a.x < b.y", "a.x > b.y", "a.x <> b.y"])
def test_outer_theta_join(duck, ab, how, cond):
    """An OUTER theta join keeps the preserved side's unmatched rows, null-extended.

    Cross+filter alone cannot express this — the filter removes exactly the rows an outer
    join must keep — so the unmatched half is recovered by tagging the preserved side with
    a row index and anti-joining the indices that survived.
    """
    a, b = ab
    query = f"SELECT * FROM a {how} JOIN b ON {cond}"
    assert_same(bt.sql(query, a=a, b=b).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("how", ["LEFT", "RIGHT"])
def test_outer_theta_join_when_nothing_matches(duck, ab, how):
    """No pair satisfies the predicate — every preserved row survives, null-extended."""
    a, b = ab
    query = f"SELECT * FROM a {how} JOIN b ON a.x > b.y + 1000"
    assert_same(bt.sql(query, a=a, b=b).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("how", ["LEFT", "RIGHT"])
def test_outer_theta_join_column_order(duck, ab, how):
    """The output keeps SQL's left-then-right column order, not the rewrite's order.

    A RIGHT theta join runs as a LEFT one over swapped operands, so the column order has
    to be restored afterwards or `SELECT *` silently returns the columns transposed.
    """
    a, b = ab
    out = bt.sql(f"SELECT * FROM a {how} JOIN b ON a.x < b.y", a=a, b=b).collect()
    assert out.column_names == duck.sql(f"SELECT * FROM a {how} JOIN b ON a.x < b.y").columns


@pytest.mark.differential
def test_outer_theta_join_with_nulls(duck):
    """A NULL key never satisfies the predicate, so its row must still be preserved."""
    a = pa.table({"x": [1, None, 7]})
    b = pa.table({"y": [5, None]})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT * FROM a LEFT JOIN b ON a.x < b.y"
    assert_same(bt.sql(query, a=a, b=b).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("cond", ["a.x < b.y", "a.x > b.y", "a.x <> b.y"])
def test_full_theta_join(duck, ab, cond):
    """FULL theta preserves the unmatched rows of BOTH sides, each null-extended.

    It is the symmetric case: the left side's misses are found by anti-joining the row
    indices that survived the predicate, and the right side's the same way, so both sides
    need a row index rather than just the preserved one.
    """
    a, b = ab
    query = f"SELECT * FROM a FULL JOIN b ON {cond}"
    assert_same(bt.sql(query, a=a, b=b).collect(), duck.sql(query))


@pytest.mark.differential
def test_full_theta_join_when_nothing_matches(duck, ab):
    """Nothing matches — every row of both sides survives, null-extended."""
    a, b = ab
    query = "SELECT * FROM a FULL JOIN b ON a.x > b.y + 1000"
    out = bt.sql(query, a=a, b=b).collect()
    assert out.num_rows == a.num_rows + b.num_rows
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_full_theta_join_drops_its_row_index_helpers(ab):
    """Neither side's row-index helper column may reach the output."""
    a, b = ab
    out = bt.sql("SELECT * FROM a FULL JOIN b ON a.x < b.y", a=a, b=b).collect()
    assert not any(c.startswith("__theta") for c in out.column_names)
