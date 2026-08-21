"""A third wave, from the richer random-query fuzzer.

Two of these are engine-level rather than translator-level: a join whose output columns are
all pruned away, and an aggregate over a group with nothing in it. Both returned an error
or a NULL where DuckDB returns an answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _tables() -> tuple[pa.Table, pa.Table]:
    left = pa.table(
        {
            "s": pa.array(["a", "b", None], pa.string()),
            "i": pa.array([1, 2, 3], pa.int64()),
        }
    )
    right = pa.table(
        {
            "k": pa.array(["a", "z"], pa.string()),
            "v": pa.array([1, 2], pa.int64()),
        }
    )
    return left, right


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) AS c FROM t FULL OUTER JOIN u ON t.s = u.k",
        "SELECT count(*) AS c FROM t FULL OUTER JOIN u ON t.i = u.v",
        "SELECT count(*) AS c FROM t LEFT JOIN u ON t.s = u.k",
        "SELECT count(*) AS c FROM t RIGHT JOIN u ON t.s = u.k",
        "SELECT count(*) AS c FROM t JOIN u ON t.s = u.k",
        "SELECT count(*) AS c FROM t CROSS JOIN u",
    ],
)
def test_a_join_whose_columns_are_all_pruned_still_counts(duck, sql):
    """`count(*)` reads no column, so projection pruning leaves the join emitting none.

    A zero-column batch has to state its row count — it is the only thing it carries — and
    the full-outer path did not, so the query died on *"must either specify a row count or
    at least one column"* where every other join type answered.
    """
    left, right = _tables()
    duck.register("t", left)
    duck.register("u", right)
    assert_same(bt.sql(sql, t=left, u=right).collect(), duck.sql(sql))


def _numeric() -> pa.Table:
    return pa.table(
        {
            "i": pa.array([7, -7, 1, 0], pa.int64()),
            "j": pa.array([2, 2, 0, 3], pa.int64()),
            "f": pa.array([7.0, -7.0, 1.0, 0.0], pa.float64()),
            "g": pa.array([2.0, 2.0, 0.0, 3.0], pa.float64()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT i, j, i // j AS r FROM t",
        "SELECT f, g, f // g AS r FROM t",
        "SELECT i, g, i // g AS r FROM t",
        "SELECT divide(i, j) AS r FROM t",
        "SELECT divide(f, g) AS r FROM t",
        "SELECT fdiv(i, j) AS r FROM t",
        "SELECT fdiv(f, g) AS r FROM t",
        "SELECT CAST(7 AS DOUBLE) // CAST(2 AS DOUBLE) AS r",
        "SELECT 7 // 2 AS r",
    ],
)
def test_the_two_division_operators_take_their_operands_types(duck, sql):
    """`//` truncates on integers and does *not* round on floats; `fdiv` always floors."""
    table = _numeric()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_integer_and_float_slash_slash_disagree_on_purpose():
    """The bug stated directly: `7.0 // 2.0` is 3.5, not 3.0."""
    table = _numeric()
    got = bt.sql(
        "SELECT CAST(7 AS DOUBLE) // CAST(2 AS DOUBLE) AS a, 7 // 2 AS b", t=table
    ).collect()
    assert got.to_pydict() == {"a": [3.5], "b": [3]}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT entropy(i) AS r FROM t",
        "SELECT g, entropy(i) AS r FROM t GROUP BY g",
        "SELECT entropy(i) AS r FROM t WHERE i > 100",
    ],
)
def test_entropy_of_an_empty_group_is_zero_not_null(duck, sql):
    """Every other group reports a number; a NULL there reads as "unknown", not "none"."""
    table = pa.table(
        {
            "g": pa.array(["x", "y", "y"], pa.string()),
            "i": pa.array([1, None, None], pa.int64()),
        }
    )
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


@pytest.mark.parametrize("unit", ["day", "month", "year", "hour"])
def test_the_three_argument_date_sub_is_date_diff(duck, unit):
    """DuckDB spells one function two ways; only one of them was recognised."""
    table = pa.table({"a": pa.array([1], pa.int64())})
    sql = f"SELECT date_sub('{unit}', TIMESTAMP '2020-01-01', TIMESTAMP '2021-03-04') AS r"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_json_type_refuses_with_the_reason_rather_than_a_node_name():
    """A refusal a reader can act on: it names what DuckDB reports and what to use here."""
    table = pa.table({"j": pa.array(['{"a":1}'], pa.string())})
    with pytest.raises(NotImplementedError, match="SQL type a value would cast to"):
        bt.sql("SELECT json_type(j) AS r FROM t", t=table).collect()
