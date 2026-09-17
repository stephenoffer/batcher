"""Positional derived expressions in `select`/`with_columns`/`agg`: names and aggregate shape.

`select(col("a") + 1)` names its output the way Polars does -- alias, else the leftmost leaf
(a column's own name, a literal's ``"literal"``) -- and an aggregate in `select` is the
whole-frame aggregate: one row when every output is an aggregate or a constant, broadcast
to every row when mixed with row-level outputs. DuckDB is the oracle for the *values* (it
has no name inference, so the SQL aliases the columns by hand); Polars 1.40 is the oracle
for the *names* and the row shape, compared positionally.

Every fixture runs over nulls, an empty input, one row, and duplicate values.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

_TABLES = {
    "nulls": pa.table(
        {"a": pa.array([1, None, 3, 3], pa.int64()), "b": pa.array([2.5, 1.0, None, 4.0])}
    ),
    "empty": pa.table({"a": pa.array([], pa.int64()), "b": pa.array([], pa.float64())}),
    "one_row": pa.table({"a": pa.array([7], pa.int64()), "b": pa.array([0.5])}),
    "duplicates": pa.table(
        {"a": pa.array([2, 2, 2, 5], pa.int64()), "b": pa.array([1.0, 1.0, 1.0, -0.0])}
    ),
}


@pytest.fixture(params=sorted(_TABLES))
def table(request) -> pa.Table:
    return _TABLES[request.param]


@pytest.fixture(scope="module")
def pl():
    return pytest.importorskip("polars")


def test_positional_derived_values_match_duckdb(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).select(bt.col("a") + 1, bt.col("b") * 2, bt.lit(1) + bt.col("a"))
    assert ours.columns == ["a", "b", "literal"]
    assert_same(ours.collect(), duck.sql("SELECT a + 1 AS a, b * 2 AS b, 1 + a AS literal FROM t"))


def test_all_aggregate_select_is_one_row(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).select(bt.col("a").sum(), bt.col("b").max(), bt.lit(1))
    assert ours.columns == ["a", "b", "literal"]
    out = ours.collect()
    assert out.num_rows == 1
    assert_same(out, duck.sql("SELECT sum(a) AS a, max(b) AS b, 1 AS literal FROM t"))


def test_mixed_select_broadcasts_the_aggregate(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).select(bt.col("a"), bt.col("b").sum().alias("total"))
    expected = duck.sql("SELECT a, sum(b) OVER () AS total FROM t")
    assert_same(ours.collect(), expected)


def test_with_columns_positional_replaces_by_inferred_name(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).with_columns(bt.col("a") * 10, bt.col("b").mean())
    assert ours.columns == ["a", "b"]
    assert_same(ours.collect(), duck.sql("SELECT a * 10 AS a, avg(b) OVER () AS b FROM t"))


def test_agg_positional_names_follow_the_leftmost_column(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).group_by("a").agg(bt.col("b").sum(), bt.count())
    assert ours.columns == ["a", "b", "count"]
    assert_same(
        ours.collect(), duck.sql('SELECT a, sum(b) AS b, count(*) AS "count" FROM t GROUP BY a')
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.select(bt.col("a") + 1, bt.col("a") * 2),
        lambda ds: ds.select(bt.lit(1), bt.lit(2)),
        lambda ds: ds.with_columns(bt.col("a") + 1, bt.col("a") - 1),
        lambda ds: ds.group_by("b").agg(bt.col("a").sum(), bt.col("a").mean()),
        lambda ds: ds.select(bt.col("a") + 1, a=bt.col("b")),
    ],
)
def test_colliding_inferred_names_are_refused(build):
    ds = bt.from_arrow(_TABLES["nulls"])
    with pytest.raises(PlanError, match=r"duplicate output column|positional aggregates over|both"):
        build(ds)


def test_polars_names_and_values_agree_on_row_level_outputs(pl, table):
    theirs = pl.from_arrow(table).select(pl.col("a") + 1, pl.lit(2) * pl.col("b"))
    ours = bt.from_arrow(table).select(bt.col("a") + 1, bt.lit(2) * bt.col("b"))
    # Positional: a projection keeps its input's row order in both engines.
    assert ours.to_pydict() == theirs.to_dict(as_series=False)


def test_polars_names_and_row_shape_agree_on_aggregates(pl, table):
    frame = pl.from_arrow(table)
    collapsed = frame.select(pl.col("b").max(), pl.col("a").min(), pl.lit(1))
    ours = bt.from_arrow(table).select(bt.col("b").max(), bt.col("a").min(), bt.lit(1))
    assert ours.to_pydict() == collapsed.to_dict(as_series=False)

    broadcast = frame.select(pl.col("a"), pl.col("b").max())
    mixed = bt.from_arrow(table).select(bt.col("a"), bt.col("b").max())
    assert mixed.to_pydict() == broadcast.to_dict(as_series=False)


def test_polars_and_duckdb_differ_on_an_empty_sum(pl):
    """A decision, recorded: over zero rows DuckDB's `sum` is NULL and Polars' is 0.

    Batcher keeps the DuckDB answer. The name and the one-row shape still agree.
    """
    table = _TABLES["empty"]
    theirs = pl.from_arrow(table).select(pl.col("a").sum()).to_dict(as_series=False)
    ours = bt.from_arrow(table).select(bt.col("a").sum()).to_pydict()
    assert theirs == {"a": [0]}
    assert ours == {"a": [None]}


def test_polars_and_duckdb_differ_on_a_constant_only_select(pl):
    """A decision, recorded: ``SELECT 1 FROM t`` keeps t's rows in DuckDB, Polars gives one.

    Batcher keeps the DuckDB answer; the name is Polars' ``literal``.
    """
    table = _TABLES["duplicates"]
    theirs = pl.from_arrow(table).select(pl.lit(1)).to_dict(as_series=False)
    ours = bt.from_arrow(table).select(bt.lit(1)).to_pydict()
    assert theirs == {"literal": [1]}
    assert ours == {"literal": [1, 1, 1, 1]}
