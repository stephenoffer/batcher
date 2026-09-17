"""A ``when(...).then(...)`` without ``otherwise`` is SQL CASE WHEN ... END: NULL on no match.

The builder is accepted wherever an expression is -- a `select` value, an operand, an
accessor chain -- and ``otherwise(None)`` means the same thing. The NULL takes the type of
the first non-null branch value, so a string CASE stays a string column. DuckDB's bare CASE
is the oracle for values and types; Polars 1.40's ``when/then`` for the output name.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_TABLES = {
    "nulls": pa.table(
        {
            "a": pa.array([1, None, 3, -2], pa.int64()),
            "s": pa.array(["x", "y", None, "x"]),
            "f": pa.array([1.5, float("nan"), None, -0.0]),
        }
    ),
    "empty": pa.table(
        {
            "a": pa.array([], pa.int64()),
            "s": pa.array([], pa.string()),
            "f": pa.array([], pa.float64()),
        }
    ),
    "one_row": pa.table({"a": pa.array([5], pa.int64()), "s": ["z"], "f": [2.0]}),
    "duplicates": pa.table(
        {"a": pa.array([2, 2, 2], pa.int64()), "s": ["q", "q", "q"], "f": [0.0, 0.0, 0.0]}
    ),
}


@pytest.fixture(params=sorted(_TABLES))
def table(request) -> pa.Table:
    return _TABLES[request.param]


def test_bare_case_matches_duckdb(duck, table):
    duck.register("t", table)
    ds = bt.from_arrow(table)
    ours = ds.select(
        i=bt.when(bt.col("a") > 1).then(bt.col("a")),
        s=bt.when(bt.col("a") > 1).then(bt.col("s")).when(bt.col("a") < 0).then(bt.lit("neg")),
        f=bt.when(bt.col("a") > 1).then(bt.col("f")).otherwise(None),
        b=bt.when(bt.col("s") == "x").then(True),
    )
    expected = duck.sql(
        "SELECT CASE WHEN a > 1 THEN a END AS i, "
        "CASE WHEN a > 1 THEN s WHEN a < 0 THEN 'neg' END AS s, "
        "CASE WHEN a > 1 THEN f ELSE NULL END AS f, "
        "CASE WHEN s = 'x' THEN true END AS b FROM t"
    )
    out = ours.collect()
    assert [f.type for f in out.schema] == [pa.int64(), pa.string(), pa.float64(), pa.bool_()]
    assert_same(out, expected)


def test_a_null_then_is_typed_by_the_other_branches(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).select(s=bt.when(bt.col("a") > 1).then(None).otherwise(bt.col("s")))
    out = ours.collect()
    assert out.schema.field("s").type == pa.string()
    assert_same(out, duck.sql("SELECT CASE WHEN a > 1 THEN NULL ELSE s END AS s FROM t"))


def test_the_builder_composes_as_an_expression(duck, table):
    duck.register("t", table)
    case = bt.when(bt.col("a") > 1).then(bt.col("a"))
    ours = bt.from_arrow(table).select(
        (case + 1).alias("plus"),
        case.is_null().alias("missing"),
        bt.coalesce(case, bt.lit(0)).alias("filled"),
    )
    expected = duck.sql(
        "SELECT (CASE WHEN a > 1 THEN a END) + 1 AS plus, "
        "(CASE WHEN a > 1 THEN a END) IS NULL AS missing, "
        "coalesce(CASE WHEN a > 1 THEN a END, 0) AS filled FROM t"
    )
    assert_same(ours.collect(), expected)


def test_a_shared_prefix_extends_two_ways_independently():
    base = bt.when(bt.col("a") > 1).then(bt.lit("big"))
    small = base.when(bt.col("a") < 0).then(bt.lit("neg"))
    ds = bt.from_arrow(_TABLES["nulls"])
    out = ds.select(base=base, small=small).to_pydict()
    assert out == {"base": [None, None, "big", None], "small": [None, None, "big", "neg"]}


def test_polars_names_the_case_after_its_first_then(table):
    pl = pytest.importorskip("polars")
    theirs = pl.from_arrow(table).select(
        pl.when(pl.col("a") > 1).then(pl.col("s")), pl.when(pl.col("a") < 0).then(pl.col("f"))
    )
    ours = bt.from_arrow(table).select(
        bt.when(bt.col("a") > 1).then(bt.col("s")), bt.when(bt.col("a") < 0).then(bt.col("f"))
    )
    # Positional: a projection keeps its input's order in both engines.
    assert ours.to_pydict() == theirs.to_dict(as_series=False)
