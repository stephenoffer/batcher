"""Choosing columns by type or by name pattern instead of listing them.

A selector resolves against the schema at plan time, so it works on a table whose column
list you do not know when you write the code. That is what makes a generic cleanup step, or
a "summarize every measure" aggregation, possible without reflection in Python. This file
applies selectors to TPC-H ``lineitem``; ``examples/expressions/column_selectors.py`` is the
vocabulary on a small table.

    python examples/expr_logic/column_selectors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    lineitem = tpch("lineitem")

    # By type.
    numbers = lineitem.select(bt.numeric())
    strings = lineitem.select(bt.string())
    dates = lineitem.select(bt.temporal())
    print("numeric:", numbers.columns)
    print("string:", strings.columns)
    print("temporal:", dates.columns)

    # The three families are disjoint and together cover the table.
    assert set(numbers.columns).isdisjoint(strings.columns)
    assert len(numbers.columns) + len(strings.columns) + len(dates.columns) == lineitem.width

    # By name.
    prefixed = lineitem.select(bt.matches(r"^l_ship"))
    print("l_ship*:", prefixed.columns)
    assert set(prefixed.columns) == {"l_shipdate", "l_shipinstruct", "l_shipmode"}

    # Everything except a few.
    trimmed = lineitem.select(bt.exclude("l_comment", "l_shipinstruct"))
    assert "l_comment" not in trimmed.columns
    assert trimmed.width == lineitem.width - 2

    # By exact type. The requested type is widened the way ingest widens a column, so
    # `pa.float32()` would name these float64 columns too.
    floats = lineitem.select(bt.by_dtype("float64"))
    assert set(floats.columns) == {"l_extendedprice", "l_discount", "l_tax"}

    # A selector composes with an expression, so "round every float" is one line.
    rounded = lineitem.with_columns(bt.floating().round(1))
    assert rounded.columns == lineitem.columns

    # In an aggregation a selector expands to one aggregate per matched column, and the
    # group key is never aggregated over. `.name` keeps several aggregates apart.
    measures = bt.floating() | bt.col("l_quantity")
    summary = (
        lineitem.group_by("l_returnflag")
        .agg(
            measures.sum().name.suffix("_sum"),
            measures.mean().name.suffix("_avg"),
            bt.count().alias("n"),
        )
        .sort("l_returnflag")
    )
    print("summary:", summary.columns)
    assert summary.columns == [
        "l_returnflag",
        "l_quantity_sum",
        "l_extendedprice_sum",
        "l_discount_sum",
        "l_tax_sum",
        "l_quantity_avg",
        "l_extendedprice_avg",
        "l_discount_avg",
        "l_tax_avg",
        "n",
    ]
    by_hand = (
        lineitem.group_by("l_returnflag")
        .agg(q=bt.col("l_quantity").sum(), n=bt.count())
        .sort("l_returnflag")
        .to_pydict()
    )
    table = summary.to_pydict()
    assert table["l_quantity_sum"] == by_hand["q"]
    assert table["n"] == by_hand["n"]
    assert sum(table["n"]) == lineitem.count()

    # One alias cannot name four columns, so that is refused when the query is written.
    try:
        lineitem.group_by("l_returnflag").agg(bt.floating().sum().alias("total"))
    except bt.PlanError as err:
        assert "names a single column" in str(err)
    else:
        raise AssertionError("an alias over several columns must be refused")

    # The name-taking verbs accept a selector too.
    long = lineitem.select("l_orderkey", "l_linenumber", "l_discount", "l_tax").unpivot(
        index=["l_orderkey", "l_linenumber"], on=bt.floating()
    )
    assert long.count() == 2 * lineitem.count()
    assert lineitem.drop_nulls(subset=bt.temporal()).count() == lineitem.count()


if __name__ == "__main__":
    main()
