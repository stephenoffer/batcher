"""Left-join elimination vs DuckDB — dropping a redundant left join to a unique,
unused dimension must not change the result (every left row is preserved, and a
left row with no dimension match still appears)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def test_left_join_to_groupby_unused(duck):
    from conftest import assert_same

    fact = pa.table({"k": [1, 2, 2, 3, 9], "v": [10, 20, 30, 40, 50]})
    dimraw = pa.table({"k": [1, 1, 2, 2, 3], "p": [5, 6, 7, 8, 9]})
    duck.register("fact", fact)
    duck.register("dimraw", dimraw)
    dim = bt.from_arrow(dimraw).group_by("k").agg(tot=col("p").sum())
    out = bt.from_arrow(fact).join(dim, on="k", how="left").select("k", "v").collect()
    # k=9 has no dimension row → still present (left-preserved), proving correctness.
    expected = duck.sql(
        "SELECT fact.k, fact.v FROM fact "
        "LEFT JOIN (SELECT k, sum(p) AS tot FROM dimraw GROUP BY k) d ON fact.k = d.k"
    )
    assert_same(out, expected)


def test_left_join_distinct_unused(duck):
    from conftest import assert_same

    fact = pa.table({"k": [1, 2, 2, 5], "v": [1, 2, 3, 4]})
    dimraw = pa.table({"k": [1, 1, 2, 3]})
    duck.register("f2", fact)
    duck.register("d2", dimraw)
    dim = bt.from_arrow(dimraw).distinct()
    out = bt.from_arrow(fact).join(dim, on="k", how="left").select("k", "v").collect()
    expected = duck.sql(
        "SELECT f2.k, f2.v FROM f2 LEFT JOIN (SELECT DISTINCT k FROM d2) d ON f2.k = d.k"
    )
    assert_same(out, expected)


def test_derived_projection_over_eliminated_join(duck):
    from conftest import assert_same

    fact = pa.table({"k": [1, 2, 3], "v": [10, 20, 30]})
    dimraw = pa.table({"k": [1, 2, 2, 3], "p": [1, 2, 3, 4]})
    duck.register("f3", fact)
    duck.register("d3", dimraw)
    dim = bt.from_arrow(dimraw).group_by("k").agg(tot=col("p").sum())
    # A derived expression over only L columns must still be rewritten correctly.
    out = (
        bt.from_arrow(fact)
        .join(dim, on="k", how="left")
        .select("k", doubled=col("v") * 2)
        .collect()
    )
    expected = duck.sql(
        "SELECT f3.k, f3.v * 2 AS doubled FROM f3 "
        "LEFT JOIN (SELECT k, sum(p) AS tot FROM d3 GROUP BY k) d ON f3.k = d.k"
    )
    assert_same(out, expected)
