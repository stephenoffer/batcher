"""`regexp_extract_all`'s capture-group argument — vs DuckDB.

`regexp_extract_all(s, pattern, group)` collects one *capture group* of every match,
not the whole match. The group index had nowhere to go: the engine's kernel called
`find_iter` unconditionally and the `.str` method took no group, so a query asking for
group 1 of `(\\d+)-(\\d+)` got the whole `100-200` back instead of `100`. Found by
running Spark's own documented `@ExpressionDescription` examples through `bt.sql`, then
confirmed against DuckDB.

The group is carried on the same `start` field the scalar `regexp_extract` already uses
for it, so no IR tag changed.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": pa.array(
                [
                    "100-200, 300-400",
                    "1-2",
                    "no pairs here",
                    "",
                    None,
                    "5-6, 7-8, 9-10",
                ]
            )
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.differential
@pytest.mark.parametrize("group", [0, 1, 2])
def test_capture_group_matches_duckdb(duck, t, group):
    out = (
        bt.from_arrow(t).select(r=col("s").str.regexp_extract_all(r"(\d+)-(\d+)", group)).collect()
    )
    expected = duck.sql(rf"SELECT regexp_extract_all(s, '(\d+)-(\d+)', {group}) r FROM t")
    assert_same(out, expected)


@pytest.mark.differential
@pytest.mark.parametrize("group", [0, 1, 2])
def test_capture_group_matches_duckdb_through_sql(duck, t, group):
    query = rf"SELECT regexp_extract_all(s, '(\d+)-(\d+)', {group}) r FROM t"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_a_group_that_did_not_participate_is_null_not_empty(duck, t):
    """DuckDB yields a NULL element for a non-participating alternation branch.

    This is where the list form and the scalar form genuinely differ: scalar
    `regexp_extract` returns `''` for the same case, so the list kernel cannot simply
    reuse it.
    """
    tbl = pa.table({"s": pa.array(["a1 b", "x", "1x"])})
    duck.register("u", tbl)
    query = r"SELECT regexp_extract_all(s, '(\d)|(x)', 2) r FROM u"
    assert_same(bt.sql(query, t=tbl, u=tbl).collect(), duck.sql(query))
    assert bt.sql(query, u=tbl).to_pydict()["r"][0] == [None]


@pytest.mark.differential
def test_a_group_the_pattern_does_not_have_is_rejected(t):
    """DuckDB errors rather than returning empty lists; so does the engine."""
    with pytest.raises(Exception, match="cannot access group"):
        bt.from_arrow(t).select(r=col("s").str.regexp_extract_all(r"(\d+)", 5)).collect()


@pytest.mark.differential
def test_the_default_is_still_the_whole_match(duck, t):
    """Omitting the group must keep the pre-existing behaviour exactly."""
    out = bt.from_arrow(t).select(r=col("s").str.regexp_extract_all(r"\d+")).collect()
    assert_same(out, duck.sql(r"SELECT regexp_extract_all(s, '\d+') r FROM t"))
