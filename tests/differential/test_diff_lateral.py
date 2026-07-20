"""`LATERAL (SELECT ...)` with no FROM, vs DuckDB.

A lateral subquery is evaluated once per outer row and may read that row's columns. When it
has **no FROM of its own** it yields exactly one row per outer row computed from those
columns — which is precisely `with_columns`, no join and no correlation machinery.
Previously any LATERAL raised ``unknown table ''``.

A lateral that *does* read a table is a correlated join (a varying row count per outer row)
and must reject. The final test is the important one: it covers a correlated lateral with
**no WHERE**, which an earlier version of the check let through — sqlglot spells the FROM
key `from_` in some versions and `from` in others, and reading only one key meant the
lateral was silently mistranslated into computed columns.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


def _norm(d):
    n = len(next(iter(d.values()))) if d else 0
    return sorted([tuple(c[i] for c in d.values()) for i in range(n)], key=str)


@pytest.fixture
def tu(duck):
    t = pa.table({"k": [1, 2, 3], "v": [10, 20, None]})
    u = pa.table({"k": [1, 1, 2], "w": [5, 6, 7]})
    duck.register("t", t)
    duck.register("u", u)
    return t, u


@pytest.mark.differential
@pytest.mark.parametrize(
    "lateral",
    [
        "LATERAL (SELECT v * 2 AS x)",
        "LATERAL (SELECT v * 2 AS x, v + 1 AS y)",
        "LATERAL (SELECT k + v AS x)",
        # NULL in the outer row must propagate, not error.
        "LATERAL (SELECT coalesce(v, 0) AS x)",
    ],
)
def test_lateral_without_from_is_computed_columns(duck, tu, lateral):
    """Each lateral projection becomes a column on the outer relation."""
    t, u = tu
    query = f"SELECT * FROM t, {lateral}"
    got = bt.sql(query, t=t, u=u).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert _norm(got) == _norm(exp)


@pytest.mark.differential
def test_cross_join_lateral_spelling(duck, tu):
    """`CROSS JOIN LATERAL` means the same as the comma form."""
    t, u = tu
    query = "SELECT k, x FROM t CROSS JOIN LATERAL (SELECT v * 3 AS x)"
    got = bt.sql(query, t=t, u=u).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert _norm(got) == _norm(exp)


@pytest.mark.differential
def test_lateral_row_count_is_unchanged(duck, tu):
    """A no-FROM lateral is one row in, one row out — never a fan-out."""
    t, u = tu
    query = "SELECT * FROM t, LATERAL (SELECT v * 2 AS x)"
    got = bt.sql(query, t=t, u=u).collect()
    assert got.num_rows == t.num_rows
    assert _norm(got.to_pydict()) == _norm(duck.sql(query).to_arrow_table().to_pydict())


@pytest.mark.parametrize(
    "lateral",
    [
        # With a WHERE — the shape that always rejected.
        "LATERAL (SELECT max(w) AS mw FROM u WHERE u.k = t.k) s",
        # WITHOUT a WHERE — the near-miss: only the FROM check catches this one, and it
        # would otherwise have been mistranslated into a computed column.
        "LATERAL (SELECT max(w) AS mw FROM u) s",
    ],
)
def test_correlated_lateral_rejects(tu, lateral):
    """A lateral that reads a table is a correlated join and must not be faked."""
    t, u = tu
    with pytest.raises(NotImplementedError, match="no FROM"):
        bt.sql(f"SELECT t.k FROM t, {lateral}", t=t, u=u).collect()


@pytest.mark.differential
def test_unaliased_lateral_expression_takes_its_text_as_the_name(duck, tu):
    """An unaliased expression is named after its SQL text, and its VALUES match DuckDB.

    DuckDB parenthesises the derived name (`(v * 2)`) where this gives `v * 2` — a
    pre-existing auto-naming convention difference shared with other derived columns, not a
    difference in the data. The values are compared positionally for that reason.
    """
    t, u = tu
    query = "SELECT * FROM t, LATERAL (SELECT v * 2)"
    got = bt.sql(query, t=t, u=u).collect()
    exp = duck.sql(query).to_arrow_table()
    assert got.num_rows == exp.num_rows
    assert [list(got.column(i).to_pylist()) for i in range(got.num_columns)] == [
        list(exp.column(i).to_pylist()) for i in range(exp.num_columns)
    ]
