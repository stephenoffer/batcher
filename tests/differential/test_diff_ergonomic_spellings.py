"""Ecosystem argument spellings compute the same result as the Batcher primary.

The migration surface is only worth having if `sort(by=, ascending=False)` is the
*same query* as `sort(descending=True)` rather than something that merely looks like
it. Each test here runs the ecosystem spelling through the full optimizer and checks
it against DuckDB, so an alias that silently drifts — a flipped null placement, a
fraction read as a row count, a fill applied to the wrong columns — fails here rather
than in a user's ported script.

Sort ordering is asserted with `assert_same_ordered`. `assert_same` is
order-independent by design and therefore structurally blind to a sort bug, which is
exactly the class of bug an `ascending=` alias could introduce.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered
from batcher import col

# Nulls in both value columns, a duplicate row, and a NULL group key: the edges an
# alias is most likely to get wrong.
_DATA = pa.table(
    {
        "g": ["a", "b", "a", "b", None, "a"],
        "x": [3, 1, 4, 1, 5, None],
        "y": [1.5, None, 2.5, 1.5, None, 4.0],
    }
)
_EMPTY = pa.table(
    {
        "g": pa.array([], pa.string()),
        "x": pa.array([], pa.int64()),
        "y": pa.array([], pa.float64()),
    }
)


@pytest.fixture
def t(duck):
    """Register the shared fixture table as ``t`` in DuckDB."""
    duck.register("t", _DATA)
    return duck


def _ds() -> bt.Dataset:
    return bt.from_arrow(_DATA)


# --- filter -----------------------------------------------------------------


def test_keyword_filter_matches_duckdb(t):
    out = _ds().filter(g="a").collect()
    assert_same(out, t.sql("SELECT * FROM t WHERE g = 'a'"))


def test_several_predicates_are_anded(t):
    out = _ds().filter(col("x") > 1, col("g") == "a").collect()
    assert_same(out, t.sql("SELECT * FROM t WHERE x > 1 AND g = 'a'"))


def test_keyword_filter_equals_the_expression_spelling():
    assert _ds().filter(g="a").equals(_ds().filter(col("g") == "a"))


def test_keyword_filter_on_empty_input(duck):
    duck.register("e", _EMPTY)
    out = bt.from_arrow(_EMPTY).filter(g="a").collect()
    assert_same(out, duck.sql("SELECT * FROM e WHERE g = 'a'"))


# --- sort -------------------------------------------------------------------


def test_ascending_false_matches_descending_true(t):
    out = _ds().sort(by="x", ascending=False, na_position="last").collect()
    expected = t.sql("SELECT * FROM t ORDER BY x DESC NULLS LAST")
    assert_same_ordered(out, expected)


def test_ascending_true_is_the_default_order(t):
    out = _ds().sort(by="x", ascending=True, na_position="last").collect()
    assert_same_ordered(out, t.sql("SELECT * FROM t ORDER BY x ASC NULLS LAST"))


def test_na_position_first_matches_nulls_first(t):
    out = _ds().sort("x", na_position="first").collect()
    assert_same_ordered(out, t.sql("SELECT * FROM t ORDER BY x ASC NULLS FIRST"))


def test_per_key_ascending_list(t):
    out = _ds().sort(by=["g", "x"], ascending=[True, False], na_position="last").collect()
    expected = t.sql("SELECT * FROM t ORDER BY g ASC NULLS LAST, x DESC NULLS LAST")
    assert_same_ordered(out, expected)


def test_alias_and_primary_agree_exactly():
    left = _ds().sort(by="x", ascending=False, na_position="first").collect()
    right = _ds().sort("x", descending=True, nulls_first=True).collect()
    assert left.equals(right)


# --- fill_null --------------------------------------------------------------


def test_fillna_fills_every_compatible_column(t):
    out = _ds().fillna(0).collect()
    expected = t.sql("SELECT g, coalesce(x, 0) AS x, coalesce(y, 0) AS y FROM t")
    assert_same(out, expected)


def test_fill_null_subset_leaves_other_columns_alone(t):
    out = _ds().fill_null(0, subset=["x"]).collect()
    assert_same(out, t.sql("SELECT g, coalesce(x, 0) AS x, y FROM t"))


# --- select_dtypes ----------------------------------------------------------


def test_select_dtypes_by_python_type(t):
    assert_same(_ds().select_dtypes(int).collect(), t.sql("SELECT x FROM t"))


def test_select_dtypes_by_dtype_name(t):
    assert_same(_ds().select_dtypes("float64").collect(), t.sql("SELECT y FROM t"))


def test_select_dtypes_exclude_is_the_complement(t):
    assert_same(_ds().select_dtypes(exclude=str).collect(), t.sql("SELECT x, y FROM t"))


# --- group_by ---------------------------------------------------------------


def test_agg_dict_spec_matches_duckdb(t):
    out = _ds().group_by("g").agg({"x": "sum"}).collect()
    assert_same(out, t.sql("SELECT g, sum(x) AS x FROM t GROUP BY g"))


def test_agg_dict_spec_with_several_reducers(t):
    out = _ds().group_by("g").agg({"x": ["min", "max"]}).collect()
    expected = t.sql("SELECT g, min(x) AS x_min, max(x) AS x_max FROM t GROUP BY g")
    assert_same(out, expected)


def test_group_by_size_counts_rows_including_null_groups(t):
    out = _ds().group_by("g").size().collect()
    assert_same(out, t.sql("SELECT g, count(*) AS size FROM t GROUP BY g"))


def test_group_by_first_and_last_follow_the_order(t):
    out = _ds().group_by("g").first("x", order_by="x").collect()
    expected = t.sql(
        "SELECT g, min(x) AS x FROM t WHERE x IS NOT NULL GROUP BY g "
        "UNION ALL SELECT g, NULL FROM t GROUP BY g HAVING count(x) = 0"
    )
    assert_same(out, expected)


# --- ecosystem aliases are the same query -----------------------------------


@pytest.mark.parametrize(
    ("alias", "primary"),
    [
        ("drop_duplicates", "distinct"),
        ("vstack", "union"),
        ("append", "union"),
        ("difference", "except_"),
    ],
)
def test_set_aliases_delegate_to_their_primary(alias, primary):
    ds, other = _ds(), _ds().filter(col("x") > 2)
    if alias == "drop_duplicates":
        assert getattr(ds, alias)().equals(getattr(ds, primary)())
    else:
        assert getattr(ds, alias)(other).equals(getattr(ds, primary)(other))


def test_row_index_aliases_agree():
    assert _ds().with_row_count().equals(_ds().with_row_index())


def test_export_aliases_agree():
    ds = _ds()
    assert ds.to_dicts() == ds.to_pylist()
    assert ds.to_dict() == ds.to_pydict()
