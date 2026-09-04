"""A list of columns is accepted wherever a verb takes varargs.

Polars ``df.select(["a", "b"])``, PySpark ``df.orderBy(["a", "b"])`` and Ray Data
``ds.sort(["a"])`` all pass a *list* into a varargs position. Batcher spells those
verbs ``*columns``, so the list used to arrive as one key and every verb raised —
``sort(["a", "b"])`` reported "expected a column name or expression, got list".
Thirteen verbs failed that way at once, which is a migration blocker on the first
line of a ported script.

These pin the flattening (`api._varargs.flatten_varargs`) at each entry point, that
it composes with the existing spellings rather than replacing them, and the two
places it deliberately does NOT apply: `grouping_sets`, where a sequence per
argument is the meaning, and genuinely wrong argument types, which must still raise.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def ds():
    return bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3], "y": [10, 20, 30]})


# --- the flattening itself ------------------------------------------------


@pytest.mark.unit
def test_flatten_varargs_expands_one_level():
    from batcher.api._varargs import flatten_varargs

    assert flatten_varargs((["a", "b"],)) == ("a", "b")
    assert flatten_varargs(("a", ["b", "c"])) == ("a", "b", "c")
    assert flatten_varargs((("a", "b"),)) == ("a", "b")
    gen = (x for x in ("a", "b"))
    assert flatten_varargs((gen,)) == ("a", "b")


@pytest.mark.unit
def test_flatten_varargs_leaves_a_flat_tuple_identical():
    # The common spelling must not pay an allocation.
    from batcher.api._varargs import flatten_varargs

    args = ("a", "b")
    assert flatten_varargs(args) is args


@pytest.mark.unit
def test_flatten_varargs_never_iterates_an_expression_or_dataset():
    # `Expr.__iter__` raises by design and `Dataset.__iter__` executes the plan, so a
    # duck-typed "is it iterable" check would either explode or silently run a query.
    from batcher.api._varargs import flatten_varargs

    e = bt.col("x")
    d = bt.from_pydict({"x": [1]})
    assert flatten_varargs((e, d)) == (e, d)


# --- every verb that takes varargs ---------------------------------------


@pytest.mark.unit
def test_select_accepts_a_list(ds):
    assert ds.select(["g", "x"]).collect().column_names == ["g", "x"]


@pytest.mark.unit
def test_select_mixes_a_list_with_bare_names(ds):
    assert ds.select("g", ["x", "y"]).collect().column_names == ["g", "x", "y"]


@pytest.mark.unit
def test_select_accepts_a_generator(ds):
    got = ds.select(bt.col(c) for c in ("g", "x"))
    assert got.collect().column_names == ["g", "x"]


@pytest.mark.unit
def test_with_columns_accepts_a_list(ds):
    got = ds.with_columns([bt.col("x").alias("k"), bt.col("y").alias("m")])
    assert set(got.collect().column_names) == {"g", "x", "y", "k", "m"}


@pytest.mark.unit
def test_filter_accepts_a_list(ds):
    # A list of predicates is ANDed, exactly as several positional ones are.
    assert ds.filter([bt.col("x") > 1, bt.col("y") < 30]).collect().num_rows == 1


@pytest.mark.unit
def test_sort_accepts_a_list(ds):
    assert ds.sort(["g", "x"]).to_pydict()["x"] == [1, 2, 3]


@pytest.mark.unit
def test_sort_accepts_a_list_with_per_key_descending(ds):
    # The `descending` list must line up with the *flattened* key count, which is the
    # case that used to fail with "descending list has 2 entries but there are 1 keys".
    got = ds.sort(["g", "x"], descending=[False, True]).to_pydict()
    assert got["x"] == [2, 1, 3]


@pytest.mark.unit
def test_group_by_accepts_a_list(ds):
    got = ds.group_by(["g"]).agg(s=bt.col("x").sum()).sort("g").to_pydict()
    assert got == {"g": ["a", "b"], "s": [3, 3]}


@pytest.mark.unit
def test_groupby_pandas_spelling_accepts_a_list(ds):
    got = ds.groupby(["g"]).agg(s=bt.col("x").sum()).sort("g").to_pydict()
    assert got == {"g": ["a", "b"], "s": [3, 3]}


@pytest.mark.unit
def test_rollup_and_cube_accept_a_list(ds):
    assert ds.rollup(["g"]).agg(n=bt.count()).collect().num_rows == 3
    assert ds.cube(["g"]).agg(n=bt.count()).collect().num_rows == 3


@pytest.mark.unit
def test_agg_accepts_a_list(ds):
    assert ds.agg([bt.sum("x")]).to_pydict() == {"x": [6]}


@pytest.mark.unit
def test_group_by_agg_accepts_a_list(ds):
    got = ds.group_by("g").agg([bt.sum("x")]).sort("g").to_pydict()
    assert got == {"g": ["a", "b"], "x": [3, 3]}


@pytest.mark.unit
def test_drop_accepts_a_list(ds):
    assert ds.drop(["y"]).collect().column_names == ["g", "x"]


@pytest.mark.unit
def test_union_accepts_a_list(ds):
    assert ds.union([ds]).collect().num_rows == 6


@pytest.mark.unit
def test_unnest_accepts_a_list():
    ds = bt.from_pydict({"s": [{"a": 1, "b": 2}]})
    assert set(ds.unnest(["s"]).collect().column_names) == {"a", "b"}


# --- what must NOT change -------------------------------------------------


@pytest.mark.unit
def test_grouping_sets_still_reads_each_argument_as_one_level(ds):
    # Here a sequence per argument IS the meaning, so flattening would collapse the
    # levels into one. Two levels plus the grand total over two groups = 3 rows.
    out = ds.grouping_sets(["g"], []).agg(n=bt.col("x").sum())
    assert sorted(out.to_pydict()["n"]) == [3, 3, 6]


@pytest.mark.unit
def test_a_wrong_argument_type_still_raises(ds):
    # Flattening turns a *sequence* into arguments; it must not make garbage legal.
    with pytest.raises(PlanError, match="positional select"):
        ds.select(3)
    with pytest.raises(PlanError, match="expected a column name or expression"):
        ds.sort(3)


@pytest.mark.unit
def test_an_empty_list_is_still_an_empty_call(ds):
    with pytest.raises(PlanError):
        ds.select([])
    with pytest.raises(PlanError, match="requires at least one key"):
        ds.sort([])
