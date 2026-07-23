"""The Python-ergonomics contract of `Dataset` and `GroupBy`.

These cover the surface a user reaches for by habit rather than by reading the
reference: ecosystem argument spellings (`sort(by=, ascending=)`,
`melt(id_vars=)`), the Python protocols (`np.asarray`, the DataFrame Interchange
Protocol), and — most importantly — the *messages*. A migrant meets Batcher through
its errors, so the guidance text is asserted here as behaviour, not decoration: if
`ds.set_index(...)` ever degrades back to a bare `AttributeError`, that is a
regression in the thing this file exists to protect.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


@pytest.fixture
def ds() -> bt.Dataset:
    """A small mixed-type dataset: an int, a nullable float, and a string key."""
    return bt.from_pydict(
        {"x": [1, 2, 3, 4], "y": [10.0, 20.0, None, 40.0], "g": ["a", "b", "a", "b"]}
    )


# --- filter ---------------------------------------------------------------------


def test_filter_keyword_is_equality_shorthand(ds: bt.Dataset) -> None:
    assert ds.filter(g="a").to_pydict()["x"] == [1, 3]


def test_filter_ands_several_predicates(ds: bt.Dataset) -> None:
    got = ds.filter(bt.col("x") > 1, bt.col("x") < 4).to_pydict()["x"]
    assert got == [2, 3]


def test_filter_mixes_predicates_and_keywords(ds: bt.Dataset) -> None:
    assert ds.filter(bt.col("x") > 1, g="a").to_pydict()["x"] == [3]


def test_filter_without_a_condition_says_what_to_pass(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match=r"filter\(col\('x'\) > 0\)"):
        ds.filter()


def test_filter_keyword_naming_an_unknown_column_is_rejected(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="unknown column"):
        ds.filter(nope=1)


def test_filter_with_a_string_points_at_sql(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match=r"ds\.sql"):
        ds.filter("x > 1")


# --- sample ---------------------------------------------------------------------


def test_sample_reads_a_positional_int_as_a_row_count(ds: bt.Dataset) -> None:
    assert ds.sample(2, seed=1).count() == 2


def test_sample_reads_a_positional_float_as_a_fraction(ds: bt.Dataset) -> None:
    assert ds.sample(0.0, seed=1).count() == 0


def test_sample_accepts_the_pandas_aliases(ds: bt.Dataset) -> None:
    assert ds.sample(n=2, random_state=7).count() == 2
    assert ds.sample(frac=0.0, seed=7).count() == 0


def test_sample_rejects_a_count_and_a_fraction_together(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="not both"):
        ds.sample(2, n=2)


# --- sort -----------------------------------------------------------------------


def test_sort_accepts_the_pandas_by_and_ascending(ds: bt.Dataset) -> None:
    # Deliberately order-dependent: an order-independent comparison cannot see a
    # sort bug, which is exactly the failure this asserts against.
    assert ds.sort(by="x", ascending=False).to_pydict()["x"] == [4, 3, 2, 1]


def test_sort_ascending_matches_descending(ds: bt.Dataset) -> None:
    assert (
        ds.sort("x", descending=True).to_pydict()["x"]
        == ds.sort(by="x", ascending=False).to_pydict()["x"]
    )


def test_sort_na_position_puts_nulls_first(ds: bt.Dataset) -> None:
    assert ds.sort("y", na_position="first").to_pydict()["y"][0] is None


def test_sort_rejects_conflicting_spellings(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="not both"):
        ds.sort("x", descending=True, ascending=False)


def test_sort_rejects_a_bad_na_position(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="'first' or 'last'"):
        ds.sort("x", na_position="middle")


# --- reshape and selection spellings ---------------------------------------------


def test_melt_accepts_the_pandas_argument_names(ds: bt.Dataset) -> None:
    wide = ds.select("x", "y")
    assert wide.melt(id_vars="x").columns == wide.melt(index=["x"]).columns


def test_select_dtypes_accepts_python_types_and_dtype_names(ds: bt.Dataset) -> None:
    assert ds.select_dtypes(int).columns == ["x"]
    assert ds.select_dtypes("int64").columns == ["x"]
    assert ds.select_dtypes([int, float]).columns == ["x", "y"]


def test_select_dtypes_exclude_is_the_complement(ds: bt.Dataset) -> None:
    assert ds.select_dtypes(exclude="string").columns == ["x", "y"]


def test_select_dtypes_needs_exactly_one_of_include_exclude(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="exactly one"):
        ds.select_dtypes()


def test_rename_applies_a_callable_to_every_column(ds: bt.Dataset) -> None:
    assert ds.rename(str.upper).columns == ["X", "Y", "G"]


def test_rename_rejects_a_callable_that_collides_names(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="unique"):
        ds.rename(lambda _: "same")


# --- null filling ----------------------------------------------------------------


def test_fillna_fills_only_the_columns_that_can_hold_the_value(ds: bt.Dataset) -> None:
    # The regression this guards: filling a mixed frame used to reach Rust and fail
    # with "arguments need to have the same data type", naming no column at all.
    out = ds.fillna(0).to_pydict()
    assert out["y"] == [10.0, 20.0, 0.0, 40.0]
    assert out["g"] == ["a", "b", "a", "b"]


def test_fill_null_with_an_explicit_subset_rejects_an_incompatible_column(
    ds: bt.Dataset,
) -> None:
    with pytest.raises(PlanError, match="cannot be filled"):
        ds.fill_null(0, subset=["g"])


def test_fill_null_reports_when_no_column_can_take_the_value(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="no column can be filled"):
        ds.select("g").fill_null(0)


def test_fill_null_mapping_and_subset_together_is_rejected(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="not both"):
        ds.fill_null({"y": 0.0}, subset=["y"])


# --- row-oriented consumers -------------------------------------------------------


def test_iter_rows_yields_tuples_then_dicts(ds: bt.Dataset) -> None:
    assert next(iter(ds.iter_rows())) == (1, 10.0, "a")
    assert next(iter(ds.iter_rows(named=True))) == {"x": 1, "y": 10.0, "g": "a"}


def test_iter_slices_covers_every_row(ds: bt.Dataset) -> None:
    assert sum(s.num_rows for s in ds.iter_slices()) == 4


def test_first_and_last_follow_the_sort(ds: bt.Dataset) -> None:
    assert ds.sort("x").first() == (1, 10.0, "a")
    assert ds.sort("x").last() == (4, 40.0, "b")


def test_first_of_an_empty_result_is_none(ds: bt.Dataset) -> None:
    assert ds.filter(bt.col("x") > 100).first() is None


def test_item_returns_the_single_scalar(ds: bt.Dataset) -> None:
    assert ds.agg(total=bt.col("x").sum()).item() == 10


def test_item_refuses_a_multi_row_result(ds: bt.Dataset) -> None:
    # Returning row zero here is the "worked in the demo" bug this guards against.
    with pytest.raises(PlanError, match="more than one"):
        ds.select("x").item()


def test_item_refuses_an_empty_result(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="empty"):
        ds.filter(bt.col("x") > 100).select("x").item()


def test_item_needs_a_column_when_the_result_is_wide(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="single-column"):
        ds.limit(1).item()


# --- introspection ----------------------------------------------------------------


def test_width_height_and_empty(ds: bt.Dataset) -> None:
    assert (ds.width, ds.height, ds.empty) == (3, 4, False)


def test_collect_schema_is_an_ordered_mapping(ds: bt.Dataset) -> None:
    assert [str(t) for t in ds.collect_schema().values()] == ["int64", "double", "string"]


def test_memory_usage_scales_with_row_count(ds: bt.Dataset) -> None:
    assert ds.memory_usage()["x"] == 4 * 8


def test_equals_compares_results_not_plans(ds: bt.Dataset) -> None:
    assert ds.equals(ds.filter(bt.col("x") > 0))
    assert not ds.equals(ds.filter(bt.col("x") > 1))


def test_equals_ignores_row_order_unless_asked(ds: bt.Dataset) -> None:
    assert ds.sort("x").equals(ds.sort("x", descending=True))
    assert not ds.sort("x").equals(ds.sort("x", descending=True), ordered=True)


def test_equals_is_false_for_different_columns(ds: bt.Dataset) -> None:
    assert not ds.equals(ds.select("x"))


# --- ecosystem spellings and protocols ---------------------------------------------


def test_ecosystem_aliases_match_their_primaries(ds: bt.Dataset) -> None:
    assert ds.to_dicts() == ds.to_pylist()
    assert ds.to_dict() == ds.to_pydict()
    assert ds.drop_duplicates().count() == ds.distinct().count()
    assert ds.with_row_count().columns == ds.with_row_index().columns
    assert ds.vstack(ds).count() == ds.union(ds).count()
    assert ds.append(ds).count() == ds.union(ds).count()
    assert ds.difference(ds).count() == ds.except_(ds).count()


def test_lazy_and_copy_are_identity_because_a_dataset_is_lazy_and_immutable(
    ds: bt.Dataset,
) -> None:
    assert ds.lazy() is ds
    assert ds.copy() is ds


def test_transform_is_pipe(ds: bt.Dataset) -> None:
    assert ds.transform(lambda d: d.count()) == 4


def test_numpy_array_protocol(ds: bt.Dataset) -> None:
    assert np.asarray(ds.select("x", "y")).shape == (4, 2)


def test_dataframe_interchange_protocol(ds: bt.Dataset) -> None:
    pd = pytest.importorskip("pandas")
    assert pd.api.interchange.from_dataframe(ds).shape == (4, 3)


# --- attribute guidance -------------------------------------------------------------


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("set_index", "no row index"),
        ("reset_index", "no row index"),
        ("iloc", "positional indexer"),
        ("iterrows", "iter_rows"),
        ("T", "Transposing"),
        ("rolling", "ds.window"),
        ("cumsum", "ds.window"),
        ("resample", "group_by"),
        ("apply", "map_batches"),
        ("toPandas", "to_pandas"),
        ("withColumn", "with_columns"),
    ],
)
def test_absent_ecosystem_apis_explain_themselves(
    ds: bt.Dataset, attribute: str, expected: str
) -> None:
    with pytest.raises(AttributeError, match=expected):
        getattr(ds, attribute)


def test_a_column_accessed_as_an_attribute_names_the_two_spellings(ds: bt.Dataset) -> None:
    with pytest.raises(AttributeError, match=r"ds\['x'\]"):
        _ = ds.x


def test_a_misspelled_method_gets_a_suggestion(ds: bt.Dataset) -> None:
    with pytest.raises(AttributeError, match="Did you mean 'filter'"):
        _ = ds.filtr


def test_private_lookups_stay_plain_so_copy_and_pickle_still_work(ds: bt.Dataset) -> None:
    import copy
    import pickle

    # `__getattr__` must not decorate dunder probes: copy/pickle look for
    # `__deepcopy__`/`__getstate__` and treat a decorated failure as a hard error.
    assert copy.deepcopy(ds).columns == ds.columns
    assert pickle.loads(pickle.dumps(ds)).columns == ds.columns
    assert not hasattr(ds, "definitely_not_here")


# --- GroupBy -------------------------------------------------------------------------


def test_group_by_keys(ds: bt.Dataset) -> None:
    assert ds.group_by("g").keys == ["g"]


def test_group_by_is_not_iterable(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="not iterable"):
        list(ds.group_by("g"))


def test_agg_accepts_a_pandas_dict_spec(ds: bt.Dataset) -> None:
    got = ds.group_by("g").agg({"x": "sum"}).sort("g").to_pydict()
    assert got == {"g": ["a", "b"], "x": [4, 6]}


def test_agg_dict_with_several_reducers_suffixes_the_names(ds: bt.Dataset) -> None:
    got = ds.group_by("g").agg({"x": ["min", "max"]}).sort("g").to_pydict()
    assert got == {"g": ["a", "b"], "x_min": [1, 2], "x_max": [3, 4]}


def test_agg_dict_rejects_an_unknown_column(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="unknown column"):
        ds.group_by("g").agg({"nope": "sum"})


def test_agg_dict_rejects_an_unknown_reducer(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="not an aggregate"):
        ds.group_by("g").agg({"x": "blah"})


def test_group_by_nunique_and_size_are_the_pandas_spellings(ds: bt.Dataset) -> None:
    assert ds.group_by("g").nunique().sort("g").to_pydict() == (
        ds.group_by("g").n_unique().sort("g").to_pydict()
    )
    assert ds.group_by("g").size().sort("g").to_pydict()["size"] == [2, 2]


def test_group_by_first_and_last_follow_the_order_by(ds: bt.Dataset) -> None:
    got = ds.group_by("g").first("x", order_by="x").sort("g").to_pydict()
    assert got == {"g": ["a", "b"], "x": [1, 2]}
    got = ds.group_by("g").last("x", order_by="x").sort("g").to_pydict()
    assert got == {"g": ["a", "b"], "x": [3, 4]}


def test_group_by_first_requires_an_explicit_order(ds: bt.Dataset) -> None:
    # Without an order "first" is whichever morsel arrived first — not a result.
    with pytest.raises(TypeError, match="order_by"):
        ds.group_by("g").first("x")


# --- second-wave spellings: query, explain, writer shortcuts ------------------------


def test_query_is_a_sql_where_clause(ds: bt.Dataset) -> None:
    assert ds.query("x > 2").to_pydict()["x"] == [3, 4]


def test_query_matches_the_expression_spelling(ds: bt.Dataset) -> None:
    assert ds.query("x > 2").equals(ds.filter(bt.col("x") > 2))


def test_explain_accepts_tree_as_an_alias_for_text(ds: bt.Dataset) -> None:
    assert ds.explain(format="tree") == ds.explain(format="text")


def test_explain_still_rejects_an_unknown_format(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="not supported"):
        ds.explain(format="yaml")


@pytest.mark.parametrize("fmt", ["csv", "parquet", "json"])
def test_pandas_writer_shortcuts_round_trip(ds: bt.Dataset, tmp_path, fmt: str) -> None:
    out = str(tmp_path / f"out.{fmt}")
    getattr(ds, f"to_{fmt}")(out)
    assert getattr(bt.read, fmt)(out).count() == 4


# --- third-wave spellings: drop/rename/cast/join/pivot keywords -----------------------


def test_drop_accepts_the_pandas_columns_and_labels_keywords(ds: bt.Dataset) -> None:
    assert ds.drop(columns=["x"]).columns == ["y", "g"]
    assert ds.drop(labels="x").columns == ["y", "g"]


def test_drop_mixes_positional_and_keyword_targets(ds: bt.Dataset) -> None:
    assert ds.drop("x", columns=["y"]).columns == ["g"]


def test_drop_without_a_target_is_rejected(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="at least one column"):
        ds.drop()


def test_drop_rejects_conflicting_keyword_spellings(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="not both"):
        ds.drop(columns="x", labels="y")


def test_rename_accepts_the_pandas_columns_keyword(ds: bt.Dataset) -> None:
    assert ds.rename(columns={"x": "z"}).columns == ["z", "y", "g"]


def test_cast_accepts_python_types_and_arrow_types(ds: bt.Dataset) -> None:
    import pyarrow as pa

    assert str(ds.select("x").cast(float).dtypes[0]) == "double"
    assert str(ds.select("x").astype({"x": float}).dtypes[0]) == "double"
    assert str(ds.select("x").cast(pa.float64()).dtypes[0]) == "double"


def test_cast_rejects_an_uninterpretable_dtype(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="cannot interpret"):
        ds.select("x").cast(object)


def test_join_how_cross_is_the_cartesian_product(ds: bt.Dataset) -> None:
    left = ds.select("x")
    assert left.join(left, how="cross").count() == 16


def test_join_how_cross_rejects_keys(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="unconditional"):
        ds.join(ds, on="x", how="cross")


def test_join_lists_cross_among_the_supported_types(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="cross"):
        ds.join(ds, on="x", how="sideways")


def test_value_counts_normalize_reports_shares(ds: bt.Dataset) -> None:
    got = ds.value_counts("g", normalize=True).to_pydict()
    assert got["proportion"] == [0.5, 0.5]


def test_pivot_accepts_the_pandas_aggfunc_spelling(ds: bt.Dataset) -> None:
    by_alias = ds.pivot(index=["g"], on="g", values="x", aggfunc="max")
    by_primary = ds.pivot(index=["g"], on="g", values="x", aggregate="max")
    assert by_alias.equals(by_primary)


def test_pivot_rejects_both_aggregate_spellings(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="not both"):
        ds.pivot(index=["g"], on="g", values="x", aggregate="max", aggfunc="min")


# --- drop_nulls how= ------------------------------------------------------------------


def test_drop_nulls_how_all_keeps_partially_null_rows() -> None:
    n = bt.from_pydict({"x": [1, None], "y": [None, None]})
    assert n.drop_nulls(how="all").to_pydict() == {"x": [1], "y": [None]}


def test_drop_nulls_how_any_is_the_default() -> None:
    n = bt.from_pydict({"x": [1, None], "y": [None, None]})
    assert n.drop_nulls().count() == 0


def test_drop_nulls_rejects_an_unknown_how() -> None:
    with pytest.raises(PlanError, match="'any' or 'all'"):
        bt.from_pydict({"x": [1]}).drop_nulls(how="some")


# --- GroupBy head/tail ------------------------------------------------------------------


def test_group_by_head_and_tail_keep_rows_not_aggregates() -> None:
    g = bt.from_pydict({"g": ["a", "a", "b"], "v": [2, 1, 3]})
    assert g.group_by("g").head(1, order_by="v").sort("g").to_pydict() == {
        "g": ["a", "b"],
        "v": [1, 3],
    }
    assert g.group_by("g").tail(1, order_by="v").sort("g").to_pydict() == {
        "g": ["a", "b"],
        "v": [2, 3],
    }


def test_group_by_head_rejects_a_non_positive_n() -> None:
    g = bt.from_pydict({"g": ["a"], "v": [1]})
    with pytest.raises(PlanError, match="n must be >= 1"):
        g.group_by("g").head(0, order_by="v")


def test_group_by_head_explains_why_a_derived_key_will_not_work() -> None:
    g = bt.from_pydict({"g": ["a"], "v": [1]})
    with pytest.raises(PlanError, match="with_columns"):
        g.group_by(k=bt.col("g")).head(1, order_by="v")


# --- boolean-mask indexing ------------------------------------------------------------


def test_boolean_mask_indexing_filters(ds: bt.Dataset) -> None:
    # The universal pandas/Polars df[df.x > 1] idiom.
    assert ds[ds["x"] > 1].to_pydict()["x"] == [2, 3, 4]


def test_boolean_mask_indexing_equals_filter(ds: bt.Dataset) -> None:
    assert ds[ds["x"] > 1].equals(ds.filter(bt.col("x") > 1))


def test_getitem_still_projects_and_slices_and_names_columns(ds: bt.Dataset) -> None:
    assert ds[["x", "y"]].columns == ["x", "y"]
    assert ds[:2].count() == 2
    assert repr(ds["x"]) == "col('x')"


def test_getitem_rejects_an_unindexable_key(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="boolean expression"):
        ds[1.5]
