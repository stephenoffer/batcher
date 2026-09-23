"""Column selectors at their edges, checked against `polars.selectors` as the oracle.

`test_diff_selectors.py` covers the selectors in the projection positions. This file covers
where they used to break: aggregates in `group_by().agg()` and `select`, the `.name`
accessor after an operation, the set operators (``^``, and ``col`` / scalar operands),
narrow dtypes that ingest widens, the name-taking verbs (`drop_nulls`, `unpivot`,
`group_by`, `distinct`), and a selector that matches nothing. Where Batcher deliberately
differs from Polars (a widened dtype, a refused alias) the test says so and asserts the
Batcher behaviour directly.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pl = pytest.importorskip("polars", reason="polars is the selector oracle")
cs = pytest.importorskip("polars.selectors", reason="polars is the selector oracle")

pytestmark = pytest.mark.differential

TABLE = pa.table(
    {
        "g": ["a", "a", "b", "b", "c"],
        "i": [1, 2, 3, None, 5],
        "f": [10.0, None, 30.0, 40.0, 50.0],
        "s": ["x", "y", None, "w", "v"],
        "ok": [True, False, True, True, None],
    }
)


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_arrow(TABLE)


@pytest.fixture
def df():
    return pl.from_arrow(TABLE)


def _sorted(table: pa.Table, key: str) -> dict:
    """A table's rows as a dict, ordered by `key`: group output has no order of its own."""
    return table.sort_by(key).to_pydict()


def _same(got: bt.Dataset, want, key: str) -> None:
    assert got.columns == want.columns
    assert _sorted(got.to_arrow(), key) == _sorted(want.to_arrow(), key)


# --- aggregates over a selector --------------------------------------------------------


def test_agg_name_suffix_renames_every_expanded_aggregate(ds, df):
    got = ds.group_by("g").agg(bt.numeric().sum().name.suffix("_sum"))
    want = df.group_by("g").agg(cs.numeric().sum().name.suffix("_sum"))
    _same(got, want, "g")


def test_several_aggregates_over_one_selector(ds, df):
    got = ds.group_by("g").agg(
        bt.numeric().sum().name.suffix("_sum"),
        bt.numeric().max().name.prefix("max_"),
        bt.numeric().min().name.map(lambda c: f"lo_{c}"),
    )
    want = df.group_by("g").agg(
        cs.numeric().sum().name.suffix("_sum"),
        cs.numeric().max().name.prefix("max_"),
        cs.numeric().min().name.map(lambda c: f"lo_{c}"),
    )
    _same(got, want, "g")


def test_group_keys_are_never_aggregated_over(ds, df):
    got = ds.group_by("i").agg(bt.numeric().max())
    want = df.group_by("i").agg(cs.numeric().max())
    assert got.columns == want.columns == ["i", "f"]
    assert _sorted(got.to_arrow(), "i") == _sorted(want.to_arrow(), "i")


def test_an_expression_over_selector_aggregates(ds, df):
    got = ds.group_by("g").agg((bt.numeric().sum() * 2).name.suffix("_x2"))
    want = df.group_by("g").agg((cs.numeric().sum() * 2).name.suffix("_x2"))
    _same(got, want, "g")


def test_a_whole_frame_aggregate_over_a_selector(ds, df):
    got = ds.select(bt.numeric().max())
    want = df.select(cs.numeric().max())
    assert got.to_pydict() == want.to_dict(as_series=False)


def test_alias_over_several_columns_is_refused_at_definition(ds):
    """Polars keeps the last column under the alias; Batcher refuses the ambiguity."""
    with pytest.raises(PlanError, match=r"alias\('x'\) names a single column"):
        ds.group_by("g").agg(bt.numeric().sum().alias("x"))
    with pytest.raises(PlanError, match="names a single column"):
        ds.group_by("g").agg(x=bt.numeric().sum())
    one = ds.group_by("g").agg(bt.integer().sum().alias("x"))
    assert one.columns == ["g", "x"]
    assert _sorted(one.to_arrow(), "g") == {"g": ["a", "b", "c"], "x": [3, 3, 5]}


def test_selector_aggregates_agree_with_duckdb(ds, duck):
    duck.register("t", TABLE)
    want = duck.sql(
        "SELECT g, sum(i) AS i_sum, sum(f) AS f_sum FROM t GROUP BY g ORDER BY g"
    ).to_arrow_table()
    got = ds.group_by("g").agg(bt.numeric().sum().name.suffix("_sum")).sort("g").to_arrow()
    assert got.to_pydict() == want.to_pydict()


# --- windows over a selector ------------------------------------------------------------


def test_a_window_over_a_selector_expands_per_column(ds, df):
    got = ds.with_columns(bt.numeric().sum().over(partition_by=["g"]).name.suffix("_w"))
    want = df.with_columns(cs.numeric().sum().over("g").name.suffix("_w"))
    assert got.columns == want.columns
    assert got.to_pydict() == want.to_dict(as_series=False)
    # It used to report a single column named 'literal' and fail only at execution.
    assert ds.select(bt.numeric().sum().over(partition_by=["g"])).columns == ["i", "f"]


# --- the .name accessor after an operation ----------------------------------------------


def test_name_after_an_operation(ds, df):
    got = ds.with_columns((bt.numeric() * 2).name.suffix("_x2"))
    want = df.with_columns((cs.numeric() * 2).name.suffix("_x2"))
    assert got.columns == want.columns
    assert got.to_pydict() == want.to_dict(as_series=False)


# --- set operators ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (lambda: bt.numeric() ^ bt.floating(), lambda: cs.numeric() ^ cs.float()),
        (lambda: bt.numeric() & bt.col("f"), lambda: cs.numeric() & cs.by_name("f")),
        (lambda: bt.numeric() | bt.col("s"), lambda: cs.numeric() | cs.by_name("s")),
        (lambda: bt.all() - bt.col("g"), lambda: cs.all() - cs.by_name("g")),
        (lambda: ~bt.numeric(), lambda: ~cs.numeric()),
        (lambda: (~bt.numeric()).name.prefix("p_"), lambda: (~cs.numeric()).name.prefix("p_")),
    ],
)
def test_set_operators_match_polars(ds, df, ours, theirs):
    assert ds.select(ours()).columns == df.select(theirs()).columns


def test_a_scalar_operand_is_arithmetic_not_a_set_operation(ds, df):
    got = ds.select(bt.numeric() - 1)
    want = df.select(cs.numeric() - 1)
    assert got.to_pydict() == want.to_dict(as_series=False)


def test_a_set_operation_on_a_renamed_selector_is_refused(ds):
    """The rename used to be dropped by ``~`` and picked arbitrarily by ``|``."""
    with pytest.raises(PlanError, match="cannot complement the renamed selector"):
        ~bt.numeric().name.prefix("p_")
    with pytest.raises(PlanError, match="cannot combine the renamed selector"):
        bt.numeric().name.prefix("p_") | bt.string()
    # `exclude` narrows the selection and keeps the rename.
    assert ds.select(bt.numeric().name.prefix("p_").exclude("f")).columns == ["p_i"]


# --- dtypes that ingest widens -----------------------------------------------------------


def test_a_narrow_dtype_matches_the_column_it_was_widened_to():
    narrow = pa.table(
        {"i32": pa.array([1], pa.int32()), "f32": pa.array([1.5], pa.float32()), "s": ["x"]}
    )
    ds, df = bt.from_arrow(narrow), pl.from_arrow(narrow)
    assert ds.collect().schema.field("i32").type == pa.int64()  # the widening itself
    for ours, theirs in [(pa.int32(), pl.Int32), (pa.float32(), pl.Float32)]:
        assert ds.select(bt.by_dtype(ours)).columns == df.select(cs.by_dtype(theirs)).columns
        assert ds.select(bt.col(ours)).columns == df.select(pl.col(theirs)).columns


def test_a_widened_dtype_also_matches_a_source_wide_column():
    """Where Batcher differs: the engine cannot tell a source int32 from an int64."""
    ds = bt.from_arrow(pa.table({"a": pa.array([1], pa.int32()), "b": pa.array([2], pa.int64())}))
    assert ds.select(bt.by_dtype(pa.int32())).columns == ["a", "b"]
    assert ds.select(bt.by_dtype("int32")).columns == ["a", "b"]  # a type name works too
    with pytest.raises(PlanError, match="takes pyarrow types"):
        bt.by_dtype("not_a_type")


# --- the name-taking verbs --------------------------------------------------------------


def test_drop_nulls_subset_takes_a_selector(ds, df):
    for how in ("any", "all"):
        got = ds.drop_nulls(subset=bt.numeric(), how=how)
        explicit = ds.drop_nulls(subset=["i", "f"], how=how)
        assert _sorted(got.to_arrow(), "g") == _sorted(explicit.to_arrow(), "g")
    assert ds.drop_nulls(subset=bt.numeric()).count() == df.drop_nulls(subset=cs.numeric()).height


def _rows(columns: dict) -> list[tuple]:
    """A column dict as a row multiset, sorted with nulls placed by `repr`."""
    return sorted(zip(*columns.values(), strict=True), key=repr)


def test_unpivot_on_takes_a_selector(ds, df):
    got = ds.unpivot(on=bt.numeric(), index="g")
    want = df.unpivot(on=cs.numeric(), index="g")
    assert got.columns == want.columns
    assert _rows(got.to_pydict()) == _rows(want.to_dict(as_series=False))


def test_group_by_takes_a_selector(ds, df):
    got = ds.group_by(bt.string()).agg(bt.col("i").sum())
    want = df.group_by(cs.string()).agg(pl.col("i").sum())
    assert got.columns == want.columns
    assert got.count() == want.height


def test_distinct_subset_takes_a_selector(ds, df):
    assert ds.distinct(subset=bt.boolean()).count() == df.unique(subset=cs.boolean()).height


def test_a_join_key_selector_is_refused_by_name(ds):
    with pytest.raises(PlanError, match="join keys must be named"):
        ds.join(ds, on=bt.string())


def test_a_name_argument_refuses_a_computed_selector(ds):
    with pytest.raises(PlanError, match="takes column names or a bare column selector"):
        ds.drop_nulls(subset=bt.numeric() + 1)


# --- a selector that matches nothing contributes nothing ---------------------------------


def test_an_empty_selector_contributes_no_columns(ds, df):
    nothing = bt.temporal()
    # Alongside other columns it adds nothing, as in Polars.
    assert ds.select(nothing, "g").columns == df.select(cs.temporal(), "g").columns
    assert ds.drop(nothing).columns == df.drop(cs.temporal()).columns
    assert ds.drop_nulls(subset=nothing).count() == df.drop_nulls(subset=cs.temporal()).height
    # On its own it leaves nothing to project, which Batcher refuses and names.
    with pytest.raises(PlanError, match="matched no columns"):
        ds.select(nothing)
    with pytest.raises(PlanError, match="matched no columns"):
        ds.with_columns(nothing.cast(pa.string()))
