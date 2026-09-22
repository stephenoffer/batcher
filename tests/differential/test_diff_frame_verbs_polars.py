"""The W8 frame verbs and `Expr.meta` against Polars 1.40, the engine they are named after.

The DuckDB side of the same claims is `test_diff_join_where_update_having.py`,
`test_diff_zip_split_transpose.py` and `test_diff_partition_schema_nans.py`. This file holds the
port: each case runs the Polars call a migrating user wrote and the Batcher call it maps to, and
compares the results. Where Batcher takes a parameter Polars leaves implicit, such as the
`order_by` a positional transpose needs, the Batcher side passes it and says why.

Frames are compared as row multisets where Polars' own order is arrival order, which Batcher
does not promise, and in order where the verb defines one.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

pl = pytest.importorskip("polars")


def _frames(pdf) -> tuple[pa.Table, bt.Dataset]:
    table = pdf.to_arrow()
    return table, bt.from_arrow(table)


def _same(batcher_table: pa.Table, polars_frame, *, ordered: bool = False) -> None:
    """Compare a Batcher result to a Polars frame, cell for cell."""
    expected = polars_frame.to_arrow()
    assert_tables_equal(batcher_table, expected, ordered=ordered)


EVENTS = pl.DataFrame({"t": [1, 5, 9, None, 5], "v": [1.0, 2.0, None, 4.0, 2.0]})
SPANS = pl.DataFrame({"lo": [0, 4, None], "hi": [6, 10, 3], "v": [7.0, 1.5, 0.0]})


def test_join_where_matches_polars():
    """Polars names the colliding right column `v_right`, and so does Batcher."""
    _, left = _frames(EVENTS)
    _, right = _frames(SPANS)
    ours = left.join_where(
        right,
        bt.col("t") >= bt.col("lo"),
        bt.col("t") < bt.col("hi"),
        bt.col("v") < bt.col("v_right"),
    )
    theirs = EVENTS.join_where(
        SPANS,
        pl.col("t") >= pl.col("lo"),
        pl.col("t") < pl.col("hi"),
        pl.col("v") < pl.col("v_right"),
    )
    _same(ours.collect(), theirs)


TARGET = pl.DataFrame(
    {"id": [1, 2, 2, 3, None], "x": [10, 20, 21, 30, 40], "y": ["a", "b", "c", "d", "e"]}
)
SOURCE = pl.DataFrame(
    {
        "id": [2, 3, 4, None],
        "x": [None, 300, 400, 99],
        "y": ["B", None, "D", "N"],
        "w": [1, 2, 3, 4],
    }
)


@pytest.mark.parametrize("include_nulls", [False, True])
@pytest.mark.parametrize("how", ["left", "inner", "full"])
def test_update_matches_polars(how, include_nulls):
    _, target = _frames(TARGET)
    _, source = _frames(SOURCE)
    ours = target.update(source, on="id", how=how, include_nulls=include_nulls)
    theirs = TARGET.update(SOURCE, on="id", how=how, include_nulls=include_nulls)
    _same(ours.collect(), theirs)


WIDE = pl.DataFrame({"name": ["q", "p"], "a": [1, 2], "b": [3, 4], "c": [0.5, None]})


def test_transpose_by_a_naming_column_matches_polars():
    """Polars keeps row order for the output columns; Batcher ascends by name, so the rows are
    given in name order to make the two agree."""
    ordered = WIDE.sort("name")
    _, ds = _frames(ordered)
    ours = ds.transpose(column_names="name", include_header=True)
    theirs = ordered.transpose(include_header=True, column_names="name")
    _same(ours.collect(), theirs, ordered=True)


def test_transpose_by_position_matches_polars():
    """Polars names positional columns by arrival; Batcher needs that order spelled out."""
    table, _ = _frames(WIDE.drop("name"))
    ds = bt.from_arrow(table).with_row_index("row")
    ours = ds.transpose(order_by="row").collect()
    # The row index is itself a transposed column (the first, as `with_row_index` puts it
    # first), so it is left out of the comparison.
    theirs = WIDE.drop("name").transpose()
    assert_tables_equal(ours.slice(1), theirs.to_arrow(), ordered=True)


def test_match_to_schema_matches_polars():
    frame = pl.DataFrame({"b": ["x", None], "a": [1, 2]})
    _, ds = _frames(frame)
    schema = {"a": pl.Int64, "b": pl.String, "c": pl.Float64}
    theirs = frame.lazy().match_to_schema(schema, missing_columns="insert").collect()
    ours = ds.match_to_schema(
        {"a": "int64", "b": "string", "c": "float64"}, missing_columns="insert"
    )
    _same(ours.collect(), theirs)


def test_drop_nans_matches_polars():
    frame = pl.DataFrame(
        {
            "a": [1.0, float("nan"), None, -0.0],
            "b": [float("nan"), 2.0, 3.0, 4.0],
            "s": list("wxyz"),
        }
    )
    _, ds = _frames(frame)
    _same(ds.drop_nans().collect(), frame.drop_nans())
    _same(ds.drop_nans("a").collect(), frame.drop_nans("a"))


def test_partition_by_matches_polars():
    frame = pl.DataFrame({"k": ["a", "b", "a", None], "j": [1, 1, 1, 2], "v": [1, 2, 3, 4]})
    _, ds = _frames(frame)
    ours = ds.partition_by("k", "j", include_key=False)
    theirs = frame.partition_by("k", "j", include_key=False, as_dict=True)
    assert set(ours) == set(theirs)
    for key, part in ours.items():
        _same(part.collect(), theirs[key])


def test_having_matches_polars():
    """`bt.count()` counts rows here and `pl.len()` there: the same group predicate."""
    frame = pl.DataFrame({"g": ["a", "a", "b", None, None], "v": [1, 2, 3, None, 5]})
    _, ds = _frames(frame)
    ours = ds.group_by("g").having(bt.count() > 1, bt.col("v").max() > 1).agg(s=bt.col("v").sum())
    theirs = (
        frame.group_by("g").having(pl.len() > 1, pl.col("v").max() > 1).agg(s=pl.col("v").sum())
    )
    _same(ours.collect(), theirs)


META_CASES = [
    (lambda m: m.col("a"), "plain column"),
    (lambda m: (m.col("a") + m.col("b") * m.col("a")).alias("z"), "aliased arithmetic"),
    (lambda m: m.col("x").sum(), "aggregate"),
    (lambda m: m.lit(1) + m.col("q"), "literal first"),
    (lambda m: m.col("a").alias("b"), "alias of a column"),
]


@pytest.mark.parametrize("build", [c[0] for c in META_CASES], ids=[c[1] for c in META_CASES])
def test_meta_introspection_matches_polars(build):
    ours, theirs = build(bt), build(pl)
    assert ours.meta.root_names() == theirs.meta.root_names()
    assert ours.meta.is_column() == theirs.meta.is_column()
    assert ours.meta.has_multiple_outputs() == theirs.meta.has_multiple_outputs()


@pytest.mark.parametrize("build", [c[0] for c in META_CASES], ids=[c[1] for c in META_CASES])
def test_meta_output_name_matches_polars(build):
    ours, theirs = build(bt), build(pl)
    assert ours.meta.output_name() == theirs.meta.output_name()


def test_meta_multiple_outputs_matches_polars():
    assert (
        bt.col("a", "b").meta.has_multiple_outputs() is pl.col("a", "b").meta.has_multiple_outputs()
    )
    assert bt.col("a", "b").meta.output_name(raise_if_undetermined=False) is None
    assert pl.col("a", "b").meta.output_name(raise_if_undetermined=False) is None
