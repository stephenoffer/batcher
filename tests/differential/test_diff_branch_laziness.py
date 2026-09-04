"""CASE and COALESCE evaluate only the branch a row selects — differential vs DuckDB.

Both forms used to be evaluated at full width: every ``THEN`` ran over every row and
``zip`` then discarded all but one result per row. That is invisible while the branches
are total, and wrong in two ways when they are not.

It **raises errors SQL says cannot happen**: ``CASE WHEN false THEN s::BIGINT ELSE 1
END`` over a non-numeric ``s`` failed the whole query here while DuckDB returned ``1``.
That is the divergence these tests pin, and it is the reason the change is a correctness
fix rather than only a speedup — the speed (a four-branch regex ``CASE`` cost four regex
passes per row) is the same property seen from the other side.

The rest of the file is the equivalence half: the selective path is only worth having if
it computes exactly what the full-width one did, so the branch shapes that decide *which*
path runs — a branch no row takes, a branch every row takes, one row, no rows, nulls in
the condition, mixed branch types — are each held against DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import coalesce, col, lit, when


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "i": pa.array([1, 2, 3, 4, 5, None], type=pa.int64()),
            "f": pa.array([1.5, None, 3.5, 4.5, None, 6.5], type=pa.float64()),
            "s": pa.array(["a", "b", "c", "10", None, "e"], type=pa.string()),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_an_unselected_branch_does_not_raise(duck, t):
    """The defect this change fixes: a ``THEN`` no row reaches used to fail the query."""
    out = (
        bt.from_arrow(t)
        .select(o=when(col("i") < 0).then(col("s").cast("int64")).otherwise(lit(1)))
        .collect()
    )
    assert_same(out, duck.sql("SELECT CASE WHEN i < 0 THEN s::BIGINT ELSE 1 END o FROM t"))


def test_a_selected_branch_still_raises(duck, t):
    """The other half: laziness must not swallow an error a row actually earns."""
    with pytest.raises(Exception):  # noqa: B017 - the engine's cast error, whatever it wraps
        (
            bt.from_arrow(t)
            .select(o=when(col("i") > 0).then(col("s").cast("int64")).otherwise(lit(1)))
            .collect()
        )


def test_a_condition_after_a_matching_branch_does_not_raise(duck, t):
    """A `CASE` ladder stops at the first match, so a later *condition* is not asked about
    a row an earlier arm already took. The laziness covers the `WHEN`s, not only the
    `THEN`s."""
    always = col("i").is_null() | col("i").is_not_null()
    out = (
        bt.from_arrow(t)
        .select(
            o=when(always)
            .then(lit(1))
            .when(col("s").cast("int64") > 0)
            .then(lit(2))
            .otherwise(lit(3))
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT CASE WHEN i IS NULL OR i IS NOT NULL THEN 1 "
            "WHEN s::BIGINT > 0 THEN 2 ELSE 3 END o FROM t"
        ),
    )


def test_a_coalesce_argument_that_is_never_needed_does_not_raise(duck, t):
    """``COALESCE`` stops at the first non-null, so a later argument never runs."""
    out = bt.from_arrow(t).select(o=coalesce(lit(7), col("s").cast("int64"))).collect()
    assert_same(out, duck.sql("SELECT coalesce(7, s::BIGINT) o FROM t"))


@pytest.mark.parametrize(
    ("name", "sql", "build"),
    [
        (
            "no_row_takes_the_branch",
            "CASE WHEN i > 100 THEN f * 2 ELSE f END",
            lambda: when(col("i") > 100).then(col("f") * 2).otherwise(col("f")),
        ),
        (
            "every_row_takes_the_branch",
            "CASE WHEN i IS NOT NULL OR i IS NULL THEN f * 2 ELSE f END",
            lambda: (
                when(col("i").is_not_null() | col("i").is_null())
                .then(col("f") * 2)
                .otherwise(col("f"))
            ),
        ),
        (
            "one_row_takes_the_branch",
            "CASE WHEN i = 3 THEN f * 2 ELSE f END",
            lambda: when(col("i") == 3).then(col("f") * 2).otherwise(col("f")),
        ),
        (
            "a_null_condition_falls_through",
            "CASE WHEN i > 2 THEN 'big' ELSE 'small' END",
            lambda: when(col("i") > 2).then(lit("big")).otherwise(lit("small")),
        ),
        (
            "four_disjoint_branches",
            "CASE WHEN i = 1 THEN 'a' WHEN i = 2 THEN 'b' WHEN i = 3 THEN 'c' ELSE 'z' END",
            lambda: (
                when(col("i") == 1)
                .then(lit("a"))
                .when(col("i") == 2)
                .then(lit("b"))
                .when(col("i") == 3)
                .then(lit("c"))
                .otherwise(lit("z"))
            ),
        ),
        (
            "overlapping_branches_first_wins",
            "CASE WHEN i >= 2 THEN 'ge2' WHEN i >= 1 THEN 'ge1' ELSE 'lt1' END",
            lambda: (
                when(col("i") >= 2)
                .then(lit("ge2"))
                .when(col("i") >= 1)
                .then(lit("ge1"))
                .otherwise(lit("lt1"))
            ),
        ),
        (
            "mixed_int_and_float_branches",
            "CASE WHEN i > 3 THEN 0 ELSE f END",
            lambda: when(col("i") > 3).then(lit(0)).otherwise(col("f")),
        ),
        (
            "a_string_branch_over_a_string_column",
            "CASE WHEN i > 3 THEN upper(s) ELSE lower(s) END",
            lambda: (
                when(col("i") > 3)
                .then(col("s").str.to_uppercase())
                .otherwise(col("s").str.to_lowercase())
            ),
        ),
        (
            # A gathered branch is scattered back with `take`, so a nested output type is
            # the case where that machinery could differ from the old whole-column `zip`.
            "a_list_returning_branch",
            "CASE WHEN i > 3 THEN string_split(s, 'x') ELSE string_split(s, '0') END",
            lambda: (
                when(col("i") > 3).then(col("s").str.split("x")).otherwise(col("s").str.split("0"))
            ),
        ),
        (
            "nested_case",
            "CASE WHEN i > 2 THEN (CASE WHEN i > 4 THEN 'hi' ELSE 'mid' END) ELSE 'lo' END",
            lambda: (
                when(col("i") > 2)
                .then(when(col("i") > 4).then(lit("hi")).otherwise(lit("mid")))
                .otherwise(lit("lo"))
            ),
        ),
        (
            "coalesce_over_three_columns",
            "coalesce(f, i, 0)",
            lambda: coalesce(col("f"), col("i"), lit(0)),
        ),
        (
            "coalesce_whose_first_argument_is_never_null",
            "coalesce(i + 0, f, 0)",
            lambda: coalesce(col("i") + 0, col("f"), lit(0)),
        ),
    ],
)
def test_branch_shapes_match_duckdb(duck, t, name, sql, build):
    out = bt.from_arrow(t).select(o=build()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} o FROM t"))


@pytest.mark.parametrize(
    "shape",
    ["dictionary", "all_null_condition", "all_null_values", "single_row", "one_distinct"],
)
def test_branch_selection_survives_the_column_shape(duck, shape):
    """The gather/scatter path reads and rebuilds whatever the arm produced, so the
    *encoding* of the column it reads is a second axis from the branch structure. A
    dictionary column decodes at the leaf, an all-null condition selects nothing on every
    arm, and a one-row relation is the smallest batch the heuristics ever see."""
    columns = {
        "dictionary": {
            "i": pa.array([1, 2, 3, 4], type=pa.int64()),
            "s": pa.array(["a", "b", "a", "b"]).dictionary_encode(),
        },
        "all_null_condition": {
            "i": pa.array([None, None, None], type=pa.int64()),
            "s": pa.array(["a", "b", "c"]),
        },
        "all_null_values": {
            "i": pa.array([1, 2, 3], type=pa.int64()),
            "s": pa.array([None, None, None], type=pa.string()),
        },
        "single_row": {"i": pa.array([1], type=pa.int64()), "s": pa.array(["a"])},
        "one_distinct": {
            "i": pa.array([7, 7, 7, 7], type=pa.int64()),
            "s": pa.array(["a", "a", "a", "a"]),
        },
    }[shape]
    tbl = pa.table(columns)
    duck.register("u", tbl)
    out = (
        bt.from_arrow(tbl)
        .select(
            c=when(col("i") > 2)
            .then(col("s").str.to_uppercase())
            .when(col("i") > 1)
            .then(col("s").str.reverse())
            .otherwise(col("s")),
            k=coalesce(col("i"), lit(0)),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT CASE WHEN i > 2 THEN upper(s) WHEN i > 1 THEN reverse(s) ELSE s END c, "
            "coalesce(i, 0) k FROM u"
        ),
    )


def test_an_empty_relation_keeps_the_branch_type(duck, t):
    """No rows means no branch is selected, which is the path that has to infer a type
    without evaluating anything. Getting it wrong shows up as a column type, not a value,
    so the schema is asserted rather than the (empty) contents."""
    out = (
        bt.from_arrow(t)
        .filter(col("i") < -1)
        .select(o=when(col("i") > 2).then(col("f") * 2).otherwise(lit(0.0)))
        .collect()
    )
    assert out.num_rows == 0
    assert pa.types.is_floating(out.schema.field("o").type)


def test_streaming_agrees_with_collect(t):
    """A morsel-at-a-time run splits the batch, so each morsel sees a different mix of
    selected rows — the one thing a single whole-relation `collect` cannot vary."""
    ds = bt.from_arrow(t).select(
        o=when(col("i") == 3)
        .then(col("s").str.to_uppercase())
        .when(col("i") == 5)
        .then(col("s").str.reverse())
        .otherwise(col("s"))
    )
    whole = ds.collect().to_pydict()["o"]
    streamed = [v for batch in ds.iter_batches() for v in batch.to_pydict()["o"]]
    assert streamed == whole
