"""Shapes the GPU translator used to decline or get wrong, checked against the CPU engine.

Same contract and same oracle as `test_gpu_plan.py` — the translator replayed on pandas must
equal Batcher's own engine, which is itself checked against DuckDB. This module covers the
cases that motivated widening it:

* the boolean folds over an **all-null group**, where the libraries return the fold's identity
  and the engine returns null. That one was not a missing feature but a *wrong answer* on a
  path already advertised as supported;
* a **sort on a computed key** and a sort whose keys **disagree about null placement**, both
  of which sent the entire plan to the CPU engine rather than the one operator.

Every ordering case is compared row-for-row. An order-independent comparison is exactly what
cannot see an ordering bug, which is the whole risk in the sort changes below.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain

pytestmark = pytest.mark.unit

NAN = float("nan")


@pytest.fixture
def be():
    import pandas as pd

    return DfBackend(pd)


def _rows(table: pa.Table) -> list[tuple]:
    cols = table.to_pydict()
    return [tuple(row) for row in zip(*cols.values(), strict=True)]


def _by_key(table: pa.Table) -> dict:
    """A one-reducer grouped result as `{key: value}`, so groups compare regardless of order."""
    return dict(zip(table.column("k").to_pylist(), table.column("r").to_pylist(), strict=True))


def _run(build, table, be):
    """Translate and replay `build`, alongside what the CPU engine computes for it."""
    ds = build(bt.from_arrow(table))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "shape should be GPU-translatable"
    got = be.to_arrow(run_chain(table, spec[1], be))
    expected = ds.collect()
    return got.select(expected.column_names), expected


# --- the boolean folds over a group with nothing to fold ------------------------------

BOOLS = pa.table(
    {
        # `mixed` has both, `all_true` only true, `all_false` only false, and `empty` is
        # the group that matters: every value null, so there is nothing to fold.
        "k": ["mixed", "mixed", "all_true", "all_true", "all_false", "empty", "empty"],
        "b": [True, False, True, True, False, None, None],
    }
)


@pytest.mark.parametrize("reducer", ["all", "any"])
def test_a_boolean_fold_over_an_all_null_group_is_null(be, reducer):
    """The libraries skip the nulls and return the fold's identity; the engine returns null.

    Left alone, `.all()` over a group whose values were every one of them null reads as
    "every one of them was true" — a wrong answer, not a missing feature.
    """
    got, expected = _run(lambda ds: ds.group_by("k").agg(r=getattr(col("b"), reducer)()), BOOLS, be)
    assert _by_key(got) == _by_key(expected)
    # And specifically: the empty group is null, not the identity element.
    assert _by_key(got)["empty"] is None


def test_a_boolean_fold_still_folds_the_groups_that_have_values(be):
    got, _ = _run(lambda ds: ds.group_by("k").agg(r=col("b").all()), BOOLS, be)
    by_key = _by_key(got)
    assert by_key["all_true"] is True
    assert by_key["mixed"] is False


@pytest.mark.parametrize("reducer", ["all", "any"])
def test_a_keyless_boolean_fold_over_all_nulls_is_null(be, reducer):
    """The keyless form is the distributed *combine* step, so it cannot be left wrong."""
    table = pa.table({"b": [None, None]}, schema=pa.schema([pa.field("b", pa.bool_())]))
    got, expected = _run(lambda ds: ds.agg(r=getattr(col("b"), reducer)()), table, be)
    assert got.column("r").to_pylist() == expected.column("r").to_pylist() == [None]


# --- aggregates that used to drop the whole plan --------------------------------------

STATS = pa.table(
    {
        "k": ["a", "a", "a", "a", "b", "b", "c"],
        "v": [1.0, 2.0, 4.0, 8.0, 3.0, 5.0, 7.0],
        "n": [1, 2, 4, 8, 3, 5, None],
    }
)


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.group_by("k").agg(s=col("v").skew()),
        lambda ds: ds.group_by("k").agg(a=col("n").any_value()),
        lambda ds: ds.group_by("k").agg(s=col("v").skew(), m=col("v").mean()),
        lambda ds: ds.agg(s=col("v").skew()),
    ],
)
def test_widened_aggregates_match_the_engine(be, build):
    got, expected = _run(build, STATS, be)
    for name in expected.column_names:
        pairs = zip(got.column(name).to_pylist(), expected.column(name).to_pylist(), strict=True)
        for g, e in pairs:
            assert (g is None and e is None) or g == pytest.approx(e), name


# --- sorts that used to fall back -----------------------------------------------------

SORTABLE = pa.table(
    {
        "a": [3, 1, None, 2, None, 1],
        "b": ["x", "Y", "z", None, "w", "A"],
        "c": [1.5, -2.0, 0.0, 9.0, 4.0, -1.0],
    }
)


@pytest.mark.parametrize(
    "build",
    [
        # A computed key: previously "sort on a computed key" -> the whole plan to the CPU.
        lambda ds: ds.sort(col("a") * 2),
        lambda ds: ds.sort(col("b").str.lower()),
        lambda ds: ds.sort(col("c").abs(), descending=True),
        lambda ds: ds.sort("a", col("c") + col("a")),
    ],
)
def test_a_sort_on_a_computed_key_matches_the_engine_row_for_row(be, build):
    got, expected = _run(build, SORTABLE, be)
    assert _rows(got) == _rows(expected)


@pytest.mark.parametrize(
    "build",
    [
        # Keys disagreeing about null placement: previously unexpressible, so a fallback.
        lambda ds: ds.sort("a", "b", nulls_first=[True, False]),
        lambda ds: ds.sort("a", "b", nulls_first=[False, True]),
        lambda ds: ds.sort("a", "b", descending=[True, False], nulls_first=[True, False]),
        # ...and the agreeing cases must not have regressed.
        lambda ds: ds.sort("a", "b", nulls_first=[True, True]),
        lambda ds: ds.sort("a", "b", nulls_first=[False, False]),
        lambda ds: ds.sort("a", descending=True),
    ],
)
def test_per_key_null_placement_matches_the_engine_row_for_row(be, build):
    got, expected = _run(build, SORTABLE, be)
    assert _rows(got) == _rows(expected)


def test_a_sort_does_not_leak_its_private_columns(be):
    """The indicator and computed-key columns are scaffolding, not output."""
    got, expected = _run(lambda ds: ds.sort(col("a") * 2), SORTABLE, be)
    assert got.column_names == expected.column_names
    assert not any(name.startswith("__bt_") for name in got.column_names)


def test_a_limited_sort_still_takes_the_top_rows(be):
    got, expected = _run(lambda ds: ds.sort("a").limit(3), SORTABLE, be)
    assert _rows(got) == _rows(expected)


# --- keys that two implementations disagree about ------------------------------------

NULL_LEFT = pa.table({"k": [1, None, 2, None], "l": [1, 2, 3, 4]})
NULL_RIGHT = pa.table({"k": [1, None, 5], "r": [10, 20, 50]})


def _run_join(how, left, right, be):
    """Translate and replay a join, alongside what the CPU engine computes for it."""
    from batcher.core.gpu_plan import gpu_join_spec
    from batcher.core.gpu_plan.execute import run_join

    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how=how)
    spec = gpu_join_spec(ds._plan)
    assert spec is not None, "join should be GPU-translatable"
    (_ls, lops), (_rs, rops), join_ir, ops = spec
    got = be.to_arrow(run_join(left, right, lops, rops, join_ir, ops, be))
    expected = ds.collect()
    return got.select(expected.column_names), expected


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer", "semi", "anti"])
def test_a_null_join_key_matches_nothing(be, how):
    """Null equals nothing, including itself — and `merge`/`isin` both disagree.

    Every join type was wrong here, in both directions: an inner join invented a row, an
    outer join paired up two rows it was supposed to report as unmatched, a semi join gained
    a row it should have dropped, and an anti join dropped the rows that are usually the
    reason for running one.
    """
    got, expected = _run_join(how, NULL_LEFT, NULL_RIGHT, be)
    assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))


def test_a_complete_key_still_joins(be):
    """The null handling must not cost the matches that were already right."""
    left = pa.table({"k": [1, 2, 3], "l": [1, 2, 3]})
    right = pa.table({"k": [2, 3, 4], "r": [20, 30, 40]})
    got, expected = _run_join("inner", left, right, be)
    assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))
    assert got.num_rows == 2


ZEROS = pa.table(
    {
        # Both zeros, which IEEE, SQL and the engine all call one value.
        "k": [0.0, -0.0, 1.0, float("inf"), -0.0, 2.0],
        "v": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    }
)


def test_negative_zero_groups_with_zero(be):
    """The libraries group by a hash, and the two zeros hash apart — so one group became two.

    A distributed aggregate is where this bites hardest: a shard that saw only one of the two
    zeros produces a partial that nothing later folds together.
    """
    got, expected = _run(lambda ds: ds.group_by("k").agg(s=col("v").sum()), ZEROS, be)
    assert got.num_rows == expected.num_rows
    assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))


def test_negative_zero_folds_in_a_computed_key(be):
    got, expected = _run(lambda ds: ds.group_by(z=col("k") * 0.0).agg(s=col("v").sum()), ZEROS, be)
    assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))


def test_an_integer_key_is_not_paid_for(be):
    """Only float keys need the fold, so an integer group-by must be untouched by it."""
    table = pa.table({"k": [1, 1, 2, None], "v": [1.0, 2.0, 4.0, 8.0]})
    got, expected = _run(lambda ds: ds.group_by("k").agg(s=col("v").sum()), table, be)
    assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))


# --- a keyless aggregate over nothing --------------------------------------------------

EMPTY = pa.table({"x": pa.array([], type=pa.int64()), "y": pa.array([], type=pa.float64())})


@pytest.mark.parametrize(
    "reducer", ["sum", "mean", "min", "max", "median", "std", "var", "product"]
)
def test_a_measuring_aggregate_over_no_rows_is_one_null_row(be, reducer):
    """One row is what makes a keyless aggregate keyless — a grouped one returns none.

    Grouping an empty frame produces no groups, so the translator returned nothing at all
    where SQL and the engine return a single row of nulls.
    """
    got, expected = _run(lambda ds: ds.agg(r=getattr(col("y"), reducer)()), EMPTY, be)
    assert got.num_rows == expected.num_rows == 1
    assert got.column("r").to_pylist() == expected.column("r").to_pylist() == [None]


@pytest.mark.parametrize("reducer", ["count", "count_distinct"])
def test_a_counting_aggregate_over_no_rows_is_zero(be, reducer):
    """Counting nothing is zero, not null — the one place the empty row is not null."""
    got, expected = _run(lambda ds: ds.agg(r=getattr(col("y"), reducer)()), EMPTY, be)
    assert got.column("r").to_pylist() == expected.column("r").to_pylist() == [0]


def test_the_empty_row_keeps_each_columns_type(be):
    """A null `sum` must be a null *float*, or the shard cannot concatenate with its peers."""
    got, expected = _run(lambda ds: ds.agg(s=col("y").sum(), n=col("x").count()), EMPTY, be)
    assert got.schema.field("s").type == expected.schema.field("s").type
    assert got.num_rows == 1


def test_a_grouped_aggregate_over_no_rows_still_returns_no_rows(be):
    """The counter-case: no groups means no rows, and the empty-row rule must not reach it."""
    got, expected = _run(lambda ds: ds.group_by("x").agg(s=col("y").sum()), EMPTY, be)
    assert got.num_rows == expected.num_rows == 0


def test_distinct_folds_negative_zero_but_keeps_the_row_it_saw_first(be):
    """DISTINCT is a group-by over every column, so it inherits the two-zeros problem."""
    table = pa.table({"f": [-0.0, 0.0, 1.0, 0.0], "g": ["b", "b", "a", "b"]})
    got, expected = _run(lambda ds: ds.distinct(), table, be)
    assert got.num_rows == expected.num_rows == 2
    assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))


def test_distinct_over_columns_with_no_floats_is_unchanged(be):
    table = pa.table({"a": [1, 1, 2, None, None], "b": ["x", "x", "y", None, None]})
    got, expected = _run(lambda ds: ds.distinct(), table, be)
    assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))


ZERO_LEFT = pa.table({"k": [0.0, -0.0, float("nan"), 1.0], "l": [1, 2, 3, 4]})
ZERO_RIGHT = pa.table({"k": [-0.0, float("nan"), 1.0], "r": [10, 20, 30]})


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_a_semi_or_anti_join_folds_negative_zero(be, how):
    """`isin` compares by hash, so a left `0.0` did not find a right `-0.0`.

    The third door the two-zeros problem arrives through, after the group key and DISTINCT.
    """
    got, expected = _run_join(how, ZERO_LEFT, ZERO_RIGHT, be)
    assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_an_equi_join_agrees_on_negative_zero_too(be, how):
    got, expected = _run_join(how, ZERO_LEFT, ZERO_RIGHT, be)
    assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))


def test_union_distinct_folds_negative_zero(be):
    """A UNION deduplicates rows, so it decides identity and needs the same fold."""
    from batcher.core.gpu_plan import gpu_union_spec
    from batcher.core.gpu_plan.execute import run_union

    a = pa.table({"x": [1, 2], "y": [1.0, -0.0]})
    b = pa.table({"x": [2, 3], "y": [0.0, 3.0]})
    ds = bt.from_arrow(a).union(bt.from_arrow(b), distinct=True)
    inputs, distinct, ops = gpu_union_spec(ds._plan)
    got = be.to_arrow(run_union([a, b], [o for _, o in inputs], distinct, ops, be))
    expected = ds.collect()
    assert got.num_rows == expected.num_rows == 3
    assert sorted(map(repr, got.select(expected.column_names).to_pylist())) == sorted(
        map(repr, expected.to_pylist())
    )


# --- casts the two implementations format differently ---------------------------------


def test_a_float_to_string_cast_is_declined(be):
    """Three separate formatting disagreements, none of which is the more correct one.

    An integral value keeps its `.0` on one side and not the other, the sign of zero prints
    differently, and `NaN` becomes the string `"nan"` on one side and a null on the other. A
    column of numbers would become a column of subtly different text, so the stage falls back.
    """
    from batcher.core.gpu_plan import Unsupported
    from batcher.core.gpu_plan.execute import run_chain

    table = pa.table({"f": [0.0, -0.0, 4.0, 1.5, float("nan"), None]})
    ds = bt.from_arrow(table).select(s=col("f").cast("string"))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the shape matches; the expression is what declines"
    with pytest.raises(Unsupported):
        run_chain(table, spec[1], be)


def test_an_integer_to_string_cast_still_runs(be):
    """The counter-case: integers have one spelling, so that cast is not in question."""
    table = pa.table({"i": [0, -7, 42, None]})
    got, expected = _run(lambda ds: ds.select(s=col("i").cast("string")), table, be)
    assert got.column("s").to_pylist() == expected.column("s").to_pylist()


def test_a_float_to_integer_cast_still_rounds(be):
    table = pa.table({"f": [1.5, 2.5, -1.5, None]})
    got, expected = _run(lambda ds: ds.select(i=col("f").cast("int64")), table, be)
    assert got.column("i").to_pylist() == expected.column("i").to_pylist()


# --- a semi/anti join whose key has more than one column ------------------------------

STAR_FACT = pa.table(
    {
        "d": [1, 2, 3, None, 4, 5],
        "s": ["a", "b", "a", "c", None, "z"],
        "f": [0.0, 1.0, -0.0, 2.0, 3.0, 4.0],
        "l": [1, 2, 3, 4, 5, 6],
    }
)
STAR_DIM = pa.table(
    {
        # `(1, "a")` twice: the right side has duplicates, which a membership test must not
        # turn into duplicated left rows.
        "d": [1, 1, 3, None, 5],
        "s": ["a", "a", "a", "c", "z"],
        "f": [-0.0, -0.0, 0.0, 2.0, 9.0],
        "r": [10, 11, 30, 40, 50],
    }
)


def _run_join_on(how, on, left, right, be):
    from batcher.core.gpu_plan import gpu_join_spec
    from batcher.core.gpu_plan.execute import run_join

    ds = bt.from_arrow(left).join(bt.from_arrow(right), on=on, how=how)
    spec = gpu_join_spec(ds._plan)
    assert spec is not None, "join should be GPU-translatable"
    (_ls, lops), (_rs, rops), join_ir, ops = spec
    got = be.to_arrow(run_join(left, right, lops, rops, join_ir, ops, be))
    expected = ds.collect()
    return got.select(expected.column_names), expected


@pytest.mark.parametrize("how", ["semi", "anti"])
@pytest.mark.parametrize("on", [["d", "s"], ["d", "s", "f"]])
def test_a_composite_key_semi_or_anti_join_runs_on_the_device(be, how, on):
    """A star-schema anti-join on `(date, store)` used to send the whole plan to the CPU.

    Compared row-for-row: a merge does not promise to preserve the left frame's order, and on
    the host backend it happens to — which is the shape of bug that passes here and reorders
    on the device. The position is carried through the merge and sorted back for that reason.
    """
    got, expected = _run_join_on(how, on, STAR_FACT, STAR_DIM, be)
    assert _rows(got) == _rows(expected)


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_a_composite_key_does_not_fan_out_on_a_duplicated_right_side(be, how):
    """`(1, "a")` appears twice on the right; a semi join must still emit one left row."""
    got, _ = _run_join_on(how, ["d", "s"], STAR_FACT, STAR_DIM, be)
    assert got.num_rows <= STAR_FACT.num_rows


# --- the widths the engine presents, from a reader that never crossed its boundary -----

NARROW = pa.table(
    {
        "i": pa.array([1, 2, None], type=pa.int32()),
        "s": pa.array([1, 2, 3], type=pa.int16()),
        "t": pa.array([1, 2, 3], type=pa.int8()),
        "u": pa.array([1, 2, 3], type=pa.uint8()),
        "b": pa.array([9, 9, 9], type=pa.uint64()),
        "f": pa.array([1.5, 2.5, 3.5], type=pa.float32()),
    }
)


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.select("i", "s", "t", "u", "b", "f"),
        lambda ds: ds.select(r=col("i") + 1),
        lambda ds: ds.select(r=col("i").abs()),
        lambda ds: ds.agg(r=col("i").sum()),
        lambda ds: ds.agg(r=col("f").mean()),
        lambda ds: ds.group_by("i").agg(n=col("s").count()),
        lambda ds: ds.filter(col("s") > 1).select("s", "f"),
        lambda ds: ds.sort("f").select("f"),
    ],
)
def test_a_narrow_column_comes_back_at_the_width_the_engine_presents(be, build):
    """The FFI boundary widens every integer to `int64` and every float to `double`.

    The translator reads Arrow without crossing that boundary, so left alone it hands back the
    source's own width. Not a wrong number — a wrong *column*, and worst exactly where this
    backend is used: a fan-out concatenates its shards, and a shard that fell back to the CPU
    engine contributes `int64` beside a device shard's `int32`.
    """
    got, expected = _run(build, NARROW, be)
    assert [f.type for f in got.schema] == [f.type for f in expected.schema]
    assert got.to_pylist() == expected.to_pylist()


def test_a_table_that_is_already_wide_is_not_rebuilt(be):
    """The check is over the field list, not the data, so the common case costs nothing."""
    from batcher.core.gpu_plan.backend import widen_narrow

    wide = pa.table({"a": pa.array([1], type=pa.int64()), "b": pa.array([1.0])})
    assert widen_narrow(wide) is wide


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (pa.int8(), pa.int64()),
        (pa.int32(), pa.int64()),
        (pa.uint64(), pa.int64()),
        (pa.float32(), pa.float64()),
        (pa.int64(), None),
        (pa.float64(), None),
        (pa.string(), None),
        (pa.bool_(), None),
        (pa.date32(), None),
    ],
)
def test_the_widening_rule_matches_the_boundary(source, expected):
    from batcher.core.gpu_plan.backend import widened_type

    assert widened_type(source) == expected


def test_abs_is_the_one_function_that_keeps_an_integer_integer(be):
    """Every other unary math function widens to double on both sides; `abs` does not."""
    table = pa.table({"i": pa.array([1, -2, 0, None], type=pa.int64())})
    got, expected = _run(lambda ds: ds.select(r=col("i").abs()), table, be)
    assert got.schema.field("r").type == expected.schema.field("r").type == pa.int64()
    assert got.column("r").to_pylist() == expected.column("r").to_pylist()


@pytest.mark.parametrize("fn", ["ceil", "floor", "sqrt", "exp", "sign"])
def test_the_other_unary_functions_still_widen(be, fn):
    table = pa.table({"i": pa.array([1, -2, 4, None], type=pa.int64())})
    got, expected = _run(lambda ds, f=fn: ds.select(r=getattr(col("i"), f)()), table, be)
    assert got.schema.field("r").type == expected.schema.field("r").type


# --- a pattern that is a pattern, and one that is not ----------------------------------

PATTERNS = pa.table({"s": ["123abc", "axb", "a.b", "a+b", "AAA", "a|b", None]})


@pytest.mark.parametrize(
    "pattern", [r"\d", "a.b", "a+b", "a|b", "A{2}", "[ab]", "^a", "b$", "(a)", "a*"]
)
def test_contains_matches_a_literal_not_a_regular_expression(be, pattern):
    """The engine matches a literal substring; both libraries default to a regex.

    Every one of these patterns matched rows the engine does not, and the metacharacter that
    makes it bite is `.` — which is in every path, hostname, version string and email domain
    anyone filters on. `contains("a.b")` was matching "axb".
    """
    got, expected = _run(lambda ds: ds.select(r=col("s").str.contains(pattern)), PATTERNS, be)
    assert got.column("r").to_pylist() == expected.column("r").to_pylist()


def test_contains_still_finds_a_plain_substring(be):
    got, expected = _run(lambda ds: ds.select(r=col("s").str.contains("ab")), PATTERNS, be)
    assert got.column("r").to_pylist() == expected.column("r").to_pylist()
    assert got.column("r").to_pylist()[:2] == [True, False]


@pytest.mark.parametrize("fn", ["starts_with", "ends_with"])
def test_the_other_pattern_functions_were_already_literal(be, fn):
    got, expected = _run(
        lambda ds, f=fn: ds.select(r=getattr(col("s").str, f)("a.b")), PATTERNS, be
    )
    assert got.column("r").to_pylist() == expected.column("r").to_pylist()


# --- the one arithmetic where the two disagree about the unit --------------------------

DATES = pa.table(
    {
        "a": [dt.date(2024, 3, 1), dt.date(2024, 1, 1), None, dt.date(2024, 2, 29)],
        "b": [dt.date(2024, 1, 1), dt.date(2024, 3, 1), dt.date(2024, 1, 1), dt.date(2024, 3, 1)],
        "t": [
            dt.datetime(2024, 3, 1, 12),
            dt.datetime(2024, 1, 1),
            None,
            dt.datetime(2024, 2, 29, 6),
        ],
        "u": [
            dt.datetime(2024, 1, 1),
            dt.datetime(2024, 3, 1, 6),
            dt.datetime(2024, 1, 1),
            dt.datetime(2024, 3, 1),
        ],
    }
)


def test_subtracting_two_dates_gives_a_count_of_days(be):
    """The libraries return a duration; the engine returns an integer number of days.

    The values agree and the column does not, which is the worse half: a shard contributing
    `duration[s]` beside a CPU-fallback shard's `int64` cannot be concatenated at all.
    """
    got, expected = _run(lambda ds: ds.select(r=col("a") - col("b")), DATES, be)
    assert got.schema.field("r").type == expected.schema.field("r").type == pa.int64()
    assert got.column("r").to_pylist() == expected.column("r").to_pylist()


def test_subtracting_two_timestamps_is_still_a_duration(be):
    """The counter-case: both sides call this a duration, so it must be left alone."""
    got, expected = _run(lambda ds: ds.select(r=col("t") - col("u")), DATES, be)
    assert got.schema.field("r").type == expected.schema.field("r").type
    assert got.column("r").to_pylist() == expected.column("r").to_pylist()


# --- a NaN no longer costs the whole plan its device -----------------------------------

NANS = pa.table(
    {
        "g": ["mix", "mix", "mix", "allnan", "allnan", "clean", "clean", "withnull", "one"],
        "v": [1.0, NAN, 3.0, NAN, NAN, 2.0, 4.0, None, 5.0],
    }
)


@pytest.mark.parametrize(
    "reducer",
    ["sum", "mean", "count", "std", "var", "product", "count_distinct", "any_value", "skew"],
)
def test_a_nan_bearing_column_still_reaches_the_device(be, reducer):
    """The whole aggregate used to fall back for *any* reduction over a NaN-bearing column.

    A division by zero somewhere upstream cost the entire query its device, even when every
    reduction in it handles NaN exactly as the engine does — which nine of them do.
    """
    got, expected = _run(lambda ds: ds.group_by("g").agg(r=getattr(col("v"), reducer)()), NANS, be)
    by_key = dict(zip(got.column("g").to_pylist(), got.column("r").to_pylist(), strict=True))
    want = dict(
        zip(expected.column("g").to_pylist(), expected.column("r").to_pylist(), strict=True)
    )
    assert by_key.keys() == want.keys()
    for key, value in want.items():
        actual = by_key[key]
        if value is None or actual is None:
            assert actual is value, key
        elif value != value:  # NaN
            assert actual != actual, key
        else:
            assert actual == pytest.approx(value), key


@pytest.mark.parametrize("reducer", ["min", "max", "median"])
def test_an_order_statistic_over_a_nan_still_declines(be, reducer):
    """The engine orders NaN above every number, so it wins a maximum and loses a minimum.

    Both libraries treat it as missing for those, and over a group of only NaNs they report
    missing where the engine reports NaN. Declining is the honest answer; a plausible number
    is not.
    """
    from batcher.core.gpu_plan import Unsupported
    from batcher.core.gpu_plan.execute import run_chain

    ds = bt.from_arrow(NANS).group_by("g").agg(r=getattr(col("v"), reducer)())
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the shape matches; the NaN is what declines"
    with pytest.raises(Unsupported):
        run_chain(NANS, spec[1], be)


def test_an_order_statistic_without_a_nan_is_unaffected(be):
    """The decline is on the data, not on the reduction — clean columns keep the device."""
    clean = pa.table({"g": ["a", "a", "b"], "v": [1.0, 3.0, 2.0]})
    for reducer in ("min", "max", "median"):
        got, expected = _run(
            lambda ds, r=reducer: ds.group_by("g").agg(x=getattr(col("v"), r)()), clean, be
        )
        assert sorted(map(repr, got.to_pylist())) == sorted(map(repr, expected.to_pylist()))
