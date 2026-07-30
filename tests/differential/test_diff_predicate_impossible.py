"""`filter_arithmetic_contradiction` must not drop a row DuckDB keeps — DuckDB is the oracle.

This rule is the most dangerous shape of optimization there is: it decides, without looking at
data, that a filter matches nothing and replaces it with an empty relation. A wrong refutation
does not raise, it silently returns fewer rows. So every refutation is checked against DuckDB
over a table chosen to *contain* the boundary values — the reachable maximum and minimum
remainder, both parities, the mask bits set and clear, negatives, and nulls — and every
satisfiable neighbour is checked too, so a rule that over-fires is caught by the rows it lost.

Two of the invariants rest on the engine's own arithmetic rather than on mathematics, and both
were measured rather than assumed: `%` is a *truncated* remainder (`-7 % 3` is `-1`, not `2`),
and `*` wraps mod 2^64. Where DuckDB agrees, that agreement is what these tests record.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.predicate_impossible  # registers the rule under test
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    # Spans every boundary the refutations turn on: remainders 0 and +/-9 (the reachable
    # extremes for `% 10`), both parities, values with and without bits 2 and 3 set, negatives,
    # a zero, and a null.
    tbl = pa.table(
        {
            "i": pa.array([0, 1, 2, 3, 4, 9, 10, 12, 13, 15, -1, -9, -10, -15, None], pa.int64()),
            "j": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, None], pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table({"i": pa.array([], pa.int64()), "j": pa.array([], pa.int64())})
    duck.register("empty", tbl)
    return tbl


#: `(batcher predicate, SQL predicate)` — the refuted shapes. Each must return zero rows, and
#: DuckDB confirms zero is the right answer rather than the rule's opinion.
_REFUTED = [
    (lambda: col("i") % 10 == 15, "i % 10 = 15"),
    (lambda: col("i") % 10 == -15, "i % 10 = -15"),
    (lambda: col("i") % 10 == 10, "i % 10 = 10"),
    (lambda: col("i") % 10 >= 10, "i % 10 >= 10"),
    (lambda: col("i") % 10 > 9, "i % 10 > 9"),
    (lambda: col("i") % 10 < -9, "i % 10 < -9"),
    (lambda: col("i") % 10 <= -10, "i % 10 <= -10"),
    (lambda: col("i") % -10 == 12, "i % -10 = 12"),
    (lambda: col("i") * 2 == 7, "i * 2 = 7"),
    (lambda: col("i") * 4 == 6, "i * 4 = 6"),
    (lambda: col("i") * -2 == 7, "i * -2 = 7"),
    (lambda: col("i").bitwise_and(12) == 3, "i & 12 = 3"),
    (lambda: col("i").bitwise_and(12) == 13, "i & 12 = 13"),
    (lambda: col("i").bitwise_or(12) == 3, "i | 12 = 3"),
    (lambda: col("i").bitwise_or(12) == 8, "i | 12 = 8"),
]

#: The satisfiable neighbours — one value off each boundary. These prove the rule does not
#: over-fire: if it emptied any of them, the lost rows would show up here immediately.
_SATISFIABLE = [
    (lambda: col("i") % 10 == 9, "i % 10 = 9"),
    (lambda: col("i") % 10 == -9, "i % 10 = -9"),
    (lambda: col("i") % 10 == 0, "i % 10 = 0"),
    (lambda: col("i") % 10 > 8, "i % 10 > 8"),
    (lambda: col("i") % 10 < -8, "i % 10 < -8"),
    (lambda: col("i") % 10 != 15, "i % 10 <> 15"),
    (lambda: col("i") * 2 == 8, "i * 2 = 8"),
    (lambda: col("i") * 3 == 9, "i * 3 = 9"),
    (lambda: col("i") * 2 != 7, "i * 2 <> 7"),
    (lambda: col("i").bitwise_and(12) == 4, "i & 12 = 4"),
    (lambda: col("i").bitwise_and(12) == 12, "i & 12 = 12"),
    (lambda: col("i").bitwise_and(12) == 0, "i & 12 = 0"),
    (lambda: col("i").bitwise_or(12) == 13, "i | 12 = 13"),
    (lambda: col("i").bitwise_or(12) == 12, "i | 12 = 12"),
    (lambda: col("i").bitwise_and(12) != 3, "i & 12 <> 3"),
]


@pytest.mark.parametrize(("pred", "sql"), _REFUTED)
def test_refuted_predicate_matches_duckdb(duck, t, pred, sql):
    out = bt.from_arrow(t).filter(pred()).collect()
    assert out.num_rows == 0, f"{sql} was refuted but Batcher returned rows"
    assert_same(out, duck.sql(f"SELECT * FROM t WHERE {sql}"))


@pytest.mark.parametrize(("pred", "sql"), _SATISFIABLE)
def test_satisfiable_neighbour_matches_duckdb(duck, t, pred, sql):
    out = bt.from_arrow(t).filter(pred()).collect()
    assert_same(out, duck.sql(f"SELECT * FROM t WHERE {sql}"))


def test_refuted_conjunct_empties_a_larger_conjunction(duck, t):
    out = bt.from_arrow(t).filter((col("i") > 0) & (col("i") % 10 == 15)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE i > 0 AND i % 10 = 15"))


def test_refuted_disjunct_keeps_the_other_side(duck, t):
    # The rule must not fire here: inside an OR the refuted term is not controlling.
    out = bt.from_arrow(t).filter((col("i") > 5) | (col("i") % 10 == 15)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE i > 5 OR i % 10 = 15"))


def test_refutation_over_empty_input(duck, empty):
    out = bt.from_arrow(empty).filter(col("i") % 10 == 15).collect()
    assert_same(out, duck.sql("SELECT * FROM empty WHERE i % 10 = 15"))


def test_refutation_above_an_aggregate(duck, t):
    # The empty relation has to fold through a grouped aggregate without changing its schema.
    out = (
        bt.from_arrow(t).filter(col("i") % 10 == 15).group_by("j").agg(n=col("i").count()).collect()
    )
    assert_same(out, duck.sql("SELECT j, count(i) AS n FROM t WHERE i % 10 = 15 GROUP BY j"))


def test_refutation_under_a_join(duck, t):
    # One side proving empty must not lose the other side's semantics: an inner join with an
    # empty side is empty.
    left = bt.from_arrow(t)
    right = bt.from_arrow(t).filter(col("i") % 10 == 15).select(j=col("j"))
    out = left.join(right, on="j").collect()
    assert_same(
        out,
        duck.sql("SELECT l.* FROM t l JOIN (SELECT j FROM t WHERE i % 10 = 15) r USING (j)"),
    )


def test_global_count_over_a_refuted_filter(duck, t):
    # A keyless aggregate over empty input emits one row, not zero — the empty-relation rules
    # are explicit about that, and getting it wrong would answer NULL instead of 0.
    out = bt.from_arrow(t).filter(col("i") % 10 == 15).agg(n=col("i").count()).collect()
    assert_same(out, duck.sql("SELECT count(i) AS n FROM t WHERE i % 10 = 15"))


# --- the image of a function -------------------------------------------------
#
# `filter_function_range_contradiction` claims a function cannot return a value, so DuckDB
# evaluating the same function is the only honest check: the calendar ranges and the two
# weekday conventions have to match the engine, and a mismatch at either end would silently
# drop the rows at that boundary.

_IMAGE_REFUTED = [
    (lambda: col("ts").dt.month() == 13, "extract(month FROM ts) = 13"),
    (lambda: col("ts").dt.month() == 0, "extract(month FROM ts) = 0"),
    (lambda: col("ts").dt.quarter() == 5, "extract(quarter FROM ts) = 5"),
    (lambda: col("ts").dt.day() == 32, "extract(day FROM ts) = 32"),
    (lambda: col("ts").dt.hour() >= 24, "extract(hour FROM ts) >= 24"),
    (lambda: col("ts").dt.minute() > 59, "extract(minute FROM ts) > 59"),
    (lambda: col("ts").dt.dayofweek() == 7, "extract(dayofweek FROM ts) = 7"),
    (lambda: col("ts").dt.isodow() == 0, "extract(isodow FROM ts) = 0"),
    (lambda: col("ts").dt.dayofyear() == 367, "extract(dayofyear FROM ts) = 367"),
    (lambda: col("s").str.len_chars() < 0, "length(s) < 0"),
    (lambda: col("i").abs() == -5, "abs(i) = -5"),
    (lambda: col("i").abs() < 0, "abs(i) < 0"),
    (lambda: col("i").sign() == 5, "sign(i) = 5"),
    (lambda: col("s").str.to_uppercase() == "ab", "upper(s) = 'ab'"),
    (lambda: col("s").str.to_lowercase() == "AB", "lower(s) = 'AB'"),
    # The counting functions and the two lower-bounded float ones.
    (lambda: col("s").str.len_bytes() < 0, "strlen(s) < 0"),
    (lambda: col("s").str.count_matches("a") == -1, "length(regexp_extract_all(s, 'a')) = -1"),
    (lambda: col("i").abs().sqrt() < 0, "sqrt(abs(i)) < 0"),
    (lambda: col("i").exp() < 0, "exp(i) < 0"),
    # A widening cast against a literal no integer can equal.
    (lambda: col("i").cast("float64") == 5.5, "CAST(i AS DOUBLE) = 5.5"),
    (lambda: col("i").cast("float64") == -0.5, "CAST(i AS DOUBLE) = -0.5"),
]

#: The boundary values themselves, which the engine really does return — so these must keep
#: exactly the rows DuckDB keeps. The fixture spans a full year plus a leap day so every one of
#: them is actually reachable.
_IMAGE_SATISFIABLE = [
    (lambda: col("ts").dt.month() == 12, "extract(month FROM ts) = 12"),
    (lambda: col("ts").dt.month() == 1, "extract(month FROM ts) = 1"),
    (lambda: col("ts").dt.day() == 31, "extract(day FROM ts) = 31"),
    (lambda: col("ts").dt.day() == 29, "extract(day FROM ts) = 29"),
    (lambda: col("ts").dt.dayofweek() == 0, "extract(dayofweek FROM ts) = 0"),
    (lambda: col("ts").dt.dayofweek() == 6, "extract(dayofweek FROM ts) = 6"),
    (lambda: col("ts").dt.isodow() == 1, "extract(isodow FROM ts) = 1"),
    (lambda: col("ts").dt.isodow() == 7, "extract(isodow FROM ts) = 7"),
    (lambda: col("ts").dt.dayofyear() == 366, "extract(dayofyear FROM ts) = 366"),
    (lambda: col("ts").dt.quarter() == 4, "extract(quarter FROM ts) = 4"),
    (lambda: col("s").str.len_chars() == 0, "length(s) = 0"),
    (lambda: col("i").abs() == 0, "abs(i) = 0"),
    (lambda: col("i").sign() == -1, "sign(i) = -1"),
    (lambda: col("s").str.to_uppercase() == "AB", "upper(s) = 'AB'"),
    (lambda: col("s").str.to_lowercase() == "ab", "lower(s) = 'ab'"),
    (lambda: col("ts").dt.month() != 13, "extract(month FROM ts) <> 13"),
    (lambda: col("s").str.len_bytes() == 0, "strlen(s) = 0"),
    (lambda: col("s").str.count_matches("a") == 0, "length(regexp_extract_all(s, 'a')) = 0"),
    # `sqrt(0)` is `0`, and `exp` reaches zero by underflow — both inclusive bounds are
    # reachable, so neither of these may be refuted.
    (lambda: col("i").abs().sqrt() == 0, "sqrt(abs(i)) = 0"),
    # An integral literal is reachable; `<>` against a fractional one is true everywhere the
    # row is non-null, which is the case a constant TRUE would get wrong.
    (lambda: col("i").cast("float64") == 5.0, "CAST(i AS DOUBLE) = 5.0"),
    (lambda: col("i").cast("float64") != 5.5, "CAST(i AS DOUBLE) <> 5.5"),
    (lambda: col("i").cast("float64") > 5.5, "CAST(i AS DOUBLE) > 5.5"),
]


@pytest.fixture
def cal(duck):
    """A leap year of daily timestamps plus the strings and numbers the image rules touch.

    Every calendar boundary the refutations depend on is reachable in here: day 29 and 31,
    weekday 0 and 6, ISO weekday 1 and 7, day-of-year 366, quarter 4. An empty string and a
    zero make the `length`/`abs`/`sign` boundaries reachable too.
    """
    import datetime as dt

    days = [dt.datetime(2024, 1, 1) + dt.timedelta(days=n) for n in range(366)]
    tbl = pa.table(
        {
            "ts": pa.array([*days, None], pa.timestamp("us")),
            "s": pa.array(
                [("" if n % 7 == 0 else "AB" if n % 3 else "ab") for n in range(366)] + [None]
            ),
            "i": pa.array([n - 180 for n in range(366)] + [None], pa.int64()),
        }
    )
    duck.register("cal", tbl)
    return tbl


@pytest.mark.parametrize(("pred", "sql"), _IMAGE_REFUTED)
def test_image_refuted_predicate_matches_duckdb(duck, cal, pred, sql):
    out = bt.from_arrow(cal).filter(pred()).collect()
    assert out.num_rows == 0, f"{sql} was refuted but Batcher returned rows"
    assert_same(out, duck.sql(f"SELECT * FROM cal WHERE {sql}"))


@pytest.mark.parametrize(("pred", "sql"), _IMAGE_SATISFIABLE)
def test_image_boundary_value_matches_duckdb(duck, cal, pred, sql):
    out = bt.from_arrow(cal).filter(pred()).collect()
    assert out.num_rows > 0, f"{sql} should be reachable in the fixture"
    assert_same(out, duck.sql(f"SELECT * FROM cal WHERE {sql}"))


def test_image_refuted_disjunct_keeps_the_other_side(duck, cal):
    out = bt.from_arrow(cal).filter((col("i") > 180) | (col("ts").dt.month() == 13)).collect()
    assert_same(out, duck.sql("SELECT * FROM cal WHERE i > 180 OR extract(month FROM ts) = 13"))


# --- every execution path -----------------------------------------------------
#
# `collect()`, `collect(spill=True)` and `iter_batches()` are three schedulings of the same
# semantics, and the empty relation these rules produce has to mean the same thing to all
# three. The keyless-aggregate case is the one that has gone wrong here before: over empty
# input it emits one row, and a path that yields zero instead answers nothing where SQL answers
# 0. Distributed is covered by the mergeable-algebra equivalence suite, which runs the same
# plans; what is specific to these rules is that the emptiness is decided at *plan* time, so it
# must reach every path identically.


def _stream(ds) -> pa.Table:
    """`iter_batches()` collected back into a table."""
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches, schema=batches[0].schema)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda ds: ds.filter(col("i") % 10 == 15), id="filter"),
        pytest.param(lambda ds: ds.filter(col("ts").dt.month() == 13), id="filter-image"),
        pytest.param(
            lambda ds: ds.filter(col("i") % 10 == 15).group_by("s").agg(n=col("i").count()),
            id="grouped-agg",
        ),
        pytest.param(
            lambda ds: ds.filter(col("i") % 10 == 15).agg(n=col("i").count()),
            id="keyless-agg",
        ),
        pytest.param(lambda ds: ds.filter(col("i") * 2 == 7).sort("i").limit(3), id="sort-limit"),
        pytest.param(lambda ds: ds.filter(col("i") % 10 == 15).distinct(), id="distinct"),
    ],
)
def test_every_execution_path_agrees(cal, build):
    from _harness import assert_tables_equal

    oracle = build(bt.from_arrow(cal)).collect()
    assert_tables_equal(build(bt.from_arrow(cal)).collect(spill=True), oracle)
    assert_tables_equal(_stream(build(bt.from_arrow(cal))), oracle)


def test_keyless_count_over_a_refuted_filter_answers_zero_on_every_path(duck, cal):
    # The specific historical failure: a keyless aggregate over empty input emits ONE row
    # (count 0), and a path that yields zero rows answers nothing where SQL answers 0.
    build = lambda ds: ds.filter(col("ts").dt.month() == 13).agg(n=col("i").count())  # noqa: E731
    oracle = duck.sql("SELECT count(i) AS n FROM cal WHERE extract(month FROM ts) = 13")
    assert_same(build(bt.from_arrow(cal)).collect(), oracle)
    assert_same(_stream(build(bt.from_arrow(cal))), oracle)
