"""The third-wave rewrites must match DuckDB after the full optimizer runs.

Covers the two half-integer rounding intervals, the `isnan`/`isinf` collapse through the
class-preserving functions, the `<>` complement of the year/decade sargable family, and the
extended `CASE` pushes. The float fixture is built around the ties (`.5` on both sides of
zero) that separate half-away-from-zero rounding from half-to-even, plus NaN and the
infinities.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.conditional_algebra
import batcher.kyber.rules.math_algebra
import batcher.kyber.rules.predicate_algebra
from _harness import assert_same
from batcher import col, lit
from batcher.plan.expr_ir.core import IsInf, IsNan, MathExpr


@pytest.fixture
def instants_2024(duck):
    """Instants straddling week (Monday) and quarter boundaries, including the Q4 rollover."""
    tbl = pa.table(
        {
            "ts": pa.array(
                [
                    dt.datetime(2024, 5, 12, 23, 59, 59),
                    dt.datetime(2024, 5, 13, 0, 0, 0),
                    dt.datetime(2024, 5, 15, 13, 30, 15),
                    dt.datetime(2024, 5, 19, 23, 59, 59),
                    dt.datetime(2024, 5, 20, 0, 0, 0),
                    dt.datetime(2024, 3, 31, 23, 59, 59),
                    dt.datetime(2024, 4, 1, 0, 0, 0),
                    dt.datetime(2024, 6, 30, 23, 59, 59),
                    dt.datetime(2024, 10, 1, 0, 0, 0),
                    dt.datetime(2024, 12, 31, 23, 59, 59),
                    dt.datetime(2025, 1, 1, 0, 0, 0),
                    None,
                ],
                type=pa.timestamp("us"),
            )
        }
    )
    duck.register("q", tbl)
    return tbl


@pytest.fixture
def instants(duck):
    """Instants straddling the hour, minute and second boundaries the sub-day bands use."""
    tbl = pa.table(
        {
            "ts": pa.array(
                [
                    dt.datetime(2024, 1, 5, 12, 59, 59),
                    dt.datetime(2024, 1, 5, 13, 0, 0),
                    dt.datetime(2024, 1, 5, 13, 29, 59),
                    dt.datetime(2024, 1, 5, 13, 30, 0),
                    dt.datetime(2024, 1, 5, 13, 30, 15),
                    dt.datetime(2024, 1, 5, 13, 30, 16),
                    dt.datetime(2024, 1, 5, 13, 59, 59, 999_999),
                    dt.datetime(2024, 1, 5, 14, 0, 0),
                    None,
                ],
                type=pa.timestamp("us"),
            )
        }
    )
    duck.register("i", tbl)
    return tbl


@pytest.fixture
def wide_years(duck):
    """Instants straddling every century and millennium boundary the buckets turn on."""
    tbl = pa.table(
        {
            "ts": pa.array(
                [
                    dt.datetime(1900, 12, 31), dt.datetime(1901, 1, 1),
                    dt.datetime(2000, 12, 31), dt.datetime(2001, 1, 1),
                    dt.datetime(2020, 6, 1), dt.datetime(2100, 12, 31),
                    dt.datetime(2101, 1, 1), None,
                ],
                type=pa.timestamp("us"),
            )
        }
    )  # fmt: skip
    duck.register("w", tbl)
    return tbl


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "f": pa.array(
                [
                    0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 2.49, 2.51, 0.0, -0.0,
                    float("nan"), float("inf"), float("-inf"), None,
                ],
                type=pa.float64(),
            ),
            "x": pa.array([1] * 15, type=pa.int64()),
            "ts": pa.array(
                [dt.datetime(2019, 12, 31)] * 5
                + [dt.datetime(2020, 6, 1)] * 5
                + [dt.datetime(2021, 1, 1)] * 4
                + [None],
                type=pa.timestamp("us"),
            ),
        }
    )  # fmt: skip
    duck.register("t", tbl)
    return tbl


_ROUNDING_CASES = [
    ("round", op, k)
    for op, k in [
        ("=", 0), ("=", 1), ("=", 2), ("=", -2), ("=", 3),
        (">=", 0), (">=", 1), ("<=", 0), ("<=", 1), (">", 0), ("<", 0), ("<>", 2),
    ]
]  # fmt: skip

_SQL_OPS = {"=": "eq", ">=": "ge", "<=": "le", ">": "gt", "<": "lt", "<>": "ne"}


@pytest.mark.parametrize(
    ("fn", "op", "k"),
    _ROUNDING_CASES,
    ids=[f"{f}_{_SQL_OPS[o]}_{k}" for f, o, k in _ROUNDING_CASES],
)
def test_round_interval_matches_duckdb(duck, t, fn, op, k):
    from batcher.plan.expr_ir import Binary

    expr = Binary(_SQL_OPS[op], MathExpr(fn, col("f")), lit(k))
    out = bt.from_arrow(t).select(r=expr).collect()
    # DuckDB's `round` is half-away-from-zero, matching the engine, so it is a usable
    # oracle for this family.
    assert_same(out, duck.sql(f"SELECT {fn}(f) {op} {k} AS r FROM t"))


@pytest.mark.parametrize(
    ("op", "k"),
    [
        ("eq", 0), ("eq", 1), ("eq", 2), ("eq", -2), ("eq", 3),
        ("ge", 0), ("ge", 1), ("le", 0), ("le", 1), ("gt", 0), ("lt", 0), ("ne", 2),
    ],
)  # fmt: skip
def test_rint_interval_matches_the_engine_own_rounding(t, op, k):
    """`rint` has no DuckDB counterpart, so the oracle is the engine's own `rint`.

    DuckDB carries no half-to-even rounding function (`rint`, `round_even` and
    `nearbyint` are all absent), so there is nothing to compare the rewritten predicate
    against in SQL. What the rule claims is that its interval selects exactly the rows
    whose `rint` equals the bucket — and that is checkable directly: the bare `rint`
    projection is not a comparison, so no interval rule touches it, and comparing the two
    in Python is a genuine check of the rewrite rather than a restatement of it.
    """
    import operator

    from batcher.plan.expr_ir import Binary

    ds = bt.from_arrow(t)
    direct = ds.select(r=MathExpr("rint", col("f"))).collect().column("r").to_pylist()
    predicate = Binary(op, MathExpr("rint", col("f")), lit(k))
    got = ds.select(r=predicate).collect().column("r").to_pylist()
    compare = {
        "eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
        "le": operator.le, "gt": operator.gt, "ge": operator.ge,
    }[op]  # fmt: skip
    want = [None if v is None else compare(v, k) for v in direct]
    # A NaN sorts above every finite value in the engine's total order, so the Python
    # comparison is only the oracle on the finite rows; the NaN row is checked by the
    # plan-shape tests instead.
    for got_value, want_value, source in zip(got, want, direct, strict=True):
        if source is None or source != source:
            continue
        assert got_value == want_value, f"rint({source}) {op} {k}"


@pytest.mark.parametrize("fn", ["abs", "ceil", "floor", "round", "trunc"])
def test_non_finite_check_through_rounding_matches_duckdb(duck, t, fn):
    out = (
        bt.from_arrow(t)
        .select(a=IsNan(MathExpr(fn, col("f"))), b=IsInf(MathExpr(fn, col("f"))))
        .collect()
    )
    assert_same(out, duck.sql(f"SELECT isnan({fn}(f)) AS a, isinf({fn}(f)) AS b FROM t"))


def test_non_finite_check_through_rint_matches_the_engine_own_rounding(t):
    """`rint` again has no DuckDB counterpart, so the check is against the bare call."""
    ds = bt.from_arrow(t)
    direct = ds.select(r=MathExpr("rint", col("f"))).collect().column("r").to_pylist()
    for check, classify in (
        (IsNan, lambda v: v != v),
        (IsInf, lambda v: v in (float("inf"), float("-inf"))),
    ):
        got = ds.select(r=check(MathExpr("rint", col("f")))).collect().column("r").to_pylist()
        want = [None if v is None else classify(v) for v in direct]
        assert got == want


def test_non_finite_check_through_sign_is_untouched_and_still_matches(duck, t):
    out = bt.from_arrow(t).select(r=IsNan(MathExpr("sign", col("f")))).collect()
    assert_same(out, duck.sql("SELECT isnan(sign(f)) AS r FROM t"))


@pytest.mark.parametrize("year", [2019, 2020, 2021])
def test_year_inequality_matches_duckdb(duck, t, year):
    out = bt.from_arrow(t).select(r=col("ts").dt.year() != lit(year)).collect()
    assert_same(out, duck.sql(f"SELECT year(ts) <> {year} AS r FROM t"))


def test_year_inequality_inside_a_filter_matches_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("ts").dt.year() != lit(2020)).select(ts=col("ts")).collect()
    assert_same(out, duck.sql("SELECT ts FROM t WHERE year(ts) <> 2020"))


@pytest.mark.parametrize("fn", ["century", "millennium"])
@pytest.mark.parametrize("op", ["=", "<>", ">=", ">", "<=", "<"])
def test_century_and_millennium_bands_match_duckdb(duck, wide_years, fn, op):
    """The two 1-based year buckets added to the sargable family.

    A century is 1901-2000, not 1900-1999, so the oracle is the only thing that proves the
    origin is right — an off-by-one would shift every band by a year and still look
    plausible in a plan-shape test.
    """
    value = {"century": 21, "millennium": 3}[fn]
    expr = getattr(bt.col("ts").dt, fn)()
    predicate = {
        "=": expr == lit(value), "<>": expr != lit(value), ">=": expr >= lit(value),
        ">": expr > lit(value), "<=": expr <= lit(value), "<": expr < lit(value),
    }[op]  # fmt: skip
    out = bt.from_arrow(wide_years).select(r=predicate).collect()
    assert_same(out, duck.sql(f"SELECT {fn}(ts) {op} {value} AS r FROM w"))


def test_century_band_inside_a_filter_matches_duckdb(duck, wide_years):
    out = (
        bt.from_arrow(wide_years)
        .filter(col("ts").dt.century() == lit(21))
        .select(ts=col("ts"))
        .collect()
    )
    assert_same(out, duck.sql("SELECT ts FROM w WHERE century(ts) = 21"))


@pytest.mark.parametrize(
    ("expr", "sql"),
    [
        (
            lambda: col("ts").dt.truncate("month") != lit(dt.datetime(2020, 1, 1)),
            "date_trunc('month', ts) <> TIMESTAMP '2020-01-01'",
        ),
        (
            lambda: col("ts").dt.truncate("year") != lit(dt.datetime(2020, 1, 1)),
            "date_trunc('year', ts) <> TIMESTAMP '2020-01-01'",
        ),
        (lambda: col("ts").dt.strftime("%Y") != lit("2020"), "strftime(ts, '%Y') <> '2020'"),
        (
            lambda: col("ts").dt.strftime("%Y-%m") != lit("2020-06"),
            "strftime(ts, '%Y-%m') <> '2020-06'",
        ),
    ],
    ids=["trunc_month_ne", "trunc_year_ne", "strftime_year_ne", "strftime_month_ne"],
)
def test_temporal_band_complements_match_duckdb(duck, wide_years, expr, sql):
    """The `<>` arm added to the `date_trunc` and `strftime` band families.

    A band complement is where an off-by-one shows up as a whole extra unit of rows, and
    where three-valued logic bites: a NULL instant must stay NULL rather than becoming
    "outside the band", which the oracle checks along with the boundaries.
    """
    out = bt.from_arrow(wide_years).select(r=expr()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM w"))


@pytest.mark.parametrize(
    ("fmt", "value"),
    [
        ("%Y-%m-%d %H", "2024-01-05 13"),
        ("%Y-%m-%d %H:%M", "2024-01-05 13:30"),
        ("%Y-%m-%d %H:%M:%S", "2024-01-05 13:30:15"),
        ("%Y%m", "202401"),
        ("%Y%m%d", "20240105"),
        ("%Y/%m/%d", "2024/01/05"),
        ("%Y-%m-%dT%H:%M:%S", "2024-01-05T13:30:15"),
        ("%Y-%m-%dT%H", "2024-01-05T13"),
        ("%Y-%m-%dT%H:%M", "2024-01-05T13:30"),
    ],
)
@pytest.mark.parametrize("op", ["=", "<>", ">=", "<"])
def test_extended_strftime_bands_match_duckdb(duck, instants, fmt, value, op):
    """The nine formats added to the `strftime` band family.

    Their rendering is fixed-width and most-significant-field-first, so string order is
    chronological — but only if the padding really is what it is assumed to be, which is
    what the oracle checks. The fixture straddles each unit boundary in both directions.
    """
    expr = col("ts").dt.strftime(fmt)
    predicate = {
        "=": expr == lit(value), "<>": expr != lit(value),
        ">=": expr >= lit(value), "<": expr < lit(value),
    }[op]  # fmt: skip
    out = bt.from_arrow(instants).select(r=predicate).collect()
    assert_same(out, duck.sql(f"SELECT strftime(ts, '{fmt}') {op} '{value}' AS r FROM i"))


def test_sub_day_strftime_band_inside_a_filter_matches_duckdb(duck, instants):
    out = (
        bt.from_arrow(instants)
        .filter(col("ts").dt.strftime("%Y-%m-%d %H") == lit("2024-01-05 13"))
        .select(ts=col("ts"))
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT ts FROM i WHERE strftime(ts, '%Y-%m-%d %H') = '2024-01-05 13'")
    )


@pytest.mark.parametrize(
    ("unit", "literal"),
    [
        ("week", "2024-05-13"),
        ("quarter", "2024-04-01"),
        ("quarter", "2024-10-01"),
        ("month", "2024-05-01"),
        ("decade", "2020-01-01"),
        ("century", "2000-01-01"),
        ("millennium", "2000-01-01"),
    ],
)
@pytest.mark.parametrize("op", ["=", "<>", ">=", "<"])
def test_coarse_truncation_bands_match_duckdb(duck, instants_2024, unit, literal, op):
    """The `week`, `quarter`, `decade`, `century` and `millennium` units added to the
    `date_trunc` band family.

    These are where a truncation band is easiest to get wrong: a week starts on Monday (not
    Sunday, and not the 1st), a quarter step of three calendar months has to roll the year
    over from Q4, and the multi-year buckets are **0-based** (2000-2099 is a century) —
    which is the opposite convention from the `century()` extraction. The oracle checks
    each boundary in both directions.
    """
    value = dt.datetime.fromisoformat(literal)
    expr = col("ts").dt.truncate(unit)
    predicate = {
        "=": expr == lit(value), "<>": expr != lit(value),
        ">=": expr >= lit(value), "<": expr < lit(value),
    }[op]  # fmt: skip
    out = bt.from_arrow(instants_2024).select(r=predicate).collect()
    assert_same(
        out,
        duck.sql(
            f"SELECT date_trunc('{unit}', ts) {op} TIMESTAMP '{literal} 00:00:00' AS r FROM q"
        ),
    )


@pytest.mark.parametrize(
    ("unit", "literal"),
    [
        ("week", "2024-05-14"),
        ("quarter", "2024-05-01"),
        # `date_trunc` buckets centuries 0-based (2000-2099), so 2001 is *not* a boundary —
        # the opposite of the `century()` extraction, where century 21 starts in 2001.
        ("century", "2001-01-01"),
        ("decade", "2024-01-01"),
    ],
)
def test_unaligned_truncation_literal_is_untouched_and_still_matches(
    duck, instants_2024, unit, literal
):
    """An unaligned literal must be left alone, and still select the same rows.

    Compared *inside a filter* rather than as a projected boolean, deliberately. The
    predicate is unsatisfiable, and DuckDB's own optimizer folds an unsatisfiable
    comparison to `false` — including on the NULL row, where the engine (correctly)
    answers NULL. Under a filter both spellings drop the row, so the comparison is
    against the rows selected, which is what the rule is about.
    """
    out = (
        bt.from_arrow(instants_2024)
        .filter(col("ts").dt.truncate(unit) == lit(dt.datetime.fromisoformat(literal)))
        .select(ts=col("ts"))
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            f"SELECT ts FROM q WHERE date_trunc('{unit}', ts) = TIMESTAMP '{literal} 00:00:00'"
        ),
    )


def test_extended_case_pushes_match_duckdb(duck, t):
    out = (
        bt.from_arrow(t)
        .select(
            a=bt.when(col("x") > lit(0)).then(col("ts")).otherwise(col("ts")).dt.offset_by("1d"),
            b=bt.when(col("x") > lit(0)).then(lit(1)).otherwise(lit(3)).is_in([1, 2]),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT (CASE WHEN x > 0 THEN ts ELSE ts END) + INTERVAL 1 DAY AS a, "
            "(CASE WHEN x > 0 THEN 1 ELSE 3 END) IN (1, 2) AS b FROM t"
        ),
    )


def test_membership_over_a_literal_input_matches_duckdb(duck, t):
    out = bt.from_arrow(t).select(a=lit(1).is_in([1, 2]), b=lit(9).is_in([1, 2])).collect()
    assert_same(out, duck.sql("SELECT 1 IN (1, 2) AS a, 9 IN (1, 2) AS b FROM t"))


# --- constant folding of the date extractions -------------------------------

_FOLD_BOUNDARIES = [
    # The weekday extractions have seven possible outputs, so two consecutive days at the
    # Sunday/Monday seam cover the whole space where a convention mismatch could hide.
    ("day_of_week", dt.date(2024, 1, 7)),
    ("day_of_week", dt.date(2024, 1, 8)),
    ("isodow", dt.date(2024, 1, 7)),
    ("isodow", dt.date(2024, 1, 8)),
    # The Gregorian century rule: 2100 is divisible by four and is *not* a leap year.
    ("is_leap_year", dt.date(2100, 2, 1)),
    ("is_leap_year", dt.date(2024, 2, 1)),
    ("days_in_month", dt.date(2100, 2, 1)),
    ("days_in_month", dt.date(2024, 2, 1)),
    # ISO week numbering at a year boundary, where a calendar-week implementation diverges.
    ("week", dt.date(2021, 1, 1)),
    ("iso_year", dt.date(2021, 1, 1)),
    ("week", dt.date(2024, 12, 30)),
    ("iso_year", dt.date(2024, 12, 30)),
    # `epoch` floors rather than truncating, so a pre-1970 instant is the discriminating case.
    ("epoch", dt.datetime(1969, 12, 31, 23, 59, 59, 500_000)),
    ("epoch", dt.date(2021, 1, 1)),
]


@pytest.mark.parametrize(
    ("fn", "value"), _FOLD_BOUNDARIES, ids=[f"{f}_{v}" for f, v in _FOLD_BOUNDARIES]
)
def test_folded_date_extraction_equals_the_engine_kernel(fn, value):
    """A plan-time fold must produce exactly what the engine's kernel would.

    The fold runs in Python, so the risk is a convention mismatch rather than a bug:
    Sunday-0 against Monday-0, ISO weeks against calendar weeks, floor against truncate.
    The oracle here is the engine itself — the same function over a one-row *column*, which
    no fold rule touches — and every case is a boundary where the two conventions differ.
    """
    from batcher.plan.expr_ir.func_nodes import DateFunc

    column = "d" if not isinstance(value, dt.datetime) else "t"
    source = bt.from_pydict({column: [value]})
    kernel = source.select(r=DateFunc(fn, col(column))).collect().column("r").to_pylist()[0]
    folded = source.select(r=DateFunc(fn, lit(value))).collect().column("r").to_pylist()[0]
    assert folded == kernel
