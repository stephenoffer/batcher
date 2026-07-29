"""The expression rewrite rules must be semantics-preserving — checked against DuckDB.

`kyber/rules/exprs/` holds roughly fifty rules that rewrite one expression into a cheaper one:
`x = x` to a constant, `x * 0` to zero, an anchored regex to `starts_with`, a reduction pushed
through `unique`. None of them was mentioned by any test, and this is the rule family where being
wrong is worst: an optimizer that changes an answer produces no error, and the cheaper plan is
the one that ships.

The traps are null and NaN. `x = x` is NULL for a null `x`, not TRUE. `x * 0` is NULL, not 0.
`min(unique(x))` equals `min(x)` but `sum(unique(x))` does not equal `sum(x)`. Each case below
pins the semantics DuckDB gives, so a rule that widens past its guard fails here.

All of these passed when written; the file exists so a future rule cannot quietly break them.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

_SELF_COMPARISONS = [
    ("eq", lambda c: c == c, "="),
    ("ne", lambda c: c != c, "<>"),
    ("ge", lambda c: c >= c, ">="),
    ("le", lambda c: c <= c, "<="),
    ("gt", lambda c: c > c, ">"),
    ("lt", lambda c: c < c, "<"),
]


@pytest.fixture
def nullable() -> pa.Table:
    return pa.table(
        {
            "i": pa.array([1, 2, None], pa.int64()),
            "f": pa.array([1.5, float("nan"), None], pa.float64()),
        }
    )


@pytest.mark.parametrize(("name", "build", "op"), _SELF_COMPARISONS)
@pytest.mark.parametrize("column", ["i", "f"])
def test_a_self_comparison_keeps_null_semantics(duck, nullable, name, build, op, column):
    """`x = x` folds to TRUE only where `x` is known: a null row stays null, not true."""
    got = bt.from_arrow(nullable).select(r=build(bt.col(column))).collect()
    duck.register("t", nullable)
    assert_same(got, duck.sql(f"SELECT {column} {op} {column} AS r FROM t"))


@pytest.mark.parametrize(
    ("label", "build", "sql"),
    [
        ("mul by zero", lambda: bt.col("i") * bt.lit(0), "i * 0"),
        ("mod by one", lambda: bt.col("i") % bt.lit(1), "i % 1"),
        ("div by one", lambda: bt.col("i") / bt.lit(1), "i / 1.0"),
    ],
)
def test_an_arithmetic_fold_keeps_null_semantics(duck, nullable, label, build, sql):
    """`x * 0` is NULL for a null `x`; folding it to a literal zero would invent a value."""
    got = bt.from_arrow(nullable).select(r=build()).collect()
    duck.register("t", nullable)
    assert_same(got, duck.sql(f"SELECT {sql} AS r FROM t"))


@pytest.mark.parametrize("column", ["i"])
@pytest.mark.parametrize(
    ("label", "method"), [("is_nan", "is_nan"), ("is_infinite", "is_infinite")]
)
def test_a_nan_check_on_an_integer_column_keeps_nulls(duck, nullable, column, label, method):
    """An integer is never NaN, but a *null* integer is not `False` either."""
    got = bt.from_arrow(nullable).select(r=getattr(bt.col(column), method)()).collect()
    duck.register("t", nullable)
    fn = "isnan" if method == "is_nan" else "isinf"
    assert_same(got, duck.sql(f"SELECT {fn}(CAST({column} AS DOUBLE)) AS r FROM t"))


@pytest.fixture
def strings() -> pa.Table:
    return pa.table(
        {"s": pa.array(["hello", "HELLO", "hel", "", None, "a.c", "xhellox", "hello\nworld"])}
    )


@pytest.mark.parametrize(
    "pattern",
    [
        "^hel",  # -> starts_with
        "^hello",
        "lo$",  # -> ends_with
        "hello$",
        "^hello$",  # -> equality
        "^$",  # the empty string, anchored
        "ell",  # -> contains
        "a.c",  # the dot is a metacharacter, so this must NOT become a literal
        r"a\.c",  # ...and this one must
        "^",
        "$",
        "^HELLO$",
        "hel.*",
        "(?i)^hel",
        "^h.l",
        r"\d",
        "^[a-z]+$",
    ],
)
def test_a_rewritten_regex_matches_duckdb(duck, strings, pattern):
    """The prefix/suffix/anchored rewrites must not change which rows match."""
    got = bt.from_arrow(strings).select(r=bt.col("s").str.regexp_matches(pattern)).collect()
    duck.register("t", strings)
    assert_same(got, duck.sql("SELECT regexp_matches(s, ?) AS r FROM t", params=[pattern]))


# --------------------------------------------------------------------------- #
# List rewrites. DuckDB is the oracle for the reduction semantics, including
# `list_sum` of an empty list being NULL rather than zero.
# --------------------------------------------------------------------------- #
_LISTS = [[3, 1, 2, 1, 3], [5], [], None, [1, None, 2], [7, 7, 7], [-2, -2, 5]]


@pytest.fixture
def lists() -> pa.Table:
    return pa.table({"a": pa.array(_LISTS, pa.list_(pa.int64()))})


@pytest.mark.parametrize(
    ("label", "method", "sql"),
    [
        ("min", "min", "list_min(list_distinct(a))"),
        ("max", "max", "list_max(list_distinct(a))"),
    ],
)
def test_a_set_reading_reduction_survives_being_pushed_through_unique(
    duck, lists, label, method, sql
):
    """`min(unique(x))` is `min(x)`: dedup changes multiplicity, never the set of values."""
    through_unique = (
        bt.from_arrow(lists).select(r=getattr(bt.col("a").list.unique().list, method)()).collect()
    )
    direct = bt.from_arrow(lists).select(r=getattr(bt.col("a").list, method)()).collect()
    assert through_unique.to_pydict() == direct.to_pydict(), (
        f"{label} must be unaffected by a preceding unique()"
    )
    duck.register("t", lists)
    assert_same(through_unique, duck.sql(f"SELECT {sql} AS r FROM t"))


@pytest.mark.parametrize("method", ["sum", "len", "mean"])
def test_a_multiplicity_reading_reduction_is_not_pushed_through_unique(lists, method):
    """`sum(unique(x))` is NOT `sum(x)`, so the rule must leave these alone."""
    through_unique = (
        bt.from_arrow(lists).select(r=getattr(bt.col("a").list.unique().list, method)()).to_pydict()
    )
    direct = bt.from_arrow(lists).select(r=getattr(bt.col("a").list, method)()).to_pydict()
    # The duplicate-bearing rows must differ; if they match, a reduction was pushed through.
    assert through_unique != direct, (
        f"{method}(unique(x)) equals {method}(x), so the dedup was dropped or pushed through"
    )


@pytest.mark.parametrize("method", ["sum", "min", "max", "len"])
def test_a_reduction_is_unchanged_by_a_preceding_reverse(lists, method):
    """`reverse` is a permutation, so every reduction is invariant under it."""
    reduce = getattr(bt.col("a").list.reverse().list, method)
    reversed_first = bt.from_arrow(lists).select(r=reduce()).to_pydict()
    direct = bt.from_arrow(lists).select(r=getattr(bt.col("a").list, method)()).to_pydict()
    assert reversed_first == direct


def test_reverse_is_an_involution(lists) -> None:
    got = bt.from_arrow(lists).select(r=bt.col("a").list.reverse().list.reverse()).to_pydict()["r"]
    assert got == _LISTS


def test_unique_is_idempotent_and_keeps_first_seen_order(lists) -> None:
    once = bt.from_arrow(lists).select(r=bt.col("a").list.unique()).to_pydict()["r"]
    twice = bt.from_arrow(lists).select(r=bt.col("a").list.unique().list.unique()).to_pydict()["r"]
    assert once == twice
    assert once[0] == [3, 1, 2], "first-seen order, as documented"


def test_a_full_slice_is_the_input(lists) -> None:
    assert bt.from_arrow(lists).select(r=bt.col("a").list.slice(0)).to_pydict()["r"] == _LISTS


def test_a_slice_of_a_slice_composes(lists) -> None:
    got = bt.from_arrow(lists).select(r=bt.col("a").list.slice(1, 3).list.slice(1, 2)).to_pydict()
    want = [None if v is None else v[1:4][1:3] for v in _LISTS]
    assert got["r"] == want


def test_a_get_past_the_end_is_null(lists) -> None:
    assert bt.from_arrow(lists).select(r=bt.col("a").list.get(99)).to_pydict()["r"] == [None] * 7


def test_a_field_of_a_constructed_struct_is_the_input() -> None:
    ds = bt.from_pydict({"x": [1, 2, None], "y": ["a", "b", None]})
    got = ds.select(r=bt.struct(x=bt.col("x"), y=bt.col("y")).struct.field("x")).to_pydict()
    assert got["r"] == [1, 2, None]


# --------------------------------------------------------------------------- #
# Temporal rewrites. `combine_adjacent_date_offsets` fires only when both
# offsets carry zero months, and that guard is the correctness argument.
# --------------------------------------------------------------------------- #
_MONTH_END_DATES = [
    dt.date(2024, 1, 31),  # +1mo clamps to Feb 29 (leap), +1mo again lands Mar 29
    dt.date(2023, 1, 31),  # ...and Feb 28 in a non-leap year, so Mar 28
    dt.date(2024, 3, 31),
    dt.date(2024, 5, 31),
    dt.date(2024, 1, 29),  # no clamping, so the two forms agree here
    dt.date(2024, 2, 29),
    dt.date(2024, 12, 31),
]


@pytest.fixture
def month_ends() -> pa.Table:
    return pa.table({"d": pa.array(_MONTH_END_DATES, pa.date32())})


def test_stacked_month_offsets_are_not_fused(month_ends) -> None:
    """Month arithmetic clamps, so it is not associative and the offsets must stay separate.

    January 31 plus one month is February 29, and one further month is March 29, while January 31
    plus *two* months is March 31. Fusing the pair would move four of these seven dates. If this
    test fails, `combine_adjacent_date_offsets` has widened past its zero-months guard.
    """
    stacked = (
        bt.from_arrow(month_ends)
        .select(r=bt.col("d").dt.offset_by("1mo").dt.offset_by("1mo"))
        .to_pydict()["r"]
    )
    fused = bt.from_arrow(month_ends).select(r=bt.col("d").dt.offset_by("2mo")).to_pydict()["r"]

    differing = [d for d, a, b in zip(_MONTH_END_DATES, stacked, fused, strict=True) if a != b]
    assert len(differing) == 4, (
        f"expected the clamping dates to differ between +1mo+1mo and +2mo, got {differing}"
    )
    assert stacked[0] == dt.date(2024, 3, 29)
    assert fused[0] == dt.date(2024, 3, 31)


def test_stacked_month_offsets_match_duckdb(duck, month_ends) -> None:
    got = (
        bt.from_arrow(month_ends)
        .select(r=bt.col("d").dt.offset_by("1mo").dt.offset_by("1mo"))
        .collect()
    )
    duck.register("t", month_ends)
    assert_same(
        got,
        duck.sql("SELECT CAST((d + INTERVAL 1 MONTH) + INTERVAL 1 MONTH AS DATE) AS r FROM t"),
    )


def test_stacked_day_offsets_fuse_to_the_same_dates(duck, month_ends) -> None:
    """Days are exact durations with no clamping, so fusing them is the same function."""
    stacked = (
        bt.from_arrow(month_ends)
        .select(r=bt.col("d").dt.offset_by("5d").dt.offset_by("3d"))
        .to_pydict()["r"]
    )
    fused = bt.from_arrow(month_ends).select(r=bt.col("d").dt.offset_by("8d")).to_pydict()["r"]
    assert stacked == fused

    duck.register("t", month_ends)
    assert_same(
        bt.from_arrow(month_ends).select(r=bt.col("d").dt.offset_by("8d")).collect(),
        duck.sql("SELECT CAST(d + INTERVAL 8 DAY AS DATE) AS r FROM t"),
    )


def test_a_zero_offset_leaves_the_date_alone(month_ends) -> None:
    got = bt.from_arrow(month_ends).select(r=bt.col("d").dt.offset_by("0d")).to_pydict()["r"]
    assert got == _MONTH_END_DATES


# --------------------------------------------------------------------------- #
# Constant folds must agree with the runtime. A fold only fires on a literal,
# so a fold that computes a different value than the engine makes the *same*
# expression return two answers depending on where its input came from.
# --------------------------------------------------------------------------- #
_FOLD_INPUTS = ["hello", "", "Hello World", "MiXeD cAsE", "  pad  ", "123", "hi"]

_STRING_FOLDS = [
    ("md5", lambda e: e.str.md5()),
    ("sha1", lambda e: e.str.sha1()),
    ("sha256", lambda e: e.str.sha256()),
    ("crc32", lambda e: e.str.crc32()),
    ("hex", lambda e: e.str.hex()),
    ("ascii", lambda e: e.str.ascii()),
    ("initcap", lambda e: e.str.initcap()),
    ("reverse", lambda e: e.str.reverse()),
    ("bit_length", lambda e: e.str.bit_length()),
    ("octet_length", lambda e: e.str.octet_length()),
    ("trim", lambda e: e.str.strip_chars()),
    ("ltrim", lambda e: e.str.strip_chars_start()),
    ("rtrim", lambda e: e.str.strip_chars_end()),
    ("repeat", lambda e: e.str.repeat(2)),
    ("lpad", lambda e: e.str.pad_start(10)),
    ("rpad", lambda e: e.str.pad_end(10)),
]


@pytest.mark.parametrize(("name", "build"), _STRING_FOLDS)
def test_folding_a_literal_equals_running_on_a_column(name, build):
    """The invariant that catches a fold reimplementing a function slightly differently.

    Three folds failed this when it was written. `hex` used Python's `bytes.hex()`, which is
    lowercase where the engine returns uppercase, so `hex(col) = hex('needle')` could never
    match. `lpad`/`rpad` used `str.rjust`/`str.ljust`, which leave an over-long input alone
    where SQL truncates it, so `lpad('Hello World', 10)` folded to `'Hello World'` against the
    runtime's `'Hello Worl'`.
    """
    table = pa.table({"s": pa.array(_FOLD_INPUTS)})
    ds = bt.from_arrow(table)

    on_column = ds.select(r=build(bt.col("s"))).to_pydict()["r"]
    on_literal = [
        ds.limit(1).select(r=build(bt.lit(value))).to_pydict()["r"][0] for value in _FOLD_INPUTS
    ]

    assert on_literal == on_column, (
        f"{name}: folding a literal disagrees with running on a column, so the same expression "
        f"returns two different answers depending on where its input came from"
    )


@pytest.mark.parametrize("value", _FOLD_INPUTS)
def test_hex_folds_to_uppercase_like_the_engine(duck, value):
    ds = bt.from_arrow(pa.table({"s": pa.array([value])}))
    folded = ds.select(r=bt.lit(value).str.hex()).collect()
    duck.register("t", pa.table({"s": pa.array([value])}))
    assert_same(folded, duck.sql("SELECT hex(s) AS r FROM t"))


@pytest.mark.parametrize("width", [1, 5, 10, 11, 20])
@pytest.mark.parametrize("method", ["pad_start", "pad_end"])
def test_a_pad_fold_truncates_an_over_long_literal(duck, width, method):
    """SQL pads *to* a width, so an input already wider than it is cut down, not left alone."""
    value = "Hello World"
    ds = bt.from_arrow(pa.table({"s": pa.array([value])}))
    folded = ds.select(r=getattr(bt.lit(value).str, method)(width)).collect()
    duck.register("t", pa.table({"s": pa.array([value])}))
    fn = "lpad" if method == "pad_start" else "rpad"
    assert_same(folded, duck.sql(f"SELECT {fn}(s, {width}, ' ') AS r FROM t"))


# --------------------------------------------------------------------------- #
# `fold_cast_of_literal` is the highest-risk fold: a rounding convention or an
# overflow handled differently from the engine changes a value, not a plan.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [1.4, 1.5, 2.5, -1.5, -2.5, 0.5, -0.5, 3.7])
def test_folding_a_float_to_int_cast_uses_the_engine_rounding(duck, value):
    """`.5` is the whole question: half-even, half-away, or truncate."""
    table = pa.table({"c": pa.array([value], pa.float64())})
    ds = bt.from_arrow(table)

    on_literal = ds.select(r=bt.lit(value).cast("int64")).to_pydict()["r"]
    on_column = ds.select(r=bt.col("c").cast("int64")).to_pydict()["r"]
    assert on_literal == on_column

    duck.register("t", table)
    assert_same(
        ds.select(r=bt.lit(value).cast("int64")).collect(),
        duck.sql("SELECT CAST(c AS BIGINT) AS r FROM t"),
    )


@pytest.mark.parametrize(
    ("value", "arrow_type", "dtype"),
    [
        (float("nan"), pa.float64(), "int64"),
        (float("inf"), pa.float64(), "int64"),
        (1e30, pa.float64(), "int64"),
        (2**31, pa.int64(), "int32"),
        (2**63 - 1, pa.int64(), "int32"),
        ("abc", pa.string(), "int64"),
        ("", pa.string(), "int64"),
        ("abc", pa.string(), "float64"),
    ],
)
def test_a_cast_with_no_answer_fails_on_both_paths(value, arrow_type, dtype):
    """A fold must not quietly succeed where the engine raises, or the reverse."""
    ds = bt.from_arrow(pa.table({"c": pa.array([value], arrow_type)}))

    def outcome(expr):
        try:
            return ("ok", ds.select(r=expr).to_pydict()["r"][0])
        except Exception as exc:  # the exception *type* is the assertion here
            return ("raised", type(exc).__name__)

    assert outcome(bt.lit(value).cast(dtype)) == outcome(bt.col("c").cast(dtype))


@pytest.mark.parametrize("value", ["1", "-1", "3.7", " 5 ", "1e3"])
def test_folding_a_string_to_number_cast_matches_the_engine(duck, value):
    table = pa.table({"c": pa.array([value], pa.string())})
    ds = bt.from_arrow(table)
    assert (
        ds.select(r=bt.lit(value).cast("int64")).to_pydict()["r"]
        == ds.select(r=bt.col("c").cast("int64")).to_pydict()["r"]
    )
    duck.register("t", table)
    assert_same(
        ds.select(r=bt.lit(value).cast("int64")).collect(),
        duck.sql("SELECT CAST(c AS BIGINT) AS r FROM t"),
    )


@pytest.mark.parametrize("value", ["hello", "", "Straße", "ÄÖÜ", "MiXeD", "İstanbul"])
@pytest.mark.parametrize("method", ["to_uppercase", "to_lowercase", "len"])
def test_folding_a_case_or_length_function_matches_the_column_path(value, method):
    """Unicode case mapping is where two implementations most plausibly diverge."""
    ds = bt.from_arrow(pa.table({"c": pa.array([value], pa.string())}))
    on_literal = getattr(bt.lit(value).str, method)()
    on_column = getattr(bt.col("c").str, method)()
    assert ds.select(r=on_literal).to_pydict()["r"] == ds.select(r=on_column).to_pydict()["r"], (
        f"{method} folds differently than it runs for {value!r}"
    )


@pytest.mark.parametrize(("start", "length"), [(0, 3), (1, 3), (2, 10), (5, 2), (0, 0)])
def test_folding_a_substring_matches_the_column_path(start, length):
    """An off-by-one in the fold's index base would shift every folded substring."""
    values = ["hello", "", "Straße", "a b"]
    ds = bt.from_arrow(pa.table({"c": pa.array(values)}))
    on_column = ds.select(r=bt.col("c").str.slice(start, length)).to_pydict()["r"]
    on_literal = [
        ds.limit(1).select(r=bt.lit(v).str.slice(start, length)).to_pydict()["r"][0] for v in values
    ]
    assert on_literal == on_column
