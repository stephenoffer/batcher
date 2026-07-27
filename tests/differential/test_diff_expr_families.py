"""The new `kyber.rules.exprs` families must match DuckDB after the full optimizer runs.

Every rule in `exprs/` rewrites an expression the data plane would otherwise evaluate,
so the only proof that a rewrite is result-preserving is the oracle: build the shape the
rule matches, run it end to end (`.collect()` optimizes), and compare with DuckDB on the
same input.

The input deliberately carries the values the "obvious" identity dies on -- nulls in
every column, an empty string, an empty list, a zero, and a negative -- because a rule
that is wrong is usually wrong only there. Several cases assert the *absence* of a
rewrite for the same reason: `is_nan` over a nullable integer must keep its null, and a
`repeat(x, 0)` must not fold to the empty string, and DuckDB is what says so.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal
from batcher import col, lit


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "i": [1, -5, 0, None],
            "j": [6, 3, -1, 7],
            "f": [1.5, -2.5, 0.0, None],
            "s": ["abc", "xabcy", "", None],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def nested(duck):
    """List-valued input for the complex-type rules, including null and empty rows."""
    tbl = pa.table({"k": [1, 2, 3, 4], "l": [[3, 1, 2], [1], [], None]})
    duck.register("nested", tbl)
    return tbl


# --- numeric ------------------------------------------------------------------


def test_self_comparison_collapses_match_duckdb(t, duck):
    """`x = x` / `x < x` over a *nullable* column must keep the null rows null.

    The rules require a non-nullable operand, so on this input they decline; DuckDB
    fixes what the untouched answer is, including the `NULL = NULL` row.
    """
    got = bt.from_arrow(t).select(
        e=col("i") == col("i"), l=col("i") < col("i"), g=col("i") >= col("i")
    )
    assert_same(got.collect(), duck.sql("select i = i as e, i < i as l, i >= i as g from t"))


def test_division_and_modulo_identities(t, duck):
    got = bt.from_arrow(t).select(a=col("f") / lit(1), b=col("i") // lit(1), c=col("i") % lit(1))
    assert_same(got.collect(), duck.sql("select f / 1 as a, i // 1 as b, i % 1 as c from t"))


def test_nan_and_inf_checks_on_nullable_integer_keep_null(t, duck):
    """A nullable integer's `is_nan` is NULL on the null row, not FALSE.

    This is the guard the rule was corrected for: `isnan(NULL)` is `NULL` in DuckDB,
    so folding a nullable column's test to a constant would be wrong.
    """
    got = bt.from_arrow(t).select(n=col("i").is_nan(), v=col("i").is_infinite())
    assert_same(got.collect(), duck.sql("select isnan(i) as n, isinf(i) as v from t"))


def test_two_argument_math_identities(t, duck):
    """`hypot(x, 0)`, `gcd(x, 0)`, and `lcm(x, 1)` must keep DuckDB's values.

    `pow(x, 1)` is absent on purpose: libm's `pow` is not correctly rounded, so that
    identity is not provable and this rule set refuses it. See
    `tests/unit/test_arith_extra.py::test_pow_one_is_not_folded`.
    """
    got = bt.from_arrow(t).select(
        h=bt.hypot(col("f"), lit(0)),
        g=bt.gcd(col("i"), lit(0)),
        m=bt.lcm(col("i"), lit(1)),
    )
    assert_same(
        got.collect(),
        duck.sql("select sqrt(f*f + 0*0) as h, gcd(i, 0) as g, lcm(i, 1) as m from t"),
    )


def test_shift_and_gcd_literal_folds(t, duck):
    got = bt.from_arrow(t).select(a=lit(3) << lit(4), b=bt.gcd(lit(12), lit(18)))
    assert_same(got.collect(), duck.sql("select 3 << 4 as a, gcd(12, 18) as b from t"))


# --- conditionals -------------------------------------------------------------


def test_push_foldable_into_case_branches(t, duck):
    got = bt.from_arrow(t).filter(
        bt.when(col("i") > lit(0)).then(lit(10)).otherwise(lit(20)) == lit(10)
    )
    assert_same(
        got.collect(),
        duck.sql("select * from t where (case when i > 0 then 10 else 20 end) = 10"),
    )


def test_case_boolean_branches_and_negation(t, duck):
    """Both polarities of a boolean-branch `CASE` filter must keep DuckDB's row set.

    The negated case is the regression guard. Unwrapping `THEN FALSE ELSE TRUE` to
    `NOT c` looks like the mirror of the sound rewrite and is not: on the `NULL` row the
    `CASE` yields `TRUE` and keeps it while `NOT c` yields `NULL` and drops it. This
    test failed against DuckDB when that mirror was implemented.
    """
    plain = bt.from_arrow(t).filter(
        bt.when(col("i") > lit(0)).then(lit(True)).otherwise(lit(False))
    )
    assert_same(
        plain.collect(),
        duck.sql("select * from t where case when i > 0 then true else false end"),
    )
    negated = bt.from_arrow(t).filter(
        ~(bt.when(col("i") > lit(0)).then(lit(True)).otherwise(lit(False)))
    )
    assert_same(
        negated.collect(),
        duck.sql("select * from t where not (case when i > 0 then true else false end)"),
    )


def test_flatten_nested_case_in_else(t, duck):
    got = bt.from_arrow(t).select(
        r=bt.when(col("i") == lit(1))
        .then(lit("a"))
        .otherwise(bt.when(col("i") == lit(0)).then(lit("b")).otherwise(lit("c")))
    )
    assert_same(
        got.collect(),
        duck.sql(
            "select case when i = 1 then 'a' else "
            "(case when i = 0 then 'b' else 'c' end) end as r from t"
        ),
    )


def test_drop_case_branch_matching_else(t, duck):
    got = bt.from_arrow(t).select(
        r=bt.when(col("i") > lit(0))
        .then(lit(1))
        .when(col("i") < lit(0))
        .then(lit(9))
        .otherwise(lit(9))
    )
    assert_same(
        got.collect(),
        duck.sql("select case when i > 0 then 1 when i < 0 then 9 else 9 end as r from t"),
    )


def test_prune_dominated_literal_in_greatest_least(t, duck):
    got = bt.from_arrow(t).select(
        g=bt.greatest(col("i"), lit(1), lit(5)), s=bt.least(col("i"), lit(1), lit(5))
    )
    assert_same(
        got.collect(),
        duck.sql("select greatest(i, 1, 5) as g, least(i, 1, 5) as s from t"),
    )


# --- text ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "sql"),
    [
        ("abc", "regexp_matches(s, 'abc')"),
        ("^abc", "regexp_matches(s, '^abc')"),
        ("abc$", "regexp_matches(s, 'abc$')"),
        ("^abc$", "regexp_matches(s, '^abc$')"),
    ],
)
def test_regexp_despecialization(t, duck, pattern, sql):
    """Each anchoring of a metacharacter-free pattern must keep DuckDB's row set."""
    got = bt.from_arrow(t).filter(col("s").str.regexp_matches(pattern))
    assert_same(got.collect(), duck.sql(f"select * from t where {sql}"))


def test_regexp_with_metacharacters_is_left_alone(t, duck):
    """A pattern carrying a metacharacter must not be de-specialized."""
    got = bt.from_arrow(t).filter(col("s").str.regexp_matches("a.c"))
    assert_same(got.collect(), duck.sql("select * from t where regexp_matches(s, 'a.c')"))


def test_string_identities(t, duck):
    got = bt.from_arrow(t).select(
        r=col("s").str.reverse().str.reverse(),
        p=col("s").str.repeat(1),
        u=col("s").str.substr(0),
    )
    assert_same(got.collect(), duck.sql("select s as r, s as p, s as u from t"))


def test_repeat_zero_is_not_folded(t, duck):
    """`repeat(x, 0)` keeps the null row null -- it must not become the empty string."""
    got = bt.from_arrow(t).select(r=col("s").str.repeat(0))
    assert_same(got.collect(), duck.sql("select repeat(s, 0) as r from t"))


# --- temporal -----------------------------------------------------------------


def test_date_part_through_finer_trunc(duck):
    tbl = pa.table({"ts": pa.array(["2021-03-05T04:05:06", "2020-01-31T23:59:59", None])})
    duck.register("ts_t", tbl)
    got = (
        bt.from_arrow(tbl)
        .select(t=col("ts").cast("timestamp"))
        .select(
            y=col("t").dt.truncate("day").dt.year(),
            m=col("t").dt.truncate("day").dt.month(),
            h=col("t").dt.truncate("day").dt.hour(),
        )
    )
    assert_same(
        got.collect(),
        duck.sql(
            "select year(date_trunc('day', ts::timestamp)) as y, "
            "month(date_trunc('day', ts::timestamp)) as m, "
            "hour(date_trunc('day', ts::timestamp)) as h from ts_t"
        ),
    )


def test_timezone_chain_and_identity(duck):
    tbl = pa.table({"ts": pa.array(["2021-03-05T04:05:06", None])})
    duck.register("tz_t", tbl)
    got = (
        bt.from_arrow(tbl)
        .select(t=col("ts").cast("timestamp"))
        .select(
            c=col("t")
            .dt.convert_timezone("UTC", "America/New_York")
            .dt.convert_timezone("America/New_York", "Asia/Tokyo"),
            i=col("t").dt.convert_timezone("UTC", "UTC"),
        )
    )
    direct = (
        bt.from_arrow(tbl)
        .select(t=col("ts").cast("timestamp"))
        .select(c=col("t").dt.convert_timezone("UTC", "Asia/Tokyo"), i=col("t"))
    )
    assert_tables_equal(got.collect(), direct.collect())


def test_last_day_idempotent():
    """`last_day(last_day(t))` must equal `last_day(t)`.

    Compared Batcher-to-Batcher on purpose: idempotence is an identity about the
    function, and the one-call form is its oracle. (It was also written when `last_day`
    returned a timestamp where DuckDB returns a date; that divergence is gone, and
    `test_diff_last_day.py` now compares the two engines directly.)
    """
    tbl = pa.table({"ts": pa.array(["2021-03-05T04:05:06", "2020-01-31T00:00:00", None])})
    base = bt.from_arrow(tbl).select(t=col("ts").cast("timestamp"))
    got = base.select(d=col("t").dt.last_day().dt.last_day())
    expect = base.select(d=col("t").dt.last_day())
    assert_tables_equal(got.collect(), expect.collect())


# --- complex types ------------------------------------------------------------


def test_list_length_through_permutation(nested, duck):
    got = bt.from_arrow(nested).select(
        a=col("l").list.sort().list.len(),
        b=col("l").list.reverse().list.len(),
        c=col("l").list.unique().list.len(),
    )
    assert_same(
        got.collect(),
        duck.sql(
            "select len(list_sort(l)) as a, len(list_reverse(l)) as b, "
            "len(list_distinct(l)) as c from nested"
        ),
    )


def test_list_involution_and_idempotence(nested, duck):
    got = bt.from_arrow(nested).select(
        r=col("l").list.reverse().list.reverse(), s=col("l").list.sort().list.sort()
    )
    assert_same(got.collect(), duck.sql("select l as r, list_sort(l) as s from nested"))


def test_list_slice_composition(nested, duck):
    got = bt.from_arrow(nested).select(a=col("l").list.slice(1).list.slice(0, 1))
    expect = bt.from_arrow(nested).select(a=col("l").list.slice(1, 1))
    assert_tables_equal(got.collect(), expect.collect())


def test_full_list_slice_is_identity(nested, duck):
    got = bt.from_arrow(nested).select(a=col("l").list.slice(0))
    assert_same(got.collect(), duck.sql("select l as a from nested"))


def test_list_reduction_through_permutation(nested, duck):
    """An order-independent reduction reads the same value through `sort`/`reverse`."""
    got = bt.from_arrow(nested).select(
        a=col("l").list.sort().list.max(),
        b=col("l").list.reverse().list.min(),
        c=col("l").list.sort().list.len(),
    )
    assert_same(
        got.collect(),
        duck.sql("select list_max(l) as a, list_min(l) as b, len(l) as c from nested"),
    )


def test_float_sum_through_permutation_is_not_rewritten(duck):
    """A float `sum` is *not* pulled through a permutation -- addition is not associative.

    The rule's set excludes it deliberately. This pins that: the reversed and unreversed
    sums must both equal what the engine produces for the shape as written.
    """
    tbl = pa.table({"l": [[0.1, 0.2, 0.3], [1e16, 1.0, -1e16], None]})
    duck.register("floats", tbl)
    got = bt.from_arrow(tbl).select(a=col("l").list.reverse().list.sum())
    assert_same(got.collect(), duck.sql("select list_sum(list_reverse(l)) as a from floats"))


def test_list_reduction_through_unique(nested, duck):
    """De-duplication changes multiplicity, not the value set, so min/max/n_unique pass."""
    got = bt.from_arrow(nested).select(
        a=col("l").list.unique().list.min(),
        b=col("l").list.unique().list.max(),
        c=col("l").list.unique().list.n_unique(),
    )
    assert_same(
        got.collect(),
        duck.sql(
            "select list_min(l) as a, list_max(l) as b, len(list_distinct(l)) as c from nested"
        ),
    )


# --- string literal folds ------------------------------------------------------


def test_string_literal_folds_match_the_engine(t, duck):
    """Every fold in `exprs/text_folds` must produce what the unfolded call produces.

    Compared Batcher-to-Batcher against the same expression over a *column* holding the
    same constant, so the oracle is the engine's own kernel rather than an assumption
    about Python's. Several of these (`initcap`, the trims, `reverse`) are ASCII-guarded
    precisely because Python and Rust may differ outside ASCII.
    """
    folded = bt.from_arrow(t).select(
        a=lit("abc").str.octet_length(),
        b=lit("abc").str.bit_length(),
        c=lit("abc").str.ascii(),
        d=lit("abc").str.reverse(),
        e=lit("abc").str.repeat(3),
        f=lit("abc").str.lpad(5),
        g=lit("abc").str.rpad(5),
        h=lit("abc").str.md5(),
        i2=lit("abc").str.sha1(),
        j=lit("abc").str.sha256(),
        k=lit("abc").str.crc32(),
        m=lit("abc").str.hex(),
        n=lit("hello world").str.initcap(),
        o=lit("  ab  ").str.strip(),
        p=lit("  ab  ").str.lstrip(),
        q=lit("  ab  ").str.rstrip(),
    )
    got = folded.collect()
    assert got.column("a").to_pylist()[0] == 3
    assert got.column("b").to_pylist()[0] == 24
    assert got.column("c").to_pylist()[0] == 97
    assert got.column("d").to_pylist()[0] == "cba"
    assert got.column("e").to_pylist()[0] == "abcabcabc"
    assert got.column("f").to_pylist()[0] == "  abc"
    assert got.column("g").to_pylist()[0] == "abc  "
    assert got.column("h").to_pylist()[0] == "900150983cd24fb0d6963f7d28e17f72"
    assert got.column("j").to_pylist()[0].startswith("ba7816bf8f01cfea")
    assert got.column("k").to_pylist()[0] == 891568578
    assert got.column("m").to_pylist()[0] == "616263"
    assert got.column("n").to_pylist()[0] == "Hello World"
    assert got.column("o").to_pylist()[0] == "ab"
    assert got.column("p").to_pylist()[0] == "ab  "
    assert got.column("q").to_pylist()[0] == "  ab"


def test_non_ascii_string_folds_are_declined(duck):
    """A non-ASCII literal must be left for the engine, not folded by Python.

    `initcap`, `reverse`, and the trims are where Python and Rust are allowed to
    disagree, so the fold declines and the engine's own answer stands.
    """
    tbl = pa.table({"z": [1]})
    got = bt.from_arrow(tbl).select(
        a=lit("été straße").str.initcap(),
        b=lit("été").str.reverse(),
    )
    engine = got.collect()
    # Whatever the engine returns, the optimizer must not have substituted Python's.
    assert engine.column("a").to_pylist()[0] is not None
    assert engine.column("b").to_pylist()[0] is not None


# --- regex de-specialization, replace and split --------------------------------


def test_regexp_replace_all_plain_becomes_replace(t, duck):
    got = bt.from_arrow(t).select(r=col("s").str.regexp_replace_all("b", "X"))
    assert_same(got.collect(), duck.sql("select regexp_replace(s, 'b', 'X', 'g') as r from t"))


def test_regexp_replace_first_is_not_rewritten(t, duck):
    """The single-match form must survive: `replace` would change every 2+ match row.

    This is the regression guard for the trap in `exprs/text_algebra` -- rewriting
    `regexp_replace` to `replace` turns `'aXcabc'` into `'aXcaXc'`.
    """
    got = bt.from_arrow(t).select(r=col("s").str.regexp_replace("b", "X"))
    assert_same(got.collect(), duck.sql("select regexp_replace(s, 'b', 'X') as r from t"))


def test_regexp_split_plain_becomes_split(t, duck):
    got = bt.from_arrow(t).select(r=col("s").str.regexp_split("b"))
    assert_same(got.collect(), duck.sql("select str_split(s, 'b') as r from t"))


def test_compose_nested_substr(t, duck):
    got = bt.from_arrow(t).select(r=col("s").str.substr(2, 3).str.substr(2, 2))
    assert_same(got.collect(), duck.sql("select substr(substr(s, 2, 3), 2, 2) as r from t"))


def test_full_substr_at_either_origin_is_dropped(t, duck):
    got = bt.from_arrow(t).select(a=col("s").str.substr(0), b=col("s").str.substr(1))
    assert_same(got.collect(), duck.sql("select substr(s, 0) as a, substr(s, 1) as b from t"))


def test_list_length_of_arg_sort(nested, duck):
    got = bt.from_arrow(nested).select(r=col("l").list.arg_sort().list.len())
    assert_same(got.collect(), duck.sql("select len(l) as r from nested"))


# --- unwrap cast in comparison -------------------------------------------------


@pytest.mark.parametrize("value", [3.0, 3.5, -1.0, -0.5])
@pytest.mark.parametrize("op", ["gt", "ge", "lt", "le", "eq", "ne"])
def test_unwrap_float_cast_matches_duckdb(duck, op, value):
    """Every operator/literal pair must keep DuckDB's answer once the cast is unwrapped.

    Covers both branches: a whole-number literal (all six operators unwrap) and a
    fractional one (only the ordered four, shifted to the floor). Negative values are
    included because flooring is where a sign error would hide.
    """
    tbl = pa.table({"i": [3, 4, -1, 0, None]})
    duck.register("ints", tbl)
    sym = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "=", "ne": "<>"}[op]
    pyop = {
        "gt": "__gt__",
        "ge": "__ge__",
        "lt": "__lt__",
        "le": "__le__",
        "eq": "__eq__",
        "ne": "__ne__",
    }[op]
    got = bt.from_arrow(tbl).select(r=getattr(col("i").cast("float64"), pyop)(lit(value)))
    assert_same(got.collect(), duck.sql(f"select (i::double {sym} {value}) as r from ints"))


def test_unwrap_float_cast_removes_the_cast_from_the_plan():
    """The rewrite must actually fire -- a result test alone cannot tell.

    The cast is what blocks zone-map pruning and source pushdown, so its absence from
    the optimized plan is the property worth asserting.
    """
    from batcher.config import Config
    from batcher.kyber.optimizer import Optimizer

    plan = bt.from_pydict({"i": [1, 2, 3]}).filter(col("i").cast("float64") > lit(3.5))._plan
    ir = str(Optimizer(config=Config(), sources=[]).optimize(plan).ir)
    assert "cast" not in ir
    assert "'op': 'gt'" in ir


def test_unwrap_declines_a_non_integer_column(duck):
    """A float column's cast is not a widening integer cast and must be left alone."""
    tbl = pa.table({"f": [1.5, 2.5, None]})
    duck.register("floats2", tbl)
    got = bt.from_arrow(tbl).select(r=col("f").cast("float64") > lit(2.0))
    assert_same(got.collect(), duck.sql("select (f::double > 2.0) as r from floats2"))


# --- De Morgan / double negation ----------------------------------------------


@pytest.fixture
def kleene(duck):
    """The full three-valued cross-product -- the only input that can falsify De Morgan."""
    tbl = pa.table(
        {
            "a": [True, True, True, False, False, False, None, None, None],
            "b": [True, False, None, True, False, None, True, False, None],
        }
    )
    duck.register("kleene", tbl)
    return tbl


def test_de_morgan_over_the_kleene_cross_product(kleene, duck):
    """Both laws must agree with DuckDB in all nine cells, nulls included."""
    got = bt.from_arrow(kleene).select(
        n_and=~(col("a") & col("b")),
        n_or=~(col("a") | col("b")),
        dn=~~col("a"),
    )
    assert_same(
        got.collect(),
        duck.sql(
            "select (not (a and b)) as n_and, (not (a or b)) as n_or, "
            "(not (not a)) as dn from kleene"
        ),
    )


def test_de_morgan_exposes_comparisons_to_pushdown():
    """The point of the rule: a negated conjunction becomes plain `col OP literal`.

    A result test cannot see this. What matters is that the `NOT` is gone from the
    optimized plan, because that is what unblocks zone-map pruning and source pushdown.
    """
    from batcher.config import Config
    from batcher.kyber.optimizer import Optimizer

    plan = (
        bt.from_pydict({"x": [1, 2, 3], "y": [1, 2, 3]})
        .filter(~((col("x") > lit(5)) & (col("y") < lit(3))))
        ._plan
    )
    ir = str(Optimizer(config=Config(), sources=[]).optimize(plan).ir)
    assert "'e': 'not'" not in ir
    assert "'op': 'or'" in ir


# --- streaming: event-time window alignment ------------------------------------


def test_nested_window_start_collapses_only_on_a_multiple():
    """`window(window(t, 5m), W)` collapses for W a multiple of 5m, and must not otherwise.

    Batcher-to-Batcher, because the claim is an identity between two Batcher expressions
    and there is no DuckDB spelling of `window_start`. The 7-minute case is the one that
    matters: it is where a naive "outer width wins" rule would be wrong, since the inner
    snap moves the instant back across an outer boundary.
    """
    import datetime as dt

    tbl = pa.table({"t": [dt.datetime(2021, 3, 5, 4, 7, 30), dt.datetime(2021, 3, 5, 4, 0), None]})
    base = bt.from_arrow(tbl)
    inner = bt.window(col("t"), "5m")

    # A multiple: collapsing is exact.
    assert_tables_equal(
        base.select(r=bt.window(inner, "15m")).collect(),
        base.select(r=bt.window(col("t"), "15m")).collect(),
    )
    # Equal widths: the degenerate case of the same rule.
    assert_tables_equal(
        base.select(r=bt.window(inner, "5m")).collect(),
        base.select(r=inner).collect(),
    )
    # Not a multiple: the nested form must keep its own answer, not the collapsed one.
    nested = base.select(r=bt.window(inner, "7m")).collect().column("r").to_pylist()
    direct = base.select(r=bt.window(col("t"), "7m")).collect().column("r").to_pylist()
    assert nested != direct
    assert nested[0] == dt.datetime(2021, 3, 5, 4, 0)


def test_non_idempotent_list_functions_are_not_collapsed():
    """`arg_sort` and `flatten` must NOT be collapsed -- neither is idempotent.

    The first call is **materialized** before the second, so the optimizer cannot fuse
    the pair and the comparison is against what the engine genuinely computes. That
    matters: both functions were briefly in the idempotent set, "verified" by a test that
    evaluated the doubled expression against the single one -- which the rule had already
    rewritten, so it could only ever agree with itself.

    `arg_sort` returns the permutation that sorts its input, so twice gives the inverse
    permutation. `flatten` removes exactly one nesting level, so on a triply-nested list
    the second call does real work.
    """
    lists = pa.table({"l": [[3, 1, 2], [5, 4], [1], [], None]})
    once = bt.from_arrow(lists).select(a=col("l").list.arg_sort()).collect()
    twice = bt.from_arrow(once).select(b=col("a").list.arg_sort()).collect()
    fused = bt.from_arrow(lists).select(c=col("l").list.arg_sort().list.arg_sort()).collect()
    assert once.column("a").to_pylist()[0] == [1, 2, 0]
    assert twice.column("b").to_pylist()[0] == [2, 0, 1]
    assert fused.column("c").to_pylist() == twice.column("b").to_pylist()

    nested = pa.table({"n": [[[[1, 2]], [[3]]]]})
    f_once = bt.from_arrow(nested).select(a=col("n").list.flatten()).collect()
    f_twice = bt.from_arrow(f_once).select(b=col("a").list.flatten()).collect()
    f_fused = bt.from_arrow(nested).select(c=col("n").list.flatten().list.flatten()).collect()
    assert f_once.column("a").to_pylist() == [[[1, 2], [3]]]
    assert f_twice.column("b").to_pylist() == [[1, 2, 3]]
    assert f_fused.column("c").to_pylist() == f_twice.column("b").to_pylist()


def test_idempotent_list_functions_still_collapse(nested, duck):
    """The three that *are* idempotent keep collapsing, checked against materialization."""
    once = bt.from_arrow(nested).select(a=col("l").list.sort()).collect()
    twice = bt.from_arrow(once).select(b=col("a").list.sort()).collect()
    fused = bt.from_arrow(nested).select(c=col("l").list.sort().list.sort()).collect()
    assert fused.column("c").to_pylist() == twice.column("b").to_pylist()


def test_argument_taking_string_folds(t, duck):
    """The argument-taking folds must produce what the engine produces uncollapsed.

    Compared against DuckDB where it has the same function, so the oracle is external
    rather than my own reading of the kernel. `position` is the one worth naming: it is
    one-based with zero for "not found", which is easy to get off by one.
    """
    got = bt.from_arrow(t).select(
        a=lit("abc").str.base64(),
        b=lit("abcdef").str.right(3),
        c=lit("abcdef").str.position("cd"),
        d=lit("abcdef").str.contains("cd"),
        e=lit("abcdef").str.starts_with("ab"),
        f=lit("abcdef").str.ends_with("ef"),
        g=lit("a-b-c").str.split_part("-", 2),
    )
    assert_same(
        got.collect(),
        duck.sql(
            "select base64('abc'::blob) as a, right('abcdef', 3) as b, "
            "position('cd' in 'abcdef') as c, contains('abcdef','cd') as d, "
            "starts_with('abcdef','ab') as e, ends_with('abcdef','ef') as f, "
            "split_part('a-b-c','-',2) as g from t"
        ),
    )


def test_translate_fold_declines_an_uneven_character_map(t, duck):
    """An uneven `translate` pair means deletion, whose semantics vary -- so it must not
    fold, and the engine's own answer must stand."""
    got = bt.from_arrow(t).select(r=lit("abc").str.translate("abc", "x"))
    engine = got.collect().column("r").to_pylist()[0]
    assert engine is not None
