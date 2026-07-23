"""The `extra.strings` rewrites must match DuckDB after the full optimizer runs.

Every rule in `kyber.rules.extra.strings` claims to preserve results exactly. These run each
rewritten shape end to end through the optimizer (via `.collect()`) and assert equality vs
DuckDB — over NULL rows, empty strings, empty input, and the wildcard/escape shapes the rules
must *not* touch (whose results must still match, because they were left alone).

The data deliberately contains a NULL and an empty string in every string column: those are
the two rows where a wrong NULL-vs-FALSE or empty-pattern decision shows up, and an
order-independent multiset comparison would still catch them (a dropped or kept row changes
the multiset).

Case conversion is exercised on ASCII only. The engine's `upper` is Rust's full-Unicode
`to_uppercase` (`'ß'` → `'SS'`) while DuckDB's is ICU's (`'ß'` → `'ẞ'`) — a *pre-existing*
divergence that has nothing to do with these rules, so the idempotence of `upper` on such
input is pinned against Batcher itself instead (`test_upper_idempotent_non_ascii_batcher_only`).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.strings  # importing runs the @rule decorators (registration)
from _harness import assert_same
from batcher import col
from batcher.plan.expr_ir import Binary, Col, Lit


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": ["abc", "abcd", "xabc", "a_c", "", None],
            "n": [1, 2, 3, 4, 5, 6],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table({"s": pa.array([], type=pa.string()), "n": pa.array([], type=pa.int64())})
    duck.register("t", tbl)
    return tbl


# --- like_without_wildcards_to_eq -------------------------------------------


def test_like_literal_filter(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("abc")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE 'abc'"))


def test_like_literal_project(duck, t):
    out = bt.from_arrow(t).select(r=col("s").str.like("abc")).collect()
    assert_same(out, duck.sql("SELECT s LIKE 'abc' AS r FROM t"))


def test_like_empty_literal(duck, t):
    # Matches only the empty string; NULL stays NULL.
    out = bt.from_arrow(t).select(r=col("s").str.like("")).collect()
    assert_same(out, duck.sql("SELECT s LIKE '' AS r FROM t"))


def test_like_underscore_wildcard_is_not_an_equality(duck, t):
    # The rule must NOT fire: 'a_c' matches 'abc' too. The result still has to match.
    out = bt.from_arrow(t).filter(col("s").str.like("a_c")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE 'a_c'"))


def test_like_backslash_pattern_untouched(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("a\\_c")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE 'a\\_c'"))


def test_like_literal_empty_input(duck, empty):
    out = bt.from_arrow(empty).filter(col("s").str.like("abc")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE 'abc'"))


# --- like → starts_with / ends_with / contains -------------------------------


def test_like_prefix(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("ab%")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE 'ab%'"))


def test_like_prefix_non_ascii(duck):
    # The prefix the range rule declines (non-incrementable trailing char) → starts_with.
    tbl = pa.table({"s": ["éclair", "eclair", "é", "", None]})
    duck.register("t2", tbl)
    out = bt.from_arrow(tbl).filter(col("s").str.like("é%")).collect()
    assert_same(out, duck.sql("SELECT * FROM t2 WHERE s LIKE 'é%'"))


def test_like_suffix(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("%abc")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE '%abc'"))


def test_like_suffix_project_keeps_nulls(duck, t):
    out = bt.from_arrow(t).select(r=col("s").str.like("%abc")).collect()
    assert_same(out, duck.sql("SELECT s LIKE '%abc' AS r FROM t"))


def test_like_contains(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("%bc%")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE '%bc%'"))


def test_like_contains_project_keeps_nulls(duck, t):
    out = bt.from_arrow(t).select(r=col("s").str.like("%bc%")).collect()
    assert_same(out, duck.sql("SELECT s LIKE '%bc%' AS r FROM t"))


# --- always-true predicates → IS NOT NULL (Filter conjuncts only) ------------


def test_like_only_wildcard(duck, t):
    # Keeps every non-null row (the empty string included), drops the NULL.
    out = bt.from_arrow(t).filter(col("s").str.like("%")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE '%'"))


def test_like_only_wildcard_conjunct(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("%") & (col("n") > 2)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE '%' AND n > 2"))


def test_not_like_only_wildcard_is_not_rewritten(duck, t):
    # NOT(s LIKE '%') is NULL for the null row → dropped; the rule must not turn it into
    # NOT(s IS NOT NULL), which would KEEP it. This is the test that catches that bug.
    out = bt.from_arrow(t).filter(~col("s").str.like("%")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE NOT (s LIKE '%')"))


def test_like_only_wildcard_under_or_is_not_rewritten(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("%") | (col("n") > 5)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE '%' OR n > 5"))


def test_like_only_wildcard_in_project_keeps_null(duck, t):
    # In a Project the value must stay NULL (not become FALSE) for the null row.
    out = bt.from_arrow(t).select(r=col("s").str.like("%")).collect()
    assert_same(out, duck.sql("SELECT s LIKE '%' AS r FROM t"))


def test_starts_with_empty_pattern(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.starts_with("")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE starts_with(s, '')"))


def test_contains_empty_pattern(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.contains("")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE contains(s, '')"))


def test_ends_with_empty_pattern(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.ends_with("")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE ends_with(s, '')"))


def test_contains_empty_pattern_in_project_keeps_null(duck, t):
    out = bt.from_arrow(t).select(r=col("s").str.contains("")).collect()
    assert_same(out, duck.sql("SELECT contains(s, '') AS r FROM t"))


# --- idempotence / identity collapse ----------------------------------------


def test_upper_of_upper(duck, t):
    out = bt.from_arrow(t).select(r=col("s").str.upper().str.upper()).collect()
    assert_same(out, duck.sql("SELECT upper(upper(s)) AS r FROM t"))


def test_lower_of_lower(duck, t):
    out = bt.from_arrow(t).select(r=col("s").str.lower().str.lower()).collect()
    assert_same(out, duck.sql("SELECT lower(lower(s)) AS r FROM t"))


def test_trim_of_trim(duck):
    tbl = pa.table({"s": ["  pad  ", "", "   ", None]})
    duck.register("t3", tbl)
    out = bt.from_arrow(tbl).select(r=col("s").str.trim().str.trim()).collect()
    assert_same(out, duck.sql("SELECT trim(trim(s)) AS r FROM t3"))


def test_trim_absorbs_ltrim(duck):
    tbl = pa.table({"s": ["  pad  ", "", "   ", None]})
    duck.register("t3", tbl)
    out = bt.from_arrow(tbl).select(r=col("s").str.lstrip().str.trim()).collect()
    assert_same(out, duck.sql("SELECT trim(ltrim(s)) AS r FROM t3"))


def test_trim_absorbs_rtrim_with_chars(duck):
    tbl = pa.table({"s": ["xxpadxx", "xx", "", None]})
    duck.register("t3", tbl)
    out = bt.from_arrow(tbl).select(r=col("s").str.rstrip("x").str.trim("x")).collect()
    assert_same(out, duck.sql("SELECT trim(rtrim(s, 'x'), 'x') AS r FROM t3"))


def test_replace_identity(duck, t):
    out = bt.from_arrow(t).select(r=col("s").str.replace("a", "a")).collect()
    assert_same(out, duck.sql("SELECT replace(s, 'a', 'a') AS r FROM t"))


def test_replace_identity_empty(duck, t):
    out = bt.from_arrow(t).select(r=col("s").str.replace("", "")).collect()
    assert_same(out, duck.sql("SELECT replace(s, '', '') AS r FROM t"))


# --- literal folding --------------------------------------------------------


def test_fold_case_of_literal(duck, t):
    out = bt.from_arrow(t).select(r=bt.lit("aBc").str.upper()).collect()
    assert_same(out, duck.sql("SELECT upper('aBc') AS r FROM t"))


def test_fold_len_of_literal(duck, t):
    out = bt.from_arrow(t).select(r=bt.lit("héllo").str.len()).collect()
    assert_same(out, duck.sql("SELECT length('héllo') AS r FROM t"))


def test_fold_octet_length_of_literal(duck, t):
    out = bt.from_arrow(t).select(r=bt.lit("héllo").str.octet_length()).collect()
    assert_same(out, duck.sql("SELECT strlen('héllo') AS r FROM t"))


def test_fold_concat_of_literals(duck, t):
    # The `concat` binary op is SQL `||` (null-propagating) — the shape the SQL front end
    # builds for `'a' || 'b'`, and the one this rule folds. (`bt.concat` is DuckDB's
    # null-*absorbing* `concat()` function and lowers through `coalesce`, not this shape.)
    out = bt.from_arrow(t).select(r=Binary("concat", Lit("a"), Lit("b"))).collect()
    assert_same(out, duck.sql("SELECT 'a' || 'b' AS r FROM t"))


def test_fold_concat_of_literals_under_a_column_concat(duck, t):
    # Only the literal pair folds; the column side still concatenates — and stays NULL for
    # the null row, which is what proves the fold did not disturb `||`'s null propagation.
    expr = Binary("concat", Col("s"), Binary("concat", Lit("a"), Lit("b")))
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(out, duck.sql("SELECT s || ('a' || 'b') AS r FROM t"))


def test_public_concat_function_is_unaffected(duck, t):
    # `bt.concat` treats NULL as absent (DuckDB `concat()`); no rule here may change that.
    out = bt.from_arrow(t).select(r=bt.concat(col("s"), bt.lit("a"), bt.lit("b"))).collect()
    assert_same(out, duck.sql("SELECT concat(s, 'a', 'b') AS r FROM t"))


def test_fold_substr_of_literal(duck, t):
    out = bt.from_arrow(t).select(r=bt.lit("abcdef").str.substr(2, 3)).collect()
    assert_same(out, duck.sql("SELECT substring('abcdef', 2, 3) AS r FROM t"))


def test_fold_substr_of_literal_zero_start(duck, t):
    out = bt.from_arrow(t).select(r=bt.lit("abcdef").str.substr(0, 3)).collect()
    assert_same(out, duck.sql("SELECT substring('abcdef', 0, 3) AS r FROM t"))


def test_fold_substr_of_literal_negative_start(duck, t):
    out = bt.from_arrow(t).select(r=bt.lit("abcdef").str.substr(-2, 4)).collect()
    assert_same(out, duck.sql("SELECT substring('abcdef', -2, 4) AS r FROM t"))


# --- Batcher-only oracle: Unicode case idempotence ---------------------------


def test_upper_idempotent_non_ascii_batcher_only():
    # `collapse_idempotent_str_func` rewrites upper(upper(s)) → upper(s); the claim is that
    # the engine's own `upper` is idempotent. DuckDB cannot be the oracle here (its ICU
    # `upper('ß')` is 'ẞ' where the engine's Rust `to_uppercase` is 'SS'), so the oracle is
    # the engine itself: the rewritten and unrewritten forms must agree.
    tbl = pa.table({"s": ["ß", "İ", "ﬁ", "straße", None]})
    ds = bt.from_arrow(tbl)
    once = ds.select(r=col("s").str.upper()).collect()
    twice = ds.select(r=col("s").str.upper().str.upper()).collect()
    assert once.column("r").to_pylist() == twice.column("r").to_pylist()
    lower_once = ds.select(r=col("s").str.lower()).collect()
    lower_twice = ds.select(r=col("s").str.lower().str.lower()).collect()
    assert lower_once.column("r").to_pylist() == lower_twice.column("r").to_pylist()
