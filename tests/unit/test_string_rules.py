"""Plan-shape, idempotence, and does-NOT-fire tests for the `extra.strings` rule family.

Each rule must fire into the intended shape, be idempotent (a second application is a
no-op — the fixpoint driver requires it), and stay well clear of the three unsound shapes
the family is built to avoid:

* a `_` (single-character) wildcard or a backslash in a LIKE pattern — neither is a literal;
* a NULL-vs-FALSE swap anywhere but a top-level `Filter` conjunct (`x LIKE '%'` is NULL on a
  null input, `x IS NOT NULL` is FALSE) — so never in a `Project`, and never under a `NOT`;
* dropping the last string function around a value whose Utf8-ness isn't provable.

Checked without the native engine; `tests/differential/test_diff_string_rules.py` proves the
same rewrites against DuckDB end to end.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import strings as st
from batcher.plan.expr_ir import Binary, Col, IsNotNull, Lit, StrFunc
from batcher.plan.logical import Filter

_RULE_NAMES = [
    "collapse_idempotent_str_func",
    "empty_pattern_match_to_not_null",
    "fold_case_of_literal",
    "fold_concat_of_literals",
    "fold_len_of_literal",
    "fold_substr_of_literal",
    "like_contains_to_contains",
    "like_only_wildcard_to_not_null",
    "like_prefix_to_starts_with",
    "like_suffix_to_ends_with",
    "like_without_wildcards_to_eq",
    "replace_identity_to_input",
    "trim_absorbs_inner_side_trim",
]


def _ds():
    """A dataset with a Utf8 column `s`, a second Utf8 `t`, and an Int64 `n`."""
    return bt.from_pydict({"s": ["abc", None], "t": ["x", "y"], "n": [1, 2]})


def _binary_ds():
    """A dataset whose `b` column is Arrow `Binary` — the shape a string function coerces to
    Utf8, and which therefore must not have its last string function stripped."""
    tbl = pa.table({"b": pa.array([b"abc", None], type=pa.binary())})
    return bt.from_arrow(tbl)


def _proj(expr):
    return _ds().select(r=expr)._plan


def _flt(pred):
    return _ds().filter(pred)._plan


def _expr_ir(node):
    return node.items[0].expr.to_ir()


# --- registration -----------------------------------------------------------


def test_all_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    for n in _RULE_NAMES:
        assert n in names, f"{n} not registered"


def test_rule_count():
    assert len(_RULE_NAMES) == len(st.__all__) == 13


# --- like_without_wildcards_to_eq -------------------------------------------


def test_like_literal_becomes_eq():
    out = st.like_without_wildcards_to_eq(_flt(col("s").str.like("abc")), None)
    assert out.predicate.to_ir() == Binary("eq", Col("s"), Lit("abc")).to_ir()


def test_like_literal_becomes_eq_in_project():
    # NULL-preserving on both sides, so it is sound outside a Filter too.
    out = st.like_without_wildcards_to_eq(_proj(col("s").str.like("abc")), None)
    assert _expr_ir(out) == Binary("eq", Col("s"), Lit("abc")).to_ir()


def test_like_empty_pattern_becomes_eq_empty():
    out = st.like_without_wildcards_to_eq(_flt(col("s").str.like("")), None)
    assert out.predicate.to_ir() == Binary("eq", Col("s"), Lit("")).to_ir()


def test_like_underscore_wildcard_does_not_fire():
    # `_` matches exactly one character: 'a_c' is NOT the literal 'a_c'.
    assert st.like_without_wildcards_to_eq(_flt(col("s").str.like("a_c")), None) is None


def test_like_percent_does_not_fire():
    assert st.like_without_wildcards_to_eq(_flt(col("s").str.like("ab%")), None) is None


def test_like_backslash_does_not_fire():
    # An escape character would change the pattern's meaning; refuse it outright.
    assert st.like_without_wildcards_to_eq(_flt(col("s").str.like("a\\%c")), None) is None


def test_ilike_does_not_fire():
    # Case-insensitive matching has no `eq` twin.
    assert st.like_without_wildcards_to_eq(_flt(col("s").str.ilike("abc")), None) is None


def test_like_to_eq_does_not_fire_on_binary_column():
    # `b` is Arrow Binary: the LIKE coerces it to Utf8, so dropping the LIKE would change
    # the comparison's typing. The guard must refuse.
    plan = _binary_ds().filter(col("b").str.like("abc"))._plan
    assert st.like_without_wildcards_to_eq(plan, None) is None


def test_like_to_eq_idempotent():
    once = st.like_without_wildcards_to_eq(_flt(col("s").str.like("abc")), None)
    assert st.like_without_wildcards_to_eq(once, None) is None


# --- like_prefix_to_starts_with (only where the range rule declines) ---------


def test_like_prefix_ascii_left_to_the_range_rule():
    # 'ab%' has an incrementable prefix → `like_prefix_to_range` owns it.
    assert st.like_prefix_to_starts_with(_flt(col("s").str.like("ab%")), None) is None


def test_like_prefix_non_ascii_becomes_starts_with():
    # 'é%' cannot be incremented into an exact range, so the range rule declines and this
    # rule takes it.
    out = st.like_prefix_to_starts_with(_flt(col("s").str.like("é%")), None)
    assert out.predicate.to_ir() == StrFunc("starts_with", Col("s"), pattern="é").to_ir()


def test_like_prefix_with_inner_wildcard_does_not_fire():
    assert st.like_prefix_to_starts_with(_flt(col("s").str.like("é_%")), None) is None


def test_like_prefix_idempotent():
    once = st.like_prefix_to_starts_with(_flt(col("s").str.like("é%")), None)
    assert st.like_prefix_to_starts_with(once, None) is None


# --- like_suffix_to_ends_with -----------------------------------------------


def test_like_suffix_becomes_ends_with():
    out = st.like_suffix_to_ends_with(_flt(col("s").str.like("%bc")), None)
    assert out.predicate.to_ir() == StrFunc("ends_with", Col("s"), pattern="bc").to_ir()


def test_like_suffix_in_project():
    out = st.like_suffix_to_ends_with(_proj(col("s").str.like("%bc")), None)
    assert _expr_ir(out) == StrFunc("ends_with", Col("s"), pattern="bc").to_ir()


def test_like_suffix_with_underscore_does_not_fire():
    assert st.like_suffix_to_ends_with(_flt(col("s").str.like("%b_c")), None) is None


def test_like_suffix_only_percent_does_not_fire():
    # '%' has no suffix — that is the every-non-null-string case, not an ends_with.
    assert st.like_suffix_to_ends_with(_flt(col("s").str.like("%")), None) is None


def test_like_suffix_idempotent():
    once = st.like_suffix_to_ends_with(_flt(col("s").str.like("%bc")), None)
    assert st.like_suffix_to_ends_with(once, None) is None


# --- like_contains_to_contains ----------------------------------------------


def test_like_infix_becomes_contains():
    out = st.like_contains_to_contains(_flt(col("s").str.like("%b%")), None)
    assert out.predicate.to_ir() == StrFunc("contains", Col("s"), pattern="b").to_ir()


def test_like_infix_with_underscore_does_not_fire():
    assert st.like_contains_to_contains(_flt(col("s").str.like("%a_b%")), None) is None


def test_like_double_percent_does_not_fire():
    # '%%' has an empty middle — the every-non-null-string case, owned by the NOT NULL rule.
    assert st.like_contains_to_contains(_flt(col("s").str.like("%%")), None) is None


def test_like_contains_idempotent():
    once = st.like_contains_to_contains(_flt(col("s").str.like("%b%")), None)
    assert st.like_contains_to_contains(once, None) is None


# --- like_only_wildcard_to_not_null (Filter conjuncts only) -----------------


def test_like_all_wildcard_becomes_not_null():
    out = st.like_only_wildcard_to_not_null(_flt(col("s").str.like("%")), None)
    assert out.predicate.to_ir() == IsNotNull(Col("s")).to_ir()


def test_like_repeated_wildcard_becomes_not_null():
    out = st.like_only_wildcard_to_not_null(_flt(col("s").str.like("%%")), None)
    assert out.predicate.to_ir() == IsNotNull(Col("s")).to_ir()


def test_like_all_wildcard_fires_on_a_top_level_conjunct():
    pred = (col("n") > 1) & col("s").str.like("%")
    out = st.like_only_wildcard_to_not_null(_flt(pred), None)
    assert out.predicate.to_ir() == ((col("n") > 1) & IsNotNull(Col("s"))).to_ir()


def test_like_all_wildcard_does_not_fire_under_not():
    # NOT(x LIKE '%') is NULL for a null x (row dropped); NOT(x IS NOT NULL) is TRUE (kept).
    assert st.like_only_wildcard_to_not_null(_flt(~col("s").str.like("%")), None) is None


def test_like_all_wildcard_does_not_fire_under_or():
    pred = (col("n") > 1) | col("s").str.like("%")
    assert st.like_only_wildcard_to_not_null(_flt(pred), None) is None


def test_null_swapping_rules_match_only_filter():
    # In a Project, NULL and FALSE are different *values*, so these rules must never be
    # offered a Project at all — the declared match set is what the driver keys off.
    for name in ("like_only_wildcard_to_not_null", "empty_pattern_match_to_not_null"):
        r = next(x for x in DEFAULT_REGISTRY.rules() if x.name == name)
        assert r.matches == frozenset({Filter})
    # And the whole-plan wrapper leaves a Project carrying the same shape untouched.
    plan = _proj(col("s").str.like("%"))
    r = next(x for x in DEFAULT_REGISTRY.rules() if x.name == "like_only_wildcard_to_not_null")
    assert r.apply(plan, None) is plan


def test_like_all_wildcard_idempotent():
    once = st.like_only_wildcard_to_not_null(_flt(col("s").str.like("%")), None)
    assert st.like_only_wildcard_to_not_null(once, None) is None


def test_like_all_wildcard_does_not_fire_on_binary_column():
    plan = _binary_ds().filter(col("b").str.like("%"))._plan
    assert st.like_only_wildcard_to_not_null(plan, None) is None


# --- empty_pattern_match_to_not_null ----------------------------------------


def test_starts_with_empty_becomes_not_null():
    out = st.empty_pattern_match_to_not_null(_flt(col("s").str.starts_with("")), None)
    assert out.predicate.to_ir() == IsNotNull(Col("s")).to_ir()


def test_ends_with_empty_becomes_not_null():
    out = st.empty_pattern_match_to_not_null(_flt(col("s").str.ends_with("")), None)
    assert out.predicate.to_ir() == IsNotNull(Col("s")).to_ir()


def test_contains_empty_becomes_not_null():
    out = st.empty_pattern_match_to_not_null(_flt(col("s").str.contains("")), None)
    assert out.predicate.to_ir() == IsNotNull(Col("s")).to_ir()


def test_non_empty_pattern_does_not_fire():
    assert st.empty_pattern_match_to_not_null(_flt(col("s").str.contains("a")), None) is None


def test_empty_pattern_does_not_fire_under_not():
    assert st.empty_pattern_match_to_not_null(_flt(~col("s").str.contains("")), None) is None


def test_empty_pattern_idempotent():
    once = st.empty_pattern_match_to_not_null(_flt(col("s").str.contains("")), None)
    assert st.empty_pattern_match_to_not_null(once, None) is None


# --- collapse_idempotent_str_func -------------------------------------------


def test_upper_of_upper_collapses():
    out = st.collapse_idempotent_str_func(_proj(col("s").str.upper().str.upper()), None)
    assert _expr_ir(out) == StrFunc("upper", Col("s")).to_ir()


def test_lower_of_lower_collapses():
    out = st.collapse_idempotent_str_func(_proj(col("s").str.lower().str.lower()), None)
    assert _expr_ir(out) == StrFunc("lower", Col("s")).to_ir()


def test_trim_of_trim_collapses():
    out = st.collapse_idempotent_str_func(_proj(col("s").str.trim().str.trim()), None)
    assert _expr_ir(out) == StrFunc("trim", Col("s")).to_ir()


def test_trim_stack_collapses_in_one_pass():
    out = st.collapse_idempotent_str_func(_proj(col("s").str.trim().str.trim().str.trim()), None)
    assert _expr_ir(out) == StrFunc("trim", Col("s")).to_ir()


def test_lstrip_of_lstrip_collapses():
    out = st.collapse_idempotent_str_func(_proj(col("s").str.lstrip("ab").str.lstrip("ab")), None)
    assert _expr_ir(out) == StrFunc("l_trim", Col("s"), pattern="ab").to_ir()


def test_trim_with_different_char_sets_does_not_collapse():
    # Two different sets are two different functions — trim(trim(x,'ab'),'cd') != trim(x,'cd').
    assert (
        st.collapse_idempotent_str_func(_proj(col("s").str.trim("ab").str.trim("cd")), None) is None
    )


def test_upper_of_lower_does_not_collapse():
    # upper(lower(x)) is NOT upper(x) in general (Unicode: 'İ').
    assert st.collapse_idempotent_str_func(_proj(col("s").str.lower().str.upper()), None) is None


def test_reverse_of_reverse_does_not_collapse():
    # reverse is an involution, not idempotent — it is not in the set.
    assert (
        st.collapse_idempotent_str_func(_proj(col("s").str.reverse().str.reverse()), None) is None
    )


def test_idempotent_collapse_is_idempotent():
    once = st.collapse_idempotent_str_func(_proj(col("s").str.upper().str.upper()), None)
    assert st.collapse_idempotent_str_func(once, None) is None


# --- trim_absorbs_inner_side_trim -------------------------------------------


def test_trim_absorbs_lstrip():
    out = st.trim_absorbs_inner_side_trim(_proj(col("s").str.lstrip().str.trim()), None)
    assert _expr_ir(out) == StrFunc("trim", Col("s")).to_ir()


def test_trim_absorbs_rstrip_with_chars():
    out = st.trim_absorbs_inner_side_trim(_proj(col("s").str.rstrip("ab").str.trim("ab")), None)
    assert _expr_ir(out) == StrFunc("trim", Col("s"), pattern="ab").to_ir()


def test_trim_does_not_absorb_a_different_char_set():
    assert (
        st.trim_absorbs_inner_side_trim(_proj(col("s").str.lstrip("ab").str.trim("cd")), None)
        is None
    )


def test_lstrip_does_not_absorb_trim():
    # The other direction is NOT sound: ltrim(trim(x)) != ltrim(x) (the tail stays trimmed).
    assert st.trim_absorbs_inner_side_trim(_proj(col("s").str.trim().str.lstrip()), None) is None


def test_trim_absorb_idempotent():
    once = st.trim_absorbs_inner_side_trim(_proj(col("s").str.lstrip().str.trim()), None)
    assert st.trim_absorbs_inner_side_trim(once, None) is None


# --- replace_identity_to_input ----------------------------------------------


def test_replace_same_from_and_to_drops_the_call():
    out = st.replace_identity_to_input(_proj(col("s").str.replace("a", "a")), None)
    assert _expr_ir(out) == Col("s").to_ir()


def test_replace_empty_from_and_to_drops_the_call():
    out = st.replace_identity_to_input(_proj(col("s").str.replace("", "")), None)
    assert _expr_ir(out) == Col("s").to_ir()


def test_replace_different_from_and_to_does_not_fire():
    assert st.replace_identity_to_input(_proj(col("s").str.replace("a", "b")), None) is None


def test_replace_identity_does_not_fire_on_binary_column():
    # Dropping the `replace` would leave the column Binary instead of the Utf8 the
    # function's coercion produced — an output-type change.
    plan = _binary_ds().select(r=col("b").str.replace("a", "a"))._plan
    assert st.replace_identity_to_input(plan, None) is None


def test_replace_identity_idempotent():
    once = st.replace_identity_to_input(_proj(col("s").str.replace("a", "a")), None)
    assert st.replace_identity_to_input(once, None) is None


# --- fold_case_of_literal ---------------------------------------------------


def test_upper_of_ascii_literal_folds():
    out = st.fold_case_of_literal(_proj(StrFunc("upper", Lit("abc"))), None)
    assert _expr_ir(out) == Lit("ABC").to_ir()


def test_lower_of_ascii_literal_folds():
    out = st.fold_case_of_literal(_proj(StrFunc("lower", Lit("AbC"))), None)
    assert _expr_ir(out) == Lit("abc").to_ir()


def test_case_of_non_ascii_literal_does_not_fold():
    # Full Unicode case mapping is the engine's to define, not Python's.
    assert st.fold_case_of_literal(_proj(StrFunc("upper", Lit("ß"))), None) is None


def test_case_of_column_does_not_fold():
    assert st.fold_case_of_literal(_proj(col("s").str.upper()), None) is None


def test_fold_case_idempotent():
    once = st.fold_case_of_literal(_proj(StrFunc("upper", Lit("abc"))), None)
    assert st.fold_case_of_literal(once, None) is None


# --- fold_len_of_literal ----------------------------------------------------


def test_len_of_literal_folds_to_character_count():
    out = st.fold_len_of_literal(_proj(StrFunc("len", Lit("héllo"))), None)
    assert _expr_ir(out) == Lit(5).to_ir()


def test_octet_length_of_literal_folds_to_byte_count():
    out = st.fold_len_of_literal(_proj(StrFunc("octet_length", Lit("héllo"))), None)
    assert _expr_ir(out) == Lit(6).to_ir()


def test_bit_length_of_literal_folds():
    out = st.fold_len_of_literal(_proj(StrFunc("bit_length", Lit("héllo"))), None)
    assert _expr_ir(out) == Lit(48).to_ir()


def test_len_of_column_does_not_fold():
    assert st.fold_len_of_literal(_proj(col("s").str.len()), None) is None


# --- fold_concat_of_literals ------------------------------------------------


def test_concat_of_two_literals_folds():
    out = st.fold_concat_of_literals(_proj(Binary("concat", Lit("a"), Lit("b"))), None)
    assert _expr_ir(out) == Lit("ab").to_ir()


def test_concat_chain_of_literals_folds_in_one_pass():
    expr = Binary("concat", Binary("concat", Lit("a"), Lit("b")), Lit("c"))
    out = st.fold_concat_of_literals(_proj(expr), None)
    assert _expr_ir(out) == Lit("abc").to_ir()


def test_concat_with_a_column_does_not_fold():
    assert st.fold_concat_of_literals(_proj(Binary("concat", Col("s"), Lit("b"))), None) is None


def test_concat_with_an_int_literal_does_not_fold():
    # The engine casts the int to Utf8 with Arrow's formatting, which is not Python's.
    assert st.fold_concat_of_literals(_proj(Binary("concat", Lit(1), Lit("b"))), None) is None


# --- fold_substr_of_literal -------------------------------------------------


def test_substr_of_literal_folds():
    out = st.fold_substr_of_literal(
        _proj(StrFunc("substr", Lit("abcdef"), start=2, length=3)), None
    )
    assert _expr_ir(out) == Lit("bcd").to_ir()


def test_substr_of_literal_clips_a_zero_start():
    # DuckDB/engine semantics: the window [0, 2] clips to [1, 2] → 'ab' (not 'abc').
    out = st.fold_substr_of_literal(
        _proj(StrFunc("substr", Lit("abcdef"), start=0, length=3)), None
    )
    assert _expr_ir(out) == Lit("ab").to_ir()


def test_substr_of_literal_negative_start_counts_from_the_end():
    out = st.fold_substr_of_literal(
        _proj(StrFunc("substr", Lit("abcdef"), start=-2, length=4)), None
    )
    assert _expr_ir(out) == Lit("ef").to_ir()


def test_substr_of_literal_without_length_runs_to_the_end():
    out = st.fold_substr_of_literal(_proj(StrFunc("substr", Lit("abcdef"), start=2)), None)
    assert _expr_ir(out) == Lit("bcdef").to_ir()


def test_substr_of_literal_empty_window():
    out = st.fold_substr_of_literal(
        _proj(StrFunc("substr", Lit("abcdef"), start=9, length=2)), None
    )
    assert _expr_ir(out) == Lit("").to_ir()


def test_substr_of_column_does_not_fold():
    assert st.fold_substr_of_literal(_proj(col("s").str.substr(2, 3)), None) is None
