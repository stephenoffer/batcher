"""String-expression rewrites — LIKE despecialization, idempotence collapse, literal folding.

NORMALIZE-phase, node-local rules over the expressions a `Filter` or `Project` carries. They
turn an opaque regex-backed `LIKE` into a cheap anchored search (`=` / `starts_with` /
`ends_with` / `contains` / `IS NOT NULL`), collapse redundant nested string functions, and fold
a string function over a literal. Each is a `@rule`; the driver supplies bottom-up traversal and
fixpoint iteration.

Three hazards govern every rule, and each is resolved by *declining* to fire, never by guessing.
A rule's docstring names the ones it navigates; the reasoning lives here.

* **The LIKE pattern language.** `bc-expr` compiles a pattern to an anchored regex: `%` → `.*`,
  `_` → `.` (exactly one character), every other character literal (`regex::escape`d, with `(?s)`
  so `.` matches a newline too). A pattern is a plain string only if it holds **no `%` and no
  `_`** — `x LIKE 'a_c'` is emphatically *not* `x = 'a_c'`. A **backslash** is refused outright:
  neither the engine nor DuckDB gives it a meaning today, but `LIKE … ESCAPE` does, and baking in
  "backslash is literal" would silently change meaning the day an escape clause reaches the IR.
  `ILIKE` is never rewritten (nothing case-insensitive reproduces it).

* **NULL versus FALSE.** `LIKE`/`contains`/`starts_with`/`ends_with` return NULL on a NULL input;
  `IS NOT NULL` returns FALSE. They differ *as values*, so swapping them is unsound in a `Project`
  and under a `NOT`/`OR`/`CASE` (`NOT(x LIKE '%')` is NULL → row dropped, `NOT(x IS NOT NULL)` is
  TRUE → row kept). They are interchangeable in exactly one place: a **top-level conjunct of a
  `Filter`**, which keeps a row only when the predicate is TRUE — NULL and FALSE both fail that, so
  the same rows survive. The two rules that swap them match `Filter` alone and rewrite only its
  top-level conjuncts (`_apply_conjuncts`). Every other rule is NULL-preserving on both sides.

* **The type of the value.** A string function *coerces* a `Binary` column to `Utf8` (the
  ClickBench `hits` shape); dropping it would leave the column `Binary` and change the output type.
  So every rule that removes the last string function around a value is "Utf8-guarded" (`_is_utf8`,
  which admits only a provably-Utf8 operand). Rules that leave one in place need no guard.

Literal folds evaluate in Python only where Python and Rust provably agree: substring search and
`substr` (identical character-indexed semantics), length (`len(s)` counts code points, exactly
`chars().count()`), and case conversion **restricted to ASCII** — full Unicode case mapping is
where implementations diverge (`upper('ß')` is `'SS'` in the engine, `'ẞ'` in DuckDB).
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.normalize import ranges
from batcher.plan.expr_ir import Binary, Col, Expr, IsNotNull, Lit, StrFunc
from batcher.plan.expr_rewrite import (
    combine_conjuncts,
    map_node_expressions,
    split_conjuncts,
    transform_expr_up,
)
from batcher.plan.logical import Filter, LogicalPlan, Project
from batcher.plan.schema import SchemaRef

__all__ = [
    "collapse_idempotent_str_func",
    "empty_pattern_match_to_not_null",
    "like_contains_to_contains",
    "like_only_wildcard_to_not_null",
    "like_prefix_to_starts_with",
    "like_suffix_to_ends_with",
    "like_without_wildcards_to_eq",
    "replace_identity_to_input",
    "trim_absorbs_inner_side_trim",
]

# What disqualifies a LIKE pattern (or a run inside one) from being a plain literal: the two
# wildcards, plus the backslash an `ESCAPE` clause would reinterpret.
_LIKE_SPECIAL = frozenset("%_\\")

# Idempotent string functions: `f(f(x)) == f(x)` for every input, given the *same* arguments.
# Unicode's case mappings are idempotent (the engine's `upper('ß')` is `'SS'` and `upper('SS')` is
# `'SS'`; a lowercased string lowercases to itself, final sigma included), trimming a character set
# removes nothing the first pass left at the ends, and `initcap` re-title-cases an already-title-
# cased word to itself. `reverse` is an involution, not idempotent — it is deliberately absent.
_IDEMPOTENT_STR_FNS = frozenset({"upper", "lower", "trim", "l_trim", "r_trim", "initcap"})

# String functions whose result is Utf8 — the vocabulary of the `_is_utf8` type guard.
_UTF8_RESULT_FNS = frozenset(
    {
        "aes_decrypt", "aes_encrypt", "base64", "from_base64", "hex", "hmac_sha256", "initcap",
        "json_extract_string", "l_trim", "lower", "lpad", "mask", "md5", "overlay", "r_trim",
        "regexp_extract", "regexp_replace", "regexp_replace_all", "repeat", "replace", "reverse",
        "right", "rpad", "sha1", "sha256", "soundex", "split_part", "strip_html", "substr",
        "substring_index", "translate", "trim", "unhex", "upper",
    }
)  # fmt: skip

# The substring-search predicates: with an *empty* pattern each is TRUE for every non-null string.
_SEARCH_FNS = frozenset({"contains", "starts_with", "ends_with"})


def _str_cols(schema: SchemaRef | None) -> frozenset[str]:
    """Names of the schema's Utf8 columns — an empty set when the schema is unknown, which makes
    every type-guarded rule a no-op (the safe default)."""
    if schema is None:
        return frozenset()
    return frozenset(
        f.name
        for f in schema.arrow
        if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)
    )


def _is_utf8(expr: Expr, str_cols: frozenset[str]) -> bool:
    """Whether `expr` is provably Utf8: a string literal, a Utf8 column, a `concat`, or a
    Utf8-returning string function. A `Binary` column (the engine would coerce it) and an
    unknown-schema column are refused, so a rewrite can never change an output type."""
    if isinstance(expr, Lit):
        return type(expr.value) is str
    if isinstance(expr, Col):
        return expr.name in str_cols
    if isinstance(expr, StrFunc):
        return expr.fn in _UTF8_RESULT_FNS
    return isinstance(expr, Binary) and expr.op == "concat"


def _cols_of(node: LogicalPlan) -> frozenset[str]:
    """The Utf8 columns of `node`'s input, memoized on the (immutable) input node — the same
    `__dict__` cache `to_ir`/`available_schema` use, since every rule asks for it on every node on
    every fixpoint iteration."""
    inp = node.input
    cached = inp.__dict__.get("_c_str_cols")
    if cached is None:
        cached = _str_cols(inp.available_schema())
        inp.__dict__["_c_str_cols"] = cached
    return cached


#: `id(schema) -> (schema, utf8 column names)`. The fused chain hands each leaf the node's
#: schema, but these leaves want the *Utf8 column set* derived from it, and deriving that is
#: a scan of every field. Schemas are immutable and shared across a plan's nodes, so one
#: entry per distinct schema serves every leaf and every expression. The schema is stored
#: alongside to pin the id against reuse. (`_cols_of` memoizes the same answer on the input
#: node, for the standalone path that still resolves the schema itself.)
_STR_COLS_BY_SCHEMA: dict[int, tuple[object, frozenset[str]]] = {}
_STR_COLS_MAX = 256


def _str_cols_of(schema: SchemaRef | None) -> frozenset[str]:
    if schema is None:
        return frozenset()
    cached = _STR_COLS_BY_SCHEMA.get(id(schema))
    if cached is not None and cached[0] is schema:
        return cached[1]
    found = _str_cols(schema)
    if len(_STR_COLS_BY_SCHEMA) >= _STR_COLS_MAX:
        _STR_COLS_BY_SCHEMA.clear()
    _STR_COLS_BY_SCHEMA[id(schema)] = (schema, found)
    return found


def _schema_leaf(leaf):
    """Adapt a `leaf(expr, str_cols)` to the `(Expr, SchemaRef) -> Expr` shape the driver's
    fused traversal runs, so these rules share its one walk instead of each making their own.

    Only for the leaves applied to *every* expression a node carries. The two rules that go
    through `_apply_conjuncts` must not be adapted: they are sound only at the top level of a
    filter predicate, where a NULL result and a FALSE result are interchangeable, and the
    fused chain would offer them every nested sub-expression as well.
    """

    def bound(expr: Expr, schema: SchemaRef | None) -> Expr:
        return leaf(expr, _str_cols_of(schema))

    return bound


def _apply(node: LogicalPlan, leaf) -> LogicalPlan | None:
    """Run `leaf(expr, str_cols)` bottom-up over every expression `node` carries, returning the
    rebuilt node — or `None` when nothing changed, so the driver's fixpoint terminates."""
    str_cols = _cols_of(node)
    rebuilt = map_node_expressions(
        node, lambda e: transform_expr_up(e, lambda x: leaf(x, str_cols))
    )
    return None if rebuilt is node else rebuilt


def _apply_conjuncts(node: Filter, leaf) -> LogicalPlan | None:
    """Run `leaf(conjunct, str_cols)` over the *top-level conjuncts* of a `Filter` predicate —
    never a nested sub-expression. That is the one position where a NULL result and a FALSE result
    are interchangeable (module docstring), so it is the only place the `IS NOT NULL` rules look."""
    str_cols = _cols_of(node)
    conjuncts = split_conjuncts(node.predicate)
    rewritten = [leaf(c, str_cols) for c in conjuncts]
    if all(a is b for a, b in zip(rewritten, conjuncts, strict=True)):
        return None
    return dataclasses.replace(node, predicate=combine_conjuncts(rewritten))


def _like_pattern(expr: Expr) -> str | None:
    """The pattern of a `LIKE` whose pattern holds no backslash, else None — the gate every LIKE
    rule passes through (it also excludes `ILIKE`, which has no case-insensitive twin)."""
    if not (isinstance(expr, StrFunc) and expr.fn == "like"):
        return None
    pattern = expr.pattern
    if not isinstance(pattern, str) or "\\" in pattern:
        return None
    return pattern


def _is_literal_run(text: str) -> bool:
    """Whether `text` holds no LIKE wildcard and no backslash — i.e. the engine's pattern compiler
    emits it as an escaped literal run matching only itself."""
    return not any(c in _LIKE_SPECIAL for c in text)


def _same_args(a: StrFunc, b: StrFunc) -> bool:
    """Whether two string-function nodes carry identical arguments (everything but the input)."""
    return (
        a.pattern == b.pattern
        and a.replacement == b.replacement
        and a.start == b.start
        and a.length == b.length
    )


def _str_lit(expr: Expr) -> str | None:
    """The value of a string `Lit`, else None."""
    return expr.value if isinstance(expr, Lit) and type(expr.value) is str else None


def _leaf_like_to_eq(expr: Expr, str_cols: frozenset[str]) -> Expr:
    pattern = _like_pattern(expr)
    if pattern is None or not _is_literal_run(pattern) or not _is_utf8(expr.input, str_cols):
        return expr
    return Binary("eq", expr.input, Lit(pattern))


@rule(
    name="like_without_wildcards_to_eq",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("like",),
    expr_schema=_schema_leaf(_leaf_like_to_eq),
)
def like_without_wildcards_to_eq(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Rewrite a wildcard-free `x LIKE 'abc'` to the equality `x = 'abc'`.

    With no `%`/`_` the engine compiles the pattern to `^abc$` over an escaped literal run, matching
    iff the string *equals* `'abc'` byte for byte — what `eq` computes on Utf8. Both are NULL on a
    NULL input (sound in a `Project` too), and the raw column now reaches pushdown, bloom probing
    and zone maps. Declines on any `%`/`_`, on a backslash and on `ILIKE`; Utf8-guarded.
    """
    return _apply(node, _leaf_like_to_eq)


def _leaf_like_prefix(expr: Expr, _str_cols: frozenset[str]) -> Expr:
    pattern = _like_pattern(expr)
    if pattern is None or len(pattern) < 2 or not pattern.endswith("%"):
        return expr
    prefix = pattern[:-1]
    if not _is_literal_run(prefix):
        return expr
    # Mutual exclusion with `like_prefix_to_range`: where that rule can build an exact range (an
    # incrementable trailing character), it owns this pattern — a range exposes the raw column to
    # zone-map pruning, which `starts_with` cannot. Only the prefixes it declines land here.
    if ranges._prefix_upper_bound(pattern) is not None:
        return expr
    return StrFunc("starts_with", expr.input, pattern=prefix)


@rule(
    name="like_prefix_to_starts_with",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("like",),
    expr_schema=_schema_leaf(_leaf_like_prefix),
)
def like_prefix_to_starts_with(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Rewrite a pure-prefix `x LIKE 'abc%'` to `starts_with(x, 'abc')`.

    `'abc%'` compiles to `(?s)^abc.*$`, which holds iff the string starts with `'abc'` — precisely
    Rust's `str::starts_with`; NULL-preserving on both sides (sound in a `Project` too), a memcmp
    instead of a regex. **Complements `normalize.ranges.like_prefix_to_range`**, which turns the
    same shape into the strictly-better `x >= 'abc' AND x < 'abd'` (a range prunes zone maps) but
    only for a safely-incrementable ASCII tail: this fires *only* where that rule declines, so the
    two are mutually exclusive by construction, not by rule order.
    """
    return _apply(node, _leaf_like_prefix)


def _leaf_like_suffix(expr: Expr, _str_cols: frozenset[str]) -> Expr:
    pattern = _like_pattern(expr)
    if pattern is None or len(pattern) < 2 or not pattern.startswith("%"):
        return expr
    suffix = pattern[1:]
    return StrFunc("ends_with", expr.input, pattern=suffix) if _is_literal_run(suffix) else expr


@rule(
    name="like_suffix_to_ends_with",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("like",),
    expr_schema=_schema_leaf(_leaf_like_suffix),
)
def like_suffix_to_ends_with(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Rewrite a pure-suffix `x LIKE '%abc'` to `ends_with(x, 'abc')`.

    `'%abc'` compiles to `(?s)^.*abc$`, and `$` in the `regex` crate matches only the end of the
    haystack (no Perl-style "before a trailing newline" exception), so it holds iff the string ends
    with `'abc'` — precisely Rust's `str::ends_with`. NULL-preserving on both sides (sound in a
    `Project` too). Declines on a further `%`/`_` in the suffix, and on a backslash.
    """
    return _apply(node, _leaf_like_suffix)


def _leaf_like_contains(expr: Expr, _str_cols: frozenset[str]) -> Expr:
    pattern = _like_pattern(expr)
    if pattern is None or len(pattern) < 3:
        return expr
    if not (pattern.startswith("%") and pattern.endswith("%")):
        return expr
    middle = pattern[1:-1]
    if not middle or not _is_literal_run(middle):
        return expr
    return StrFunc("contains", expr.input, pattern=middle)


@rule(
    name="like_contains_to_contains",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("like",),
    expr_schema=_schema_leaf(_leaf_like_contains),
)
def like_contains_to_contains(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Rewrite an infix `x LIKE '%abc%'` to `contains(x, 'abc')`.

    `'%abc%'` compiles to `(?s)^.*abc.*$`, and because `(?s)` lets `.` match a newline that holds
    iff `'abc'` occurs anywhere in the string: precisely Rust's `str::contains`. NULL-preserving on
    both sides (sound in a `Project` too). Declines when the middle is empty (`'%%'` is the
    every-non-null-string case, owned by `like_only_wildcard_to_not_null`) or holds a wildcard.
    """
    return _apply(node, _leaf_like_contains)


def _leaf_all_wildcard(expr: Expr, str_cols: frozenset[str]) -> Expr:
    pattern = _like_pattern(expr)
    if pattern is None or pattern == "" or any(c != "%" for c in pattern):
        return expr
    return IsNotNull(expr.input) if _is_utf8(expr.input, str_cols) else expr


@rule(
    name="like_only_wildcard_to_not_null",
    phase=Phase.NORMALIZE,
    matches=(Filter,),
    expr_matches=(StrFunc,),
    expr_ops=("like",),
)
def like_only_wildcard_to_not_null(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Rewrite a top-level `Filter` conjunct `x LIKE '%'` (or `'%%'`, …) to `x IS NOT NULL`.

    `%` matches any run of characters *including the empty one*, so `x LIKE '%'` is TRUE for every
    non-null string and NULL for a null one — a null check written as a regex. The two are **not**
    equal as expressions (NULL vs FALSE), so this fires only on a top-level `Filter` conjunct, where
    they are interchangeable: the filter keeps a row iff the predicate is TRUE, and NULL and FALSE
    both fail that. Never under a `NOT`/`OR`/`CASE`, never in a `Project`, and Utf8-guarded.
    """
    return _apply_conjuncts(node, _leaf_all_wildcard)


def _leaf_empty_pattern(expr: Expr, str_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn in _SEARCH_FNS):
        return expr
    if expr.pattern != "" or not _is_utf8(expr.input, str_cols):
        return expr
    return IsNotNull(expr.input)


@rule(
    name="empty_pattern_match_to_not_null",
    phase=Phase.NORMALIZE,
    matches=(Filter,),
    expr_matches=(StrFunc,),
    expr_ops=tuple(sorted(_SEARCH_FNS)),
)
def empty_pattern_match_to_not_null(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Rewrite a top-level `Filter` conjunct `starts_with(x, '')` / `ends_with(x, '')` /
    `contains(x, '')` to `x IS NOT NULL`.

    Every string — the empty one included — starts with, ends with and contains `''` (Rust and
    DuckDB agree), so each is TRUE for every non-null value and NULL for a null one: a null check in
    disguise, usually left by a parameterized query whose search term came through empty. Sound for
    exactly the reason `like_only_wildcard_to_not_null` is, and restricted the same way: top-level
    `Filter` conjuncts only, never under a `NOT`/`OR`/`CASE`, never in a `Project`.
    """
    return _apply_conjuncts(node, _leaf_empty_pattern)


def _leaf_idempotent(expr: Expr, _str_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn in _IDEMPOTENT_STR_FNS):
        return expr
    inner = expr.input
    if isinstance(inner, StrFunc) and inner.fn == expr.fn and _same_args(expr, inner):
        return inner
    return expr


@rule(
    name="collapse_idempotent_str_func",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=tuple(sorted(_IDEMPOTENT_STR_FNS)),
    expr_schema=_schema_leaf(_leaf_idempotent),
)
def collapse_idempotent_str_func(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Collapse a nested idempotent string function: `upper(upper(x))` → `upper(x)`, and the same
    for `lower`, `initcap`, `trim`, `l_trim` and `r_trim`.

    Each is a projection onto its own image — a second application cannot move a value the first
    already normalized (`_IDEMPOTENT_STR_FNS` carries the per-function argument, Unicode
    case-mapping edges included) — so the outer call is a full pass over every string, wasted. The
    nodes must carry **identical arguments** (`trim(trim(x, 'ab'), 'cd')` trims two different sets);
    the outer function survives, so type, coercion and NULL are unchanged.
    """
    return _apply(node, _leaf_idempotent)


def _leaf_trim_absorb(expr: Expr, _str_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn == "trim"):
        return expr
    inner = expr.input
    if not (isinstance(inner, StrFunc) and inner.fn in ("l_trim", "r_trim")):
        return expr
    if inner.pattern != expr.pattern:
        return expr
    return StrFunc("trim", inner.input, pattern=expr.pattern)


@rule(
    name="trim_absorbs_inner_side_trim",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("trim",),
    expr_schema=_schema_leaf(_leaf_trim_absorb),
)
def trim_absorbs_inner_side_trim(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Absorb a one-sided trim into an enclosing two-sided one: `trim(ltrim(x))` → `trim(x)`, and
    the `rtrim` twin — when both trim the same character set.

    `trim` strips that set from *both* ends, so whatever `l_trim` removed from the front (or
    `r_trim` from the back) `trim` would have removed anyway: the inner call changes the cost, not
    the result. Requires identical trim sets, and the reverse nesting is *not* rewritten
    (`ltrim(trim(x))` keeps its tail trimmed). The outer `trim` survives, so NULL and type are too.
    """
    return _apply(node, _leaf_trim_absorb)


def _leaf_replace_identity(expr: Expr, str_cols: frozenset[str]) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn == "replace"):
        return expr
    if expr.pattern is None or expr.pattern != expr.replacement:
        return expr
    return expr.input if _is_utf8(expr.input, str_cols) else expr


@rule(
    name="replace_identity_to_input",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("replace",),
    expr_schema=_schema_leaf(_leaf_replace_identity),
)
def replace_identity_to_input(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a no-op replacement: `replace(x, s, s)` → `x`.

    Replacing every occurrence of `s` with `s` rewrites each match to itself, so the result is the
    input unchanged — for every `s`, the empty string included (Rust's `"abc".replace("", "")`
    splices an empty string between the characters and still yields `"abc"`). It costs a scan and an
    allocation per row and buys nothing. Utf8-guarded (it drops the last string function around the
    value); NULL is preserved.
    """
    return _apply(node, _leaf_replace_identity)
