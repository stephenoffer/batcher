"""``LIKE`` / ``ILIKE`` lowering, including the shapes that skip the pattern matcher.

A boundary-only ``%`` with no ``_`` is a prefix/suffix/substring test, which the engine has
dedicated kernels for and which Kyber can turn into a zone-map-prunable range; anything
richer goes to the native matcher. A pattern that is not a constant cannot be classified at
plan time at all, and takes the per-row form (`lowering.dynamic`).
"""

from __future__ import annotations

from batcher._sql.parser.expressions.literals import _like_to_regex
from batcher._sql.parser.expressions.lowering.dynamic import const_str, str_call
from batcher.plan.expr_ir import Expr, lit

__all__ = ["like"]


def like(tr, node, case_insensitive: bool = False, escape: str | None = None) -> Expr:
    pattern_node = node.expression
    pattern = const_str(pattern_node)
    if pattern is None:
        # A per-row pattern (`s LIKE pattern_col`) goes to the native matcher directly:
        # the prefix/suffix specializations below all read the pattern at plan time, and
        # ESCAPE desugars to a regex built from it, so neither is available here.
        if escape is not None:
            raise NotImplementedError("LIKE ... ESCAPE needs a constant pattern")
        tag = "ilike" if case_insensitive else "like"
        result = str_call(tr, tag, node.this, pattern=pattern_node)
        return ~result if node.args.get("negate") else result
    target = tr._scalar(node.this)
    # ILIKE: fold both sides to lower case for a case-insensitive match.
    if case_insensitive:
        target = target.str.lower()
        pattern = pattern.lower()

    # Boundary-only `%` with no `_`/ESCAPE lowers to the anchored
    # starts_with/ends_with/contains kernels — leanest, and the shape Kyber's
    # `like_prefix_to_range` can further turn into a zone-map-prunable range.
    #
    # Anything richer goes to the native `like`, whose Rust matcher classifies the
    # pattern *once per morsel* into the cheapest shape (prefix/suffix/ordered
    # `memmem` segment scan) and falls back to a cached anchored regex only for `_`.
    # It must not be spelled as `regexp_matches(_like_to_regex(...))`: that runs a
    # regex automaton per row for `%a%b%`, which measured ~7x DuckDB on TPC-H q13's
    # `o_comment NOT LIKE '%special%requests%'` (76ms vs 11ms) — and, lacking `(?s)`,
    # also made `%` stop at a newline, which SQL says it must not.
    #
    # ESCAPE keeps the Python-desugared regex: the native matcher has no escape char.
    simple = escape is None and "_" not in pattern and "%" not in pattern.strip("%")
    if simple:
        result = _like_simple(target, pattern)
    elif escape is None:
        result = target.str.like(pattern)
    else:
        result = target.str.regexp_matches(_like_to_regex(pattern, escape))

    # `x NOT LIKE p` parses as a Like node with negate=True.
    if node.args.get("negate"):
        result = ~result
    return result


def _like_simple(target: Expr, pattern: str) -> Expr:
    starts = pattern.startswith("%")
    ends = pattern.endswith("%")
    # Strip *all* boundary `%`, not just one: a pattern like `%%c` / `a%%` carries
    # consecutive leading/trailing wildcards, and the caller's `simple` guard already
    # proved the stripped core holds no `%`/`_`, so it is a pure literal. Peeling only a
    # single `%` left an interior `%` in `inner` that `starts_with`/`ends_with`/`contains`
    # then matched literally (`'abc' LIKE '%%c'` → false instead of true).
    inner = pattern.strip("%")
    if starts and ends:
        return target.str.contains(inner)
    if ends:  # 'abc%'
        return target.str.starts_with(inner)
    if starts:  # '%abc'
        return target.str.ends_with(inner)
    return target == lit(inner)  # no wildcards → exact match
