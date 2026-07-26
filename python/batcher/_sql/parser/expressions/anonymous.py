"""DuckDB function names sqlglot leaves as `Anonymous` → the Batcher expression surface.

sqlglot promotes the SQL functions it knows to typed nodes (`exp.Levenshtein`,
`exp.Hex`, …), which `functions._scalar_function` dispatches by class name. Everything
else arrives as `exp.Anonymous` carrying only a name and an argument list, and used to
raise ``unknown function 'x'`` — even when the engine already had the function under
its DataFrame spelling. This module is the name-keyed half of the dispatch: one table
per argument shape, mapping the DuckDB spelling onto the expression method that already
implements it.

The tables are deliberately *not* a single flat dict. The argument shape decides how a
call is built — a constant string, an expression, an integer index — and folding them
together would mean re-deriving the shape at the call site from the arity, which is how
``lpad(s, -1, '*')`` and ``substr(s, -2)`` were mis-parsed before. One table per shape
makes the shape a property of the entry.

Only names whose result is *bit-identical* to DuckDB's are listed. A name whose closest
Batcher equivalent differs in semantics (DuckDB's character-bigram ``jaccard`` against
the engine's list-valued `.list.jaccard`, say) is left out, so it keeps raising a clear
"not supported" rather than returning a plausible wrong answer.
"""

from __future__ import annotations

from batcher.plan.expr_ir import Binary, Expr, array, atan2
from batcher.plan.functions.scalar import gcd, lcm

__all__ = ["anonymous_scalar"]


# --- one table per argument shape -------------------------------------------

# `f(x)` → a method on the value expression itself.
_UNARY_EXPR = {
    "bit_count": "bit_count",
    "isfinite": "is_finite",
    "isinf": "is_infinite",
    "isnan": "is_nan",
}

# `f(s)` → a `.str` method.
_UNARY_STR = {
    "base64": "base64",
    "to_base64": "base64",
    "from_base64": "from_base64",
    "octet_length": "octet_length",
    "strlen": "len",
    "ord": "ascii",
    "crc32": "crc32",
    "initcap": "initcap",
    "soundex": "soundex",
}

# `f(ts)` → a `.dt` method. These are the DuckDB date-part spellings sqlglot has no
# typed node for; the typed ones (`Year`, `Quarter`, …) stay in `literals._DATE_PART`.
_UNARY_DT = {
    "century": "century",
    "decade": "decade",
    "millennium": "millennium",
    "epoch_ms": "epoch_ms",
    "epoch_ns": "epoch_ns",
    "epoch_us": "epoch_us",
    # DuckDB's `weekday` is Sunday-based (`dayofweek`), not the ISO Monday-based
    # `.dt.weekday()` — mapping it by name made Sunday 7 where DuckDB says 0.
    "weekday": "dayofweek",
    "isodow": "isodow",
    "isoyear": "iso_year",
    "days_in_month": "days_in_month",
}

# `f(ts)` where DuckDB's answer is *not* the bare `.dt` field of the same name.
#
# `microsecond`/`millisecond`/`nanosecond` in DuckDB include the seconds component —
# `microsecond('…:08.123456')` is 8_123_456, not the 123_456 the `.dt` field returns —
# and `yearweek` packs the ISO year and week into one integer. Mapping these to the
# same-named `.dt` method returned a plausible number that was not DuckDB's.
_UNARY_DT_DERIVED = {
    "microsecond": lambda ts: ts.dt.second() * 1_000_000 + ts.dt.microsecond(),
    "millisecond": lambda ts: ts.dt.second() * 1_000 + ts.dt.millisecond(),
    "nanosecond": lambda ts: ts.dt.second() * 1_000_000_000 + ts.dt.nanosecond(),
    "yearweek": lambda ts: ts.dt.iso_year() * 100 + ts.dt.week_of_year(),
}

# `f(s, t)` where `t` must be a constant string — the engine's string-metric and
# splitting methods take the second operand as a Python `str`, not an expression.
_STR_TEXT = {
    "levenshtein": "levenshtein",
    "editdist3": "levenshtein",
    "damerau_levenshtein": "damerau_levenshtein",
    "jaro_similarity": "jaro_similarity",
    "jaro_winkler_similarity": "jaro_winkler_similarity",
    "str_split": "split",
    "string_split": "split",
    "string_to_array": "split",
    "regexp_split_to_array": "regexp_split",
}

# `f(a, b)` → an arithmetic operator. DuckDB exposes the operators under function
# names too; `divide` is *integer* division on integer operands (`divide(3, 2)` is 1),
# so it lowers to floor division, not `/`.
_ARITH = {
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "floordiv",
    "fdiv": "truediv",
    "mod": "mod",
    "fmod": "mod",
}

# `f(a, b)` → a two-argument top-level builder.
_BINARY_FN = {
    "gcd": gcd,
    "greatest_common_divisor": gcd,
    "lcm": lcm,
    "least_common_multiple": lcm,
    "atan2": atan2,
}

# `f(list, i)` → 1-based element access. DuckDB's `list_extract`/`array_extract`/
# `list_element` are the same function under three names.
_ELEMENT_AT = frozenset(
    {"list_extract", "array_extract", "list_element", "array_element", "element_at"}
)

# `f(a, b)` on two list expressions → a binary `.list` method.
_LIST_PAIR = {
    "list_intersect": "intersect",
    "array_intersect": "intersect",
    "list_difference": "difference",
    "array_difference": "difference",
    "list_union": "union",
    "array_union": "union",
}

# `f(v, …)` → a list literal of every argument (DuckDB's list constructors).
_LIST_PACK = frozenset({"list_pack", "list_value", "array_value"})


def anonymous_scalar(tr, node):
    """Translate an `exp.Anonymous` scalar call, or return None if the name is unknown.

    Args:
        tr: The translator instance (for recursive `_scalar` calls).
        node: The `exp.Anonymous` node.

    Returns:
        The `Expr` this call denotes, or `None` when the name is not one this
        module serves (the caller then raises its own "unknown function" error).
    """
    from batcher._sql.parser.expressions.literals import _const_int_arg, _const_str_arg

    name = node.name.lower()
    args = list(node.expressions)

    if len(args) == 1:
        one = args[0]
        if name in _UNARY_EXPR:
            return getattr(tr._scalar(one), _UNARY_EXPR[name])()
        if name in _UNARY_STR:
            return getattr(tr._scalar(one).str, _UNARY_STR[name])()
        if name in _UNARY_DT:
            return getattr(tr._scalar(one).dt, _UNARY_DT[name])()
        derived = _UNARY_DT_DERIVED.get(name)
        if derived is not None:
            return derived(tr._scalar(one))

    if len(args) == 2:
        left, right = args
        if name in _STR_TEXT:
            text = _const_str_arg(right, f"{name}()", "second argument")
            return getattr(tr._scalar(left).str, _STR_TEXT[name])(text)
        if name in _ARITH:
            return _arith(tr, name, left, right)
        builder = _BINARY_FN.get(name)
        if builder is not None:
            return builder(tr._scalar(left), tr._scalar(right))
        if name in _ELEMENT_AT:
            # DuckDB indexes lists from 1; `.list.get` is 0-based.
            return tr._scalar(left).list.get(_const_int_arg(right, f"{name}(): index") - 1)
        if name in _LIST_PAIR:
            return getattr(tr._scalar(left).list, _LIST_PAIR[name])(tr._scalar(right))

    if name in _LIST_PACK:
        return array(*(tr._scalar(a) for a in args))

    return None


def _arith(tr, name: str, left, right) -> Expr:
    """Build the arithmetic operator `name` over two translated operands.

    `floordiv`/`truediv`/`mod` are `Expr` methods rather than `Binary` opcodes (they
    lower to the engine's own division semantics, including its divide-by-zero rule),
    so they are dispatched by attribute; `add`/`sub`/`mul` are plain binary nodes.
    """
    op = _ARITH[name]
    lhs, rhs = tr._scalar(left), tr._scalar(right)
    if op in ("floordiv", "truediv", "mod"):
        return getattr(lhs, op)(rhs)
    return Binary(op, lhs, rhs)
