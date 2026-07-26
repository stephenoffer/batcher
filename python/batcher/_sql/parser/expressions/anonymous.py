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
Batcher equivalent differs in semantics is left out, so it keeps raising a clear "not
supported" rather than returning a plausible wrong answer — `first`/`last`/`any_value`
(DuckDB returns an unspecified row's value; the engine's require an explicit ordering)
and `fsum`/`kahan_sum` (compensated summation, which the engine's `sum` is not) are the
current members of that list.

Note the two `jaccard`s: DuckDB's is over the two strings' character *sets*, which is
`.str.jaccard`, **not** the engine's list-valued `.list.jaccard`. Mapping the name to the
nearer-looking one would have been exactly the failure this module's rule exists to
prevent.
"""

from __future__ import annotations

import math

from batcher.plan.expr_ir import Binary, Expr, array, atan2, lit
from batcher.plan.functions.scalar import gcd, hypot, lcm, nanvl, next_after
from batcher.plan.functions.temporal import current_date, make_date

__all__ = ["anonymous_scalar"]


# --- one table per argument shape -------------------------------------------

# `f(x)` → a method on the value expression itself.
_UNARY_EXPR = {
    "bit_count": "bit_count",
    "isfinite": "is_finite",
    "isinf": "is_infinite",
    "isnan": "is_nan",
    "even": "even",
    "gamma": "gamma",
    "lgamma": "lgamma",
    "sec": "sec",
    "csc": "csc",
    "rint": "rint",
    # Spark spellings whose Batcher method is identically named.
    "isnull": "is_null",
    "isnotnull": "is_not_null",
    "log1p": "log1p",
    "expm1": "expm1",
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
    "from_hex": "unhex",
    "url_encode": "url_encode",
    "url_decode": "url_decode",
    "regexp_escape": "regexp_escape",
    "parse_filename": "parse_filename",
    "parse_dirname": "parse_dirname",
    "parse_dirpath": "parse_dirpath",
    "parse_path": "parse_path",
    "to_binary": "to_binary",
    "from_binary": "from_binary",
    "xxhash64": "xxhash64",
    # Spark's `try_*` string forms differ from the plain ones only by returning null
    # instead of raising on a *conversion* failure, and these two already return null
    # rather than raising, so the two spellings mean the same thing here.
    "try_to_binary": "to_binary",
    "try_url_decode": "url_decode",
}

# `f(m)` → a `.map` method. `map_keys` reaches the typed dispatch; `map_values` does not.
_UNARY_MAP = {
    "map_values": "values",
    "map_keys": "keys",
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
    "hamming": "hamming",
    "mismatches": "hamming",
    "jaccard": "jaccard",
    "prefix": "starts_with",
    "suffix": "ends_with",
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
    "nextafter": next_after,
    "hypot": hypot,
    "nanvl": nanvl,
    # Spark's `try_mod` returns null on a zero divisor where `mod` raises; the engine's
    # `%` already yields null there, so the two spellings coincide.
    "try_mod": lambda a, b: a.mod(b),
}

# `f(a, b, c)` → a three-argument builder over the translated operands.
#
# `list_slice` is DuckDB's, and its two bounds are an inclusive 1-based `begin`..`end`
# pair — *not* Spark's `slice(l, start, length)`, which sqlglot gives its own node. The
# two spellings differ in both the base and the meaning of the second operand, so they
# cannot share a row.
_TERNARY_FN = {
    "make_date": make_date,
}

# `f()` → a constant. DuckDB spells these as nullary functions; the engine has no node
# for them because there is nothing per-row to compute, so they fold to a literal at
# plan-build time — which is also what makes them constant-foldable downstream.
_NULLARY = {
    "pi": lambda: lit(math.pi),
    "today": current_date,
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

    if not args:
        nullary = _NULLARY.get(name)
        if nullary is not None:
            return nullary()

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
        if name in _UNARY_MAP:
            return getattr(tr._scalar(one).map, _UNARY_MAP[name])()

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

    if len(args) == 3:
        if name in ("list_slice", "array_slice"):
            begin = _const_int_arg(args[1], f"{name}(): begin")
            end = _const_int_arg(args[2], f"{name}(): end")
            return tr._scalar(args[0]).list.slice(begin - 1, max(end - begin + 1, 0))
        builder = _TERNARY_FN.get(name)
        if builder is not None:
            return builder(*(tr._scalar(a) for a in args))

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
