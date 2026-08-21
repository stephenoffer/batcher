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

from batcher.plan.expr_ir import Binary, Expr, array, atan2, lit, nullif, when
from batcher.plan.functions.collection import element
from batcher.plan.functions.partitioning import (
    partition_days,
    partition_hours,
    partition_months,
    partition_truncate,
    partition_years,
)
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
    # DuckDB's `strlen` counts **bytes**, not characters — `strlen('Ünicode')` is 8 where
    # `length('Ünicode')` is 7. Mapping it onto `len` made the two synonyms, which they
    # are only for ASCII.
    "strlen": "octet_length",
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
# `map_entries` is here because the kernel already answered DuckDB's shape exactly — a list
# of `{key, value}` structs, in insertion order, with those two field names — and only the
# SQL name was missing. Unlike `map_extract` below, no adjustment is involved: the two
# results are equal element for element, which is what makes this a wiring fix rather than
# a semantic one.
_UNARY_MAP = {
    "map_values": "values",
    "map_keys": "keys",
    "map_entries": "entries",
    "cardinality": "len",
}

# `f(m, key)` → a `.map` method taking one literal key. Spark's `map_contains_key` is
# absent because sqlglot gives it a *typed* node, so it never reaches this table; adding
# it here would be dead code.
# `map_extract` is deliberately absent. DuckDB returns a **list** — `[1]` for a present
# key and `[]` for a missing one — where `m[k]` and Spark's `element_at(m, k)` return the
# bare value. Mapping it to `.map.get` would answer `1`/`NULL`: a plausible result that is
# not DuckDB's, which is the failure this census exists to catch. It needs a kernel that
# wraps the hit in a list, not a table row.
_MAP_KEY = {
    "map_contains": "contains",
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

#: The engine tag for a `_STR_TEXT` entry whose `.str` method name differs from it. Only
#: `jaccard` does; the rest are the same word, and listing them would be a second copy to
#: keep in step.
_STR_TEXT_TAG = {"jaccard": "jaccard_similarity"}

# `f(a, b)` → an arithmetic operator. DuckDB exposes the operators under function names
# too, and the two division spellings are *not* the ones the names suggest:
#
# * `divide(a, b)` is the `//` operator — integer division truncating **toward zero**
#   (`divide(-1, 3)` is 0, `divide(-7, 2)` is -3). Lowering it to `floordiv` rounded the
#   other way on every negative quotient, and returned NaN rather than NULL on `0.0/0.0`.
# * `fdiv(a, b)` is **floor** division returning a DOUBLE (`fdiv(-1, 3)` is -1.0). It is
#   not true division: mapping it to `/` answered -0.333 where DuckDB answers -1. It is
#   also not `//`: a zero divisor is IEEE there (`fdiv(1.0, 0)` is `inf`, `fdiv(0, 0)` is
#   `NaN`) where `//` and `divide()` return NULL, so it is built from the float division
#   rather than from the `floor_div` opcode.
_ARITH = {
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "truncdiv",
    "fdiv": "floordiv_double",
    "mod": "mod",
}


def _floored_mod(a: Expr, b: Expr) -> Expr:
    """DuckDB `fmod`: the remainder carrying the *divisor's* sign, as a double.

    Written as `a - b * floor(a / b)` rather than as a dedicated opcode because that
    expression already reproduces every edge DuckDB has: a zero divisor makes `a / b`
    infinite and `b * floor(...)` a NaN, which is what DuckDB returns there; a null
    operand propagates through the arithmetic; and the double division fixes the result
    type for integer operands, which DuckDB also widens to DOUBLE.
    """
    left, right = a.cast("float64"), b.cast("float64")
    return left - right * (left / right).floor()


# `f(v)` → a one-argument top-level builder. The lakehouse partition transforms are
# reachable from SQL under the same names the DataFrame API gives them, so a query can
# group or filter by the value a table is partitioned by without leaving SQL. Iceberg's
# own one-word spellings (`years`, `days`) are deliberately *not* aliased here: DuckDB
# already gives `days(5)` a different meaning (an INTERVAL), and quietly redefining it
# would change the answer of a query that never mentioned partitioning.
def _signbit(value: Expr) -> Expr:
    """DuckDB `signbit(x)` — is the value's IEEE sign bit set?

    True for every negative value **and for `-0.0`**, which is the whole reason this is
    not `x < 0`. It is also not `1/x < 0` on its own: that reads `-0.0` correctly but
    reads `-inf` as false, because `1 / -inf` is `-0.0` and `-0.0 < 0` is false. The
    disjunction is what covers both ends, and NaN stays false in every term, which is
    DuckDB's answer for it.
    """
    x = value.cast("float64")
    return (x < lit(0.0)) | ((lit(1.0) / x) < lit(0.0))


_UNARY_FN = {
    "signbit": _signbit,
    "partition_years": partition_years,
    "partition_months": partition_months,
    "partition_days": partition_days,
    "partition_hours": partition_hours,
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
    # `fmod` is *not* a spelling of `mod`, despite the name. DuckDB's takes the sign of
    # the divisor (`fmod(-2.25, 4) = 1.75`) where `mod` takes the sign of the dividend
    # (`-2.25`), and it returns NaN on a zero divisor where `mod` returns null. Mapping
    # it onto `mod` agreed with DuckDB only when both operands shared a sign.
    "fmod": _floored_mod,
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
    "e": lambda: lit(math.e),
    "today": current_date,
    # The lambda placeholder: `transform(xs, x -> x + 1)` rewrites its parameter to
    # `element()` before translating the body, and this row is what resolves it.
    "element": element,
}

# `f(list, i)` → 1-based element access. DuckDB's `list_extract`/`array_extract`/
# `list_element` are the same function under three names.
_ELEMENT_AT = frozenset(
    {"list_extract", "array_extract", "list_element", "array_element", "element_at"}
)

# `f(struct)` → a unary `.struct` method. A struct's keys come from its *type*, so the
# same list is repeated on every row — and a null struct row still answers null, which is
# what keeps this from being a constant.
_UNARY_STRUCT = {
    "struct_keys": "keys",
}

# `f(list)` → a unary `.list` method. These are DuckDB names whose `array_` twin does not
# exist, so they never reached the `array_`/`list_` alias sweep that found the rest.
_UNARY_LIST = {
    "list_first": "first",
    "list_last": "last",
    "list_median": "median",
}

# `f(list, value)` → a `.list` method taking one *literal* value (not an expression):
# `list_position(l, 3)` is the 1-based index of `3`, and `.list.position` already returns
# it 1-based, so unlike `list_extract` there is no origin to shift.
_LIST_VALUE = {
    "list_position": "position",
    "array_position": "position",
    "list_indexof": "position",
    "array_indexof": "position",
}

# `f(a, b)` on two list expressions → a binary `.list` method.
_LIST_PAIR = {
    "list_intersect": "intersect",
    "array_intersect": "intersect",
    "list_concat": "concat",
    "array_concat": "concat",
    "list_cat": "concat",
    "array_cat": "concat",
    "list_has_all": "has_all",
    "array_has_all": "has_all",
    "list_has_any": "has_any",
    "array_has_any": "has_any",
    "list_difference": "difference",
    "array_difference": "difference",
    "list_union": "union",
    "array_union": "union",
    # DuckDB gives the fixed-size `ARRAY` type its own spelling of each vector metric;
    # the kernel is the same, so the `array_` names are aliases rather than a second
    # implementation.
    "array_cosine_similarity": "cosine_similarity",
    "array_cosine_distance": "cosine_distance",
    "array_distance": "euclidean_distance",
    "array_inner_product": "dot",
    "array_dot_product": "dot",
}

# `f(a, b)` on two list expressions → the negation of a `.list` method. DuckDB's
# `list_negative_dot_product`/`list_negative_inner_product` are the plain products with
# the sign flipped, which is the form a maximum-inner-product search minimizes.
_LIST_PAIR_NEGATED = {
    "list_negative_dot_product": "dot",
    "list_negative_inner_product": "dot",
    "array_negative_dot_product": "dot",
    "array_negative_inner_product": "dot",
}

# `f(l, idx)` → gather by a list of 1-based positions (DuckDB's two spellings).
_LIST_SELECT = frozenset({"list_select", "array_select"})

# `f(v, …)` → a list literal of every argument (DuckDB's list constructors).
_LIST_PACK = frozenset({"list_pack", "list_value", "array_value"})

# `grade_up(l)` — the 1-based positions that sort the list. `.list.arg_sort` is the same
# permutation 0-based, so the whole difference is the index origin.
_GRADE_UP = frozenset({"grade_up", "list_grade_up", "array_grade_up"})


def anonymous_scalar(tr, node):
    """Translate an `exp.Anonymous` scalar call, or return None if the name is unknown.

    Args:
        tr: The translator instance (for recursive `_scalar` calls).
        node: The `exp.Anonymous` node.

    Returns:
        The `Expr` this call denotes, or `None` when the name is not one this
        module serves (the caller then raises its own "unknown function" error).
    """
    from batcher._sql.parser.expressions.literals import _const_int_arg

    name = node.name.lower()
    args = list(node.expressions)

    if not args:
        nullary = _NULLARY.get(name)
        if nullary is not None:
            return nullary()

    if name == "ord" and len(args) == 1:
        # `ord('')` is -1 where `ascii('')` is 0; the kernel implements `ascii`, so the
        # difference is layered on rather than forked into a second kernel.
        from batcher._sql.parser.expressions.functions import _empty_string_is_minus_one

        return _empty_string_is_minus_one(tr._scalar(args[0]))

    if name in _GRADE_UP and len(args) == 1:
        return tr._scalar(args[0]).list.arg_sort().list.transform(element() + lit(1))

    if name == "constant_or_null" and len(args) >= 2:
        # `constant_or_null(v, x, …)` is `v` unless any guard is null, in which case
        # null — DuckDB's way of writing "this constant, but propagate the nulls of the
        # arguments it replaced". Folding the guards with `&` keeps one pass per guard.
        guard = tr._scalar(args[1]).is_not_null()
        for extra in args[2:]:
            guard = guard & tr._scalar(extra).is_not_null()
        value = tr._scalar(args[0])
        return when(guard).then(value).otherwise(nullif(value, value))

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
        if name in _UNARY_LIST:
            return getattr(tr._scalar(one).list, _UNARY_LIST[name])()
        if name in _UNARY_STRUCT:
            return getattr(tr._scalar(one).struct, _UNARY_STRUCT[name])()
        unary = _UNARY_FN.get(name)
        if unary is not None:
            return unary(tr._scalar(one))

    if len(args) == 2:
        left, right = args
        if name in _STR_TEXT:
            from batcher._sql.parser.expressions.lowering.dynamic import str_call

            method = _STR_TEXT[name]
            return str_call(tr, _STR_TEXT_TAG.get(method, method), left, pattern=right)
        if name in _ARITH:
            return _arith(tr, name, left, right)
        builder = _BINARY_FN.get(name)
        if builder is not None:
            return builder(tr._scalar(left), tr._scalar(right))
        if name in _ELEMENT_AT:
            from batcher._sql.parser.expressions.lowering.dynamic import const_int

            index = const_int(right)
            if index is None:
                return _element_at_dyn(tr._scalar(left), tr._scalar(right))
            return _element_at(tr, left, index)
        if name in _MAP_KEY:
            return getattr(tr._scalar(left).map, _MAP_KEY[name])(_raw_literal(right))
        if name in _LIST_VALUE:
            return getattr(tr._scalar(left).list, _LIST_VALUE[name])(_raw_literal(right))
        if name in _LIST_PAIR:
            return getattr(tr._scalar(left).list, _LIST_PAIR[name])(tr._scalar(right))
        if name in _LIST_SELECT:
            # `list_select(l, idx)` gathers by a *list* of 1-based positions, repeats and
            # reorderings included. `.list.gather` is the same operation 0-based, so the
            # whole difference is the index origin — shifted inside the index list rather
            # than at its call site, because the list is a column, not a constant.
            return tr._scalar(left).list.gather(
                tr._scalar(right).list.transform(element() - lit(1))
            )
        if name == "partition_truncate":
            return _partition_truncate(tr, left, right)
        negated = _LIST_PAIR_NEGATED.get(name)
        if negated is not None:
            return lit(0.0) - getattr(tr._scalar(left).list, negated)(tr._scalar(right))

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


def _element_at_dyn(value: Expr, index: Expr) -> Expr:
    """`list_extract(l, i)` with `i` computed per row.

    The constant form folds SQL's 1-based, 0-is-out-of-range, negatives-from-the-end rule
    into one integer at plan time. Per row that folding becomes a `CASE`: index 0 names no
    element, a positive index shifts down to the 0-based accessor, and a negative one is
    already what the accessor means.

    Args:
        value: The list expression.
        index: The 1-based index expression.

    Returns:
        The element expression.
    """
    from batcher.plan.expr_ir import ListGet, ListGetDyn

    zero_based = when(index > lit(0)).then(index - lit(1)).otherwise(index)
    element = ListGetDyn(value, zero_based)
    # Index 0 names no element. `nullif(x, x)` is a NULL of the element's own type.
    empty = ListGet(value, 0)
    return when(index == lit(0)).then(nullif(empty, empty)).otherwise(element)


def _element_at(tr, value, index: int) -> Expr:
    """`list_extract(l, i)` — DuckDB's 1-based element access, negatives from the end.

    Blanket-subtracting one from the written index (the old rule) is right only for a
    positive one. Index **0** is not the last element, it is out of range and NULL; and a
    **negative** index already counts from the end, so shifting it moved every such call
    one place too far — `list_extract(l, -1)` answered the second-to-last value.

    Args:
        tr: The translator.
        value: The list argument node.
        index: The constant index as written.

    Returns:
        The element expression, or a NULL of the element's type for index 0.
    """
    element = tr._scalar(value).list.get(0)
    if index == 0:
        # SQL indexes from 1, so 0 names no element. `nullif(x, x)` is a NULL of the
        # element's own type, which keeps the column's type right.
        return nullif(element, element)
    return tr._scalar(value).list.get(index - 1 if index > 0 else index)


def _raw_literal(node) -> object:
    """The Python value a constant argument denotes — map lookups take a plan-time key."""
    from sqlglot import expressions as exp

    if not isinstance(node, exp.Literal):
        raise NotImplementedError("a map key must be a constant")
    if node.is_string:
        return node.name
    text = node.name
    return float(text) if ("." in text or "e" in text.lower()) else int(text)


def _arith(tr, name: str, left, right) -> Expr:
    """Build the arithmetic operator `name` over two translated operands.

    `floordiv`/`truediv`/`mod` are `Expr` methods rather than `Binary` opcodes (they
    lower to the engine's own division semantics, including its divide-by-zero rule),
    so they are dispatched by attribute; `add`/`sub`/`mul` are plain binary nodes.
    """
    from batcher._sql.parser.expressions.literals import _sql_int_div

    op = _ARITH[name]
    lhs, rhs = tr._scalar(left), tr._scalar(right)
    if op == "truncdiv":
        return _sql_int_div(tr, lhs, rhs)
    if op == "floordiv_double":
        return (lhs.cast("float64") / rhs.cast("float64")).floor()
    if op in ("floordiv", "truediv", "mod"):
        return getattr(lhs, op)(rhs)
    return Binary(op, lhs, rhs)


def _partition_truncate(tr, value, width):
    """`partition_truncate(v, W)` — Iceberg's transform, with its two readings separated.

    On text the transform is the first `W` characters; on a number it is the largest
    multiple of `W` at or below the value. They are different operations, and only the
    argument's type says which one a call means — which is why `plan.functions.partitioning`
    exposes the numeric one alone and leaves the text one to `.str.substr`. Here the type
    *is* in scope, so both readings are reachable under the one name a user would write.

    A column whose type the plan cannot state takes the numeric reading, matching the
    DataFrame surface rather than guessing from the literal's shape.
    """
    import pyarrow as pa

    from batcher._sql.parser.expressions.literals import _const_int_arg

    chars = _const_int_arg(width, "partition_truncate(): width")
    column = tr.column_type(value)
    if column is not None and (pa.types.is_string(column) or pa.types.is_large_string(column)):
        return tr._scalar(value).str.substr(1, chars)
    return partition_truncate(tr._scalar(value), chars)
