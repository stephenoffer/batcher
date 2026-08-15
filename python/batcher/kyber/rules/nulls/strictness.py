"""Push `IS NULL` / `IS NOT NULL` through the scalar functions that are null-strict.

A function is *null-strict and total* when its result is null exactly when one of its
arguments is null: a null in gives a null out, and a non-null in can never give a null
out (and can never raise). For such an `f`, `f(x) IS NULL` and `x IS NULL` are the same
predicate on every row, so the call is pure overhead inside a null test.

Rewriting the test onto the bare column is worth much more than the saved kernel pass.
A null check on a column is what predicate pushdown can move below a join, what the
zonemap pruner can answer from a file's null count without reading the file, and what
`push_is_not_null_from_join_key` needs to see in order to reject null keys early.
`upper(name) IS NOT NULL` is opaque to all three; `name IS NOT NULL` is visible to all
three.

Totality is the guard that does the real work, and it is why every family here is an
explicit vocabulary rather than "any function node":

* `factorial` **raises** on a negative argument, so dropping it would delete an error
  the query is entitled to. It is excluded from `STRICT_MATH_FNS`.
* `bit_count` returns null for a NaN or an infinity, so it is null-strict on an integer
  and not on a float. Rather than carry a type guard for one function, it is excluded.
* `list.sum` / `list.min` / `list.mean` return null for an *empty* list, which is not a
  null input at all — only `len` and `n_unique` are total over the list functions.
* `json_value`, `regexp_extract`, `unhex`, `from_base64` and the rest of the parsing
  family all answer null for a well-formed non-null input that simply does not match.
* `list_simhash` answers null for an **empty** list, which is again not a null input — so
  it is absent while the other two-list operations (`list_add`, `array_union`, `dot` and
  their siblings) are present.
* An `IN` list holding a null makes a *non-null* operand answer null for a value the set
  does not contain, so the test stops tracking the operand's own nullness. The family
  fires only on a set with no null in it.

Each vocabulary was verified against the engine rather than assumed, and a function is
in it only when a null result provably requires a null argument.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.rules.leaf_rewrite import register_leaf_rule
from batcher.plan.expr_ir import Expr, InList, IsNotNull, IsNull, ListJoin, Not
from batcher.plan.expr_ir.core import Binary, IsInf, IsNan, Math2Expr, MathExpr
from batcher.plan.expr_ir.func_nodes import (
    ConvertTimezone,
    DateFunc,
    DateOffset,
    DateTrunc,
    ListBinary,
    ListFunc,
    ListSet,
    ListSlice,
    ListZip,
    Strftime,
    StrFunc,
    WindowBuckets,
    WindowStart,
)
from batcher.plan.ir_tags import COMPARISON_OPS

__all__ = [
    "NULL_STRICTNESS_BINARY_RULES",
    "NULL_STRICTNESS_UNARY_RULES",
    "STRICT_LIST_FNS",
    "STRICT_MATH_FNS",
    "STRICT_STR_FNS",
]

# Plan nodes that carry expressions a null test can appear in.

#: Unary math functions whose result is null exactly when the argument is null.
#: `MATH_FNS` minus `factorial` (raises on a negative argument) and `bit_count`
#: (null for a NaN or an infinity, so strict only on an integer).
STRICT_MATH_FNS = frozenset(
    {
        "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "ceil",
        "cos", "cosh", "cot", "csc", "degrees", "even", "exp", "floor", "gamma",
        "lgamma", "ln", "log10", "log2", "radians", "rint", "round", "sec", "sign",
        "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
    }
)  # fmt: skip

#: String functions whose result is null exactly when the input column is null.
#: The case/trim/pad/hash/length/predicate/similarity families, which all answer for
#: every well-formed string. The parsing and decoding families (`json_*`, `unhex`,
#: `from_base64`, `to_base`, `chr`, `from_binary`, the `aes_*` pair, `compress`) are
#: absent: each returns null for a non-null input it cannot interpret.
STRICT_STR_FNS = frozenset(
    {
        "ascii", "base64", "bit_length", "contains", "crc32", "damerau_levenshtein",
        "ends_with", "hash64", "hex", "ilike", "initcap", "jaccard_similarity",
        "jaro_similarity", "jaro_winkler_similarity", "l_trim", "len", "levenshtein",
        "like", "lower", "lpad", "md5", "octet_length", "position", "r_trim",
        "regexp_count", "regexp_escape", "regexp_matches", "regexp_replace",
        "regexp_replace_all", "repeat", "replace", "reverse", "right", "rpad", "sha1",
        "sha256", "soundex", "split_part", "starts_with", "strip_html",
        "substring_index", "substr", "translate", "trim", "upper", "url_decode",
        "url_encode", "xxhash64",
    }
)  # fmt: skip

#: List reductions that answer for *every* non-null list, the empty one included.
#: The numeric reductions (`sum`/`min`/`max`/`mean`/`median`/`std`/`var`/…) are absent
#: because an empty list has no value to report and they answer null.
STRICT_LIST_FNS = frozenset({"len", "n_unique"})

#: Binary operators that are null-strict: null in either operand nulls the result, and
#: two non-null operands always produce a value. Wrapping add/sub/mul and the six
#: comparisons qualify. `div`/`mod` are absent (a zero divisor aborts the query, so
#: dropping the operation would delete an error), and so are the bitwise operators and
#: the shifts (they coerce through Int64 and a shift count out of range can abort).
_STRICT_ARITH_OPS = frozenset({"add", "sub", "mul"})


def _math_operand(expr: Expr) -> Expr | None:
    if isinstance(expr, MathExpr) and expr.fn in STRICT_MATH_FNS:
        return expr.input
    return None


def _str_operand(expr: Expr) -> Expr | None:
    if isinstance(expr, StrFunc) and expr.fn in STRICT_STR_FNS:
        return expr.input
    return None


def _date_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, DateFunc) else None


def _date_trunc_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, DateTrunc) else None


def _date_offset_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, DateOffset) else None


def _strftime_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, Strftime) else None


def _timezone_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, ConvertTimezone) else None


def _list_operand(expr: Expr) -> Expr | None:
    if isinstance(expr, ListFunc) and expr.fn in STRICT_LIST_FNS:
        return expr.input
    return None


def _not_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, Not) else None


def _nan_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, IsNan) else None


def _inf_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, IsInf) else None


def _in_list_operand(expr: Expr) -> Expr | None:
    # `x IN (…)` is null exactly when `x` is — but only while the set itself holds no
    # null, which would make a non-null `x` answer null for a value it does not contain.
    if isinstance(expr, InList) and all(v is not None for v in expr.values):
        return expr.input
    return None


def _list_slice_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, ListSlice) else None


def _list_join_operand(expr: Expr) -> Expr | None:
    # Total as well as strict: an *empty* list joins to the empty string rather than to
    # null, which is where the list reductions fail the same test.
    return expr.input if isinstance(expr, ListJoin) else None


def _window_start_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, WindowStart) else None


def _window_buckets_operand(expr: Expr) -> Expr | None:
    return expr.input if isinstance(expr, WindowBuckets) else None


#: `(family, extractor)` for every unary node a null test collapses through. The name is
#: what the rule is called (`is_null_through_<family>`); the extractor returns the operand
#: the test moves onto, or ``None`` when the node is not of that family.
_UNARY_FAMILIES: tuple[tuple[str, Callable[[Expr], Expr | None]], ...] = (
    ("math_fn", _math_operand),
    ("str_fn", _str_operand),
    ("date_fn", _date_operand),
    ("date_trunc", _date_trunc_operand),
    ("date_offset", _date_offset_operand),
    ("strftime", _strftime_operand),
    ("convert_timezone", _timezone_operand),
    ("list_reduction", _list_operand),
    ("boolean_negation", _not_operand),
    ("nan_check", _nan_operand),
    ("inf_check", _inf_operand),
    ("list_slice", _list_slice_operand),
    ("list_join", _list_join_operand),
    ("window_start", _window_start_operand),
    ("window_buckets", _window_buckets_operand),
    ("in_list", _in_list_operand),
)


def _unary_leaf(
    operand: Callable[[Expr], Expr | None], *, positive: bool
) -> Callable[[Expr], Expr]:
    """The leaf rewrite moving one null test through a strict unary family.

    `positive` selects the test being moved: ``True`` for `IS NULL`, ``False`` for
    `IS NOT NULL`. Both directions are sound for the same reason and neither can
    re-fire on its own output, because the operand it produces is by construction not
    a node of the family it was extracted from.
    """
    check: type = IsNull if positive else IsNotNull

    def leaf(expr: Expr) -> Expr:
        if isinstance(expr, check):
            inner = operand(expr.input)
            if inner is not None:
                return check(inner)
        return expr

    return leaf


def _register_unary(family: str, operand: Callable[[Expr], Expr | None], *, positive: bool):
    leaf = _unary_leaf(operand, positive=positive)
    name = f"is_null_through_{family}" if positive else f"is_not_null_through_{family}"
    return register_leaf_rule(name, leaf, expr_matches=(IsNull,) if positive else (IsNotNull,))


#: `f(x) IS [NOT] NULL -> x IS [NOT] NULL` for every null-strict, total unary family:
#: the math functions, the string functions, the date-part extractions, `date_trunc`,
#: `offset_by`, `strftime`, `convert_timezone`, the total list reductions, and `NOT`.
#: Two rules per family, one per direction, so each can be enabled, traced, and tested
#: on its own.
NULL_STRICTNESS_UNARY_RULES = [
    _register_unary(family, operand, positive=positive)
    for family, operand in _UNARY_FAMILIES
    for positive in (True, False)
]


#: Sentinels standing in for the binary families selected by *node type* rather than by an
#: operator string: the two-argument math functions, and the three two-list operations
#: (`list_add`/`list_subtract`/…, `array_union`/`array_intersect`/…, `dot`/`cosine`/…).
#: All are null-strict on both operands and total on every pair of non-null lists — the
#: empty list included, which is where `list_simhash` fails the same test and is absent.
_MATH2_MARKER = frozenset({"__math2__"})
_LIST_ZIP_MARKER = frozenset({"__list_zip__"})
_LIST_SET_MARKER = frozenset({"__list_set__"})
_LIST_BINARY_MARKER = frozenset({"__list_binary__"})

_NODE_MARKERS: dict[frozenset[str], type] = {
    _MATH2_MARKER: Math2Expr,
    _LIST_ZIP_MARKER: ListZip,
    _LIST_SET_MARKER: ListSet,
    _LIST_BINARY_MARKER: ListBinary,
}


def _binary_operands(expr: Expr, ops: frozenset[str]) -> tuple[Expr, Expr] | None:
    node_type = _NODE_MARKERS.get(ops)
    if node_type is not None:
        return (expr.left, expr.right) if isinstance(expr, node_type) else None
    if isinstance(expr, Binary) and expr.op in ops:
        return expr.left, expr.right
    return None


def _binary_leaf(ops: frozenset[str], *, positive: bool) -> Callable[[Expr], Expr]:
    """The leaf rewrite splitting one null test across a strict binary node.

    `a ⊕ b IS NULL` becomes `a IS NULL OR b IS NULL`, and `a ⊕ b IS NOT NULL` becomes
    `a IS NOT NULL AND b IS NOT NULL`. The split is what makes the test pushable: a
    conjunction of per-column null tests can be divided between the two sides of a join
    or pushed into separate scans, while the fused test can go nowhere.
    """
    check: type = IsNull if positive else IsNotNull
    connective = "or" if positive else "and"

    def leaf(expr: Expr) -> Expr:
        if isinstance(expr, check):
            pair = _binary_operands(expr.input, ops)
            if pair is not None:
                left, right = pair
                return Binary(connective, check(left), check(right))
        return expr

    return leaf


def _register_binary(family: str, ops: frozenset[str], *, positive: bool):
    leaf = _binary_leaf(ops, positive=positive)
    name = f"is_null_through_{family}" if positive else f"is_not_null_through_{family}"
    return register_leaf_rule(name, leaf, expr_matches=(IsNull,) if positive else (IsNotNull,))


#: `a ⊕ b IS NULL -> a IS NULL OR b IS NULL` (and the `IS NOT NULL` / `AND` dual) for
#: the three strict binary families: wrapping arithmetic, the six comparisons, and the
#: two-argument math functions. Division, modulo, the bitwise operators and the shifts
#: are deliberately absent — each can abort on operands that are perfectly non-null, so
#: the split would drop an error the query would otherwise have raised.
NULL_STRICTNESS_BINARY_RULES = [
    _register_binary(family, ops, positive=positive)
    for family, ops in (
        ("arithmetic", _STRICT_ARITH_OPS),
        ("comparison", COMPARISON_OPS),
        ("math2_fn", _MATH2_MARKER),
        ("list_zip", _LIST_ZIP_MARKER),
        ("list_set_op", _LIST_SET_MARKER),
        ("list_binary", _LIST_BINARY_MARKER),
    )
    for positive in (True, False)
]
