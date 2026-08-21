"""Typing a bare SQL ``NULL`` from the position it is written in.

SQL leaves ``NULL`` untyped and lets the surrounding operator decide; the IR has no untyped
null (a bare ``NULL`` lowers to the Int64 ``nullif(1, 1)``), so the type has to be chosen
here. Getting it wrong is not a wrong answer but a *failed query*: ``NULL OR x`` reached the
engine as ``or(Int64, Bool)`` and ``s = NULL`` on a text column as ``Utf8 == Int64``.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Binary, Cast, Expr, lit, nullif

__all__ = [
    "binop_with_null",
    "null_boolean",
    "positional_null",
    "typed_null",
]


#: sqlglot node types whose (first) operand is text, so a bare ``NULL`` written there is a
#: *string*-typed null rather than the Int64 default. Read from the string-dispatch tables
#: so the two cannot drift, plus the operators and multi-argument forms those tables miss.
def _string_arg_parents() -> frozenset[str]:
    from batcher._sql.parser.expressions.literals import _UNARY_STR

    return frozenset(_UNARY_STR) | {
        "Concat",
        "ConcatWs",
        "DPipe",
        "Lower",
        "Upper",
        "Trim",
        "Substring",
        "Split",
        "SplitPart",
        "Replace",
        "Repeat",
        "Pad",
        "LPad",
        "RPad",
        "Left",
        "Right",
        "StrPosition",
        "Like",
        "ILike",
        "RegexpLike",
        "RegexpReplace",
        "RegexpExtract",
        "StartsWith",
        "EndsWith",
        "Contains",
        "Levenshtein",
        "Initcap",
        "Reverse",
        "Length",
        "Ascii",
        "Chr",
        "MD5",
        "SHA",
        "Hex",
        "Unhex",
    }


_STRING_ARG_PARENTS = _string_arg_parents()


def positional_null(node) -> Expr:
    """A ``NULL`` literal typed by the position it is written in.

    SQL leaves a bare ``NULL`` untyped and lets its context decide; the IR has no untyped
    null, so it has to be given one here. The enclosing node names the family — a string
    function's argument is a text null, everything else keeps the Int64 default, which
    arithmetic and comparison already widen correctly.

    Args:
        node: The `exp.Null` node, read for its parent.

    Returns:
        The typed NULL expression.
    """
    typed = nullif(lit(1), lit(1))
    parent = node.parent
    if parent is not None and type(parent).__name__ in _STRING_ARG_PARENTS:
        return Cast(typed, "string")
    return typed


#: Binary operators whose result is a *boolean* NULL when either operand is a bare NULL.
_NULL_BOOLEAN_OPS = (
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
)


def binop_with_null(tr, node) -> Expr:
    """Translate a binary operator one of whose operands is a bare, untyped ``NULL``.

    SQL leaves ``NULL`` untyped and lets the operator decide; the IR has no untyped null,
    so the type has to be chosen here. A comparison yields a boolean NULL, ``AND``/``OR``
    take a boolean NULL and stay three-valued, and anything else (arithmetic, ``||``)
    yields a NULL of the *other* operand's type, which ``nullif(x, x)`` supplies exactly.

    Args:
        tr: The translator, for recursing into the non-null operand.
        node: The binary node, with `exp.Null` on at least one side.

    Returns:
        The expression for the operator applied to a correctly typed NULL.
    """
    left_null = isinstance(node.this, exp.Null)
    right_null = isinstance(node.expression, exp.Null)
    if isinstance(node, _NULL_BOOLEAN_OPS):
        return null_boolean()
    if isinstance(node, (exp.And, exp.Or)):
        null = null_boolean()
        if left_null and right_null:
            return null
        other = tr._scalar(node.expression if left_null else node.this)
        return (null & other) if isinstance(node, exp.And) else (null | other)
    if left_null and right_null:
        return nullif(lit(1), lit(1))
    # A NULL of the other operand's type: `nullif(x, x)` is NULL for every row and
    # carries `x`'s type, which is what the operator's result type would have been.
    other = tr._scalar(node.expression if left_null else node.this)
    if isinstance(node, exp.DPipe):
        return nullif(Binary("concat", other, lit("")), Binary("concat", other, lit("")))
    return nullif(other, other)


def typed_null(arrow_type) -> Expr:
    """A NULL literal typed to match `arrow_type` (the subquery's output column).

    Built as `NULLIF(1, 1)` (a typed NULL of int) cast to the target type, so the
    output schema matches DuckDB's — a scalar subquery yields a column of its own
    type even when it produces no row.
    """
    import pyarrow as pa

    typed = nullif(lit(1), lit(1))
    if pa.types.is_floating(arrow_type):
        return Cast(typed, "float64")
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return Cast(typed, "string")
    if pa.types.is_boolean(arrow_type):
        return Cast(typed, "bool")
    if pa.types.is_date(arrow_type):
        return Cast(typed, "date")
    if pa.types.is_timestamp(arrow_type):
        return Cast(typed, "timestamp")
    return typed  # integer (and any other) → the int-typed NULL


def null_boolean() -> Expr:
    """A NULL of boolean type. `lit(None)` has no type to give it, so NULLIF supplies one."""
    return nullif(lit(True), lit(True))
