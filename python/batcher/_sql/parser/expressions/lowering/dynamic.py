"""String functions whose parameters are columns rather than constants.

The engine's string kernels take ``pattern``/``replacement``/``start``/``length`` as
plan-time constants, and the `.str` namespace is typed that way. SQL does not require
them to be constant: ``replace(s, old_col, new_col)``, ``substr(s, from_col, len_col)``,
``s LIKE pattern_col`` and roughly thirty other spellings are ordinary, and every one of
them used to be refused with *"requires a constant string pattern"*.

`StrFuncDyn` is the same function with its parameters as sub-expressions; the engine
groups rows by their distinct parameter tuple and calls the *same* kernel per group, so
there is one definition of each function's semantics. Which of the two nodes to build is
decided in one neutral place, `batcher.plan.expr_ir.namespaces.dynamic.str_call`, which the
`.str` namespace builds through too. This module is only the translator's side of that:
:func:`str_call` reads a sqlglot literal as a constant and lowers anything else, then
hands the parameters over.
"""

from __future__ import annotations

from typing import Any

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Expr, StrFunc, StrFuncDyn, lit, when
from batcher.plan.expr_ir.namespaces import dynamic

__all__ = ["const_bool", "const_float", "const_int", "const_str", "str_call"]

#: The `StrFunc` slots that take text; every other slot (`start`, `length`) takes an integer.
_TEXT_SLOTS = ("pattern", "replacement")


def const_str(node) -> str | None:
    """The Python string a node denotes, or None when it is not a string literal."""
    if isinstance(node, exp.Literal) and node.is_string:
        return node.this
    return None


def const_int(node) -> int | None:
    """The Python int a node denotes, or None when it is not an integer literal.

    A negative literal parses as `exp.Neg` wrapping the magnitude, so the sign is folded
    here rather than left to a caller that would read straight through to the child.
    """
    if isinstance(node, exp.Neg):
        inner = const_int(node.this)
        return None if inner is None else -inner
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return int(node.this)
        except ValueError:
            return None
    return None


def const_float(node) -> float | None:
    """The Python float a node denotes, or None when it is not a numeric literal.

    Sibling of `const_int`, and negative literals fold the same way. An integer literal
    reads as a float: SQL's `2` is a perfectly good argument to a parameter typed `float`,
    and refusing it would make `image_adjust_brightness(i, 2)` an error next to
    `image_adjust_brightness(i, 2.0)` working.
    """
    if isinstance(node, exp.Neg):
        inner = const_float(node.this)
        return None if inner is None else -inner
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return float(node.this)
        except ValueError:
            return None
    return None


def const_bool(node) -> bool | None:
    """The Python bool a node denotes, or None when it is not a boolean literal."""
    return node.this if isinstance(node, exp.Boolean) else None


def str_call(tr, fn: str, value, **params: Any) -> Expr:
    """Build the string function `fn` over `value`, per-row where a parameter is not constant.

    Args:
        tr: The translator, for lowering a non-constant parameter.
        fn: The engine's string-function tag (`STR_FNS`).
        value: The string argument — a sqlglot node or an already-built `Expr`.
        **params: Any of ``pattern``/``replacement``/``start``/``length``, each a sqlglot
            node, an already-constant Python value, an already-lowered `Expr`, or None.

    Returns:
        A `StrFunc` when every supplied parameter is a plan-time constant, else a
        `StrFuncDyn` carrying each parameter as an expression.
    """
    subject = value if isinstance(value, Expr) else tr._scalar(value)
    lowered: dict[str, dynamic.StrParam] = {}
    for slot, node in params.items():
        # Absent, already a constant, or already lowered (a caller that had to build the
        # parameter itself, such as a regex pattern carrying an inline flag prefix).
        if node is None or (isinstance(node, (str, int, Expr)) and not isinstance(node, bool)):
            lowered[slot] = node
            continue
        constant = const_str(node) if slot in _TEXT_SLOTS else const_int(node)
        lowered[slot] = constant if constant is not None else tr._scalar(node)
    return dynamic.str_call(fn, subject, **lowered)


def dynamic_left(tr, value, count) -> Expr:
    """``left(s, n)`` with a non-constant `n`.

    There is no `left` kernel: the constant path composes it as ``substr(s, 1, n)``, and
    for a negative `n` as "all but the last |n| characters". Which of the two applies is a
    per-row question once `n` is a column, so both are built and a `CASE` picks.

    Args:
        tr: The translator.
        value: The string argument node.
        count: The length node.

    Returns:
        The expression for `left`.
    """
    subject = tr._scalar(value)
    n = tr._scalar(count)
    from_start = StrFuncDyn("substr", subject, start=lit(1), length=n)
    without_tail = StrFuncDyn("substr", subject, start=lit(1), length=StrFunc("len", subject) + n)
    return when(n >= lit(0)).then(from_start).otherwise(without_tail)
