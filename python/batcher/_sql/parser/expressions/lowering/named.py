"""Named SQL arguments (``name => value``): consume them, or refuse them by name.

sqlglot parses ``round(x, 0, mode => 'half_to_even')`` into a typed `Round` node and files
the named argument in whatever slot comes next, and no handler read it there. The call then
answered with the default tie rule: 2.5 rounded to 3.0 where the query asked for 2.0. A
wrong answer with nothing to show for it, and the same held for any made-up name.

So the rule is total. A named argument is either **consumed** by a handler that gives it a
meaning, which marks it, or the call is refused with the argument named. The marking is a
flag in the node's `meta` so it needs no bookkeeping in the translator; the check runs once
per translated node, in `_Translator._scalar`, after the handler has built the expression.

Two kinds of handler consume one. `round` takes ``mode =>``, the keyword-only parameter
`Expr.round` has. The two *derived* dispatches (`families`, `accessors`) map a named argument
onto the Python parameter of the same name, so every keyword the public surface has is
spelled the same way in SQL. A named argument a typed builder dropped outright never reaches
this module; the parser refuses it (`batcher._internal.sql_errors`).
"""

from __future__ import annotations

import inspect
import typing
from typing import Any

from sqlglot import expressions as exp

from batcher._internal.sql_errors import unsupported
from batcher._sql.parser.expressions.lowering.dynamic import const_int, const_str
from batcher._sql.parser.expressions.lowering.signatures import (
    LITERAL_READERS,
    annotation_kind,
    hints,
    read_argument,
)
from batcher.plan.expr_ir import Expr

__all__ = [
    "consume",
    "keyword_arguments",
    "misplaced",
    "named_arguments",
    "refuse_unconsumed",
    "round_with_mode",
    "split_keywords",
]

#: The `meta` flag a handler sets on a named argument it gave a meaning to.
_CONSUMED = "bc_consumed"


def named_arguments(node) -> list:
    """The ``name => value`` arguments sitting directly in `node`'s argument slots."""
    found = []
    for value in node.args.values():
        for item in value if isinstance(value, list) else (value,):
            if isinstance(item, exp.Kwarg):
                found.append(item)
    return found


def consume(kwarg) -> None:
    """Mark `kwarg` as given a meaning, so `refuse_unconsumed` lets its call through."""
    kwarg.meta[_CONSUMED] = True


def refuse_unconsumed(node) -> None:
    """Raise when `node` carries a named argument no handler consumed.

    Args:
        node: The sqlglot node just translated.

    Raises:
        SQLUnsupportedError: Naming the first ignored argument.
    """
    if not isinstance(node, exp.Func):
        return
    for kwarg in named_arguments(node):
        if not kwarg.meta.get(_CONSUMED):
            raise _refusal(node, kwarg)


def misplaced(kwarg) -> Exception:
    """The error for a named argument a handler tried to read as an ordinary value.

    Args:
        kwarg: The `Kwarg` node.

    Returns:
        The `SQLUnsupportedError` to raise.
    """
    return _refusal(kwarg.parent, kwarg)


def _refusal(call, kwarg) -> Exception:
    name = _written(call) if call is not None else "this function"
    return unsupported(
        f"{name}() does not take a named argument {kwarg.this.name!r}; Batcher refuses a "
        f"named argument it cannot honour rather than ignore it. Pass the argument by "
        f"position, or drop it",
        call,
    )


def _written(node) -> str:
    """How a call was written: its name, or the sqlglot node type for a typed call."""
    if isinstance(node, exp.Anonymous):
        return str(node.name)
    return node.sql_name().lower() if isinstance(node, exp.Func) else type(node).__name__


def split_keywords(args: list) -> tuple[list, list]:
    """Separate a call's argument nodes into positional ones and ``name => value`` ones."""
    positional = [a for a in args if not isinstance(a, exp.Kwarg)]
    return positional, [a for a in args if isinstance(a, exp.Kwarg)]


def round_with_mode(tr, node) -> Expr | None:
    """``round(x[, digits], mode => '...')``, or None when `node` is not that call.

    `mode` is `Expr.round`'s tie rule, and it takes the same values: ``'half_away_from_zero'``
    (the default, DuckDB's ``round``) or ``'half_to_even'``. Any other value raises the
    `PlanError` `Expr.round` raises, rather than a SQL-only spelling of the same rule.

    Args:
        tr: The translator.
        node: The sqlglot node.

    Returns:
        The rounded expression, or None when `node` is not a `Round` with a named argument.
    """
    if not isinstance(node, exp.Round):
        return None
    named = named_arguments(node)
    if not named:
        return None
    for kwarg in named:
        if kwarg.this.name.lower() != "mode":
            raise _refusal(node, kwarg)
    if len(named) > 1:
        raise unsupported("round() takes `mode =>` once", node)
    mode = const_str(named[0].expression)
    if mode is None:
        raise unsupported("round(..., mode => ...) takes a constant string", node)
    decimals = node.args.get("decimals")
    digits = None
    if decimals is not None and not isinstance(decimals, exp.Kwarg):
        digits = const_int(decimals)
        if digits is None:
            raise unsupported(
                "round(x, digits, mode => ...) takes a constant digit count; a per-row digit "
                "count rounds with the default tie rule only",
                node,
            )
    consume(named[0])
    return tr._scalar(node.this).round(digits, mode=mode)


def keyword_arguments(
    tr, fn, keywords: list, written: str, *, skip_first: bool = False, filled: int = 0
) -> dict[str, Any]:
    """Map ``name => value`` arguments onto `fn`'s parameters of the same name.

    A parameter that takes a column lowers its value as an expression; one that takes a
    plan-time constant reads the literal, exactly as a positional argument would.

    Args:
        tr: The translator, for lowering a column argument.
        fn: The Python function being called.
        keywords: The call's `Kwarg` nodes.
        written: How the call was written, for the error messages.
        skip_first: `fn`'s first parameter is a bound `self`.
        filled: How many positional parameters the call already filled by position.

    Returns:
        Parameter name to value, ready to pass as ``**kwargs``.

    Raises:
        SQLUnsupportedError: For a name `fn` has no parameter for, a parameter given twice,
            or one whose type SQL cannot spell.
    """
    parameters = list(inspect.signature(fn).parameters.values())[1 if skip_first else 0 :]
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    named = {
        p.name.lower(): p for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    annotations = hints(fn)
    out: dict[str, Any] = {}
    for kwarg in keywords:
        parameter = named.get(kwarg.this.name.lower())
        if parameter is None:
            raise unsupported(
                f"{written}() has no parameter named {kwarg.this.name!r}; its named "
                f"parameters are {sorted(named)}",
                kwarg.parent,
            )
        if parameter.name in out or (
            parameter in positional and positional.index(parameter) < filled
        ):
            raise unsupported(f"{written}() was given {parameter.name!r} twice", kwarg.parent)
        kind = annotation_kind(_choice_type(annotations.get(parameter.name)))
        if kind is Expr:
            value = tr._scalar(kwarg.expression)
        elif kind in LITERAL_READERS:
            value = read_argument(kind, kwarg.expression, written)
        else:
            raise unsupported(
                f"{written}()'s parameter {parameter.name!r} takes a Python value SQL cannot "
                f"spell; use the DataFrame API for it",
                kwarg.parent,
            )
        consume(kwarg)
        out[parameter.name] = value
    return out


def _choice_type(annotation: Any) -> Any:
    """The value type of a ``Literal["a", "b"]`` choice, else `annotation` unchanged.

    A keyword such as ``mode: Literal["fast", "exact"]`` takes a constant of that type.
    """
    if typing.get_origin(annotation) is not typing.Literal:
        return annotation
    values = {type(v) for v in typing.get_args(annotation)}
    return values.pop() if len(values) == 1 else None
