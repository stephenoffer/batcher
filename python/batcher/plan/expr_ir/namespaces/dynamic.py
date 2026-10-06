"""Build a string function whose parameters may be columns rather than constants.

The engine's string kernels take ``pattern``/``replacement``/``start``/``length`` as
plan-time constants (`StrFunc`). Nothing about the functions requires that: a per-row
delimiter, offset or comparison string is ordinary, in SQL (``replace(s, old_col, new_col)``)
and in a DataFrame (``col("a").str.levenshtein(col("b"))``) alike. `StrFuncDyn` is the same
function with its parameters as sub-expressions; the engine groups rows by their distinct
parameter tuple and calls the *same* kernel per group, so there is one definition of each
function's semantics.

:func:`str_call` is the one place that chooses between the two nodes. The SQL translator and
the `.str` namespace both build through it, so a call site states the function once and does
not have to know which form it will get. Neutral (layer 1): it knows about expression nodes,
never about a parser.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr, Lit
from batcher.plan.expr_ir.func_nodes import StrFunc, StrFuncDyn

__all__ = ["StrParam", "str_call"]

#: A `StrFunc` parameter as a caller may give it: a plan-time value, an expression, or
#: absent.
StrParam = str | int | Expr | None

#: The slots that take text; every other slot (`start`, `length`) takes an integer.
_TEXT_SLOTS = frozenset({"pattern", "replacement"})


def _constant(slot: str, value: Expr) -> str | int | None:
    """The plan-time value a literal expression denotes in `slot`, or None if it is not one.

    ``lit("x")`` passed where a delimiter is expected *is* a constant; folding it keeps the
    plan on the constant kernel instead of grouping rows by a parameter that never varies.
    """
    if not isinstance(value, Lit):
        return None
    v = value.value
    if slot in _TEXT_SLOTS:
        return v if isinstance(v, str) else None
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def str_call(fn: str, subject: Expr, **params: StrParam) -> Expr:
    """Build the string function `fn` over `subject`, per row where a parameter is not constant.

    Args:
        fn: The engine's string-function tag (`STR_FNS`).
        subject: The string argument.
        **params: Any of ``pattern``/``replacement``/``start``/``length``, each a plan-time
            ``str``/``int``, an `Expr`, or None (absent).

    Returns:
        A `StrFunc` when every supplied parameter is a plan-time constant, else a
        `StrFuncDyn` carrying each parameter as an expression.
    """
    consts: dict[str, str | int] = {}
    dynamic: dict[str, Expr] = {}
    for slot, value in params.items():
        if value is None:
            continue
        if isinstance(value, Expr):
            folded = _constant(slot, value)
            if folded is None:
                dynamic[slot] = value
            else:
                consts[slot] = folded
            continue
        consts[slot] = value
    if not dynamic:
        return StrFunc(fn, subject, **consts)
    # A mixed call lifts its constants to literals so every slot is an expression.
    args: dict[str, Expr] = {k: Lit(v) for k, v in consts.items()}
    args.update(dynamic)
    return StrFuncDyn(fn, subject, **args)
