"""Turning a wrong-typed user argument into a typed error, at the API edge.

Expression builders lower their scalar arguments straight into the JSON IR, so an argument
of the wrong type is not caught until the Rust deserializer refuses it -- and what the user
then sees is ``malformed plan IR: invalid type: string "3", expected i64``, which names the
engine's wire format rather than the call they got wrong. Builders that instead compare the
argument while building (``if n < 1``) fail even less helpfully, with a bare ``TypeError:
'<' not supported between instances of 'str' and 'int'``.

`require_int` is the one place that turns both into a `PlanError` naming the method and the
parameter, before a plan is ever built.

This lives in layer 0 rather than beside its callers because those callers span
`plan.expr_ir.core`, the accessor namespaces, and the compat shims -- and `core` is the
module every one of the others imports, so any home inside `expr_ir` would be a cycle.
"""

from __future__ import annotations

import operator
from typing import Any

from batcher._internal.errors.hierarchy import PlanError

__all__ = ["require_float", "require_int"]


def require_int(value: Any, *, func: str, arg: str, minimum: int | None = None) -> int:
    """Coerce `value` to an `int` for lowering into the IR, or raise `PlanError`.

    Anything implementing ``__index__`` is accepted, so a NumPy integer works where a
    Python one does. `bool` is rejected despite being an `int` subclass: ``str.repeat(True)``
    would otherwise silently mean ``repeat(1)``, which is a wrong answer rather than an error.

    Args:
        value: The argument as the caller passed it.
        func: Dotted method name for the message, such as ``"str.lpad"``.
        arg: Parameter name for the message.
        minimum: If given, the smallest value the parameter accepts.

    Returns:
        `value` as a plain `int`.

    Raises:
        PlanError: If `value` is not an integer, or is below `minimum`.
    """
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise PlanError(f"{func}(): {arg} must be an integer, got {type(value).__name__} {value!r}")
    number = operator.index(value)
    if minimum is not None and number < minimum:
        raise PlanError(f"{func}(): {arg} must be >= {minimum}, got {number}")
    return number


def require_float(value: Any, *, func: str, arg: str) -> float:
    """Coerce `value` to a `float` for lowering into the IR, or raise `PlanError`.

    Only the *type* is checked here. Callers keep their own domain checks (``0 <= q <= 1``)
    because those messages name the real constraint; what they could not do is run at all
    when handed a string, because ``0.0 <= "abc"`` raises `TypeError` before the check has
    a chance to say anything useful.

    Anything implementing ``__float__`` is accepted, so a NumPy float or a `Decimal` works.
    `bool` is rejected for the same reason `require_int` rejects it: ``sample_frac(True)``
    would silently mean "all rows" rather than being the mistake it is.

    Args:
        value: The argument as the caller passed it.
        func: Dotted method name for the message, such as ``"quantile"``.
        arg: Parameter name for the message.

    Returns:
        `value` as a plain `float`.

    Raises:
        PlanError: If `value` is not a number.
    """
    if isinstance(value, bool) or not hasattr(value, "__float__"):
        raise PlanError(f"{func}(): {arg} must be a number, got {type(value).__name__} {value!r}")
    return float(value)
