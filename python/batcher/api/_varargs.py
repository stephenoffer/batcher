"""Sequence flattening for the `Dataset` verbs' varargs positions.

Every mature DataFrame API accepts a *list* of columns where its signature takes
varargs: Polars ``df.select(["a", "b"])``, PySpark ``df.orderBy(["a", "b"])``,
Ray Data ``ds.sort(["a"])``, pandas ``df.sort_values(by=["a", "b"])``. Batcher
spells those verbs ``*columns``, so without this the list arrived as a *single*
key and every one of them raised — ``sort(["a", "b"])`` reported "expected a
column name or expression, got list". That is a migration blocker on the first
verb of a ported script, and it was true of eleven verbs at once.

One helper applied at each varargs entry point is what keeps the parity from being
re-derived, and re-forgotten, per verb.

**Why this is safe rather than a widening of meaning.** A `list`/`tuple`/generator
is not a valid argument in any of these positions today — every one of them raises.
So flattening can only turn an error into the result the caller meant; it cannot
change the meaning of a call that works now. The one varargs verb where a sequence
*is* already meaningful, ``grouping_sets(*sets)``, deliberately does not use this.
"""

from __future__ import annotations

from types import GeneratorType
from typing import Any

__all__ = ["flatten_varargs"]

# Flattened by *type*, never by "is it iterable": `Expr`, `Selector` and `Dataset`
# all define `__iter__` (an expression's raises, a dataset's executes the plan), so
# a duck-typed check would either explode or silently run a query.
_SEQUENCE_TYPES = (list, tuple, GeneratorType)


def flatten_varargs(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Expand any list, tuple, or generator argument into the surrounding varargs.

    Flattens exactly one level, so ``f(["a", "b"])``, ``f("a", ["b", "c"])`` and
    ``f(col(c) for c in names)`` all reach the verb as the same flat argument tuple
    that ``f("a", "b")`` does.

    Args:
        args: The varargs tuple exactly as the verb received it.

    Returns:
        The tuple with one level of list/tuple/generator arguments expanded. The
        input tuple itself is returned when there is nothing to expand.

    Examples:
        .. doctest::

            >>> from batcher.api._varargs import flatten_varargs
            >>> flatten_varargs((["a", "b"],))
            ('a', 'b')

            >>> flatten_varargs(("a", ["b", "c"]))
            ('a', 'b', 'c')

            >>> flatten_varargs(("a", "b"))
            ('a', 'b')
    """
    if not any(isinstance(a, _SEQUENCE_TYPES) for a in args):
        return args
    out: list[Any] = []
    for a in args:
        if isinstance(a, _SEQUENCE_TYPES):
            out.extend(a)
        else:
            out.append(a)
    return tuple(out)
