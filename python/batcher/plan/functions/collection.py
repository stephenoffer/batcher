"""Collection-construction free functions (`struct`, `named_struct`, `sequence`).

`struct`/`named_struct` build a `MakeStruct` node — the construction counterpart of
the `.struct.field` read accessor; `sequence` builds a per-row integer list. `struct`
takes ``name=expr`` keywords (Pythonic); `named_struct` takes alternating name/value
positional arguments (SQL ``named_struct``).
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr, IntoExpr, _wrap
from batcher.plan.expr_ir.nodes import Col, MakeStruct, Sequence

#: The reserved column name the list higher-order ops bind each element to. Must
#: match the Rust `eval/list_hof.rs` ELEMENT constant.
_ELEMENT_COL = "element"


def struct(**fields: IntoExpr) -> Expr:
    """Build a struct column from ``name=expr`` fields (Spark ``struct``).

    ``struct(x=col("a"), y=col("b") + 1)`` produces a struct ``{x, y}`` per row; read
    a field back with ``col("s").struct.field("x")``. Requires at least one field.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [2]})
            >>> ds.select(s=bt.struct(x=bt.col("a"), y=bt.col("b"))).to_pydict()
            {'s': [{'x': 1, 'y': 2}]}
    """
    if not fields:
        raise PlanError("struct() requires at least one field")
    return MakeStruct([(name, _wrap(value)) for name, value in fields.items()])


def named_struct(*args: object) -> Expr:
    """Build a struct from alternating ``name, value`` arguments (SQL ``named_struct``).

    ``named_struct("x", col("a"), "y", col("b"))`` is equivalent to
    ``struct(x=col("a"), y=col("b"))``. Field names must be strings.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [2]})
            >>> ds.select(s=bt.named_struct("x", bt.col("a"), "y", bt.col("b"))).to_pydict()
            {'s': [{'x': 1, 'y': 2}]}
    """
    if not args or len(args) % 2 != 0:
        raise PlanError("named_struct() requires an even number of name, value arguments")
    fields: list[tuple[str, Expr]] = []
    for i in range(0, len(args), 2):
        name = args[i]
        if not isinstance(name, str):
            raise PlanError(f"named_struct field name must be a string, got {name!r}")
        fields.append((name, _wrap(args[i + 1])))  # type: ignore[arg-type]
    return MakeStruct(fields)


def sequence(start: IntoExpr, stop: IntoExpr, step: IntoExpr = 1) -> Expr:
    """Build a per-row integer list from ``start`` to ``stop`` inclusive (Spark ``sequence``).

    ``sequence(1, 5)`` yields ``[1, 2, 3, 4, 5]``; ``sequence(col("a"), col("b"), 2)``
    steps by 2. The bounds and step may be columns or literals (cast to Int64); a null
    argument yields a null list, and a ``step`` of 0 raises. Pair with ``explode`` to
    fan a range out into rows.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [4]})
            >>> ds.select(s=bt.sequence(bt.col("a"), bt.col("b"))).to_pydict()
            {'s': [[1, 2, 3, 4]]}
    """
    return Sequence(_wrap(start), _wrap(stop), _wrap(step))


def element() -> Expr:
    """The current element inside ``list.transform`` / ``list.filter`` (Polars
    ``element()``).

    Use it to build the per-element expression: ``col("a").list.transform(element() * 2)``
    doubles each element, ``col("a").list.filter(element() > 0)`` keeps the positives.
    Outside a list higher-order op it has no binding.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [[1, 2, 3]]})
            >>> ds.select(d=bt.col("a").list.transform(bt.element() * 2)).to_pydict()
            {'d': [[2, 4, 6]]}
    """
    return Col(_ELEMENT_COL)
