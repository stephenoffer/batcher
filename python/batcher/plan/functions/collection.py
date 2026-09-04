"""Collection-construction free functions (`struct`, `named_struct`, `map_from_arrays`, `sequence`).

`struct`/`named_struct` build a `MakeStruct` node — the construction counterpart of
the `.struct.field` read accessor; `map_from_arrays` is the same counterpart for the
`.map` accessors; `sequence` builds a per-row integer list. `struct`
takes ``name=expr`` keywords (Pythonic); `named_struct` takes alternating name/value
positional arguments (SQL ``named_struct``).
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr, IntoExpr, _wrap
from batcher.plan.expr_ir.nodes import Col, MakeMap, MakeStruct, Sequence

__all__ = [
    "element",
    "map_from_arrays",
    "named_struct",
    "sequence",
    "struct",
]

#: The reserved column name the list higher-order ops bind each element to. Must
#: match the Rust `eval/list_hof.rs` ELEMENT constant.
_ELEMENT_COL = "element"


def struct(**fields: IntoExpr) -> Expr:
    """Build a struct column from ``name=expr`` fields (Spark ``struct``).

    ``struct(x=col("a"), y=col("b") + 1)`` produces a struct ``{x, y}`` per row; read
    a field back with ``col("s").struct.field("x")``. Requires at least one field.

    Args:
        fields: The struct fields as ``name=expr`` keyword arguments.

    Returns:
        A struct-typed expression with one field per keyword argument.

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


def map_from_arrays(keys: IntoExpr, values: IntoExpr) -> Expr:
    """Build a map column by pairing a list of keys with a list of values.

    This is the construction counterpart of the ``.map`` read accessors, and the name is
    Spark's. SQL spells it ``map(keys, values)`` (DuckDB) or ``map_from_arrays`` (Spark);
    both reach this node. The Python name avoids ``map`` because that is a builtin.

    Three inputs raise rather than being coerced, matching DuckDB, because each has a
    plausible wrong answer instead: a null key (Arrow map keys are non-nullable), a
    duplicate key (keeping first or last is a guess), and key/value lists of different
    lengths (truncating silently drops data). A null *value* is fine, and a null list on
    either side yields a null map.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"k": [["a", "b"]], "v": [[1, 2]]})
            >>> m = ds.select(bt.map_from_arrays(bt.col("k"), bt.col("v")).alias("m"))
            >>> m.to_pydict()["m"]
            [[('a', 1), ('b', 2)]]

            >>> keys = m.select(bt.col("m").map.keys().alias("ks"))
            >>> keys.to_pydict()["ks"]
            [['a', 'b']]

    Args:
        keys: A list column (or literal list) of map keys, one list per row.
        values: A list column of the matching values, the same length per row as `keys`.

    Returns:
        An expression producing a ``Map`` column.
    """
    return MakeMap(_wrap(keys), _wrap(values))


def named_struct(*args: object) -> Expr:
    """Build a struct from alternating ``name, value`` arguments (SQL ``named_struct``).

    ``named_struct("x", col("a"), "y", col("b"))`` is equivalent to
    ``struct(x=col("a"), y=col("b"))``. Field names must be strings.

    Args:
        args: Alternating field ``name, value`` positional arguments.

    Returns:
        A struct-typed expression with one field per name/value pair.

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
    seen: set[str] = set()
    for i in range(0, len(args), 2):
        name = args[i]
        if not isinstance(name, str):
            raise PlanError(f"named_struct field name must be a string, got {name!r}")
        if name in seen:
            raise PlanError(f"named_struct() has a duplicate field name {name!r}")
        seen.add(name)
        fields.append((name, _wrap(args[i + 1])))  # type: ignore[arg-type]
    return MakeStruct(fields)


def sequence(start: IntoExpr, stop: IntoExpr, step: IntoExpr = 1) -> Expr:
    """Build a per-row integer list from ``start`` to ``stop`` inclusive (Spark ``sequence``).

    ``sequence(1, 5)`` yields ``[1, 2, 3, 4, 5]``; ``sequence(col("a"), col("b"), 2)``
    steps by 2. The bounds and step may be columns or literals (cast to Int64); a null
    argument yields a null list, and a ``step`` of 0 raises. Pair with ``explode`` to
    fan a range out into rows.

    Args:
        start: The first value of the range (column or literal, cast to Int64).
        stop: The inclusive last value of the range.
        step: The increment between successive values (defaults to 1).

    Returns:
        A list-typed expression holding the integer range for each row.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [4]})
            >>> ds.select(s=bt.sequence(bt.col("a"), bt.col("b"))).to_pydict()
            {'s': [[1, 2, 3, 4]]}
    """
    return Sequence(_wrap(start), _wrap(stop), _wrap(step))


def element() -> Expr:
    """The current element inside ``list.transform`` / ``list.filter`` (Polars ``element``).

    Use it to build the per-element expression: ``col("a").list.transform(element() * 2)``
    doubles each element, ``col("a").list.filter(element() > 0)`` keeps the positives.
    Outside a list higher-order op it has no binding.

    Returns:
        An expression referencing the current list element.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [[1, 2, 3]]})
            >>> ds.select(d=bt.col("a").list.transform(bt.element() * 2)).to_pydict()
            {'d': [[2, 4, 6]]}
    """
    return Col(_ELEMENT_COL)
