"""String-building free functions (`concat`, `concat_ws`, `format_string`).

All three lower to existing IR — the `concat` binary op (SQL ``||``), `array` + `list.join`,
and casts — so they add public surface without touching the engine. Null handling matches
DuckDB: `concat`/`concat_ws` treat NULL as absent (the differential oracle), not
null-propagating like the raw ``||`` operator.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError, require_bool
from batcher.plan.expr_ir.core import Binary, Coalesce, Expr, IntoExpr, Lit, _wrap
from batcher.plan.expr_ir.nodes import Array, ListJoin

__all__ = ["concat", "concat_ws", "format_string"]


def concat(*exprs: IntoExpr, ignore_nulls: bool = True) -> Expr:
    """Concatenate values into one string (DuckDB ``concat``).

    Each argument is cast to text; NULLs are treated as the empty string (DuckDB
    semantics), so ``concat("a", lit(None), "b")`` is ``"ab"`` — unlike the raw
    ``a || b`` operator, which propagates NULL. Requires at least one argument.

    Spark ``concat``, Polars ``concat_str`` and Daft ``concat`` return null when any
    argument is null instead; ``ignore_nulls=False`` does that.

    Args:
        exprs: The values to concatenate, cast to text.
        ignore_nulls: Treat a null argument as the empty string. ``False`` makes the
            whole result null, as SQL ``||`` does.

    Returns:
        A string expression joining every argument.

    Raises:
        PlanError: If no argument is given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": ["x", "y"], "b": ["1", None]})
            >>> ds.select(c=bt.concat_str(bt.col("a"), bt.col("b"))).to_pydict()
            {'c': ['x1', 'y']}

            >>> ds.select(c=bt.concat_str(bt.col("a"), bt.col("b"), ignore_nulls=False)).to_pydict()
            {'c': ['x1', None]}
    """
    if not exprs:
        raise PlanError("concat() requires at least one argument")
    text = [_wrap(e).cast("string") for e in exprs]
    if require_bool(ignore_nulls, func="concat_str", arg="ignore_nulls"):
        # NULL → '' so a null contributes nothing (DuckDB concat, not `||`).
        text = [Coalesce([t, Lit("")]) for t in text]
    result = text[0]
    for part in text[1:]:
        result = Binary("concat", result, part)
    return result


def concat_ws(separator: str, *exprs: IntoExpr) -> Expr:
    """Concatenate values with `separator` between them (DuckDB/Spark ``concat_ws``).

    NULL arguments are skipped entirely — no doubled separator — matching DuckDB:
    ``concat_ws(",", "a", lit(None), "b")`` is ``"a,b"``. Each argument is cast to
    text. Requires at least one value argument.

    Args:
        separator: The text inserted between adjacent non-null values.
        exprs: The values to concatenate, cast to text (nulls skipped).

    Returns:
        A string expression joining the arguments with ``separator``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": ["x", "y"], "b": ["1", "2"]})
            >>> ds.select(c=bt.concat_ws("-", bt.col("a"), bt.col("b"))).to_pydict()
            {'c': ['x-1', 'y-2']}
    """
    if not exprs:
        raise PlanError("concat_ws() requires at least one value argument")
    # array(...).list.join skips nulls, which is exactly concat_ws's contract.
    # `list.join` of an all-null (non-empty) list is NULL, but DuckDB `concat_ws`
    # returns the empty string when every value argument is NULL — coalesce to "".
    elements = [_wrap(e).cast("string") for e in exprs]
    return Coalesce([ListJoin(Array(elements), separator), Lit("")])


def format_string(format: str, *exprs: IntoExpr, ignore_nulls: bool = True) -> Expr:
    """Interpolate values into a template with ``{}`` placeholders (Polars ``format``).

    ``format_string("{} = {}", col("k"), col("v"))`` yields ``"k = v"`` per row. The
    number of ``{}`` placeholders must equal the number of arguments. Values are cast
    to text with the same NULL-as-empty rule as :func:`concat_str`. The placeholder is the
    literal two-character ``{}`` (no printf width/precision — keep formatting in SQL).

    Polars ``format`` and Daft ``format`` return null when any argument is null, which
    ``ignore_nulls=False`` does. Spark's ``format_string`` is a printf template (``%s``,
    ``%d``) and renders a null as the text ``null``; it is not this function's template
    language.

    Args:
        format: The template string with one ``{}`` per value argument.
        exprs: The values to interpolate, cast to text.
        ignore_nulls: Render a null argument as the empty string. ``False`` makes the
            whole result null.

    Returns:
        A string expression with each ``{}`` replaced by its argument.

    Raises:
        PlanError: If the number of ``{}`` placeholders differs from the argument count.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": ["x", "y"], "b": ["1", None]})
            >>> ds.select(c=bt.format_string("{}={}", bt.col("a"), bt.col("b"))).to_pydict()
            {'c': ['x=1', 'y=']}

            >>> strict = bt.format_string("{}={}", bt.col("a"), bt.col("b"), ignore_nulls=False)
            >>> ds.select(c=strict).to_pydict()
            {'c': ['x=1', None]}
    """
    chunks = format.split("{}")
    if len(chunks) - 1 != len(exprs):
        raise PlanError(
            f"format_string: {len(exprs)} argument(s) but {len(chunks) - 1} '{{}}' placeholder(s)"
        )
    parts: list[IntoExpr] = []
    for i, chunk in enumerate(chunks):
        if chunk:
            parts.append(Lit(chunk))
        if i < len(exprs):
            parts.append(exprs[i])
    if not parts:
        return Lit("")
    return concat(*parts, ignore_nulls=ignore_nulls)
