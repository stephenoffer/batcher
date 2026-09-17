"""The `.meta` accessor: questions about an expression's *shape*, answered without data.

``col("a").meta.root_names()`` reads the expression tree, never a row, so every method here
is plan-time and needs no `Dataset`. It carries the introspection with a Batcher meaning:
the name a projection would give the expression, the columns it reads, whether it is a bare
column, whether it expands to several columns, and the tree the engine is handed.

Polars' remaining ``meta`` members are deliberately absent. ``eq``/``ne``/``pop``/
``undo_aliases``/``is_literal`` describe Polars' own node layout rather than a question a
Batcher plan needs answered, ``as_selector``/``as_expression`` convert between two types
Batcher does not separate (a selector *is* an `Expr` here), and ``serialize``/
``write_json``/``show_graph`` wait on a stable, round-trippable expression serialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import AggExpr, Aliased, Expr, Lit
from batcher.plan.expr_ir.node_base import IRNode, scalar_fields_of

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["_MetaNamespace"]


def _window_key_exprs(keys: list[Any]) -> Iterator[Expr]:
    """The expressions inside window keys, which may be names or ``(key, desc, ...)`` tuples."""
    from batcher.plan.expr_ir.nodes import Col

    for key in keys:
        key = key[0] if isinstance(key, tuple) else key
        if isinstance(key, str):
            yield Col(key)
        elif isinstance(key, (Expr, AggExpr)):
            yield key


def _children(node: Expr | AggExpr) -> list[Expr | AggExpr]:
    """`node`'s sub-expressions in evaluation order, including an aggregate's or window's.

    The scalar node kinds come from the one structural table every rewrite uses
    (`expr_rewrite.traverse`), so a node added there is walked here too. Aggregates and
    windows are not in it, because a rewrite treats them as opaque leaves; introspection
    has to see through them to answer "which columns does this read".
    """
    from batcher.plan.expr_ir.nodes import WindowExpr
    from batcher.plan.expr_rewrite.traverse import _EXPR_KIDS

    if isinstance(node, AggExpr):
        return [k for k in (node.input, node.input2) if k is not None]
    if isinstance(node, WindowExpr):
        own = [node.input] if node.input is not None else []
        return [*own, *_window_key_exprs(node.partition_by), *_window_key_exprs(node.order_by)]
    if isinstance(node, Aliased):
        return [node.inner]
    kids_of = _EXPR_KIDS.get(type(node))
    return list(kids_of(node)) if kids_of is not None else []


def _label(node: Expr | AggExpr) -> str:
    """One line naming `node` and its own parameters, without its children."""
    from batcher.plan.expr_ir.nodes import Col, WindowExpr
    from batcher.plan.expr_ir.selectors import Selector

    if isinstance(node, Col):
        return f"col({node.name})"
    if isinstance(node, Lit):
        return f"lit({node.value!r})"
    if isinstance(node, Aliased):
        return f"alias({node.name})"
    if isinstance(node, AggExpr):
        return f"agg({node.func})"
    if isinstance(node, WindowExpr):
        return f"window({node.func})"
    if isinstance(node, Selector):
        return f"selector({node!r})"
    if isinstance(node, IRNode):
        params = []
        for field in scalar_fields_of(type(node)):
            value = getattr(node, field)
            # Only plain parameters belong on the line. A node that predates the
            # declarative base (`Case`) keeps its sub-expressions in undeclared fields,
            # and those are drawn as children below rather than repeated here.
            if not isinstance(value, (str, int, float, bool)):
                continue
            params.append(str(value) if isinstance(value, str) else f"{field}={value!r}")
        tag = getattr(node.tag, "value", node.tag)
        return f"{tag}({', '.join(params)})" if params else str(tag)
    return type(node).__name__.lower()


def _tree_lines(node: Expr | AggExpr, prefix: str, connector: str) -> Iterator[str]:
    """`node` and its subtree, one line each, drawn with the `explain` tree glyphs."""
    yield f"{prefix}{connector}{_label(node)}"
    kids = _children(node)
    if connector == "└─ ":
        prefix += "   "
    elif connector:
        prefix += "│  "
    for i, kid in enumerate(kids):
        yield from _tree_lines(kid, prefix, "└─ " if i == len(kids) - 1 else "├─ ")


class _MetaNamespace:
    """Expression introspection: ``col("a").meta.root_names()``.

    Every method reads the expression tree only, so none needs a `Dataset` or runs a query.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> (bt.col("a") + bt.col("b")).alias("total").meta.output_name()
            'total'
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr | AggExpr) -> None:
        """Wrap the parent expression so its `.meta` methods can inspect it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.meta accessor of col('c')>``."""
        return f"<.meta accessor of {self._e!r}>"

    def output_name(self, *, raise_if_undetermined: bool = True) -> str | None:
        """The column name this expression takes as an unnamed projection.

        It is the name ``ds.select(expr)`` would give the result: an ``alias`` wins, then
        the leftmost column the expression reads, and ``"literal"`` for an expression that
        reads none. A selector such as ``col("a", "b")`` or ``numeric()`` expands to several
        columns and so has no single name.

        Args:
            raise_if_undetermined: Raise when there is no single name, rather than return
                ``None``.

        Returns:
            The output name, or ``None`` when there is none and `raise_if_undetermined`
            is false.

        Raises:
            PlanError: If the expression expands to several columns and
                `raise_if_undetermined` is true.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> (bt.col("price") * 2).meta.output_name()
                'price'
                >>> bt.lit(1).meta.output_name()
                'literal'
                >>> print(bt.numeric().meta.output_name(raise_if_undetermined=False))
                None
        """
        from batcher.plan.expr_rewrite.naming import output_name

        if self.has_multiple_outputs():
            if raise_if_undetermined:
                raise PlanError(
                    f"{self._e!r} expands to several columns, so it has no single output name; "
                    "give each expanded column a name with .name.prefix()/.name.suffix()"
                )
            return None
        return output_name(self._e)

    def root_names(self) -> list[str]:
        """The input columns the expression reads, in order of appearance, repeats kept.

        This is what column pruning keeps for the expression: ``(col("a") + col("b") +
        col("a"))`` reads ``["a", "b", "a"]``. A selector names no column until it meets a
        schema, so it contributes none.

        Returns:
            The column names, left to right, once per reference.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> (bt.col("a") + bt.col("b") * bt.col("a")).meta.root_names()
                ['a', 'b', 'a']
                >>> bt.col("x").sum().over(partition_by="g").meta.root_names()
                ['x', 'g']
        """
        from batcher.plan.expr_ir.nodes import Col

        names: list[str] = []
        stack: list[Expr | AggExpr] = [self._e]
        while stack:
            node = stack.pop()
            if isinstance(node, Col):
                names.append(node.name)
            stack.extend(reversed(_children(node)))
        return names

    def is_column(self) -> bool:
        """Whether the expression is a bare column reference, with no alias or computation.

        Returns:
            True for ``col("a")``; false for ``col("a").alias("b")``, ``col("a") + 1``, and
            a multi-column selector.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.col("a").meta.is_column(), (bt.col("a") + 1).meta.is_column()
                (True, False)
        """
        from batcher.plan.expr_ir.nodes import Col

        return type(self._e) is Col

    def has_multiple_outputs(self) -> bool:
        """Whether the expression expands to more than one column when projected.

        A selector (``col("a", "b")``, ``numeric()``, ``matches("^x")``), or any expression
        built over one, expands to one output column per matched input column.

        Returns:
            True when the expression holds a column selector.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.col("a", "b").meta.has_multiple_outputs()
                True
                >>> (bt.col("a") * 2).meta.has_multiple_outputs()
                False
        """
        from batcher.plan.expr_ir.selectors import Selector

        stack: list[Expr | AggExpr] = [self._e]
        while stack:
            node = stack.pop()
            if isinstance(node, Selector):
                return True
            stack.extend(_children(node))
        return False

    def tree_format(self, *, return_as_string: bool = False) -> str | None:
        """Draw the expression as a tree, one node per line, children indented beneath.

        Each line names a node by its engine tag and its own parameters, so the tree shows
        what the engine is handed rather than how the expression was spelled. The layout
        uses the glyphs `Dataset.explain` draws a plan with, not Polars' boxed diagram.

        Args:
            return_as_string: Return the drawing instead of printing it.

        Returns:
            The drawing when `return_as_string` is true, otherwise ``None``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> (bt.col("a") * 2 + bt.col("b")).meta.tree_format()
                binary(add)
                ├─ binary(mul)
                │  ├─ col(a)
                │  └─ lit(2)
                └─ col(b)
        """
        text = "\n".join(_tree_lines(self._e, "", ""))
        if return_as_string:
            return text
        print(text)
        return None
