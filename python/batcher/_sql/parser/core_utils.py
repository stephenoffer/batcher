"""Small stateless AST helpers shared across translator theme modules.

Kept in their own leaf module so every theme module can import them without
creating an import cycle through the translator class.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import PlanError


def _columns_selector(node) -> Any:
    """Translate DuckDB ``COLUMNS(*)`` / ``COLUMNS('regex')`` to a column selector.

    The selector is expanded against the input schema by the projection builder, so
    ``COLUMNS('^sales')`` projects every matching column and ``func(COLUMNS(*))``
    applies ``func`` to each column. Reuses the DataFrame selector engine.
    """
    from sqlglot import expressions as exp

    from batcher.plan.expr_ir.selectors import all as select_all
    from batcher.plan.expr_ir.selectors import matches

    inner = node.this
    if isinstance(inner, exp.Star):
        return select_all()
    if isinstance(inner, exp.Literal) and inner.is_string:
        return matches(inner.name)
    raise NotImplementedError(
        f"COLUMNS(...) supports COLUMNS(*) or COLUMNS('regex'); got {type(inner).__name__}"
    )


def _positional(projections, literal, clause: str):
    """Resolve a 1-based positional reference (`ORDER BY 2`) to its SELECT item."""
    idx = int(literal.this)
    if not 1 <= idx <= len(projections):
        raise PlanError(
            f"{clause} position {idx} is out of range: the SELECT list has "
            f"{len(projections)} item(s)"
        )
    return projections[idx - 1]


def _unwrap_alias(p):
    from sqlglot import expressions as exp

    return p.this if isinstance(p, exp.Alias) else p


def _alias_of(p) -> str:
    from sqlglot import expressions as exp

    if isinstance(p, exp.Alias):
        return p.alias
    if isinstance(p, exp.Column):
        return p.name
    # No explicit `AS`: derive the output name from the expression, matching the
    # convention of the reference engines (DuckDB/Polars) so a column the user did not
    # alias lines up across engines — `sum(l_quantity)`, `count_star()` — rather than a
    # bespoke `SUM_l_quantity`. `count(*)` is DuckDB's special `count_star()`.
    if isinstance(p, exp.Count) and isinstance(p.this, exp.Star):
        return "count_star()"
    return p.sql().lower()


def _has_aggregate(node) -> bool:
    from sqlglot import expressions as exp

    # An aggregate inside a window (e.g. SUM(x) OVER (...)) is a window
    # function, not a GROUP-BY aggregate, so ignore those. An aggregate
    # inside a (scalar) subquery belongs to the inner query, not this one.
    for a in node.find_all(exp.AggFunc):
        if a.find_ancestor(exp.Window) is not None:
            continue
        if a.find_ancestor(exp.Subquery) is not None:
            continue
        return True
    return False
