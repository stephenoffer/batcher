"""Small stateless AST helpers shared across translator theme modules.

Kept in their own leaf module so every theme module can import them without
creating an import cycle through the translator class.
"""

from __future__ import annotations


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
