"""The one builder every geospatial function is written in terms of.

Each `st_*` function in this package is a thin, typed, documented wrapper around a
single `GeoFunc` node. Keeping the node construction here means the 113 wrappers carry
no logic at all — they are a signature, a docstring, and one call — so the family reads
uniformly and a new function cannot accidentally lower differently from its neighbours.

Geometry arguments are coerced the way every other Batcher function coerces: a bare
string is a *literal geometry* (WKT, EWKT, GeoJSON or hex WKB, detected by the engine),
not a column name. That is the opposite of the string-means-column convention the
relational verbs use, and it is deliberate — ``st_intersects(col("g"), "POINT(1 2)")``
is what a user means every time, and a column of geometries is always named with
``col()`` because it is a geometry, not a name.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr, Lit
from batcher.plan.expr_ir.func_nodes import GeoFunc

__all__ = ["geo_call", "geometry", "value"]


def geometry(g: Expr | str | bytes) -> Expr:
    """Coerce a geometry argument to an `Expr`.

    A `str` or `bytes` becomes a literal geometry rather than a column reference, so a
    literal geometry can be written inline without wrapping it in `lit`.

    Args:
        g: An expression producing a geometry column, or an inline geometry as WKT,
            EWKT, GeoJSON, hex WKB, or raw WKB bytes.

    Returns:
        The argument as an expression.

    Examples:
        .. doctest::

            >>> from batcher.plan.functions.geo._build import geometry
            >>> geometry("POINT(1 2)").to_ir()["e"]
            'lit'
    """
    if isinstance(g, Expr):
        return g
    return Lit(g)


def value(v: Expr | float | int | str) -> Expr:
    """Coerce a non-geometry argument (a radius, a precision, an SRID) to an `Expr`.

    Args:
        v: An expression or a plain Python constant.

    Returns:
        The argument as an expression.

    Examples:
        .. doctest::

            >>> from batcher.plan.functions.geo._build import value
            >>> value(4326).to_ir()
            {'e': 'lit', 'value': {'int': 4326}}
    """
    return v if isinstance(v, Expr) else Lit(v)


def geo_call(fn: str, *args: Expr) -> Expr:
    """Build the `GeoFunc` node for `fn` over `args`.

    Args:
        fn: The engine function name, validated against the `GEO_FNS` vocabulary.
        *args: The already-coerced argument expressions, in engine order.

    Returns:
        The geospatial expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.functions.geo._build import geo_call, geometry
            >>> geo_call("st_area", geometry(bt.col("g"))).to_ir()["fn"]
            'st_area'
    """
    return GeoFunc(fn, list(args))
