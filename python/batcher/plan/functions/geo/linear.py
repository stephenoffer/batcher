"""Positions along a chain, expressed as a fraction of its length.

This is the vocabulary route and network data is described in: "the incident is 0.32 of
the way along segment 4471", "give me the stretch between mile 3 and mile 7", "snap this
GPS fix to the road".

Everything here is parameterized on a `[0, 1]` fraction rather than an absolute
distance, which is what makes a query portable across chains of different lengths and is
the convention PostGIS standardizes on. `st_line_interpolate_point` and
`st_line_locate_point` are exact inverses of each other.

Every function here operates on a **single** chain. A multi-chain geometry has no
defined traversal order, so it yields null rather than an answer against whichever
member the encoder happened to write first.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry, value

__all__ = [
    "st_closest_point",
    "st_line_interpolate_point",
    "st_line_locate_point",
    "st_line_substring",
    "st_project",
    "st_shortest_line",
]


def st_line_interpolate_point(geom: Expr | str, fraction: Expr | float) -> Expr:
    """The point a given fraction of the way along a chain.

    A fraction outside `[0, 1]` raises rather than clamping: it means the caller
    computed the fraction wrongly, and clamping would hide that on every row.

    Args:
        geom: A chain.
        fraction: A position along it, from 0 to 1.

    Returns:
        The position as a point, or null for a non-chain.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 10 0, 10 10)']})
            >>> got = bt.st_as_text(bt.st_line_interpolate_point(bt.col("g"), 0.5))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(10 0)']}
    """
    return geo_call("st_line_interpolate_point", geometry(geom), value(fraction))


def st_line_locate_point(geom: Expr | str, point: Expr | str) -> Expr:
    """How far along a chain it passes closest to a point.

    The snap-to-network primitive, and the exact inverse of
    `st_line_interpolate_point`. Pair it with `st_closest_point` when you need the
    snapped position rather than the measure.

    Args:
        geom: A chain.
        point: The point to locate.

    Returns:
        The fraction along the chain, from 0 to 1, or null for a non-chain.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 10 0)'], 'p': ['POINT(3 5)']})
            >>> got = bt.st_line_locate_point(bt.col("g"), bt.col("p"))
            >>> ds.select(v=got).to_pydict()
            {'v': [0.3]}
    """
    return geo_call("st_line_locate_point", geometry(geom), geometry(point))


def st_line_substring(geom: Expr | str, start: Expr | float, end: Expr | float) -> Expr:
    """The stretch of a chain between two fractions.

    The endpoints are inserted as real positions, so the result starts and ends
    exactly where asked rather than at the nearest original vertex — which matters when
    the substring is then measured. Reversed fractions are normalized rather than
    producing nothing.

    Args:
        geom: A chain.
        start: The starting fraction.
        end: The ending fraction.

    Returns:
        The stretch of chain, or a point when the fractions are equal.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 10 0, 10 10)']})
            >>> got = bt.st_as_text(bt.st_line_substring(bt.col("g"), 0.25, 0.75))
            >>> ds.select(v=got).to_pydict()
            {'v': ['LINESTRING(5 0, 10 0, 10 5)']}
    """
    return geo_call("st_line_substring", geometry(geom), value(start), value(end))


def st_closest_point(a: Expr | str, b: Expr | str) -> Expr:
    """The point of one geometry closest to another.

    Asymmetric on purpose, like PostGIS: it returns a position **on the first
    geometry**, which is what a snap-to-road or snap-to-boundary step needs.

    Args:
        a: The geometry to return a position on.
        b: The geometry to be near.

    Returns:
        A point on `a`, or null when either is empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'a': ['LINESTRING(0 0, 10 0)'], 'b': ['POINT(4 7)']})
            >>> got = bt.st_as_text(bt.st_closest_point(bt.col("a"), bt.col("b")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(4 0)']}
    """
    return geo_call("st_closest_point", geometry(a), geometry(b))


def st_shortest_line(a: Expr | str, b: Expr | str) -> Expr:
    """The line joining the closest positions of two geometries.

    The drawable form of `st_distance`: the same number, as a geometry you can put on
    a map to see *where* the gap is.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        A two-position line joining the closest positions, or null when either is
        empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'a': ['LINESTRING(0 0, 10 0)'], 'b': ['POINT(4 7)']})
            >>> got = bt.st_as_text(bt.st_shortest_line(bt.col("a"), bt.col("b")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['LINESTRING(4 0, 4 7)']}
    """
    return geo_call("st_shortest_line", geometry(a), geometry(b))


def st_project(point: Expr | str, distance: Expr | float, azimuth: Expr | float) -> Expr:
    """The lon/lat point reached by travelling a distance along a bearing.

    Geodesic, on the mean-radius sphere, and the inverse of `st_distance_sphere` plus
    a bearing. The practical use is building a proximity box correctly: the four
    cardinal destinations at radius `r` give a bounding box in degrees that actually
    covers `r` metres, which a fixed degree offset does not.

    Args:
        point: A lon/lat point.
        distance: The distance to travel, in metres.
        azimuth: The bearing, in degrees clockwise from north.

    Returns:
        The destination point, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(0 0)']})
            >>> got = bt.st_as_text(bt.st_project(bt.col("g"), 111195.0, 0.0))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(0 0.9999992784435)']}
    """
    return geo_call("st_project", geometry(point), value(distance), value(azimuth))
