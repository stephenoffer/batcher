"""Polygon overlay: the union, intersection and difference of two areal geometries.

Clipping parcels to a boundary, erasing a lake from a land-use polygon, merging two
coverage areas: each is a set operation on the points two shapes cover. These run the
same noding-and-tracing overlay `st_buffer` is built on, row by row in Rust.

The operands must be areal, a ``POLYGON`` or ``MULTIPOLYGON``, and valid by
`st_is_valid`. A point or line operand, an invalid polygon such as a bowtie, and a shape
the overlay cannot trace robustly give null for that row rather than a result with no
meaning, and `st_is_valid_reason` names what is wrong with an invalid one. An empty
result, such as the intersection of two disjoint parcels, is ``POLYGON EMPTY`` rather than
null, which is what GEOS and DuckDB write. The overlay is planar and two-dimensional, like
`st_buffer`, so a z coordinate does not survive into the result.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry

__all__ = ["st_difference", "st_intersection", "st_union"]


def st_union(a: Expr | str, b: Expr | str) -> Expr:
    """The points covered by either of two areal geometries, as polygons.

    Where `st_collect` keeps two overlapping polygons as two members that overlap, this
    dissolves the shared area and the seam between them into one boundary.

    Args:
        a: The first polygon or multipolygon.
        b: The second polygon or multipolygon.

    Returns:
        A polygon or multipolygon, or null when either input is null, not areal, or not
        valid.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'],
            ...      'b': ['POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))']}
            ... )
            >>> got = bt.st_area(bt.st_union(bt.col("a"), bt.col("b")))
            >>> ds.select(v=got).to_pydict()
            {'v': [28.0]}
    """
    return geo_call("st_union", geometry(a), geometry(b))


def st_intersection(a: Expr | str, b: Expr | str) -> Expr:
    """The points covered by both of two areal geometries, as polygons.

    The clip: ``st_intersection(parcel, county)`` is the part of each parcel inside the
    county. Disjoint operands intersect in ``POLYGON EMPTY``.

    Args:
        a: The first polygon or multipolygon.
        b: The second polygon or multipolygon.

    Returns:
        A polygon, a multipolygon, or ``POLYGON EMPTY``, or null when either input is
        null, not areal, or not valid.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'],
            ...      'b': ['POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))']}
            ... )
            >>> got = bt.st_area(bt.st_intersection(bt.col("a"), bt.col("b")))
            >>> ds.select(v=got).to_pydict()
            {'v': [4.0]}
    """
    return geo_call("st_intersection", geometry(a), geometry(b))


def st_difference(a: Expr | str, b: Expr | str) -> Expr:
    """The points covered by the first areal geometry and not the second, as polygons.

    The erase: ``st_difference(land, lake)`` is the land with the lake cut out, which is
    a polygon with a hole when the lake lies wholly inside it. Not symmetric: swapping
    the operands answers the other question.

    Args:
        a: The polygon or multipolygon to cut from.
        b: The polygon or multipolygon to remove.

    Returns:
        A polygon, a multipolygon, or ``POLYGON EMPTY``, or null when either input is
        null, not areal, or not valid.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'],
            ...      'b': ['POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))']}
            ... )
            >>> got = bt.st_area(bt.st_difference(bt.col("a"), bt.col("b")))
            >>> ds.select(v=got).to_pydict()
            {'v': [12.0]}
    """
    return geo_call("st_difference", geometry(a), geometry(b))
