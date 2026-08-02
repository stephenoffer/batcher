"""Picking a geometry apart, and asking whether it is well formed.

The first half addresses members and vertices: a multi-geometry's `n`-th part, a
chain's endpoints, a polygon's rings. The second half is the validity surface, and it
is the more important of the two.

Real geometry columns are full of invalid polygons — self-intersecting rings from a
digitizing error, holes poking outside their shell, rings with two vertices. Every areal
predicate produces nonsense on those, silently. `st_is_valid_reason` is therefore not a
nicety: it is the difference between a spatial join that is wrong and one that is wrong
*and undetected*. Run it over a new column before trusting anything else here.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry, value

__all__ = [
    "st_end_point",
    "st_exterior_ring",
    "st_geometry_n",
    "st_has_z",
    "st_interior_ring_n",
    "st_is_closed",
    "st_is_collection",
    "st_is_empty",
    "st_is_ring",
    "st_is_simple",
    "st_is_valid",
    "st_is_valid_reason",
    "st_point_n",
    "st_start_point",
]


def st_geometry_n(geom: Expr | str, n: Expr | int) -> Expr:
    """The 1-based n-th member of a multi-geometry.

    A simple geometry has exactly one member — itself — so `st_geometry_n(g, 1)` is
    the identity on one. Indexing is 1-based to match PostGIS and SQL.

    Args:
        geom: The geometry.
        n: The 1-based member index.

    Returns:
        The member, or null when the index is out of range.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['MULTIPOINT((0 0), (5 5))']})
            >>> got = bt.st_as_text(bt.st_geometry_n(bt.col("g"), 2))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(5 5)']}
    """
    return geo_call("st_geometry_n", geometry(geom), value(n))


def st_point_n(geom: Expr | str, n: Expr | int) -> Expr:
    """The 1-based n-th position of a chain.

    Negative indices count from the end, so `-1` is the last position. Null for
    anything but a lone `LINESTRING`: a multi-chain has no defined position order, and
    answering from whichever member the encoder wrote first is how a route's start
    silently changes after a re-export.

    Args:
        geom: A chain.
        n: The 1-based position index; negative counts from the end.

    Returns:
        The position as a point, or null when out of range or not a chain.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 1 1, 2 2)']})
            >>> got = bt.st_as_text(bt.st_point_n(bt.col("g"), -1))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(2 2)']}
    """
    return geo_call("st_point_n", geometry(geom), value(n))


def st_start_point(geom: Expr | str) -> Expr:
    """The first position of a chain.

    Null for anything but a lone `LINESTRING`; see `st_point_n`.

    Args:
        geom: A chain.

    Returns:
        The first position as a point, or null for a non-chain.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 3 4)']})
            >>> got = bt.st_as_text(bt.st_start_point(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(0 0)']}
    """
    return geo_call("st_start_point", geometry(geom))


def st_end_point(geom: Expr | str) -> Expr:
    """The last position of a chain.

    Null for anything but a lone `LINESTRING`; see `st_point_n`.

    Args:
        geom: A chain.

    Returns:
        The last position as a point, or null for a non-chain.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 3 4)']})
            >>> got = bt.st_as_text(bt.st_end_point(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(3 4)']}
    """
    return geo_call("st_end_point", geometry(geom))


def st_exterior_ring(geom: Expr | str) -> Expr:
    """The outer ring of a polygon.

    The complement of `st_interior_ring_n`. Together they are how you get at a
    polygon's rings as chains, which is what the linear-referencing functions need.

    Args:
        geom: A polygon.

    Returns:
        The outer ring as a chain, or null for a non-polygon.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_geometry_type(bt.st_exterior_ring(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['LINESTRING']}
    """
    return geo_call("st_exterior_ring", geometry(geom))


def st_interior_ring_n(geom: Expr | str, n: Expr | int) -> Expr:
    """The 1-based n-th hole of a polygon.

    Pairs with `st_num_interior_rings` to walk a polygon's holes.

    Args:
        geom: A polygon.
        n: The 1-based hole index.

    Returns:
        The hole as a chain, or null when out of range.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4
            ...     4))']}
            ... )
            >>> got = bt.st_length(bt.st_interior_ring_n(bt.col("g"), 1))
            >>> ds.select(v=got).to_pydict()
            {'v': [8.0]}
    """
    return geo_call("st_interior_ring_n", geometry(geom), value(n))


def st_is_empty(geom: Expr | str) -> Expr:
    """Whether a geometry holds no positions at all.

    Emptiness is about positions, not members: a collection of empty geometries is
    empty. An empty geometry satisfies no positive predicate, which is why a column
    quietly full of them makes a spatial join return nothing.

    Args:
        geom: The geometry.

    Returns:
        True when the geometry holds no positions.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['POLYGON EMPTY', 'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']}
            ... )
            >>> got = bt.st_is_empty(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True, False]}
    """
    return geo_call("st_is_empty", geometry(geom))


def st_is_valid(geom: Expr | str) -> Expr:
    """Whether a geometry satisfies OGC validity.

    Use `st_is_valid_reason` instead when the answer is no — a boolean tells you a
    row is broken, a reason tells you which vertex to fix.

    Args:
        geom: The geometry.

    Returns:
        True when the geometry satisfies OGC validity.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))', 'POLYGON((0 0, 4 4, 4 0, 0
            ...     4, 0 0))']}
            ... )
            >>> got = bt.st_is_valid(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True, False]}
    """
    return geo_call("st_is_valid", geometry(geom))


def st_is_valid_reason(geom: Expr | str) -> Expr:
    """Why a geometry is invalid, or null when it is not.

    Null means valid, which makes `WHERE st_is_valid_reason(g) IS NOT NULL` the query
    that finds every broken row in a column and explains each one. Where the failure is
    a single location the message names the position.

    Args:
        geom: The geometry.

    Returns:
        The reason the geometry is invalid, or null when it is valid.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 4, 4 0, 0 4, 0 0))']})
            >>> got = bt.st_is_valid_reason(bt.col("g")).str.contains("self-intersects")
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_is_valid_reason", geometry(geom))


def st_is_closed(geom: Expr | str) -> Expr:
    """Whether a chain's ends meet.

    A polygon reports true — its rings are closed by definition. Closed is not the
    same as being a ring: a figure-eight is closed and crosses itself, which is what
    `st_is_ring` separates.

    Args:
        geom: The geometry.

    Returns:
        True when every chain's first and last positions coincide.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['LINESTRING(0 0, 1 0, 1 1, 0 0)', 'LINESTRING(0 0, 1 1)']}
            ... )
            >>> got = bt.st_is_closed(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True, False]}
    """
    return geo_call("st_is_closed", geometry(geom))


def st_is_ring(geom: Expr | str) -> Expr:
    """Whether a chain is a valid linear ring.

    Closed *and* simple. This is the test a chain has to pass before
    `st_make_polygon` produces something meaningful from it.

    Args:
        geom: The geometry.

    Returns:
        True when the chain is closed and does not cross itself.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['LINESTRING(0 0, 4 0, 4 4, 0 0)', 'LINESTRING(0 0, 4 4, 4 0, 0 4,
            ...     0 0)']}
            ... )
            >>> got = bt.st_is_ring(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True, False]}
    """
    return geo_call("st_is_ring", geometry(geom))


def st_is_simple(geom: Expr | str) -> Expr:
    """Whether a geometry has no anomalous self-intersection.

    For a chain that means it does not touch or cross itself except at a closing
    endpoint; for a point set it means no duplicates. Areal geometries are simple once
    they are valid, so this is not a second validity check on a polygon.

    Args:
        geom: The geometry.

    Returns:
        True when the geometry has no anomalous self-intersection.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['LINESTRING(0 0, 2 2, 2 0, 0 2)', 'LINESTRING(0 0, 1 1)']}
            ... )
            >>> got = bt.st_is_simple(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [False, True]}
    """
    return geo_call("st_is_simple", geometry(geom))


def st_is_collection(geom: Expr | str) -> Expr:
    """Whether a geometry has multiple parts.

    True for every `MULTI*` type and for `GEOMETRYCOLLECTION`, false for the four
    simple types. Pairs with `st_num_geometries` when a pipeline has to branch on
    whether the parts need walking.

    Args:
        geom: The geometry.

    Returns:
        True for a multi-geometry or a collection.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['MULTIPOINT((0 0))', 'POINT(2 2)']})
            >>> got = bt.st_is_collection(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True, False]}
    """
    return geo_call("st_is_collection", geometry(geom))


def st_has_z(geom: Expr | str) -> Expr:
    """Whether a geometry carries a z ordinate.

    A mixed 2D/3D column is legal and common, and it is what makes a bare `st_z`
    return nulls for half a column. Check this before assuming.

    Args:
        geom: The geometry.

    Returns:
        True when the geometry carries elevations.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT Z(1 2 3)', 'POINT(2 2)']})
            >>> got = bt.st_has_z(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [True, False]}
    """
    return geo_call("st_has_z", geometry(geom))
