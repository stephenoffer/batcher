"""Reading the parts of a geometry: ordinates, bounds, type, and counts.

These are the cheap functions. Every one of them reads a decoded geometry and returns a
plain scalar, so they are what you reach for when a query needs a number rather than a
shape — a group key, a sort key, a filter bound.

`st_xmin` and its three siblings are the ones to know: together they are the bounding
box, and a bounding-box filter is the single highest-leverage thing you can put in front
of a spatial predicate. Four range comparisons on Float64 columns push down to the scan
and to Parquet statistics; `st_intersects` does not.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry, value

__all__ = [
    "st_coord_dim",
    "st_dimension",
    "st_geometry_type",
    "st_num_geometries",
    "st_num_interior_rings",
    "st_num_points",
    "st_set_srid",
    "st_srid",
    "st_x",
    "st_xmax",
    "st_xmin",
    "st_y",
    "st_ymax",
    "st_ymin",
    "st_z",
]


def st_x(geom: Expr | str) -> Expr:
    """The x ordinate of a point.

    Null for anything but a point, which is what PostGIS does. The x of a polygon is
    not its first vertex — it is undefined, and returning a vertex would be a
    plausible wrong answer rather than an obvious absent one.

    Args:
        geom: The geometry.

    Returns:
        The x ordinate, or null unless the geometry is a point.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['POINT(2 2)', 'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']}
            ... )
            >>> got = bt.st_x(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [2.0, None]}
    """
    return geo_call("st_x", geometry(geom))


def st_y(geom: Expr | str) -> Expr:
    """The y ordinate of a point.

    Null for anything but a point; see `st_x`.

    Args:
        geom: The geometry.

    Returns:
        The y ordinate, or null unless the geometry is a point.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_y(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [2.0]}
    """
    return geo_call("st_y", geometry(geom))


def st_z(geom: Expr | str) -> Expr:
    """The z ordinate of a 3D point.

    Null for a 2D point as well as for a non-point: a 2D geometry has no elevation,
    and reporting the zero it happens to be stored with would invent data.

    Args:
        geom: The geometry.

    Returns:
        The elevation, or null unless the geometry is a 3D point.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT Z(1 2 3)']})
            >>> got = bt.st_z(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [3.0]}
    """
    return geo_call("st_z", geometry(geom))


def st_xmin(geom: Expr | str) -> Expr:
    """The minimum x of a geometry's bounding box.

    The four bound accessors are the cheapest useful thing you can compute from a
    geometry column. Materialize them once alongside the geometry and every subsequent
    region filter becomes four Float64 comparisons that push down to the scan.

    Args:
        geom: The geometry.

    Returns:
        The minimum x, or null for an empty geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_xmin(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [0.0]}
    """
    return geo_call("st_xmin", geometry(geom))


def st_ymin(geom: Expr | str) -> Expr:
    """The minimum y of a geometry's bounding box.

    See `st_xmin` for why these four are worth materializing.

    Args:
        geom: The geometry.

    Returns:
        The minimum y, or null for an empty geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_ymin(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [0.0]}
    """
    return geo_call("st_ymin", geometry(geom))


def st_xmax(geom: Expr | str) -> Expr:
    """The maximum x of a geometry's bounding box.

    See `st_xmin` for why these four are worth materializing.

    Args:
        geom: The geometry.

    Returns:
        The maximum x, or null for an empty geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_xmax(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [4.0]}
    """
    return geo_call("st_xmax", geometry(geom))


def st_ymax(geom: Expr | str) -> Expr:
    """The maximum y of a geometry's bounding box.

    See `st_xmin` for why these four are worth materializing.

    Args:
        geom: The geometry.

    Returns:
        The maximum y, or null for an empty geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_ymax(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [4.0]}
    """
    return geo_call("st_ymax", geometry(geom))


def st_geometry_type(geom: Expr | str) -> Expr:
    """The OGC type name of a geometry.

    Returns the bare name — `POINT`, `MULTIPOLYGON` — without the `ST_` prefix
    PostGIS prepends, matching the OGC spelling and the WKT keyword. A mixed geometry
    column is common and this is how you find out what is in it.

    Args:
        geom: The geometry.

    Returns:
        The uppercase OGC type name, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['POINT(2 2)', 'LINESTRING(0 0, 3 4)', 'POLYGON((0 0, 4 0, 4 4, 0
            ...     4, 0 0))']}
            ... )
            >>> got = bt.st_geometry_type(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT', 'LINESTRING', 'POLYGON']}
    """
    return geo_call("st_geometry_type", geometry(geom))


def st_dimension(geom: Expr | str) -> Expr:
    """The topological dimension of a geometry.

    0 for points, 1 for chains, 2 for areas. This is what several predicates turn on:
    `st_crosses` and `st_overlaps` are both defined by comparing the dimension of an
    intersection against the dimensions of its operands.

    Args:
        geom: The geometry.

    Returns:
        0, 1 or 2, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['POINT(2 2)', 'LINESTRING(0 0, 3 4)', 'POLYGON((0 0, 4 0, 4 4, 0
            ...     4, 0 0))']}
            ... )
            >>> got = bt.st_dimension(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [0, 1, 2]}
    """
    return geo_call("st_dimension", geometry(geom))


def st_srid(geom: Expr | str) -> Expr:
    """The spatial reference system a geometry is stated in.

    0 means unknown, not WGS 84. A geometry parsed from GeoJSON carries 4326 because
    RFC 7946 fixes it; one parsed from bare WKT carries 0 because nothing said.

    Args:
        geom: The geometry.

    Returns:
        The EPSG code, 0 when unknown, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_srid(bt.st_set_srid(bt.col("g"), 4326))
            >>> ds.select(v=got).to_pydict()
            {'v': [4326]}
    """
    return geo_call("st_srid", geometry(geom))


def st_set_srid(geom: Expr | str, srid: Expr | int) -> Expr:
    """Label a geometry with a spatial reference system.

    An assertion about what the coordinates already mean, not a conversion. It moves
    nothing. `st_transform` is the conversion, and it needs a correct source label to
    work from — which is what this is for when the source data did not carry one.

    Args:
        geom: The geometry.
        srid: The EPSG code to label it with.

    Returns:
        The same geometry, relabelled.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_srid(bt.st_set_srid(bt.col("g"), 3857))
            >>> ds.select(v=got).to_pydict()
            {'v': [3857]}
    """
    return geo_call("st_set_srid", geometry(geom), value(srid))


def st_num_points(geom: Expr | str) -> Expr:
    """The number of positions in a geometry.

    Counts every position in every part, including a ring's repeated closing vertex.
    The practical use is finding the geometries that dominate a column's size before
    deciding whether to `st_simplify`.

    Args:
        geom: The geometry.

    Returns:
        The position count, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_num_points(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [5]}
    """
    return geo_call("st_num_points", geometry(geom))


def st_num_geometries(geom: Expr | str) -> Expr:
    """The number of top-level members of a geometry.

    1 for a simple geometry — it is its own only member — and the member count for a
    multi-geometry or collection. Pairs with `st_geometry_n` to walk the parts.

    Args:
        geom: The geometry.

    Returns:
        The member count, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['MULTIPOINT((0 0), (1 1), (2 2))', 'POINT(2 2)']})
            >>> got = bt.st_num_geometries(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [3, 1]}
    """
    return geo_call("st_num_geometries", geometry(geom))


def st_num_interior_rings(geom: Expr | str) -> Expr:
    """The number of holes in a polygon.

    Reads the first polygon of the geometry. A column where this is unexpectedly
    non-zero is usually a sign of a failed union upstream.

    Args:
        geom: The geometry.

    Returns:
        The hole count, or 0 for a non-polygon.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4
            ...     4))', 'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']}
            ... )
            >>> got = bt.st_num_interior_rings(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [1, 0]}
    """
    return geo_call("st_num_interior_rings", geometry(geom))


def st_coord_dim(geom: Expr | str) -> Expr:
    """The number of ordinates each position carries.

    3 when the geometry is tagged Z, 2 otherwise. Dimensionality is a property of the
    whole geometry rather than of individual positions, which is what every geometry
    encoding assumes and what makes a half-3D chain unrepresentable.

    Args:
        geom: The geometry.

    Returns:
        2 or 3, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)', 'POINT Z(1 2 3)']})
            >>> got = bt.st_coord_dim(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [2, 3]}
    """
    return geo_call("st_coord_dim", geometry(geom))
