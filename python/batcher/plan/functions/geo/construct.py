"""Building a geometry, and deriving a simpler shape from one.

The constructors are how a table of coordinate columns becomes a table of geometries:
almost every real dataset stores `lat`/`lon` as floats, and `st_point` is the bridge.
The derived shapes — centroid, envelope, hull — go the other way, reducing a geometry
to something cheaper to index, join or draw.

`st_envelope` and `st_convex_hull` are worth knowing as a pair. The envelope is the
bounding box: four corners, exact to compute, and what every spatial index actually
stores. The hull is tighter but costs a sort. Filter on the envelope first and the hull
second, and the expensive predicate runs on a fraction of the rows.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry, value

__all__ = [
    "st_boundary",
    "st_buffer",
    "st_centroid",
    "st_collect",
    "st_convex_hull",
    "st_envelope",
    "st_expand",
    "st_make_envelope",
    "st_make_line",
    "st_make_polygon",
    "st_point",
    "st_point_on_surface",
    "st_point_z",
]


def st_point(x: Expr | float, y: Expr | float) -> Expr:
    """Build a point from two ordinate columns.

    Argument order is x-then-y, which for geographic data means **longitude first**.
    That is the order WKT, GeoJSON and PostGIS all use, and it is the opposite of how
    latitude and longitude are usually spoken. Getting it backwards puts Zurich in the
    Indian Ocean, which is at least obvious; getting it backwards near the equator is
    not.

    Args:
        x: The x ordinate, or longitude.
        y: The y ordinate, or latitude.

    Returns:
        A point geometry, null where either ordinate is null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'lon': [-122.4194], 'lat': [37.7749]})
            >>> got = bt.st_as_text(bt.st_point(bt.col("lon"), bt.col("lat")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(-122.4194 37.7749)']}
    """
    return geo_call("st_point", value(x), value(y))


def st_point_z(x: Expr | float, y: Expr | float, z: Expr | float) -> Expr:
    """Build a 3D point from three ordinate columns.

    The z ordinate is carried through every function that preserves structure and
    ignored by every planar measurement: `st_area` of a 3D polygon is its area in the
    xy plane, matching PostGIS.

    Args:
        x: The x ordinate, or longitude.
        y: The y ordinate, or latitude.
        z: The elevation.

    Returns:
        A 3D point geometry, null where any ordinate is null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'x': [1.0], 'y': [2.0], 'z': [3.0]})
            >>> got = bt.st_as_text(bt.st_point_z(bt.col("x"), bt.col("y"), bt.col("z")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT Z(1 2 3)']}
    """
    return geo_call("st_point_z", value(x), value(y), value(z))


def st_make_line(start: Expr | str, end: Expr | str) -> Expr:
    """Build a two-position line joining two points.

    The segment primitive: origin-to-destination, sensor-to-target, before-and-after.
    For a longer chain, build the geometry as WKT or read it from a file — an
    N-position line has no natural expression form, because the positions live in N
    rows rather than N columns.

    Args:
        start: The first point.
        end: The second point.

    Returns:
        A two-position line, null where either input is null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'a': ['POINT(0 0)'], 'b': ['POINT(3 4)']})
            >>> got = bt.st_length(bt.st_make_line(bt.col("a"), bt.col("b")))
            >>> ds.select(v=got).to_pydict()
            {'v': [5.0]}
    """
    return geo_call("st_make_line", geometry(start), geometry(end))


def st_make_polygon(ring: Expr | str) -> Expr:
    """Build a polygon from a closed chain.

    The chain is closed for you if its last position does not repeat its first. The
    result has no holes: a polygon with holes has to come from a file or from WKT,
    since the holes are separate chains.

    Args:
        ring: A closed chain to use as the exterior ring.

    Returns:
        A polygon, or null when the chain has fewer than three positions.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'r': ['LINESTRING(0 0, 4 0, 4 4, 0 4, 0 0)']})
            >>> got = bt.st_area(bt.st_make_polygon(bt.col("r")))
            >>> ds.select(v=got).to_pydict()
            {'v': [16.0]}
    """
    return geo_call("st_make_polygon", geometry(ring))


def st_make_envelope(
    xmin: Expr | float, ymin: Expr | float, xmax: Expr | float, ymax: Expr | float
) -> Expr:
    """Build a rectangle from explicit bounds.

    The literal region filter: build the box once and intersect a whole column
    against it. Inverted bounds raise rather than producing an empty rectangle,
    because swapping two arguments is far more likely than wanting nothing.

    Args:
        xmin: Minimum x.
        ymin: Minimum y.
        xmax: Maximum x.
        ymax: Maximum y.

    Returns:
        A rectangle, null where any bound is null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'n': [1]})
            >>> got = bt.st_area(bt.st_make_envelope(0.0, 0.0, 2.0, 3.0))
            >>> ds.select(v=got).to_pydict()
            {'v': [6.0]}
    """
    return geo_call("st_make_envelope", value(xmin), value(ymin), value(xmax), value(ymax))


def st_collect(a: Expr | str, b: Expr | str) -> Expr:
    """Combine two geometries into one, without computing an overlay.

    Concatenation, not union: two adjacent polygons collected stay two polygons that
    happen to touch. That is the cheap, lossless operation, and it is what you want
    before a single `st_envelope` or `st_convex_hull` over the pair. The result is the
    narrowest container that fits — two points make a `MULTIPOINT`, a point and a line
    make a `GEOMETRYCOLLECTION`.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        A multi-geometry or collection, null where either input is null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'a': ['POINT(0 0)'], 'b': ['POINT(4 4)']})
            >>> got = bt.st_as_text(bt.st_collect(bt.col("a"), bt.col("b")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['MULTIPOINT((0 0), (4 4))']}
    """
    return geo_call("st_collect", geometry(a), geometry(b))


def st_centroid(geom: Expr | str) -> Expr:
    """The centre of mass of a geometry.

    Weighted by the highest dimension present: areas by area, lines by length, point
    sets by count. A collection of a polygon and a point therefore ignores the point.

    A centroid can fall **outside** its geometry — the centre of a crescent is in the
    gap, and the centre of a ring-shaped polygon is in the hole. When the result must
    lie on the geometry, use `st_point_on_surface`.

    Args:
        geom: The geometry.

    Returns:
        The centroid as a point, or null for an empty geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_as_text(bt.st_centroid(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(2 2)']}
    """
    return geo_call("st_centroid", geometry(geom))


def st_envelope(geom: Expr | str) -> Expr:
    """The axis-aligned bounding box of a geometry.

    Degenerate inputs degrade rather than producing a zero-area polygon: a point's
    envelope is a point and a horizontal line's is a line. A zero-area 'polygon' would
    be accepted by every areal predicate and answer all of them wrongly.

    Args:
        geom: The geometry.

    Returns:
        The bounding box as a geometry, or an empty polygon for an empty input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 3 4)']})
            >>> got = bt.st_as_text(bt.st_envelope(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POLYGON((0 0, 3 0, 3 4, 0 4, 0 0))']}
    """
    return geo_call("st_envelope", geometry(geom))


def st_boundary(geom: Expr | str) -> Expr:
    """The boundary of a geometry: rings for areas, endpoints for chains.

    A closed chain has no boundary at all, which is the topological fact that makes
    'closed' mean something — and it is why `st_boundary` of a ring is empty rather
    than its start vertex.

    Args:
        geom: The geometry.

    Returns:
        The boundary geometry, or an empty geometry when there is none.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 1 1)']})
            >>> got = bt.st_as_text(bt.st_boundary(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['MULTIPOINT((0 0), (1 1))']}
    """
    return geo_call("st_boundary", geometry(geom))


def st_convex_hull(geom: Expr | str) -> Expr:
    """The smallest convex shape containing a geometry.

    Degrades honestly: fewer than three distinct positions, or three collinear ones,
    yield a point or a line rather than a degenerate polygon. Cheaper than a buffer
    and tighter than an envelope, which makes it the middle rung of a filter ladder.

    Args:
        geom: The geometry.

    Returns:
        The hull, or a point or line when the input has no area.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['MULTIPOINT((0 0), (4 0), (4 4), (0 4), (2 2))']})
            >>> got = bt.st_area(bt.st_convex_hull(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': [16.0]}
    """
    return geo_call("st_convex_hull", geometry(geom))


def st_point_on_surface(geom: Expr | str) -> Expr:
    """A point guaranteed to lie on the geometry.

    Unlike a centroid, which can land in a hole or outside a crescent, this is always
    *on* the shape. It is what to label a polygon with, and what to use as a
    representative point in a spatial join that must not miss.

    Args:
        geom: The geometry.

    Returns:
        A point on the geometry, or null for an empty geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['POLYGON((0 0, 10 0, 10 2, 2 2, 2 8, 10 8, 10 10, 0 10, 0 0))']}
            ... )
            >>> got = bt.st_contains(bt.col("g"), bt.st_point_on_surface(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_point_on_surface", geometry(geom))


def st_buffer(geom: Expr | str, radius: Expr | float, quad_segs: Expr | int) -> Expr:
    """An approximate buffer of a geometry.

    **This is an approximation and over-estimates for a concave input.** It buffers
    each vertex with a regular polygon and takes the convex hull of the result, which
    is exact for a convex geometry up to the arc discretization and is the buffer *of
    the hull* otherwise.

    That makes it sound as a candidate filter feeding an exact `st_dwithin`, and wrong
    as a number to report. When the answer is 'is this within X', use `st_dwithin`,
    which is exact and cheaper.

    The radius is in the geometry's own units. On EPSG:4326 that is degrees, not
    metres; project first with `st_transform`.

    Args:
        geom: The geometry.
        radius: The buffer distance, in the geometry's own units.
        quad_segs: Segments per quarter circle; higher is smoother and larger.

    Returns:
        The buffered polygon, or an empty polygon for a non-positive radius.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(0 0)']})
            >>> got = bt.st_area(bt.st_buffer(bt.col("g"), 1.0, 16)).round(3)
            >>> ds.select(v=got).to_pydict()
            {'v': [3.137]}
    """
    return geo_call("st_buffer", geometry(geom), value(radius), value(quad_segs))


def st_expand(geom: Expr | str, dx: Expr | float, dy: Expr | float) -> Expr:
    """The bounding box of a geometry, grown on every side.

    The cheap 'everything near this' region: one rectangle, exact to compute and exact
    to index. Prefer it over `st_buffer` as the first stage of a proximity filter —
    it is both faster and, unlike the buffer, never under-covers.

    Args:
        geom: The geometry.
        dx: Horizontal growth on each side.
        dy: Vertical growth on each side.

    Returns:
        The grown bounding box as a rectangle, null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(5 5)']})
            >>> got = bt.st_as_text(bt.st_expand(bt.col("g"), 1.0, 2.0))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POLYGON((4 3, 6 3, 6 7, 4 7, 4 3))']}
    """
    return geo_call("st_expand", geometry(geom), value(dx), value(dy))
