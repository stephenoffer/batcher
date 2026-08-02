"""Measuring geometry: areas, lengths, distances, bearings.

The single most important thing on this page is that these come in two families and
mixing them up is silent.

**The planar functions** — `st_area`, `st_length`, `st_distance` — are Cartesian. They
treat coordinates as points on a flat plane and answer in whatever unit the coordinates
are stated in. On EPSG:4326 that unit is **degrees**, and a degree is not a distance: a
`st_distance` of 0.01 is 1.1 km north-south and anywhere from 1.1 km to nothing
east-west depending on latitude. This is exactly what PostGIS's `geometry` type does,
and it is what makes spatial joins affordable, because the planar metric is the one a
bounding box can bound.

**The geodesic functions** — the `_sphere` and `_spheroid` suffixes — answer in metres
on the Earth, and take longitude/latitude in degrees. They are the ones to use when the
number is the deliverable.

The third option, and usually the best one for a whole pipeline, is to `st_transform`
into a projected CRS once and use the planar functions everywhere after. Then the
coordinate unit *is* metres and the two families agree.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry

__all__ = [
    "st_area",
    "st_area_spheroid",
    "st_azimuth",
    "st_distance",
    "st_distance_sphere",
    "st_distance_spheroid",
    "st_hausdorff_distance",
    "st_length",
    "st_length_spheroid",
    "st_max_distance",
    "st_perimeter",
    "st_perimeter_spheroid",
]


def st_area(geom: Expr | str) -> Expr:
    """The planar area of a geometry.

    Holes are subtracted. Points and chains have zero area rather than null, matching
    PostGIS, so summing a mixed column is well defined.

    In squared *coordinate* units. On lon/lat that is square degrees, which is not a
    quantity anyone wants; use `st_area_spheroid` or project first.

    Args:
        geom: The geometry.

    Returns:
        The area in squared coordinate units; 0 for a non-areal geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         'g': ['POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4))'],
            ...     }
            ... )
            >>> got = bt.st_area(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [96.0]}
    """
    return geo_call("st_area", geometry(geom))


def st_length(geom: Expr | str) -> Expr:
    """The total length of a geometry's chains.

    Polygon boundaries are **excluded** — that is `st_perimeter`. PostGIS draws the
    same line, and it is what keeps a mixed geometry column from silently summing two
    different quantities into one number.

    Args:
        geom: The geometry.

    Returns:
        The total chain length; 0 for a polygon or point.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['LINESTRING(0 0, 3 4)', 'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']}
            ... )
            >>> got = bt.st_length(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [5.0, 0.0]}
    """
    return geo_call("st_length", geometry(geom))


def st_perimeter(geom: Expr | str) -> Expr:
    """The total boundary length of a geometry's polygons.

    Holes count: a donut's perimeter includes the inner ring, because that ring is
    part of the boundary.

    Args:
        geom: The geometry.

    Returns:
        The total polygon boundary length; 0 for a non-areal geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))', 'LINESTRING(0 0, 3 4)']}
            ... )
            >>> got = bt.st_perimeter(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': [16.0, 0.0]}
    """
    return geo_call("st_perimeter", geometry(geom))


def st_distance(a: Expr | str, b: Expr | str) -> Expr:
    """The smallest planar distance between two geometries.

    0 when the geometries touch or overlap, including when one contains the other.
    Null when either is empty, because there is no pair of points to measure between.

    In coordinate units. See the module note about degrees.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        The distance in coordinate units, 0 when they intersect, or null when either
        is empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'], 'b': ['POINT(6 2)']}
            ... )
            >>> got = bt.st_distance(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [2.0]}
    """
    return geo_call("st_distance", geometry(a), geometry(b))


def st_max_distance(a: Expr | str, b: Expr | str) -> Expr:
    """The largest distance between any pair of vertices of two geometries.

    Vertex-to-vertex, so it is exact for point inputs and an under-estimate for
    curved boundaries between vertices. The usual use is bounding how far apart two
    clusters can possibly be.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        The largest vertex-to-vertex distance, or null when either is empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['MULTIPOINT((0 0), (1 1))'], 'b': ['MULTIPOINT((3 4), (0 0))']}
            ... )
            >>> got = bt.st_max_distance(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [5.0]}
    """
    return geo_call("st_max_distance", geometry(a), geometry(b))


def st_hausdorff_distance(a: Expr | str, b: Expr | str) -> Expr:
    """How far apart two shapes are at their worst-matching point.

    The largest distance from any vertex of one geometry to the nearest point of the
    other, symmetrized. The standard measure of 'how similar are these two shapes',
    used to compare a simplified geometry against its original.

    **Discrete**: it samples only the vertices, like PostGIS. On densified inputs it
    converges to the continuous value; on sparse ones it under-reports. Run
    `st_segmentize` first when that matters.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        The symmetric discrete Hausdorff distance, or null when either is empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['LINESTRING(0 0, 10 0)'], 'b': ['LINESTRING(0 1, 10 1)']}
            ... )
            >>> got = bt.st_hausdorff_distance(bt.col("a"), bt.col("b"))
            >>> ds.select(v=got).to_pydict()
            {'v': [1.0]}
    """
    return geo_call("st_hausdorff_distance", geometry(a), geometry(b))


def st_azimuth(a: Expr | str, b: Expr | str) -> Expr:
    """The planar bearing from one point to another.

    Radians clockwise from north, in `[0, 2*pi)`, and null for two identical points
    because coincident points have no direction.

    Planar: it uses the coordinate axes, so on lon/lat it is only correct near the
    equator. There is no geodesic bearing function here; project to a local CRS with
    `st_transform` when the bearing has to be right at high latitude.

    Args:
        a: The origin point.
        b: The target point.

    Returns:
        The angle in radians clockwise from north, or null when the points coincide.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'a': ['POINT(0 0)'], 'b': ['POINT(1 0)']})
            >>> got = bt.st_azimuth(bt.col("a"), bt.col("b")).round(6)
            >>> ds.select(v=got).to_pydict()
            {'v': [1.570796]}
    """
    return geo_call("st_azimuth", geometry(a), geometry(b))


def st_distance_sphere(a: Expr | str, b: Expr | str) -> Expr:
    """The distance between two lon/lat geometries, in metres on a sphere.

    Haversine on the mean-radius sphere: accurate to about 0.5%, cheap, and with no
    failure mode. Coordinates must be longitude and latitude in degrees.

    Measured **vertex to vertex**, not between the nearest points of the shapes. For
    two points — the overwhelming majority of proximity queries — those coincide
    exactly. For extended geometries it over-reports by at most a segment length, so
    it is an upper bound and safe to filter with; `st_segmentize` first when the answer
    must be tight.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        The distance in metres, 0 when the geometries intersect, or null when either
        is empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POINT(-122.4194 37.7749)'], 'b': ['POINT(-0.1278 51.5074)']}
            ... )
            >>> got = (bt.st_distance_sphere(bt.col("a"), bt.col("b")) / 1000).round(0)
            >>> ds.select(v=got).to_pydict()
            {'v': [8616.0]}
    """
    return geo_call("st_distance_sphere", geometry(a), geometry(b))


def st_distance_spheroid(a: Expr | str, b: Expr | str) -> Expr:
    """The distance between two lon/lat geometries, in metres on the WGS 84 ellipsoid.

    Vincenty's inverse formula: accurate to under a millimetre, iterative, and roughly
    an order of magnitude slower than `st_distance_sphere`. Use it when the number is
    the deliverable and the sphere's 0.5% is too much.

    Near-antipodal pairs, where the iteration does not converge, are skipped rather
    than reported wrong; if every pair is antipodal the result is null.

    Args:
        a: The first geometry.
        b: The second geometry.

    Returns:
        The distance in metres, 0 when the geometries intersect, or null when either
        is empty.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'a': ['POINT(-0.1278 51.5074)'], 'b': ['POINT(-74.006 40.7128)']}
            ... )
            >>> got = (bt.st_distance_spheroid(bt.col("a"), bt.col("b")) / 1000).round(1)
            >>> ds.select(v=got).to_pydict()
            {'v': [5585.2]}
    """
    return geo_call("st_distance_spheroid", geometry(a), geometry(b))


def st_area_spheroid(geom: Expr | str) -> Expr:
    """The geodesic area of a lon/lat geometry, in square metres.

    Computed from the spherical excess, so it is correct for a ring of any size —
    including one spanning a hemisphere, where projecting to a plane and taking the
    shoelace area is wrong by an unbounded factor. Holes are subtracted.

    Args:
        geom: A lon/lat geometry.

    Returns:
        The area in square metres; 0 for a non-areal geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))']})
            >>> got = (bt.st_area_spheroid(bt.col("g")) / 1e6).round(0)
            >>> ds.select(v=got).to_pydict()
            {'v': [12364.0]}
    """
    return geo_call("st_area_spheroid", geometry(geom))


def st_length_spheroid(geom: Expr | str) -> Expr:
    """The geodesic length of a lon/lat geometry's chains, in metres.

    Sums the great-circle distance of each segment, so a long segment is measured
    along the sphere rather than through it. Polygon boundaries are excluded, matching
    `st_length`.

    Args:
        geom: A lon/lat geometry.

    Returns:
        The chain length in metres; 0 for a polygon or point.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 1 0)']})
            >>> got = (bt.st_length_spheroid(bt.col("g")) / 1000).round(1)
            >>> ds.select(v=got).to_pydict()
            {'v': [111.2]}
    """
    return geo_call("st_length_spheroid", geometry(geom))


def st_perimeter_spheroid(geom: Expr | str) -> Expr:
    """The geodesic perimeter of a lon/lat geometry's polygons, in metres.

    Holes count, matching `st_perimeter`.

    Args:
        geom: A lon/lat geometry.

    Returns:
        The boundary length in metres; 0 for a non-areal geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))']})
            >>> got = (bt.st_perimeter_spheroid(bt.col("g")) / 1000).round(0)
            >>> ds.select(v=got).to_pydict()
            {'v': [445.0]}
    """
    return geo_call("st_perimeter_spheroid", geometry(geom))
