"""Moving, reshaping and reprojecting a geometry.

Three groups live here. The **affine transforms** — translate, scale, rotate, affine —
move positions without touching structure, so they cannot turn a valid geometry into an
invalid one. **Normalization** — winding, dimension, vertex order, repeated points — is
the set of fixes for a column that came from somewhere with different conventions.

And **`st_transform`**, which is the one that matters most. Batcher supports a
deliberately small set of reference systems rather than shipping a projection database,
and it **refuses an unsupported EPSG code** rather than silently returning the input:

| EPSG | System | Use it for |
| --- | --- | --- |
| 4326 | WGS 84 lon/lat degrees | storage, interchange, the geodesic functions |
| 3857 | Web Mercator metres | map tiles, anything drawn on a slippy map |
| 326xx / 327xx | UTM zone metres | local distance and area, to a metre, within a zone |
| 6933 | Cylindrical equal area metres | density comparisons across latitudes |

Every one of these is on the WGS 84 datum, so a conversion between them is a projection
change and loses nothing. Reprojecting between *datums* — NAD 27, OSGB 36 — needs a grid
shift that is not here, and those codes are rejected.

A projection moves positions but does not add them, so a long straight segment
reprojects to a straight segment that should have been curved. Run `st_segmentize`
first when a segment spans degrees.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry, value

__all__ = [
    "st_affine",
    "st_flip_coordinates",
    "st_force_2d",
    "st_force_3d",
    "st_force_polygon_ccw",
    "st_force_polygon_cw",
    "st_remove_repeated_points",
    "st_reverse",
    "st_rotate",
    "st_scale",
    "st_segmentize",
    "st_simplify",
    "st_snap_to_grid",
    "st_transform",
    "st_translate",
]


def st_simplify(geom: Expr | str, tolerance: Expr | float) -> Expr:
    """Reduce a geometry's vertex count with Douglas-Peucker.

    The highest-leverage thing you can do to a large geometry column: vertex count
    drives the cost of every predicate, every shuffle and every byte written.

    A ring is never collapsed below a triangle — dropping it would turn a polygon
    column into a mix of polygons and nothing, and every areal predicate would then
    answer differently for the survivors. Compare the result against the original with
    `st_hausdorff_distance` to see what the tolerance cost you.

    Args:
        geom: The geometry.
        tolerance: The maximum deviation, in coordinate units.

    Returns:
        The simplified geometry, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 1 0.001, 2 0, 3 0)']})
            >>> got = bt.st_num_points(bt.st_simplify(bt.col("g"), 0.01))
            >>> ds.select(v=got).to_pydict()
            {'v': [2]}
    """
    return geo_call("st_simplify", geometry(geom), value(tolerance))


def st_reverse(geom: Expr | str) -> Expr:
    """Reverse the position order of every chain and ring.

    Direction is meaningful for a route and for a ring's winding, and this is how you
    flip it. For winding specifically, prefer `st_force_polygon_ccw`, which is
    idempotent.

    Args:
        geom: The geometry.

    Returns:
        The geometry with every chain reversed.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 1 1, 2 2)']})
            >>> got = bt.st_as_text(bt.st_reverse(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['LINESTRING(2 2, 1 1, 0 0)']}
    """
    return geo_call("st_reverse", geometry(geom))


def st_force_2d(geom: Expr | str) -> Expr:
    """Drop the z ordinate from a geometry.

    Halves the storage of a 3D column whose elevations nothing reads, and makes a
    mixed 2D/3D column uniform so `st_has_z` stops being a thing callers must check.

    Args:
        geom: The geometry.

    Returns:
        The geometry with elevations dropped.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT Z(1 2 3)']})
            >>> got = bt.st_as_text(bt.st_force_2d(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(1 2)']}
    """
    return geo_call("st_force2d", geometry(geom))


def st_force_3d(geom: Expr | str, z: Expr | float) -> Expr:
    """Give a 2D geometry a constant elevation.

    The complement of `st_force_2d`, for making a column uniform in the other
    direction — usually so it can be written to a format that requires 3D.

    Args:
        geom: The geometry.
        z: The elevation to set on every position.

    Returns:
        The geometry with a constant elevation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_as_text(bt.st_force_3d(bt.col("g"), 10.0))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT Z(2 2 10)']}
    """
    return geo_call("st_force3d", geometry(geom), value(z))


def st_force_polygon_ccw(geom: Expr | str) -> Expr:
    """Force every polygon exterior counter-clockwise.

    GeoJSON's right-hand rule. Holes are forced the opposite way, because that
    opposition is the encoding of 'this ring subtracts' — forcing them the same way
    is how a renderer ends up punching holes in the wrong places.

    Args:
        geom: The geometry.

    Returns:
        The geometry with counter-clockwise exteriors and clockwise holes.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))']})
            >>> got = bt.st_area(bt.st_force_polygon_ccw(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': [16.0]}
    """
    return geo_call("st_force_polygon_ccw", geometry(geom))


def st_force_polygon_cw(geom: Expr | str) -> Expr:
    """Force every polygon exterior clockwise.

    The Shapefile convention, and the mirror of `st_force_polygon_ccw`. Winding is
    not an area change; both leave `st_area` untouched.

    Args:
        geom: The geometry.

    Returns:
        The geometry with clockwise exteriors and counter-clockwise holes.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_area(bt.st_force_polygon_cw(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': [16.0]}
    """
    return geo_call("st_force_polygon_cw", geometry(geom))


def st_flip_coordinates(geom: Expr | str) -> Expr:
    """Swap the x and y of every position.

    The fix for the most common geospatial data bug there is: a lat/lon source loaded
    as lon/lat. Everything is in the wrong hemisphere and nothing errors, because both
    orderings are valid coordinates.

    Args:
        geom: The geometry.

    Returns:
        The geometry with x and y swapped.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(37.7749 -122.4194)']})
            >>> got = bt.st_as_text(bt.st_flip_coordinates(bt.col("g")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(-122.4194 37.7749)']}
    """
    return geo_call("st_flip_coordinates", geometry(geom))


def st_translate(geom: Expr | str, dx: Expr | float, dy: Expr | float) -> Expr:
    """Shift a geometry by an offset.

    Structure is untouched, so ring nesting, closure and vertex count all survive.

    Args:
        geom: The geometry.
        dx: The x shift.
        dy: The y shift.

    Returns:
        The shifted geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_as_text(bt.st_translate(bt.col("g"), 3.0, 4.0))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(5 6)']}
    """
    return geo_call("st_translate", geometry(geom), value(dx), value(dy))


def st_scale(geom: Expr | str, sx: Expr | float, sy: Expr | float) -> Expr:
    """Scale a geometry about the origin.

    About the origin, not about the geometry's own centre. To scale in place,
    translate to the origin, scale, and translate back.

    Args:
        geom: The geometry.
        sx: The x factor.
        sy: The y factor.

    Returns:
        The scaled geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_area(bt.st_scale(bt.col("g"), 2.0, 3.0))
            >>> ds.select(v=got).to_pydict()
            {'v': [96.0]}
    """
    return geo_call("st_scale", geometry(geom), value(sx), value(sy))


def st_rotate(geom: Expr | str, radians: Expr | float) -> Expr:
    """Rotate a geometry counter-clockwise about the origin.

    Radians, counter-clockwise, about the origin — matching PostGIS `ST_Rotate`.

    Args:
        geom: The geometry.
        radians: The counter-clockwise angle, in radians.

    Returns:
        The rotated geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(1 0)']})
            >>> got = bt.st_x(bt.st_rotate(bt.col("g"), 3.14159265358979)).round(6)
            >>> ds.select(v=got).to_pydict()
            {'v': [-1.0]}
    """
    return geo_call("st_rotate", geometry(geom), value(radians))


def st_affine(
    geom: Expr | str,
    a: Expr | float,
    b: Expr | float,
    d: Expr | float,
    e: Expr | float,
    xoff: Expr | float,
    yoff: Expr | float,
) -> Expr:
    """Apply a general 2D affine transform.

    `x' = a*x + b*y + xoff`, `y' = d*x + e*y + yoff`. Argument names match PostGIS
    `ST_Affine`, so a transform matrix carries over from an existing pipeline without
    re-deriving it. Every other transform on this page is a special case of it.

    Args:
        geom: The geometry.
        a: The x-from-x coefficient.
        b: The x-from-y coefficient.
        d: The y-from-x coefficient.
        e: The y-from-y coefficient.
        xoff: The x offset.
        yoff: The y offset.

    Returns:
        The transformed geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_as_text(bt.st_affine(bt.col("g"), 1.0, 0.0, 0.0, 1.0, 5.0, 6.0))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(7 8)']}
    """
    return geo_call(
        "st_affine",
        geometry(geom),
        value(a),
        value(b),
        value(d),
        value(e),
        value(xoff),
        value(yoff),
    )


def st_snap_to_grid(geom: Expr | str, size: Expr | float) -> Expr:
    """Round every position onto a grid.

    The compression lever: snapping to a grid coarser than the data's real precision
    makes vertices repeat, which makes the column compress and makes an equality join
    on geometry actually hit.

    Snapping can move two distinct vertices onto one position. Follow it with
    `st_remove_repeated_points` when the result feeds an areal predicate.

    Args:
        geom: The geometry.
        size: The grid cell size.

    Returns:
        The snapped geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(1.234 5.678)']})
            >>> got = bt.st_as_text(bt.st_snap_to_grid(bt.col("g"), 0.5))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(1 5.5)']}
    """
    return geo_call("st_snap_to_grid", geometry(geom), value(size))


def st_segmentize(geom: Expr | str, max_length: Expr | float) -> Expr:
    """Insert positions so no segment is longer than a bound.

    The prerequisite for reprojecting or geodesically measuring a long segment: a
    straight line in one CRS is curved in another, so a two-position transatlantic
    segment reprojects to a visibly wrong path. Segments are split evenly rather than
    in `max_length` steps with a short remainder.

    Args:
        geom: The geometry.
        max_length: The maximum segment length, in coordinate units.

    Returns:
        The densified geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 10 0)']})
            >>> got = bt.st_num_points(bt.st_segmentize(bt.col("g"), 3.0))
            >>> ds.select(v=got).to_pydict()
            {'v': [5]}
    """
    return geo_call("st_segmentize", geometry(geom), value(max_length))


def st_remove_repeated_points(geom: Expr | str, tolerance: Expr | float) -> Expr:
    """Drop consecutive positions closer together than a tolerance.

    A tolerance of 0 drops exact duplicates only. Rings stay closed: thinning that
    would drop a ring below a triangle leaves it alone rather than producing something
    no areal predicate can read.

    Args:
        geom: The geometry.
        tolerance: Merge consecutive positions closer than this.

    Returns:
        The thinned geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['LINESTRING(0 0, 0 0, 1 1, 1 1, 2 2)']})
            >>> got = bt.st_num_points(bt.st_remove_repeated_points(bt.col("g"), 0.0))
            >>> ds.select(v=got).to_pydict()
            {'v': [3]}
    """
    return geo_call("st_remove_repeated_points", geometry(geom), value(tolerance))


def st_transform(geom: Expr | str, from_srid: Expr | int, to_srid: Expr | int) -> Expr:
    """Reproject a geometry between spatial reference systems.

    The single highest-value function on this page for anyone doing real measurement.
    Project once into UTM or the equal-area system, and every planar function after it
    answers in metres — which is both correct and far cheaper than calling the geodesic
    functions per row.

    An unsupported EPSG code raises, naming the supported set. See the module note for
    which systems those are and why the list is short.

    Args:
        geom: The geometry.
        from_srid: The EPSG code the coordinates are currently in.
        to_srid: The EPSG code to convert to.

    Returns:
        The reprojected geometry, labelled with `to_srid`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(-122.4194 37.7749)']})
            >>> got = bt.st_x(bt.st_transform(bt.col("g"), 4326, 3857)).round(0)
            >>> ds.select(v=got).to_pydict()
            {'v': [-13627665.0]}
    """
    return geo_call("st_transform", geometry(geom), value(from_srid), value(to_srid))
