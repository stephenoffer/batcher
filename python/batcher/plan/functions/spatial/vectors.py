"""Distances and directions between points in three dimensions.

The measurements a point cloud is filtered and bucketed by, once its points are in
whichever frame the question is asked in. A lidar return's *range* decides whether it is
a real obstacle or sensor noise, its *azimuth* decides which sector of the sweep it came
from, and its *elevation* decides whether it is ground, a vehicle or an overhead sign.

Everything here composes existing arithmetic and introduces no IR node of its own. Each
is a couple of multiplications and a transcendental function, which the engine already
evaluates as fast as a fused kernel would; the value these add is a correct, named,
documented spelling of a formula that is otherwise retyped per query and gets the
argument order of `atan2` wrong about half the time.

These are Cartesian and answer in the coordinate column's own units, which for a
robotics log means metres and radians. They are not the geodesic functions: a
`lat`/`lon` column wants ``st_distance_sphere`` from the geospatial family instead.
"""

from __future__ import annotations

from batcher.plan.expr_ir import atan2
from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.spatial._build import Numeric, Point, value

__all__ = ["azimuth_3d", "distance_3d", "elevation_3d", "norm_3d", "voxel_index"]


def _squared_norm(p: Point) -> Expr:
    """The sum of the three components squared."""
    x, y, z = (value(c) for c in p)
    return x * x + y * y + z * z


def norm_3d(point: Point) -> Expr:
    """Measure how far a point is from the origin.

    A lidar return's range, in the sensor frame. The cheapest and most effective filter
    a point cloud has: returns beyond the sensor's rated range are noise, and returns
    inside a metre or two are usually the vehicle itself.

    Comparing a range against a threshold is faster if you square the threshold and
    compare against the squared distance instead, which skips a square root per point.
    That is worth doing in an inner loop and not worth doing anywhere else.

    Args:
        point: The point, as ``(x, y, z)``.

    Returns:
        The Euclidean distance from the origin, in the coordinates' own units.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [3.0], "y": [4.0], "z": [12.0]})
            >>> ds.select(r=bt.norm_3d(("x", "y", "z"))).to_pydict()
            {'r': [13.0]}
    """
    return _squared_norm(point).sqrt()


def distance_3d(a: Point, b: Point) -> Expr:
    """Measure the straight-line distance between two points.

    Both points must already be in the same frame. Distances between points in different
    frames are the most common wrong answer in this whole area, and nothing here can
    detect it — move one of them first with ``se3_transform``.

    Args:
        a: The first point, as ``(x, y, z)``.
        b: The second point, as ``(x, y, z)``.

    Returns:
        The Euclidean distance, in the coordinates' own units.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"ax": [0.0], "ay": [0.0], "az": [0.0],
            ...      "bx": [3.0], "by": [4.0], "bz": [0.0]}
            ... )
            >>> got = bt.distance_3d(("ax", "ay", "az"), ("bx", "by", "bz"))
            >>> ds.select(d=got).to_pydict()
            {'d': [5.0]}
    """
    delta = tuple(value(bi) - value(ai) for ai, bi in zip(a, b, strict=True))
    return _squared_norm(delta).sqrt()


def azimuth_3d(point: Point) -> Expr:
    """Measure a point's bearing in the horizontal plane.

    The angle from the positive X axis, turning towards positive Y, which is the
    right-handed convention a robotics frame uses throughout. Bucketing a sweep by
    azimuth is how you split it into the sectors a rotating lidar actually measures in.

    Args:
        point: The point, as ``(x, y, z)``. The Z component is ignored.

    Returns:
        The bearing in radians on ``[-pi, pi]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0], "y": [0.0], "z": [5.0]})
            >>> ds.select(a=bt.azimuth_3d(("x", "y", "z"))).to_pydict()
            {'a': [0.0]}
    """
    x, y, _ = (value(c) for c in point)
    return atan2(y, x)


def elevation_3d(point: Point) -> Expr:
    """Measure how far a point rises above the horizontal plane.

    The angle from the XY plane towards positive Z. In a sensor frame it separates
    ground returns from vehicles and from overhead structure, which is the first cut
    almost every point-cloud pipeline makes.

    Args:
        point: The point, as ``(x, y, z)``.

    Returns:
        The elevation in radians on ``[-pi/2, pi/2]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0], "y": [0.0], "z": [0.0]})
            >>> ds.select(e=bt.elevation_3d(("x", "y", "z"))).to_pydict()
            {'e': [0.0]}
    """
    x, y, z = (value(c) for c in point)
    return atan2(z, (x * x + y * y).sqrt())


def voxel_index(point: Point, size: Numeric, *, prefix: str = "") -> dict[str, Expr]:
    """Put a point in a cubic bin, as three named integer columns.

    Voxel downsampling is the first thing almost every point-cloud pipeline does: a lidar
    sweep is far denser near the sensor than it is far away, and reducing each occupied
    cube to one point makes the density uniform and the cloud an order of magnitude
    smaller. In Batcher that is a `group_by` over these three columns, so the whole
    operation is native, spills, and distributes with no special handling::

        cloud.group_by(**bt.voxel_index(("x", "y", "z"), 0.2)).agg(
            x=col("x").mean(), y=col("y").mean(), z=col("z").mean()
        )

    The bin index is a **floor**, not a truncation, and the result is an **integer**.
    Both matter. Truncating toward zero would make the cell straddling the origin twice
    as wide as every other cell, which is a real and famously hard-to-see defect in a
    downsampled cloud. And a float group key can split one group in two — the value that
    ought to be a single key arrives as two representations that do not compare equal —
    so the cast is what makes this safe to group by, distributed or not.

    Bin the points in whichever frame the question is asked in. Binning sensor-frame
    coordinates and then transforming gives different cells than transforming first,
    because the grid is fixed to the frame you binned in.

    Args:
        point: The point, as ``(x, y, z)``.
        size: The cube's edge length, in the coordinates' own units.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``ix``/``iy``/``iz`` (each with `prefix`) to the bin index along
        that axis.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"x": [0.05, 0.15, -0.05], "y": [0.0, 0.0, 0.0], "z": [0.0, 0.0, 0.0]}
            ... )
            >>> ds.select(**bt.voxel_index(("x", "y", "z"), 0.1)).to_pydict()
            {'ix': [0, 1, -1], 'iy': [0, 0, 0], 'iz': [0, 0, 0]}
    """
    edge = value(size)
    return {
        f"{prefix}i{axis}": (value(c) / edge).floor().cast("int64")
        for axis, c in zip("xyz", point, strict=True)
    }
