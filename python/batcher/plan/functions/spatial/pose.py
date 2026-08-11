"""Applying a whole pose — a rotation and a translation — to a point.

A pose is seven numbers: the translation ``(tx, ty, tz)`` first, then the rotation
``(qx, qy, qz, qw)``. Read a pose named ``world_from_lidar`` as "where the lidar is,
expressed in the world frame", and `se3_transform_x` and its siblings take a point *out
of* the lidar frame and *into* the world frame. Naming a pose for the two frames it
relates, target first, is the ROS ``tf2`` convention, and it is what makes a chain of
them compose by cancelling adjacent names.

The transform rotates and then translates. The other order is a different transform and
gets a different, wrong answer for every point that is not at the origin.

These six functions are the innermost loop of an autonomous-driving pipeline: a single
lidar sweep is a hundred thousand points and a log is tens of thousands of sweeps. They
are also plain `Float64` projections, so the engine pushes them down, spills them and
shuffles them exactly like arithmetic.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.spatial._build import Numeric, spatial_call

__all__ = [
    "se3_inverse_transform_x",
    "se3_inverse_transform_y",
    "se3_inverse_transform_z",
    "se3_transform_x",
    "se3_transform_y",
    "se3_transform_z",
]


def se3_transform_x(
    tx: Numeric,
    ty: Numeric,
    tz: Numeric,
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Move a point into the pose's parent frame and take its X coordinate.

    The function that turns a lidar return into a world-frame point. Rotation is applied
    before translation.

    Args:
        tx: The pose's X translation.
        ty: The pose's Y translation.
        tz: The pose's Z translation.
        qx: The pose's rotation X component.
        qy: The pose's rotation Y component.
        qz: The pose's rotation Z component.
        qw: The pose's rotation scalar component.
        px: The point's X coordinate, in the pose's own frame.
        py: The point's Y coordinate, in the pose's own frame.
        pz: The point's Z coordinate, in the pose's own frame.

    Returns:
        The point's X coordinate in the parent frame, or null if the rotation is all
        zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"tx": [10.0], "ty": [0.0], "tz": [0.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> pose = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
            >>> ds.select(v=bt.se3_transform_x(*pose, "x", "y", "z")).to_pydict()
            {'v': [11.0]}
    """
    return spatial_call("se3_transform_x", tx, ty, tz, qx, qy, qz, qw, px, py, pz)


def se3_transform_y(
    tx: Numeric,
    ty: Numeric,
    tz: Numeric,
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Move a point into the pose's parent frame and take its Y coordinate.

    Args:
        tx: The pose's X translation.
        ty: The pose's Y translation.
        tz: The pose's Z translation.
        qx: The pose's rotation X component.
        qy: The pose's rotation Y component.
        qz: The pose's rotation Z component.
        qw: The pose's rotation scalar component.
        px: The point's X coordinate, in the pose's own frame.
        py: The point's Y coordinate, in the pose's own frame.
        pz: The point's Z coordinate, in the pose's own frame.

    Returns:
        The point's Y coordinate in the parent frame, or null if the rotation is all
        zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"tx": [0.0], "ty": [10.0], "tz": [0.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> pose = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
            >>> ds.select(v=bt.se3_transform_y(*pose, "x", "y", "z")).to_pydict()
            {'v': [12.0]}
    """
    return spatial_call("se3_transform_y", tx, ty, tz, qx, qy, qz, qw, px, py, pz)


def se3_transform_z(
    tx: Numeric,
    ty: Numeric,
    tz: Numeric,
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Move a point into the pose's parent frame and take its Z coordinate.

    Args:
        tx: The pose's X translation.
        ty: The pose's Y translation.
        tz: The pose's Z translation.
        qx: The pose's rotation X component.
        qy: The pose's rotation Y component.
        qz: The pose's rotation Z component.
        qw: The pose's rotation scalar component.
        px: The point's X coordinate, in the pose's own frame.
        py: The point's Y coordinate, in the pose's own frame.
        pz: The point's Z coordinate, in the pose's own frame.

    Returns:
        The point's Z coordinate in the parent frame, or null if the rotation is all
        zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"tx": [0.0], "ty": [0.0], "tz": [10.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> pose = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
            >>> ds.select(v=bt.se3_transform_z(*pose, "x", "y", "z")).to_pydict()
            {'v': [13.0]}
    """
    return spatial_call("se3_transform_z", tx, ty, tz, qx, qy, qz, qw, px, py, pz)


def se3_inverse_transform_x(
    tx: Numeric,
    ty: Numeric,
    tz: Numeric,
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Move a parent-frame point into the pose's own frame and take its X coordinate.

    The inverse of `se3_transform_x`, and the direction that answers "where is this
    world-frame obstacle relative to the vehicle". Subtracts the translation, then
    rotates by the inverse rotation; doing those two in the other order is the mistake
    this function exists to prevent.

    Args:
        tx: The pose's X translation.
        ty: The pose's Y translation.
        tz: The pose's Z translation.
        qx: The pose's rotation X component.
        qy: The pose's rotation Y component.
        qz: The pose's rotation Z component.
        qw: The pose's rotation scalar component.
        px: The point's X coordinate, in the parent frame.
        py: The point's Y coordinate, in the parent frame.
        pz: The point's Z coordinate, in the parent frame.

    Returns:
        The point's X coordinate in the pose's own frame, or null if the rotation is all
        zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"tx": [10.0], "ty": [0.0], "tz": [0.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [11.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> pose = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
            >>> ds.select(v=bt.se3_inverse_transform_x(*pose, "x", "y", "z")).to_pydict()
            {'v': [1.0]}
    """
    return spatial_call("se3_inverse_transform_x", tx, ty, tz, qx, qy, qz, qw, px, py, pz)


def se3_inverse_transform_y(
    tx: Numeric,
    ty: Numeric,
    tz: Numeric,
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Move a parent-frame point into the pose's own frame and take its Y coordinate.

    Args:
        tx: The pose's X translation.
        ty: The pose's Y translation.
        tz: The pose's Z translation.
        qx: The pose's rotation X component.
        qy: The pose's rotation Y component.
        qz: The pose's rotation Z component.
        qw: The pose's rotation scalar component.
        px: The point's X coordinate, in the parent frame.
        py: The point's Y coordinate, in the parent frame.
        pz: The point's Z coordinate, in the parent frame.

    Returns:
        The point's Y coordinate in the pose's own frame, or null if the rotation is all
        zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"tx": [0.0], "ty": [10.0], "tz": [0.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [12.0], "z": [3.0]}
            ... )
            >>> pose = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
            >>> ds.select(v=bt.se3_inverse_transform_y(*pose, "x", "y", "z")).to_pydict()
            {'v': [2.0]}
    """
    return spatial_call("se3_inverse_transform_y", tx, ty, tz, qx, qy, qz, qw, px, py, pz)


def se3_inverse_transform_z(
    tx: Numeric,
    ty: Numeric,
    tz: Numeric,
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Move a parent-frame point into the pose's own frame and take its Z coordinate.

    Args:
        tx: The pose's X translation.
        ty: The pose's Y translation.
        tz: The pose's Z translation.
        qx: The pose's rotation X component.
        qy: The pose's rotation Y component.
        qz: The pose's rotation Z component.
        qw: The pose's rotation scalar component.
        px: The point's X coordinate, in the parent frame.
        py: The point's Y coordinate, in the parent frame.
        pz: The point's Z coordinate, in the parent frame.

    Returns:
        The point's Z coordinate in the pose's own frame, or null if the rotation is all
        zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"tx": [0.0], "ty": [0.0], "tz": [10.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [13.0]}
            ... )
            >>> pose = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
            >>> ds.select(v=bt.se3_inverse_transform_z(*pose, "x", "y", "z")).to_pydict()
            {'v': [3.0]}
    """
    return spatial_call("se3_inverse_transform_z", tx, ty, tz, qx, qy, qz, qw, px, py, pz)
