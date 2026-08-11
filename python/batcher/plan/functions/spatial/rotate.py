"""Composing rotations, comparing them, interpolating between them, and applying one.

The four things you do *with* a rotation once you have it. Reading one apart and
building one live in `quaternion`; the same operations with a translation attached live
in `pose`.

`quat_slerp_*` deserves singling out. Poses arrive at the localizer's rate and
measurements arrive at each sensor's, so almost every measurement needs the pose
*between* two logged poses. Interpolating the four components independently and
renormalizing is the obvious alternative and it is wrong: it sweeps the angle at a
non-constant rate, which shows up as a lidar sweep that bends.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.spatial._build import Numeric, spatial_call

__all__ = [
    "quat_angular_distance",
    "quat_inverse_rotate_x",
    "quat_inverse_rotate_y",
    "quat_inverse_rotate_z",
    "quat_multiply_w",
    "quat_multiply_x",
    "quat_multiply_y",
    "quat_multiply_z",
    "quat_rotate_x",
    "quat_rotate_y",
    "quat_rotate_z",
    "quat_slerp_w",
    "quat_slerp_x",
    "quat_slerp_y",
    "quat_slerp_z",
]


def quat_multiply_x(
    ax: Numeric,
    ay: Numeric,
    az: Numeric,
    aw: Numeric,
    bx: Numeric,
    by: Numeric,
    bz: Numeric,
    bw: Numeric,
) -> Expr:
    """Take the X component of two rotations applied in turn.

    The result applies ``b`` first and then ``a``, which is the order function
    composition and matrix multiplication use and the *opposite* of the order the frame
    names read in. Going from a sensor frame to the world through the vehicle multiplies
    ``world_from_ego`` by ``ego_from_sensor``, in that order.

    Args:
        ax: The second rotation's X component.
        ay: The second rotation's Y component.
        az: The second rotation's Z component.
        aw: The second rotation's scalar component.
        bx: The first rotation's X component.
        by: The first rotation's Y component.
        bz: The first rotation's Z component.
        bw: The first rotation's scalar component.

    Returns:
        The X component of the composed rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0]})
            >>> args = ["nil", "nil", "nil", "one", "nil", "nil", "nil", "one"]
            >>> ds.select(x=bt.quat_multiply_x(*args)).to_pydict()
            {'x': [0.0]}
    """
    return spatial_call("quat_multiply_x", ax, ay, az, aw, bx, by, bz, bw)


def quat_multiply_y(
    ax: Numeric,
    ay: Numeric,
    az: Numeric,
    aw: Numeric,
    bx: Numeric,
    by: Numeric,
    bz: Numeric,
    bw: Numeric,
) -> Expr:
    """Take the Y component of two rotations applied in turn.

    Args:
        ax: The second rotation's X component.
        ay: The second rotation's Y component.
        az: The second rotation's Z component.
        aw: The second rotation's scalar component.
        bx: The first rotation's X component.
        by: The first rotation's Y component.
        bz: The first rotation's Z component.
        bw: The first rotation's scalar component.

    Returns:
        The Y component of the composed rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0]})
            >>> args = ["nil", "nil", "nil", "one", "nil", "nil", "nil", "one"]
            >>> ds.select(v=bt.quat_multiply_y(*args)).to_pydict()
            {'v': [0.0]}
    """
    return spatial_call("quat_multiply_y", ax, ay, az, aw, bx, by, bz, bw)


def quat_multiply_z(
    ax: Numeric,
    ay: Numeric,
    az: Numeric,
    aw: Numeric,
    bx: Numeric,
    by: Numeric,
    bz: Numeric,
    bw: Numeric,
) -> Expr:
    """Take the Z component of two rotations applied in turn.

    Args:
        ax: The second rotation's X component.
        ay: The second rotation's Y component.
        az: The second rotation's Z component.
        aw: The second rotation's scalar component.
        bx: The first rotation's X component.
        by: The first rotation's Y component.
        bz: The first rotation's Z component.
        bw: The first rotation's scalar component.

    Returns:
        The Z component of the composed rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0]})
            >>> args = ["nil", "nil", "nil", "one", "nil", "nil", "nil", "one"]
            >>> ds.select(v=bt.quat_multiply_z(*args)).to_pydict()
            {'v': [0.0]}
    """
    return spatial_call("quat_multiply_z", ax, ay, az, aw, bx, by, bz, bw)


def quat_multiply_w(
    ax: Numeric,
    ay: Numeric,
    az: Numeric,
    aw: Numeric,
    bx: Numeric,
    by: Numeric,
    bz: Numeric,
    bw: Numeric,
) -> Expr:
    """Take the scalar component of two rotations applied in turn.

    Args:
        ax: The second rotation's X component.
        ay: The second rotation's Y component.
        az: The second rotation's Z component.
        aw: The second rotation's scalar component.
        bx: The first rotation's X component.
        by: The first rotation's Y component.
        bz: The first rotation's Z component.
        bw: The first rotation's scalar component.

    Returns:
        The scalar component of the composed rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0]})
            >>> args = ["nil", "nil", "nil", "one", "nil", "nil", "nil", "one"]
            >>> ds.select(w=bt.quat_multiply_w(*args)).to_pydict()
            {'w': [1.0]}
    """
    return spatial_call("quat_multiply_w", ax, ay, az, aw, bx, by, bz, bw)


def quat_angular_distance(
    ax: Numeric,
    ay: Numeric,
    az: Numeric,
    aw: Numeric,
    bx: Numeric,
    by: Numeric,
    bz: Numeric,
    bw: Numeric,
) -> Expr:
    """Measure the angle between two rotations.

    The honest error metric for an orientation estimate, and the one to aggregate when
    scoring a localizer or a pose model against ground truth. A component-wise
    difference is not, because a quaternion and its negation are the same rotation and
    would score as maximally far apart.

    Args:
        ax: The first rotation's X component.
        ay: The first rotation's Y component.
        az: The first rotation's Z component.
        aw: The first rotation's scalar component.
        bx: The second rotation's X component.
        by: The second rotation's Y component.
        bz: The second rotation's Z component.
        bw: The second rotation's scalar component.

    Returns:
        The angle in radians on ``[0, pi]``, or null if either rotation is all zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0], "neg": [-1.0]})
            >>> args = ["nil", "nil", "nil", "one", "nil", "nil", "nil", "neg"]
            >>> ds.select(d=bt.quat_angular_distance(*args)).to_pydict()
            {'d': [0.0]}
    """
    return spatial_call("quat_angular_distance", ax, ay, az, aw, bx, by, bz, bw)


def quat_slerp_x(
    ax: Numeric,
    ay: Numeric,
    az: Numeric,
    aw: Numeric,
    bx: Numeric,
    by: Numeric,
    bz: Numeric,
    bw: Numeric,
    t: Numeric,
) -> Expr:
    """Take the X component of a rotation interpolated between two others.

    Spherical linear interpolation: the first rotation at ``t = 0``, the second at
    ``t = 1``, sweeping the angle between them at a constant rate. ``t`` is not clamped,
    so a value outside ``[0, 1]`` extrapolates along the same arc, which is what you want
    when a measurement's timestamp falls just past the last logged pose.

    Args:
        ax: The first rotation's X component.
        ay: The first rotation's Y component.
        az: The first rotation's Z component.
        aw: The first rotation's scalar component.
        bx: The second rotation's X component.
        by: The second rotation's Y component.
        bz: The second rotation's Z component.
        bw: The second rotation's scalar component.
        t: The interpolation fraction.

    Returns:
        The X component of the interpolated rotation, or null if either input rotation
        is all zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0], "half": [0.5]})
            >>> args = ["nil", "nil", "nil", "one", "nil", "nil", "nil", "one", "half"]
            >>> ds.select(x=bt.quat_slerp_x(*args)).to_pydict()
            {'x': [0.0]}
    """
    return spatial_call("quat_slerp_x", ax, ay, az, aw, bx, by, bz, bw, t)


def quat_slerp_y(
    ax: Numeric,
    ay: Numeric,
    az: Numeric,
    aw: Numeric,
    bx: Numeric,
    by: Numeric,
    bz: Numeric,
    bw: Numeric,
    t: Numeric,
) -> Expr:
    """Take the Y component of a rotation interpolated between two others.

    Args:
        ax: The first rotation's X component.
        ay: The first rotation's Y component.
        az: The first rotation's Z component.
        aw: The first rotation's scalar component.
        bx: The second rotation's X component.
        by: The second rotation's Y component.
        bz: The second rotation's Z component.
        bw: The second rotation's scalar component.
        t: The interpolation fraction.

    Returns:
        The Y component of the interpolated rotation, or null if either input rotation
        is all zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0], "half": [0.5]})
            >>> args = ["nil", "nil", "nil", "one", "nil", "nil", "nil", "one", "half"]
            >>> ds.select(v=bt.quat_slerp_y(*args)).to_pydict()
            {'v': [0.0]}
    """
    return spatial_call("quat_slerp_y", ax, ay, az, aw, bx, by, bz, bw, t)


def quat_slerp_z(
    ax: Numeric,
    ay: Numeric,
    az: Numeric,
    aw: Numeric,
    bx: Numeric,
    by: Numeric,
    bz: Numeric,
    bw: Numeric,
    t: Numeric,
) -> Expr:
    """Take the Z component of a rotation interpolated between two others.

    Args:
        ax: The first rotation's X component.
        ay: The first rotation's Y component.
        az: The first rotation's Z component.
        aw: The first rotation's scalar component.
        bx: The second rotation's X component.
        by: The second rotation's Y component.
        bz: The second rotation's Z component.
        bw: The second rotation's scalar component.
        t: The interpolation fraction.

    Returns:
        The Z component of the interpolated rotation, or null if either input rotation
        is all zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0], "half": [0.5]})
            >>> args = ["nil", "nil", "nil", "one", "nil", "nil", "nil", "one", "half"]
            >>> ds.select(v=bt.quat_slerp_z(*args)).to_pydict()
            {'v': [0.0]}
    """
    return spatial_call("quat_slerp_z", ax, ay, az, aw, bx, by, bz, bw, t)


def quat_slerp_w(
    ax: Numeric,
    ay: Numeric,
    az: Numeric,
    aw: Numeric,
    bx: Numeric,
    by: Numeric,
    bz: Numeric,
    bw: Numeric,
    t: Numeric,
) -> Expr:
    """Take the scalar component of a rotation interpolated between two others.

    Args:
        ax: The first rotation's X component.
        ay: The first rotation's Y component.
        az: The first rotation's Z component.
        aw: The first rotation's scalar component.
        bx: The second rotation's X component.
        by: The second rotation's Y component.
        bz: The second rotation's Z component.
        bw: The second rotation's scalar component.
        t: The interpolation fraction.

    Returns:
        The scalar component of the interpolated rotation, or null if either input
        rotation is all zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0], "half": [0.5]})
            >>> args = ["nil", "nil", "nil", "one", "nil", "nil", "nil", "one", "half"]
            >>> ds.select(w=bt.quat_slerp_w(*args)).to_pydict()
            {'w': [1.0]}
    """
    return spatial_call("quat_slerp_w", ax, ay, az, aw, bx, by, bz, bw, t)


def quat_rotate_x(
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Rotate a vector and take its X component.

    Rotation only, with no translation — the right function for a velocity, an
    acceleration or a surface normal, all of which turn with a frame but do not move
    with it. A *position* almost always wants `se3_transform_x` instead.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.
        px: The vector's X component.
        py: The vector's Y component.
        pz: The vector's Z component.

    Returns:
        The X component of the rotated vector, or null if the rotation is all zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import math
            >>> ds = bt.from_pydict(
            ...     {
            ...         "qx": [0.0],
            ...         "qy": [0.0],
            ...         "qz": [math.sin(math.pi / 4)],
            ...         "qw": [math.cos(math.pi / 4)],
            ...         "x": [1.0],
            ...         "y": [0.0],
            ...         "z": [0.0],
            ...     }
            ... )
            >>> got = bt.quat_rotate_y("qx", "qy", "qz", "qw", "x", "y", "z")
            >>> ds.select(v=got.round(9)).to_pydict()
            {'v': [1.0]}
    """
    return spatial_call("quat_rotate_x", qx, qy, qz, qw, px, py, pz)


def quat_rotate_y(
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Rotate a vector and take its Y component.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.
        px: The vector's X component.
        py: The vector's Y component.
        pz: The vector's Z component.

    Returns:
        The Y component of the rotated vector, or null if the rotation is all zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> got = bt.quat_rotate_y("qx", "qy", "qz", "qw", "x", "y", "z")
            >>> ds.select(v=got).to_pydict()
            {'v': [2.0]}
    """
    return spatial_call("quat_rotate_y", qx, qy, qz, qw, px, py, pz)


def quat_rotate_z(
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Rotate a vector and take its Z component.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.
        px: The vector's X component.
        py: The vector's Y component.
        pz: The vector's Z component.

    Returns:
        The Z component of the rotated vector, or null if the rotation is all zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> got = bt.quat_rotate_z("qx", "qy", "qz", "qw", "x", "y", "z")
            >>> ds.select(v=got).to_pydict()
            {'v': [3.0]}
    """
    return spatial_call("quat_rotate_z", qx, qy, qz, qw, px, py, pz)


def quat_inverse_rotate_x(
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Rotate a vector by the opposite rotation and take its X component.

    The direction a world-frame vector travels to reach a body frame — a velocity
    expressed in world coordinates, read as forward, left and up relative to the vehicle.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.
        px: The vector's X component.
        py: The vector's Y component.
        pz: The vector's Z component.

    Returns:
        The X component of the inversely-rotated vector, or null if the rotation is all
        zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> got = bt.quat_inverse_rotate_x("qx", "qy", "qz", "qw", "x", "y", "z")
            >>> ds.select(v=got).to_pydict()
            {'v': [1.0]}
    """
    return spatial_call("quat_inverse_rotate_x", qx, qy, qz, qw, px, py, pz)


def quat_inverse_rotate_y(
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Rotate a vector by the opposite rotation and take its Y component.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.
        px: The vector's X component.
        py: The vector's Y component.
        pz: The vector's Z component.

    Returns:
        The Y component of the inversely-rotated vector, or null if the rotation is all
        zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> got = bt.quat_inverse_rotate_y("qx", "qy", "qz", "qw", "x", "y", "z")
            >>> ds.select(v=got).to_pydict()
            {'v': [2.0]}
    """
    return spatial_call("quat_inverse_rotate_y", qx, qy, qz, qw, px, py, pz)


def quat_inverse_rotate_z(
    qx: Numeric,
    qy: Numeric,
    qz: Numeric,
    qw: Numeric,
    px: Numeric,
    py: Numeric,
    pz: Numeric,
) -> Expr:
    """Rotate a vector by the opposite rotation and take its Z component.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.
        px: The vector's X component.
        py: The vector's Y component.
        pz: The vector's Z component.

    Returns:
        The Z component of the inversely-rotated vector, or null if the rotation is all
        zeros.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> got = bt.quat_inverse_rotate_z("qx", "qy", "qz", "qw", "x", "y", "z")
            >>> ds.select(v=got).to_pydict()
            {'v': [3.0]}
    """
    return spatial_call("quat_inverse_rotate_z", qx, qy, qz, qw, px, py, pz)
