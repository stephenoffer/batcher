"""Reading a rotation, and building one from the other two spellings a log uses.

A rotation column in a robotics log is four numbers — ``qx``, ``qy``, ``qz``, ``qw`` —
and the functions here answer questions about it or produce it from something else.
Composing, interpolating and applying rotations live in `rotate`; whole poses live in
`pose`.

Every function takes the four components separately and in ``(x, y, z, w)`` order,
scalar last. That is the ROS, SciPy and Eigen-storage order, and taking them one by one
means the order is written at the call site instead of assumed: a quaternion read out of
a scalar-first source such as nuScenes and fed to a scalar-last function is not an error
anything can detect, it is a different and entirely plausible rotation.

A quaternion of length zero names no rotation, and every function here returns null for
one rather than raising. `quat_norm` is the exception and the escape hatch — it reports
the zero, so ``filter(quat_norm(...) == 0)`` finds the rows the others nulled.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.spatial._build import Numeric, spatial_call

__all__ = [
    "quat_angle",
    "quat_from_euler_w",
    "quat_from_euler_x",
    "quat_from_euler_y",
    "quat_from_euler_z",
    "quat_from_rotmat_w",
    "quat_from_rotmat_x",
    "quat_from_rotmat_y",
    "quat_from_rotmat_z",
    "quat_inverse_w",
    "quat_inverse_x",
    "quat_inverse_y",
    "quat_inverse_z",
    "quat_norm",
    "quat_normalize_w",
    "quat_normalize_x",
    "quat_normalize_y",
    "quat_normalize_z",
    "quat_to_pitch",
    "quat_to_roll",
    "quat_to_yaw",
]


def quat_norm(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Measure a quaternion's four-component length.

    A rotation has length one. A quaternion that has been logged, rounded to
    ``float32``, or multiplied a few thousand times is only nearly unit, and this is how
    you see how far it has drifted. It is also the one function in the family that
    answers for a zero quaternion rather than nulling it, which makes it the way to find
    the rows every other function nulled.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The Euclidean length of the four components.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]}
            ... )
            >>> ds.select(n=bt.quat_norm("qx", "qy", "qz", "qw")).to_pydict()
            {'n': [1.0]}
    """
    return spatial_call("quat_norm", qx, qy, qz, qw)


def quat_normalize_x(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Take the X component of the same rotation at unit length.

    Normalizing a rotation column is worth doing once on ingest when the source is
    lossy, so that a downstream comparison of raw components means what it looks like it
    means. Every function in this family normalizes internally anyway, so this is for
    storage and for equality, not for correctness of the maths.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The X component of the unit-length rotation, or null if every component is zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [3.0], "qy": [0.0], "qz": [0.0], "qw": [4.0]})
            >>> ds.select(x=bt.quat_normalize_x("qx", "qy", "qz", "qw")).to_pydict()
            {'x': [0.6]}
    """
    return spatial_call("quat_normalize_x", qx, qy, qz, qw)


def quat_normalize_y(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Take the Y component of the same rotation at unit length.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The Y component of the unit-length rotation, or null if every component is zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [0.0], "qy": [3.0], "qz": [0.0], "qw": [4.0]})
            >>> ds.select(y=bt.quat_normalize_y("qx", "qy", "qz", "qw")).to_pydict()
            {'y': [0.6]}
    """
    return spatial_call("quat_normalize_y", qx, qy, qz, qw)


def quat_normalize_z(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Take the Z component of the same rotation at unit length.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The Z component of the unit-length rotation, or null if every component is zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [0.0], "qy": [0.0], "qz": [3.0], "qw": [4.0]})
            >>> ds.select(z=bt.quat_normalize_z("qx", "qy", "qz", "qw")).to_pydict()
            {'z': [0.6]}
    """
    return spatial_call("quat_normalize_z", qx, qy, qz, qw)


def quat_normalize_w(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Take the scalar component of the same rotation at unit length.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The scalar component of the unit-length rotation, or null if every component is
        zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [3.0], "qy": [0.0], "qz": [0.0], "qw": [4.0]})
            >>> ds.select(w=bt.quat_normalize_w("qx", "qy", "qz", "qw")).to_pydict()
            {'w': [0.8]}
    """
    return spatial_call("quat_normalize_w", qx, qy, qz, qw)


def quat_inverse_x(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Take the X component of the opposite rotation.

    The inverse of a *rotation*, not the multiplicative inverse of the four numbers:
    the input is normalized first, so a drifted quaternion cannot smuggle a scale factor
    into the result and quietly shrink whatever it is applied to.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The X component of the inverse rotation, or null if every component is zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [0.6], "qy": [0.0], "qz": [0.0], "qw": [0.8]})
            >>> ds.select(x=bt.quat_inverse_x("qx", "qy", "qz", "qw")).to_pydict()
            {'x': [-0.6]}
    """
    return spatial_call("quat_inverse_x", qx, qy, qz, qw)


def quat_inverse_y(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Take the Y component of the opposite rotation.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The Y component of the inverse rotation, or null if every component is zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [0.0], "qy": [0.6], "qz": [0.0], "qw": [0.8]})
            >>> ds.select(y=bt.quat_inverse_y("qx", "qy", "qz", "qw")).to_pydict()
            {'y': [-0.6]}
    """
    return spatial_call("quat_inverse_y", qx, qy, qz, qw)


def quat_inverse_z(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Take the Z component of the opposite rotation.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The Z component of the inverse rotation, or null if every component is zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [0.0], "qy": [0.0], "qz": [0.6], "qw": [0.8]})
            >>> ds.select(z=bt.quat_inverse_z("qx", "qy", "qz", "qw")).to_pydict()
            {'z': [-0.6]}
    """
    return spatial_call("quat_inverse_z", qx, qy, qz, qw)


def quat_inverse_w(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Take the scalar component of the opposite rotation.

    The scalar part is unchanged by inversion once the quaternion is unit length, so
    this differs from `quat_normalize_w` only for a drifted input.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The scalar component of the inverse rotation, or null if every component is
        zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [0.6], "qy": [0.0], "qz": [0.0], "qw": [0.8]})
            >>> ds.select(w=bt.quat_inverse_w("qx", "qy", "qz", "qw")).to_pydict()
            {'w': [0.8]}
    """
    return spatial_call("quat_inverse_w", qx, qy, qz, qw)


def quat_angle(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Measure how far a rotation turns, ignoring which way.

    The magnitude of the rotation in radians, always on ``[0, pi]``. Useful as a filter
    on its own — a calibration whose rotation is meant to be small, an IMU sample whose
    orientation change between ticks should be bounded.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The rotation angle in radians, or null if every component is zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]}
            ... )
            >>> ds.select(a=bt.quat_angle("qx", "qy", "qz", "qw")).to_pydict()
            {'a': [0.0]}
    """
    return spatial_call("quat_angle", qx, qy, qz, qw)


def quat_to_roll(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Read a rotation's roll, the angle about the X axis.

    Part of the intrinsic Z-Y-X decomposition: yaw about Z, then pitch about Y, then
    roll about X. At pitch of plus or minus a quarter turn roll and yaw stop being
    separately determined, and this reports roll as zero there rather than picking one
    of the infinitely many equivalent splits.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The roll angle in radians, or null if every component is zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]}
            ... )
            >>> ds.select(r=bt.quat_to_roll("qx", "qy", "qz", "qw")).to_pydict()
            {'r': [0.0]}
    """
    return spatial_call("quat_to_roll", qx, qy, qz, qw)


def quat_to_pitch(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Read a rotation's pitch, the angle about the Y axis.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The pitch angle in radians, on ``[-pi/2, pi/2]``, or null if every component is
        zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]}
            ... )
            >>> ds.select(p=bt.quat_to_pitch("qx", "qy", "qz", "qw")).to_pydict()
            {'p': [0.0]}
    """
    return spatial_call("quat_to_pitch", qx, qy, qz, qw)


def quat_to_yaw(qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric) -> Expr:
    """Read a rotation's yaw, the angle about the Z axis.

    The heading, and the one of the three angles a map-matching, planning or
    lane-association query actually asks for. Ground vehicles are near-level almost
    always, which makes yaw the whole of the orientation for most purposes.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.

    Returns:
        The yaw angle in radians, on ``[-pi, pi]``, or null if every component is zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import math
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [math.sin(0.25)], "qw": [math.cos(0.25)]}
            ... )
            >>> ds.select(y=bt.quat_to_yaw("qx", "qy", "qz", "qw").round(6)).to_pydict()
            {'y': [0.5]}
    """
    return spatial_call("quat_to_yaw", qx, qy, qz, qw)


def quat_from_euler_x(roll: Numeric, pitch: Numeric, yaw: Numeric) -> Expr:
    """Build the X component of the rotation three Euler angles describe.

    The inverse of `quat_to_roll` and its siblings, in the same intrinsic Z-Y-X
    sequence. Use it when the source publishes angles — an IMU, a map annotation, a
    human-authored calibration — and the rest of the pipeline wants a quaternion.

    Args:
        roll: Rotation about the X axis, in radians.
        pitch: Rotation about the Y axis, in radians.
        yaw: Rotation about the Z axis, in radians.

    Returns:
        The X component of the resulting rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"r": [0.0], "p": [0.0], "y": [0.0]})
            >>> ds.select(x=bt.quat_from_euler_x("r", "p", "y")).to_pydict()
            {'x': [0.0]}
    """
    return spatial_call("quat_from_euler_x", roll, pitch, yaw)


def quat_from_euler_y(roll: Numeric, pitch: Numeric, yaw: Numeric) -> Expr:
    """Build the Y component of the rotation three Euler angles describe.

    Args:
        roll: Rotation about the X axis, in radians.
        pitch: Rotation about the Y axis, in radians.
        yaw: Rotation about the Z axis, in radians.

    Returns:
        The Y component of the resulting rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"r": [0.0], "p": [0.0], "y": [0.0]})
            >>> ds.select(v=bt.quat_from_euler_y("r", "p", "y")).to_pydict()
            {'v': [0.0]}
    """
    return spatial_call("quat_from_euler_y", roll, pitch, yaw)


def quat_from_euler_z(roll: Numeric, pitch: Numeric, yaw: Numeric) -> Expr:
    """Build the Z component of the rotation three Euler angles describe.

    Args:
        roll: Rotation about the X axis, in radians.
        pitch: Rotation about the Y axis, in radians.
        yaw: Rotation about the Z axis, in radians.

    Returns:
        The Z component of the resulting rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"r": [0.0], "p": [0.0], "y": [0.0]})
            >>> ds.select(v=bt.quat_from_euler_z("r", "p", "y")).to_pydict()
            {'v': [0.0]}
    """
    return spatial_call("quat_from_euler_z", roll, pitch, yaw)


def quat_from_euler_w(roll: Numeric, pitch: Numeric, yaw: Numeric) -> Expr:
    """Build the scalar component of the rotation three Euler angles describe.

    Args:
        roll: Rotation about the X axis, in radians.
        pitch: Rotation about the Y axis, in radians.
        yaw: Rotation about the Z axis, in radians.

    Returns:
        The scalar component of the resulting rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"r": [0.0], "p": [0.0], "y": [0.0]})
            >>> ds.select(w=bt.quat_from_euler_w("r", "p", "y")).to_pydict()
            {'w': [1.0]}
    """
    return spatial_call("quat_from_euler_w", roll, pitch, yaw)


def quat_from_rotmat_x(
    m00: Numeric,
    m01: Numeric,
    m02: Numeric,
    m10: Numeric,
    m11: Numeric,
    m12: Numeric,
    m20: Numeric,
    m21: Numeric,
    m22: Numeric,
) -> Expr:
    """Build the X component of the rotation a 3x3 matrix describes.

    Sensor calibration files usually carry a rotation matrix rather than a quaternion,
    because nine numbers need no convention note to be unambiguous. Arguments are
    row-major: ``m01`` is row 0, column 1.

    A matrix that is not a rotation is not detected. Check one with
    ``quat_norm(quat_from_rotmat_x(...), ...)``, which is one only for a genuine
    rotation.

    Args:
        m00: Row 0, column 0.
        m01: Row 0, column 1.
        m02: Row 0, column 2.
        m10: Row 1, column 0.
        m11: Row 1, column 1.
        m12: Row 1, column 2.
        m20: Row 2, column 0.
        m21: Row 2, column 1.
        m22: Row 2, column 2.

    Returns:
        The X component of the resulting rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0]})
            >>> cols = ["one", "nil", "nil", "nil", "one", "nil", "nil", "nil", "one"]
            >>> ds.select(x=bt.quat_from_rotmat_x(*cols)).to_pydict()
            {'x': [0.0]}
    """
    return spatial_call("quat_from_rotmat_x", m00, m01, m02, m10, m11, m12, m20, m21, m22)


def quat_from_rotmat_y(
    m00: Numeric,
    m01: Numeric,
    m02: Numeric,
    m10: Numeric,
    m11: Numeric,
    m12: Numeric,
    m20: Numeric,
    m21: Numeric,
    m22: Numeric,
) -> Expr:
    """Build the Y component of the rotation a 3x3 matrix describes.

    Args:
        m00: Row 0, column 0.
        m01: Row 0, column 1.
        m02: Row 0, column 2.
        m10: Row 1, column 0.
        m11: Row 1, column 1.
        m12: Row 1, column 2.
        m20: Row 2, column 0.
        m21: Row 2, column 1.
        m22: Row 2, column 2.

    Returns:
        The Y component of the resulting rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0]})
            >>> cols = ["one", "nil", "nil", "nil", "one", "nil", "nil", "nil", "one"]
            >>> ds.select(v=bt.quat_from_rotmat_y(*cols)).to_pydict()
            {'v': [0.0]}
    """
    return spatial_call("quat_from_rotmat_y", m00, m01, m02, m10, m11, m12, m20, m21, m22)


def quat_from_rotmat_z(
    m00: Numeric,
    m01: Numeric,
    m02: Numeric,
    m10: Numeric,
    m11: Numeric,
    m12: Numeric,
    m20: Numeric,
    m21: Numeric,
    m22: Numeric,
) -> Expr:
    """Build the Z component of the rotation a 3x3 matrix describes.

    Args:
        m00: Row 0, column 0.
        m01: Row 0, column 1.
        m02: Row 0, column 2.
        m10: Row 1, column 0.
        m11: Row 1, column 1.
        m12: Row 1, column 2.
        m20: Row 2, column 0.
        m21: Row 2, column 1.
        m22: Row 2, column 2.

    Returns:
        The Z component of the resulting rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0]})
            >>> cols = ["one", "nil", "nil", "nil", "one", "nil", "nil", "nil", "one"]
            >>> ds.select(v=bt.quat_from_rotmat_z(*cols)).to_pydict()
            {'v': [0.0]}
    """
    return spatial_call("quat_from_rotmat_z", m00, m01, m02, m10, m11, m12, m20, m21, m22)


def quat_from_rotmat_w(
    m00: Numeric,
    m01: Numeric,
    m02: Numeric,
    m10: Numeric,
    m11: Numeric,
    m12: Numeric,
    m20: Numeric,
    m21: Numeric,
    m22: Numeric,
) -> Expr:
    """Build the scalar component of the rotation a 3x3 matrix describes.

    Args:
        m00: Row 0, column 0.
        m01: Row 0, column 1.
        m02: Row 0, column 2.
        m10: Row 1, column 0.
        m11: Row 1, column 1.
        m12: Row 1, column 2.
        m20: Row 2, column 0.
        m21: Row 2, column 1.
        m22: Row 2, column 2.

    Returns:
        The scalar component of the resulting rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "nil": [0.0]})
            >>> cols = ["one", "nil", "nil", "nil", "one", "nil", "nil", "nil", "one"]
            >>> ds.select(w=bt.quat_from_rotmat_w(*cols)).to_pydict()
            {'w': [1.0]}
    """
    return spatial_call("quat_from_rotmat_w", m00, m01, m02, m10, m11, m12, m20, m21, m22)
