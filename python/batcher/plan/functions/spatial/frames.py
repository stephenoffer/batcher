"""Whole rotations and whole poses, as a set of named columns at once.

The component functions in `quaternion`, `rotate` and `pose` are the primitives, and
they are unwieldy on their own: transforming one point through one pose means writing a
ten-argument call three times, once per output coordinate. Everything here is a thin
composition of them that returns a **mapping of column name to expression**, so a whole
transform is one call splatted into `Dataset.with_columns`:

.. code-block:: python

    lidar_to_world = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
    world = sweep.with_columns(
        **bt.se3_transform(lidar_to_world, ("x", "y", "z"), prefix="world_")
    )

Nothing here introduces a new IR node. `se3_compose` and `se3_inverse` in particular are
built out of the quaternion primitives rather than given kernels of their own: they run
once per *frame*, of which a log has thousands, while `se3_transform` runs once per
*point*, of which it has billions, so only the second could repay a kernel at all.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.spatial._build import Numeric, Point, Pose, Quaternion, value
from batcher.plan.functions.spatial.pose import (
    se3_inverse_transform_x,
    se3_inverse_transform_y,
    se3_inverse_transform_z,
    se3_transform_x,
    se3_transform_y,
    se3_transform_z,
)
from batcher.plan.functions.spatial.quaternion import (
    quat_from_euler_w,
    quat_from_euler_x,
    quat_from_euler_y,
    quat_from_euler_z,
    quat_inverse_w,
    quat_inverse_x,
    quat_inverse_y,
    quat_inverse_z,
    quat_normalize_w,
    quat_normalize_x,
    quat_normalize_y,
    quat_normalize_z,
    quat_to_pitch,
    quat_to_roll,
    quat_to_yaw,
)
from batcher.plan.functions.spatial.rotate import (
    quat_multiply_w,
    quat_multiply_x,
    quat_multiply_y,
    quat_multiply_z,
    quat_rotate_x,
    quat_rotate_y,
    quat_rotate_z,
    quat_slerp_w,
    quat_slerp_x,
    quat_slerp_y,
    quat_slerp_z,
)

__all__ = [
    "pose_interpolate",
    "quat_from_euler",
    "quat_inverse",
    "quat_multiply",
    "quat_normalize",
    "quat_slerp",
    "quat_to_euler",
    "se3_compose",
    "se3_inverse",
    "se3_inverse_transform",
    "se3_transform",
]


def _split_pose(p: Pose) -> tuple[Point, Quaternion]:
    """Split a seven-value pose into its translation and its rotation."""
    return (p[0], p[1], p[2]), (p[3], p[4], p[5], p[6])


def se3_transform(pose: Pose, point: Point, *, prefix: str = "") -> dict[str, Expr]:
    """Move a point out of a frame and into its parent, as three named columns.

    The single most-used transform in an autonomous-driving pipeline: it takes a sensor
    reading in the sensor's own coordinates and puts it in the world's.

    Args:
        pose: The frame's pose in its parent, as ``(tx, ty, tz, qx, qy, qz, qw)``.
        point: The point in the frame's own coordinates, as ``(x, y, z)``.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``x``/``y``/``z`` (each with `prefix`) to the transformed
        coordinate, ready to splat into ``with_columns``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"tx": [10.0], "ty": [20.0], "tz": [0.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [1.0], "y": [2.0], "z": [3.0]}
            ... )
            >>> pose = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
            >>> moved = bt.se3_transform(pose, ("x", "y", "z"), prefix="world_")
            >>> ds.select(**moved).to_pydict()
            {'world_x': [11.0], 'world_y': [22.0], 'world_z': [3.0]}
    """
    args = (*pose, *point)
    return {
        f"{prefix}x": se3_transform_x(*args),
        f"{prefix}y": se3_transform_y(*args),
        f"{prefix}z": se3_transform_z(*args),
    }


def se3_inverse_transform(pose: Pose, point: Point, *, prefix: str = "") -> dict[str, Expr]:
    """Move a point out of a parent frame and into a child, as three named columns.

    The inverse of `se3_transform`, and the direction that answers "where is this
    world-frame obstacle relative to the vehicle".

    Args:
        pose: The frame's pose in its parent, as ``(tx, ty, tz, qx, qy, qz, qw)``.
        point: The point in the parent frame's coordinates, as ``(x, y, z)``.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``x``/``y``/``z`` (each with `prefix`) to the transformed
        coordinate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"tx": [10.0], "ty": [20.0], "tz": [0.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "x": [11.0], "y": [22.0], "z": [3.0]}
            ... )
            >>> pose = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
            >>> local = bt.se3_inverse_transform(pose, ("x", "y", "z"), prefix="ego_")
            >>> ds.select(**local).to_pydict()
            {'ego_x': [1.0], 'ego_y': [2.0], 'ego_z': [3.0]}
    """
    args = (*pose, *point)
    return {
        f"{prefix}x": se3_inverse_transform_x(*args),
        f"{prefix}y": se3_inverse_transform_y(*args),
        f"{prefix}z": se3_inverse_transform_z(*args),
    }


def se3_compose(a: Pose, b: Pose, *, prefix: str = "") -> dict[str, Expr]:
    """Chain two poses into one, as seven named columns.

    Composes the way the frame names read: ``se3_compose(world_from_ego,
    ego_from_lidar)`` is ``world_from_lidar``. Collapsing a chain once per frame and
    then applying the single result per point is much cheaper than applying each pose in
    turn to every point, and it is the reason this function exists.

    Args:
        a: The outer pose, as ``(tx, ty, tz, qx, qy, qz, qw)``.
        b: The inner pose, applied first, in the same layout.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``tx``/``ty``/``tz``/``qx``/``qy``/``qz``/``qw`` (each with
        `prefix`) to the composed pose.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"ax": [1.0], "ay": [0.0], "az": [0.0],
            ...      "bx": [0.0], "by": [2.0], "bz": [0.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]}
            ... )
            >>> outer = ("ax", "ay", "az", "qx", "qy", "qz", "qw")
            >>> inner = ("bx", "by", "bz", "qx", "qy", "qz", "qw")
            >>> both = bt.se3_compose(outer, inner)
            >>> ds.select(tx=both["tx"], ty=both["ty"]).to_pydict()
            {'tx': [1.0], 'ty': [2.0]}
    """
    a_t, a_q = _split_pose(a)
    b_t, b_q = _split_pose(b)
    rot = (*a_q, *b_t)
    quats = (*a_q, *b_q)
    return {
        f"{prefix}tx": value(a_t[0]) + quat_rotate_x(*rot),
        f"{prefix}ty": value(a_t[1]) + quat_rotate_y(*rot),
        f"{prefix}tz": value(a_t[2]) + quat_rotate_z(*rot),
        f"{prefix}qx": quat_multiply_x(*quats),
        f"{prefix}qy": quat_multiply_y(*quats),
        f"{prefix}qz": quat_multiply_z(*quats),
        f"{prefix}qw": quat_multiply_w(*quats),
    }


def se3_inverse(pose: Pose, *, prefix: str = "") -> dict[str, Expr]:
    """Turn a pose around, as seven named columns.

    ``se3_inverse(world_from_ego)`` is ``ego_from_world``. Useful when a log carries one
    direction and a query wants the other, and when the same inverse is applied to many
    points — inverting once per frame beats calling `se3_inverse_transform` per point
    only if the pose is then composed further, so prefer the direct function when you
    are simply moving points.

    Args:
        pose: The pose to invert, as ``(tx, ty, tz, qx, qy, qz, qw)``.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``tx``/``ty``/``tz``/``qx``/``qy``/``qz``/``qw`` (each with
        `prefix`) to the inverted pose.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"tx": [5.0], "ty": [0.0], "tz": [0.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]}
            ... )
            >>> inv = bt.se3_inverse(("tx", "ty", "tz", "qx", "qy", "qz", "qw"))
            >>> ds.select(tx=inv["tx"], qw=inv["qw"]).to_pydict()
            {'tx': [-5.0], 'qw': [1.0]}
    """
    t, q = _split_pose(pose)
    inv_q = (
        quat_inverse_x(*q),
        quat_inverse_y(*q),
        quat_inverse_z(*q),
        quat_inverse_w(*q),
    )
    rot = (*inv_q, *t)
    return {
        f"{prefix}tx": -quat_rotate_x(*rot),
        f"{prefix}ty": -quat_rotate_y(*rot),
        f"{prefix}tz": -quat_rotate_z(*rot),
        f"{prefix}qx": inv_q[0],
        f"{prefix}qy": inv_q[1],
        f"{prefix}qz": inv_q[2],
        f"{prefix}qw": inv_q[3],
    }


def pose_interpolate(a: Pose, b: Pose, t: Numeric, *, prefix: str = "") -> dict[str, Expr]:
    """Find the pose between two logged poses, as seven named columns.

    The operation that makes sensor fusion work. Poses arrive at the localizer's rate
    and measurements at each sensor's, so a measurement almost never lands on a logged
    pose. Rotation is interpolated spherically and translation linearly, which is the
    standard treatment and the one that keeps a lidar sweep straight.

    `t` is a fraction, not a timestamp: compute it as
    ``(measured_at - a_at) / (b_at - a_at)``. It is not clamped, so a measurement just
    past the last pose extrapolates rather than pinning to the endpoint.

    Args:
        a: The earlier pose, as ``(tx, ty, tz, qx, qy, qz, qw)``.
        b: The later pose, in the same layout.
        t: The interpolation fraction, 0 at `a` and 1 at `b`.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``tx``/``ty``/``tz``/``qx``/``qy``/``qz``/``qw`` (each with
        `prefix`) to the interpolated pose.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"ax": [0.0], "ay": [0.0], "az": [0.0],
            ...      "bx": [10.0], "by": [0.0], "bz": [0.0],
            ...      "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
            ...      "frac": [0.25]}
            ... )
            >>> rot = ("qx", "qy", "qz", "qw")
            >>> mid = bt.pose_interpolate(
            ...     ("ax", "ay", "az", *rot), ("bx", "by", "bz", *rot), "frac"
            ... )
            >>> ds.select(tx=mid["tx"], qw=mid["qw"]).to_pydict()
            {'tx': [2.5], 'qw': [1.0]}
    """
    a_t, a_q = _split_pose(a)
    b_t, b_q = _split_pose(b)
    frac = value(t)
    slerp = (*a_q, *b_q, frac)
    lerp = {
        f"{prefix}t{axis}": value(lo) + (value(hi) - value(lo)) * frac
        for axis, lo, hi in zip("xyz", a_t, b_t, strict=True)
    }
    return {
        **lerp,
        f"{prefix}qx": quat_slerp_x(*slerp),
        f"{prefix}qy": quat_slerp_y(*slerp),
        f"{prefix}qz": quat_slerp_z(*slerp),
        f"{prefix}qw": quat_slerp_w(*slerp),
    }


def quat_multiply(a: Quaternion, b: Quaternion, *, prefix: str = "") -> dict[str, Expr]:
    """Chain two rotations into one, as four named columns.

    The result applies `b` first and then `a`, matching function composition.

    Args:
        a: The outer rotation, as ``(qx, qy, qz, qw)``.
        b: The inner rotation, applied first, in the same layout.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``qx``/``qy``/``qz``/``qw`` (each with `prefix`) to the composed
        rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]}
            ... )
            >>> q = ("qx", "qy", "qz", "qw")
            >>> ds.select(**bt.quat_multiply(q, q, prefix="c")).to_pydict()
            {'cqx': [0.0], 'cqy': [0.0], 'cqz': [0.0], 'cqw': [1.0]}
    """
    args = (*a, *b)
    return {
        f"{prefix}qx": quat_multiply_x(*args),
        f"{prefix}qy": quat_multiply_y(*args),
        f"{prefix}qz": quat_multiply_z(*args),
        f"{prefix}qw": quat_multiply_w(*args),
    }


def quat_inverse(q: Quaternion, *, prefix: str = "") -> dict[str, Expr]:
    """Turn a rotation around, as four named columns.

    Args:
        q: The rotation, as ``(qx, qy, qz, qw)``.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``qx``/``qy``/``qz``/``qw`` (each with `prefix`) to the inverse
        rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.6], "qy": [0.0], "qz": [0.0], "qw": [0.8]}
            ... )
            >>> inv = bt.quat_inverse(("qx", "qy", "qz", "qw"))
            >>> ds.select(qx=inv["qx"], qw=inv["qw"]).to_pydict()
            {'qx': [-0.6], 'qw': [0.8]}
    """
    return {
        f"{prefix}qx": quat_inverse_x(*q),
        f"{prefix}qy": quat_inverse_y(*q),
        f"{prefix}qz": quat_inverse_z(*q),
        f"{prefix}qw": quat_inverse_w(*q),
    }


def quat_normalize(q: Quaternion, *, prefix: str = "") -> dict[str, Expr]:
    """Bring a rotation to unit length, as four named columns.

    Worth doing once on ingest when the source is lossy. Every function in this family
    normalizes internally, so this is for storage and for comparing raw components, not
    for the correctness of the arithmetic.

    Args:
        q: The rotation, as ``(qx, qy, qz, qw)``.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``qx``/``qy``/``qz``/``qw`` (each with `prefix`) to the unit-length
        rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [3.0], "qy": [0.0], "qz": [0.0], "qw": [4.0]}
            ... )
            >>> unit = bt.quat_normalize(("qx", "qy", "qz", "qw"), prefix="u")
            >>> ds.select(uqx=unit["uqx"], uqw=unit["uqw"]).to_pydict()
            {'uqx': [0.6], 'uqw': [0.8]}
    """
    return {
        f"{prefix}qx": quat_normalize_x(*q),
        f"{prefix}qy": quat_normalize_y(*q),
        f"{prefix}qz": quat_normalize_z(*q),
        f"{prefix}qw": quat_normalize_w(*q),
    }


def quat_slerp(a: Quaternion, b: Quaternion, t: Numeric, *, prefix: str = "") -> dict[str, Expr]:
    """Interpolate between two rotations, as four named columns.

    Args:
        a: The rotation at ``t = 0``, as ``(qx, qy, qz, qw)``.
        b: The rotation at ``t = 1``, in the same layout.
        t: The interpolation fraction, not clamped.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``qx``/``qy``/``qz``/``qw`` (each with `prefix`) to the
        interpolated rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0], "f": [0.5]}
            ... )
            >>> q = ("qx", "qy", "qz", "qw")
            >>> mid = bt.quat_slerp(q, q, "f")
            >>> ds.select(qw=mid["qw"]).to_pydict()
            {'qw': [1.0]}
    """
    args = (*a, *b, t)
    return {
        f"{prefix}qx": quat_slerp_x(*args),
        f"{prefix}qy": quat_slerp_y(*args),
        f"{prefix}qz": quat_slerp_z(*args),
        f"{prefix}qw": quat_slerp_w(*args),
    }


def quat_to_euler(q: Quaternion, *, prefix: str = "") -> dict[str, Expr]:
    """Read a rotation as roll, pitch and yaw, as three named columns.

    The intrinsic Z-Y-X sequence. Euler angles are for reading and for reporting, never
    for computing: they are not unique at pitch of plus or minus a quarter turn, and
    round-tripping a rotation through them near that pole does not return the angles you
    started with.

    Args:
        q: The rotation, as ``(qx, qy, qz, qw)``.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``roll``/``pitch``/``yaw`` (each with `prefix`) to the angle in
        radians.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]}
            ... )
            >>> ds.select(**bt.quat_to_euler(("qx", "qy", "qz", "qw"))).to_pydict()
            {'roll': [0.0], 'pitch': [0.0], 'yaw': [0.0]}
    """
    return {
        f"{prefix}roll": quat_to_roll(*q),
        f"{prefix}pitch": quat_to_pitch(*q),
        f"{prefix}yaw": quat_to_yaw(*q),
    }


def quat_from_euler(
    roll: Numeric, pitch: Numeric, yaw: Numeric, *, prefix: str = ""
) -> dict[str, Expr]:
    """Build a rotation from roll, pitch and yaw, as four named columns.

    Args:
        roll: Rotation about the X axis, in radians.
        pitch: Rotation about the Y axis, in radians.
        yaw: Rotation about the Z axis, in radians.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``qx``/``qy``/``qz``/``qw`` (each with `prefix`) to the resulting
        rotation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"r": [0.0], "p": [0.0], "y": [0.0]})
            >>> ds.select(**bt.quat_from_euler("r", "p", "y")).to_pydict()
            {'qx': [0.0], 'qy': [0.0], 'qz': [0.0], 'qw': [1.0]}
    """
    return {
        f"{prefix}qx": quat_from_euler_x(roll, pitch, yaw),
        f"{prefix}qy": quat_from_euler_y(roll, pitch, yaw),
        f"{prefix}qz": quat_from_euler_z(roll, pitch, yaw),
        f"{prefix}qw": quat_from_euler_w(roll, pitch, yaw),
    }
