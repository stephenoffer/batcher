"""Euler angles in any axis sequence, for logs that do not use Z-Y-X.

The rest of the family reads and writes Euler angles in one convention, intrinsic Z-Y-X
(yaw, then pitch, then roll), which is the ROS and aerospace one. Other fields settled on
others: biomechanics on Y-X-Z, orbital mechanics on the proper Euler angles Z-X-Z, a
camera rig on whatever its vendor chose. The functions here take the sequence as a
string, SciPy's way: uppercase letters (``"ZYX"``) are *intrinsic*, each turn about the
axes the previous turns left behind, and lowercase (``"xyz"``) are *extrinsic*, each turn
about the fixed axes. Intrinsic ``"ZYX"`` and extrinsic ``"xyz"`` are the same rotation
with the angles listed in reverse.

The angles come back in sequence order, and at gimbal lock (a middle angle of a quarter
turn for three distinct axes, or of 0 or a half turn for a repeated one) the third angle
is reported as zero, which is SciPy's convention and, for ``"ZYX"``, the family's own.

The sequence is fixed when the expression is built, and from SQL it is a string literal:
``quat_to_euler_seq_first(qx, qy, qz, qw, 'ZXZ')``. It crosses the engine boundary as a
number (the axes as digits, ``x`` = 1, ``y`` = 2, ``z`` = 3, plus 1000 when intrinsic),
because every argument in this family is a ``Float64``; `bc_spatial::euler_seq` holds the
decoding.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.spatial._build import Numeric, Point, Quaternion, spatial_call

__all__ = [
    "quat_from_euler_seq",
    "quat_from_euler_seq_w",
    "quat_from_euler_seq_x",
    "quat_from_euler_seq_y",
    "quat_from_euler_seq_z",
    "quat_to_euler_seq",
    "quat_to_euler_seq_first",
    "quat_to_euler_seq_second",
    "quat_to_euler_seq_third",
]


def _sequence_code(sequence: str) -> int:
    """The engine's numeric code for an axis sequence, refusing one that names none."""
    ok = (
        isinstance(sequence, str)
        and len(sequence) == 3
        and (sequence.isupper() or sequence.islower())
        and set(sequence.lower()) <= {"x", "y", "z"}
        and sequence[0].lower() != sequence[1].lower()
        and sequence[1].lower() != sequence[2].lower()
    )
    if not ok:
        raise PlanError(
            f"an Euler sequence is three of x/y/z, all uppercase (intrinsic) or all "
            f"lowercase (extrinsic), with no axis repeated back to back; got {sequence!r}"
        )
    digits = int("".join(str("xyz".index(c) + 1) for c in sequence.lower()))
    return digits + (1000 if sequence.isupper() else 0)


def _from(fn: str, a1: Numeric, a2: Numeric, a3: Numeric, sequence: str) -> Expr:
    return spatial_call(fn, a1, a2, a3, _sequence_code(sequence))


def _to(fn: str, q: Quaternion, sequence: str) -> Expr:
    return spatial_call(fn, *q, _sequence_code(sequence))


def quat_from_euler_seq_x(a1: Numeric, a2: Numeric, a3: Numeric, sequence: str = "ZYX") -> Expr:
    """Build the X component of the rotation three angles describe in any axis sequence.

    Args:
        a1: The first angle of the sequence, in radians.
        a2: The second angle, in radians.
        a3: The third angle, in radians.
        sequence: Three axes, uppercase for intrinsic and lowercase for extrinsic.

    Returns:
        The X component of the rotation.

    Raises:
        PlanError: `sequence` names no axis sequence.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [3.141592653589793], "b": [0.0], "c": [0.0]})
            >>> got = bt.quat_from_euler_seq_x("a", "b", "c", sequence="XYZ").round(6)
            >>> ds.select(x=got).to_pydict()
            {'x': [1.0]}
    """
    return _from("quat_from_euler_seq_x", a1, a2, a3, sequence)


def quat_from_euler_seq_y(a1: Numeric, a2: Numeric, a3: Numeric, sequence: str = "ZYX") -> Expr:
    """Build the Y component of the rotation three angles describe in any axis sequence.

    Args:
        a1: The first angle of the sequence, in radians.
        a2: The second angle, in radians.
        a3: The third angle, in radians.
        sequence: Three axes, uppercase for intrinsic and lowercase for extrinsic.

    Returns:
        The Y component of the rotation.

    Raises:
        PlanError: `sequence` names no axis sequence.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [0.0], "b": [0.0], "c": [0.0]})
            >>> ds.select(y=bt.quat_from_euler_seq_y("a", "b", "c")).to_pydict()
            {'y': [0.0]}
    """
    return _from("quat_from_euler_seq_y", a1, a2, a3, sequence)


def quat_from_euler_seq_z(a1: Numeric, a2: Numeric, a3: Numeric, sequence: str = "ZYX") -> Expr:
    """Build the Z component of the rotation three angles describe in any axis sequence.

    Args:
        a1: The first angle of the sequence, in radians.
        a2: The second angle, in radians.
        a3: The third angle, in radians.
        sequence: Three axes, uppercase for intrinsic and lowercase for extrinsic.

    Returns:
        The Z component of the rotation.

    Raises:
        PlanError: `sequence` names no axis sequence.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [0.0], "b": [0.0], "c": [0.0]})
            >>> ds.select(z=bt.quat_from_euler_seq_z("a", "b", "c")).to_pydict()
            {'z': [0.0]}
    """
    return _from("quat_from_euler_seq_z", a1, a2, a3, sequence)


def quat_from_euler_seq_w(a1: Numeric, a2: Numeric, a3: Numeric, sequence: str = "ZYX") -> Expr:
    """Build the scalar component of the rotation three angles describe in any sequence.

    Args:
        a1: The first angle of the sequence, in radians.
        a2: The second angle, in radians.
        a3: The third angle, in radians.
        sequence: Three axes, uppercase for intrinsic and lowercase for extrinsic.

    Returns:
        The scalar component of the rotation.

    Raises:
        PlanError: `sequence` names no axis sequence.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [0.0], "b": [0.0], "c": [0.0]})
            >>> ds.select(w=bt.quat_from_euler_seq_w("a", "b", "c")).to_pydict()
            {'w': [1.0]}
    """
    return _from("quat_from_euler_seq_w", a1, a2, a3, sequence)


def quat_to_euler_seq_first(
    qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric, sequence: str = "ZYX"
) -> Expr:
    """Read the first angle of a rotation in any axis sequence.

    For the default intrinsic ``"ZYX"`` this is the yaw.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.
        sequence: Three axes, uppercase for intrinsic and lowercase for extrinsic.

    Returns:
        The angle in radians on ``[-pi, pi)``, or null for a zero quaternion.

    Raises:
        PlanError: `sequence` names no axis sequence.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0], "qy": [0.0], "qz": [0.7071067811865476],
            ...      "qw": [0.7071067811865476]}
            ... )
            >>> got = bt.quat_to_euler_seq_first("qx", "qy", "qz", "qw").round(4)
            >>> ds.select(yaw=got).to_pydict()
            {'yaw': [1.5708]}
    """
    return _to("quat_to_euler_seq_first", (qx, qy, qz, qw), sequence)


def quat_to_euler_seq_second(
    qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric, sequence: str = "ZYX"
) -> Expr:
    """Read the second angle of a rotation in any axis sequence.

    For the default intrinsic ``"ZYX"`` this is the pitch.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.
        sequence: Three axes, uppercase for intrinsic and lowercase for extrinsic.

    Returns:
        The angle in radians, or null for a zero quaternion.

    Raises:
        PlanError: `sequence` names no axis sequence.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]})
            >>> got = bt.quat_to_euler_seq_second("qx", "qy", "qz", "qw")
            >>> ds.select(pitch=got).to_pydict()
            {'pitch': [0.0]}
    """
    return _to("quat_to_euler_seq_second", (qx, qy, qz, qw), sequence)


def quat_to_euler_seq_third(
    qx: Numeric, qy: Numeric, qz: Numeric, qw: Numeric, sequence: str = "ZYX"
) -> Expr:
    """Read the third angle of a rotation in any axis sequence.

    For the default intrinsic ``"ZYX"`` this is the roll. At gimbal lock it is zero.

    Args:
        qx: The rotation's X component.
        qy: The rotation's Y component.
        qz: The rotation's Z component.
        qw: The rotation's scalar component.
        sequence: Three axes, uppercase for intrinsic and lowercase for extrinsic.

    Returns:
        The angle in radians on ``[-pi, pi)``, or null for a zero quaternion.

    Raises:
        PlanError: `sequence` names no axis sequence.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]})
            >>> got = bt.quat_to_euler_seq_third("qx", "qy", "qz", "qw")
            >>> ds.select(roll=got).to_pydict()
            {'roll': [0.0]}
    """
    return _to("quat_to_euler_seq_third", (qx, qy, qz, qw), sequence)


def quat_from_euler_seq(angles: Point, *, sequence: str, prefix: str = "") -> dict[str, Expr]:
    """Build a rotation from three angles in any axis sequence, as four named columns.

    Args:
        angles: The three angles in radians, in the order `sequence` lists the axes.
        sequence: Three axes, uppercase for intrinsic and lowercase for extrinsic.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``qx``/``qy``/``qz``/``qw`` (each with `prefix`) to the rotation.

    Raises:
        PlanError: `sequence` names no axis sequence.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [0.0], "b": [0.0], "c": [0.0]})
            >>> ds.select(**bt.quat_from_euler_seq(("a", "b", "c"), sequence="zxz")).to_pydict()
            {'qx': [0.0], 'qy': [0.0], 'qz': [0.0], 'qw': [1.0]}
    """
    a1, a2, a3 = angles
    return {
        f"{prefix}q{axis}": _from(f"quat_from_euler_seq_{axis}", a1, a2, a3, sequence)
        for axis in "xyzw"
    }


def quat_to_euler_seq(q: Quaternion, *, sequence: str, prefix: str = "") -> dict[str, Expr]:
    """Read a rotation as three angles in any axis sequence, as three named columns.

    Args:
        q: The rotation, as ``(qx, qy, qz, qw)``.
        sequence: Three axes, uppercase for intrinsic and lowercase for extrinsic.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``angle_1``/``angle_2``/``angle_3`` (each with `prefix`) to the
        angles in radians, in the order `sequence` lists the axes.

    Raises:
        PlanError: `sequence` names no axis sequence.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0]})
            >>> q = ("qx", "qy", "qz", "qw")
            >>> ds.select(**bt.quat_to_euler_seq(q, sequence="xyz")).to_pydict()
            {'angle_1': [0.0], 'angle_2': [0.0], 'angle_3': [0.0]}
    """
    names = ("first", "second", "third")
    return {
        f"{prefix}angle_{i}": _to(f"quat_to_euler_seq_{name}", q, sequence)
        for i, name in enumerate(names, start=1)
    }
