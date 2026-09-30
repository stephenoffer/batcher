"""Measuring and repairing a rotation that has drifted, and giving each rotation one sign.

`quat_from_rotmat_x` and its siblings refuse a matrix more than ``1e-4`` from being a
rotation, which is the right default: a zero matrix from a missing calibration or a scaled
one from a units mix-up is a bug to surface, not a rotation to guess at. The functions
here make the tolerance yours instead of the engine's. `rotmat_orthogonality_error` and
`rotmat_determinant` report the two things the strict reader checks, separately, so a
pipeline can see how far its matrices drift and choose its own threshold, and
`quat_from_rotmat_nearest_x` and its siblings return the rotation nearest a drifted matrix
rather than null.

`quat_canonicalize` is the other half of making rotations comparable. A quaternion and its
negation are the same rotation, so a `group_by`, a `distinct` or an equality join over raw
components treats one rotation as two. Canonicalizing first gives every rotation a single
spelling.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import greatest, when
from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.spatial._build import (
    Matrix,
    Numeric,
    Quaternion,
    spatial_call,
    value,
)
from batcher.plan.functions.spatial.quaternion import (
    quat_normalize_w,
    quat_normalize_x,
    quat_normalize_y,
    quat_normalize_z,
)

__all__ = [
    "quat_canonicalize",
    "quat_from_rotmat_nearest",
    "quat_from_rotmat_nearest_w",
    "quat_from_rotmat_nearest_x",
    "quat_from_rotmat_nearest_y",
    "quat_from_rotmat_nearest_z",
    "rotmat_determinant",
    "rotmat_orthogonality_error",
]


def _rows(matrix: Matrix) -> list[list[Expr]]:
    if len(matrix) != 9:
        raise PlanError(f"a 3x3 matrix is nine values in row-major order, got {len(matrix)}")
    m = [value(v) for v in matrix]
    return [m[0:3], m[3:6], m[6:9]]


def quat_from_rotmat_nearest_x(
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
    """Build the X component of the rotation nearest a 3x3 matrix.

    The repair for a matrix that has drifted past `quat_from_rotmat_x`'s ``1e-4``
    tolerance: the rotation closest to it in the Frobenius norm, which is the orthogonal
    factor an SVD-based repair such as SciPy's ``Rotation.from_matrix`` returns. Only a
    matrix with a non-finite entry or a determinant that is not positive gives null. A
    reflection is still refused, because no amount of rounding turns a rotation into
    one. Arguments are row-major: ``m01`` is row 0, column 1.

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
        The X component of the nearest rotation, or null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"big": [1.1], "nil": [0.0]})
            >>> cols = ["big", "nil", "nil", "nil", "big", "nil", "nil", "nil", "big"]
            >>> ds.select(x=bt.quat_from_rotmat_nearest_x(*cols)).to_pydict()
            {'x': [0.0]}
    """
    return spatial_call("quat_from_rotmat_nearest_x", m00, m01, m02, m10, m11, m12, m20, m21, m22)


def quat_from_rotmat_nearest_y(
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
    """Build the Y component of the rotation nearest a 3x3 matrix.

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
        The Y component of the nearest rotation, or null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"big": [1.1], "nil": [0.0]})
            >>> cols = ["big", "nil", "nil", "nil", "big", "nil", "nil", "nil", "big"]
            >>> ds.select(y=bt.quat_from_rotmat_nearest_y(*cols)).to_pydict()
            {'y': [0.0]}
    """
    return spatial_call("quat_from_rotmat_nearest_y", m00, m01, m02, m10, m11, m12, m20, m21, m22)


def quat_from_rotmat_nearest_z(
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
    """Build the Z component of the rotation nearest a 3x3 matrix.

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
        The Z component of the nearest rotation, or null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"big": [1.1], "nil": [0.0]})
            >>> cols = ["big", "nil", "nil", "nil", "big", "nil", "nil", "nil", "big"]
            >>> ds.select(z=bt.quat_from_rotmat_nearest_z(*cols)).to_pydict()
            {'z': [0.0]}
    """
    return spatial_call("quat_from_rotmat_nearest_z", m00, m01, m02, m10, m11, m12, m20, m21, m22)


def quat_from_rotmat_nearest_w(
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
    """Build the scalar component of the rotation nearest a 3x3 matrix.

    Never negative: the nearest rotation is returned in the canonical sign that
    `quat_canonicalize` gives, so two drifted copies of one matrix agree component
    for component.

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
        The scalar component of the nearest rotation, or null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"big": [1.1], "nil": [0.0]})
            >>> cols = ["big", "nil", "nil", "nil", "big", "nil", "nil", "nil", "big"]
            >>> ds.select(w=bt.quat_from_rotmat_nearest_w(*cols)).to_pydict()
            {'w': [1.0]}
    """
    return spatial_call("quat_from_rotmat_nearest_w", m00, m01, m02, m10, m11, m12, m20, m21, m22)


def quat_from_rotmat_nearest(matrix: Matrix, *, prefix: str = "") -> dict[str, Expr]:
    """Build the rotation nearest a 3x3 matrix, as four named columns.

    Args:
        matrix: The nine entries in row-major order.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``qx``/``qy``/``qz``/``qw`` (each with `prefix`) to the nearest
        rotation, with ``qw`` never negative.

    Raises:
        PlanError: `matrix` does not have nine entries.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"big": [1.1], "nil": [0.0]})
            >>> m = ("big", "nil", "nil", "nil", "big", "nil", "nil", "nil", "big")
            >>> ds.select(**bt.quat_from_rotmat_nearest(m)).to_pydict()
            {'qx': [0.0], 'qy': [0.0], 'qz': [0.0], 'qw': [1.0]}
    """
    flat = [v for row in _rows(matrix) for v in row]
    return {
        f"{prefix}q{axis}": spatial_call(f"quat_from_rotmat_nearest_{axis}", *flat)
        for axis in "xyzw"
    }


def rotmat_orthogonality_error(matrix: Matrix) -> Expr:
    """Measure how far a 3x3 matrix is from having orthonormal rows.

    The largest absolute entry of ``M * M^T - I``. This is exactly the first of the two
    checks `quat_from_rotmat_x` makes: it accepts a matrix when this is at most ``1e-4``
    and `rotmat_determinant` is within ``1e-4`` of 1. Reporting the two separately says
    which way a matrix went wrong, a scale or shear here and a reflection there, and
    lets a pipeline pick its own threshold before `quat_from_rotmat_nearest_x` repairs
    what is left.

    Args:
        matrix: The nine entries in row-major order.

    Returns:
        The error, 0 for an exact rotation, or null when any entry is null.

    Raises:
        PlanError: `matrix` does not have nine entries.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"big": [1.1], "nil": [0.0]})
            >>> m = ("big", "nil", "nil", "nil", "big", "nil", "nil", "nil", "big")
            >>> ds.select(e=bt.rotmat_orthogonality_error(m).round(6)).to_pydict()
            {'e': [0.21]}
    """
    rows = _rows(matrix)

    def gram(i: int, j: int) -> Expr:
        a, b = rows[i], rows[j]
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    terms = [(gram(i, i) - 1.0).abs() for i in range(3)]
    terms += [gram(i, j).abs() for i in range(3) for j in range(i + 1, 3)]
    present = terms[0].is_not_null()
    for term in terms[1:]:
        present = present & term.is_not_null()
    return when(present).then(greatest(*terms)).otherwise(None)


def rotmat_determinant(matrix: Matrix) -> Expr:
    """Compute the determinant of a 3x3 matrix.

    A rotation's is exactly 1. A negative one is a reflection, the handedness mistake a
    calibration file makes when one axis is flipped, and zero is a degenerate or missing
    matrix. `quat_from_rotmat_x` accepts a matrix only when this is within ``1e-4`` of 1
    and `rotmat_orthogonality_error` is at most ``1e-4``.

    Args:
        matrix: The nine entries in row-major order.

    Returns:
        The determinant, or null when any entry is null.

    Raises:
        PlanError: `matrix` does not have nine entries.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"one": [1.0], "neg": [-1.0], "nil": [0.0]})
            >>> m = ("one", "nil", "nil", "nil", "one", "nil", "nil", "nil", "neg")
            >>> ds.select(d=bt.rotmat_determinant(m)).to_pydict()
            {'d': [-1.0]}
    """
    (a, b, c), (d, e, f), (g, h, i) = _rows(matrix)
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def quat_canonicalize(q: Quaternion, *, prefix: str = "") -> dict[str, Expr]:
    """Give a rotation its one canonical spelling, as four named columns.

    ``q`` and ``-q`` are the same rotation, and nothing in a raw log says which of the
    two a sensor will write. That is harmless to every rotation function here, but
    grouping, deduplicating or joining on the components treats one rotation as two.
    This normalizes the quaternion and flips its sign so that ``qw`` is positive, with
    ties broken by the first non-zero of ``qx``, ``qy`` and ``qz``, which is SciPy's
    canonical form. Canonicalize before comparing components, not before interpolating:
    `quat_slerp` already takes the short way round from either sign.

    Args:
        q: The rotation, as ``(qx, qy, qz, qw)``.
        prefix: Prepended to each output column name.

    Returns:
        A mapping of ``qx``/``qy``/``qz``/``qw`` (each with `prefix`) to the canonical
        unit quaternion, null for a zero quaternion.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"qx": [0.0, -0.0], "qy": [0.0, -0.0], "qz": [0.6, -0.6], "qw": [0.8, -0.8]}
            ... )
            >>> out = ds.select(**bt.quat_canonicalize(("qx", "qy", "qz", "qw")))
            >>> out.select("qz", "qw").to_pydict()
            {'qz': [0.6, 0.6], 'qw': [0.8, 0.8]}
    """
    x, y, z, w = (
        quat_normalize_x(*q),
        quat_normalize_y(*q),
        quat_normalize_z(*q),
        quat_normalize_w(*q),
    )
    zero_w = w == 0.0
    zero_wx = zero_w & (x == 0.0)
    zero_wxy = zero_wx & (y == 0.0)
    flip = (w < 0.0) | (zero_w & (x < 0.0)) | (zero_wx & (y < 0.0)) | (zero_wxy & (z < 0.0))
    return {
        f"{prefix}q{axis}": when(flip).then(-part).otherwise(part)
        for axis, part in zip("xyzw", (x, y, z, w), strict=True)
    }
