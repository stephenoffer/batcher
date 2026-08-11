"""The one builder every rigid-body function is written in terms of.

Each function in this package is a thin, typed, documented wrapper around a single
`SpatialFunc` node. Keeping the node construction here means the wrappers carry no logic
at all — a signature, a docstring, and one call — so the family reads uniformly and a
new function cannot accidentally lower differently from its neighbours.

Arguments follow the **relational** convention, not the geospatial one: a bare string is
a *column name*, and a number is a literal. Every argument in this family is a
coordinate, an angle or an interpolation fraction, and those come out of columns far
more often than they are typed inline — ``se3_transform_x("tx", "ty", "tz", ...)`` is
what a pose table looks like. That is the opposite of ``plan.functions.geo``, where a
bare string is a literal geometry, and the two differ because their arguments do.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr, Lit
from batcher.plan.expr_ir.func_nodes import SpatialFunc
from batcher.plan.expr_ir.nodes import Col

__all__ = ["Numeric", "Point", "Pose", "Quaternion", "spatial_call", "value"]

#: A single numeric argument: an expression, a column name, or a constant.
Numeric = Expr | str | float | int

#: A rotation, as its four components in ``(x, y, z, w)`` order — scalar last, the ROS
#: and SciPy order. See :doc:`/user-guide/analyze/robotics` for why the order is spelled
#: out at every call site rather than inferred.
Quaternion = tuple[Numeric, Numeric, Numeric, Numeric]

#: A position or displacement, as its three components in ``(x, y, z)`` order.
Point = tuple[Numeric, Numeric, Numeric]

#: A rigid transform, as seven values: the translation ``(tx, ty, tz)`` first, then the
#: rotation ``(qx, qy, qz, qw)``. Translation first because that is the order
#: ``geometry_msgs/Pose`` and every log schema derived from it writes them.
Pose = tuple[Numeric, Numeric, Numeric, Numeric, Numeric, Numeric, Numeric]


def value(v: Numeric) -> Expr:
    """Coerce one argument to an `Expr`.

    Args:
        v: An expression, a column name, or a numeric constant.

    Returns:
        The argument as an expression: a string becomes a column reference, a number
        becomes a literal, and an expression passes through.

    Examples:
        .. doctest::

            >>> from batcher.plan.functions.spatial._build import value
            >>> value("qw").to_ir()
            {'e': 'col', 'name': 'qw'}

            >>> value(1.0).to_ir()
            {'e': 'lit', 'value': {'float': 1.0}}
    """
    if isinstance(v, Expr):
        return v
    if isinstance(v, str):
        return Col(v)
    return Lit(v)


def spatial_call(fn: str, *args: Numeric) -> Expr:
    """Build the `SpatialFunc` node for `fn` over `args`.

    Args:
        fn: The engine function name, validated against the ``SPATIAL_FNS`` vocabulary.
        *args: The arguments, in engine order, each coerced by `value`.

    Returns:
        The rigid-body expression.

    Examples:
        .. doctest::

            >>> from batcher.plan.functions.spatial._build import spatial_call
            >>> spatial_call("quat_norm", "qx", "qy", "qz", "qw").to_ir()["fn"]
            'quat_norm'
    """
    return SpatialFunc(fn, [value(a) for a in args])
