"""Result types of the temporal nodes whose type depends on their input's type.

The name-only temporal answers (`year` is Int64, `dayname` is String) live in `scalars`.
These are the ones that must see the operand: a calendar operation keeps the *zone* of a
tz-aware input, and a declared type that drops it while the engine keeps it (or the
reverse) is a schema-contract violation -- the device tier refuses a result whose types
disagree with `available_schema`, and a user reading ``Dataset.schema`` is told the wrong
thing. Every rule here mirrors a `bc-expr` kernel:

* `date_trunc`, `offset_by`, per-row month/day shifts and `add_business_days` run on the
  zone's wall clock and keep the label (`bc_expr::eval::temporal::timezone::on_wall_clock`).
* `convert_timezone` answers a naive wall clock; `replace_timezone` answers its target zone.
* The window keys keep the input's zone, at microsecond resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.plan.expr_ir import Expr
    from batcher.plan.schema import SchemaRef
    from batcher.plan.types.infer.arithmetic import InferFn

__all__ = ["calendar_shift_type", "temporal_node_type"]


def _zone(source: pa.DataType | None) -> str | None:
    """The zone of a timestamp type, else ``None``."""
    return source.tz if source is not None and pa.types.is_timestamp(source) else None


def calendar_shift_type(source: pa.DataType | None) -> pa.DataType | None:
    """The type of a calendar shift of `source`: a date stays a date, a timestamp keeps its zone.

    Args:
        source: The operand's type, or ``None`` if unknown.

    Returns:
        ``date32`` for a date, ``timestamp[us, tz]`` for a timestamp or text (text is parsed
        as a timestamp first), ``None`` otherwise.
    """
    if source is None:
        return None
    if pa.types.is_date32(source):
        return pa.date32()
    if pa.types.is_timestamp(source):
        return pa.timestamp("us", tz=source.tz)
    if pa.types.is_string(source) or pa.types.is_large_string(source) or pa.types.is_null(source):
        return pa.timestamp("us")
    return None


def temporal_node_type(expr: Expr, schema: SchemaRef, infer: InferFn) -> pa.DataType | None:
    """The result type of a zone-sensitive temporal node, or ``None`` if `expr` is not one.

    Args:
        expr: The expression to type.
        schema: The operator's input schema.
        infer: The recursive `infer_type`, for the operand.

    Returns:
        The Arrow type, or ``None`` when uncertain or not a node handled here.
    """
    from batcher.plan.expr_ir.func_nodes import (
        BusinessDay,
        ConvertTimezone,
        DateOffset,
        DateTrunc,
        ReplaceTimezone,
        WindowBuckets,
        WindowStart,
    )

    if isinstance(expr, ConvertTimezone):
        return pa.timestamp("us")  # the wall clock in `to_tz`, without a zone
    if isinstance(expr, ReplaceTimezone):
        return pa.timestamp("us", tz=expr.tz)
    if isinstance(expr, DateTrunc):
        # `date_trunc` answers a microsecond timestamp for a date or a timestamp, unless
        # `preserve_type` keeps a date a date; a zoned input keeps its zone.
        source = infer(expr.input, schema)
        if expr.preserve_type and source is None:
            return None  # the engine answers the input's type, which is unknown here
        if expr.preserve_type and pa.types.is_date32(source):
            return pa.date32()
        return pa.timestamp("us", tz=_zone(source))
    if isinstance(expr, DateOffset):
        return calendar_shift_type(infer(expr.input, schema))
    if isinstance(expr, (WindowStart, WindowBuckets)):
        # A timestamp whatever the input was -- NOT the input's own type, which `_sql`'s
        # bucket lowering depends on (it casts a DATE's bucket back to a date) -- but in the
        # input's zone, which the kernel keeps.
        stamp = pa.timestamp("us", tz=_zone(infer(expr.input, schema)))
        return stamp if isinstance(expr, WindowStart) else pa.list_(stamp)
    if isinstance(expr, BusinessDay):
        if expr.fn == "count":
            return pa.int64()
        if expr.fn == "is":
            return pa.bool_()
        return calendar_shift_type(infer(expr.input, schema))
    return None
