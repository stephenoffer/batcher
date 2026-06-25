"""IR node classes built by the accessor namespaces (`.str`/`.dt`/`.list`/…).

These are the `Expr` subclasses the namespace accessors in `namespaces.py`
construct (one per function family). They are split out from `namespaces.py` so
that file holds only the accessor logic and dispatch tables; the edge points one
way (`namespaces` → `func_nodes` → `core`). The `Expr` base never references them.

Each node is a declarative `IRNode`: it sets a wire ``tag`` and annotates its
fields with the `child`/`scalar`/`literal` factories, and the generic
`IRNode.to_ir` assembles the JSON. See `node_base` for the mechanism.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.fn_names import (
    DATE_FNS,
    LIST_FNS,
    STR_FNS,
    ListBinaryFn,
    ListSetFn,
    MapFn,
)
from batcher.plan.expr_ir.node_base import IRNode, child, expr_node, literal, scalar
from batcher.plan.ir_tags import ExprTag


@expr_node
class StrFunc(IRNode):
    """A string function over a sub-expression. Built via the `.str` namespace."""

    tag = ExprTag.STR
    vocab = STR_FNS
    fn: str = scalar()
    input: Expr = child()
    pattern: str | None = scalar(omit_none=True, default=None)
    replacement: str | None = scalar(omit_none=True, default=None)
    start: int | None = scalar(omit_none=True, default=None)
    length: int | None = scalar(omit_none=True, default=None)


@expr_node
class DateFunc(IRNode):
    """A date/time field extraction. Built via the `.dt` namespace."""

    tag = ExprTag.DATE
    vocab = DATE_FNS
    fn: str = scalar()
    input: Expr = child()


@expr_node
class DateTrunc(IRNode):
    """`date_trunc(unit, ts)` — truncate to the start of a unit. → Timestamp."""

    tag = ExprTag.DATE_TRUNC
    input: Expr = child()
    unit: str = scalar()


@expr_node
class ListBinary(IRNode):
    """A pairwise reduction over two List columns (`dot`/`cosine_similarity`/
    `l2_distance`). Built via the `.list` namespace. → Float64."""

    tag = ExprTag.LIST_BINARY
    vocab = frozenset(ListBinaryFn)
    fn: str = scalar()
    left: Expr = child()
    right: Expr = child()


@expr_node
class ListSet(IRNode):
    """A set op between two List columns (`array_intersect`/`array_except`) — the
    distinct left elements present in / absent from the right list. Built via
    ``.list.intersect`` / ``.list.difference``. → List."""

    tag = ExprTag.LIST_SET
    vocab = frozenset(ListSetFn)
    fn: str = scalar()
    left: Expr = child()
    right: Expr = child()


@expr_node
class ListTransform(IRNode):
    """`list.transform(func)` — apply the element sub-expression `func` (over
    ``element()``) to every list element, preserving lengths. → List."""

    tag = ExprTag.LIST_TRANSFORM
    input: Expr = child()
    func: Expr = child()


@expr_node
class ListFilter(IRNode):
    """`list.filter(pred)` — keep elements where the boolean element predicate `pred`
    (over ``element()``) is true. → List."""

    tag = ExprTag.LIST_FILTER
    input: Expr = child()
    pred: Expr = child()


@expr_node
class Strftime(IRNode):
    """`strftime(ts, format)` — format a Date/Timestamp with a chrono/strftime
    format string (e.g. ``%Y-%m-%d``). → Utf8."""

    tag = ExprTag.STRFTIME
    input: Expr = child()
    format: str = scalar()


@expr_node
class Strptime(IRNode):
    """`strptime(s, format)` — parse a string column into a Timestamp using a
    chrono/strftime format (e.g. ``%Y-%m-%d %H:%M:%S``). Values that do not match
    become NULL (DuckDB ``try_strptime``). → Timestamp(us)."""

    tag = ExprTag.STRPTIME
    input: Expr = child()
    format: str = scalar()


@expr_node
class ConvertTimezone(IRNode):
    """`convert_timezone(from_tz, to_tz, ts)` — shift each naive timestamp's
    wall-clock from `from_tz` to `to_tz` (DST-aware). Type-preserving (Timestamp)."""

    tag = ExprTag.CONVERT_TIMEZONE
    input: Expr = child()
    from_tz: str = scalar()
    to_tz: str = scalar()


@expr_node
class DateOffset(IRNode):
    """`offset_by` — shift a Date/Timestamp by `months` (calendar), `days`, and
    `micros` (exact). Built via ``.dt.offset_by``. Type-preserving.

    Zero components are omitted from the IR (serde defaults them) to keep it compact.
    """

    tag = ExprTag.DATE_OFFSET
    input: Expr = child()
    months: int = scalar(omit_falsy=True)
    days: int = scalar(omit_falsy=True)
    micros: int = scalar(omit_falsy=True)


@expr_node
class WindowStart(IRNode):
    """`window_start(ts, width, origin)` — the start of the fixed-width tumbling
    window containing each instant. Built via `window(col, duration)`. → Timestamp."""

    tag = ExprTag.WINDOW_START
    input: Expr = child()
    width_micros: int = scalar()
    origin_micros: int = scalar(omit_falsy=True, default=0)


@expr_node
class WindowBuckets(IRNode):
    """`window_buckets(ts, width, slide)` — the starts of every sliding window that
    contains each instant, as a `List<Timestamp>`. Built via `window(col, duration,
    slide)`; fan out with `unnest` then group-by the start."""

    tag = ExprTag.WINDOW_BUCKETS
    input: Expr = child()
    width_micros: int = scalar()
    slide_micros: int = scalar()


@expr_node
class ListFunc(IRNode):
    """A per-row scalar reduction over a List column. Built via the `.list`
    namespace. `len`/`n_unique` → Int64; `sum`/`min`/`max`/`mean` → Float64."""

    tag = ExprTag.LIST
    vocab = LIST_FNS
    fn: str = scalar()
    input: Expr = child()


@expr_node
class ListGet(IRNode):
    """`list[index]` — the element at `index` of each row's list (null where the
    row is null or the index is out of range). Negative indices count from the end.
    Type-preserving."""

    tag = ExprTag.LIST_GET
    input: Expr = child()
    index: int = scalar()


@expr_node
class ListContains(IRNode):
    """`list.contains(value)` — true where any element equals the literal. → Bool."""

    tag = ExprTag.LIST_CONTAINS
    input: Expr = child()
    value: int | float | bool | str = literal()


@expr_node
class ListPosition(IRNode):
    """`list.position(value)` — the 1-based index of the first element equal to the
    literal, 0 if absent (DuckDB `list_position`). → Int64."""

    tag = ExprTag.LIST_POSITION
    input: Expr = child()
    value: int | float | bool | str = literal()


@expr_node
class ListSlice(IRNode):
    """`list.slice(offset, length)` — the 0-based sub-range of each row's list."""

    tag = ExprTag.LIST_SLICE
    input: Expr = child()
    offset: int = scalar()
    length: int | None = scalar(omit_none=True, default=None)


@expr_node
class StructField(IRNode):
    """`struct.field` — a named field of a Struct column (type-preserving;
    null where the struct row is null)."""

    tag = ExprTag.STRUCT_FIELD
    input: Expr = child()
    field: str = scalar()


@expr_node
class MapFunc(IRNode):
    """A Map-column accessor (``map_keys``/``map_values``/``element_at``) built via
    the ``.map`` namespace. `element_at` carries a literal lookup ``key``."""

    tag = ExprTag.MAP
    vocab = frozenset(MapFn)
    fn: str = scalar()
    input: Expr = child()
    key: object | None = literal(omit_none=True, default=None)
