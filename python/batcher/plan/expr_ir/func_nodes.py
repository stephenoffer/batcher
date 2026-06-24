"""IR node classes built by the accessor namespaces (`.str`/`.dt`/`.list`/…).

These are the `Expr` subclasses the namespace accessors in `namespaces.py`
construct (one per function family). They are split out from `namespaces.py` so
that file holds only the accessor logic and dispatch tables; the edge points one
way (`namespaces` → `func_nodes` → `core`). The `Expr` base never references them.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir.core import Expr, Lit
from batcher.plan.ir_tags import ExprTag


class StrFunc(Expr):
    """A string function over a sub-expression. Built via the `.str` namespace."""

    __slots__ = ("fn", "input", "length", "pattern", "replacement", "start")

    def __init__(
        self,
        fn: str,
        input: Expr,
        *,
        pattern: str | None = None,
        replacement: str | None = None,
        start: int | None = None,
        length: int | None = None,
    ) -> None:
        self.fn = fn
        self.input = input
        self.pattern = pattern
        self.replacement = replacement
        self.start = start
        self.length = length

    def to_ir(self) -> dict[str, Any]:
        ir: dict[str, Any] = {"e": ExprTag.STR, "fn": self.fn, "input": self.input.to_ir()}
        if self.pattern is not None:
            ir["pattern"] = self.pattern
        if self.replacement is not None:
            ir["replacement"] = self.replacement
        if self.start is not None:
            ir["start"] = self.start
        if self.length is not None:
            ir["length"] = self.length
        return ir


class DateFunc(Expr):
    """A date/time field extraction. Built via the `.dt` namespace."""

    __slots__ = ("fn", "input")

    def __init__(self, fn: str, input: Expr) -> None:
        self.fn = fn
        self.input = input

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.DATE, "fn": self.fn, "input": self.input.to_ir()}


class DateTrunc(Expr):
    """`date_trunc(unit, ts)` — truncate to the start of a unit. → Timestamp."""

    __slots__ = ("input", "unit")

    def __init__(self, input: Expr, unit: str) -> None:
        self.input = input
        self.unit = unit

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.DATE_TRUNC, "input": self.input.to_ir(), "unit": self.unit}


class ListBinary(Expr):
    """A pairwise reduction over two List columns (`dot`/`cosine_similarity`/
    `l2_distance`). Built via the `.list` namespace. → Float64."""

    __slots__ = ("fn", "left", "right")

    def __init__(self, fn: str, left: Expr, right: Expr) -> None:
        self.fn = fn
        self.left = left
        self.right = right

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.LIST_BINARY,
            "fn": self.fn,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
        }


class ListSet(Expr):
    """A set op between two List columns (`array_intersect`/`array_except`) — the
    distinct left elements present in / absent from the right list. Built via
    ``.list.intersect`` / ``.list.difference``. → List."""

    __slots__ = ("fn", "left", "right")

    def __init__(self, fn: str, left: Expr, right: Expr) -> None:
        self.fn = fn
        self.left = left
        self.right = right

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.LIST_SET,
            "fn": self.fn,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
        }


class ListTransform(Expr):
    """`list.transform(func)` — apply the element sub-expression `func` (over
    ``element()``) to every list element, preserving lengths. → List."""

    __slots__ = ("func", "input")

    def __init__(self, input: Expr, func: Expr) -> None:
        self.input = input
        self.func = func

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.LIST_TRANSFORM, "input": self.input.to_ir(), "func": self.func.to_ir()}


class ListFilter(Expr):
    """`list.filter(pred)` — keep elements where the boolean element predicate `pred`
    (over ``element()``) is true. → List."""

    __slots__ = ("input", "pred")

    def __init__(self, input: Expr, pred: Expr) -> None:
        self.input = input
        self.pred = pred

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.LIST_FILTER, "input": self.input.to_ir(), "pred": self.pred.to_ir()}


class Strftime(Expr):
    """`strftime(ts, format)` — format a Date/Timestamp with a chrono/strftime
    format string (e.g. ``%Y-%m-%d``). → Utf8."""

    __slots__ = ("format", "input")

    def __init__(self, input: Expr, format: str) -> None:
        self.input = input
        self.format = format

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.STRFTIME, "input": self.input.to_ir(), "format": self.format}


class Strptime(Expr):
    """`strptime(s, format)` — parse a string column into a Timestamp using a
    chrono/strftime format (e.g. ``%Y-%m-%d %H:%M:%S``). Values that do not match
    become NULL (DuckDB ``try_strptime``). → Timestamp(us)."""

    __slots__ = ("format", "input")

    def __init__(self, input: Expr, format: str) -> None:
        self.input = input
        self.format = format

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.STRPTIME, "input": self.input.to_ir(), "format": self.format}


class ConvertTimezone(Expr):
    """`convert_timezone(from_tz, to_tz, ts)` — shift each naive timestamp's
    wall-clock from `from_tz` to `to_tz` (DST-aware). Type-preserving (Timestamp)."""

    __slots__ = ("from_tz", "input", "to_tz")

    def __init__(self, input: Expr, from_tz: str, to_tz: str) -> None:
        self.input = input
        self.from_tz = from_tz
        self.to_tz = to_tz

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.CONVERT_TIMEZONE,
            "input": self.input.to_ir(),
            "from_tz": self.from_tz,
            "to_tz": self.to_tz,
        }


class DateOffset(Expr):
    """`offset_by` — shift a Date/Timestamp by `months` (calendar), `days`, and
    `micros` (exact). Built via ``.dt.offset_by``. Type-preserving."""

    __slots__ = ("days", "input", "micros", "months")

    def __init__(self, input: Expr, months: int, days: int, micros: int) -> None:
        self.input = input
        self.months = months
        self.days = days
        self.micros = micros

    def to_ir(self) -> dict[str, Any]:
        ir: dict[str, Any] = {"e": ExprTag.DATE_OFFSET, "input": self.input.to_ir()}
        # Omit zero components (serde defaults them) to keep the IR compact.
        if self.months:
            ir["months"] = self.months
        if self.days:
            ir["days"] = self.days
        if self.micros:
            ir["micros"] = self.micros
        return ir


class WindowStart(Expr):
    """`window_start(ts, width, origin)` — the start of the fixed-width tumbling
    window containing each instant. Built via `window(col, duration)`. → Timestamp."""

    __slots__ = ("input", "origin_micros", "width_micros")

    def __init__(self, input: Expr, width_micros: int, origin_micros: int = 0) -> None:
        self.input = input
        self.width_micros = width_micros
        self.origin_micros = origin_micros

    def to_ir(self) -> dict[str, Any]:
        ir: dict[str, Any] = {
            "e": ExprTag.WINDOW_START,
            "input": self.input.to_ir(),
            "width_micros": self.width_micros,
        }
        if self.origin_micros:
            ir["origin_micros"] = self.origin_micros
        return ir


class WindowBuckets(Expr):
    """`window_buckets(ts, width, slide)` — the starts of every sliding window that
    contains each instant, as a `List<Timestamp>`. Built via `window(col, duration,
    slide)`; fan out with `unnest` then group-by the start."""

    __slots__ = ("input", "slide_micros", "width_micros")

    def __init__(self, input: Expr, width_micros: int, slide_micros: int) -> None:
        self.input = input
        self.width_micros = width_micros
        self.slide_micros = slide_micros

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.WINDOW_BUCKETS,
            "input": self.input.to_ir(),
            "width_micros": self.width_micros,
            "slide_micros": self.slide_micros,
        }


class ListFunc(Expr):
    """A per-row scalar reduction over a List column. Built via the `.list`
    namespace. `len`/`n_unique` → Int64; `sum`/`min`/`max`/`mean` → Float64."""

    __slots__ = ("fn", "input")

    def __init__(self, fn: str, input: Expr) -> None:
        self.fn = fn
        self.input = input

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.LIST, "fn": self.fn, "input": self.input.to_ir()}


class ListGet(Expr):
    """`list[index]` — the element at `index` of each row's list (null where the
    row is null or the index is out of range). Negative indices count from the end.
    Type-preserving."""

    __slots__ = ("index", "input")

    def __init__(self, input: Expr, index: int) -> None:
        self.input = input
        self.index = index

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.LIST_GET, "input": self.input.to_ir(), "index": self.index}


class ListContains(Expr):
    """`list.contains(value)` — true where any element equals the literal. → Bool."""

    __slots__ = ("input", "value")

    def __init__(self, input: Expr, value: int | float | bool | str) -> None:
        self.input = input
        self.value = value

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.LIST_CONTAINS,
            "input": self.input.to_ir(),
            "value": Lit(self.value).to_ir()["value"],
        }


class ListPosition(Expr):
    """`list.position(value)` — the 1-based index of the first element equal to the
    literal, 0 if absent (DuckDB `list_position`). → Int64."""

    __slots__ = ("input", "value")

    def __init__(self, input: Expr, value: int | float | bool | str) -> None:
        self.input = input
        self.value = value

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.LIST_POSITION,
            "input": self.input.to_ir(),
            "value": Lit(self.value).to_ir()["value"],
        }


class ListSlice(Expr):
    """`list.slice(offset, length)` — the 0-based sub-range of each row's list."""

    __slots__ = ("input", "length", "offset")

    def __init__(self, input: Expr, offset: int, length: int | None = None) -> None:
        self.input = input
        self.offset = offset
        self.length = length

    def to_ir(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "e": ExprTag.LIST_SLICE,
            "input": self.input.to_ir(),
            "offset": self.offset,
        }
        if self.length is not None:
            d["length"] = self.length
        return d


class StructField(Expr):
    """`struct.field` — a named field of a Struct column (type-preserving;
    null where the struct row is null)."""

    __slots__ = ("field", "input")

    def __init__(self, input: Expr, field: str) -> None:
        self.input = input
        self.field = field

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.STRUCT_FIELD, "input": self.input.to_ir(), "field": self.field}


class MapFunc(Expr):
    """A Map-column accessor (``map_keys``/``map_values``/``element_at``) built via
    the ``.map`` namespace. `element_at` carries a literal lookup ``key``."""

    __slots__ = ("fn", "input", "key")

    def __init__(self, fn: str, input: Expr, key: object | None = None) -> None:
        self.fn = fn
        self.input = input
        self.key = key

    def to_ir(self) -> dict[str, Any]:
        ir: dict[str, Any] = {"e": ExprTag.MAP, "fn": self.fn, "input": self.input.to_ir()}
        if self.key is not None:
            ir["key"] = Lit(self.key).to_ir()["value"]
        return ir
