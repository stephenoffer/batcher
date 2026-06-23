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
