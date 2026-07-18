"""Window-function logical nodes: `WindowFuncSpec` and `Window`.

These use the `WINDOW_*` frozensets from `ir_tags` to validate function names and
their input/order requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Expr
from batcher.plan.ir_tags import (
    WINDOW_AGGREGATES,
    WINDOW_FILL,
    WINDOW_FRAMEABLE,
    WINDOW_FUNCS,
    WINDOW_RANKING,
    WINDOW_VALUE,
    Op,
)
from batcher.plan.logical.aggregate import SortKeySpec
from batcher.plan.logical.base import LogicalPlan, _validate_refs
from batcher.plan.schema import SchemaRef
from batcher.plan.types import infer_type, widen

__all__ = ["Window", "WindowFrame", "WindowFuncSpec"]


def _window_func_type(fn: WindowFuncSpec, input_schema: SchemaRef) -> pa.DataType | None:
    """The Arrow type a window function appends, or ``None`` if not certain."""
    # `percent_rank`/`cume_dist` are ranking functions but produce a fraction in [0, 1],
    # so they are Float64 — the plain-int64 ranking branch below would misreport the schema.
    if fn.func in ("percent_rank", "cume_dist"):
        return pa.float64()
    if fn.func in WINDOW_RANKING or fn.func == "count":
        return pa.int64()
    if fn.func == "avg":
        return pa.float64()
    if fn.input is None:  # value/min/max/sum all need an input
        return None
    t = infer_type(fn.input, input_schema)
    if t is None:
        return None
    if fn.func == "sum":
        return widen(t)
    if fn.func in WINDOW_VALUE or fn.func in {"min", "max"}:
        return t
    return None


_FRAME_UNITS = ("rows", "range", "groups")


@dataclass(frozen=True, slots=True)
class WindowFrame:
    """An explicit window frame.

    `start` and `end` are signed offsets from the current row: negative =
    *preceding*, ``0`` = current row, positive = *following*; ``None`` = unbounded
    (`UNBOUNDED PRECEDING` for `start`, `UNBOUNDED FOLLOWING` for `end`). `units`
    selects how the offsets are counted:

    - ``"rows"`` (default) — physical rows. ``WindowFrame(-2, 0)`` is
      ``ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`` (a trailing 3-row window).
    - ``"groups"`` — peer groups (rows sharing an ORDER BY value).
      ``WindowFrame(-1, 0, "groups")`` covers the current peer group and the one
      before it.
    - ``"range"`` — value-based peers; only peer bounds (current row / unbounded)
      are honored, e.g. ``WindowFrame(None, 0, "range")``. A numeric ``range``
      offset (value-based ``n PRECEDING``/``FOLLOWING``) is not supported and raises
      — use ``"rows"`` for a physical-row frame.
    """

    start: int | None
    end: int | None
    units: str = "rows"

    def __post_init__(self) -> None:
        if self.units not in _FRAME_UNITS:
            raise PlanError(f"window frame units must be one of {_FRAME_UNITS}, got {self.units!r}")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise PlanError(f"window frame start {self.start} is after end {self.end}")
        # A numeric RANGE offset (`n PRECEDING`/`FOLLOWING`, i.e. a non-zero bound) is
        # value-based — it selects peers whose ORDER BY *value* is within `n`, not `n`
        # rows. The engine does not implement that typed order-key arithmetic and
        # silently ran the default running frame instead, a wrong result vs SQL. Reject
        # it (only peer bounds — CURRENT ROW / UNBOUNDED — are honored for `range`),
        # matching the SQL translator, which already rejects `RANGE BETWEEN n …`.
        if self.units == "range" and (
            (self.start is not None and self.start != 0) or (self.end is not None and self.end != 0)
        ):
            raise PlanError(
                "numeric RANGE window frame offsets are not supported; only peer "
                "bounds (CURRENT ROW / UNBOUNDED) are honored for range units. Use "
                "'rows' units for a physical-row frame."
            )

    def to_ir(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "start": _bound_ir(self.start, preceding=True),
            "end": _bound_ir(self.end, preceding=False),
        }


def _bound_ir(offset: int | None, *, preceding: bool) -> dict[str, Any]:
    """One frame edge → the Rust `FrameBound` serde shape."""
    if offset is None:
        return {"kind": "unbounded_preceding" if preceding else "unbounded_following"}
    if offset == 0:
        return {"kind": "current_row"}
    if offset < 0:
        return {"kind": "preceding", "n": -offset}
    return {"kind": "following", "n": offset}


@dataclass(frozen=True, slots=True)
class WindowFuncSpec:
    """One window function: a function name, optional input expression, and alias.

    `func` is one of the Rust `WindowFn` snake_case tags (ranking
    `row_number`/`rank`/`dense_rank`; aggregates `sum`/`avg`/`min`/`max`/`count`;
    value `first_value`/`last_value`/`lag`/`lead`). The ranking functions take no
    `input`; the aggregates and value functions require one. `offset` is the
    lag/lead distance (ignored otherwise). `frame` is an explicit frame, valid on
    the aggregate functions and the positional value functions
    (`first_value`/`last_value`/`nth_value`) — which then pick the frame's
    first/last/nth row.
    """

    func: str
    input: Expr | None
    alias: str
    offset: int = 1
    frame: WindowFrame | None = None

    def __post_init__(self) -> None:
        if self.func not in WINDOW_FUNCS:
            raise PlanError(
                f"unknown window function {self.func!r}; expected one of {sorted(WINDOW_FUNCS)}"
            )
        if self.func in (WINDOW_AGGREGATES | WINDOW_VALUE) and self.input is None:
            raise PlanError(f"window function {self.func!r} requires an input column")
        if self.func in WINDOW_RANKING and self.input is not None:
            raise PlanError(f"window ranking function {self.func!r} takes no input")
        if self.frame is not None and self.func not in WINDOW_FRAMEABLE:
            raise PlanError(f"window function {self.func!r} does not support an explicit frame")

    def to_ir(self) -> dict[str, Any]:
        item: dict[str, Any] = {"func": self.func, "alias": self.alias, "offset": self.offset}
        if self.input is not None:
            item["input"] = self.input.to_ir()
        if self.frame is not None:
            item["frame"] = self.frame.to_ir()
        return item


@dataclass(frozen=True, slots=True)
class Window(LogicalPlan):
    """Window functions: partition, order within partition, append one column per
    function. The input columns are preserved. A pipeline breaker.

    `partition_keys` may be empty (one partition over all rows). The ranking
    functions (`row_number`/`rank`/`dense_rank`) require order keys. The aggregates
    (`sum`/`avg`/`min`/`max`/`count`) are **whole-partition** when `order_keys` is
    empty (every row in a partition gets the same value), or **running/cumulative**
    over the ordered partition when order keys are given — with `RANGE` peer
    semantics (tied rows share the end-of-peer-group value), matching SQL's default
    window frame.
    """

    input: LogicalPlan
    partition_keys: tuple[Expr, ...]
    order_keys: tuple[SortKeySpec, ...]
    functions: tuple[WindowFuncSpec, ...]
    # Fused per-partition top-N (`QUALIFY <rank> <= k`): keep only rows whose ranking
    # value is `<= rank_limit`. Set by the `qualify_to_partition_topn` optimizer rule
    # for a single ranking function; None = a plain window. See the Rust
    # `RelOp::Window::rank_limit`.
    rank_limit: int | None = None

    def __post_init__(self) -> None:
        if not self.functions:
            raise PlanError("window requires at least one function")
        if self.rank_limit is not None:
            if len(self.functions) != 1 or self.functions[0].func not in WINDOW_RANKING:
                raise PlanError(
                    "window rank_limit requires exactly one ranking function "
                    "(row_number/rank/dense_rank)"
                )
            if self.rank_limit < 0:
                raise PlanError(f"window rank_limit must be non-negative, got {self.rank_limit}")
        available = set(self.input.available_columns())
        for expr in self.partition_keys:
            _validate_refs(expr, available, what="window partition key")
        for key in self.order_keys:
            _validate_refs(key.expr, available, what="window order key")
        for fn in self.functions:
            if fn.func in WINDOW_RANKING and not self.order_keys:
                raise PlanError(f"window ranking function {fn.func!r} requires order_by keys")
            if fn.func in WINDOW_FILL and not self.order_keys:
                # Without an order there is no "previous" row: the result would depend on
                # arrival order, which a morselized/distributed scan does not fix.
                raise PlanError(
                    f"window function {fn.func!r} requires order_by keys — a fill carries "
                    "values along a defined row order, and an unordered relation has none"
                )
            if fn.input is not None:
                _validate_refs(fn.input, available, what=f"window function {fn.alias!r}")
        # Aliases must not collide with input columns or each other.
        seen = set(self.input.available_columns())
        for fn in self.functions:
            if fn.alias in seen:
                raise PlanError(
                    f"window output column {fn.alias!r} collides with an existing column"
                )
            seen.add(fn.alias)

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.WINDOW,
            "input": self.input.to_ir(),
            "partition_keys": [e.to_ir() for e in self.partition_keys],
            "order_keys": [
                {
                    "expr": k.expr.to_ir(),
                    "descending": k.descending,
                    "nulls_first": k.nulls_first,
                }
                for k in self.order_keys
            ],
            "functions": [fn.to_ir() for fn in self.functions],
            "rank_limit": self.rank_limit,
        }

    def available_columns(self) -> list[str]:
        return self.input.available_columns() + [fn.alias for fn in self.functions]

    def available_schema(self) -> SchemaRef | None:
        inp = self.input.available_schema()
        if inp is None:
            return None
        fields: list[pa.Field] = list(inp.arrow)
        for fn in self.functions:
            t = _window_func_type(fn, inp)
            if t is None:
                return None
            fields.append(pa.field(fn.alias, t))
        return SchemaRef.from_arrow(pa.schema(fields))
