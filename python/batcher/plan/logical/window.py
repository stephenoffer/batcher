"""Window-function logical nodes: `WindowFuncSpec` and `Window`.

These use the `WINDOW_*` frozensets from `ir_tags` to validate function names and
their input/order requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Expr
from batcher.plan.ir_tags import (
    WINDOW_AGGREGATES,
    WINDOW_FUNCS,
    WINDOW_RANKING,
    WINDOW_VALUE,
    Op,
)
from batcher.plan.logical.aggregate import SortKeySpec
from batcher.plan.logical.base import LogicalPlan, _validate_refs

__all__ = ["Window", "WindowFrame", "WindowFuncSpec"]


@dataclass(frozen=True, slots=True)
class WindowFrame:
    """An explicit ``ROWS`` window frame.

    `start` and `end` are signed row offsets from the current row: negative =
    *preceding*, ``0`` = current row, positive = *following*; ``None`` = unbounded
    (`UNBOUNDED PRECEDING` for `start`, `UNBOUNDED FOLLOWING` for `end`). For
    example ``WindowFrame(-2, 0)`` is ``ROWS BETWEEN 2 PRECEDING AND CURRENT ROW``
    (a trailing 3-row window). Only ``ROWS`` frames are supported; the default
    (no frame) is SQL's ``RANGE`` running/whole-partition frame.
    """

    start: int | None
    end: int | None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise PlanError(f"window frame start {self.start} is after end {self.end}")

    def to_ir(self) -> dict[str, Any]:
        return {
            "units": "rows",
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
    lag/lead distance (ignored otherwise). `frame` is an explicit ``ROWS`` frame,
    valid only on the aggregate functions.
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
        if self.frame is not None and self.func not in WINDOW_AGGREGATES:
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
