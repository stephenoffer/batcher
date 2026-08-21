"""Window-function logical nodes: `WindowFuncSpec` and `Window`.

These use the `WINDOW_*` frozensets from `ir_tags` to validate function names and
their input/order requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, Expr
from batcher.plan.ir_tags import (
    FRAME_UNITS,
    WINDOW_AGGREGATES,
    WINDOW_EWM,
    WINDOW_FILL,
    WINDOW_FRAMEABLE,
    WINDOW_FUNCS,
    WINDOW_RANKING,
    WINDOW_SERIES,
    WINDOW_VALUE,
    Op,
)
from batcher.plan.logical.base import (
    LogicalPlan,
    SortKeySpec,
    _validate_refs,
    available_column_set,
)
from batcher.plan.schema import SchemaRef
from batcher.plan.types import infer_type, widen

__all__ = ["Window", "WindowFrame", "WindowFuncSpec"]


def _window_func_type(fn: WindowFuncSpec, input_schema: SchemaRef) -> pa.DataType | None:
    """The Arrow type a window function appends, or ``None`` if not certain."""
    # `percent_rank`/`cume_dist` are ranking functions but produce a fraction in [0, 1],
    # so they are Float64 — the plain-int64 ranking branch below would misreport the schema.
    if fn.func in ("percent_rank", "cume_dist"):
        return pa.float64()
    if fn.func in WINDOW_RANKING or fn.func in ("count", "rle_id"):
        return pa.int64()
    # The EWM statistics and `interpolate` are ratios of weighted sums, so an integer
    # input widens: the value between two integers is generally not one.
    if fn.func == "avg" or fn.func in WINDOW_EWM or fn.func == "interpolate":
        return pa.float64()
    # `bc_runtime::window::agg` builds a `Float64Array` for every one of these regardless of
    # the input's width -- the moments and the median are ratios, and the fold accumulates in
    # a double. They fell through to the `None` below, so `Dataset.schema` could not answer a
    # column the engine's type was never in doubt about.
    if fn.func in ("var", "stddev", "median", "product"):
        return pa.float64()
    if fn.input is None:  # value/min/max/sum all need an input
        return None
    t = infer_type(fn.input, input_schema)
    if t is None:
        return None
    if fn.func == "sum":
        # A **windowed** sum folds in f64 (`bc_runtime::window::agg` accumulates every
        # numeric fold there as a double), so a `decimal` input comes back as a double —
        # unlike the *grouped* `sum`, which keeps the decimal, and unlike DuckDB, whose
        # windowed `SUM` over a DECIMAL is a DECIMAL. `widen` returns the decimal unchanged,
        # so the declared type said `decimal(10,2)` where the engine produced `double`: the
        # one place in the window family where `Dataset.schema` disagreed with a run.
        #
        # Declared as what the engine makes rather than what DuckDB makes, because that is
        # the contract this function has. The representation divergence itself is the same
        # one the math family already carries for decimals and is recorded beside it in
        # `competitor_parity_census.md`; closing it means giving the window fold a decimal
        # accumulator, which is a kernel change rather than a declaration.
        return pa.float64() if pa.types.is_decimal(t) else widen(t)
    if fn.func in WINDOW_VALUE or fn.func in {"min", "max"}:
        return t
    return None


def _validate_window_input_types(source: LogicalPlan, functions: tuple) -> None:
    """Reject a window function over an input type it cannot mean anything over.

    The window twin of `plan.logical.aggregate._validate_agg_input_types`, sharing its rule
    (`plan.types.domains`) because the window form of an aggregate computes the same
    statistic over a frame and so has the same input domain. Runs on the same terms too: at
    build time when the input's types are known, silently when they are not.
    """
    from batcher.plan.types.domains import window_domain_error

    schema = source.available_schema()
    if schema is None:
        return
    for fn in functions:
        if fn.input is None:
            continue
        dt = infer_type(fn.input, schema)
        if dt is None:
            continue
        label = (
            f"column {fn.input.name!r}" if isinstance(fn.input, Col) else f"{fn.alias!r}'s input"
        )
        problem = window_domain_error(fn.func, label, widen(dt))
        if problem is not None:
            raise PlanError(problem)


#: Fixed order for the error message; membership is `FRAME_UNITS`, which is the wire
#: vocabulary and lives with every other IR tag in `plan/ir_tags.py`.
_FRAME_UNIT_ORDER = ("rows", "range", "groups")


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
    - ``"range"`` — the ORDER BY key's own **values**. ``WindowFrame(-300_000_000, 0,
      "range")`` over a microsecond timestamp is "the last five minutes", a window whose
      row count varies with how densely the series was sampled, where a ``"rows"`` frame
      of the same shape is always the same number of rows. A ``0`` bound (or ``None``) is
      the peer/unbounded form, e.g. ``WindowFrame(None, 0, "range")``.

    A value-based ``range`` offset needs exactly one ORDER BY key and a numeric or
    temporal one, because the bound is arithmetic on the key. The engine rejects a key it
    cannot subtract rather than quietly substituting the peer frame.
    """

    start: int | None
    end: int | None
    units: str = "rows"

    def __post_init__(self) -> None:
        if self.units not in FRAME_UNITS:
            raise PlanError(
                f"window frame units must be one of {_FRAME_UNIT_ORDER}, got {self.units!r}"
            )
        if self.start is not None and self.end is not None and self.start > self.end:
            raise PlanError(f"window frame start {self.start} is after end {self.end}")

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
    #: EWM smoothing factor in ``(0, 1]``; set only for `ewm_mean`/`ewm_var`/`ewm_std`.
    alpha: float | None = None
    #: EWM half-life in the ORDER BY key's units (microseconds for a temporal key). Set
    #: instead of `alpha` to decay by elapsed key value; only `ewm_mean` takes it.
    half_life: float | None = None

    def __post_init__(self) -> None:
        if self.func not in WINDOW_FUNCS:
            raise PlanError(
                f"unknown window function {self.func!r}; expected one of {sorted(WINDOW_FUNCS)}"
            )
        if self.func in (WINDOW_AGGREGATES | WINDOW_VALUE | WINDOW_SERIES) and self.input is None:
            raise PlanError(f"window function {self.func!r} requires an input column")
        if self.func in WINDOW_RANKING and self.input is not None:
            raise PlanError(f"window ranking function {self.func!r} takes no input")
        if self.frame is not None and self.func not in WINDOW_FRAMEABLE:
            raise PlanError(f"window function {self.func!r} does not support an explicit frame")
        # An alpha on a non-EWM function would be silently dropped by the engine, and an
        # EWM function without one has no curve to compute — reject both rather than pick
        # a default, which would return a plausible answer to a question nobody asked.
        if self.func in WINDOW_EWM:
            # Exactly one decay: an alpha per row, or a half-life per unit of elapsed key.
            # Both would be two answers to one question, and neither leaves nothing to decay.
            if (self.alpha is None) == (self.half_life is None):
                raise PlanError(
                    f"window function {self.func!r} requires exactly one of a smoothing "
                    f"factor alpha or a half_life, got alpha={self.alpha!r} "
                    f"half_life={self.half_life!r}"
                )
            if self.alpha is not None and not 0.0 < self.alpha <= 1.0:
                raise PlanError(
                    f"window function {self.func!r} requires a smoothing factor alpha in "
                    f"(0, 1], got {self.alpha!r}"
                )
            if self.half_life is not None and self.func != "ewm_mean":
                raise PlanError(
                    f"window function {self.func!r} does not take a half_life; only "
                    "ewm_mean decays by elapsed time"
                )
            if self.half_life is not None and not self.half_life > 0:
                raise PlanError(
                    f"window function {self.func!r} requires a positive half_life, got "
                    f"{self.half_life!r}"
                )
        elif self.alpha is not None or self.half_life is not None:
            raise PlanError(f"window function {self.func!r} does not take a smoothing factor")

    def to_ir(self) -> dict[str, Any]:
        item: dict[str, Any] = {"func": self.func, "alias": self.alias, "offset": self.offset}
        if self.input is not None:
            item["input"] = self.input.to_ir()
        if self.frame is not None:
            item["frame"] = self.frame.to_ir()
        if self.alpha is not None:
            item["alpha"] = self.alpha
        if self.half_life is not None:
            item["half_life"] = self.half_life
        return item


def _key_label(expr: Expr) -> str:
    """How to name a window key in an error: its column, or the expression's rendering."""
    from batcher.plan.expr_ir import Col

    return expr.name if isinstance(expr, Col) else str(expr)


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
        available = available_column_set(self.input)
        for expr in self.partition_keys:
            _validate_refs(expr, available, what="window partition key")
        for key in self.order_keys:
            _validate_refs(key.expr, available, what="window order key")
        # A window partitions and orders through the same row encoder `group_by` keys on,
        # so a `map` in either key list fails the same way and is refused the same way.
        from batcher.plan.logical.base import validate_key_domains

        validate_key_domains(
            self.input,
            [(e, _key_label(e)) for e in self.partition_keys],
            operation="over(partition_by=...)",
        )
        validate_key_domains(
            self.input,
            [(k.expr, _key_label(k.expr)) for k in self.order_keys],
            operation="over(order_by=...)",
        )
        for fn in self.functions:
            if fn.func in WINDOW_RANKING and not self.order_keys:
                # Spark rejects this too (WINDOW_FUNCTION_FRAME_NOT_ORDERED), for the reason
                # that applies here: without an order there is no "first" row, so the answer
                # would depend on arrival order, which a morselized or distributed scan does
                # not fix. DuckDB and Polars accept it because a single-node engine can
                # define arrival order cheaply; an engine whose contract is
                # single-node == distributed cannot.
                #
                # `row_number()` over the whole relation is the one that ports, because the
                # thing a migrant wants from it — a positional column, any order — is a
                # capability Batcher has under another name. Naming it here is what turns a
                # refusal into a fix. The suggestion is withheld when `partition_by` is set:
                # `with_row_index` numbers the relation, not each group, so offering it there
                # would trade a clear refusal for a wrong answer.
                hint = (
                    " — for a plain positional column with no ordering, use ds.with_row_index('n')"
                    if fn.func == "row_number" and not self.partition_keys
                    else ""
                )
                raise PlanError(f"window ranking function {fn.func!r} requires order_by keys{hint}")
            if fn.func in (WINDOW_FILL | WINDOW_SERIES) and not self.order_keys:
                # Without an order there is no "previous" row: the result would depend on
                # arrival order, which a morselized/distributed scan does not fix.
                raise PlanError(
                    f"window function {fn.func!r} requires order_by keys — it carries "
                    "values along a defined row order, and an unordered relation has none"
                )
            if fn.input is not None:
                _validate_refs(fn.input, available, what=f"window function {fn.alias!r}")
        _validate_window_input_types(self.input, self.functions)
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
