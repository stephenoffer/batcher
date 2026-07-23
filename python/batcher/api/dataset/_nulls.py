"""Null handling behind `Dataset.fill_null` / `Dataset.drop_nulls` (the `api` layer).

Three shapes of fill live here, and they lower very differently:

* a **constant** (or per-column dict) → a plain `coalesce` projection;
* a **statistic** (``mean``/``min``/``max``/``zero``) → one whole-relation window
  aggregate broadcast into a coalesce, so it needs no second pass over the data;
* a **carry** (``forward``/``backward``) → a `Window` value function along an explicit
  `order_by`. This one is only meaningful with an order, which is why it demands one:
  a morselized or distributed scan has no inherent row order to fall back on.

Everything here composes out of existing IR (`Window`, `coalesce`, `filter`) — the null
sugar adds no new plan node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

__all__ = ["build_drop_nulls", "build_fill_null", "build_fill_null_strategy"]


def _fillable_columns(ds: Dataset, value: Any, subset: list[str] | None) -> list[str]:
    """The columns a single scalar `value` can legally fill, or a `PlanError` naming why not.

    `coalesce(col, literal)` is typed in the engine: filling a string column with 0
    is not a wide cast, it is a type error, and it used to surface as a raw
    ``arguments need to have the same data type`` from Rust with no column named. So
    the compatible columns are worked out here, up front, where the schema and the
    offending value are both in hand and the message can name them.

    With an explicit `subset` every named column must accept the value (the user said
    which ones they meant). Without one, an all-column fill that matches nothing at
    all is the mistake worth reporting; a fill that matches some columns and skips
    incompatible ones is the pandas/Polars-shaped behaviour users expect from
    ``fillna(0)`` on a mixed frame.
    """
    cols = list(ds.columns) if subset is None else list(subset)
    unknown = set(cols) - set(ds.columns)
    if unknown:
        raise PlanError(f"fill_null(): unknown column(s) {sorted(unknown)}")

    schema = ds._plan.available_schema()
    if schema is None:
        return cols  # types unknown at plan time; let the engine judge
    arrow = schema.arrow
    compatible = [c for c in cols if _accepts_fill(arrow, c, value)]
    if subset is not None and len(compatible) != len(cols):
        rejected = sorted(set(cols) - set(compatible))
        raise PlanError(
            f"fill_null(): column(s) {rejected} cannot be filled with {value!r} "
            f"(incompatible types); pass a per-column mapping, e.g. "
            f"fill_null({{'{rejected[0]}': <a matching value>}})"
        )
    if not compatible:
        raise PlanError(
            f"fill_null(): no column can be filled with {value!r}; the dataset's "
            f"types are {[str(f.type) for f in arrow]}"
        )
    return compatible


def _accepts_fill(arrow: Any, column: str, value: Any) -> bool:
    """Whether `column`'s Arrow type can hold `value` as a null replacement."""
    import pyarrow as pa

    index = arrow.get_field_index(column)
    if index < 0:  # pragma: no cover - guarded by the caller's unknown-column check
        return True
    dtype = arrow.field(index).type
    if isinstance(value, bool):
        return pa.types.is_boolean(dtype)
    if isinstance(value, (int, float)):
        return (
            pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)
        )
    if isinstance(value, str):
        return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
    return True  # an exotic literal: let the engine have the final say


def build_fill_null(
    ds: Dataset, value: Any | dict[str, Any], subset: list[str] | None = None
) -> Dataset:
    """Replace nulls — one fill `value` for every column, or per-column via a dict."""
    if isinstance(value, dict):
        if subset is not None:
            raise PlanError(
                "fill_null(): pass a per-column mapping or a `subset`, not both — "
                "the mapping already names its columns"
            )
        unknown = set(value) - set(ds.columns)
        if unknown:
            raise PlanError(f"fill_null(): unknown column(s) {sorted(unknown)}")
        return ds.with_columns(**{c: Col(c).fill_null(value[c]) for c in value})
    targets = _fillable_columns(ds, value, subset)
    return ds.with_columns(**{c: Col(c).fill_null(value) for c in targets})


# Strategies that lower to a whole-relation window aggregate broadcast into a coalesce.
_FILL_AGG_STRATEGIES = {"mean": "avg", "min": "min", "max": "max"}
# Strategies that carry a value along an explicit row order (a window value function).
_FILL_ORDERED_STRATEGIES = {"forward": "forward_fill", "backward": "backward_fill"}


def build_fill_null_strategy(
    ds: Dataset,
    strategy: str,
    subset: list[str] | None = None,
    order_by: list[str] | None = None,
    partition_by: list[str] | None = None,
) -> Dataset:
    """Replace nulls using a `strategy` rather than a constant.

    ``"zero"`` fills with 0; ``"mean"``/``"min"``/``"max"`` fill with the column's
    whole-relation aggregate (a single window-aggregate pass, distributed-safe).

    ``"forward"``/``"backward"`` carry the nearest non-null value along `order_by`,
    optionally per `partition_by` group — a window value function, so it is equally
    correct on one core and on a cluster. `order_by` is mandatory for them: an
    unordered relation has no "previous row", and a morselized scan would otherwise
    produce an arrival-order result that changes run to run.

    ``"median"`` remains unsupported (it is not a window aggregate).
    """
    cols = subset if subset is not None else ds.columns
    unknown = set(cols) - set(ds.columns)
    if unknown:
        raise PlanError(f"fill_null(): unknown column(s) {sorted(unknown)}")
    if strategy == "zero":
        return ds.with_columns(**{c: Col(c).fill_null(0) for c in cols})
    if strategy in _FILL_ORDERED_STRATEGIES:
        return _build_ordered_fill(ds, strategy, cols, order_by, partition_by)
    if strategy not in _FILL_AGG_STRATEGIES:
        raise PlanError(
            f"fill_null(strategy={strategy!r}) is not supported; use one of "
            "'mean'/'min'/'max'/'zero', 'forward'/'backward' (with `order_by`), "
            "a constant value, or fill from another column. "
            "(median is not a window aggregate.)"
        )
    agg = _FILL_AGG_STRATEGIES[strategy]
    helpers = {f"__fill_{c}": (agg, c) for c in cols}
    filled = ds.window(partition_by=[], order_by=[], functions=helpers)
    filled = filled.with_columns(**{c: Col(c).fill_null(Col(f"__fill_{c}")) for c in cols})
    return filled.drop(*helpers.keys())


def _build_ordered_fill(
    ds: Dataset,
    strategy: str,
    cols: list[str],
    order_by: list[str] | None,
    partition_by: list[str] | None,
) -> Dataset:
    """Forward/backward fill: one `Window` pass carrying values along `order_by`."""
    if not order_by:
        raise PlanError(
            f"fill_null(strategy={strategy!r}) requires `order_by` — a fill carries "
            "values along a defined row order, and a relation has none by itself"
        )
    keys = set(order_by) | set(partition_by or ())
    if unknown := keys - set(ds.columns):
        raise PlanError(f"fill_null(): unknown order_by/partition_by column(s) {sorted(unknown)}")
    # The order and partition keys are the frame of reference, not data to be filled.
    targets = [c for c in cols if c not in keys]
    if not targets:
        return ds
    fn = _FILL_ORDERED_STRATEGIES[strategy]
    helpers = {f"__fill_{c}": (fn, c) for c in targets}
    filled = ds.window(
        partition_by=list(partition_by or ()), order_by=list(order_by), functions=helpers
    )
    # The fill is the identity wherever the column is already non-null, so the helper
    # column *is* the filled column — no coalesce needed.
    filled = filled.with_columns(**{c: Col(f"__fill_{c}") for c in targets})
    return filled.drop(*helpers.keys())


def build_drop_nulls(ds: Dataset, subset: list[str] | None) -> Dataset:
    """Drop rows that are null in any of `subset` (or any column when `subset` is None)."""
    cols = subset if subset is not None else ds.columns
    unknown = set(cols) - set(ds.columns)
    if unknown:
        raise PlanError(f"drop_nulls(): unknown column(s) {sorted(unknown)}")
    if not cols:
        return ds
    predicate = Col(cols[0]).is_not_null()
    for c in cols[1:]:
        predicate = predicate & Col(c).is_not_null()
    return ds.filter(predicate)
