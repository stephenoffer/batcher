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


def build_fill_null(ds: Dataset, value: Any | dict[str, Any]) -> Dataset:
    """Replace nulls — one fill `value` for every column, or per-column via a dict."""
    cols = ds.columns
    if isinstance(value, dict):
        unknown = set(value) - set(cols)
        if unknown:
            raise PlanError(f"fill_null(): unknown column(s) {sorted(unknown)}")
        return ds.with_columns(**{c: Col(c).fill_null(value[c]) for c in value})
    return ds.with_columns(**{c: Col(c).fill_null(value) for c in cols})


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
