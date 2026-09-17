"""Binding ``.over(...)`` onto any expression: the one implementation behind every `over`.

`Expr.over`, `AggExpr.over` and `WindowExpr.over` all lower here, so the three spellings
cannot drift apart in what a partition, an order, or a frame means. The rules are Polars'
``over``, restricted to what the relational `Window` operator can compute:

* an **aggregate** becomes the window aggregate over the partition (``sum(x) OVER (...)``);
* a **window function** is bound to the partition and order: the outer partition adds to
  the one it already has (``rank()`` over ``"g"`` ranks within ``g``), and an outer order,
  when given, replaces its own -- ``col("b").rank().over(order_by="b")`` is the long-standing
  spelling of the same rank, so the outer order stays the one that wins;
* a **composite** expression binds every aggregate and window inside it, so
  ``(col("x") / col("x").sum()).over("g")`` is the share within ``g``;
* a **row-level** expression with no aggregate or window is returned unchanged, because a
  per-row value is the same computed per group (Polars answers it identically).

`mapping_strategy="group_to_rows"` is the only strategy the `Window` operator implements;
``"join"`` and ``"explode"`` are refused by name rather than approximated.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import AggExpr, Expr, WindowExpr
from batcher.plan.expr_ir.core import FrameSpec, normalize_key_list
from batcher.plan.expr_ir.nodes import Case, Col, NullIf
from batcher.plan.expr_rewrite.traverse import transform_expr_up

__all__ = ["MAPPING_STRATEGIES", "bind_over"]

#: The `mapping_strategy` values `over` understands; only the first is computed.
MAPPING_STRATEGIES = ("group_to_rows", "join", "explode")

#: The aggregates that pick one row by an order key (`first`/`last`, `min_by`/`max_by`).
_ORDERED_PICKS = frozenset({"arg_min", "arg_max", "arg_min_null", "arg_max_null"})


def _order_keys(
    order_by: Iterable[Any] | Any, descending: bool | Iterable[bool], nulls_last: bool
) -> list[Any]:
    """Normalize `order_by` into window keys carrying their direction and null placement.

    A key is kept as given when it is ascending with nulls last (the SQL default), so a
    plain ``over(order_by="t")`` builds exactly the key it always did; otherwise it becomes
    ``(key, descending, nulls_first)``.
    """
    keys = normalize_key_list(order_by)
    flags = [descending] * len(keys) if isinstance(descending, bool) else list(descending)
    if len(flags) != len(keys):
        raise PlanError(
            f"over(): descending has {len(flags)} flag(s) for {len(keys)} order_by key(s)"
        )
    out: list[Any] = []
    for key, desc in zip(keys, flags, strict=True):
        if isinstance(key, tuple) or (not desc and nulls_last):
            out.append(key)
        else:
            out.append((key, bool(desc), not nulls_last))
    return out


def _as_key(key: Any) -> Expr:
    """A key expression: a string names a column."""
    return Col(key) if isinstance(key, str) else key


def _first_last_window(
    agg: AggExpr, partition: list[Any], order: list[Any], frame: FrameSpec | None
) -> WindowExpr:
    """``first``/``last`` (and ``min_by``/``max_by``) as a window, with the aggregate's rules.

    The grouped form (`arg_min`/`arg_max`) skips a row whose value *or* order key is null,
    and breaks a tie on the key toward the smaller value. `first_value` over an order that
    sorts the skipped rows last, then the key, then the value, picks the same row; the value
    is masked to null where the key is null so a partition of only skipped rows is null.

    The `_null` forms (``ignore_nulls=False``) skip only a row whose *key* is null, so a null
    value can win: the skipped rows sort last by the key alone, and a tie on the key still
    breaks toward the smaller value, nulls after.
    """
    if agg.input2 is not None and not order:
        order = [agg.input2]
    if not order:
        name = "first" if agg.func.startswith("arg_min") else "last"
        raise PlanError(_missing_order_message(name))
    if len(order) != 1:
        raise PlanError("first()/last() over a window take exactly one order_by key")
    key = order[0]
    by, key_desc = (key[0], bool(key[1])) if isinstance(key, tuple) else (key, False)
    by = _as_key(by)
    descending = key_desc != agg.func.startswith("arg_max")
    value = Case([(by.is_not_null(), agg.input)], NullIf(agg.input, agg.input))
    skipped = by.is_null() if agg.func.endswith("_null") else value.is_null()
    keys = [(skipped, False, False), (by, descending, False), (value, False, False)]
    return WindowExpr("first_value", value, partition, keys, frame)


def _missing_order_message(func: str) -> str:
    """The one refusal every order-dependent expression gives when it has no order."""
    from batcher.plan.logical.window import missing_order_message

    return missing_order_message(func)


def _bind_agg(
    agg: AggExpr, partition: list[Any], order: list[Any], frame: FrameSpec | None
) -> WindowExpr:
    if agg.func in _ORDERED_PICKS and agg.input is not None:
        return _first_last_window(agg, partition, order, frame)
    if agg.input2 is not None:
        raise PlanError(
            f"the two-input aggregate {agg.func!r} has no window form; compute it with "
            "group_by(...).agg(...) and join the result back"
        )
    if agg.order_by:
        raise PlanError(
            "array_agg(order_by=...) has no window form; compute it with "
            "group_by(...).agg(...) and join the result back"
        )
    # `mean` is the DataFrame spelling; the window engine names the aggregate `avg`.
    func = "avg" if agg.func == "mean" else agg.func
    return WindowExpr(func, agg.input, partition, order, frame)


def _bind_window(
    we: WindowExpr, partition: list[Any], order: list[Any], frame: FrameSpec | None
) -> WindowExpr:
    """Merge the outer keys into a window's own: partitions add up, an outer order wins."""
    inner = we.input if we.input is None else _bind_tree(we.input, partition, order, frame)
    return WindowExpr(
        we.func,
        inner,
        [*partition, *we.partition_by],
        order or we.order_by,
        frame if frame is not None else we.frame,
        we.offset,
        we.alpha,
        we.half_life,
        we.ignore_nulls,
    )


def _bind_tree(expr: Expr, partition: list[Any], order: list[Any], frame: Any) -> Expr:
    def rule(node: Any) -> Any:
        if isinstance(node, AggExpr):
            return _bind_agg(node, partition, order, frame)
        if isinstance(node, WindowExpr):
            return _bind_window(node, partition, order, frame)
        return node

    return transform_expr_up(expr, rule)


def bind_over(
    expr: Expr | AggExpr,
    partition_by: Iterable[Any] | Any = (),
    order_by: Iterable[Any] | Any = (),
    frame: FrameSpec | None = None,
    *,
    descending: bool | Iterable[bool] = False,
    nulls_last: bool = True,
    mapping_strategy: str = "group_to_rows",
) -> Expr:
    """Bind `expr` to a window partition and order (see the module docstring).

    Args:
        expr: The expression to evaluate over the window.
        partition_by: Key expressions (or column names) the window is computed within.
        order_by: Key expressions (or column names) giving the row order.
        frame: An explicit frame for the aggregates the expression holds.
        descending: Order every key, or each key, from largest to smallest.
        nulls_last: Place nulls after the non-null keys (the SQL default).
        mapping_strategy: How a per-group result maps back to rows; only
            ``"group_to_rows"`` is supported.

    Returns:
        The expression with every aggregate and window it holds bound to the window.

    Raises:
        PlanError: For an unsupported `mapping_strategy` or a malformed key list.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_rewrite.over import bind_over
            >>> w = bind_over(bt.col("x").sum(), partition_by="g")
            >>> w.func, [k.name for k in w.partition_by if hasattr(k, "name")] or w.partition_by
            ('sum', ['g'])
    """
    if mapping_strategy not in MAPPING_STRATEGIES:
        raise PlanError(
            f"over(): mapping_strategy must be one of {MAPPING_STRATEGIES}, got "
            f"{mapping_strategy!r}"
        )
    if mapping_strategy != "group_to_rows":
        raise PlanError(
            f"over(mapping_strategy={mapping_strategy!r}) is not supported: the window "
            "operator maps each group's result back to its rows ('group_to_rows'). Collect "
            "per-group lists with group_by(...).agg(col(...).array_agg()) and join them back."
        )
    partition = normalize_key_list(partition_by)
    order = _order_keys(order_by, descending, nulls_last)
    if isinstance(expr, AggExpr):
        return _bind_agg(expr, partition, order, frame)
    if isinstance(expr, WindowExpr):
        return _bind_window(expr, partition, order, frame)
    return _bind_tree(expr, partition, order, frame)
