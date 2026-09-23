# Expr: aggregates and windows

This page lists the {py:obj}`Expr <batcher.plan.expr_ir.core.Expr>` methods that collapse many rows into one, the window and ranking methods that compute over a frame, and the {py:obj}`AggExpr <batcher.AggExpr>` an aggregate returns.

Every aggregate here is mergeable in the engine, as a `partial` then `combine` then `finalize` triple. That is why the same call is correct on one core, on every core, and across a cluster, and why a windowed form costs no separate implementation.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.core
```

## Aggregation

The per-group reductions used in `group_by().agg(...)` or bound to a window with `.over(...)`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.sum
   Expr.count
   Expr.mean
   Expr.min
   Expr.max
   Expr.median
   Expr.count_distinct
   Expr.approx_count_distinct
   Expr.first
   Expr.last
   Expr.any_value
   Expr.min_by
   Expr.max_by
   Expr.arg_min
   Expr.arg_max
   Expr.array_agg
   Expr.histogram
```

## Statistical, logical, and bitwise aggregates

Aggregates that describe a group's distribution (spread, quantiles, most frequent values, shape, and the assembly-contiguity statistics) or fold its non-null values by AND, OR, XOR, product, or a compensated sum.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.std
   Expr.var
   Expr.quantile
   Expr.quantile_disc
   Expr.approx_quantile
   Expr.approx_median
   Expr.mode
   Expr.mode_top_k
   Expr.top_k
   Expr.skew
   Expr.kurtosis
   Expr.mad
   Expr.entropy
   Expr.n50
   Expr.n90
   Expr.l50
   Expr.aun
   Expr.bool_and
   Expr.bool_or
   Expr.bit_and
   Expr.bit_or
   Expr.bit_xor
   Expr.product
   Expr.kahan_sum
```

## Windows, ranking, and offsets

Bind an expression to a window with `over`, or use the window helpers that rank rows, accumulate up to the current row, look back by `n` rows, or flag duplicates, runs, and peaks.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.over
   Expr.cum_sum
   Expr.cum_count
   Expr.cum_min
   Expr.cum_max
   Expr.cum_prod
   Expr.shift
   Expr.diff
   Expr.pct_change
   Expr.rank
   Expr.rank_pct
   Expr.rle_id
   Expr.is_first_distinct
   Expr.is_last_distinct
   Expr.is_duplicated
   Expr.is_unique
   Expr.peak_max
   Expr.peak_min
```

## Rolling, expanding, and exponentially weighted windows

Moving statistics over a fixed count of rows, a time window, every row so far, or an exponential decay.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.rolling_sum
   Expr.rolling_mean
   Expr.rolling_min
   Expr.rolling_max
   Expr.rolling_count
   Expr.rolling_std
   Expr.rolling_var
   Expr.rolling_sum_by
   Expr.rolling_mean_by
   Expr.rolling_min_by
   Expr.rolling_max_by
   Expr.rolling_count_by
   Expr.expanding_mean
   Expr.expanding_std
   Expr.expanding_var
   Expr.ewm_mean
   Expr.ewm_std
   Expr.ewm_var
   Expr.ewm_mean_by
```

## AggExpr

Call an aggregate or a window function on an `Expr` and you get an `AggExpr` back, which adds `.over(...)` for binding a window.

```{eval-rst}
.. currentmodule:: batcher

.. autoclass:: batcher.AggExpr
   :no-members:
```

## Windows, naming, and casting on an aggregate

Bind an aggregate to a window, name or cast its result, or lower it to its JSON `AggregateItem`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   AggExpr.over
   AggExpr.alias
   AggExpr.cast
   AggExpr.to_ir
```

## Reading an aggregate's own tree

These read the aggregate rather than the data: which expressions it consumes, and what it would be named. {py:obj}`meta <batcher.AggExpr.meta>` is the same introspection namespace {py:obj}`Expr.meta <batcher.plan.expr_ir.core.Expr.meta>` returns.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   AggExpr.meta
   AggExpr.name
   AggExpr.operands
   AggExpr.map_operands
```

## Math on an aggregate result

The scalar math that applies to an aggregate's result, with the same meaning as on `Expr`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   AggExpr.round
   AggExpr.floor
   AggExpr.ceil
   AggExpr.trunc
   AggExpr.abs
   AggExpr.sign
   AggExpr.clip
   AggExpr.sqrt
   AggExpr.cbrt
   AggExpr.square
   AggExpr.exp
   AggExpr.expm1
   AggExpr.ln
   AggExpr.log10
   AggExpr.log2
   AggExpr.log1p
```

## See also

- {doc}`expression-methods`: the same object's per-row methods.
- {doc}`/api/relational/functions`: the free aggregate and window functions, such as {py:obj}`bt.sum <batcher.sum>` and {py:obj}`bt.rank <batcher.rank>`.
- {doc}`/user-guide/analyze/aggregations`: grouping, and what each aggregate does with nulls.
- {doc}`/user-guide/analyze/window-functions`: partitions, ordering, and frame bounds.
- {doc}`/architecture/deep-dives/operators/mergeable-algebra`: why one implementation serves every scale.
