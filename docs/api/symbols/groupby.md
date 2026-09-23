# GroupBy

This page lists the {py:obj}`GroupBy <batcher.GroupBy>` surface: the builder {py:obj}`Dataset.group_by <batcher.Dataset.group_by>` returns, and the aggregates you call on it. Nothing here evaluates either. A `GroupBy` is a partial plan, and it becomes a dataset again at {py:obj}`agg <batcher.GroupBy.agg>` or at one of the shorthand aggregates below.

```{eval-rst}
.. currentmodule:: batcher

.. autoclass:: GroupBy
   :no-members:
```

## Common aggregates

`agg` takes any aggregate expressions, and the rest are shorthands that apply one aggregate to every value column per group.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GroupBy.agg
   GroupBy.len
   GroupBy.count
   GroupBy.count_distinct
   GroupBy.sum
   GroupBy.mean
   GroupBy.min
   GroupBy.max
   GroupBy.first
   GroupBy.last
```

## Statistical and collecting aggregates

These shorthands compute a distribution statistic per group, or collect each group's values into a list.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GroupBy.median
   GroupBy.mode
   GroupBy.quantile
   GroupBy.std
   GroupBy.var
   GroupBy.skew
   GroupBy.kurtosis
   GroupBy.product
   GroupBy.array_agg
```

## Filtering, rows, and functions per group

{py:obj}`having <batcher.GroupBy.having>` is SQL's `HAVING`: it keeps the groups for which a predicate over the aggregates holds, which is the one filter that cannot be written before the grouping. The rest keep the first or last rows of each group, apply a Python function to each group, or report the keys.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GroupBy.having
   GroupBy.head
   GroupBy.tail
   GroupBy.map_groups
   GroupBy.keys
   GroupBy.__iter__
```

## See also

- {doc}`dataset-transforms`: the method that produces a `GroupBy`.
- {doc}`expression-aggregates`: the aggregate expressions `agg` takes.
- {doc}`/user-guide/analyze/aggregations`: grouping semantics, null handling, and multi-key groups.
