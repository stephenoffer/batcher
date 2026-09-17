# Parallelism and the operator core

These pages cover how Batcher's operators turn one definition into parallel, spilling and distributed execution. The first two describe the shape every operator shares: work cut into morsels, and state that merges. The rest take the stateful operators one at a time, each an instance of that shape, with the sort's behaviour at cluster scale on a page of its own.

- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: why work is cut at 16,384 rows or 1 MiB, whichever comes first.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: `partial → combine → finalize`, and why one core and one cluster run the same code.
- {doc}`Aggregation internals </architecture/deep-dives/operators/aggregation-internals>`: a {py:meth}`group_by().agg() <batcher.Dataset.group_by>` from the morsel to the output rows, and the decisions made at runtime rather than at plan time.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: the one row-index primitive every join type and strategy is built on, and the range joins that avoid a cartesian product.
- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: the only operator whose order is the answer, and why an order-independent test cannot see its bugs.
- {doc}`Sorting at scale </architecture/deep-dives/operators/sort-at-scale>`: which phases of a distributed sort grow with the cluster, and how the sort adapts when the data is ordered, low-cardinality, or skewed.
- {doc}`Window internals </architecture/deep-dives/operators/window-internals>`: a pipeline breaker that must return every input row, in the original order.

```{toctree}
:hidden:

morsel-parallelism
mergeable-algebra
aggregation-internals
join-algorithms
sort-internals
sort-at-scale
window-internals
```
