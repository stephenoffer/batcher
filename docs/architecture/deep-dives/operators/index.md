# Parallelism and the operator core

The first two pages give you the shape shared by every operator. The last four are the four
stateful operators themselves, each one an instance of that shape.

- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: why work is cut into 16,384-row chunks.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: `partial → combine → finalize`, and why one core and one cluster run the same code.
- {doc}`Aggregation internals </architecture/deep-dives/operators/aggregation-internals>`: a `group_by().agg()` from the morsel to the output rows, and the decisions made at runtime rather than at plan time.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: the one row-index primitive every join type and strategy is built on.
- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: the only operator whose order is the answer, and why an order-independent test cannot see its bugs.
- {doc}`Window internals </architecture/deep-dives/operators/window-internals>`: a pipeline breaker that must return every input row, in the original order.

```{toctree}
:hidden:

aggregation-internals
join-algorithms
mergeable-algebra
morsel-parallelism
sort-internals
window-internals
```
