# vs PyArrow

This page compares Batcher and PyArrow's compute kernels on the suites where PyArrow can
express the workload.

PyArrow is the substrate rather than a competitor in the usual sense: Batcher's own data
plane speaks Arrow, and on several suites the two read byte-identical buffers. What differs
is everything above the buffer -- scheduling, the query optimizer, and whether an operator
streams or materializes.

## Where a comparison exists at all

PyArrow has no SQL surface and no query planner. It has `Table` and `compute`, so a
benchmark case can be expressed for it only when the case is a single operator or a short
chain of them. The suites split cleanly:

| Suite | Comparable | Why |
|---|---|---|
| operators | yes, 19 of 23 | each case is one operator over one table |
| scan | yes, 27 of 27 | a read plus an aggregate |
| TPC-H, TPC-DS, ClickBench, JSON, H2O | no | multi-table SQL with joins and subqueries |

A blank column on the other suites means PyArrow cannot express the query, not that it lost.
This site marks the two states differently on purpose.

## The measured standing

48-core box, `batcher,pyarrow` pairwise, best of five, every row correctness-gated.

| Suite | b/pyarrow | Cases |
|---|---:|---:|
| operators | 0.04 | 19 |
| scan | 0.07 | 27 |

The margin is wide because the comparison is not really kernel against kernel. PyArrow
executes an operator over a whole table on one core; Batcher morselizes the same work into
16,384-row batches across every core, and on the shapes where PyArrow must materialize an
intermediate (a sort feeding a limit, a window over an ordered partition) it also pays for
memory Batcher never allocates. Where the work is a single vectorized pass over one column,
the two converge -- that is the same Arrow kernel underneath.

## What this does not say

It does not say Batcher's kernels beat Arrow's. On a single-threaded single-column
reduction they are the same order, and several of Batcher's own kernels call into Arrow's.
The result above is a statement about scheduling and planning.

## See also

- {doc}`/benchmarks/methodology`: the correctness gate and the quiet-box rule.
- {doc}`/architecture/index`: how the morsel scheduler sits over the Arrow kernels.
