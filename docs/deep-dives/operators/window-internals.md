# Window internals

A window function computes a value per row from the *other* rows in the same partition. It is a
pipeline breaker (nothing can be emitted until the partition is complete), it needs an ordering
inside each partition, and, unlike a group-by, every input row survives. That last point is
what shapes the implementation: the output columns must land back in **original row order**,
not partition order.

```text
Window { partition_keys, order_keys, functions, rank_limit }
```

Partition by the keys (an empty key list is one partition over all rows), order within each
partition, compute one output column per function, scatter each column back to the row positions
it came from, and append them to the input columns.

```text
   input rows in original order, and every one of them survives
   ┌────┬────┬────┬────┬────┬────┬────┬────┐
   │ r0 │ r1 │ r2 │ r3 │ r4 │ r5 │ r6 │ r7 │
   └────┴────┴────┴────┴────┴────┴────┴────┘
                      │
             hash-partition by the PARTITION BY key
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
   bucket A: r0 r2 r5           bucket B: r1 r3 r4 r6 r7
        │  order by the ORDER BY key       │  order by the ORDER BY key
        │  run the serial kernel           │  run the serial kernel
        ▼                                  ▼
   values for r0 r2 r5          values for r1 r3 r4 r6 r7
        └─────────────┬──────────────┘
                      │  scatter each value back to the row it came from
                      ▼
   ┌────┬────┬────┬────┬────┬────┬────┬────┐
   │ v0 │ v1 │ v2 │ v3 │ v4 │ v5 │ v6 │ v7 │  appended to the input columns
   └────┴────┴────┴────┴────┴────┴────┴────┘
```

:::{important}
A window partition never spans buckets, and the final scatter restores positions, so the
per-row result is bit-identical to the serial kernel. That last step is the one that is easy to
get wrong: unlike a group-by, every input row survives, and the output columns must land back
in **original row order**, not partition order.
:::

## Three families

`crates/bc-runtime/src/window.rs` implements them, and they have genuinely different costs.

| Family | Functions | Needs an ordering | Shape of the work |
|---|---|---|---|
| Ranking | `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile` | yes | a sort of each partition |
| Aggregate | `sum`, `avg`, `min`, `max`, `count` | only with an `ORDER BY` | two very different kernels wear one name (below) |
| Value | `first_value`, `last_value`, `lag`, `lead`, `nth_value`, `forward_fill`, `backward_fill` | yes | *select a row*, so they are type-generic: one pass builds a per-row source-index map and Arrow's `take` does the rest |

The aggregate row is where the subtlety is.

::::{tab-set}
:::{tab-item} sum() OVER (PARTITION BY g)
A *whole-partition* aggregate: one value, broadcast to every row of the partition. No ordering
is needed at all, because the order within a partition cannot affect the result. This takes the
dense-group-id shortcut below, and it is the cheapest window in the engine.
:::

:::{tab-item} sum() OVER (PARTITION BY g ORDER BY t)
A *running* (cumulative) aggregate over the ordered partition, with `RANGE` peer semantics: tied
rows all take the end-of-peer-group value. That is SQL's default frame
(`RANGE UNBOUNDED PRECEDING TO CURRENT ROW`), and getting the peer rule wrong is the classic
window bug.
:::
::::

## The whole-partition shortcut

`window_partition_agg.rs` exists because a no-`ORDER BY` aggregate does not need any of the
window machinery. It computes via **dense group ids**: reduce each group in one linear pass over
the rows, then broadcast its value back by index. That is exactly a group-by aggregate followed
by a scatter, and it skips the per-partition index lists and the scattered gather they force.

It's the cheapest window in the engine, and it's the shape where Batcher's margin over DuckDB is
narrowest, because there is so little work left to remove.

## Explicit `ROWS` frames

`window_frame.rs` handles `ROWS BETWEEN <start> AND <end>`: for each row, aggregate the physical
rows in `[start, end]` of its ordered partition.

Both frame edges are non-decreasing in the row position, since each is `pos + const` clamped to the
partition, so the frame only ever slides right and never rewinds. That makes the frame a FIFO
queue, and the kernel exploits it to run in **one pass** with no frame ever rescanned. A naive
implementation that re-aggregates each frame is O(n·k), and for a wide frame that is the whole cost
of the query.

Which one-pass structure a function uses depends on its arithmetic. `count`, and `sum` over
integers, keep a running accumulator in O(n): drop the leaving row, add the entering one. `sum` and
`avg` over floats can't do that, because subtracting floats is catastrophically unstable, so they
use a `FifoSum`, a two-stack sliding aggregate that never subtracts. `min` and `max` keep a
monotonic deque, O(n) amortized. Only the aggregate functions take a frame at all.

:::{note}
`FrameBound` and `FrameUnits` here are *mirror* types of `bc_ir::FrameBound`/`FrameUnits`.
`bc-runtime` does not depend on `bc-ir`, because the crate DAG points one way and does not bend
for convenience, so the interpreter maps the IR enum onto these exactly as it does for
`WindowFn`. `Rows` counts physical rows while `Range` and `Groups` count peer groups, and `Range`
is reached only for peer bounds such as `CURRENT ROW` and `UNBOUNDED`. A numeric `RANGE` offset is
not supported and falls back upstream.
:::

## Parallelism

`window_parallel.rs`. Hash-partition the rows by the `PARTITION BY` keys into buckets, so every
window partition lands **wholly inside one bucket**, run the serial kernel on each bucket across
rayon cores, and scatter each function's output column back to original row order. Partitioning
only regroups whole partitions across buckets, and the final scatter restores positions, so the
per-row result is bit-identical to the serial kernel.

There is a load-balance guard for the pathological case. When a *single* partition holds most of
the rows (a near-constant `PARTITION BY`), bucketing collapses to one busy bucket and the
partition/gather/scatter plumbing is pure overhead over the serial kernel. So if the largest
bucket holds more than half the rows, the parallel path bails to `window_serial`. A handful of
*balanced* large partitions still parallelizes fine (each rides a core), so the guard fires only
when one bucket dominates.

Below `window_parallel_row_threshold` (2^15 rows) the serial path runs regardless, so a small
window stays sub-millisecond instead of paying for pool fan-out.

## Spilling

`crates/bc-interp/src/window_spill.rs`. Window functions are per-partition independent, and equal
`PARTITION BY` keys hash to the same bucket, so the input can be grace-partitioned by those keys
into disk-backed buckets and the in-memory kernel run one bucket at a time. Each bucket holds
*complete* partitions, so the result is the same multiset as the single-pass kernel, with peak
resident memory bounded to the largest bucket.

This is the same grace algebra and the same `DiskSpillStore` the aggregate spill uses. One
mechanism, a different operator. It requires a non-empty `partition_keys`: a single global
partition cannot be split for a ranking or running aggregate.

## `QUALIFY` fusion

`Window { rank_limit }` is the fused per-partition top-N behind `QUALIFY <rank> <= k`. The
optimizer sets it only when the window has a single ranking function, so the bound applies to the
one appended column. For `row_number` that is the top k per partition; for `rank` / `dense_rank`
it correctly keeps peers tied at the boundary.

Without the fusion, "top 3 products per category" computes the full ranking over every partition
and then throws almost all of it away.

## Using it

```python
import batcher as bt

ds = bt.from_pydict({"dept": ["a", "a", "b"], "sal": [10, 20, 30]})

out = ds.with_columns(
    rk=bt.rank().over(partition_by=["dept"], order_by=["sal"]),
    running=bt.col("sal").sum().over(partition_by=["dept"], order_by=["sal"]),
    total=bt.col("sal").sum().over(partition_by=["dept"]),
).sort("dept", "sal")

print(out.to_pydict())
print(out.explain())
```

```text
{'dept': ['a', 'a', 'b'], 'sal': [10, 20, 30],
 'rk': [1, 2, 1], 'running': [10, 30, 30], 'total': [30, 30, 30]}
```

`running` is cumulative within the ordered partition; `total` is the whole-partition value
broadcast to every row. Same `sum()`, two different kernels, chosen by whether an `order_by` is
present.

:::{dropdown} Why the plan has two `window` nodes
```text
sort                            est≈3 (exact)
  window                        est≈3 (exact)
    window                      est≈3 (exact)
      scan                      est≈3 (exact)
```

The three functions do not share one `(partition, order)` spec, so they cannot share one node.
The no-`ORDER BY` aggregate is planned separately and takes the dense-group-id shortcut.
:::

## Where it stands

All four measured window shapes beat DuckDB on the operator sweep, though by very different
margins. The table below reads as a speedup factor, so 2.6x means Batcher is 2.6 times faster.
The rows run from the widest margin to the narrowest, and PyArrow has no window operator to
compare against.

| Shape | vs DuckDB | vs Polars |
|---|---:|---:|
| running `sum()` | 2.6x | 6.3x |
| `lag()` | 1.9x | 25x |
| `rank()` | 1.4x | 6.7x |
| `sum()` over partition | 1.1x | 1.0x |

The two aggregate shapes bracket the range for a reason. A running `sum()` streams the ordered
partition once and wins comfortably. A whole-partition `sum()` is already so cheap through the
dense-group-id shortcut that there's little left to win, which is why it sits at parity with
Polars. The ranking and value functions land in between, because both need the partition ordered
and pay a per-partition sort plus the scatter back to row order.

These figures come from the operator-mix sweep in `benchmarks/BENCHMARK_RESULTS.md`, measured on
a 16-core release build with every correctness check passing.

## Where the code lives

- `crates/bc-runtime/src/window.rs`: `WindowFn`, the serial kernel, ranking and value functions
- `crates/bc-runtime/src/window_frame.rs`: explicit `ROWS` frames, one-pass accumulator/deque
- `crates/bc-runtime/src/window_partition_agg.rs`: whole-partition aggregates via dense ids
- `crates/bc-runtime/src/window_parallel.rs`: bucket-parallel execution and the skew guard
- `crates/bc-runtime/src/window_fill.rs`: `forward_fill` / `backward_fill`
- `crates/bc-interp/src/window_spill.rs`: grace partitioning for bounded memory

## See also

- {doc}`Architecture </architecture/index>`: pipeline breakers, and why a window is one.
- {doc}`Execution engine </internals/execution>`: where the window kernel is driven from.
- {doc}`Kyber </internals/kyber>`: the pass that sets `rank_limit`.
- {doc}`Window functions </user-guide/analyze/window-functions>`: the API, frames, and `QUALIFY`.
- {doc}`Analytics benchmarks </benchmarks/analytics>`: the four window shapes measured above.
- {doc}`Aggregation internals </deep-dives/operators/aggregation-internals>`: the dense group ids the shortcut reuses.
- {doc}`Sort internals </deep-dives/operators/sort-internals>`: the per-partition ordering the ranking functions pay for.
- {doc}`Spilling </deep-dives/memory/spilling>`: the same grace partitioning, bounding a window.
