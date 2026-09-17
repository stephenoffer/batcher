# Window internals

A window function computes a value per row from the *other* rows in the same partition. It is a
pipeline breaker (nothing can be emitted until the partition is complete), it needs an ordering
inside each partition, and, unlike a group-by, every input row survives. That last point is
what shapes the implementation: the output columns must land back in **original row order**,
not partition order.

```text
Window { partition_keys, order_keys, functions, rank_limit }
```

Partition by the keys, ordering within each partition, then compute one output column per
function, scatter each column back to the row positions it came from, and append them to the
input columns. An empty key list is one partition over all rows.

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
per-row result is bit-identical to the serial kernel. The scatter is the step that is easy to
get wrong, because a result in partition order looks plausible and is not the answer.
:::

## Four families

`crates/bc-runtime/src/window/mod.rs` defines `WindowFn`, and its variants fall into four
families with genuinely different costs.

| Family | Functions | Needs an ordering | Shape of the work |
|---|---|---|---|
| Ranking | `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile` | yes | a sort of each partition |
| Aggregate | `sum`, `avg`, `min`, `max`, `count`, plus `var`, `stddev`, `product`, `bool_and`, `bool_or`, `bit_and`, `bit_or`, `bit_xor`, `count_distinct` and `median` | only with an `ORDER BY` | two very different kernels wear one name (below) |
| Value | `first_value`, `last_value`, `lag`, `lead`, `nth_value`, `forward_fill`, `backward_fill` | yes | *select a row*, so they are type-generic: one pass builds a per-row source-index map and Arrow's `take` does the rest |
| Series | `ewm_mean`, `ewm_var`, `ewm_std`, `interpolate`, `rle_id` | yes | one sequential recurrence over the ordered partition, so no frame applies |

The ten aggregates past the first five (`window/agg/`) are the ones whose running form costs
O(1) per row, apart from `median`, which keeps a two-heap at `O(log n)` per row. `var` and
`stddev` carry the same Welford state a `GROUP BY` does, and a whole-partition `median` is
the `GROUP BY` median's quickselect, so a window and a group-by over the same rows agree by
construction.

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

`window/partition_agg.rs` exists because a no-`ORDER BY` aggregate does not need any of the
window machinery. It computes via **dense group ids**: reduce each group in one linear pass over
the rows, then broadcast its value back by index. That is exactly a group-by aggregate followed
by a scatter, and it skips the per-partition index lists and the scattered gather they force.

It's the cheapest window in the engine. It is also the shape where Batcher's margin over DuckDB
is narrowest, because there is so little work left to remove.

## Explicit `ROWS` frames

`window/frame/` handles `ROWS BETWEEN <start> AND <end>`: for each row, aggregate the physical
rows in `[start, end]` of its ordered partition.

Both frame edges are non-decreasing in the row position, since each is `pos + const` clamped to
the partition. The frame only ever slides right. That makes it a FIFO queue, and the kernel
exploits the fact to run in **one pass** with no frame ever rescanned. A naive
implementation that re-aggregates each frame is O(n·k), and for a wide frame that is the whole cost
of the query.

Which one-pass structure a function uses depends on its arithmetic. `count`, and `sum` over
integers, keep a running accumulator in O(n): drop the leaving row, add the entering one. `sum` and
`avg` over floats can't do that, because subtracting floats is catastrophically unstable, so they
use a `FifoSum`, a two-stack sliding aggregate that never subtracts. `min` and `max` keep a
monotonic deque, O(n) amortized. Only the aggregate functions and the positional value
functions (`first_value`, `last_value`, `nth_value`) take a frame.

:::{note}
`FrameBound` and `FrameUnits` here are *mirror* types of `bc_ir::FrameBound`/`FrameUnits`.
`bc-runtime` does not depend on `bc-ir`, because the crate DAG points one way and does not bend
for convenience, so the interpreter maps the IR enum onto these exactly as it does for
`WindowFn`. `Rows` counts physical rows while `Range` and `Groups` count peer groups, and `Range`
is reached for peer bounds such as `CURRENT ROW` and `UNBOUNDED`, and for a numeric `RANGE`
offset too: `frame_bounds` dispatches `is_value_range()` to `value_range_bounds`, a binary search
over the order key's values rather than a walk over peers.
:::

The figure below sets one tied row's `ROWS` frame beside its `RANGE` frame, which is the point
where the two units stop agreeing.

![What a tie in the ORDER BY key does to a frame. The rows are sorted once by the PARTITION BY keys and then the ORDER BY keys, so every partition comes out contiguous and each function's output column is scattered back to the row it came from in the original row order. Inside one ordered partition holding three rows tied on the order key, ROWS ... CURRENT ROW stops at the current row, which is pure position arithmetic, so each tied row gets a different answer, while RANGE ... CURRENT ROW runs to the end of the peer group, so all three tied rows get the same answer. GROUPS counts peer groups the way ROWS counts rows, a numeric RANGE offset is a binary search over the key's values rather than a walk over peers, and a null order key frames only its own null peer group. Both frame edges only ever slide right, so a frame is a FIFO queue and no frame is ever rescanned.](/_static/diagrams/window_frame_eval.svg)

## Parallelism

`window/parallel.rs`. Hash-partition the rows by the `PARTITION BY` keys into buckets, so every
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

Bucketing needs partition keys, so a *global* window (no `PARTITION BY`) and a skewed one that
bailed both land on a single partition. `window/running_par.rs` keeps those off one core for a
running aggregate: it cuts the ordered partition into chunks on peer-group boundaries, folds
each chunk, scans the chunk totals, and re-walks each chunk seeded with its prefix. That
re-associates the fold, which is exact for integer `+`, `min`, `max` and counting and not for
floating-point `+`, so float `sum` and `avg` keep the serial walk and the result stays
bit-identical to it.

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

The fusion removes the separate filter node and stops the full windowed batch from reaching the
operator above it. It also **bounds the work**, which it did not always do: `rank_limit` used to
be a mask applied *after* the ranking, so "top 3 products per category" ordered every partition
and then discarded almost all of it.

`rank_limit` is threaded down to `window_serial`, and when the only function is `row_number`
with a single numeric order key, `bc_runtime::window::topk` selects each partition's best `k` with
a bounded max-heap instead of ordering it. That is `O(n log k)` against `O(n log n)`, and for the
usual `k` of one to ten the `log k` is two or three comparisons. Spark and Daft build operators for
exactly this (`WindowGroupLimitExec`, `window_partition_and_dynamic_frame`).

**Where the bound is applied is the whole difference.** A first version hooked the selection above
`window_with`, over the whole batch, and was **2 to 4x slower**. It traded the operator's
bucketed parallelism for the better complexity, running `O(n log k)` on one core against
`O(n log n)` on ninety-six. Applied inside the per-bucket kernel it inherits that parallelism
instead, and each worker heaps only the partitions it owns.

Measured on 6M rows, interleaved best-of-three, `QUALIFY row_number() <= k`
(`docs/architecture/internals/competitor_technique_review.md`):

| Partitions | `k` | Ordering | Bounded | |
|---|---|---|---|---|
| 100 x 60,000 rows | 10 | 63.2 ms | 45.1 ms | **1.40x** |
| 100 x 60,000 rows | 3 | 60.8 ms | 45.2 ms | **1.35x** |
| 2,000 x 3,000 rows | 10 | 58.0 ms | 44.4 ms | **1.31x** |
| 50,000 x 120 rows | 3 | 65.6 ms | 52.5 ms | **1.25x** |
| 1,000,000 x 6 rows | 2 | 116.7 ms | 98.4 ms | **1.19x** |
| `rank()` instead of `row_number` (declines) | 3 | 101.5 ms | 97.4 ms | 1.04x |

The win grows with partition size, which is what the complexity predicts. `k = 1` is unchanged
because Kyber sends it down a different route entirely: `row_number() = 1` rewrites onto
`DISTINCT ON`, a per-key argmin rather than any kind of sort.

The parallel window also stopped doing work for rows it throws away. It used to rank every row, scatter all of those ranks back into input order, and then mask all but `k` per partition. On H2O `groupby` q8's shape (10M rows, 100,000 partitions, `k = 2`), `perf` put 21% of the query in that scatter, spent keeping 200,000 rows. `bc_runtime::window::window_with_rank_limit` runs the same buckets but keeps only each bucket's survivors, named by input row, and orders them once. The interpreter's window then gathers just those rows. The rows, their order and their values are the mask's by construction, and the Rust test `rank_limited_equals_masking_the_full_window` holds the two forms equal across `row_number`, `rank` and `dense_rank`, with ties, null keys, every `k`, and both the parallel and serial paths.

Measured on a 48-core box under load, best of five across four alternating rounds, so the cells are ranges (`benchmarks/BENCHMARK_RESULTS.md`):

| Query | Mask every row | Keep survivors |
|---|---|---|
| `row_number() <= 2` over 100,000 partitions (H2O q8's shape) | 121 to 252 ms | **63 to 68 ms** |
| `row_number() <= 2` over 100 partitions | 126 to 143 ms | **50 to 88 ms** |
| `row_number() <= 10` over 100,000 partitions | 142 to 182 ms | 84 to 158 ms |
| `rank() <= 3` per `l_suppkey`, TPC-H sf1 `lineitem` | 306 to 657 ms | 253 to 444 ms |

The bounded path declines to the ordering path on anything it does not cover: more than one
order key, a non-numeric or nullable one, more than one partition key, or a `groups x k` heap
large next to the rows it selects from. A non-survivor is marked `k + 1` rather than null or
zero, because each bucket keeps the rows whose rank is `<= k` and a zero would pass that test.

Keep it in proportion: on this shape Batcher was already 10-20x faster than DuckDB and 2.5-10x
faster than Polars, so this makes a win larger rather than closing a gap.

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

:::{dropdown} Why the plan has three `window` nodes
```text
query plan (planned)                      7 operators
─────────────────────────────────────────────────────
OPERATOR                            ESTIMATE  NOTES
sort  [dept, sal]                      est≈3  (exact)
└─ project                             est≈3  (exact)
   └─ window  [global]                 est≈3  (exact)
      └─ project                       est≈3  (exact)
         └─ window  [global]           est≈3  (exact)
            └─ window  [global]        est≈3  (exact)
               └─ scan  [source 0]     est≈3  (exact)
```

One node per `.over(...)`, stacked. Nothing merges two windows that happen to share a
`(partition, order)` spec, which `rk` and `running` here do, so the merge is available work
rather than something the plan already did. The no-`ORDER BY` `total` could not have joined
them anyway: it takes the dense-group-id shortcut. The `project` nodes carry each appended
column forward to the next window.
:::

## Where it stands

All four measured window shapes beat DuckDB on the operator sweep, though by very different
margins. The table below reads as a speedup factor, so 2.6x means Batcher is 2.6 times faster.
The rows run from the widest margin to the narrowest, and PyArrow has no window operator to
compare against.

| Shape | vs DuckDB | vs Polars |
|---|---:|---:|
| running `sum()` | 2.6x | 6.3x |
| {py:func}`lag() <batcher.lag>` | 1.9x | 25x |
| `rank()` | 1.4x | 6.7x |
| `sum()` over partition | 1.1x | 1.0x |

The two aggregate shapes bracket the range for a reason. A running `sum()` streams the ordered
partition once and wins comfortably. A whole-partition `sum()` is already so cheap through the
dense-group-id shortcut that there's little left to win, which is why it sits at parity with
Polars. The ranking and value functions land in between, because both need the partition ordered
and pay a per-partition sort plus the scatter back to row order.

These figures come from an operator-mix sweep recorded in `benchmarks/BENCHMARK_RESULTS.md`,
measured on a 16-core release build with every correctness check passing. The
{doc}`analytics benchmarks </benchmarks/results/analytics>` page carries the last full published
sweep, with narrower margins against DuckDB on the running `sum()` (0.71x) and the
whole-partition `sum()` (0.93x), so read the table for the ordering of the shapes rather than as
a current ratio.

## Where the code lives

- `crates/bc-runtime/src/window/mod.rs`: `WindowFn`, the serial kernel, ranking and value functions
- `crates/bc-runtime/src/window/frame/`: explicit `ROWS` frames, one-pass accumulator/deque
- `crates/bc-runtime/src/window/partition_agg.rs`: whole-partition aggregates via dense ids
- `crates/bc-runtime/src/window/parallel.rs`: bucket-parallel execution and the skew guard
- `crates/bc-runtime/src/window/running_par.rs`: the parallel prefix scan for a single large partition
- `crates/bc-runtime/src/window/agg/`, `series.rs`: the extra aggregates, EWM, `interpolate` and `rle_id`
- `crates/bc-runtime/src/window/topk.rs`: the bounded per-partition top-k behind `QUALIFY`
- `crates/bc-runtime/src/window/fill.rs`: {py:meth}`forward_fill <batcher.plan.expr_ir.core.Expr.forward_fill>` / {py:meth}`backward_fill <batcher.plan.expr_ir.core.Expr.backward_fill>`
- `crates/bc-interp/src/window_spill.rs`: grace partitioning for bounded memory

## See also

- {doc}`Architecture </architecture/index>`: pipeline breakers, and why a window is one.
- {doc}`Execution engine </architecture/internals/execution>`: where the window kernel is driven from.
- {doc}`Kyber </architecture/internals/kyber>`: the pass that sets `rank_limit`.
- {doc}`Window functions </user-guide/analyze/window-functions>`: the API, frames, and `QUALIFY`.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: the four window shapes measured above.
- {doc}`Aggregation internals </architecture/deep-dives/operators/aggregation-internals>`: the dense group ids the shortcut reuses.
- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: the per-partition ordering the ranking functions pay for.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: the same grace partitioning, bounding a window.
