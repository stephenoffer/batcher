# Window internals

A window function computes a value per row from the other rows in the same partition. It's a pipeline breaker, it needs an ordering inside each partition, and unlike a group-by every input row survives. That last point shapes the implementation: the output columns must land back in **original row order**, not partition order. This page covers the four function families, frames, parallelism, spilling, and the fused top-N behind `QUALIFY`.

```python
import batcher as bt

ds = bt.from_pydict({"dept": ["a", "a", "b"], "sal": [10, 20, 30]})
out = ds.with_columns(
    rk=bt.rank().over(partition_by=["dept"], order_by=["sal"]),
    running=bt.col("sal").sum().over(partition_by=["dept"], order_by=["sal"]),
    total=bt.col("sal").sum().over(partition_by=["dept"]),
).sort("dept", "sal")
print(out.to_pydict())
# {'dept': ['a', 'a', 'b'], 'sal': [10, 20, 30], 'rk': [1, 2, 1], 'running': [10, 30, 30], 'total': [30, 30, 30]}
```

`running` is cumulative within the ordered partition, and `total` is the whole-partition value broadcast to every row: the same `sum()`, two different kernels, chosen by whether an `order_by` is present.

```text
   input rows in original order, and every one of them survives
   ┌────┬────┬────┬────┬────┬────┬────┬────┐
   │ r0 │ r1 │ r2 │ r3 │ r4 │ r5 │ r6 │ r7 │
   └────┴────┴────┴────┴────┴────┴────┴────┘
             hash-partition by the PARTITION BY key
        ┌─────────────┴──────────────┐
   bucket A: r0 r2 r5           bucket B: r1 r3 r4 r6 r7
        │  order, run the serial kernel    │  order, run the serial kernel
   values for r0 r2 r5          values for r1 r3 r4 r6 r7
        └─────────────┬──────────────┘
                      │  scatter each value back to the row it came from
   ┌────┬────┬────┬────┬────┬────┬────┬────┐
   │ v0 │ v1 │ v2 │ v3 │ v4 │ v5 │ v6 │ v7 │  appended to the input columns
   └────┴────┴────┴────┴────┴────┴────┴────┘
```

The plan node is `Window { partition_keys, order_keys, functions, rank_limit }`. A partition never spans buckets, and the final scatter restores positions, so the per-row result is bit-identical to the serial kernel.

## Four families

[`crates/bc-runtime/src/window/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/window/mod.rs) defines `WindowFn`, whose variants fall into four families with different costs:

| Family | Functions | Needs an ordering | Shape of the work |
|---|---|---|---|
| Ranking | `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile` | yes | a sort of each partition |
| Aggregate | `sum`, `avg`, `min`, `max`, `count`, plus `var`, `stddev`, `product`, `bool_and`, `bool_or`, `bit_and`, `bit_or`, `bit_xor`, `count_distinct` and `median` | only with an `ORDER BY` | a whole-partition or a running kernel |
| Value | `first_value`, `last_value`, `lag`, `lead`, `nth_value`, `forward_fill`, `backward_fill` | yes | select a row: a per-row source-index map, then Arrow's `take` |
| Series | `ewm_mean`, `ewm_var`, `ewm_std`, `interpolate`, `rle_id` | yes | one sequential recurrence over the ordered partition |

The extra aggregates (`window/agg/`) cost O(1) per row in their running form, apart from `median`, which keeps a two-heap at `O(log n)`. `var` and `stddev` carry the same Welford state a `GROUP BY` does, so a window and a group-by over the same rows agree by construction.

A **whole-partition** aggregate (no `ORDER BY`) takes the shortcut in `window/partition_agg.rs`: reduce each group through dense group ids in one pass, then broadcast its value back by index, exactly a group-by followed by a scatter. A **running** aggregate (with `ORDER BY`) uses SQL's default frame, `RANGE UNBOUNDED PRECEDING TO CURRENT ROW`, so tied rows all take the end-of-peer-group value.

## Explicit frames

`window/frame/` handles `ROWS BETWEEN <start> AND <end>`. Both frame edges are non-decreasing in the row position, so the frame only slides right and behaves as a FIFO queue, and the kernel runs in one pass with no frame rescanned:

```python
r = bt.from_pydict({"t": [1, 2, 3, 4, 5], "v": [1.0, 4.0, 2.0, 8.0, 5.0]})
print(r.with_columns(s3=bt.col("v").rolling_sum(3).over(order_by=["t"])).to_pydict()["s3"])
# [1.0, 5.0, 7.0, 14.0, 15.0]
```

`count` and integer `sum` keep a running accumulator. Float `sum` and `avg` use a `FifoSum`, a two-stack sliding aggregate that never subtracts, because float subtraction is unstable. `min` and `max` keep a monotonic deque, O(n) amortized.

![What a tie in the ORDER BY key does to a frame. The rows are sorted once by the PARTITION BY keys and then the ORDER BY keys, so every partition comes out contiguous and each function's output column is scattered back to the row it came from in the original row order. Inside one ordered partition holding three rows tied on the order key, ROWS ... CURRENT ROW stops at the current row, which is pure position arithmetic, so each tied row gets a different answer, while RANGE ... CURRENT ROW runs to the end of the peer group, so all three tied rows get the same answer. GROUPS counts peer groups the way ROWS counts rows, a numeric RANGE offset is a binary search over the key's values rather than a walk over peers, and a null order key frames only its own null peer group. Both frame edges only ever slide right, so a frame is a FIFO queue and no frame is ever rescanned.](/_static/diagrams/window_frame_eval.svg)

:::{dropdown} Frame types and the crate DAG
`FrameBound` and `FrameUnits` in `bc-runtime` mirror `bc_ir::FrameBound`/`FrameUnits`, because `bc-runtime` doesn't depend on `bc-ir`. The interpreter maps the IR enum onto them, as it does for `WindowFn`. `Rows` counts physical rows while `Range` and `Groups` count peer groups. A numeric `RANGE` offset dispatches through `is_value_range()` to `value_range_bounds`, a binary search over the order key's values.
:::

## Parallelism

`window/parallel.rs` hash-partitions rows by the `PARTITION BY` keys so every window partition lands wholly inside one bucket, runs the serial kernel per bucket across rayon cores, and scatters each output column back to original row order.

- Below `window_parallel_row_threshold` (2^15 rows) the serial path runs, so a small window stays sub-millisecond.
- When the largest bucket holds more than half the rows, as with a near-constant `PARTITION BY`, the parallel path hands off to `window_serial`. Balanced large partitions still each ride a core.
- A global window (no `PARTITION BY`) running an exact aggregate uses `window/running_par.rs`: it cuts the ordered partition on peer-group boundaries, folds each chunk, scans the chunk totals, and re-walks each chunk seeded with its prefix. That reassociation is exact for integer `+`, `min`, `max` and counting, so float `sum` and `avg` keep the serial walk.

## Spilling

[`crates/bc-interp/src/window_spill.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/window_spill.rs) grace-partitions the input by the `PARTITION BY` keys into disk-backed buckets and runs the in-memory kernel one bucket at a time. Each bucket holds complete partitions, so peak memory is bounded by the largest bucket and the result is unchanged. It's the same `DiskSpillStore` the aggregate uses, and it requires a non-empty `partition_keys`.

```python
from batcher.config import Config, MemoryConfig, config_context

w = bt.from_pydict({"g": [i % 4 for i in range(1000)], "t": list(range(1000))})
q = w.with_columns(rn=bt.row_number().over(partition_by=["g"], order_by=["t"])).sort("t")
with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=1))):
    print(q.to_pydict()["rn"][:6])  # same answer under a one-byte budget
# [1, 1, 1, 1, 2, 2]
```

## `QUALIFY` fusion

A filter on a ranking column, `QUALIFY <rank> <= k`, is fused into the window as `rank_limit`. The optimizer sets it only when the window has a single ranking function, and for `rank` / `dense_rank` it keeps peers tied at the boundary. The plan shows no separate filter node:

```python
sales = bt.from_pydict({"cat": ["a", "a", "a", "b", "b"], "amt": [5, 9, 7, 1, 3]})
rn = bt.row_number().over(partition_by=["cat"], order_by=["amt"], descending=True)
top2 = sales.with_columns(rn=rn).filter(bt.col("rn") <= 2).sort("cat", "rn")
print(top2.to_pydict())
# {'cat': ['a', 'a', 'b', 'b'], 'amt': [9, 7, 3, 1], 'rn': [1, 2, 1, 2]}
print(top2.explain())
# sort  [cat, rn]                        est≈4  (default)
# └─ window  [over cat · row_number]     est≈4  (default)
#    └─ scan  [source 0]                 est≈5  (exact)
```

When the only function is `row_number` with a single numeric order key, `bc_runtime::window::topk` selects each partition's best `k` with a bounded max-heap instead of ordering it: `O(n log k)` against `O(n log n)`. It runs inside the per-bucket kernel, so it keeps the bucket parallelism. `window_with_rank_limit` keeps only each bucket's survivors rather than scattering every rank back, and `k = 1` rewrites onto `DISTINCT ON`, a per-key argmin. The Rust test `rank_limited_equals_masking_the_full_window` holds the fused form equal to masking the full window across `row_number`, `rank` and `dense_rank`, with ties, null keys, every `k`, and both execution paths.

:::{dropdown} Measured effect of the bounded top-k
6M rows, interleaved best-of-three, `QUALIFY row_number() <= k` (`docs/architecture/internals/competitor_technique_review.md`):

| Partitions | `k` | Ordering | Bounded | |
|---|---|---|---|---|
| 100 x 60,000 rows | 10 | 63.2 ms | 45.1 ms | **1.40x** |
| 100 x 60,000 rows | 3 | 60.8 ms | 45.2 ms | **1.35x** |
| 2,000 x 3,000 rows | 10 | 58.0 ms | 44.4 ms | **1.31x** |
| 50,000 x 120 rows | 3 | 65.6 ms | 52.5 ms | **1.25x** |
| 1,000,000 x 6 rows | 2 | 116.7 ms | 98.4 ms | **1.19x** |

The win grows with partition size, as the complexity predicts. The bounded path hands off to the ordering path on more than one order key, a non-numeric or nullable one, more than one partition key, or a `groups x k` heap large next to the rows.
:::

## Where it stands

All four measured window shapes beat DuckDB on the operator sweep. The table reads as a speedup factor, so 2.6x means Batcher is 2.6 times faster:

| Shape | vs DuckDB | vs Polars |
|---|---:|---:|
| running `sum()` | 2.6x | 6.3x |
| {py:func}`lag() <batcher.lag>` | 1.9x | 25x |
| `rank()` | 1.4x | 6.7x |
| `sum()` over partition | 1.1x | 1.0x |

These come from an operator-mix sweep in [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md) on a 16-core release build with every correctness check passing. The {doc}`analytics benchmarks </benchmarks/results/analytics>` page carries the last full published sweep, which records the running `sum()` at 0.71x and the whole-partition `sum()` at 0.93x of DuckDB's time.

## Where the code lives

- [`crates/bc-runtime/src/window/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/window/mod.rs): `WindowFn`, the serial kernel, ranking and value functions
- [`crates/bc-runtime/src/window/frame/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-runtime/src/window/frame): explicit `ROWS` frames, one-pass accumulator/deque
- [`crates/bc-runtime/src/window/partition_agg.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/window/partition_agg.rs): whole-partition aggregates via dense ids
- [`crates/bc-runtime/src/window/parallel.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/window/parallel.rs): bucket-parallel execution and the skew guard
- [`crates/bc-runtime/src/window/running_par.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/window/running_par.rs): the parallel prefix scan for a single large partition
- [`crates/bc-runtime/src/window/agg/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-runtime/src/window/agg), `series.rs`: the extra aggregates, EWM, `interpolate` and `rle_id`
- [`crates/bc-runtime/src/window/topk.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/window/topk.rs): the bounded per-partition top-k behind `QUALIFY`
- [`crates/bc-runtime/src/window/fill.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/window/fill.rs): {py:meth}`forward_fill <batcher.plan.expr_ir.core.Expr.forward_fill>` / {py:meth}`backward_fill <batcher.plan.expr_ir.core.Expr.backward_fill>`
- [`crates/bc-interp/src/window_spill.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/window_spill.rs): grace partitioning for bounded memory

## See also

- {doc}`Architecture </architecture/index>`: pipeline breakers, and why a window is one.
- {doc}`Execution engine </architecture/internals/execution>`: where the window kernel is driven from.
- {doc}`Kyber </architecture/internals/kyber>`: the pass that sets `rank_limit`.
- {doc}`Window functions </user-guide/analyze/window-functions>`: the API, frames, and `QUALIFY`.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: the four window shapes measured above.
- {doc}`Aggregation internals </architecture/deep-dives/operators/aggregation-internals>`: the dense group ids the shortcut reuses.
- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: the per-partition ordering the ranking functions pay for.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: the same grace partitioning, bounding a window.
