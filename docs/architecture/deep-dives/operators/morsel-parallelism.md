# Morsel parallelism

A *morsel* is the unit of work Batcher schedules across cores. It's an Arrow `RecordBatch` (`bc_arrow::Morsel` is a type alias, not a wrapper), sized to 16,384 rows **or** 1 MiB, whichever it hits first (`bc_arrow::MorselTarget`). The row bound keeps a narrow batch cache-resident, and the byte bound keeps a wide one, such as a table of images, from ballooning. This page covers how morsels are made, how they're scheduled, and how wide the schedule goes.

You can see the row bound from the API, and that the answer doesn't depend on the schedule:

```python
import dataclasses

import batcher as bt
from batcher import Config, config_context

ds = bt.from_pydict({"x": list(range(50_000))})
print([b.num_rows for b in ds.filter(bt.col("x") >= 0).iter_batches()])
# [16384, 16384, 16384, 848]

for width in (1, 8):
    cfg = Config().replace(execution=dataclasses.replace(Config().execution, parallelism=width))
    with config_context(cfg):
        print(width, ds.filter(bt.col("x") % 2 == 0).agg(s=bt.col("x").sum()).to_pydict())
# 1 {'s': [624975000]}
# 8 {'s': [624975000]}
```

```text
  input relation: whatever the source happened to emit
  ┌────────────────────────┬─────┬───────────────────────────────┬──┬──┬──┐
  │       row group        │     │           row group           │  │  │  │
  └────────────────────────┴─────┴───────────────────────────────┴──┴──┴──┘
                     │  morselize: split what is too large, coalesce what is too small
                     ▼
  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
  │ morsel │ morsel │ morsel │ morsel │ morsel │ morsel │ morsel │ morsel │
  └───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┘
      └────────┴───┐    └────────┴───┐    └───┬────┘        └───┬────┘
                   ▼                 ▼        ▼                 ▼
              ┌────────┐        ┌────────┐ ┌────────┐      ┌────────┐
              │ worker │        │ worker │ │ worker │      │ worker │   rayon work-stealing:
              │   0    │        │   1    │ │   2    │      │   3    │   a slow morsel does not
              └───┬────┘        └───┬────┘ └───┬────┘      └───┬────┘   stall the others
                  └─────────────────┴────┬─────┴───────────────┘
                                         ▼
                            combine  →  finalize     (stateless operators just emit)
```

## Making morsels

Sources don't produce well-sized batches. A Parquet reader emits row groups, a streaming reader emits whatever arrived, and a selective filter emits crumbs. `morselize` ([`crates/bc-interp/src/ops/morsel.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/morsel.rs)) splits what is too large and coalesces a run of undersized batches up to the target before splitting. A batch already at half the target or more stands alone and passes through zero-copy, so well-sized input pays nothing. Coalescing matters because each tiny batch would otherwise become its own task and its own partial state to merge.

:::{dropdown} How byte sizing stays cheap
Byte sizing is measured, not estimated, but only when it can matter. For an all-fixed-width batch the per-row width is constant, so the split is O(1). For a variable-width batch whose average row is narrow enough that the row target always trips first, it is also O(1). Only when the average row approaches the byte budget does the morselizer walk the offset buffers for each row's true cost. That is what makes a single row wider than the whole budget its own one-row morsel.
:::

## Scheduling

`bc-interp::par` is the multi-core executor. It shares the operator primitives in `ops/` with the sequential oracle and changes only how they are scheduled:

| Operator | Schedule |
|---|---|
| filter, project, unnest | per morsel, embarrassingly parallel |
| aggregate, distinct | partial-aggregate each morsel in parallel, then `combine` + `finalize` |
| join | hash-shuffle both sides into one bucket per worker, join the buckets in parallel |
| sort | sample-sort: range-partition by the leading key, sort each range |
| window | hash-partition by the `PARTITION BY` keys and run the serial kernel per bucket; with no partition keys, exact running aggregates run as a parallel prefix scan |

Every one of these is a scheduling choice. The hash-shuffle that parallelizes a join across threads is the same mechanism the distributed layer uses across actors, so operator semantics never change.

![Three zoom levels on how a filter-and-project chain reaches the cores. First, morselize splits and coalesces whatever the source emitted into morsels, each full at 16,384 rows or 1 MiB, whichever bound trips first, so one over-budget row becomes a one-row morsel. Second, the morsel vector is handed to one cached pool through a single par_iter() call, one task per morsel, at a width of operator_cores() capped by the number of morsels the input can produce; an idle worker steals from a busy one, and that stealing is rayon's scheduler rather than a queue Batcher owns. Third, one worker's pass over one morsel: rows in, a filter compiled once by the JIT, survivors projected without ever being materialized, and a morsel out, collected in index order so filter and project preserve row order where the hash operators do not.](/_static/diagrams/morsel_scheduling.svg)

Result order for the hash-based operators (aggregate, distinct, join) depends on the worker count, so it isn't stable across machines and the tests compare those relations as multisets. A sort is order-defining, and its parallel path produces a bit-identical permutation. See {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`.

## How wide?

`EngineConfig.parallelism` is honored verbatim when set, and the hash-shuffle bucket count keys off it. At `0`, meaning all cores, `par::auto_width` takes the smaller of two numbers:

- `bc_arrow::operator_cores()`: every physical core plus a third of the SMT siblings, from `usable_cores()`, which clamps `available_parallelism()` by the cgroup CPU quota. A container or Ray actor gets the width it may actually use.
- The number of morsels the inputs can produce, counted by bytes as well as rows, so a 176 MB audio batch of 2,000 rows counts as ~176 pieces. The cap keeps a one-row query from spinning up a 96-thread pool, and it never changes a result.

Two kinds of plan lift the morsel cap to every usable core, because their work isn't bounded by the leaf's morsel count: a plan that multiplies rows (`Unnest`, `Unpivot`, `RangeJoin`) and a media decode, whose tiny encoded input hides heavy per-row work.

:::{dropdown} Why not every logical CPU
A plan interleaves work that stalls on memory, such as hash build and probe, with work that saturates bandwidth, such as gather and scan. SMT siblings help the first while doubling cache pressure on the second. With the same binary and only the width changed, 164 of 198 benchmark queries preferred 64 workers to 96 on a 96-CPU box. The fifteen H2O queries, each one large operator over 10M rows, paid 4% to 6% for it.

Pools are cached per width (`par::pool_for`), so the small and streaming paths don't build a `ThreadPool` per execution, and the total worker-thread count stays bounded when several queries run at once.
:::

:::{dropdown} The allocator
Every morsel-parallel operator allocates output buffers per morsel. glibc's malloc serves buffers of that size through `mmap`/`munmap`, and each `munmap` broadcasts a TLB-shootdown interrupt to every core. On a 6M-row filter on 96 cores, glibc scaled only 5.3x and regressed past 32 workers. With mimalloc's per-thread heaps, which recycle pages instead of returning them, the same filter scales 15x ([`benchmarks/TPCH_FINDINGS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/TPCH_FINDINGS.md)). `bc-py` installs mimalloc as the `#[global_allocator]` ([`crates/bc-py/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/lib.rs)).
:::

## How far it scales

The scan-and-aggregate core scales well: `GROUP BY` alone over TPC-H `lineitem` at scale factor 1 reaches **19.2x** on 16 cores against 1 (`docs/architecture/internals/parity/databricks_parity.md`, in the repository rather than on this site). A join is bounded by the serial work around the per-bucket join, and two of those prefixes are already parallel: the radix partition runs as a parallel histogram/prefix-sum/scatter, and the probe side is gathered once via `interleave`.

## Where the code lives

- [`crates/bc-arrow/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-arrow/src/lib.rs): `Morsel`, `MorselTarget`, `DEFAULT_MORSEL_ROWS`, `DEFAULT_MORSEL_BYTES`, `RuntimeTuning`
- [`crates/bc-arrow/src/hardware.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-arrow/src/hardware.rs): `usable_cores`, `operator_cores`
- [`crates/bc-interp/src/ops/morsel.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/morsel.rs): `morselize`, the split/coalesce rules
- [`crates/bc-interp/src/par.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/par.rs): `auto_width`, `pool_for`, the per-operator schedules
- [`crates/bc-interp/src/ops/repartition.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/repartition.rs): gather-once hash partitioning of morsels
- [`crates/bc-py/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/lib.rs): the global allocator

## See also

- {doc}`Architecture </architecture/index>`: where scheduling sits relative to operator semantics.
- {doc}`Execution engine </architecture/internals/execution>`: the sequential oracle this path must agree with.
- {doc}`Carbonite </architecture/internals/carbonite>`: who decides how big a morsel may get under pressure.
- {doc}`Performance </user-guide/operate/tuning/performance>`: the `morsel_rows` and `parallelism` knobs, applied.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: the single-node numbers.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: what the same schedule does across machines.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why the parallel schedule computes the same answer.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: the shuffle-and-bucket join in detail.
- {doc}`Arrow and memory </architecture/deep-dives/memory/arrow-memory>`: the byte budget a morsel is bounded by.
