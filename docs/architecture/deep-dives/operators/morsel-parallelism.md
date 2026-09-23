# Morsel parallelism

A *morsel* is the unit of work Batcher schedules across cores. It's an Arrow `RecordBatch`,
and `bc_arrow::Morsel` is a type alias rather than a wrapper, so there's no second data
structure. Its target size is 16,384 rows **or** 1 MiB, whichever it hits first
(`bc_arrow::MorselTarget`). The row bound keeps a narrow batch cache-resident, and the byte
bound keeps a wide one from ballooning. `DEFAULT_MORSEL_ROWS` and `DEFAULT_MORSEL_BYTES` in `bc-arrow` hold the two defaults. This page describes how morsels are made, how they're
scheduled, and where the granularity stops paying.

The three sizes a morsel is chosen against are the three ways cutting a query goes wrong. One
piece per core lets a single slow partition stall everyone. Too fine, and scheduling overhead
eats the gain. Cut by row count alone, and a table of 5 MB images blows out memory.

```text
  input relation: whatever the source happened to emit
  ┌────────────────────────┬─────┬───────────────────────────────┬──┬──┬──┐
  │       row group        │     │           row group           │  │  │  │
  └────────────────────────┴─────┴───────────────────────────────┴──┴──┴──┘
                     │
                     │  morselize:  split what is too large
                     │              coalesce what is too small
                     ▼
  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
  │ morsel │ morsel │ morsel │ morsel │ morsel │ morsel │ morsel │ morsel │
  └───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┘
      │        │        │        │        │        │        │        │
      └────────┴───┐    └────────┴───┐    └───┬────┘        └───┬────┘
                   ▼                 ▼        ▼                 ▼
              ┌────────┐        ┌────────┐ ┌────────┐      ┌────────┐
              │ worker │        │ worker │ │ worker │      │ worker │   rayon, work-stealing:
              │   0    │        │   1    │ │   2    │      │   3    │   a slow morsel does not
              └───┬────┘        └───┬────┘ └───┬────┘      └───┬────┘   stall the others
                  │                 │          │               │
                  └─────────────────┴────┬─────┴───────────────┘
                                         ▼
                            combine  →  finalize            (stateless operators skip
                                         │                   this and just emit)
                                         ▼
                                     result morsels
```

## Making morsels

Sources don't produce well-sized batches. A Parquet reader emits row groups, a streaming
reader emits whatever arrived, and a selective upstream filter emits crumbs. `morselize`
([`crates/bc-interp/src/ops/morsel.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/morsel.rs)) corrects both directions. It splits what is too large into row- and byte-bounded pieces, and coalesces a run of
undersized batches up to the target *before* splitting. A batch already at half the target or
more counts as standing alone: it's neither buffered nor copied, and it splits or passes
through zero-copy, so well-sized input pays nothing.

:::{warning}
Skipping the coalesce step is expensive in a way that's easy to miss. Each tiny batch becomes
its own task and its own partial aggregate state to merge, so the per-morsel overhead is paid
in full for a fraction of the work. A source that emits crumbs defeats the granularity the
rest of this page depends on.
:::

Byte sizing is measured, not estimated, but only when it can matter. All-fixed-width batch:
the per-row width is constant, so the split is O(1). Variable-width batch whose *average*
row is narrow enough that the row target always trips first: also O(1), because per-row
widths can't move the boundaries. Only when the average row approaches the byte budget does
the morselizer walk the offset buffers for each row's true cost. That is what makes a
single row wider than the whole budget become its own one-row morsel, instead of sharing a
morsel with 16,383 others.

## Scheduling

`bc-interp::par` is the multi-core executor. It shares the operator primitives in
`ops/` with the sequential oracle and changes only *how* they are scheduled:

| Operator | Schedule |
|---|---|
| filter, project, unnest | per morsel, embarrassingly parallel |
| aggregate, distinct | partial-aggregate each morsel in parallel, then `combine` + `finalize` |
| join | hash-shuffle both sides into one bucket per worker, join the buckets in parallel |
| sort | sample-sort: range-partition by the leading key, sort each range |
| window | hash-partition by the `PARTITION BY` keys, run the serial kernel per bucket; a window with no partition keys runs its exact running aggregates as a parallel prefix scan |

Every one of these is a *scheduling* choice. The hash-shuffle that parallelizes a join across
threads is the same mechanism the distributed layer uses across actors, and the per-bucket
join is the same primitive. Nothing about operator semantics changes.

The three zoom levels below follow one filter-and-project chain from the batches a source
happened to emit down to a single worker's pass over a single morsel.

![Three zoom levels on how a filter-and-project chain reaches the cores. First, morselize splits and coalesces whatever the source emitted into morsels, each full at 16,384 rows or 1 MiB, whichever bound trips first, so one over-budget row becomes a one-row morsel. Second, the morsel vector is handed to one cached pool through a single par_iter() call, one task per morsel, at a width of operator_cores() capped by the number of morsels the input can produce; an idle worker steals from a busy one, and that stealing is rayon's scheduler rather than a queue Batcher owns. Third, one worker's pass over one morsel: rows in, a filter compiled once by the JIT, survivors projected without ever being materialized, and a morsel out, collected in index order so filter and project preserve row order where the hash operators do not.](/_static/diagrams/morsel_scheduling.svg)

Result *order* for the hash-based operators (aggregate, distinct, join) depends on the worker
count and is therefore not stable across machines. These are unordered relations, so the tests
compare them as multisets. A sort, by contrast, is order-defining, and its parallel path is
deliberately built to produce a **bit-identical** permutation. See
{doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`.

## How wide?

`EngineConfig.parallelism` is honored verbatim when set: the control plane asked for that
width, and the hash-shuffle bucket count keys off it. When it's `0`, meaning all cores,
`par::auto_width` picks the width from two numbers. The first is `bc_arrow::operator_cores()`:
every physical core plus a third of the SMT siblings, taken from `usable_cores()`, which is
`available_parallelism()` clamped by the cgroup CPU quota. A container or a Ray actor therefore
gets the width it may actually use rather than the width of the host. The second is the number
of morsels the inputs can produce, which caps the first.

Not every logical CPU gets a worker, and that was measured rather than assumed. A plan
interleaves work that stalls on memory, such as hash build and probe, with work that saturates
bandwidth, such as gather and scan, and SMT siblings help the first while doubling cache pressure
on the second. Same binary, only the width changed, 164 of 198 benchmark queries preferred 64
workers to 96 on a 96-CPU box. The fifteen H2O queries, each one large operator over 10M rows,
paid 4% to 6% for it.

The morsel cap matters because an idle worker isn't free. Rayon still wakes it and it contends
for the job queue, and because pools are cached per width, a one-row query would otherwise
install and spin up a 96-thread pool. The engine's low-fixed-overhead goal is exactly this case.
The cap only bounds parallelism at the leaves, and it never changes a result.

The morsel count is byte-aware for the same reason morsels are: a 176 MB audio batch of 2,000
rows morselizes into ~176 pieces, and counting by rows alone scheduled it on one core.

Two kinds of plan lift the cap to every usable core, because the work below a leaf is not
bounded by that leaf's morsel count. A plan that multiplies rows is the first
(`RelOp::multiplies_rows`: `Unnest`, `Unpivot` and `RangeJoin`). A hundred rows of
ten-thousand-element lists are one morsel at the leaf and a million rows one operator later, so
capping on the leaf would pin the whole downstream to one core. A media decode is the second,
gated on `RelOp::contains_media_decode`. Decode does heavy, embarrassingly-parallel work *inside* a morsel,
and its input is tiny encoded bytes, so a whole corpus of JPEGs looks like one morsel to the
morsel counter and would decode on one core. The decode kernel's own rayon fan-out shares the
same pool, so there is no oversubscription.

Pools themselves are cached per width (`par::pool_for`). Building a `ThreadPool` per execution
is a real cost on the small and streaming paths, where streaming means a new pool per
micro-batch. Sharing one pool per width also bounds the total worker-thread count when
several queries run at once.

## The allocator is part of the story

Every morsel-parallel operator allocates its output buffers per
morsel, and glibc's malloc serves buffers of that size through `mmap`/`munmap`. Each `munmap`
must invalidate the mapping on every core, so it broadcasts a TLB-shootdown IPI. With 96
workers each freeing a buffer per morsel, that interrupt storm becomes a serialization point
in the middle of an embarrassingly parallel scan.

:::{warning}
Measured on a 6M-row filter on 96 cores, glibc's allocator scaled only 5.3x and then
*regressed* past 32 workers. Adding cores made it slower. With mimalloc's per-thread heaps,
which recycle pages instead of returning them, the same filter scales 15x and doesn't regress
([`benchmarks/TPCH_FINDINGS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/TPCH_FINDINGS.md)). `bc-py` installs mimalloc as the `#[global_allocator]`
because it's the cdylib every crate links into ([`crates/bc-py/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/lib.rs)).
:::

## Seeing it

Morsel size is observable through {py:meth}`iter_batches <batcher.Dataset.iter_batches>`, and the result is invariant to it:

```python
import dataclasses
import batcher as bt
from batcher import Config, config_context

ds = bt.from_pydict({"x": list(range(50_000))})
base = Config()

for rows in (16_384, 4_096):
    cfg = base.replace(execution=dataclasses.replace(base.execution, morsel_rows=rows))
    with config_context(cfg):
        sizes = [b.num_rows for b in ds.filter(bt.col("x") >= 0).iter_batches()]
    print(rows, "->", len(sizes), "morsels", sizes[:3], "total", sum(sizes))
```

```text
16384 -> 4 morsels [16384, 16384, 16384] total 50000
4096 -> 13 morsels [4096, 4096, 4096] total 50000
```

Same 50,000 rows, different scheduling granularity. `parallelism` is the other knob. Both are
shipped to Rust inside `EngineConfig`, so the two sides can't disagree about them.

## What it costs, and where it stops paying

Morsel granularity buys load balance and cache residency. It costs a per-morsel scheduling
and state-merge overhead that only amortizes if the morsel is big enough. 16,384
rows was chosen to sit in L2/L3 for narrow data; an un-coalesced source is what defeats it.

How far a shape scales is set by Amdahl rather than by the morsel loop. The
scan-and-aggregate core scales well, and `GROUP BY` alone over TPC-H `lineitem` at scale
factor 1 reaches **19.2x** on 16 cores against 1
(`docs/architecture/internals/parity/databricks_parity.md`, in the repository rather than on
this site). The join is bounded by the serial prefixes around the
per-bucket work: materializing a side, gathering the probe side, and the shuffle itself. Two
of those have been removed, so the radix partition is now a parallel
histogram/prefix-sum/scatter and the probe side is gathered once via `interleave` rather than
concatenated and then re-gathered.

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
