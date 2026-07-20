# Morsel parallelism

A *morsel* is the unit of work Batcher schedules across cores. It's an Arrow `RecordBatch`,
and `bc_arrow::Morsel` is a type alias rather than a wrapper, so there's no second data
structure. Its target size is 16,384 rows **or** 1 MiB, whichever it hits first
(`bc_arrow::MorselTarget`). The row bound keeps a narrow batch cache-resident, and the byte
bound keeps a wide one from ballooning. This page describes how morsels are made, how they're
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
(`crates/bc-interp/src/ops/morsel.rs`) corrects both directions. It splits what is too large into row- and byte-bounded pieces, and coalesces a run of
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
| window | hash-partition by `PARTITION BY` key, run the serial kernel per bucket |

Every one of these is a *scheduling* choice. The hash-shuffle that parallelizes a join across
threads is the same mechanism the distributed layer uses across actors, and the per-bucket
join is the same primitive. Nothing about operator semantics changes.

Result *order* for the hash-based operators (aggregate, distinct, join) depends on the worker
count and is therefore not stable across machines. These are unordered relations, so the tests
compare them as multisets. A sort, by contrast, is order-defining, and its parallel path is
deliberately built to produce a **bit-identical** permutation. See
[Sort internals](sort-internals.md).

## How wide?

`EngineConfig.parallelism` is honored verbatim when set: the control plane asked for that
width, and the hash-shuffle bucket count keys off it. When it's `0`, meaning all cores, the
width is `bc_arrow::usable_cores()` **capped by the number of morsels the inputs can produce**
(`par::auto_width`). `usable_cores` is `available_parallelism()` clamped by the cgroup CPU
quota, so a container or a Ray actor gets the width it may actually use rather than the width
of the host.

The cap matters because an idle worker isn't free. Rayon still wakes it and it contends for the
job queue, and because pools are cached per width, a one-row query would otherwise install and
spin up a 96-thread pool. The engine's low-fixed-overhead goal is exactly this case. The cap is
an upper bound on *useful* parallelism at the leaves, so it can never remove parallelism a plan
could have used, and it never changes a result.

The morsel count is byte-aware for the same reason morsels are: a 176 MB audio batch of 2,000
rows morselizes into ~176 pieces, and counting by rows alone scheduled it on one core.

There is one deliberate exception. A plan containing a media decode lifts the cap to all
cores, gated on `RelOp::contains_media_decode`, the plan-level walker over the expression
predicate of the same name. Decode does heavy, embarrassingly-parallel work *inside* a morsel,
and its input is tiny encoded bytes, so a whole corpus of JPEGs looks like one morsel to the
morsel counter and would decode on one core. The decode kernel's own rayon fan-out shares the
same pool, so there is no oversubscription.

Pools themselves are cached per width (`par::pool_for`). Building a `ThreadPool` per execution
is a real cost on the small and streaming paths, where streaming means a new pool per
micro-batch. Sharing one pool per width also bounds the total worker-thread count when
several queries run at once.

## The allocator is part of the story

This isn't a footnote. Every morsel-parallel operator allocates its output buffers per
morsel, and glibc's malloc serves buffers of that size through `mmap`/`munmap`. Each `munmap`
must invalidate the mapping on every core, so it broadcasts a TLB-shootdown IPI. With 96
workers each freeing a buffer per morsel, that interrupt storm becomes a serialization point
in the middle of an embarrassingly parallel scan.

:::{warning}
Measured on a 6M-row filter on 96 cores, glibc's allocator scaled only 5.3x and then
*regressed* past 32 workers. Adding cores made it slower. With mimalloc's per-thread heaps,
which recycle pages instead of returning them, the same filter scales 15x and doesn't regress
(`benchmarks/TPCH_FINDINGS.md`). `bc-py` installs mimalloc as the `#[global_allocator]`
because it's the cdylib every crate links into (`crates/bc-py/src/lib.rs`).
:::

## Seeing it

Morsel size is observable through `iter_batches`, and the result is invariant to it:

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

The honest headline number: **single-node parallelism reaches only about 1.7x to 3.8x on 16
cores** on the TPC-H shapes (`benchmarks/BENCHMARK_RESULTS.md`). The scan-and-aggregate core scales
well. The join doesn't, and the reason is Amdahl rather than mystery. Serial prefixes
before the per-bucket join (materializing a side, gathering the probe side, the shuffle
itself) cap the achievable speedup. Two of those have already been removed (the radix
partition is now a parallel histogram/prefix-sum/scatter; the probe side is gathered once via
`interleave` rather than concatenated and then re-gathered), and the remaining ones are the
open work. Closing that gap is what would close the join-heavy TPC-H gap against DuckDB.

## Where the code lives

- `crates/bc-arrow/src/lib.rs`: `Morsel`, `MorselTarget`, `DEFAULT_MORSEL_ROWS`, `RuntimeTuning`
- `crates/bc-interp/src/ops/morsel.rs`: `morselize`, the split/coalesce rules
- `crates/bc-interp/src/par.rs`: `auto_width`, `pool_for`, the per-operator schedules
- `crates/bc-interp/src/ops/repartition.rs`: gather-once hash partitioning of morsels
- `crates/bc-py/src/lib.rs`: the global allocator

## See also

:::{seealso}
- [Architecture](../architecture/index.md): where scheduling sits relative to operator semantics
- [Execution engine](../internals/execution.md): the sequential oracle this path must agree with
- [Carbonite](../internals/carbonite.md): who decides how big a morsel may get under pressure
- [Performance](../user-guide/performance.md): the `morsel_rows` and `parallelism` knobs, applied
- [Analytics benchmarks](../benchmarks/analytics.md): the single-node numbers
- [Scaling benchmarks](../benchmarks/scaling.md): where the 1.7x to 3.8x figure comes from
- [Mergeable algebra](mergeable-algebra.md): why the parallel schedule computes the same answer
- [Join algorithms](join-algorithms.md): the shuffle-and-bucket join in detail
- [Arrow and memory](arrow-memory.md): the byte budget a morsel is bounded by
:::
