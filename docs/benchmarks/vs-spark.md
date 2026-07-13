# vs Spark

:::{warning}
**There are no head-to-head Spark timings in this repository, and this page invents none.**
Everything else on the benchmarks site is a measurement; this page is a positioning
argument, and it is labeled as one. Every other `vs-*` page opens with a scorecard of wins
and losses. This one cannot, because nothing has been run. When a Spark run lands in
`benchmarks/BENCHMARK_RESULTS.md`, the numbers will appear here and this admonition will go.
:::

What follows is an architectural comparison against the system Spark actually is, which is
worth writing down because the design difference is specific and testable.

| | Status |
|---|---|
| Head-to-head Spark timings | **None.** Not run, not published, not claimed. |
| Architectural comparison | Below, and it is an argument rather than a measurement. |
| Measured distributed results | Real, but against [Ray Data](vs-ray-data.md) and [Daft](vs-daft.md), not Spark. See [scaling](scaling.md). |
| API migration | Mapped verb by verb in the [migration guide](../migration/index.md). |

## Adaptation: stage boundaries versus pipeline breakers

Spark's Adaptive Query Execution re-plans between stages. When a shuffle finishes, AQE reads
the materialized shuffle statistics and can coalesce partitions, switch a sort-merge join to
a broadcast join, or split a skewed partition. It is a real and valuable capability, and it
is why Spark survives bad estimates that would sink a purely static optimizer.

The constraint is where those decision points sit. A stage boundary is a shuffle boundary.
Inside a stage (a scan, a filter, a projection, a hash-aggregate's build) Spark is committed
to the plan it entered with, however wrong the estimate that produced it turns out to be.

Batcher re-optimizes at every **pipeline breaker**: a sort, an aggregate, a join build. A
breaker is a point where the engine has just *measured* the true size of what it processed,
and there are more of them than there are shuffles. When an estimate is off by more than
`optimizer.reoptimize_error` (2× by default), the rest of the query is re-planned on the
measured numbers before it continues. The result is identical whichever way it runs;
only the plan changes.

That measured feedback also outlives the query. Core records actual cardinalities, operator
times and peak memory into the MetadataHub, and Kyber reads them on the *next* run, so a
recurring query gets a better plan each time it executes.

## Where the two engines differ

| | Spark | Batcher |
|---|---|---|
| Re-optimization points | Stage (shuffle) boundaries | Every pipeline breaker |
| Data plane | JVM, row and columnar hybrid | Rust over Arrow, columnar throughout |
| Expression evaluation | Whole-stage codegen (JVM bytecode) | Interpreter oracle plus a Cranelift JIT, bit-for-bit identical on its subset |
| Small-query overhead | JVM start, driver, and scheduler | In-process; a metadata `count()` answers in 0.05 ms |
| Single-node story | The cluster case, shrunk | A first-class, in-process engine |
| Distributed story | The design center | The *same* mergeable operators, scheduled across nodes |
| Bulk data movement | Shuffle files, exchange service | Arrow Flight with credit-based flow control; the Ray object store is bypassed |

The row that carries the most weight is the second-to-last. Batcher's stateful operators are
built once as `partial → combine → finalize`, so one implementation serves a single core,
many cores, and many machines. There is no separate distributed engine with its own
semantics, and a distributed result is bit-identical to the single-node one. Spark's
single-node mode is its cluster machinery running with one executor.

## What this does not tell you

:::{important}
An architecture table is not a benchmark. Spark at petabyte scale with a tuned cluster is a
serious system, and the claim that Batcher beats it is exactly the sort of claim this site
refuses to make without a correctness-gated measurement. Do not cite this page as a speed
result. It is not one.
:::

What *is* measured, and does bear on the comparison, is the layer beneath: Batcher's
distributed path beats Ray Data on every pipeline at every scale tested, beats Daft on 4 of
5 distributed pipelines, and keeps per-node memory bounded through the mergeable algebra and
spill. See [scaling](scaling.md).

## Migrating

If you are coming from Spark, the API is deliberately close. `Session`, SQL, `write` modes,
triggers, watermarks, and output modes all mirror the Spark spelling. The
[migration guide](../migration/index.md) maps them verb by verb.

## See also

- [Scaling](scaling.md): the distributed measurements that do exist.
- [Methodology](methodology.md): what has to be true before a number is published.
- [Optimization](../architecture/optimization.md): how breaker-level re-planning works.
- [Adaptive re-optimization](../deep-dives/adaptive-reoptimization.md): the pipeline-breaker
  mechanism, in detail.
- [Learned metadata](../deep-dives/learned-metadata.md): the feedback that outlives the
  query.
- [Migration guide](../migration/index.md): `Session`, SQL, triggers, watermarks, output
  modes, verb by verb.
