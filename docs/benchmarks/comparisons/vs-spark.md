# vs Spark

This page compares Batcher and Spark on architecture: where each one re-plans a query, what
moves the bulk data, and what that means for a single node.

## The measured standing

Spark 4.2 on OpenJDK 17, local mode, 48-core box, `batcher,spark` pairwise, best of five,
every row correctness-gated. Measured 2026-09-11. Each cell is `batcher_ms / spark_ms`, so
lower is better and every number here is a Batcher win.

| Suite | b/spark | Cases |
|---|---:|---:|
| ClickBench | 0.02 | 39 |
| operators | 0.03 | 23 |
| TPC-H sf1 | 0.03 | 22 |
| H2O `join` | 0.04 | 5 |
| TPC-DS | 0.05 | 87 |
| H2O `groupby` | 0.07 | 10 |
| JSON | 0.01 | 5 |
| scan | 0.21 | 27 |

Batcher is between 5x and 100x faster on every suite. Read that as the single-node case it
is: Spark's per-stage machinery is priced for a cluster, and none of it is amortized here.
`scan` is the closest column because it is the one dominated by reading bytes from S3, which
both engines pay in full.

Three TPC-DS cases and two ClickBench cases are excluded from those counts because Spark
disagreed with the result DuckDB and Batcher both produced: `count(*)` aliased as `count1`
rather than `count_star` (three queries), a timezone rendered as `2013-07-15 19:40:00Z`
against `12:40:00`, and one row count. The naming differences are cosmetic; they are
excluded rather than waved through because the gate compares column names.

TPC-DS needs `BENCH_SPARK_DRIVER_MEMORY=12g` on a 92 GiB box. At the default 32g heap the
pair was OOM-killed.

:::{note}
The section below is an architectural comparison rather than a benchmark, and is labeled so
that nothing in it reads as a speed result.
:::

The design difference is specific and testable, which is what makes it worth writing down.

| | Where to look |
|---|---|
| Architectural comparison | Below |
| Measured distributed results | {doc}`/benchmarks/results/scaling`, measured against {doc}`/benchmarks/comparisons/vs-daft` |
| API migration | Mapped verb by verb in the {doc}`/getting-started/migration/index` |

## Adaptation granularity

Spark's Adaptive Query Execution re-plans between stages. When a shuffle finishes, AQE reads
the materialized shuffle statistics and can coalesce partitions, switch a sort-merge join to
a broadcast join, or split a skewed partition. It is a real and valuable capability, and it
is why Spark survives bad estimates that would sink a purely static optimizer.

The decision points sit where the shuffles do, and that is the constraint.
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

The differences that matter are architectural rather than incidental. Each row names one
and gives both engines' answer:

| | Spark | Batcher |
|---|---|---|
| Re-optimization points | Stage (shuffle) boundaries | Every pipeline breaker |
| Data plane | JVM, row and columnar hybrid | Rust over Arrow, columnar throughout |
| Expression evaluation | Whole-stage codegen (JVM bytecode) | Interpreter oracle plus a Cranelift JIT, bit-for-bit identical on its subset |
| Small-query overhead | JVM start, driver, and scheduler | In-process; a metadata `count()` answers in 0.05 ms |
| Single-node story | The cluster case, shrunk | A first-class, in-process engine |
| Distributed story | The design center | The *same* mergeable operators, scheduled across nodes |
| Bulk data movement | Shuffle files, exchange service | Arrow Flight with credit-based flow control; the Ray object store is bypassed |

The second-to-last row carries the most weight. Batcher's stateful operators are
built once as `partial → combine → finalize`, so one implementation serves a single core,
many cores, and many machines. There is no separate distributed engine with its own
semantics, and a distributed result holds the same rows and types as the single-node one
(float reductions agree to the last bits, since the partition count sets the summation
order). Spark's
single-node mode is its cluster machinery running with one executor.

## What this does not tell you

:::{important}
An architecture table is not a benchmark. Spark at petabyte scale with a tuned cluster is a
serious system, and the claim that Batcher beats it is exactly the sort of claim this site
refuses to make without a correctness-gated measurement. Do not cite this page as a speed
result. It is not one.
:::

The layer beneath it is measured, and it does bear on the comparison. On a 128-CPU cluster
Batcher's distributed path takes the join, the group-by, and the metadata count against
Daft's Ray runner, and keeps per-node memory bounded through the mergeable algebra and spill.
See {doc}`/benchmarks/results/scaling`.

## Three things Spark has that Batcher does not

The measured board above is a single-node board, and single-node is where Spark is weakest.
Three of Spark's advantages do not appear on it at all.

**An external shuffle service.** Batcher's shuffle replicas live in RAM, so they cost memory,
and losing a whole node takes every copy with it and forces a recompute. Spark's shuffle
outlives the executor that wrote it. This is the gap the scorecard in
`docs/architecture/internals/competitive_architecture.md` records as structural rather than
a tuning item.

**Streaming guarantees.** Batcher's streaming is micro-batch. It cannot express what a
continuous-operator engine expresses, which is a limitation against Flink first and Spark
Structured Streaming second.

**Lakehouse formats.** Batcher reaches Iceberg, Delta and Hudi through `pyiceberg` and
`delta-rs` rather than through a native writer, so format support tracks those libraries.

## Migrating

If you are coming from Spark, the API is deliberately close. {py:class}`Session <batcher.Session>`, SQL, `write` modes,
triggers, watermarks, and output modes all mirror the Spark spelling. The
{doc}`/getting-started/migration/index` maps them verb by verb.

## See also

- {doc}`/benchmarks/results/scaling`: the distributed measurements that do exist.
- {doc}`/benchmarks/methodology`: what has to be true before a number is published.
- {doc}`/architecture/optimization`: how breaker-level re-planning works.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization`: the pipeline-breaker
  mechanism, in detail.
- {doc}`/architecture/deep-dives/adaptive/learned-metadata`: the feedback that outlives the
  query.
- {doc}`/getting-started/migration/index`: `Session`, SQL, triggers, watermarks, output
  modes, verb by verb.
