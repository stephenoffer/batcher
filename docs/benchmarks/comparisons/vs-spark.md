# vs Spark

This page compares Batcher with Spark: the measured standing on a single node, and the architecture behind it, from where each engine re-plans a query to what moves its bulk data.

## The measured standing

On one machine Batcher is far ahead. The following results are recorded in [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md), each correctness-gated against DuckDB:

| Workload | Result | Measured |
|---|---|---|
| TPC-H sf1, local-mode Spark | Batcher **20x to 50x** faster | 2026-08-15, 96-core box |
| Streaming drain of a Parquet backlog, 4M rows and 1,000 keys | **211.1 ms against 666.6 ms** for Spark Structured Streaming, 3.2x | 2026-08-18 |
| Operator mix, sf1 | Batcher faster on **11 of 11** operators | 2026-07-25 |

The operator sweep predates the Spark configuration fixes below, which the record notes changed no winner. The TPC-H figure was taken after three handicaps on Spark's side were removed: a pandas round trip on every result, 8 shuffle partitions on a 96-core box, and a Parquet re-read on every query where the other engines queried loaded tables. Spark now reads with `DataFrame.toArrow()`, uses one shuffle partition per core, and caches its tables before the clock starts. It is still 20x to 50x behind, because local-mode Spark carries about 90 ms of fixed cost per query that nothing amortizes on 6M rows.

Read these as single-node results. Spark's per-stage machinery is priced for a cluster, so a single-node board measures Spark where it is weakest. {doc}`/benchmarks/results/scaling` has Batcher's distributed measurements, taken against Daft's Ray runner.

## Where each engine re-plans

Spark's Adaptive Query Execution re-plans between stages. When a shuffle finishes, AQE reads the materialized shuffle statistics and can coalesce partitions, switch a sort-merge join to a broadcast join, or split a skewed partition. It is why Spark survives estimates that would sink a purely static optimizer.

Batcher re-plans the same way, at stage boundaries on measured cardinalities, with the same granularity as AQE. When an estimate is off by more than `optimizer.reoptimize_error` (2x by default), the rest of the query is re-planned on the measured numbers, and the result is identical either way. Two things differ. The loop runs on a single node too, where AQE needs shuffle stages to exist. And what it measures outlives the query: Core records actual cardinalities, operator times and peak memory into the metadata hub, sketches and calibrated costs feed Kyber on the next run, so a recurring query gets a better plan each time it executes.

The within-query loop isn't always on. Single-node, it engages on a query with a join once the input clears 5M rows, or about 320 MB, for each pipeline breaker it would cut at, so the simplest joined shape qualifies at about 10M rows.

## Where the two engines differ

The following table sets the architectural choices side by side:

| | Spark | Batcher |
|---|---|---|
| Re-optimization | Stage boundaries, cluster only | Stage boundaries, single node or cluster, plus statistics learned across runs |
| Data plane | JVM, row and columnar hybrid | Rust over Arrow, columnar throughout |
| Expression evaluation | Whole-stage codegen to JVM bytecode | Interpreter oracle plus a Cranelift JIT, bit-for-bit identical on its subset |
| Small-query overhead | JVM, driver and scheduler | In process |
| Single-node story | The cluster machinery with one executor | A first-class in-process engine |
| Distributed story | The design center | The same mergeable operators, scheduled across nodes |
| Bulk data movement | Shuffle files and an external shuffle service | Arrow Flight with credit-based flow control, bypassing the Ray object store |

The distributed row carries the most weight. Batcher's stateful operators are built once as `partial`, `combine` and `finalize`, so one implementation serves a single core, many cores and many machines. There is no separate distributed engine with its own semantics, and a distributed result holds the same rows and types as the single-node one, with floating-point reductions agreeing to the last bits because the partition count sets the summation order.

## Migrating

The API is deliberately close to Spark's. {py:class}`Session <batcher.Session>`, SQL, `write` modes, triggers, watermarks and output modes all mirror the Spark spelling, and {doc}`/getting-started/migration/index` maps them verb by verb.

## Requirements and limitations

The single-node board doesn't show what Spark does best. The following gaps are where Spark leads:

- **Shuffle survivability.** Batcher's shuffle can spill to disk, but the spill directory is worker-local, so a bucket outlives its worker only if a replica does. Spark's external shuffle service keeps shuffle output after the executor that wrote it is gone.
- **Streaming guarantees.** Batcher's streaming is micro-batch. It can't express what a continuous-operator engine expresses, which is a limitation against Flink first and Spark Structured Streaming second.
- **Lakehouse formats.** Batcher reaches Iceberg and Delta through `pyiceberg` and `delta-rs` rather than through its own table-format implementation, so format support tracks those libraries.
- **Cluster scale.** No recorded benchmark sets Batcher against a tuned Spark cluster. Don't read this page as a claim about petabyte-scale Spark.

## Reproduce

The following commands rerun the Spark measurements. Spark needs a JVM as well as the `pyspark` wheel, and without one its adapter reports unavailable and the lineup drops it:

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,spark
python benchmarks/scenarios/streaming_throughput.py
```

## See also

- {doc}`/benchmarks/results/scaling`: the distributed measurements.
- {doc}`/benchmarks/methodology`: what has to be true before a number is published.
- {doc}`/architecture/optimization`: how stage-boundary re-planning works.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization`: the mechanism and its size floor, in detail.
- {doc}`/architecture/deep-dives/adaptive/learned-metadata`: the feedback that outlives the query.
- {doc}`/getting-started/migration/index`: `Session`, SQL, triggers, watermarks and output modes, verb by verb.
