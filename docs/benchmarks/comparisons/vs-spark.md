# vs Spark

This page compares Batcher with Spark: the measured standing on a single node, and the architecture behind it, from where each engine re-plans a query to what moves its bulk data.

## The measured standing

On one machine Batcher is far ahead. The following results are recorded in [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md), each correctness-gated against DuckDB:

| Workload | Result | Measured |
|---|---|---|
| TPC-H sf1, local-mode Spark | Batcher **20x to 50x** faster | 2026-08-15, 96-core box |
| Streaming drain of a Parquet backlog, 4M rows and 1,000 keys | **211.1 ms against 666.6 ms** for Spark Structured Streaming, 3.2x | 2026-08-18 |
| Operator mix, sf1 | Batcher faster on **11 of 11** operators | 2026-07-25 |

Spark reads results with `DataFrame.toArrow()`, uses one shuffle partition per core, and caches its tables before the clock starts. These are single-node results. {doc}`/benchmarks/results/scaling` has Batcher's distributed measurements.

## Where each engine re-plans

Spark's Adaptive Query Execution re-plans between stages on materialized shuffle statistics. Batcher re-plans the same way, at stage boundaries on measured cardinalities, with the same granularity as AQE. When an estimate is off by more than `optimizer.reoptimize_error` (2x by default), the rest of the query is re-planned on the measured numbers, and the result is identical either way.

Two things differ. Batcher's loop runs inside the Python process rather than in a JVM beside it. And what it measures outlives the query: actual cardinalities, operator times and peak memory feed the optimizer on the next run, so a recurring query gets a better plan each time it executes. Single-node, the within-query loop engages on a joined query once the input clears 5M rows, or about 320 MB, per pipeline breaker it would cut at.

```python
import batcher as bt

print(bt.active_config().optimizer.reoptimize_error)
# 2.0
```

## Where the two engines differ

The following table sets the architectural choices side by side:

| | Spark | Batcher |
|---|---|---|
| Re-optimization | Stage boundaries, on by default since 3.2, local mode or cluster | Stage boundaries, inside the Python process, single node or cluster, plus statistics learned across runs |
| Data plane | JVM, row and columnar hybrid | Rust over Arrow, columnar throughout |
| Expression evaluation | Whole-stage codegen to JVM bytecode | Interpreter oracle plus a Cranelift JIT, bit-for-bit identical on its subset |
| Small-query overhead | JVM, driver and scheduler | In process |
| Single-node story | The cluster machinery with one executor | A first-class in-process engine |
| Distributed story | The design center | The same mergeable operators, scheduled across nodes |
| Bulk data movement | Shuffle files and an external shuffle service | Arrow Flight with credit-based flow control, bypassing the Ray object store |

The distributed row carries the most weight. Batcher's stateful operators are built once as `partial`, `combine` and `finalize`, so one implementation serves a single core, many cores and many machines. There is no separate distributed engine with its own semantics, and a distributed result holds the same rows and types as the single-node one, with floating-point reductions agreeing to the last bits because the partition count sets the summation order.

## Migrating

The API is deliberately close to Spark's. {py:class}`Session <batcher.Session>`, SQL, `write` modes, triggers, watermarks and output modes all mirror the Spark spelling, and {doc}`/getting-started/migration/index` maps them verb by verb:

```python
import batcher as bt

session = bt.Session()
session.register("sales", bt.from_pydict({"region": ["eu", "us", "eu"], "amount": [10, 20, 5]}))
print(session.sql("SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY region").to_pydict())
# {'region': ['eu', 'us'], 'total': [15, 20]}
```

:::{dropdown} Practical differences
- Batcher's shuffle spill directory is worker-local, so a bucket outlives its worker only through a replica. Spark has an external shuffle service.
- Batcher's streaming is micro-batch.
- Iceberg and Delta are reached through `pyiceberg` and `delta-rs`, so format support tracks those libraries.
- These measurements are single node. No recorded benchmark sets Batcher against a tuned Spark cluster.
:::

## Reproduce

Spark needs a JVM as well as the `pyspark` wheel.

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
