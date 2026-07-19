# Methodology

How the numbers were produced, and how to reproduce them.

## Correctness gates the timer

:::{important}
The harness (`benchmarks/harness.py`) runs each query on every engine, compares the results
as a sorted row multiset within float tolerance, and refuses to record a time if they
disagree. A query whose result doesn't match reports `FAILED` and contributes no number. A
fast wrong answer is a bug, not a win, and it never reaches these pages as a timing.
:::

This is the same discipline the engine is built under: every relational operator is
differential-tested against DuckDB, and the Tier-0 interpreter is the oracle that the
parallel executor and the JIT must agree with bit-for-bit.

It also means the benchmark occasionally reports on *other* engines' correctness. On TPC-H
q6, both Daft and Polars return the wrong revenue, having folded `0.06 + 0.01` in IEEE double to `0.06999999999999999`, which drops every `l_discount = 0.07` row.
The harness declines to time them rather than crediting a fast wrong answer.

## Hardware

Different workload families were measured on different machines, because a GPU benchmark
needs GPUs and a distributed benchmark needs a cluster. Each is labeled where it appears:

| Family | Hardware |
|---|---|
| Operators, TPC-H, connectors | Single node, 16 cores, 30 GB |
| Multimodal ingest (image, point cloud) | Single node, 96 cores |
| Ray Data map / ETL / training ingest | Single node, 96 cores, 188 GB |
| GPU inference and embeddings | 8×T4 across a Ray cluster |
| Distributed scale-out | 9-node Ray cluster, 128 CPUs |

:::{note}
Comparing a number from one row against a number from another row is not meaningful. A
16-core operator timing and an 8×T4 throughput figure describe different machines running
different work, and the arithmetic you can do between them is arithmetic about nothing. The
ratios *within* a row are the claim; the absolute numbers are context for it.
:::

## Engine configuration

Each engine runs on its own home turf rather than in a configuration chosen to make it
look bad. Data is read once into Arrow and shared byte-identically across engines, so nobody
is timed on a different copy of the input. Timings are best-of-N warm.

| Engine | Configuration | Why |
|---|---|---|
| Batcher | Single-node, in-process | Its low-overhead strength |
| Ray Data | Attached to the live cluster (`ray.init(address="auto")`) | Where it is designed to be strongest |
| Daft | Native multithreaded local engine (`DAFT_RUNNER=native`) | Its fastest runner for single-node work |
| DuckDB, Polars | In-process | The only way they run |

## Data

TPC-H at scale factor 1 (`lineitem` = 6,001,215 rows), and scale factor 10 (60M) where
noted, from `s3://ray-benchmark-data/tpch/parquet/`.

## Reproducing

:::{dropdown} Every command, by workload family
```bash
export BENCH_S3_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2
export DAFT_RUNNER=native

# analytics: batcher vs duckdb vs polars vs pyarrow
python benchmarks/run.py --benchmark tpch      --tier single
python benchmarks/run.py --benchmark operators --tier single

# the AI/data-plane lineup: batcher vs ray data vs daft
python benchmarks/run.py --benchmark operators --tier multi

# multimodal ingest
python benchmarks/scenarios/image_decode.py
python benchmarks/scenarios/point_cloud_load.py

# distributed batcher on a live cluster
python benchmarks/scenarios/dist_bench.py --workers 4
```
:::

`python benchmarks/run.py --list` prints every registered benchmark.

## The full log

`benchmarks/BENCHMARK_RESULTS.md` is the complete record, including the runs that went
badly. It keeps the regressions and what fixed them: the JSON writer that once took 65
seconds, the image pipeline that started at 350 img/s and lost to both competitors before
five fixes took it to 5,700, the distributed path that used 1 of 8 GPUs. A benchmark file
with no failures in it is a marketing document, not a measurement.

## See also

- [TPC-H](tpch.md): the suite where the gate has the most to say about other engines.
- [Analytics and I/O](analytics.md) and [AI and GPU](ai-and-gpu.md): the two halves of the
  measurement.
- [Testing strategy](../internals/testing-strategy.md): the same discipline applied to the
  engine itself, where DuckDB is the differential oracle.
- [Performance tuning](../user-guide/performance.md): the levers you have on your own query.
