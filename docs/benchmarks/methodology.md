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
| Suite geomeans (TPC-H, TPC-DS, ClickBench, JOB, H2O, operators, JSON), 2026-08-15 | Single node, 96 cores, 184 GiB |
| Older per-operator and connector figures | Single node, 16 cores, 30 GB |
| Multimodal ingest (image, point cloud) | Single node, 96 cores |
| `map_batches` ETL and training ingest | Single node, 96 cores, 188 GB |
| GPU inference and embeddings | 8xT4 across a Ray cluster |
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
| DuckDB (`duckdb`) | Its **native** store, ingested untimed before the query | DuckDB at its best — the harder bar |
| DuckDB (`duckdb_arrow`) | The **same zero-copy Arrow** Batcher runs on | The like-for-like execution comparison |
| Daft | Native multithreaded local engine (`DAFT_RUNNER=native`), or its Ray runner for the cluster grid | Its fastest runner for each shape |
| DuckDB, Polars | In-process | The only way they run |
| Distributed engines | Attached to the live cluster (`ray.init(address="auto")`) | Where they are designed to be strongest |

## Suite coverage

The harness registers **346 benchmarks across ten suites**, spanning the industry-standard
analytics set and the workload families that are specific to Batcher's range:

| Suite | Queries | What it covers |
|---|---:|---|
| TPC-DS | 99 | The full official set, vendored from DuckDB's `tpcds` extension |
| Join Order Benchmark | 113 | Join planning against the real IMDb dataset, 21 tables |
| ClickBench | 43 | Wide-table scan and aggregate on web analytics data |
| Scan and I/O | 27 | Parquet, CSV, JSON, and the connectors |
| TPC-H | 22 | The full official set, at scale factors 1 and 10 |
| Operators | 11 | The data-plane kernel lineup in isolation |
| H2O.ai db-benchmark, group-by | 10 | Grouping from 100 groups to 10M, the standard cardinality sweep |
| H2O.ai db-benchmark, join | 5 | Five join shapes across small, medium, and large build sides |
| Semi-structured JSON | 5 | Nested extraction and projection |
| Images | 3 | Decode to tensor |

TPC-DS and the Join Order Benchmark exercise planning far harder than TPC-H does. The median
JOB query joins 8 tables and the largest joins 17, which is the regime where join ordering
and cardinality estimation decide the runtime rather than the kernels.

## Data

TPC-H at scale factor 1 (`lineitem` = 6,001,215 rows), and scale factor 10 (60M) where
noted, from `s3://ray-benchmark-data/tpch/parquet/`. TPC-DS is generated locally through
DuckDB's `dsdgen`. The Join Order Benchmark runs on the real IMDb snapshot from
`event.cwi.nl/da/job/imdb.tgz`, converted once to Parquet. H2O.ai tables are generated to
match the reference R generators, seed 108, so the group cardinalities are the published
ones.

## Reproducing

:::{dropdown} Every command, by workload family
```bash
export BENCH_S3_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2
export DAFT_RUNNER=native

# analytics: batcher vs duckdb vs polars vs pyarrow
python benchmarks/run.py --benchmark tpch      --tier single
python benchmarks/run.py --benchmark operators --tier single

# the planning-heavy suites
python benchmarks/run.py --benchmark tpcds
python benchmarks/run.py --benchmark job
python benchmarks/run.py --benchmark h2o-groupby
python benchmarks/run.py --benchmark h2o-join

# the AI and data-plane lineup
python benchmarks/run.py --benchmark operators --tier multi

# multimodal ingest
python benchmarks/scenarios/image_decode.py
python benchmarks/scenarios/point_cloud_load.py

# distributed batcher on a live cluster
python benchmarks/scenarios/dist_bench.py --workers 4
```
:::

`python benchmarks/run.py --list` prints every registered benchmark, and
`--skip SUBSTRING` drops matching cases from a run and reports what it dropped.

## The full log

`benchmarks/BENCHMARK_RESULTS.md` is the complete engineering record. It tracks every
optimization from first measurement to shipped result, which is what makes the numbers on
these pages auditable: the JSON writer that went from 65 seconds to sub-second, the image
pipeline that five fixes took from 350 img/s to 5,700, the distributed path that now uses
all 8 GPUs instead of 1. Each entry names the change, the measurement method, and the
result.

## See also

- {doc}`/benchmarks/results/tpch`: the suite where the gate has the most to say about other engines.
- {doc}`/benchmarks/results/analytics` and {doc}`/benchmarks/results/ai-and-gpu`: the two halves of the
  measurement.
- {doc}`/architecture/internals/testing-strategy`: the same discipline applied to the
  engine itself, where DuckDB is the differential oracle.
- {doc}`/user-guide/operate/tuning/performance`: the levers you have on your own query.
