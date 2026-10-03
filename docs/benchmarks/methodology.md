# Methodology

This page describes how the benchmark numbers on this site are produced and how to reproduce them.

## The correctness gate

:::{important}
A fast wrong answer never gets a ratio. The harness in [`benchmarks/harness/`](https://github.com/stephenoffer/batcher/tree/main/benchmarks/harness) runs each query on every engine, compares the results, and times only the engines that agree.
:::

- **Results compare as a sorted row multiset.** Integers, strings, booleans and decimals compare exactly. Floats compare within a symmetric tolerance.
- **Ordered results are checked for order.** Every case with an outermost `ORDER BY` is also checked for monotonicity per engine ([`benchmarks/harness/order.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/harness/order.py)).
- **A disagreement withholds the ratio.** The case reports `FAILED` and the ratio prints as `n/c`.
- **Known semantic differences are recorded, not excused.** [`benchmarks/harness/divergences.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/harness/divergences.py) names which engine is right for each one, with a citation. Such a case reports `DIVERGENT` and still carries no ratio.
- **Self-comparisons are type-strict.** Scripts that compare Batcher against itself, single node against distributed or CPU against GPU, pass `strict_types=True`.

The gate is the same check you can run by hand. This block computes a group-by in Batcher and in DuckDB over the same Arrow table and asserts that they agree:

```python
import batcher as bt
import duckdb
import pyarrow as pa

events = pa.table({"k": ["a", "b", "a", "c"], "v": [1, 2, 3, 4]})
ours = bt.from_arrow(events).group_by("k").agg(s=bt.col("v").sum()).sort("k").to_pydict()
theirs = duckdb.sql("SELECT k, SUM(v)::BIGINT AS s FROM events GROUP BY k ORDER BY k").fetchall()
assert list(zip(ours["k"], ours["s"])) == theirs
print(theirs)
# [('a', 4), ('b', 2), ('c', 4)]
```

[`tests/unit/test_benchmark_comparator.py`](https://github.com/stephenoffer/batcher/blob/main/tests/unit/test_benchmark_comparator.py) pins every comparison rule, and {doc}`/architecture/internals/testing-strategy` covers the same discipline applied to the engine itself.

## How a ratio is computed

- Every headline figure is a geometric mean of per-query `batcher_ms / engine_ms`. Lower is better, and below 1.00 is a Batcher win.
- Timings are best of N, warm: one execution is discarded, then N are timed.
- Every engine reads the same source data, loaded once into Arrow.
- Each table names the number of cases it averages and the machine it ran on.

## Hardware

Each workload family ran on the machine it needs. Every table on this site names its machine:

| Family | Hardware |
|---|---|
| Current single-node board, 2026-09-13, and the TPC-DS sweep of 2026-09-11 | 48 cores (24 physical plus SMT), 92 GiB |
| Five-engine board with Daft and PyArrow, 2026-08-28 | 92-core box |
| Suite sweeps of 2026-08-15 to 2026-08-25, including TPC-H sf10 and the Join Order Benchmark | 96 cores, 184 GiB |
| Per-query TPC-H board, 2026-07-28 | c5d.24xlarge, 96 vCPU, 184 GiB |
| Older per-operator, connector and kernel figures | 16 cores, 30 GB |
| Multimodal ingest (image, point cloud) | One 96-core node |
| GPU inference and embeddings | 8xT4 across a Ray cluster |
| GPU inference against Ray Data and Daft | Six single-T4 nodes, or 17 nodes with 8 T4s, as each table states |
| Distributed scale-out | 16 worker nodes of 8 CPUs each, 128 CPUs |

Compare ratios within a table. Absolute times from different machines aren't comparable.

:::{dropdown} Engine configuration
| Engine | Configuration |
|---|---|
| Batcher | Single node, in process |
| DuckDB (`duckdb`) | Its native store, ingested untimed before the query |
| DuckDB (`duckdb_arrow`) | The same zero-copy Arrow Batcher runs on. Opt in with `--engines batcher,duckdb,duckdb_arrow` |
| DuckDB, both bars | CPU and memory budget pinned to Batcher's (`benchmarks/engines/duckdb.py::match_batcher_budget`) |
| Polars | In process. Native lazy DataFrame pipelines on TPC-H |
| Daft | Native multithreaded engine (`DAFT_RUNNER=native`), or its Ray runner for the cluster grid |
| Ray Data | Tables written to Parquet once, untimed, then read back with Ray-sized row groups |
| Distributed engines | Attached to the live cluster with `ray.init(address="auto")` |

On TPC-H, Batcher runs DataFrame pipelines on 8 of the 22 queries ([`benchmarks/suites/standard/tpch_dataframe.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/suites/standard/tpch_dataframe.py)) and {py:func}`bt.sql <batcher.sql>` on the other 14. DuckDB and Spark run the SQL text throughout.
:::

:::{dropdown} What the harness prints beside best-of-N
The harness keeps every timed repetition ([`benchmarks/harness/timing.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/harness/timing.py)) and prints, per engine and case:

- `first_ms`, the first call, which is also the correctness run.
- `best_ms`, the headline figure.
- `median_ms` and `p95_ms` over the timed repetitions.
- `cpu_ms`, the CPU time of the best repetition in this process.

A cell with no timing says why: `n/a` (no form of the query), `OOM`, `ERR`, or `-` (not measured). `python benchmarks/run.py --repeat N` reruns a selection and reports the spread, and `--isolate` runs each case in its own subprocess. `run.py` names the mode in its header.
:::

## Suite coverage

`python benchmarks/run.py --list` prints the live count, which stands at 373 benchmarks across ten suites:

| Suite | Cases | What it covers |
|---|---:|---|
| Join Order Benchmark | 113 | Join planning over the real IMDb dataset, 3 to 16 tables per query |
| TPC-DS | 99 | The full official set, vendored from DuckDB's `tpcds` extension |
| Operators | 46 | The data-plane kernels in isolation |
| ClickBench | 43 | Wide-table scans and aggregates over web analytics data |
| Scan and I/O | 27 | Parquet, CSV, JSON and the connectors |
| TPC-H | 22 | The full official set, at scale factors 1 and 10 |
| H2O.ai db-benchmark, group-by | 10 | Grouping from 100 groups to 10M |
| H2O.ai db-benchmark, join | 5 | Five join shapes across small, medium and large build sides |
| Semi-structured JSON | 5 | Nested extraction and projection |
| Images | 3 | Decode to tensor |

:::{dropdown} Data sources
- TPC-H runs at scale factor 1 (`lineitem` holds 6,001,215 rows) and scale factor 10 (60M rows), from `s3://ray-benchmark-data/tpch/parquet/`.
- TPC-DS is generated locally by the `dsdgen` in DuckDB's `tpcds` extension.
- The Join Order Benchmark runs on the IMDb snapshot at `https://event.cwi.nl/da/job/imdb.tgz`, converted once to Parquet.
- The H2O.ai tables come from [`benchmarks/datagen/h2o_tables.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/datagen/h2o_tables.py) with a fixed seed of 108, so absolute times compare across engines in one run, not against the H2O.ai leaderboard.
:::

## Reproduce

[`benchmarks/run.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/run.py) drives every analytics suite. The following commands reproduce each family:

```bash
export BENCH_S3_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2
export DAFT_RUNNER=native

python benchmarks/run.py --benchmark tpch --engines batcher,duckdb,duckdb_arrow,polars
python benchmarks/run.py --benchmark operators --tier single
python benchmarks/run.py --benchmark tpcds --engines batcher,duckdb
python benchmarks/run.py --benchmark job --engines batcher,duckdb
python benchmarks/run.py --benchmark h2o-groupby
python benchmarks/run.py --benchmark h2o-join
python benchmarks/scenarios/image_decode.py
python benchmarks/scenarios/point_cloud_load.py
python benchmarks/cluster/vs_ray_daft.py 10
python benchmarks/scenarios/scaling/ladder.py --rungs 1,2,4
```

[`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md) is the complete record of every run, and [`benchmarks/results/`](https://github.com/stephenoffer/batcher/tree/main/benchmarks/results) holds the standalone boards.

## See also

- {doc}`/benchmarks/results/analytics` and {doc}`/benchmarks/results/ai-and-gpu`: the two halves of the measurement.
- {doc}`/architecture/internals/testing-strategy`: the same discipline applied to the engine.
- {doc}`/user-guide/operate/tuning/performance`: the levers you have on your own query.
