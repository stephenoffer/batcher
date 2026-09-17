# Methodology

This page describes how the benchmark numbers on this site are produced and how to reproduce them. Read it before quoting a ratio from any other page in this section.

## Correctness gates the timer

:::{important}
The harness in `benchmarks/harness/` runs each query on every engine and compares the results as a sorted row multiset within float tolerance. A query whose result doesn't match reports `FAILED`, and the ratio for the engine that disagreed is withheld and printed as `n/c`. A fast wrong answer never reaches these pages as a ratio.
:::

The engine that disagreed is still timed, and its milliseconds still appear in the harness output. How fast a wrong answer was is diagnostic, and hiding it would make a failing engine look the same as an absent one. The harness only refuses to divide the two times, because a ratio is a claim about which engine is faster.

A result that asks for an order gets a second check. The multiset comparison sorts both sides first, so on its own it can't tell a sorted result from an unsorted one, and an engine that skipped its `ORDER BY` would pass. Every case with an outermost `ORDER BY` is therefore also checked for monotonicity in its own order, per engine (`benchmarks/harness/order.py`), and an engine that fails that check is disqualified the same way a wrong value is.

This is the discipline the engine itself is built under. Every relational operator is differential-tested against DuckDB, and the Tier-0 interpreter is the oracle that the parallel executor and the JIT must match bit for bit. {doc}`/architecture/internals/testing-strategy` covers that side.

### When the oracle is wrong

A difference from DuckDB is evidence about the pair of engines, not a verdict on Batcher. Tightening the comparison to include column order once produced 49 failures on `SELECT * ... USING (k)`, where Batcher follows SQL:2016 section 7.7 and DuckDB does not. Batcher's `var_samp` over values near 2^53 is also more accurate than DuckDB's. Both would have read as Batcher failing.

So the suite records known semantic differences in `benchmarks/harness/divergences.py`, each with a verdict naming which engine is right and a citation. A case whose every difference is recorded reports `DIVERGENT`. It never reads `OK`, its ratio is still withheld, its reason prints beneath the table, and it doesn't fail the run. An unrecorded difference is still `FAILED`. An entry can't excuse a Batcher defect, because each entry names the engine that is the odd one out.

The gate also reports on other engines. On TPC-H q6, Daft and the Polars SQL frontend fold `0.06 + 0.01` in IEEE double to `0.06999999999999999`, which drops every `l_discount = 0.07` row. Their q6 result reports `FAILED` and carries no ratio, so neither is credited with a fast wrong answer.

## Hardware

Different workload families ran on different machines, because a GPU benchmark needs GPUs and a distributed benchmark needs a cluster. Every table on this site names its machine. The following table lists them by family:

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

:::{note}
A number from one row set against a number from another row means nothing. A 16-core operator timing and an 8xT4 throughput figure describe different machines running different work. The ratios within a table are the claim, and the absolute times are context for it.
:::

## Engine configuration

Each engine runs in the configuration its users run, not one chosen to make it look bad. Every engine is handed the same source data, loaded once into Arrow, so none is timed on a different dataset. Timings are best of N, warm.

Two engines don't execute over that Arrow directly, and the table below says why. DuckDB's default bar ingests it into DuckDB's native store first. Ray Data reads it back through Parquet, because a single in-memory Arrow table becomes one Ray block, which would pin every operator to one core. Both steps are untimed setup, and the first is why the like-for-like `duckdb_arrow` bar exists beside the native one.

The following table describes how each engine is configured:

| Engine | Configuration | Why |
|---|---|---|
| Batcher | Single node, in process | Its low-overhead mode |
| DuckDB (`duckdb`) | Its native store, ingested untimed before the query | DuckDB at its best, and the harder bar |
| DuckDB (`duckdb_arrow`) | The same zero-copy Arrow Batcher runs on | The like-for-like execution comparison |
| DuckDB, both bars | CPU and memory budget pinned to Batcher's | Left to defaults, DuckDB takes 80% of RAM and Batcher 90%: 147.1 against 165.6 GiB on the 184 GiB box, a 13% headroom advantage to Batcher before it spills |
| Polars | In process | The only way it runs |
| Daft | Native multithreaded engine (`DAFT_RUNNER=native`), or its Ray runner for the cluster grid | Its fastest runner for each shape |
| Ray Data | Tables written to Parquet once, untimed, then read back with Ray-sized row groups | A one-block dataset runs every operator on a single core |
| Distributed engines | Attached to the live cluster with `ray.init(address="auto")` | Where they're designed to be strongest |

`duckdb_arrow` isn't in the default lineup. Pass it explicitly with `--engines batcher,duckdb,duckdb_arrow`, and `benchmarks/engines/lineup.py` records why it is opt-in. The memory pinning lives in `benchmarks/engines/duckdb.py::match_batcher_budget`.

### Which surface each engine runs

TPC-H is the one suite where engines run different surfaces. Ray Data has no SQL surface and runs hand-written `ray.data.Dataset` pipelines. Polars runs native lazy DataFrame pipelines, as its own published TPC-H benchmark does, because its SQL frontend can't parse the suite. Batcher runs DataFrame pipelines on 8 of the 22 queries (`benchmarks/suites/standard/tpch_dataframe.py`) and {py:func}`bt.sql <batcher.sql>` on the other 14. DuckDB and Spark run the SQL text throughout.

Batcher's two paths plan the same way. Their operator sequences were diffed at sf1-proportional cardinalities and agree exactly on q1, q3, q5 and q6, the six-table join included, and elsewhere differ only in where one or two projection and scan-pushdown nodes sit.

## How much precision a geomean has

Every headline figure is a geometric mean of per-query `batcher_ms / engine_ms` ratios. That is a ratio of times, so lower is better and anything below 1.00 is a Batcher win. Each table on this site repeats the convention in its lead-in, because the inverted form reads identically.

Two things decide whether one geomean is comparable to another, and both are printed with it: the number of cases it averages, and the best-of-N the run used, which `run.py` varies by scale.

A third decides how many digits it is entitled to. The operator mix measured at a 4.1% spread across three runs on one machine (0.619, 0.598 and 0.594), with one case varying 1.78x. A single case on the 2026-09-13 board has a median spread of 11% across runs, against 1.4% to 2.3% for a suite geomean. So a figure quoted to three decimals asserts a stability nobody measured. `python benchmarks/run.py --repeat N` reruns a selection and reports the minimum, median, maximum and spread. Quote the median, to the precision the spread supports.

### Three asymmetries, stated rather than removed

**Warm-up.** `bench()` discards one execution and reports the best of the next N, so every figure is steady state. The engines don't pay the same price for the discarded run. On TPC-H sf1 a first-seen query costs Batcher 2.60x its steady state against DuckDB's 1.15x (`benchmarks/scenarios/claims/learning_curve.py`, geomean over 22 queries, 8 executions each in a fresh process). DuckDB's 1.15x is the page-cache, JIT and allocator floor both engines pay. The 2.27x excess is Batcher's plan cache and learned statistics filling, and it is gone by the second or third execution.

This is the largest of the three. A user who runs a query once sees a number the board never prints, and for Batcher the gap is about 2.3x. The board answers how fast an engine is on a query it has seen before, and `benchmarks/scenarios/claims/cold_start.py` and `learning_curve.py` answer the other question. Neither feeds a headline ratio, and run 1 to run 2 folds the plan cache and the learned statistics together without separating them.

**Planning.** Batcher's {py:class}`Session <batcher.Session>` caches the parsed statement, the optimized plan and the prepared physical plan, so across warm-up and repeats it plans once. DuckDB's `con.sql(query)` re-parses and re-plans on every call. Both are the ordinary way each engine is used, so the asymmetry is left in place.

**Output format.** Batcher returns Arrow natively, and DuckDB converts to it. Two attempts to isolate that conversion disagreed in sign, so no figure is quoted.

### Process isolation

`run.py --isolate` runs each case in its own subprocess, which the harness needs when a query takes the process down rather than raising. It changes the numbers. On TPC-H sf1 over four alternated passes, the suite geomean reads 0.725 isolated against 0.693 in process, a 4.4% difference from a flag that changes nothing about the queries, against a pass-to-pass spread under 1%.

The benefit of one shared process is Batcher's. Run alone, DuckDB barely cares which mode it is in (1.3%, with per-query signs split 11 to 11) while Batcher gains 3.8%, because cross-query carry-over is exactly what Batcher has and DuckDB doesn't. So the shared-process mode credits that carry-over into the headline ratio. A figure is only comparable to one taken the same way, and `run.py` names the mode in its header. The 2026-09-13 board runs one process per case.

`--isolate` isn't a cold-start measurement. The child still executes the query once for the correctness check, once as a warm-up and N more times, so the plan cache and learned statistics are warm when the number is taken. Isolated, Batcher reads about 1.01x its in-process time, not the 2.6x of a first run.

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

Registered isn't the same as timed. In the 2026-08 sweeps TPC-DS timed 98 of its 99, because q67 failed the gate on both engines: float reassociation moves group sums in their last bits, which changes which sums tie, which moves an integer `rank()`. The Join Order Benchmark timed 109 of its 113. Every geomean on this site prints the count it averaged.

TPC-DS and the Join Order Benchmark exercise planning far harder than TPC-H does. Every Join Order Benchmark query is a 3-way to 16-way join, which is the regime where join ordering and cardinality estimation decide the runtime rather than the kernels.

## Data

TPC-H runs at scale factor 1, where `lineitem` holds 6,001,215 rows, and at scale factor 10 (60M rows) where noted, from `s3://ray-benchmark-data/tpch/parquet/`. TPC-DS is generated locally by the `dsdgen` in DuckDB's `tpcds` extension. The Join Order Benchmark runs on the IMDb snapshot at `https://event.cwi.nl/da/job/imdb.tgz`, converted once to Parquet. The H2O.ai tables come from `benchmarks/datagen/h2o_tables.py`, which reproduces the reference R generators' cardinalities and value ranges with a fixed seed of 108. The draws differ from the published CSVs, so absolute times compare across engines in one run and not against the H2O.ai leaderboard.

## Reproducing

`benchmarks/run.py` drives every analytics suite. The following commands reproduce each family:

:::{dropdown} Every command, by workload family
```bash
export BENCH_S3_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2
export DAFT_RUNNER=native

# analytics, both DuckDB bars
python benchmarks/run.py --benchmark tpch --engines batcher,duckdb,duckdb_arrow,polars
python benchmarks/run.py --benchmark operators --tier single

# the planning-heavy suites
python benchmarks/run.py --benchmark tpcds --engines batcher,duckdb
python benchmarks/run.py --benchmark job --engines batcher,duckdb
python benchmarks/run.py --benchmark h2o-groupby
python benchmarks/run.py --benchmark h2o-join

# multimodal ingest
python benchmarks/scenarios/image_decode.py
python benchmarks/scenarios/point_cloud_load.py

# distributed Batcher against Daft on a live cluster, one scale factor per run
python benchmarks/cluster/vs_ray_daft.py 10

# does N times the cluster give N times the throughput?
python benchmarks/scenarios/scaling/ladder.py --rungs 1,2,4
```
:::

`--skip SUBSTRING` drops matching cases from a run and reports what it dropped. It exists for a query that kills the process, never for hiding a wrong answer.

## The full log

`benchmarks/BENCHMARK_RESULTS.md` is the complete engineering record, and it is what makes the numbers on these pages auditable. Each entry names the change, the measurement method and the result, from the JSON writer that went from over 65 seconds to about one to the image pipeline that five fixes took from 350 img/s to 5,693. `benchmarks/results/` holds the standalone boards, such as `TPCH_SF1_SF10_RESULTS.md` and `LOSS_BACKLOG.md`.

## See also

- {doc}`/benchmarks/results/tpch`: the suite where the gate has the most to say about other engines.
- {doc}`/benchmarks/results/analytics` and {doc}`/benchmarks/results/ai-and-gpu`: the two halves of the measurement.
- {doc}`/architecture/internals/testing-strategy`: the same discipline applied to the engine, with DuckDB as the differential oracle.
- {doc}`/user-guide/operate/tuning/performance`: the levers you have on your own query.
