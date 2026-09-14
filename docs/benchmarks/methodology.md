# Methodology

How the numbers were produced, and how to reproduce them. Read it before quoting a ratio
off any other page in this section.

## Correctness gates the timer

:::{important}
The harness (`benchmarks/harness/`) runs each query on every engine and compares the results
as a sorted row multiset within float tolerance. A query whose result doesn't match reports
`FAILED`, and the ratio for the engine that disagreed is withheld and printed as `n/c`. A
fast wrong answer is a bug, not a win, and it never reaches these pages as a ratio.
:::

The engine that disagreed is still *timed*, and its milliseconds still appear in the table.
That is deliberate. How fast a wrong answer was is diagnostic, and hiding it would make a
failing engine indistinguishable from an absent one. The harness refuses only to *divide*
the two, because a ratio is the suite's claim about which engine is faster, and a number
carries its disqualification only as long as the status column travels beside it.

A result that asked for an order gets a second, separate check. The multiset comparison
sorts both sides before comparing, so on its own it cannot tell a sorted result from an
unsorted one. An engine that skipped its `ORDER BY` entirely would match. Every case
carrying an outermost `ORDER BY` is therefore also checked for monotonicity in its own
order, per engine, and an engine that fails it is disqualified the same way a wrong value
is.

### When the oracle is the one that is wrong

A difference from DuckDB is evidence about the *pair*, not about Batcher, and treating the
two as the same claim has a cost. Tightening a comparison to include column order produced
49 failures against Batcher for `SELECT * ... USING (k)`, where Batcher follows SQL:2016
§7.7 and DuckDB does not; and Batcher's `var_samp` over values near 2^53 is more accurate
than DuckDB's, not less. Both would read as Batcher failing.

So the suite records known semantic differences in `benchmarks/harness/divergences.py`, each
with a verdict naming which engine is right and a citation. A row whose every difference is
recorded reports **`DIVERGENT`**: it never reads `OK`, its ratio is still withheld, its
reason prints beneath the table, and it does not fail the run. An unrecorded difference is
still `FAILED`. No entry can excuse a Batcher defect: entries name the engine that is the
odd one out, so a Batcher error on the same query is unaffected.

This is the same discipline the engine is built under: every relational operator is
differential-tested against DuckDB, and the Tier-0 interpreter is the oracle that the
parallel executor and the JIT must agree with bit-for-bit.

It also means the benchmark occasionally reports on *other* engines' correctness. On TPC-H
q6, both Daft and Polars return the wrong revenue, having folded `0.06 + 0.01` in IEEE double to `0.06999999999999999`, which drops every `l_discount = 0.07` row.
Their q6 row reports `FAILED` and carries no ratio, so neither is credited with a fast wrong answer.

## Hardware

Different workload families were measured on different machines, because a GPU benchmark
needs GPUs and a distributed benchmark needs a cluster. Each is labeled where it appears:

| Family | Hardware |
|---|---|
| Suite geomeans (TPC-H, TPC-DS, ClickBench, JOB, H2O, operators, JSON), 2026-08-15 and 2026-08-25 | Single node, 96 cores, 184 GiB |
| The full engine matrix and the Spark and PyArrow boards, 2026-09-11 | Single node, 48-core Xeon Platinum 8275CL, 92 GiB |
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
look bad. Every engine is handed the same source data, loaded once into Arrow, so no engine
is timed on a different *dataset*. Two engines do not execute over that Arrow directly, and
the table below says which and why: DuckDB's default bar ingests it into its own native
store first, and Ray Data reads it back through Parquet because a single in-memory Arrow
table becomes one Ray block, which would pin it to one core. Both are untimed setup, both
are the representative way that engine is run, and both are why the like-for-like
`duckdb_arrow` bar exists alongside the native one. Timings are best-of-N warm.

| Engine | Configuration | Why |
|---|---|---|
| Batcher | Single-node, in-process | Its low-overhead strength |
| DuckDB (`duckdb`) | Its **native** store, ingested untimed before the query | DuckDB at its best, and the harder bar |
| DuckDB (`duckdb_arrow`) | The **same zero-copy Arrow** Batcher runs on | The like-for-like execution comparison |
| Daft | Native multithreaded local engine (`DAFT_RUNNER=native`), or its Ray runner for the cluster grid | Its fastest runner for each shape |
| DuckDB, Polars | In-process | The only way they run |
| Ray Data | Tables written to Parquet once, untimed, then read back with Ray-sized row groups | `from_arrow` makes one block, and a block is Ray's unit of parallelism, so a one-block dataset runs every operator on a single core. The cost is that Ray decodes Parquet inside each timed run while the in-process engines read Arrow; the alternative was measuring Ray single-threaded |
| DuckDB (both bars) | CPU and memory budget **pinned to Batcher's** | Left to their defaults DuckDB takes 80% of RAM and Batcher takes 90% of the whole machine: 147.1 against 165.6 GiB here, a 13% headroom advantage to Batcher before it spills. That decides nothing at sf1 and decides whether a query spills at all above 10M rows, which is the regime the project concedes it loses in. Threads are pinned for a second reason: the two agree at 92 on this box by separate auto-detections, and parity that holds by coincidence is parity nobody notices losing |
| Distributed engines | Attached to the live cluster (`ray.init(address="auto")`) | Where they are designed to be strongest |

### Which surface each engine runs

TPC-H is the one suite where this is not uniform, and it matters for reading its geomean.
Ray Data has no SQL surface and runs hand-written `ray.data.Dataset` pipelines; Polars runs
native lazy-DataFrame pipelines, as its own published TPC-H benchmark does; Batcher runs
DataFrame pipelines on 8 of the 22 queries and `bt.sql()` on the other 14; DuckDB and Spark
run the SQL string throughout. The planned operator sequences of Batcher's two paths were
diffed at sf1-proportional cardinalities and agree exactly on q1/q3/q5/q6, the
six-table join included, differing elsewhere only in the placement of one or two projection and
scan-pushdown nodes. The hand-written pipelines are not out-planning the SQL front-end.

## How much precision a geomean is entitled to

Every headline figure here is a geometric mean of per-query `batcher_ms / engine_ms` ratios.
That is a ratio of *times*, so lower is better and anything below 1.00 is a Batcher win.
Every table on this site states the convention again in its own lead-in, because the
inverted form reads identically and a reader who guesses wrong reads every row backwards.

Two things decide whether one geomean is comparable to another, and both are printed with it:
the **number of cases it averages** and the exclusions by status, since a mean over an
unstated denominator is not comparable to a different one; and the **best-of-N** the run
used, which `run.py` varies by scale.

A third thing decides how many digits it is entitled to, and it is not visible in a single
run at all. The operator mix measures at a **4.1% spread** across three runs on one machine
(0.619 / 0.598 / 0.594), with one case varying 1.78x. At that spread a figure quoted to
three decimals asserts a stability nobody measured, and two figures agreeing to 0.001 is a
coincidence rather than evidence that two boards agree. `python benchmarks/run.py --repeat N`
re-runs a selection and reports min / median / max and the spread; quote the median, to a
precision the spread supports.

### Three asymmetries that remain, stated rather than removed

**Planning.** Batcher's `Session.sql` caches the parsed AST, the optimized plan and the
prepared physical plan, so across `bench()`'s warm-up and repeats it plans once. DuckDB's
`con.sql(query)` re-parses and re-plans on every call, because that is the API a DuckDB user
writes. Measured on TPC-H sf1, giving DuckDB `PREPARE`d statements moves the geomean from
**0.764 to 0.783, 2.3% against Batcher**, with 15 of 22 queries moving against it and
per-query planning running 8–20% on the plan-heavy shapes. It is left as it is because both
sides are the ordinary way each engine is used, and it is recorded because 2.3% is real even
though it sits below this suite's own 2.9% run-to-run spread.

**Output format.** Batcher returns Arrow natively; DuckDB converts to it. Two attempts to
isolate that conversion here disagreed in sign, so **no figure is quoted for it**. It is
named as a known asymmetry of unmeasured size rather than estimated.

**Warm-up.** `bench()` discards one execution and reports the best of the next N, so every
figure on the board is a *steady-state* one. That is the standard way to benchmark. The
asymmetry is not the warm-up itself, it is the price of the discarded execution, and the two
engines do not pay the same one: on TPC-H sf1 a first-seen query costs Batcher **2.60x** its steady state
against DuckDB's **1.15x** (`benchmarks/scenarios/claims/learning_curve.py`, geomean over 22
queries, 8 executions each in a fresh process). DuckDB's 1.15x is the page-cache, JIT and
allocator floor both engines pay; the **2.27x excess** is Batcher's plan cache and learned
store filling up, and it is gone by the second or third execution, which is why best-of-N
lands both engines in steady state and the comparison stays fair.

It is recorded here because it is by far the largest of the three, and because of what it
implies about scope rather than about bias: a user who runs a query **once**, the most common
thing a user does, sees a number this board never prints, and the gap between the board and
that number is engine-specific and roughly 2.3x, not the 2.3% of the planning row above. The
board answers "how fast is this engine on a query it has seen before". `cold_start.py` and
`learning_curve.py` are where the other question is answered, and neither feeds a headline
ratio.

One honest limit, inherited from the experiment: run 1 to run 2 folds the plan cache and the
learned store together and does not separate them.

**Process isolation.** `run.py --isolate` runs each case in its own subprocess, which the
harness needs whenever a query takes the process down rather than raising. It is not a
neutral packaging choice. On TPC-H sf1 over four alternated passes the suite geomean reads
**0.725 isolated against 0.693 in-process**, a 4.4% difference from a flag that changes
nothing about the queries, against a pass-to-pass spread inside each mode of under 1%. Three
quarters of it lands on the comparator, which is misleading about the cause. Run **alone**,
DuckDB does not care which mode it is in (+1.3%, per-query signs 11 of 22, a coin flip)
while Batcher gains **3.8%** from the shared process (15 of 22 queries faster). The capacity
to benefit from one process is Batcher's, because cross-query carry-over is exactly what this
engine has and DuckDB does not; in the paired lineup part of that gain is spent being crowded
by a co-resident DuckDB, so it shows up on DuckDB's side of the table instead. **The
shared-process mode credits Batcher's cross-query carry-over into the headline ratio**, and
it is the mode TPC-H, TPC-DS and ClickBench are published from, while JOB is published
isolated.

Two consequences. A figure is only comparable to another figure taken the same way, so
`run.py` now names the mode in its header rather than printing an identical table for both.
And `--isolate` is **not** a cold-start measurement, which a note in
`benchmarks/BENCHMARK_RESULTS.md` claimed until it was checked: the child still executes the
query once for the correctness check, once as a warm-up, and N more times reporting only the
best, so the plan cache and the learned store are warm when the number is taken. If it were
the cold-start case Batcher would read ~2.6x its steady state. It reads 1.012x. The flag
removes cross-*query* carry-over and nothing else.

## Suite coverage

`python benchmarks/run.py --list` prints the live count. It stands at **373 benchmarks across
ten suites**, spanning the industry-standard analytics set and the workload families that are
specific to Batcher's range:

| Suite | Cases | What it covers |
|---|---:|---|
| Join Order Benchmark | 113 | Join planning against the real IMDb dataset, 21 tables |
| TPC-DS | 99 | The full official set, vendored from DuckDB's `tpcds` extension |
| Operators | 46 | The data-plane kernel lineup in isolation |
| ClickBench | 43 | Wide-table scan and aggregate on web analytics data |
| Scan and I/O | 27 | Parquet, CSV, JSON, and the connectors |
| TPC-H | 22 | The full official set, at scale factors 1 and 10 |
| H2O.ai db-benchmark, group-by | 10 | Grouping from 100 groups to 10M, the standard cardinality sweep |
| H2O.ai db-benchmark, join | 5 | Five join shapes across small, medium, and large build sides |
| Semi-structured JSON | 5 | Nested extraction and projection |
| Images | 3 | Decode to tensor |

Registered is not the same as timed, and two suites publish a smaller denominator than the
column above. TPC-DS times **98 of its 99**: q67 fails the correctness gate on both engines,
because float reassociation moves group sums in their last bits, which changes which sums tie,
which moves an integer `rank()`. The Join Order Benchmark times **109 of its 113**; the other
four do not clear the gate. Neither denominator is rounded up, and every geomean on this site
prints the count it averaged.

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

# does N times the cluster give N times the throughput?
python benchmarks/scenarios/scaling/ladder.py --rungs 1,2,4
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
