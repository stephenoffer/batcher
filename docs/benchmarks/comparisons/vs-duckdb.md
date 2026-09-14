# vs DuckDB

This page compares Batcher against DuckDB on single-node analytics: the operator shapes, the query suites, and the architecture behind the results.

DuckDB is the single-node analytical engine to beat. On identical Arrow input Batcher beats it on every TPC-H and every ClickBench query. Against DuckDB's own native compressed store, which is the harder bar, Batcher leads TPC-H, ClickBench, JSON and the operator mix at scale factor 1.

:::{important}
Every number below was produced by a run that had to pass the correctness gate first: the
harness compares the two engines' results as a sorted row multiset within float tolerance
and refuses to record a time when they disagree. Batcher matches DuckDB's result on all 22
TPC-H queries, so nothing here is a correctness argument. It is a speed argument.
:::

## Scorecard

Every ratio on this page, this table included, is `batcher_ms / duckdb_ms`. **Lower is
better, and anything below 1.00 is a Batcher win.** Each row is one workload shape, and the
table further down that produced it says on what hardware:

| Shape | `batcher / duckdb` |
|---|---:|
| Global aggregate, filtered count | **0.19x to 0.20x** |
| Group-by (one or two keys) | **0.68x to 0.76x** |
| Window running `sum()` | **0.71x** |
| `MEDIAN` / `QUANTILE_CONT` per group | **0.91x to 0.92x** |
| Delta file skipping (`count(*)` with a predicate) | **0.34x** |
| TPC-H sf1, DuckDB reading the same Arrow | **0.26x**, wins all 22 |
| ClickBench (43), DuckDB reading the same Arrow | **0.07x**, wins all 43 |
| TPC-H sf1, DuckDB on its native store | **0.79x**, wins 16 of 22 |
| TPC-H sf10, DuckDB on its native store | **0.963x**, a win as of 2026-08-25 |
| H2O.ai `groupby` (10), DuckDB on its native store | 1.19x, wins 4 of 10 |
| Join Order Benchmark (113), DuckDB on its native store | 1.29x, wins 35 of 109 |

## Operators

Single node, 16 cores, 30 GB, `python benchmarks/run.py --benchmark operators --tier single`.
TPC-H `lineitem` at scale factor 1 (6,001,215 rows) held in Arrow and shared byte-identically
between the engines. The ratio is `batcher / duckdb`, so **below 1.0 means Batcher is
faster**.

| Operator | Batcher | DuckDB | vs DuckDB |
|---|---:|---:|---:|
| global sum | 0.5 ms | 2.7 ms | **0.19x** |
| filter → count | 0.6 ms | 2.7 ms | **0.20x** |
| group-by, two keys | 11.6 ms | 16.9 ms | **0.68x** |
| window running `sum()` | 171 ms | 240 ms | **0.71x** |
| group-by sum, one key | 7.6 ms | 10.0 ms | **0.76x** |
| window `sum()` over partition | 92.7 ms | 99.9 ms | **0.93x** |

The filtered count is the widest margin, and it is not a micro-optimization. `.count()`
over a filter compiles to a `COUNT(*)` aggregate, so projection pushdown prunes the scan
to the one column the predicate touches and the count fuses into a single {py:func}`count_if <batcher.count_if>` pass.
Nothing else is read, and no matching row is ever materialized.

:::{note}
`join → aggregate` has moved since this table was published. `BroadcastProbe::probe` was building a full 16,384-entry null mask per morsel and reading it per row, for a foreign-key probe whose key is never null. Skipping both when the probe key has no nulls took the operator to **0.90x to 0.97x**, verified bit-identical across the 84 join and stream oracle tests. `benchmarks/BENCHMARK_RESULTS.md` carries the measurement.
:::

:::{tip}
The same margins are reachable from your own query. {py:meth}`ds.explain() <batcher.Dataset.explain>` shows whether the
predicate reached the scan and which columns survived pruning; {py:meth}`ds.stats() <batcher.Dataset.stats>` reports what
each operator actually cost. {doc}`/getting-started/tutorials/foundations/optimizing-a-slow-query` walks the loop.
:::

## Exact aggregates

Batcher takes the exact order-statistic aggregates, on algorithms rather than tuning.
16 cores, same fixture:

| Query | Batcher | DuckDB |
|---|---:|---:|
| `MEDIAN(x) GROUP BY flag` (5M rows, 3 groups) | **210 ms** | 232 ms |
| `QUANTILE_CONT(x, 0.9) GROUP BY flag` (5M rows) | **208 ms** | 226 ms |
| `COUNT(DISTINCT id) GROUP BY flag` (2M rows) | **163 ms** | 181 ms |

Median and quantile need the value at one rank, not a sorted list, so the finalize does
quickselect instead of a full sort and runs each group's selection on its own core. The
count-distinct is a Kyber rewrite: a lone `COUNT(DISTINCT x) GROUP BY g` becomes a
distinct over `(g, x)` followed by a count, which parallelizes across the distinct values
instead of the handful of groups.

## Two bars, and both are published

DuckDB can be measured two ways, and the difference between them is not a detail:

`duckdb_arrow`
    DuckDB executing over the *same zero-copy Arrow* Batcher runs on. This is the
    like-for-like comparison of two execution engines, and the one Batcher's Arrow-only
    contract makes fair.
`duckdb`
    DuckDB over its own native store, ingested before the clock starts: compressed,
    dictionary-encoded, zone-mapped. This measures DuckDB's *storage engine plus* its
    execution engine against Batcher's execution engine alone. It is DuckDB at its best,
    and it is the harder bar.

Both are reported, because quoting only the first would be choosing the flattering one.
Suite geometric means of per-query `batcher / duckdb` ratios, 96 cores / 184 GiB, scale
factor 1, measured 2026-08-15. **Below 1.0 means Batcher is faster**, and the count beside
each ratio is queries won:

| Suite | vs `duckdb` (native store) | vs `duckdb_arrow` (same Arrow) |
|---|---:|---:|
| Semi-structured JSON (5) | **0.25x**, 5 of 5 | **0.04x**, 5 of 5 |
| ClickBench (43) | **0.64x**, 28 of 43 | **0.07x**, 43 of 43 |
| Operator mix (19) | **0.66x**, 11 of 19 | **0.36x**, 15 of 19 |
| TPC-H (22) | **0.79x**, 16 of 22 | **0.26x**, 22 of 22 |
| H2O.ai `join` (5) | **0.93x**, 3 of 5 | **0.24x**, 5 of 5 |
| TPC-DS (98 of 99 timed) | **0.96x**, 38 of 98 | — |
| H2O.ai `groupby` (10) | 1.19x, 4 of 10 | **0.09x**, 10 of 10 |
| Join Order Benchmark (113) | 1.29x, 35 of 109 | — |

The `—` in the right-hand column means no figure exists, not a tie. DuckDB over registered
Arrow views is killed on TPC-DS q64, and on the Join Order Benchmark its planner has no
storage statistics to order a many-way join with, so the comparison would measure a
handicapped planner rather than an executor.

TPC-DS reports 98 rather than 99 because q67 fails the correctness gate on both engines.
Float reassociation moves group sums in their last bits, which changes which sums tie, which
moves an integer `rank()`. Neither engine is deterministic there, so the query is excluded
rather than scored. The Join Order Benchmark times 109 of its 113; the other four do not
clear the gate either, and the record does not attribute them to one cause.

Every suite in the right-hand column is a Batcher win. Read that column for a question about
*engines*. The left-hand one answers a question about engines **and** storage formats
together.

On identical input Batcher's execution engine is **3.9x DuckDB's on TPC-H and 14x on
ClickBench**, and it wins every query of both. Against DuckDB's native store the margin
narrows to 1.3x and 1.6x, because that comparison puts a storage engine and an execution
engine together against an execution engine alone.

The suite's residue is concentrated in its two single-key high-cardinality queries, one
string column with 100,000 distinct values. A *composite* string key no longer pays for being
strings. Each column's distinct values are numbered in first-seen order and the ranked columns
take the ordinary integer grouper, which brought a two-string-key group-by over 10M rows from
41.7 ms to 32.6 ms. That is identical to the same query with two `int64` keys, which is the
check that nothing about the strings is left to pay for.

:::{warning}
Five of the 43 ClickBench queries and two of the 19 operator cases are answered from
Batcher's recorded column statistics rather than executed: an unfiltered `SUM`, `AVG` or
`COUNT(DISTINCT)` over an immutable in-memory relation. The answers are exact, but the timing
is a memo lookup rather than a scan. **Excluding them**, ClickBench is **0.77x over 38
queries** and the operator mix **0.76x over 17**. Quote those when the claim is about
execution speed.
:::


## Lakehouse reads

A selective predicate on a Delta table should open one data file, not all of them. The
transaction log records each file's column bounds, and reading it at plan time is the whole
game. `python benchmarks/scenarios/lakehouse_bench.py`, single node, 10M rows across 200
Delta data files with one `day` per file, measured 2026-07-13:

| `count(*) WHERE day = 42` | Time | Files opened |
|---|---:|---:|
| **Batcher** | **7.4 ms** | **1** |
| Batcher, before file skipping | 98.8 ms | 200 |
| DuckDB `delta_scan` | 21.8 ms | not reported |

Batcher is 2.9x faster than `delta_scan` here, having been 2.7x slower before the log was
consulted at plan time, and it opens one file where the predicate selects one file. An
unfiltered `count(*)` is 0.85 ms and opens nothing, answered from the log. The record does
not name the machine for this run, so read the ratios and not the absolute times.

Predicates are recovered from the user's plan, where a `Filter` on a `Scan` constrains that
scan whatever the optimizer does above it. The most ordinary lakehouse query in existence
therefore pushes down and skips the rest of the table.

## What DuckDB cannot do

The gap that matters most doesn't appear on this page as a number. DuckDB is single-node and its optimizer is static. It commits to a plan before the first row is read and can't change its mind.

Batcher re-optimizes at stage boundaries on measured cardinalities. That is the same granularity Spark AQE works at, available single-node too, and it carries a sketch-backed cross-query learned-stats loop that DuckDB has no equivalent for. The same mergeable operators then run across a cluster and return the same rows. Float reductions agree to the last bits rather than to every bit, because the partition count sets the summation order.

The loop is not free and it is not always on. It engages once a joined query clears 5M rows, or roughly 320 MB, for each pipeline breaker it would cut at, so the simplest joined shape qualifies at about 10M rows and a query with no join never qualifies at any size. See {doc}`/benchmarks/results/scaling`.

## Reproduce

```bash
python benchmarks/run.py --benchmark operators --tier single --scale 1
python benchmarks/run.py --benchmark tpch      --tier single --scale 1
python benchmarks/scenarios/lakehouse_bench.py
```

## See also

- {doc}`/benchmarks/results/tpch` for the per-query breakdown.
- {doc}`/benchmarks/results/analytics` for operators, connectors, and the lazy control plane.
- {doc}`/benchmarks/comparisons/vs-polars` and {doc}`/benchmarks/comparisons/vs-daft` for the other two single-node engines.
- {doc}`/architecture/deep-dives/operators/aggregation-internals` for the quickselect finalize behind the median and quantile wins.
- {doc}`/architecture/deep-dives/operators/join-algorithms` for the join strategies behind the TPC-H results.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization` for what a static optimizer can't do.
- {doc}`/benchmarks/methodology` for hardware, gating, and why cross-hardware comparison is meaningless.
