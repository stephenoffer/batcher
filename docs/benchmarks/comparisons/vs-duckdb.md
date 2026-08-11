# vs DuckDB

This page compares Batcher against DuckDB on single-node analytics: the operator shapes, the query suites, and the architecture behind the results.

DuckDB is the single-node analytical engine to beat, and on identical Arrow input Batcher beats it: 22 of 22 TPC-H queries at sf1, 21 of 22 at sf10, and the operator mix.

:::{important}
Every number below was produced by a run that had to pass the correctness gate first: the
harness compares the two engines' results as a sorted row multiset within float tolerance
and refuses to record a time when they disagree. Batcher matches DuckDB's result on all 22
TPC-H queries, so nothing here is a correctness argument. It is a speed argument.
:::

## Scorecard

Each row is one workload shape, and by how much Batcher takes it. The methodology above is
what makes these numbers comparable, so read them together:

| Shape | Batcher's margin |
|---|---|
| Global aggregate, filtered count | 5× |
| Group-by (one or two keys) | 1.3x to 1.5x |
| Window running `sum()` | 1.4× |
| `MEDIAN` / `QUANTILE_CONT` per group | 1.1× |
| TPC-H overall (sf1), DuckDB reading the same Arrow | All 22 queries |
| TPC-H overall (sf10), DuckDB reading the same Arrow | 1.89×; wins 21 of 22 |
| Delta file skipping (`count(*)` with a predicate) | 1.42× |

## Operators

Single node, 16 cores, 30 GB. TPC-H `lineitem` at scale factor 1 (6,001,215 rows) held
in Arrow and shared byte-identically between the engines. The ratio is
`batcher / duckdb`, so **below 1.0 means Batcher is faster**.

| Operator | Batcher | DuckDB | vs DuckDB |
|---|---:|---:|---:|
| global sum | 0.5 ms | 2.7 ms | **0.19×** |
| filter → count | 0.6 ms | 2.7 ms | **0.20×** |
| group-by, two keys | 11.6 ms | 16.9 ms | **0.68×** |
| window running `sum()` | 171 ms | 240 ms | **0.71×** |
| group-by sum, one key | 7.6 ms | 10.0 ms | **0.76×** |
| window `sum()` over partition | 92.7 ms | 99.9 ms | **0.93×** |

The filtered count is the widest margin, and it is not a micro-optimization: `.count()`
over a filter compiles to a `COUNT(*)` aggregate, so projection pushdown prunes the scan
to the one column the predicate touches and the count fuses into a single {py:func}`count_if <batcher.count_if>` pass.
Nothing else is read, and no matching row is ever materialized.

:::{note}
`join → aggregate` has moved since this table was published. `BroadcastProbe::probe` was building a full 16,384-entry null mask per morsel and reading it per row, for a foreign-key probe whose key is never null. Skipping both when the probe key has no nulls took the operator to **0.90x to 0.97x**, verified bit-identical across the 84 join and stream oracle tests. `benchmarks/BENCHMARK_RESULTS.md` carries the measurement.
:::

:::{tip}
The same margins are reachable from your own query. {py:meth}`ds.explain() <batcher.Dataset.explain>` shows whether the
predicate reached the scan and which columns survived pruning; {py:meth}`ds.stats() <batcher.Dataset.stats>` reports what
each operator actually cost. {doc}`/tutorials/foundations/optimizing-a-slow-query` walks the loop.
:::

## Exact aggregates

Batcher takes the exact order-statistic aggregates, on algorithms rather than tuning.
16 cores.

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

## TPC-H

Against DuckDB reading the same Arrow, **Batcher wins all 22 queries**, by 1.1x to 7.1x. That is the like-for-like execution comparison, and it is the one Batcher's Arrow-only contract makes fair.

At scale factor 1 on 16 cores, re-measured 2026-08-02, the per-query geometric mean against DuckDB's native compressed store is **0.99x**. That comparison measures a storage engine plus an execution engine against an execution engine alone, because DuckDB decompresses its own format as it scans and never pays an Arrow ingest. {doc}`/benchmarks/results/tpch` has the per-query detail.

## Lakehouse reads

A selective predicate on a Delta table should open one data file, not all of them. The
transaction log records each file's column bounds, and reading it at plan time is the whole
game. Single node, 10M rows across 200 Delta files, one `day` per file:

| `count(*) WHERE day = 42` | Time | Files opened |
|---|---:|---:|
| **Batcher** | **13.4 ms** | **1** |
| DuckDB `delta_scan` | 19.0 ms | baseline |

Batcher is 1.42× faster here, and it opens one file where the predicate selects one file.
Predicates are recovered from the user's plan, where a `Filter` on a `Scan` constrains that
scan whatever the optimizer does above it, so the most ordinary lakehouse query in existence
pushes down and skips the rest of the table.

## What DuckDB cannot do

The gap that matters most doesn't appear on this page as a number. DuckDB is single-node and its optimizer is static. It commits to a plan before the first row is read and can't change its mind. Batcher re-optimizes at stage boundaries on measured cardinalities, the same granularity Spark AQE works at but available single-node too, and it carries a sketch-backed cross-query learned-stats loop that DuckDB has no equivalent for. The same mergeable operators then run across a cluster with a bit-identical result.

Stage-boundary re-optimization engages at 20M input rows and above, and it runs single-node, where Spark cannot go. See {doc}`/benchmarks/results/scaling`.

## Reproduce

```bash
python benchmarks/run.py --benchmark operators --tier single --scale 1
python benchmarks/run.py --benchmark tpch      --tier single --scale 1
```

## See also

- {doc}`/benchmarks/results/tpch` for the per-query breakdown.
- {doc}`/benchmarks/results/analytics` for operators, connectors, and the lazy control plane.
- {doc}`/benchmarks/comparisons/vs-polars` and {doc}`/benchmarks/comparisons/vs-daft` for the other two single-node engines.
- {doc}`/architecture/deep-dives/operators/aggregation-internals` for the quickselect finalize behind the median and quantile wins.
- {doc}`/architecture/deep-dives/operators/join-algorithms` for the join strategies behind the TPC-H results.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization` for what a static optimizer can't do.
- {doc}`/benchmarks/methodology` for hardware, gating, and why cross-hardware comparison is meaningless.
