# vs DuckDB

DuckDB is the single-node analytical engine to beat, and on join-heavy SQL it is still
ahead of us. This page gives the split: the operator shapes Batcher takes, the query
shapes DuckDB takes, and why.

:::{important}
Every number below was produced by a run that had to pass the correctness gate first: the
harness compares the two engines' results as a sorted row multiset within float tolerance
and refuses to record a time when they disagree. Batcher matches DuckDB's result on all 22
TPC-H queries, so nothing here is a correctness argument. It is a speed argument, in both
directions.
:::

## Scorecard

| Shape | Winner |
|---|---|
| Global aggregate, filtered count | Batcher, by 5× |
| Group-by (one or two keys) | Batcher, 1.3–1.5× |
| Window running `sum()` | Batcher, 1.4× |
| Window `rank()`, `lag()` | DuckDB |
| `MEDIAN` / `QUANTILE_CONT` per group | Batcher, 1.1× |
| Join → aggregate | DuckDB, 1.15× |
| TPC-H overall (sf1) | DuckDB, geomean ~1.36× |
| Delta file skipping (`count(*)` with a predicate) | Batcher, 1.42× |

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
| sort → top-N (`LIMIT`) | 14.1 ms | 13.3 ms | 1.06× |
| filter → project | 13.9 ms | 12.9 ms | 1.08× |
| join → aggregate | 98.3 ms | 85.6 ms | 1.15× |
| window `lag()` | 180 ms | 151 ms | 1.19× |
| window `rank()` | 221 ms | 133 ms | 1.66× |

The filtered count is the widest margin, and it is not a micro-optimization: `.count()`
over a filter compiles to a `COUNT(*)` aggregate, so projection pushdown prunes the scan
to the one column the predicate touches and the count fuses into a single `count_if` pass.
Nothing else is read, and no matching row is ever materialized.

`rank()` and `lag()` go the other way. DuckDB's window operator is better than ours on the
ordered frame kernels, and we have not closed that.

:::{tip}
Both directions are reachable from your own query. `ds.explain()` shows whether the
predicate reached the scan and which columns survived pruning; `ds.stats()` reports what
each operator actually cost. [Optimizing a slow
query](../tutorials/optimizing-a-slow-query.md) walks the loop.
:::

## In-memory kernels

To separate compute from I/O, the in-memory microbenchmark in the run log loads roughly 60M
TPC-H rows into Arrow once and times each engine's kernels. Single node, 16 cores:

| Operator | Batcher | DuckDB |
|---|---:|---:|
| filter | 28 ms | 1,601 ms |
| group-by | 359 ms | 2,729 ms |
| sum | 10 ms | 92 ms |

The Rust kernels are not the problem. That matters for the next section, because it means
the TPC-H gap is not kernel speed.

## Aggregates DuckDB used to win

Two exact aggregates were slower than DuckDB and are now faster. Both were algorithmic,
not tuning. 16 cores.

| Query | Before | After | DuckDB |
|---|---:|---:|---:|
| `MEDIAN(x) GROUP BY flag` (5M rows, 3 groups) | 427 ms | **210 ms** | 232 ms |
| `QUANTILE_CONT(x, 0.9) GROUP BY flag` (5M rows) | 406 ms | **208 ms** | 226 ms |
| `COUNT(DISTINCT id) GROUP BY flag` (2M rows) | 287 ms | **163 ms** | 181 ms |

Median and quantile need the value at one rank, not a sorted list, so the finalize does
quickselect instead of a full sort and runs each group's selection on its own core. The
count-distinct is a Kyber rewrite: a lone `COUNT(DISTINCT x) GROUP BY g` becomes a
distinct over `(g, x)` followed by a count, which parallelizes across the distinct values
instead of the handful of groups.

## Where DuckDB wins: TPC-H

:::{warning}
All 22 queries, scale factor 1, 16 cores. **DuckDB is faster on 16 of the 21 comparable
queries**, with a geometric mean of about **1.36× in DuckDB's favor**. This is the headline
loss on the site and it is not going to be argued away by the operator table above it.
:::

Re-measured 2026-07-18 on a release build, against DuckDB's **native compressed store**:

| | Queries |
|---|---|
| Batcher faster | q15 (0.46×), q12 (0.74×), q11 (0.80×), q1 (0.88×), q9 (0.88×), q18 (0.91×), q6 (0.92×) |
| Batcher slower | the other 15; worst are q17 (7.91×), q20 (2.81×), q3 (2.57×), q21 (2.38×) |
| Not comparable | none — **all 22 run**. Correlated subqueries are now supported, so q21 is measured |

Against **DuckDB reading the same Arrow**, the result inverts: Batcher wins **all 22**, by
1.1×–6.9×. That is the like-for-like execution comparison; the table above is what a user gets
from `duckdb` at a prompt, where DuckDB also decompresses its own format as it scans and never
pays an Arrow ingest. Both are published, on [the TPC-H page](tpch.md).

The pattern is clean. Batcher wins the scan-and-aggregate queries and loses the multi-join
ones. Given the kernel numbers above, the cause is not the aggregation or the filter. It is
that single-node parallelism currently reaches only about 1.7–3.8× on 16 cores where DuckDB
and Daft use effectively all of them, and Batcher does roughly 2× more CPU work per query.
Closing that is a runtime-parallelism and kernel-efficiency effort, and it is the top open
lever in `benchmarks/BENCHMARK_RESULTS.md`. It is not a knob.

[The TPC-H page](tpch.md) has the per-query detail.

## Lakehouse reads

A selective predicate on a Delta table should open one data file, not all of them. The
transaction log records each file's column bounds, and reading it at plan time is the whole
game. Single node, 10M rows across 200 Delta files, one `day` per file:

| `count(*) WHERE day = 42` | Time | Files opened |
|---|---:|---:|
| Batcher (before the fix) | 98.8 ms | 200 |
| **Batcher** | **13.4 ms** | **1** |
| DuckDB `delta_scan` | 19.0 ms | baseline |

We were 2.7× slower than DuckDB here and are now 1.42× faster. The bug was ours and it was
in the optimizer, not the connector: the `COUNT(*)`-over-`Filter` fusion deleted the
`Filter` node that source-predicate extraction was looking for, so the most ordinary
lakehouse query in existence pushed nothing down and scanned the whole table. Predicates
are now recovered from the user's plan, where a `Filter` on a `Scan` constrains that scan
whatever the optimizer does above it.

## What DuckDB cannot do

The gap that matters most is not on this page as a number. DuckDB is single-node and its
optimizer is static: it commits to a plan before the first row is read and cannot change
its mind. Batcher re-optimizes *during* the query at pipeline breakers, on measured
cardinalities, and the same mergeable operators run across a cluster with a bit-identical
result. See [scaling](scaling.md).

## Reproduce

```bash
python benchmarks/run.py --benchmark operators --tier single --scale 1
python benchmarks/run.py --benchmark tpch      --tier single --scale 1
```

## See also

- [TPC-H](tpch.md): the per-query breakdown.
- [Analytics and I/O](analytics.md): operators, connectors, the lazy control plane.
- [vs Polars](vs-polars.md) and [vs Daft](vs-daft.md): the other two single-node engines.
- [Aggregation internals](../deep-dives/aggregation-internals.md): the quickselect finalize
  behind the median and quantile wins.
- [Join algorithms](../deep-dives/join-algorithms.md): the operator the TPC-H gap lives in.
- [Adaptive re-optimization](../deep-dives/adaptive-reoptimization.md): what a static
  optimizer cannot do, and the reason the last section is not a number.
- [Methodology](methodology.md): hardware, gating, and why cross-hardware comparison is
  meaningless.
