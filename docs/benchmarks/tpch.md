# TPC-H

TPC-H is the benchmark Batcher currently *loses*, and it is the most useful page on this
site for understanding where the engine stands. All 22 queries, scale factor 1
(`lineitem` = 6,001,215 rows), single node, 16 cores, 30 GB.

## Correctness first

:::{important}
**Batcher matches DuckDB's result on all 22 queries.** That is the only claim on this page
that is not about speed, and it is the one that gates every other number: the harness runs
the query on each engine, compares the results as a sorted row multiset within float
tolerance, and refuses to record a time when they disagree. A wrong answer produces no
timing. It does not produce a fast timing.
:::

The gate earns its keep on other engines, too:

| Engine | Correctness on the suite |
|---|---|
| DuckDB | Reference. |
| **Batcher** | **Matches DuckDB on all 22.** |
| Daft | **q6 is wrong** (mishandles `interval '1' year`: 75.2M against the correct 123.1M); cannot parse `SUBSTRING(x FROM a FOR b)` in q22. |
| Polars | **q6 is wrong**; its SQL frontend errors on most of the suite (multi-table `FROM`, `EXISTS`, non-equi joins). |

## Where the suite stands

:::{warning}
**DuckDB is faster on 16 of the 21 comparable queries**, with a geometric mean of about
**1.36× in DuckDB's favor**. **Daft is 2–12× faster on the join-heavy queries** and roughly
2× faster on a per-batch Python UDF. Batcher wins the five scan-and-aggregate-dominated
queries and loses the multi-join ones. That is the whole story, and it is consistent across
every other benchmark we run.
:::

**q21 is not comparable.** It raises `NotImplementedError`, because correlated subqueries are
not supported yet. It raises rather than returning a wrong answer, which is the behavior we
want from an unsupported feature.

:::{dropdown} Per-query ratios vs DuckDB
The ratio is `batcher / duckdb`, so below 1.0 means Batcher is faster.

| Query | vs DuckDB |
|---|---:|
| q14 | **0.71×** |
| q1 | **0.80×** |
| q6 | **0.82×** |
| q12 | **0.86×** |
| q16 | **0.99×** |
| q7 | 2.15× |
| q8 | 2.30× |
| q17 | 2.46× |
| q5 | 2.99× |
:::

:::{dropdown} Per-query ratios vs Daft
The ratio is `batcher / daft`, so above 1 means Daft is faster.

| Query | vs Daft |
|---|---:|
| q7 | 12× |
| q5 | 9.6× |
| q20 | 8.6× |
| q17 | 6.7× |
| q9 | 5.9× |
| q3 | 3.8× (was 7.7×) |

Batcher ties Daft on global aggregation, group-by, and single-stage expression ETL, and it
wins top-N and sort-limit by 8–10× (a fused top-N heap against a full sort).
:::

## Why

It would be convenient if this were a slow kernel. It is not. The in-memory microbenchmark
in the run log loads ~60M rows into Arrow once, so no I/O is in the way, and times each
engine's kernels on the same 16 cores:

| Operator | Batcher | Daft | DuckDB |
|---|---:|---:|---:|
| filter | 28 ms | 188 ms | 1,601 ms |
| group-by | 359 ms | 487 ms | 2,729 ms |
| sum | 10 ms | 181 ms | 92 ms |

The kernels win. The queries built out of them lose. Two things account for it:

1. **Single-node parallelism reaches only about 1.7–3.8× on 16 cores**, where DuckDB and
   Daft use effectively all of them. The shuffle path is the visible culprit: `key_indices`
   and `partition_by_keys` run serially over the 6M-row probe side before the per-bucket
   join, and the parallel broadcast join rebuilds its build-side hash table in every probe
   chunk instead of building once and sharing it.
2. **Batcher does roughly 2× more CPU work per query.**

Neither is a tuning knob. Both are tracked as open levers in
`benchmarks/BENCHMARK_RESULTS.md`.

## What did move

Two planner fixes landed against this suite, and they are worth naming because they show
where the remaining gap is *not*.

**Build-side selection.** Broadcast eligibility used to be checked only on the right input,
so when the small side arrived on the left, the join shuffled a 6M-row build instead of
broadcasting. It is now decided from `min(left_bytes, right_bytes)`. q3 went 7.7× → 3.8×
against Daft, and the q5 `orders ⋈ lineitem` join went 419 ms → 175 ms.

**Cold-start join cardinality.** The estimator's join model is right (`|L||R| / max(ndv)`),
but its NDV map read only *learned* NDV from past runs, so a cold join fell back to
`max(left, right)`, which under-estimates a low-NDV many-to-many join badly enough to steer
join order into 12–18M-row intermediates. Cold q5 ran 7,115 ms against a warm 300 ms. NDV is
now seeded from source statistics (footer and written-file HLL sketches), so the cold plan is
no longer flying blind.

## Reproduce

```bash
python benchmarks/run.py --benchmark tpch --tier single --scale 1  # vs DuckDB / Polars
python benchmarks/run.py --benchmark tpch --engines batcher,daft   # vs Daft (Ray Data has no SQL)
```

## See also

- [vs DuckDB](vs-duckdb.md) and [vs Daft](vs-daft.md): the full engine scorecards.
- [Analytics and I/O](analytics.md): operators and connectors.
- [Join algorithms](../deep-dives/join-algorithms.md): the shuffle and broadcast paths the
  gap lives in.
- [Cardinality estimation](../deep-dives/cardinality-estimation.md): the cold-start NDV
  problem that ran q5 at 7,115 ms.
- [Cost model](../deep-dives/cost-model.md): how the build side is now chosen.
- [SQL guide](../user-guide/sql.md): the supported SQL surface, including what q21 needs.
- [Methodology](methodology.md): the correctness gate in detail.
