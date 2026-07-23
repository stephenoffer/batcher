# TPC-H

This page reports Batcher's TPC-H results against DuckDB, Daft, and Polars, and explains where the remaining gap comes from.

TPC-H shows both sides of the engine. Against DuckDB reading the same Arrow, Batcher wins all 22 queries. Against DuckDB on its own compressed store, where DuckDB never pays an ingest and decompresses as it scans, DuckDB still wins 15 of 22. Both numbers are below. Publishing only the first would be marketing and publishing only the second would be false modesty.

All 22 queries, scale factor 1 (`lineitem` = 6,001,215 rows), single node, 16 cores, 30 GB,
release build, measured 2026-07-18.

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
| **Batcher** | **Matches DuckDB on all 22**, and matches the *official* TPC-H answer on q6. |
| Daft | **q6 is wrong** (returns 75.2M against the official 123,141,078.2283); cannot parse `SUBSTRING(x FROM a FOR b)` in q22. |
| Polars | **q6 is wrong** (same 75.2M); its SQL frontend errors on most of the suite (multi-table `FROM`, `EXISTS`, non-equi joins). |

:::{dropdown} What Daft and Polars actually get wrong on q6
The predicate is `l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01`. In IEEE double,
`0.06 + 0.01` is `0.06999999999999999`, a hair under `0.07`, so an engine that folds the bound in floating point drops every `l_discount = 0.07` row and loses about 39% of the revenue. TPC-H defines `l_discount` as `DECIMAL`, so the 0.07 rows belong in the answer.

Ground truth was computed independently in PyArrow over the identical input and equals the
official sf1 answer, `123141078.2283`. Batcher returns exactly that. This page previously
attributed the error to `interval '1' year`, which was wrong.
:::

## Where the suite stands

**Against DuckDB reading the same Arrow (a like-for-like *execution* comparison), Batcher
wins all 22 queries**, by 1.1x to 6.9x. That is the comparison Batcher's Arrow-only contract makes fair. q21 runs too, because correlated subqueries are supported, so all 22 are comparable.

:::{warning}
**Against DuckDB on its own native compressed store, DuckDB is faster on 15 of 22**, geometric
mean **≈1.40× in DuckDB's favor** (≈1.29× excluding q17, an 8× outlier). That is not a
like-for-like execution comparison, because DuckDB decompresses its own format on the fly and never pays an Arrow ingest. It is the number you get from `duckdb` at a prompt, so it's published here. Batcher wins the scan-and-aggregate-dominated queries (q15 0.46×, q12 0.74×,
q11 0.80×, q1/q9 0.88×) and loses the join- and subquery-heavy ones (q17 7.9×, q20 2.8×,
q3 2.6×, q21 2.4×).
:::

:::{dropdown} Per-query ratios vs DuckDB
All 22 queries, measured 2026-07-18 on a release build, correctness-gated. The ratio is
`batcher / duckdb`, so **below 1.0 means Batcher is faster**.

| Query | vs DuckDB-on-Arrow<br>(same input) | vs native DuckDB<br>(its own store) |
|---|---:|---:|
| q1  | **0.51×** | **0.88×** |
| q2  | **0.42×** | 1.64× |
| q3  | **0.59×** | 2.57× |
| q4  | **0.50×** | 1.77× |
| q5  | **0.24×** | 1.63× |
| q6  | **0.27×** | **0.92×** |
| q7  | **0.35×** | 1.32× |
| q8  | **0.35×** | 1.72× |
| q9  | **0.27×** | **0.88×** |
| q10 | **0.43×** | 1.05× |
| q11 | **0.15×** | **0.80×** |
| q12 | **0.38×** | **0.74×** |
| q13 | **0.81×** | 1.17× |
| q14 | **0.40×** | 1.14× |
| q15 | **0.14×** | **0.46×** |
| q16 | **0.64×** | 1.54× |
| q17 | **0.91×** | 7.91× |
| q18 | **0.43×** | **0.91×** |
| q19 | **0.91×** | 1.52× |
| q20 | **0.77×** | 2.81× |
| q21 | **0.53×** | 2.38× |
| q22 | **0.78×** | 2.08× |
| **total** | **22 of 22 won** | 7 of 22 won, geomean 1.40× |
:::

:::{dropdown} Per-query ratios vs Daft (re-measured 2026-07-18)
Both single-node on the same 16 cores and the same Arrow input. Ratio is `batcher / daft`,
so **below 1.0 means Batcher is faster**.

| Batcher faster | | Daft faster | |
|---|---:|---|---:|
| q12 | **0.14×** | q20 | 2.03× |
| q11 | **0.18×** | q3 | 1.55× |
| q10 | **0.26×** | q4 | 1.51× |
| q15 | **0.43×** | q17 | 1.35× |
| q2 | **0.53×** | q5 | 1.13× |
| q8 | **0.55×** | q16 | 1.07× |
| q1 | **0.58×** | q19 | 1.06× |
| q14 | **0.60×** | | |
| q9 | **0.63×** | | |
| q7 | **0.64×** | | |
| q13 | **0.99×** | | |

**Batcher is faster on 11 of the 18 queries Daft answers correctly**, and the spread is wider
in Batcher's favour (up to 7×) than against it (up to 2×). Daft additionally **cannot
complete four**: q6 is wrong (above), q18 returns an unaliased column, q21 fails to bind a
correlated subquery, and q22 cannot parse `SUBSTRING(x FROM a FOR b)`.

Earlier revisions of this page reported Daft 4x to 12x ahead on the join-heavy queries. That gap
is gone: the parallel radix join and the whole-partition window kernel landed since, taking
q3 from 3.8× behind to 1.55× and q4 to 1.51×.
:::

## Why

It would be convenient if this were a slow kernel. It isn't. The in-memory microbenchmark in the run log (`benchmarks/microbench.py`) loads about 60M rows into Arrow once, so no I/O is in the way, and times each engine's kernels on the same 16 cores:

| Operator | Batcher | Daft | Polars |
|---|---:|---:|---:|
| filter | 28 ms | 188 ms | 156 ms |
| group-by | 359 ms | 487 ms | 223 ms |
| sum | 10 ms | 181 ms | 6 ms |

Batcher's kernels beat Daft's on all three and trade with Polars, which is ahead on group-by and sum. Nothing here is slow enough to explain the query-level gap. The kernels hold up and the queries built out of them lose, so two other things account for it:

1. **Single-node parallelism plateaus after about 8 cores.** Measured on q3 with the plan
   built once: 171 ms at 1 core → 43 ms at 8 → 42.8 ms at 16, a 4.0× speedup from 16 cores
   (~25% parallel efficiency). Narrowed by operator, `GROUP BY` alone scales **19.2×** and
   the join alone only **5.9×**, so the join is the ceiling, not the aggregate.
2. **Batcher does roughly 2× more CPU work per query.**

One documented cause of (1) is now fixed: the radix join's partition loop ran on a single
core, so a join too large to broadcast funnelled a fully-parallel build and probe into a
serial kernel. Joining the partitions concurrently, concatenating them in partition order so the sequential output is reproduced exactly, took TPC-H q4 from 115.6 ms to 43.0 ms and q3 from 110.3 ms to 66.3 ms. What remains is an unexplained roughly-constant serial section
inside the join; three plausible causes (serial hash-table build, fixed per-query overhead,
memory-bandwidth-bound output gather) were each tested and **ruled out**, so the next step is
a real profiler on `bc-interp`'s join path rather than more black-box timing.

Both are tracked as open levers in `benchmarks/BENCHMARK_RESULTS.md`.

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
join order into 12M to 18M-row intermediates. Cold q5 ran 7,115 ms against a warm 300 ms. NDV is
now seeded from source statistics (footer and written-file HLL sketches), so the cold plan is
no longer flying blind.

## Reproduce

```bash
python benchmarks/run.py --benchmark tpch --tier single --scale 1  # vs DuckDB / Polars
python benchmarks/run.py --benchmark tpch --engines batcher,daft   # vs Daft (Ray Data has no SQL)
```

## See also

- {doc}`vs-duckdb` and {doc}`vs-daft` for the full engine scorecards.
- {doc}`analytics` for operators and connectors.
- {doc}`../deep-dives/join-algorithms` for the shuffle and broadcast paths the gap lives in.
- {doc}`../deep-dives/cardinality-estimation` for the cold-start NDV problem that ran q5 at 7,115 ms.
- {doc}`../deep-dives/cost-model` for how the build side is chosen.
- {doc}`../user-guide/sql` for the supported SQL surface.
- {doc}`methodology` for the correctness gate in detail.
