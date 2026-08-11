# TPC-H

This page reports Batcher's TPC-H results against DuckDB, Daft, and Polars.

Against DuckDB reading the same Arrow, which is the like-for-like execution comparison, Batcher wins the suite at both scales measured, and wins every query at scale factor 1.

Three runs are published, and they are not interchangeable:

| Run | Scale | Hardware | Measured |
|---|---|---|---|
| Suite standing | sf10 | 96 cores | 2026-07-27 |
| Suite standing | sf1 (`lineitem` = 6,001,215 rows) | 16 cores | 2026-07-31 |
| Per-query detail | sf1 | 16 cores, 30 GB | 2026-07-18 |

The sf10 run is the most demanding, so it leads. The July 31 sf1 run is the current sf1
standing. The July 18 sf1 run is kept because it is the only one published query by query.

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
official sf1 answer, `123141078.2283`. Batcher returns exactly that.
:::

## At scale factor 10

![Bar chart of the TPC-H scale-factor-10 suite ratio. Batcher is 1.89x faster than DuckDB reading the same Arrow, winning 21 of 22 queries, and 2.26x faster than Polars, winning 17 of 22.](/_static/diagrams/tpch_sf10.svg)

At sf10 on 96 cores, correctness-gated with all 22 queries reporting `OK`, the suite total is
**4,453 ms**, down from 4,993 ms earlier in the same session. The two like-for-like
comparisons:

| Against | Suite ratio | Queries won |
|---|---|---|
| DuckDB on the same Arrow (`duckdb_arrow`) | **1.89x faster** | **21 of 22**; q9 is 1.01x, a tie |
| Polars | **2.26x faster** | 17 of 22 |

The single query that moved most is q5, which more than halved once the optimizer was given a
distinct-count estimate before it chose the join order. Its peak resident memory is the result
that matters more than the timing: q5 no longer takes the process past 110 GB.

:::{warning}
Treat the sf10 totals as indicative rather than exact. The box was shared with other work
during the run, and at load average 16 to 41 a repeated run of the same build swings about
25% (the same build measured 4,992 ms and 4,411 ms an hour apart). Per-query results quoted
here were reproduced at least twice, and anything that moved by less than 2x was taken from a
run at load average under 5.
:::

## Where the suite stands at scale factor 1

The current sf1 standing was measured 2026-07-31 on 16 cores, release build, all 22 queries
reporting `OK`. The suite total is **871 ms**:

| Against | Its suite total | Ratio |
|---|---:|---|
| DuckDB on the same Arrow (`duckdb_arrow`) | 2,062 ms | **2.37x faster** |
| Polars | 1,101 ms | **1.26x faster** |

**Against DuckDB reading the same Arrow (a like-for-like *execution* comparison), Batcher
wins all 22 queries at sf1.** That is the comparison Batcher's Arrow-only contract makes
fair. q21 runs too, because correlated subqueries are supported, so all 22 are comparable.

The suite total more than halved in this run, from 1,843 ms to **871 ms**, on a single
cardinality fix. Core measures a quantile grid from raw Arrow values, so a `date32` column's
grid counts epoch days, while Kyber read it with `date.toordinal()`, which counts from year 1
and is therefore 719,163 days out. Every temporal literal landed outside its own column's
grid, which interpolates to "no rows match", so
`o_orderdate BETWEEN '1995-01-01' AND '1996-12-31'` estimated **0 rows against a true
455,112** and a join with a zero-row side priced as free. Two queries carry almost all of the
recovery:

| Query | Before | After |
|---|---:|---:|
| q8 | 735.0 ms | **20.7 ms** |
| q7 | 309.5 ms | **30.5 ms** |
| suite | 1,843 ms | **871 ms** |

The defect bit only from a query's *second* execution, because the first has no measured grid
to read. That is why it survived so long: a benchmark warms up before it times, so every
timed run measured the broken state and the cold run that would have shown the good plan was
the one thrown away.

:::{dropdown} Per-query ratios vs DuckDB on the same Arrow
All 22 queries at **scale factor 1**, measured 2026-07-18 on a release build,
correctness-gated. The ratio is `batcher / duckdb`, so **below 1.0 means Batcher is faster**.
The sf10 standing above is a separate, later run on different hardware.

| Query | vs DuckDB-on-Arrow (same input) |
|---|---:|
| q1  | **0.51×** |
| q2  | **0.42×** |
| q3  | **0.59×** |
| q4  | **0.50×** |
| q5  | **0.24×** |
| q6  | **0.27×** |
| q7  | **0.35×** |
| q8  | **0.35×** |
| q9  | **0.27×** |
| q10 | **0.43×** |
| q11 | **0.15×** |
| q12 | **0.38×** |
| q13 | **0.81×** |
| q14 | **0.40×** |
| q15 | **0.14×** |
| q16 | **0.64×** |
| q17 | **0.91×** |
| q18 | **0.43×** |
| q19 | **0.91×** |
| q20 | **0.77×** |
| q21 | **0.53×** |
| q22 | **0.78×** |
| **total** | **22 of 22 won** |
:::

:::{dropdown} Per-query ratios vs Daft (re-measured 2026-07-18)
Both single-node on the same 16 cores and the same Arrow input. Ratio is `batcher / daft`,
so **below 1.0 means Batcher is faster**.

| Query | vs Daft |
|---|---:|
| q12 | **0.14×** |
| q11 | **0.18×** |
| q10 | **0.26×** |
| q15 | **0.43×** |
| q2 | **0.53×** |
| q8 | **0.55×** |
| q1 | **0.58×** |
| q14 | **0.60×** |
| q9 | **0.63×** |
| q7 | **0.64×** |
| q13 | **0.99×** |

**Batcher is faster on 11 of the 18 queries Daft answers correctly**, and the spread is wider
in Batcher's favor (up to 7x). Daft also **cannot complete four**: q6 is wrong (above), q18
returns an unaliased column, q21 fails to bind a correlated subquery, and q22 cannot parse
`SUBSTRING(x FROM a FOR b)`.

The parallel radix join and the whole-partition window kernel took q3 from 3.8× to 1.55× and
q4 to 1.51× against Daft.
:::

## Kernels

The in-memory microbenchmark in the run log (`benchmarks/microbench.py`) loads about 60M rows into Arrow once, so no I/O is in the way, and times each engine's kernels on the same 16 cores:

| Operator | Batcher | Daft | Polars |
|---|---:|---:|---:|
| filter | 28 ms | 188 ms | 156 ms |
| group-by | 359 ms | 487 ms | 223 ms |
| sum | 10 ms | 181 ms | 6 ms |

Batcher's kernels beat Daft's on all three and trade with Polars.

## Planner work behind these numbers

Two planner fixes landed against this suite, and they are worth naming because they show
where the wins come from.

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

The radix join's partition loop also ran on a single core, so a join too large to broadcast
funnelled a fully-parallel build and probe into a serial kernel. Joining the partitions
concurrently, concatenating them in partition order so the sequential output is reproduced
exactly, took TPC-H q4 from 115.6 ms to 43.0 ms and q3 from 110.3 ms to 66.3 ms.

## Reproduce

```bash
python benchmarks/run.py --benchmark tpch --tier single --scale 1   # sf1, vs DuckDB / Polars
python benchmarks/run.py --benchmark tpch --tier single --scale 10  # sf10, the suite standing
python benchmarks/run.py --benchmark tpch --engines batcher,daft    # vs Daft
```

## See also

- {doc}`/benchmarks/comparisons/vs-duckdb` and {doc}`/benchmarks/comparisons/vs-daft` for the full engine scorecards.
- {doc}`/benchmarks/results/analytics` for operators and connectors.
- {doc}`/architecture/deep-dives/operators/join-algorithms` for the shuffle and broadcast paths behind these results.
- {doc}`/architecture/deep-dives/adaptive/cardinality-estimation` for the cold-start NDV problem that ran q5 at 7,115 ms.
- {doc}`/architecture/deep-dives/adaptive/cost-model` for how the build side is chosen.
- {doc}`/user-guide/analyze/sql` for the supported SQL surface.
- {doc}`/benchmarks/methodology` for the correctness gate in detail.
