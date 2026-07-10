# TPC-H / ClickBench vs DuckDB, Polars, PyArrow, Daft, Spark — where Batcher's time goes

Companion to `BENCHMARK_RESULTS.md` (which covers Ray Data / Daft on the workloads Batcher
already wins by 50–450×). This document measures the ones it *loses*: the 22-query TPC-H
decision-support suite and the single-table operator mix, against the engines with a
comparable single-node native execution model.

Reproduce from a local mirror of the public data:

```bash
python tools/mirror_bench_data.py --dataset tpch --scale 1 --scale 10 --scale 100
python tools/mirror_bench_data.py --dataset clickbench --parts 100
export JAVA_HOME=$(python -c "import jdk; print(jdk.install('17', jre=True))")   # for Spark
python benchmarks/run.py --benchmark tpch --scale 1 --source ~/bench-data/tpch \
    --engines batcher,duckdb,polars,daft,spark
python benchmarks/run.py --benchmark operators --source ~/bench-data/tpch \
    --engines batcher,duckdb,polars,pyarrow
```

The mirror is not generated data — it is the same public parquet
(`s3://ray-benchmark-data/tpch`, `datasets.clickhouse.com/hits_compatible`) rewritten once
with the canonical column names and decimal→float64 normalization that `sources.py`
otherwise applies after materializing into Arrow. That normalization has to happen *before*
the scan for the `--scan` path to work at sf100, where the tables do not fit in memory.

## Read this first: the measurement caveat

Every wall-clock number here was taken on a machine **shared with other benchmark
processes**. Repeated runs of the same query varied by up to 2×. Treat the timings as
*indicative of magnitude*, not as a publishable measurement.

The structural numbers — q-error, peak intermediate bytes, estimate-vs-actual, the
parallel-scaling curve, the native/Python split — are deterministic or ratio-based and
survive that noise. Those are the ones the conclusions rest on.

## Where we stand

**Batcher is 1.3×–7.2× slower than DuckDB across TPC-H sf1**, and does not reach the 2×
lead over any of the five comparators. It *is* faster than Polars on 8 of 13 comparable
queries and faster than Spark on nearly all of them, but DuckDB is the bar that matters.

TPC-H sf1, best-of-5, `b/x` = batcher_ms / x_ms (lower is better for Batcher):

| query | batcher | duckdb | polars | daft | spark | b/duckdb |
|-------|--------:|-------:|-------:|-----:|------:|---------:|
| q1  |  23.9 | 15.2 |  78.8 |  43.4 |  431 | 1.57× |
| q3  |  79.0 | 22.2 |  27.6 |  37.4 |  496 | 3.55× |
| q5  | 139.3 | 21.1 |  36.6 |  34.2 |  829 | 6.59× |
| q8  | 163.8 | 22.8 | 240.7 |  74.6 |  480 | 7.17× |
| q9  | 209.7 | 64.4 | 841.0 |  85.9 | 1038 | 3.25× |
| q13 |  54.6 | 41.5 |   ERR |  52.4 | 1621 | 1.31× |
| q18 | 154.7 | 33.8 |   ERR |  40.9 |  904 | 4.58× |

Three comparator caveats, all confirmed:

* **Polars' SQL frontend fails 9 of 22 queries** (`subquery comparisons with '=' are not
  supported`, `unsupported join constraint`, `cross joins would produce more rows than fits
  into 2^32`) and returns a **wrong answer on q6**. A fair Polars comparison must drive its
  LazyFrame API (`suites/standard/tpch_dataframe.py`), not `pl.sql`.
* **Daft** is also wrong on q6, returns the wrong columns on q18, and cannot express q21/q22.
* **PyArrow and Ray Data have no SQL surface at all**, so they cannot run TPC-H. They appear
  only in the operator mix below. Claiming a TPC-H win over them would be meaningless.
* **Batcher cannot run q21** — `NotImplementedError: correlated subqueries not supported`.

On q6, Polars and Daft both drop the `l_discount = 0.07` rows (their 75,207,768 is exactly
the 0.05+0.06 revenue share); Batcher's 123,141,078.23 matches DuckDB *and* the published
TPC-H answer. The harness used to treat "the first engine that produced a result" as the
reference, which meant Batcher — so a comparator's bug was reported as Batcher's. It now
prefers DuckDB, the project's designated oracle (`.claude/rules/testing.md`).

## The gap widens with scale

TPC-H sf10, batcher vs duckdb (all 21 runnable queries **correct**):

| query | batcher | duckdb | b/duckdb |   | query | batcher | duckdb | b/duckdb |
|-------|--------:|-------:|---------:|---|-------|--------:|-------:|---------:|
| q1  |  154.9 |  71.6 | 2.16× | | q12 |  561.5 |  73.7 |  7.62× |
| q3  |  686.4 |  83.1 | 8.26× | | q17 |  740.2 |  55.2 | 13.42× |
| q5  | 1341.7 |  88.8 |**15.10×**| | q18 | 2050.1 | 162.1 | 12.65× |
| q7  |  720.3 |  55.2 |13.04× | | q19 |  607.9 | 101.8 |  5.97× |
| q8  |  880.2 |  83.5 |10.54× | | q20 |  552.0 |  64.4 |  8.57× |
| q9  | 1381.6 | 176.4 | 7.83× | | q22 |  127.1 |  58.4 |  2.18× |

Compare with sf1's 1.3×–7.2×. **Ten times the data roughly doubles the deficit.** DuckDB's
per-query cost grows sub-linearly because it spreads the extra work across cores; Batcher's
does not, because it stops scaling at ~16–32 workers (see Root cause 0). This is the
strongest evidence that parallel efficiency — not any single operator — is the dominant
problem, and it is why the sf1 numbers understate the gap.

## ClickBench (20M-row subset)

`run.py --benchmark clickbench` — 43 single-table analytic queries. Batcher is **2×–38×
slower than DuckDB** wherever both engines execute the query:

| query | batcher | duckdb | b/duckdb |   | query | batcher | duckdb | b/duckdb |
|-------|--------:|-------:|---------:|---|-------|--------:|-------:|---------:|
| q08 |  113.6 |  45.5 | 2.49× | | q19 |   58.4 |   1.8 | **32.91×** |
| q12 |  223.7 |  35.1 | 6.37× | | q26 |   51.1 |   7.5 |  6.83× |
| q13 |  801.9 |  80.9 | 9.91× | | q33 | 1218.3 | 121.4 | 10.03× |
| q14 |  415.4 |  39.2 |10.61× | | q34 | 1276.3 | 137.7 |  9.27× |
| q16 |  790.3 |  72.9 |10.85× | | q41 |   56.2 |   4.8 | 11.68× |
| q18 | 1391.4 | 111.6 |12.47× | | q42 |    ERR |   9.1 |      – |

**The apparent wins on q00–q06 (0.03×–0.42×) are not execution wins.** `COUNT(DISTINCT
UserID)` takes 602 ms on its first execution and 0.9 ms on every one after: Batcher answers
it from the exact distinct count its own previous run measured and recorded. The answer is
*correct* — verified equal to DuckDB's 1,972,146 — but the harness warms up before timing,
so Batcher is being timed on a memoized answer while DuckDB re-executes. Any headline that
cites those ratios is comparing a cache to an engine. The honest number for q04 is the cold
one: 602 ms vs DuckDB's 30 ms, i.e. **20× slower**.

Two dataset/harness bugs were fixed to get this far, and both had made every engine fail:

* **ClickBench's temporal columns were never reconstructed.** `hits_compatible` parquet
  stores `EventDate` as `uint16` days and the `*EventTime`s as `int64` seconds, while the
  queries compare them to date strings and call `extract(minute FROM EventTime)`. The
  reference loaders rebuild the types on ingest; `sources.py` did not, so q18/q36/q37/q41
  failed identically on Batcher *and* DuckDB (`Could not convert string '2013-07-01' to
  UINT16`). Now normalized in `_reconstruct_clickbench_temporals`.
* **The result comparison was case-sensitive on column names.** DuckDB folds `avg(UserID)`
  to `avg(userid)`; Batcher preserves the case. Six of thirteen "failures" were only this.

What remains failing is mostly the harness being stricter than ClickBench itself, which
compares only timings:

* `q17` is `GROUP BY … LIMIT 10` with **no ORDER BY** — the rows returned are arbitrary.
* `q31`, `q32`, `q35`, `q38`, `q39`, `q40` order by a column with ties and then `LIMIT`, so
  the tie-break is arbitrary.
* `q29`/`q35` differ only in a derived column's generated *name* (`sum((ResolutionWidth +
  0))` vs `sum(resolutionwidth + 0)`).
* The harness canonicalizes every integer through `float`, which loses precision on the
  `SUM(UserID)` values (≈8.7e18) these queries produce.

Running ClickBench at its full 100M rows needs a scan path: the suite materializes the whole
table into shared Arrow (~100 GB) and is OOM-killed. `sources.table_uris` raises for
ClickBench. That is the next harness gap.

## Root cause 0 (continued) — the join's serial prefix

Per-operator scaling of `join-agg` (`lineitem ⋈ orders`, sum), from `explain(analyze)` at
1 / 8 / 48 workers, sf1:

| op | p=1 | p=8 | p=48 | speedup |
|----|----:|----:|-----:|--------:|
| hash_join |  2981.6 | 1415.8 | 240.8 | 12.4× |
| project   |   209.0 |   39.7 |  20.5 | 10.2× |
| aggregate (group-by, separate query) | 1773.9 | 244.0 | 89.7 | **19.8×** |

The group-by aggregate scales half again as well as the join. Solving Amdahl on the join's
numbers gives a serial fraction of ≈183 ms against ≈2798 ms of parallel probe — a hard
ceiling of ~16×, and 12.4× is what it delivers.

The serial part is the **hash-table build**, and — worse — `radix_partition`, the scatter
that both radix joins begin with, was a single sequential loop over every build *and* probe
row. So even `radix_join_scalar_parallel`, the join the executor picks for a large broadcast
build, had a fully sequential prefix over (at sf10) 60M probe rows.

**Fixed:** `bc_runtime::join::radix` partitions in the textbook three parallel phases —
per-chunk histogram, exclusive prefix sum over `(chunk, partition)`, parallel scatter into
the reserved disjoint slices. It is **bit-identical to the serial scatter**: because a
chunk's slice of a partition is offset by all earlier chunks' counts and each chunk walks
its rows in increasing order, every partition ends up in ascending `abs_row` order, exactly
as the serial `push` loop left it. That matters — the join's output row order, and hence the
`seq == par` oracle, depends on it. Four unit tests pin the equivalence, including one that
compares against the serial scatter directly and one that checks ordering across chunk
boundaries; the existing `radix_matches_flat` oracle still passes.

The scaling curve now keeps improving through 96 workers where it previously plateaued at
32, but efficiency remains ~9-14%: the build side's hash-table construction, the
`materialize` copy of both join inputs, and the per-morsel output allocation are the next
serial costs. **A caveat on the numbers: this was measured while the machine carried a load
average of 97 from other work, so treat the absolute times as noise-bounded.** The
correctness of the change is not in question; the size of its win is not yet cleanly
measured.

## A correction, and a warning about measuring this engine

An earlier version of this document reported that TPC-H Q5 at sf10 carried 2,161,115,018
bytes of `n_name` through a join. **That number was wrong, and so was the conclusion drawn
from it.** The overflow check summed `StringArray::value_data().len()` per morsel — but a
morsel is a zero-copy *slice*, and a sliced Arrow array keeps the whole values buffer while
narrowing only its offsets. Summing it across the 3,663 morsels of a 60M-row `lineitem`
counted one 264 MB buffer 3,663 times and "found" a 966 GB column.

The consequence was not academic: the false positive widened ordinary `Utf8` columns to
`LargeUtf8`, and `bc-expr`'s scalar kernels accept `Utf8` only — so TPC-H sf10 Q12 began
failing with `Invalid comparison operation: LargeUtf8 == Utf8` and Q14 with `string function
StartsWith expected a Utf8 argument`. Both were green before and are green again. The check
now measures `offsets[len] - offsets[0]`, with a regression test that a sliced morsel counts
only its own bytes.

The original panic (`attempt to add with overflow` in arrow's `GenericBytesBuilder`) was
real and reproducible before any of this code existed, and the guard against it stands. But
which column actually overflowed is no longer known — the diagnosis that named `n_name` came
from the same broken measurement. The lesson generalizes: **on an engine whose morsels are
slices, any per-batch byte accounting that reads a buffer length rather than an offset range
is wrong by the number of morsels.**

## Root cause 0a — two engine bugs the better plans exposed

Improving the cold plans surfaced two defects that a worse plan simply never reached.

**1. A panic on user data.** `bc_interp::ops::materialize` — "the first step of every
pipeline breaker" — concatenates a whole relation into one `RecordBatch` per column.
Arrow's `concat` builds `Utf8`/`Binary` through a `GenericBytesBuilder` that accumulates
lengths into an `i32` and **panics on overflow**. TPC-H Q5 at sf10 carries `n_name`
through a 12M-row join whose concatenated value bytes reach **2,161,115,018** — past
`i32::MAX` — and the engine aborted:

```
thread panicked at arrow-array/src/builder/generic_bytes_builder.rs:
attempt to add with overflow
```

That violates the crate's own rule that nothing on a data path may panic. Fixed: an
overflowing `Utf8`/`Binary` column is now widened to its 64-bit-offset counterpart
(`LargeUtf8`/`LargeBinary`) and the rebuilt schema follows the columns actually produced.
The values are unchanged; only the offset width grows, which the downstream kernels
(`take`, radix grouping, sorting, shuffle) already handle. Confirmed unrelated to the
worker-width change: it reproduces with `parallelism=96` pinned.

**2. The adaptive executor cannot survive a large stage boundary.** With the panic fixed,
cold Q5 at sf10 is OOM-killed with 162 GB free. The same query, same plan, with
`collect(adaptive=False)`:

```
q5 sf10 adaptive=False: 2745 ms  rows=5  peak RSS 23.4 GB   (23.4 GB is the loaded tables)
```

**2.7 seconds, versus 77 seconds for the baseline plan** — and versus an OOM through the
adaptive path. `resolve_adaptive` enables stage-by-stage re-optimization whenever a join
operand is a breaker-produced, `Provenance.DEFAULT`-sized relation and the query reads
≥ 20M rows — which is exactly sf10. The loop then materializes that boundary. A sharper
plan makes the *first* stage boundary the big one (12M wide rows, 9.2 GB), and the staging
blows past memory.

This is Batcher's headline feature defeating itself, and it is where Carbonite is supposed
to intervene: the adaptive loop must refuse to materialize a stage boundary that does not
fit the memory envelope, and fall back to one-shot execution. Relatedly, Kyber ranks join
orders by **rows**, not bytes — it has no reason to prefer the order that carries fewer
string payloads through the breaker, even though `__column_avg_bytes__` is already learned.
Both are open.

Until they are fixed, `collect(adaptive=False)` is the fast path at sf10 for join-heavy
queries.

## Cold-start planning: what the ndv seeding is actually worth

Fresh `MetadataHub` per query (a query nobody has run before), batcher-only, ms:

| query | baseline | with seeding | speedup |
|-------|---------:|-------------:|--------:|
| sf1 q1  | 3808 | 736 | **5.2×** |
| sf1 q5  | 4327 | 301 | **14.4×** |
| sf1 q8  | 3470 | 286 | **12.1×** |
| sf1 q9  | 4116 | 327 | **12.6×** |
| sf1 q18 | 2120 | 283 | **7.5×** |
| sf10 q7 | ~75000 | 2300 | **~33×** |
| sf10 q8 | 75332 | 2097 | **35.9×** |
| sf10 q9 | 80931 | 4855 | **16.7×** |
| sf10 q5 | 77638 | 2745 (`adaptive=False`) | **28.3×** |

This is the largest wall-clock effect measured in this document. It does **not** show up in
the benchmark table, because the harness warms up before timing and the post-run learning
loop supplies `ndv` from run 2 onward. It is what a query that runs once — most ad-hoc
analytics — actually experiences.

## The operator mix says the engine is not uniformly slow

`run.py --benchmark operators` over the same TPC-H tables — one relational operator each:

| op | batcher | duckdb | polars | pyarrow | b/duckdb |
|----|--------:|-------:|-------:|--------:|---------:|
| global-sum      |   0.6 |   1.2 |   1.4 |    5.5 | **0.53×** |
| filter-count    |   0.7 |   1.3 |   6.2 |  254.3 | **0.56×** |
| window-runsum   |  90.5 | 233.0 | 929.1 |    n/a | **0.39×** |
| window-rank     |  76.3 | 124.7 | 927.7 |    n/a | **0.61×** |
| window-lag      | 116.2 | 163.0 |2422.7 |    n/a | **0.71×** |
| groupby-sum     |   6.2 |   3.3 |  14.5 |    5.0 | 1.86× |
| join-agg        |  81.3 |  42.7 |  49.7 |  155.0 | 1.90× |
| sort-limit      |  11.0 |   5.8 | 235.0 | 2397.6 | 1.91× |
| filter-project  |  22.4 |   7.8 |  11.9 |  186.3 | 2.88× |

Batcher already **beats DuckDB by 1.6–2.6×** on window functions, global aggregation and
filter-count. It loses on projection, sort, group-by and join. So the deficit is not "the
engine is slow" — it is concentrated in four operators plus per-query overhead.

## Root cause 0 — the engine barely uses the machine

The single largest finding. Sweeping `execution.parallelism` (sf1, ms, best-of-5):

| workers | filter-agg | groupby | groupby-hi | join-agg |
|--------:|-----------:|--------:|-----------:|---------:|
|   1 |   8.48 | 173.91 | 216.16 | 597.20 |
|   4 |   4.74 |  47.85 |  73.66 | 173.33 |
|  16 | **3.76** |  23.76 | **43.27** |  83.02 |
|  32 |   7.01 |  19.43 |  58.70 | **75.83** |
|  48 |   7.24 |  19.57 |  59.77 |  79.29 |
|  96 |   9.42 | **19.07** |  67.68 |  92.69 |

Two things are wrong here.

1. **Scaling stops at ~16–32 workers** on a 48-core / 96-thread box. Past that, three of the
   four shapes get *worse*. The suspects are SMT (96 logical vs 48 physical cores), the
   2-socket NUMA split, allocator contention across `filter_batch_jit`'s per-morsel output
   allocation, and a hash-shuffle bucket count that keys off `current_num_threads()` so the
   merge cost grows with the worker count.
2. **The default is the worst setting.** `ExecutionConfig.parallelism = 0` resolves to
   `std::thread::available_parallelism()` = 96, which is slower than 16 workers on
   `filter-agg` (2.5×), `groupby-hi` (1.6×) and `join-agg` (1.2×) — and on `filter-agg`,
   slower than running on **one** thread.

Note also that `RAYON_NUM_THREADS` has no effect: `execute_parallel_with_metrics` builds its
own scoped pool sized from `available_parallelism()` (deliberately — a Ray worker's global
pool is built before CPU affinity is applied). Anyone benchmarking Batcher by setting
`RAYON_NUM_THREADS` is measuring nothing.

This is the highest-value open item and it is architectural, not a constant: the right width
depends on the operator and the data size, which is exactly the kind of decision Kyber's
`annotate_ops` already computes per operator (it scales parallelism with row count) and which
the engine then ignores in favour of one global width. Wiring the per-operator parallelism
annotation through to the scoped pool — and capping the default at physical cores, and never
spawning more workers than there are morsels — is the change to make.

## Root cause 0b — how much time is even in the engine

Wrapping the `_native.execute_plan` / `execute_plan_metered` FFI entry points:

| query | total | native | Python | Python % |
|-------|------:|-------:|-------:|---------:|
| filter-project |  11.3 |   6.8 |   4.5 | 40% |
| groupby-sum    |   5.9 |   4.1 |   1.8 | 31% |
| tpch-q1        |  23.7 |  18.7 |   5.0 | 21% |
| tpch-q9        | 148.1 | 110.7 |  37.4 | 25% |
| tpch-q8        | 132.4 |  68.7 |  63.8 | **48%** |

On q8, **half the query is the Python optimizer**. But note the other half: even at zero
planning cost, q8's native execution (68.7 ms) is 3× DuckDB's entire query (22.8 ms). Both
sides need work; neither alone closes the gap.

## Root cause 1 — the optimizer is the query

Splitting `collect()` into optimize vs. execute (sf1, best-of-5):

| query | total_ms | kyber_ms | exec_ms |
|-------|---------:|---------:|--------:|
| q2  |  99.7 | **72.2** | 27.5 |
| q8  | 186.9 | **90.6** | 96.3 |
| q7  | 130.2 | **58.7** | 71.4 |
| q1  |  26.2 |    4.3   | 21.9 |

Kyber spent **90 ms planning q8 — 3.4× DuckDB's entire end-to-end query time.** Profiling
showed 6.8M Python calls per `optimize_full`. The cause was structural, not algorithmic:
`transform_expr_up` and `map_node_expressions` rebuilt every expression node and every
plan node on every rule, on every fixpoint iteration — discarding each node's memoized
`to_ir` and re-running the enclosing node's column validation (38k `_validate_refs`
calls and 22k `to_ir` calls per optimize).

**Fixed:** structural sharing in the neutral `plan/` rewrite layer. A node whose
rewritten children are all the same objects (`is`) is not rebuilt. Nodes are immutable
and value-typed, so reusing one is indistinguishable from rebuilding an equal copy —
except that the optimizer's `is`-based fixpoint detection and per-node memo caches now
hit. **Kyber time roughly halved** (q8 90→58 ms, q1 4.3→2.4 ms).

This is necessary but nowhere near sufficient: 58 ms of planning still dwarfs DuckDB's
27 ms total. See "What remains".

## Root cause 2 — cold-start cardinality is blind

`explain(analyze=True)` on q9, first execution:

```
hash_join  est≈6,001,215  actual=6,001,215   out=2.6GB
hash_join  est≈6,001,215  actual=6,001,215   out=2.0GB
  ...
hash_join  est≈6,001,215  actual=  319,404   out=28MB    ← the selective join, last
  filter   est≈  100,000  actual=   10,664             ← p_name LIKE '%green%'
```

Batcher joined `lineitem ⋈ partsupp ⋈ orders` first — materializing **2.0 GB and then
2.6 GB** — and only then applied the `part` filter that cuts 200k rows to 10.7k. Three
independent estimator defects produced that plan:

1. **`LIKE '%green%'` was estimated at 0.5 selectivity.** Every string-pattern predicate
   fell through to `default_filter_selectivity`. So did every `IN` list. (`p_name LIKE
   '%green%'` actually keeps 5.3%.)
2. **A range predicate against a column with known exact `min`/`max` used the Selinger
   1/3 constant.** Footer statistics carry exact bounds for every column from the first
   query on; quantiles only appear after the learning loop has measured a run. TPC-H q1,
   q3–q7, q12, q14, q15 and q20 all filter on a date interval, and all of them estimated
   a flat 1/3 against an exactly-known date span.
3. **`join_columns` dropped `ndv` unconditionally**, so every join *above* a join had no
   key distinct-count and `_estimate_join` fell back to `max(|L|, |R|)` — which, for any
   join involving `lineitem`, is 6,001,215 regardless of what the join actually does.

Worse, defect 3 was masking a deeper one: **`ndv` was `None` for every column, cold.**
No Parquet footer carries a distinct count. `ndv` only ever arrived from the *post-run*
learning loop (`learn_column_stats`), which is why the second execution of q9 picks a
completely different — and correct — plan:

```
run 2:  hash_join est≈302,330 actual=319,404 (1.1x)  out=32MB
        hash_join est≈300,258 actual=319,404 (1.1x)  out=56MB
```

The adaptive loop works. The *first* run of every query planned blind.

**Fixed:**

* String-pattern and `IN`-list selectivity (`substring_selectivity` = 0.05,
  `prefix_selectivity` = 0.10 — the conventional optimizer defaults; cold-start values
  only, since the learning loop replaces them with the measured selectivity).
* Range selectivity now interpolates across a column's exact `min`/`max` under a
  uniformity assumption when no quantile grid exists yet, with `date`/`datetime`/
  `Decimal` mapped to a common ordinal.
* `join_columns` carries `ndv` forward as `min(ndv_in, rows_out)` — sound, because a
  join invents no values and the output row count trivially bounds the distinct count.
* **`core.column_ndv`** — a new HLL-only, rayon-parallel, GIL-releasing FFI entry point.
  `column_stats` could not serve cold-start seeding because it also builds a KLL quantile
  sketch (~50 ns/row, ~7× the HLL). The new path measures **5 columns × 6M rows in
  11.9 ms** with <1% error, and is bit-identical to the sequential HLL (the sketch is
  `Mergeable`, so the parallel fold is exact, and a unit test pins that).
* **`seed_column_ndv`** — the conductor sketches a resident source's not-yet-measured
  columns *before* Kyber runs. Idempotent, once per column, budget-capped
  (`optimizer.ndv_sketch_max_cells`), and recorded through the existing `SKETCH`-provenance
  learned channel so an approximate distinct count can never answer an exact
  `count_distinct`.

A subtlety worth recording: `learn_column_stats` used to gate on "does this column have
an `ndv` yet?". Seeding `ndv` early would have silently suppressed it, losing the
quantile grids and most-common-values that only it produces. It now gates on the average
byte width, which is written for every column it measures.

### Effect (cold, first execution, sf1 — deterministic, contention-independent)

| | before | after |
|---|---:|---:|
| total peak intermediate bytes, 21 queries | **87.98 GB** | **25.08 GB** |
| geomean q-error across all operators | **9.06** | **3.34** |
| q5 peak intermediate | 64 GB | 1.5 GB |
| q1 geomean q-error | 293.3 | 1.23 |
| q12 geomean q-error | 50.2 | 2.21 |
| q9 geomean q-error | 4.51 | 1.92 |

The cold plan for q9 is now identical to the warm one.

## Root cause 3 — sketch primitives are far slower than they look

Measured on 6M-row `lineitem` columns via `core.column_statistics`:

| column | type | ns/row |
|--------|------|-------:|
| `l_orderkey` | int64 | 343 |
| `l_extendedprice` | double | 454 |
| `l_comment` | string | 335 |
| `l_shipdate` | date32 | **39** |

An HLL pass over int64 should cost single-digit nanoseconds per row. Two findings:

* **`date32`/`timestamp` columns get no quantiles at all.** `column_stats_full` never
  applies `temporal_cols_as_i64`, and `KllSketch::add_array` bails when `cast(date32 →
  float64)` fails. `l_shipdate`'s quantiles come back `None` — which is why it is the
  *fast* column here, and why every TPC-H date range predicate had no histogram to use.
  (Working around this is what motivated the exact-bounds interpolation above; the
  underlying gap is still open.)
* **KLL's level-0 compactor capacity collapses to 2.** `capacity(h)` is
  `k·c^(levels−1−h)` with `c < 1`, so as levels grow the *bottom* compactor's capacity
  decays to the `max(2, …)` floor. `add()` then calls `compress()` every two values, and
  `compress()` walks every level calling `powi` per level. That is the ~300 ns/row.

Neither is fixed yet. `column_ndv` sidesteps both by not building a KLL.

## What was changed, and what it did

Honest ledger. Two of these move wall-clock, three move plan quality, three are correctness
or harness fixes, and one is hygiene with no measured effect.

| change | effect |
|--------|--------|
| Structural sharing in `plan/expr_rewrite` (unchanged subtrees keep identity) | Kyber time roughly halved (q8 90→58 ms) |
| Cold-start `ndv` seeding (`core.column_ndv` + `seed_column_ndv`) | cold peak intermediates 87.98→25.08 GB; cold q-error 9.06→3.34 |
| `Cast(Lit)` constant folding | TPC-H Q1's filter estimate 3.0× off → 1.0× |
| Range selectivity from exact `min`/`max` | Q3's date filters 1.0×; unlocks 10 queries' date predicates |
| String-pattern / `IN` selectivity | Q9's `LIKE '%green%'` estimate 100k → 10.0k (actual 10.7k) |
| Harness oracle = DuckDB, not "first engine" | Q6's Polars/Daft bug no longer reported as Batcher's |
| Spark adapter: JVM detection + Parquet registration | Spark runs at all (was `JAVA_GATEWAY_EXITED`, then driver-heap OOM) |
| `auto_width` caps workers by available morsels | correct and tested, but **no measured speedup** — the pool is cached, so a wide pool is nearly free |
| Widen `Utf8`/`Binary` past 32-bit offsets in `materialize` | removed a reachable panic (`attempt to add with overflow`) |
| Slice-aware byte accounting for that check | fixed the regression it caused in sf10 Q12/Q14 |
| Parallel radix partition (`join::radix`) | join scaling now extends past 32 workers; bit-identical output, 4 new tests. Win not cleanly measurable under the machine's contention |
| Restrict `ndv` seeding to join/group/equality columns | sf10 `lineitem` now fits the sketch budget (3 keys, not 16 columns) |
| ClickBench temporal reconstruction in `sources.py` | q18/q36/q37/q41 went from "all engines failed" to passing |
| Case-insensitive column comparison in the harness | ClickBench failures 24 → 9; the survivors are real or non-deterministic |

The cold-plan wins are real and large, but note what they do *not* do: the benchmark harness
warms up before timing, and the post-run learning loop supplies `ndv` from run 2 onward. So
the wall-clock TPC-H table barely moved. Cold planning matters for queries that run once —
most ad-hoc analytics — and it is what stops sf10's Q8 from planning blind. It is not what
closes the DuckDB gap.

## What "2× faster than DuckDB" actually requires

Measured on a quiet machine, splitting each query at the `_native.execute_plan` boundary:

| query | batcher total | native | Python | duckdb | 2× target | native must get |
|-------|-------------:|-------:|-------:|-------:|----------:|----------------:|
| q1 |  26.1 |  20.9 |  5.2 | 15.0 |  7.5 | **2.8× faster** |
| q8 | 136.1 |  72.5 | 63.6 | 22.5 | 11.2 | **6.4× faster** |
| q9 | 155.0 | 116.3 | 38.6 | 62.6 | 31.3 | **3.7× faster** |

Read the last column carefully: it assumes the **entire Python control plane costs zero** —
Kyber, lowering, FFI, Arrow table construction, all of it. Even then the native engine has
to get between 2.8× and 6.4× faster. No amount of optimizer work reaches the goal; the
engine is the whole story.

The headroom exists, and it is exactly the parallel efficiency gap. `join-agg` at sf10 runs
5111 ms on one worker and 576 ms on 96 — 8.9×, where linear scaling would give ~53 ms. That
is a ~10× shortfall sitting in three places, in order of size:

1. the hash-table **build** is a serial loop over every build row;
2. `materialize` copies **both join inputs into one batch** before every join — a full
   copy of the largest relation in the query, and the thing the architecture's own
   performance rule ("don't introduce whole-relation materialization where a streaming /
   morsel path is possible") forbids;
3. per-morsel output allocation in `filter_batch_jit` and `gather_join_output`.

Fixing (2) — probing the build table directly from the probe side's morsels instead of
concatenating them — removes a copy, removes the 2 GiB `Utf8` ceiling, and is the change
the streaming/distributed story wants anyway. It is the single highest-value item in this
document.

## What remains before Batcher beats DuckDB by 2×

**The 2× goal is not met, and is not close.** Batcher is 1.3×–7.2× slower at sf1 and
2.0×–15.1× slower at sf10. Ranked by evidence:

1. **Parallel efficiency (Root cause 0).** The engine stops scaling at 16–32 of 96 threads
   and its default width is the worst setting on three of four shapes. This is the only
   finding that explains why the gap *widens* with scale. The work: cap the auto width at
   physical cores; wire Kyber's existing per-operator parallelism annotation through to the
   scoped pool instead of using one global width; investigate the per-morsel output
   allocation in `filter_batch_jit` and the `current_num_threads()`-keyed shuffle bucket
   count as contention sources. Expect the largest single win here.
2. **Kyber's per-query cost.** 48% of q8. Node-local rules are already fused into one
   traversal, but ~10 *plan* rules (6 `sarg_*`, fold, simplify, `like_prefix_to_range`,
   `date_trunc_to_range`) each re-walk every expression of every node, and `join_reorder`
   alone costs 15 ms. The fix is an `expr_rule` kind whose leaves the driver composes into a
   single `transform_expr_up` per node — the same fusion `_apply_node_rules` already does one
   level up, and what the "designed for hundreds of rules" architecture calls for.
3. **Correlated subqueries (q21).** A correctness gap. Batcher, Polars and Daft all fail it;
   DuckDB and Spark do not.
4. **`starts_with` is not range-sargable.** The SQL frontend lowers `LIKE 'abc%'` to
   `StrFunc(starts_with)`, but `like_prefix_to_range` only matches `fn == "like"` — so the
   prefix→range rewrite never fires for SQL, and zone-map pruning and Parquet predicate
   pushdown stay blind to Q14's and Q20's prefix filters. The rule also only inspects each
   node's *top-level* expression, so a nested `LIKE` under an `AND` would be missed too.
5. **KLL and date quantiles (Root cause 3).** `date32`/`timestamp` columns get no quantile
   sketch at all, and KLL's level-0 compactor thrashes at ~300 ns/row. Together they make the
   first run of every query pay seconds and deny the estimator a histogram on exactly the
   columns TPC-H filters on.
6. **A fair Polars comparison.** Wire `suites/standard/tpch_dataframe.py` into the default
   lineup so Polars is measured on its LazyFrame API rather than its SQL frontend, which
   fails 9 of 22 queries and is wrong on q6.
7. **`--scan`-mode cold `ndv`.** `seed_column_ndv` only sketches resident sources; a Parquet
   scan still plans its first query blind. Sampling row groups with a principled distinct
   estimator (GEE/Chao) is the follow-up — a naive leading-rows sample underestimates a
   clustered key like `l_orderkey` by ~10× and is worse than no estimate at all.

## Verification
Everything above is gated on correctness, not just on plans:

* `tests/differential/` — 1454 cases, all green, including a new
  `test_diff_cast_constant_fold.py` pinning the plan-time `cast` folding against DuckDB
  on both the folded cases and the cases the folder deliberately refuses.
* `tests/unit/test_cardinality_cold_start.py` — the estimator contracts (bounds
  interpolation, pattern/`IN` selectivity, `ndv` propagation, and that a join never
  reports `EXACT` provenance).
* `tests/unit/test_seed_column_ndv.py` — the parallel HLL agrees with the sequential
  one, seeding is idempotent, non-resident sources are skipped, the cell budget is
  honoured, and a broken source never breaks a query.
* `cargo test --workspace --exclude bc-py` — green.
