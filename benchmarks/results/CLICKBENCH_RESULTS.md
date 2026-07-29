# ClickBench -- Batcher against every comparator (2026-07-28)

The 43-query ClickBench suite over the ClickHouse `hits` dataset, run across every
SQL-capable engine in the harness. Companion to `TPCH_SF1_SF10_RESULTS.md`.

Every timing is correctness-gated: the harness compares each engine's result against the
DuckDB reference as a sorted row multiset before it will report a number, so a fast wrong
answer surfaces as `FAILED` and never as a win.

## Scale, and why it is not 100M rows

Upstream ClickBench is 100M rows. This harness materializes the table to **in-memory
Arrow** and shares it across every engine, each of which then builds its own handle --
DuckDB copies into its compressed native store, Polars and Daft copy, Spark writes
Parquet. At 10 parts the table is **10M rows, 105 columns, 7.21 GiB in Arrow**; at 100
parts the engine copies alone would not fit beside each other on this host. The scale is
stated rather than implied, and every engine sees byte-identical input.

## The two DuckDB bars

`duckdb` is DuckDB in its **compressed native store** (dictionary/RLE, zone maps), loaded
by an untimed `CREATE TABLE`. `duckdb_arrow` binds DuckDB to the **same zero-copy Arrow**
Batcher consumes. The first is DuckDB at its best; the second isolates execution from
storage format. ClickBench is a single wide table of low-cardinality strings, which is
exactly where compression pays most, so the gap between the two bars is unusually large
here -- and reading only one of them would badly mislead in either direction.

## Environment

| | |
|---|---|
| Host | c5d.24xlarge, 96 vCPU, 184 GiB RAM, us-west-2 |
| Batcher | `maturin develop --release`, verified `__engine_profile__ == "release"` |
| Data | `datasets.clickhouse.com/hits_compatible/athena_partitioned`, 10 parts, mirrored to local NVMe |
| Repeats | best-of-5 after one warm-up |
| Versions | duckdb 1.5.5, polars 1.36.1, pyspark 4.2.0 (JRE 17), daft 0.7.21 |

```bash
export BENCH_CLICKBENCH_BASE=/mnt/local_storage/clickbench BENCH_CLICKBENCH_PARTS=10
python3 benchmarks/run.py --benchmark clickbench \
    --engines batcher,duckdb,duckdb_arrow,polars,spark,daft
```

## Times (ms)

| q | batcher | duckdb | ddb_arrow | polars | spark | daft |
|---|--:|--:|--:|--:|--:|--:|
| q00 | 0.1 | 1.2 | 13.5 | 0.1 | 139.3 | 5.8 |
| q01 | 0.3 | 4.1 | 193.6 | 4.0 | 132.9 | 9.1 |
| q02 | 0.2 | 1.9 | 14.1 | 16.7 | 134.9 | 9.1 |
| q03 | 0.2 | 1.8 | 17.6 | 13.6 | 121.6 | wrong |
| q04 | 0.2 | 17.0 | 38.9 | 82.9 | 462.4 | 28.5 |
| q05 | 0.2 | 30.1 | 54.7 | 356.5 | 690.6 | 36.3 |
| q06 | 0.2 | 2.0 | 15.4 | 3.1 | 96.4 | 9.0 |
| q07 | 4.1 | 2.7 | 199.1 | 12.0 | 164.3 | 12.7 |
| q08 | 47.9 | 27.3 | 49.5 | 68.1 | 505.3 | 108.0 |
| q09 | 49.3 | 43.3 | 73.3 | 83.0 | 613.5 | 78.8 |
| q10 | 16.9 | 16.5 | 63.9 | 40.5 | 224.0 | 26.0 |
| q11 | 15.2 | 15.6 | 66.8 | 37.3 | 239.2 | 24.5 |
| q12 | 45.5 | 32.6 | 208.0 | 172.9 | 659.9 | 56.6 |
| q13 | 54.8 | 64.4 | 240.3 | 192.5 | 1029.2 | 246.1 |
| q14 | 56.8 | 31.3 | 204.9 | 196.8 | 732.1 | 64.0 |
| q15 | 33.4 | 26.8 | 48.2 | 96.9 | 423.4 | 39.2 |
| q16 | 81.6 | 47.6 | 86.3 | 364.3 | 1034.6 | 97.7 |
| q17 | 72.5 | 46.1 | 82.4 | 374.1 | 1101.8 | 81.8 |
| q18 | 103.2 | 75.9 | 137.3 | 861.9 | 1707.6 | n/a |
| q19 | 4.5 | 1.4 | 15.7 | 7.9 | 80.2 | 18.2 |
| q20 | 18.3 | 20.3 | 47.4 | 1651.4 | 429.4 | 21.4 |
| q21 | 12.8 | 20.9 | 211.4 | 1673.9 | 525.6 | 31.1 |
| q22 | 20.8 | 35.7 | 182.1 | 3698.3 | 1031.5 | 47.0 |
| q23 | 62.3 | 35.3 | 172.8 | 74.1 | wrong | 28.1 |
| q24 | 14.5 | 5.9 | 174.0 | 70.8 | 285.6 | 25.6 |
| q25 | 16.9 | 8.0 | 186.4 | 114.3 | 222.0 | 31.7 |
| q26 | 12.3 | 7.5 | 181.3 | 129.5 | 251.4 | 27.4 |
| q27 | 26.8 | 35.3 | 601.8 | 495.1 | 723.5 | 69.1 |
| q28 | 412.7 | 561.4 | 1913.6 | n/a | wrong | n/a |
| q29 | 5.8 | 12.4 | 24.9 | 57.1 | 613.8 | 190.8 |
| q30 | 45.1 | 26.2 | 194.5 | 104.6 | 521.6 | 35.3 |
| q31 | 63.3 | 27.4 | 208.1 | 128.8 | 643.3 | 45.3 |
| q32 | 203.7 | 159.1 | 172.0 | 756.7 | 1664.6 | 220.2 |
| q33 | 128.3 | 84.4 | 152.5 | 885.0 | 1674.1 | 219.3 |
| q34 | 132.8 | 101.6 | 146.9 | 858.2 | 1636.7 | 225.8 |
| q35 | 43.6 | 30.3 | 65.1 | n/a | 467.1 | 65.8 |
| q36 | 57.9 | 19.3 | 338.9 | 107.2 | 626.7 | 66.9 |
| q37 | 44.0 | 12.2 | 326.9 | 54.3 | 413.0 | 41.4 |
| q38 | 20.5 | 8.9 | 126.1 | 32.6 | 281.5 | 23.4 |
| q39 | 121.3 | 55.1 | 367.7 | 223.3 | 1191.7 | 249.2 |
| q40 | 12.3 | 7.4 | 144.3 | 113.7 | 172.1 | 22.6 |
| q41 | 11.1 | 5.4 | 149.4 | 30.5 | 145.3 | 20.4 |
| q42 | 15.7 | 9.5 | 286.9 | n/a | wrong | 19.8 |
| **total** | **2089.9** | **1779.1** | **8198.5** | **14244.5** | **23813.7** | **2679.0** |

## Ratios (`b/x`; below 1.00x means Batcher is faster)

| q | duckdb | ddb_arrow | polars | spark | daft |
|---|--:|--:|--:|--:|--:|
| q00 | 0.08x | 0.01x | 1.00x | 0.00x | 0.02x |
| q01 | 0.07x | 0.00x | 0.07x | 0.00x | 0.03x |
| q02 | 0.11x | 0.01x | 0.01x | 0.00x | 0.02x |
| q03 | 0.11x | 0.01x | 0.01x | 0.00x | -- |
| q04 | 0.01x | 0.01x | 0.00x | 0.00x | 0.01x |
| q05 | 0.01x | 0.00x | 0.00x | 0.00x | 0.01x |
| q06 | 0.10x | 0.01x | 0.06x | 0.00x | 0.02x |
| q07 | 1.52x | 0.02x | 0.34x | 0.02x | 0.32x |
| q08 | 1.75x | 0.97x | 0.70x | 0.09x | 0.44x |
| q09 | 1.14x | 0.67x | 0.59x | 0.08x | 0.63x |
| q10 | 1.02x | 0.26x | 0.42x | 0.08x | 0.65x |
| q11 | 0.97x | 0.23x | 0.41x | 0.06x | 0.62x |
| q12 | 1.40x | 0.22x | 0.26x | 0.07x | 0.80x |
| q13 | 0.85x | 0.23x | 0.28x | 0.05x | 0.22x |
| q14 | 1.81x | 0.28x | 0.29x | 0.08x | 0.89x |
| q15 | 1.25x | 0.69x | 0.34x | 0.08x | 0.85x |
| q16 | 1.71x | 0.95x | 0.22x | 0.08x | 0.84x |
| q17 | 1.57x | 0.88x | 0.19x | 0.07x | 0.89x |
| q18 | 1.36x | 0.75x | 0.12x | 0.06x | n/a |
| q19 | 3.21x | 0.29x | 0.57x | 0.06x | 0.25x |
| q20 | 0.90x | 0.39x | 0.01x | 0.04x | 0.86x |
| q21 | 0.61x | 0.06x | 0.01x | 0.02x | 0.41x |
| q22 | 0.58x | 0.11x | 0.01x | 0.02x | 0.44x |
| q23 | 1.76x | 0.36x | 0.84x | -- | 2.22x |
| q24 | 2.46x | 0.08x | 0.20x | 0.05x | 0.57x |
| q25 | 2.11x | 0.09x | 0.15x | 0.08x | 0.53x |
| q26 | 1.64x | 0.07x | 0.09x | 0.05x | 0.45x |
| q27 | 0.76x | 0.04x | 0.05x | 0.04x | 0.39x |
| q28 | 0.74x | 0.22x | n/a | -- | n/a |
| q29 | 0.47x | 0.23x | 0.10x | 0.01x | 0.03x |
| q30 | 1.72x | 0.23x | 0.43x | 0.09x | 1.28x |
| q31 | 2.31x | 0.30x | 0.49x | 0.10x | 1.40x |
| q32 | 1.28x | 1.18x | 0.27x | 0.12x | 0.93x |
| q33 | 1.52x | 0.84x | 0.14x | 0.08x | 0.59x |
| q34 | 1.31x | 0.90x | 0.15x | 0.08x | 0.59x |
| q35 | 1.44x | 0.67x | n/a | 0.09x | 0.66x |
| q36 | 3.00x | 0.17x | 0.54x | 0.09x | 0.87x |
| q37 | 3.61x | 0.13x | 0.81x | 0.11x | 1.06x |
| q38 | 2.30x | 0.16x | 0.63x | 0.07x | 0.88x |
| q39 | 2.20x | 0.33x | 0.54x | 0.10x | 0.49x |
| q40 | 1.66x | 0.09x | 0.11x | 0.07x | 0.54x |
| q41 | 2.06x | 0.07x | 0.36x | 0.08x | 0.54x |
| q42 | 1.65x | 0.05x | n/a | -- | 0.79x |

## Summary

| engine | total | geomean b/x | Batcher wins |
|---|--:|--:|--:|
| batcher | 2089.9 ms | -- | -- |
| duckdb (native store) | 1779.1 ms | **0.828x** | 15/43 |
| duckdb_arrow (same input) | 8198.5 ms | **0.131x** | **42/43** |
| polars | 14244.5 ms | 0.137x | 39/40 |
| spark | 32886.3 ms | 0.031x | 43/43 |
| daft | 2687.6 ms | 0.319x | 37/41 |

Batcher is **geometrically faster than DuckDB's compressed native store** (0.828x) and
7.6x faster than DuckDB's execution engine on the same Arrow input. Its total is still
higher than DuckDB's (2089.9 ms against 1779.1 ms) because a handful of heavy group-by
queries dominate the sum while the geomean weights all 43 equally -- both numbers are
reported rather than whichever flatters.

## What changed this round

Two per-row inefficiencies were found by profiling with `perf` and fixed in `bc-expr`
(commit `466ed37`). Neither changes a result.

| q | before | after | speedup |
|---|--:|--:|--:|
| q28 | 2180.4 ms | 412.7 ms | **5.28x** |
| q42 | 28.5 ms | 15.7 ms | **1.82x** |
| q38 | 30.6 ms | 20.5 ms | **1.49x** |

Suite total: **3907.9 ms -> 2089.9 ms**; geomean against DuckDB-native **0.950x ->
0.828x**; wins against `duckdb_arrow` **41/43 -> 42/43**.

### 1. `regexp_replace` allocated a capture buffer per row

q28 (`REGEXP_REPLACE` over 10M `Referer` values) was 73% of the entire deficit against
DuckDB -- 1634.7 ms of a 2236.7 ms gap. `perf` showed only ~10% of it was the regex
search:

| % | function |
|--:|---|
| 29.92% | `Regex::create_captures` (per-row allocation) |
| 18.17% | `Regex::replacen` |
| 15.94% | `CaptureMatches::next` |
| 10.81% | dropping the per-row iterator |
| 7.92% | the actual regex search |

`Regex::replace` builds a fresh `Captures` and iterator per call and returns a `Cow` that
`into_owned` copies again. The loop now reuses one `CaptureLocations` per column and
writes directly into the Arrow builder.

Hand-rolling the scan loop meant re-deriving the library's match semantics, and the
differential test written against the regex crate's own replacer **caught a bug before it
shipped**: an empty match ending where the previous match ended is skipped, so `a*` over
`"ab"` yields `"a"` then the empty match at 2, not a second empty match at 1. The test
sweeps 11 patterns x 7 templates x 14 subjects x {first, global}, including
empty-matching patterns and multibyte subjects.

### 2. A string literal against a temporal column was parsed per row

`EventDate >= '2013-07-01'` materializes the literal to an N-row `StringArray`, which
`coerce_numeric` then casts -- parsing the same ISO string once per row (6.3% of q39 in
`arrow_cast::parse::parse_date`). The scalar fast path declined the pair because the types
differ; it now casts the one-element literal and broadcasts. `cast` is elementwise, so
that is the same value, and the same error on an unparseable string, as casting the
materialized copy.

A bare date-range filter over `hits` went **24.7 ms -> 8.6 ms**, matching the
already-typed `DATE '...'` spelling (7.5 ms). This shape -- an ISO string compared against
a date column -- is ubiquitous in real SQL, so the win is not ClickBench-specific.

## Further optimization attempts, all measured and rejected

After the two wins above, 28 queries still trailed DuckDB's native store by 566 ms in
total. Every remaining structural explanation was tested and none held. Recorded so the
next attempt does not repeat them:

| hypothesis | measurement |
|---|---|
| Dictionary-encode the string group keys (Arrow has `DictionaryArray`, so it is inside the columnar contract) | The columns are **high**-cardinality: `URL` 2.62M distinct of 10M rows (26%), `Referer` 27%, `Title` 16%, `SearchPhrase` 8%. Dictionary encoding buys almost nothing here and the group-by still faces millions of distinct keys |
| Larger morsels to cut task-dispatch and epoch-GC pressure (worth 20% on a TPC-H filter) | Full-suite sweep: 16,384 rows 2107.7 ms, 65,536 2095.8, 262,144 2095.2 -- **0.6%**, inside noise, and only 22/43 queries improve. Does not generalize from the TPC-H case |
| A hotspot in the heaviest group-by queries | q37's profile is flat: nothing above 5.7%, ~22% accounted across the top eight symbols. There is no q28-shaped lever left |

### Worker over-subscription is real, but not shippable as a constant

The one effect that reproduced on **both** benchmarks is that the engine uses every logical
core and loses throughput doing it. Full-suite sweep at 10M rows:

| workers | total | vs 96 |
|--:|--:|--:|
| 48 | 2150.1 ms | -0.1% |
| 64 | **2018.9 ms** | **+6.0%**, better on 32/43 |
| 80 | 2087.8 ms | +2.8% |
| 96 (default) | 2147.6 ms | -- |

At 64 workers Batcher wins **18/43** against the native store rather than 15/43. The effect
is genuine and matches TPC-H, where 64 also beat 96 (filter-count 25.5 ms against 27.6 ms).

It is **not** committed, because the optimum is not a topology boundary this host explains:
the box is 48 physical cores / 96 logical, and **48 performs no better than 96**. A hardcoded
64 would be tuning to one machine, which `.claude/rules/performance.md` forbids. The right
form is an adaptive worker-width policy that *measures* -- a Carbonite resource decision, and
exactly the adaptive re-optimization the contract calls the moat -- rather than a constant.

### Resolved: the combine gather is memory-bound, not instruction-bound

Profiling q32 (the heaviest query, `GROUP BY WatchID, ClientIP` over ~10M near-unique
groups) showed 27% of it inside `arrow_select::interleave` -- 22.65% on `Int64` plus 4.39%
on `Float64`. The call site is **not** `ops/repartition.rs` (instrumenting that file showed
it never runs for this query) but `bc-runtime::agg::group::combine`, which gathers each key
and state column out of the partials by `(partial, row)` address.

That looked like the same trade `ops/repartition.rs` already makes on the shuffle side,
where a flat typed gather replaced `interleave` precisely because the pair form costs
"sixteen bytes of index per output row". So it was implemented the same way: `u32` planes
for the addresses instead of `Vec<(usize, usize)>`, and a direct value copy for a null-free
primitive column, falling back to `interleave` for strings, nested types and nulls.

**It did not pay, and the change was reverted.** The fast path was confirmed taken -- the
profile's top symbol became `combine::gather_primitive<Int64Type>` at 18.84%, replacing
`interleave_primitive` at 22.65% -- but end-to-end time did not move (q32 203.7 ms ->
213.8 ms, inside noise) and the string-key queries **regressed** (q33 0.91x, q16 0.87x,
q17 0.87x) because the fallback now builds both index forms.

The conclusion is the useful part: this gather is **memory-latency-bound**. It reads 10M
rows scattered across the partials, so the cost is cache misses, and halving the index
bytes or skipping Arrow's builder changes nothing. The same reasoning explains why
`assign_groups_int64_multi` (15.21%) sits beside it -- a hash table with ~10M live entries
is misses all the way down. Making these queries faster needs a different data structure
(a radix-partitioned layout with sequential access, or a cache-conscious aggregation), not
a cheaper gather.

## Where Batcher still trails the native store

The remaining losses are group-by-heavy queries with high-cardinality string keys
(`GROUP BY URL`, `GROUP BY Title`, and q39's five-key group-by including two string
columns):

| q | b/duckdb | shape |
|---|--:|---|
| q37 | ~4.2x | filter then `GROUP BY Title` |
| q36 | ~3.0x | filter then `GROUP BY URL` |
| q39 | ~2.5x | five-key group-by, two string keys |

Profiling q39 put ~13% in `crossbeam_epoch` (rayon scheduling) and the rest in the
group-by itself. This is the same storage-versus-execution split the TPC-H document
records: on identical Arrow input Batcher wins these queries comfortably, and DuckDB's
advantage comes from grouping dictionary-encoded columns that Batcher, under the
`Arrow is the only columnar contract` invariant, reads as raw strings.

## Correctness

Batcher answered **all 43 queries correctly**. Every failure in the run belongs to another
engine:

- **Daft q03** returns `-653315757734.8666` for `AVG(UserID)` where the answer is
  `2.5131007489380997e+18` -- an overflow, not a rounding difference.
- **Daft q18** cannot plan (`Expected input to minute to be temporal, got UInt32`).
- **Spark q23 / q42** render timestamps with nanosecond precision
  (`2013-07-03 04:00:27.000000000`) where the reference has microseconds; **q28** returns
  15 rows against the reference's 1.
- **Polars q35** rejects the query (`group_by keys contained duplicate output name`).

Those cells read `wrong` in the tables above and are excluded from the affected engine's
total and geomean, so no engine is credited for being fast at being wrong.

**PyArrow is absent.** ClickBench is SQL-only in this suite and, unlike TPC-H, carries no
hand-written native pipeline for it.

## Regression check

Both fixes touch shared `bc-expr` kernels, so TPC-H was re-run: sf1 batcher **625.1 ms**
against 617.5 ms before (inside run-to-run noise), duckdb 634.0 ms, duckdb_arrow
1673.4 ms, zero correctness failures. The full Rust workspace suite passes, and clippy is
clean under `-D warnings`.
