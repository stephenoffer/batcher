# Where Batcher's remaining benchmark gap actually is (2026-07-28)

A record of every optimization hypothesis tested against the TPC-H and ClickBench results,
what it measured, and what shipped. Two succeeded. Thirteen did not, and each is written
down with the number that killed it so the next attempt starts from evidence.

The scorecards themselves are `TPCH_SF1_SF10_RESULTS.md` and `CLICKBENCH_RESULTS.md`.

## The one-paragraph conclusion

**Batcher's execution engine already beats DuckDB's on identical input: 84 of 87 queries
across TPC-H sf1, TPC-H sf10 and ClickBench** (`duckdb_arrow`, geomean 0.353x / 0.421x /
0.131x). The residual against DuckDB's *native compressed store* is not a set of hot spots
that can be optimized away -- it is two structural properties of that store, and closing
either is an engine project rather than a tuning. Every attempt to find a third explanation
failed.

## What shipped

| change | effect | commit |
|---|---|---|
| `regexp_replace` reuses one `CaptureLocations` per column instead of allocating a `Captures` per row | ClickBench q28 **2180.4 ms -> 412.7 ms (5.28x)**, flipping 4.00x behind DuckDB to 0.67x ahead | `466ed37` |
| A string literal against a temporal column is parsed once, not per row | date-range filter **24.7 ms -> 8.6 ms**; ClickBench q42 28.5 -> 15.7 ms | `466ed37` |
| Ray Data registered through the Parquet connector instead of `from_arrow` (one block, so every operator ran single-threaded) plus native pipelines for all 22 TPC-H queries | coverage 4/22 -> 22/22, no longer measured on one core | `b2dab69` |

ClickBench overall: **3907.9 ms -> 2089.9 ms**, geomean against the native store
**0.950x -> 0.828x**.

Both engine changes share a shape worth naming: *work that is constant per column was being
redone per row*. That is the class of bug profiling finds reliably. Nothing else in the
remaining gap has that shape.

## The two structural causes

### 1. DuckDB reads fewer bytes

Its native store dictionary/RLE/FSST-compresses columns and skips row groups on zone maps.
Batcher reads uncompressed Arrow, because `Arrow is the only columnar contract`
(CLAUDE.md #3). On the *same* Arrow bytes Batcher is 2.4x faster at sf10, so this is a
storage-format difference, not an execution one.

### 2. The high-cardinality group-by is memory-latency-bound

ClickBench q32 (`GROUP BY WatchID, ClientIP`, ~10M near-unique groups) spends ~34% between
`assign_groups_int64_multi` and the partial gather, and neither is instruction-bound: a hash
table with ~10M live entries misses to DRAM on essentially every probe. This is why the
cheaper-gather and partition-sizing attempts below both failed -- they reduce instructions,
and instructions are not the constraint.

## Rejected: measured, and why

### Storage-side

| hypothesis | measurement |
|---|---|
| Zone-map / min-max morsel pruning over Arrow | **0%** of 16,384-row morsels are skippable for the real TPC-H predicates (`l_shipdate`, `l_discount`, `l_quantity`). Only the clustered `l_orderkey` prunes (74.9%), and no benchmark query filters on it |
| Dictionary-encode low-cardinality columns (Arrow has `DictionaryArray`, so this is in-contract) | See below -- **correct but 5x to 14x slower today**, and only an 11% byte reduction |

The dictionary result deserves its numbers. TPC-H `lineitem` looks ideal for it:
`l_shipinstruct` is **4 distinct values in 959.7 MB**, `l_returnflag` 3 in 299.9 MB,
`l_linestatus` 2 in 299.9 MB, `l_shipmode` 7 in 497.0 MB. Encoding the low-cardinality
string columns of every table and re-running:

| query | plain | dictionary | |
|---|--:|--:|--:|
| q1 | 165.8 ms | 880.7 ms | **0.19x** |
| q12 | 118.5 ms | 787.8 ms | **0.15x** |
| q19 | 128.6 ms | 1819.1 ms | **0.07x** |
| q5 | 548.2 ms | 325.0 ms | 1.69x |

Every result was **correct**, so the operators handle the type -- they just decode it. The
profile is dominated by `memmove` and kernel time, the signature of materializing the
decoded column per operator. And the whole-dataset footprint fell only 12.32 GB -> 10.98 GB
(11%), because a 4-byte dictionary index replaces a 4-byte offset plus a short string, so
the saving is real only on wide values.

Two things follow. Dictionary encoding is **not** a shortcut to DuckDB's storage advantage
-- the byte reduction is far smaller than the compression ratio suggests. And Batcher has a
**performance cliff on dictionary-encoded input**, which matters independently of
benchmarks: that is the shape Parquet produces natively, so a user reading dictionary
Parquet hits up to 14x today. Making it a win means dictionary-native group-by, filter, join
and sort -- a large engine project, not a flag.

### Aggregation-side

| hypothesis | measurement |
|---|---|
| Replace `interleave` in the combine's gather with a flat typed gather (`u32` address planes, direct value copy) -- the trade `ops/repartition.rs` already makes | Fast path confirmed taken (top symbol became `combine::gather_primitive<Int64Type>` at 18.84%, replacing `interleave_primitive` at 22.65%) but end-to-end unchanged (q32 203.7 -> 213.8 ms) and string-key queries **regressed** (q33 0.91x, q16 0.87x, q17 0.87x) because the fallback then builds both index forms. Reverted |
| Size radix partitions to a core's private cache instead of the thread count -- the standard radix-aggregation remedy | **Worse: 955.5 ms -> 1111.4 ms (0.86x)** across ten group-by queries, 8 of 10 regressing (q31 0.74x, q17 0.73x). More partitions buy hash-table locality but cost per-partition setup and a wider output concat, and the second term dominates. Reverted |
| Larger morsels, to cut task dispatch and epoch-GC pressure | Worth 20% on one TPC-H filter and **0.6%** across the ClickBench suite (16,384 rows 2107.7 ms, 262,144 rows 2095.2 ms) -- inside noise. Does not generalize |

### Execution-side

| hypothesis | measurement |
|---|---|
| Multi-key top-N comparator overhead (`sort-limit` shows 3.04x) | Batcher already **wins** every top-N shape in isolation: 1-key int 0.41x, 1-key float 0.92x, 2-key 0.90x, 3-key 0.91x. The 3.04x is against the native store, not an execution defect |
| Scan-level runtime filters (sideways information passing) in the materializing executor | A/B via `BATCHER_RUNTIME_JOIN_FILTER`: q5 584.6 -> 583.2 ms, q9 693 -> 699 ms, q21 943 -> 930 ms. Worth ~0 |
| Streaming vs materializing executor selection | The apparent 2x was an **artifact**: an in-process A/B lets the second arm inherit Batcher's own learned statistics. Measured in isolated processes, streaming wins both q5 (556.5 vs 1011.7 ms) and q18 (447.7 vs 564.4 ms) |
| Date32 SIMD lane width in the Cranelift JIT (the code documents ~10 GB/s against ~40 GB/s for f64) | Equalizing bytes-per-iteration measured flat (27.4 -> 27.2 ms). `perf` shows the predicate runs through Arrow kernels, not the JIT, on this path. Reverted |
| Null-bitmap overhead on non-nullable columns (~20% of a profile in `is_valid`/`set_bits`/`zip`) | The column already carries **no validity buffer** (`null_count = 0`); stripping it changes nothing (72.9 vs 72.8 ms) |
| A hotspot in the heavy ClickBench group-bys | q37's profile is flat: nothing above 5.7%, ~22% accounted across the top eight symbols |

## Worker over-subscription: real, and deliberately not shipped

The one effect that reproduced on **both** benchmarks is that the engine uses every logical
core and loses throughput doing it. ClickBench, full suite, 10M rows:

| workers | total | vs 96 |
|--:|--:|--:|
| 48 | 2150.1 ms | -0.1% |
| 64 | **2018.9 ms** | **+6.0%**, better on 32/43 |
| 80 | 2087.8 ms | +2.8% |
| 96 (default) | 2147.6 ms | -- |

At 64 workers Batcher wins **18/43** against the native store rather than 15/43. TPC-H
agrees (filter-count 25.5 ms at 64 against 27.6 ms at 96).

It is not committed because the optimum is not a topology boundary: this host is 48
physical / 96 logical cores, and **48 performs no better than 96**. A hardcoded 64 would be
tuning to one machine, which `.claude/rules/performance.md` forbids. The right form is an
adaptive worker-width policy that measures -- a Carbonite resource decision, and the
adaptive re-optimization the contract already calls the moat.

## What would actually close the gap

In descending order of evidence:

1. **Adaptive worker width.** Measured 6% on ClickBench and reproduced on TPC-H, with a
   clear owner (Carbonite) and no machine-specific constant if it is measured rather than
   guessed.
2. **Dictionary-native operators.** Not for the byte savings, which are 11%, but because the
   current 5x-14x cliff on dictionary-encoded input is a real defect for anyone reading
   Parquet. Group-by, filter, join and sort would each need an encoded fast path.
3. **A cache-conscious aggregation layout.** The high-cardinality group-by is
   memory-latency-bound and neither a cheaper gather nor more partitions moved it. Closing it
   needs sequential access -- a partitioned layout where probes stream rather than scatter --
   which is a redesign of `assign_groups`, not a parameter.

## A measurement note

Repeated runs of the same ClickBench query varied by up to **14%** on this host (q32
baseline measured 203.7 ms and 232.1 ms in two runs minutes apart). Any conclusion below
about a 5-10% effect needs many more repeats than the best-of-5 used here. The results that
survive that noise are the large ones -- the 5.28x, the 14x, the 0.86x -- and those are what
this document rests on.
