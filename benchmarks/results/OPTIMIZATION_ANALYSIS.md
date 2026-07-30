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

## The ISA floor: measured, and it pays on exactly one kernel family (2026-07-29)

`.cargo/config.toml` pins the ahead-of-time floor at `x86-64-v2` (SSE4.2/POPCNT) so one
shipped `_native.abi3.so` runs on the oldest worker of a mixed cluster. AVX2 is taken at
*runtime* -- but only by the Cranelift JIT. **No AOT kernel in the workspace does runtime
feature dispatch**: `grep target_feature crates/` finds only comments and
`HardwareProfile::detect`. So every Arrow compute kernel, every hash, and every aggregation
loop in the shipped binary is SSE-only. That is a large, previously unmeasured surface.

Measured by building the same commit twice in an isolated worktree -- `x86-64-v2` against
`x86-64-v3` (AVX2/FMA/BMI) -- and running single-threaded (`parallelism=1`) so the number is
the kernel and not the scheduler, minimum of 9, three interleaved rounds:

| case (2M rows) | v2 | v3 | |
|---|--:|--:|--:|
| `filter` on `f64 > lit` | 14.66 / 14.55 / 14.80 ms | 12.49 / 11.83 / 12.00 ms | **0.825x** |
| `filter` on `i64 > lit AND f64 < lit` | 20.04 / 19.39 / 19.14 ms | 16.06 / 15.40 / 15.52 ms | **0.802x** |
| `GROUP BY` int key, `SUM(f64)` | 11.33 / 11.41 / 11.22 ms | 11.51 / 10.20 / 11.66 ms | overlapping -- **no effect** |

The two filter rows do not overlap between arms in any round, and every arm returned
identical results (1,333,333 / 773,200 rows, 7 groups). A wider sweep at 5 repeats found
join (0.96x), distinct (0.95x), `GROUP BY` on a string key (0.99x), sort, top-N and the
string kernels all inside noise.

**So the prize is specific: ~18-20% on comparison/filter kernels, and nothing anywhere
else.** An earlier reading of ~7% on the int group-by did not survive more repeats and is
withdrawn.

Two consequences for whoever picks this up. The gain lives in *arrow-rs's* comparison
kernels being autovectorized wider, not in Batcher code, so capturing it at runtime means
writing multiversioned compare kernels in `bc-expr` (`#[target_feature(enable = "avx2")]`
plus `is_x86_feature_detected!` dispatch, falling back to the Arrow kernel) rather than
changing a build flag -- the shipped floor must stay `x86-64-v2` for the mixed-cluster
guarantee. And the *aggregation* side is not worth multiversioning: it is memory-latency
bound, which is the same conclusion the cheaper-gather and partition-sizing attempts above
reached from the other direction.

### Which half of the filter moves, measured

Splitting the filter into its two halves under both builds (2M rows, one batch, best of 7)
locates the gain precisely and rules out the obvious wrong target:

| component | v2 | v3 | |
|---|--:|--:|---|
| JIT-compiled predicate (`CompiledExpr::eval`) | 7.189 ms | 7.227 ms | flat -- **as designed** |
| Arrow gather (`filter_record_batch`) | 28.509 ms | 29.946 ms | flat |
| Interpreter predicate (Arrow `cmp` kernel) | 3.445 ms | **2.264 ms** | **0.66x** |

The JIT half is flat because Cranelift already selects vector width at *runtime* --
`HardwareProfile::detect` reports `simd_lanes_f64=4, avx2=true, avx512f=true` on this host,
and a coverage probe confirms all three benchmark predicates compile and evaluate. So the
existing runtime-dispatch mechanism works; it simply does not cover the AOT kernels. **The
gather is not the target** -- only the comparison kernel is.

## Rejected: gating the JIT off for boolean predicates

The split above turned up something that looked like a much larger win, and it did not
survive end-to-end measurement. Worth recording in full, because the component number is
real and will tempt the next reader.

At the engine's own morsel granularity (16,384 rows x 128 morsels, compile amortized across
all of them, JIT output asserted equal to the interpreter's on every morsel), **the JIT is
slower than the interpreter on every boolean shape and faster on every arithmetic one**:

| expression | JIT | interpreter | |
|---|--:|--:|--:|
| `f64 > lit` | 7.614 ms | 3.582 ms | 2.13x **slower** |
| `i64 > lit` | 3.042 ms | 1.631 ms | 1.87x **slower** |
| `i64 > lit AND f64 < lit` | 8.498 ms | 6.698 ms | 1.27x **slower** |
| AND-chain, depth 12 | 50.02 ms | 48.57 ms | 1.03x **slower** |
| arith chain, depth 2 | 2.668 ms | 3.805 ms | 0.70x faster |
| arith chain, depth 12 | 14.99 ms | 17.08 ms | 0.88x faster |

The asymmetry is structural. The JIT's advantage is fusion -- no intermediate array per
operation -- which is why the arithmetic chains improve with depth. A comparison has no
intermediate to avoid, and against it the interpreter fields Arrow's tuned kernel writing a
packed bitmap. So a deeper boolean tree gets *closer* without ever arriving.

That reads as a live 2x defect on the commonest operator, since `par.rs` does call
`ops::try_compile` on filter predicates. **It is not.** Gating `try_compile` to decline
boolean-result expressions, built and A/B'd against an ungated build of the same commit,
measured **no end-to-end difference at all** (8M rows, empty-result filter so the gather is
free and the predicate is essentially the whole cost, best of 11, three interleaved rounds):

| case | ungated | gated |
|---|--:|--:|
| `f64 > lit` (parallelism 8) | 7.84 / 8.08 / 8.21 ms | 7.98 / 8.15 / 8.18 ms |
| `f64 > lit AND f64 < lit` | 10.51 / 10.51 / 10.52 ms | 10.53 / 10.61 / 10.80 ms |

Removing a path that were it taken would cost 2x, and observing nothing, means **the engine
is not taking it** on these shapes -- which independently confirms the earlier `perf`
observation that "the predicate runs through Arrow kernels, not the JIT, on this path". The
sequential oracle explains part of it (`filter_batch` passes `&None`, so it never JITs), but
not the `par.rs` path, which does compile. Not shipped: an unproven change is not an
improvement, and the honest state is that the JIT's boolean arm is mostly *dead code on
these shapes* rather than a tax being paid.

Two things follow for anyone picking this up. Any future work that widens JIT coverage for
predicates should first check this measurement, because compiling *more* boolean expressions
would make things worse, not better. And a first A/B of this must not be run at
`parallelism=1` -- that takes the sequential path, where the JIT is never consulted and every
arm measures identically.

## The high-cardinality group-by: the latency is recoverable, and by how much (2026-07-29)

The two aggregation attempts above both failed for the same stated reason -- "they reduce
instructions, and instructions are not the constraint" -- leaving *"a cache-conscious
aggregation layout"* as a recommendation with no number attached. This prices it.

Prefetching is the remedy that matches the diagnosis: it does not remove work, it overlaps
the DRAM misses so several are outstanding at once instead of one dependent load at a time.
Modelled on the q32 shape (10M near-unique keys inserted into a 16.7M-slot open-addressing
table, ~0.6 load factor) as a standalone benchmark that touches no production code -- the
point is to price the redesign before anyone commits to replacing `hashbrown` on the
canonical grouping path:

| prefetch block | naive probe | batched + prefetched | |
|--:|--:|--:|--:|
| 8 | 387.1 ms | 339.5 ms | 0.877x |
| 16 | 385.7 ms | 319.8 ms | 0.829x |
| 32 | 389.7 ms | 317.4 ms | 0.814x |
| 64 | 388.0 ms | **310.8 ms** | **0.801x** |

**~20% off the probe loop**, and the naive baseline moves only ±0.5% across the four runs, so
the effect is not noise. The monotone improvement with block size is itself the confirmation:
more keys hashed before any is probed means more misses in flight, which is what memory-level
parallelism predicts and what a purely instruction-side change cannot produce. That is the
first positive evidence for this line after two rejections.

### …and it does not transfer to `assign_groups`. Built, measured, reverted.

The obvious next step is to spend that 20% by replacing `hashbrown::HashTable` in
`assign_groups_int64_multi` -- it exposes no bucket address, so prefetching requires a
hand-rolled open-addressing table. That was implemented: sized once from `num_rows` at a 0.5
load factor (`alloc_zeroed`, empty sentinel `0`, no rehash), hashing a 64-row block and
prefetching its buckets before probing any of them.

It is **correct** -- `prefetched_matches_the_hashbrown_assignment` compares it against the
shipped `HashTable` loop over 120 shape combinations (1/2/3 key columns x 1 to 100,000
distinct values x 0 to 5,000 rows, including `i64::MIN`/`MAX`) and the group ids match
element for element; the whole 1,401-test Rust workspace stays green. And it is **slower**,
in every round and on every shape (10M rows, best of 5, three interleaved rounds):

| shape | `HashTable` | prefetched | |
|---|--:|--:|--:|
| 2-key near-unique (10M groups) | 92.0 ms | 98.4 ms | **1.07x worse** |
| 1-key, 1M groups | 67.9 ms | 69.5 ms | 1.02x worse |
| 2-key low-cardinality | 93.3 ms | 98.0 ms | 1.05x worse |

The reason is the attribution, and it corrects this document. `assign_groups` is the
**per-morsel** grouping core: `agg_par.rs` records that a 16,384-row morsel's partial holds
~4,096 groups, so its table is tens of KB and sits in L1/L2. There is no DRAM latency there
to recover, and the batching pass plus the loss of `hashbrown`'s SIMD group probe are pure
cost. The ~10M-entry table that misses to DRAM on every probe -- the one the q32 profile
points at -- is only ever assembled in **`combine`**, where the per-morsel partials are merged.

So the prefetch prize above is real but mis-aimed by this file's own earlier wording. The
target is `agg/group/combine.rs`, not `assign_groups`, and any attempt should first confirm
the table it probes is actually large at the point of probing. Reverted, and not shipped.

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

## The control plane, not the kernels: where a small query's time actually goes (2026-07-29)

Everything above measures the Rust data plane. None of it measured the Python conductor, and
that is where the time goes on anything short. Profiled at 3 rows, where the kernel is free:

| stage | us/query | share |
|---|--:|--:|
| `core._execute_in_memory` (of which the **engine itself is 85 us**) | 325 | 23% |
| `write_event_log` | 283 | 20% |
| `carbonite.recommended_config` | 127 | 9% |
| the learning loops (`_close_learning_loops`) | 128 | 9% |
| `kyber._optimize` | 79 | 6% |
| conductor glue (routing, `config_context`, `_admit`, plan build) | ~500 | 35% |
| **total** | **~1,440** | |

**The engine is 6% of a small query.** The other 94% is orchestration, and it is a floor
under *every* query: 1.4 ms whether the input is 1,000 rows or 6,000,000. That is what the
operator-mix table is really showing when `op-global-sum` sits at 2.19x DuckDB -- a 3.1 ms
query with 1.4 ms of fixed cost, against DuckDB's 1.4 ms total. Row-count sweep, same shape:

| rows | 1,000 | 10,000 | 100,000 | 1,000,000 | 6,000,000 |
|---|--:|--:|--:|--:|--:|
| `GROUP BY` + `SUM` | 1.37 ms | 1.43 | 1.58 | 2.81 | 4.34 |

`.claude/rules/performance.md` already names this as a requirement ("Small queries:
sub-second, **low fixed overhead**"). It was never measured against.

### Shipped: the learned-memory refit was O(session history)

`carbonite.memory.learned._fit` re-derived a per-row footprint for **every** row of
`op_stats` on every refit, to absorb the `_REFIT_AFTER = 64` rows that were new. py-spy put
it at ~6% of all samples -- above `json`, above everything but the FFI call itself.

The cost grows with how long the session has been up, and the docstring's claim that the
per-kind cap keeps "a long-lived session's planning cost flat" did not hold:

| queries run | op_stats rows | `_fit` |
|--:|--:|--:|
| 0 | 0 | 0.002 ms |
| 500 | 1,000 | 0.507 ms |
| 5,000 | 10,000 | 6.064 ms |
| 20,000 | 15,418 | 9.646 ms |

It plateaus only because the buckets trim at `PER_KIND_MAX * _TRIM_SLACK` = 8,192 rows, and
the plateau is expensive: a 6-10 ms refit every ~28 queries is **215-300 us on every query**,
in steady state, forever.

The samples are append-only -- a row's derived footprint never changes -- so the derivation
now extends the previous one instead of repeating it, keyed on the bucket's first row
*object* (an `id()` would accept a stale list after a freed row's address was reused). A
bucket the hub trimmed from the front fails that check and re-derives in full.

Refit **4.8 ms -> 0.070 ms (~65x)**, and the fit is bit-identical to a from-scratch pass at
every step through 24 rounds of growth including 3 trims. End to end, **seven interleaved runs**
of 6,000 queries each after a 2,000-query warm-up (both arms in the same process against the
same hub, so host drift hits them equally):

| | mean (ms) | p99 (ms) | max (ms) |
|---|---|---|---|
| from-scratch | 1.760, 1.752, 1.708, 1.835, 1.800, 1.852, 1.657 | 13.93, 13.06, 11.38, 16.33, 15.79, 16.26, 11.48 | 18.2 - 31.3 |
| incremental | **1.433, 1.501, 1.445, 1.446, 1.496, 1.516, 1.417** | **3.354, 3.432, 3.433, 3.427, 3.612, 3.485, 3.472** | **5.0 - 6.7** |

**The two arms do not overlap on either statistic in any round**, which is what makes this
survive a host that showed up to 38 % run-to-run spread on `op-groupby-2key` (wider than the
14 % this document assumed -- see the measurement note). Mean **1.766 -> 1.465 (1.21x)**; p99
**~14.0 -> ~3.46 (4.0x)**. The incremental arm's p99 is also far *tighter* (3.35-3.61 against
11.4-16.3), which is the mechanism showing through: the stall is gone, not merely smaller.

One trap worth recording, because the first version of this change had it: the reused prefix
must be **copied, not appended to**. `execution.max_concurrent_queries` runs several queries
per process, so two threads can refit one hub at once; extending the cached lists in place let
both append the same new rows and double-count them, inflating the learned footprint and every
reservation sized from it. The from-scratch version was safe by construction because it
allocated fresh lists each time. Copying costs a list-of-floats memcpy; the refit still
measures **0.070 ms**.

The **tail is the honest headline** -- the refit was a periodic multi-millisecond stall on a
1.4 ms query. Note that a best-of-N benchmark cannot see this at all: `benchmarks/run.py`
reports the minimum of 5, and the minimum is precisely the statistic that excludes a stall
occurring every ~28 queries. The operator-mix table is unchanged by this fix, and that is
expected rather than a contradiction.

### Also shipped: `_resolve_dir` re-resolved a constant per query

The event log is on by default, so `_resolve_dir` ran on every query, and its comment
("cheap string work") was wrong -- `expanduser` can consult the password database. Measured
31 us/query. Memoized on `(configured, BATCHER_HOME)`, which keeps the property the
per-call resolution existed for.

### Not shipped, and why

- **`event_log = True` by default costs 283 us/query** (22% of the floor), writing a JSON
  document per query plus a directory prune, *and* forcing the metered FFI path. Turning it
  off measures **1.464 -> 1.140 ms**. But `event_log: bool = True` is a deliberate product
  decision ("the Spark event-log analog"), so this is the owner's call, not a tuning. The
  right fix keeps the feature and takes it off the critical path (write asynchronously);
  note the function's own docstring already claims to be off the critical path "with none
  attached (the default)", which the shipped default contradicts.
- **Caching the CPU-topology probes.** `recommend_parallelism` spends 29 us/query re-reading
  `sched_getaffinity` + the CFS quota -- machine constants. A TTL cache is only ~2% and
  breaks `tests/unit/test_hardware.py`, which patches those inputs and expects the next call
  to see them. Not worth it at that price.

## The dictionary cliff is worse than recorded, and half of it is not performance

Re-measured at 6 M rows, 4 distinct values, plain against dictionary-encoded (correctness
compared on every case):

| case | plain | dictionary | |
|---|--:|--:|--:|
| `GROUP BY s`, `SUM(v)` | 6.5 ms | 83.1 ms | **12.8x** |
| `filter s == lit` | 5.4 ms | 80.5 ms | **15.0x** |
| `distinct s` | 7.3 ms | 81.5 ms | **11.2x** |
| `sort s limit 100` | 19.4 ms | 94.2 ms | 4.9x |
| `join on s` | -- | **`PlanError`** | did not run |

The dictionary input is **half the bytes** (72.0 MB against 144.0 MB) and every case lands
at a uniform ~80-94 ms, which is the tell: the cost is one shared decode, not per-operator.
`bc_py::normalize.rs::normalize_to` decodes every `Dictionary` column to its value type at
the boundary so no operator has to special-case the encoding. Preserving it is RFC
`rfc-streaming-executor.md` Proposal 3 (separate the plan's *logical* type from the morsel's
*physical* encoding) and remains an engine project.

**But the join failure was not that project.** `plan.types.lattice.widen` documents itself as
the mirror of `normalize_to` and then omitted its `Dictionary` and `LargeUtf8` arms, so:

- `Dataset.schema` reported `dictionary<values=string, indices=int32>` where `collect()`
  returns plain `string` -- the cheapest of `_schema`'s three arms was the only one that
  lied, since the other two predict normalization or execute for real.
- Joining a dictionary key against a plain-string key was **rejected at build time** with
  "join key type mismatch", for a join the engine runs correctly because both sides reach it
  decoded. Fact table from Parquet, dimension table as plain strings, is the ordinary
  star-schema shape.

Both fixed by completing the mirror (`widen`) and widening the scan arm of `_schema`;
`Dataset.schema` now agrees with `collect()` on every type. This does nothing for the 11-15x,
which is still the decode.

## Scaling with nodes: the guarantee is verified, the measurement is blocked (2026-07-29)

The property that makes scale-out *mean* anything is the mergeable decomposition, and it
holds. `bc-interp`'s `dist` invariant tests pass on the composition the Python orchestrator
actually runs across machines:

- `every_aggregate_survives_the_distributed_pipeline` -- `partial_aggregate` ->
  `partition_batches` -> `combine_finalize` equals single-node for every aggregate
- `partition_batches_matches_materialize_then_partition`
- `spilling_reduce_matches_single_node` -- the out-of-core reducer agrees too
- `edge_float_group_keys_route_identically` -- the `-0.0`/NaN key identity, which is how a
  float group silently splits in two across partitions

Rust lib suites: 518 passed, 0 failed. Above them, `tests/integration/test_distributed.py`
carries **61** single-node-equivalence tests spanning aggregate, distinct, window, sort,
union, inner/left/right/outer/semi/anti join, broadcast join, skew-salted join, as-of join,
runtime-bloom join, `map_batches`, and breaker-free scan fan-out -- on both the Ray-object and
Arrow Flight transports. Every stateful operator has a mergeable form with a test that would
fail if it lost one. That is the guard CLAUDE.md asks for, and it is intact.

Driver-side cost was audited for super-linearity: the aggregate path submits O(M) map tasks
and O(R) reduce tasks, and assembles O(M x R) shuffle-path *strings* on the driver. The
O(M x R) term is hash-shuffle metadata (Spark has the same), not data movement, so it is
inherent rather than a defect -- but it is the term that would bite first at very high
partition counts and is worth remembering.

**What is not here: measured multi-node speedup.** This host's Ray cluster is a single node
whose 96 CPUs are **fully reserved in placement groups** by other sessions -- `ray status`
reported `96.0/96.0 CPU (96.0 used of 96.0 reserved in placement groups)` with 21+ pending
32-CPU tasks -- so a distributed arm never receives a core. A worker-scaling sweep hung on
its first arm at 0% CPU and was killed by its own timeout after 28 minutes. That is an
environment condition, not an engine finding, and no speedup number is recorded rather than
one manufactured from a contended box. Re-run `benchmarks/scenarios/scale_bench.py` on a real
multi-node cluster to fill this in.

## A note on verifying anything in this tree

Two traps cost real time here, both from the tree being shared with many concurrent sessions.

**`.pytest_cache` is shared, so `--lf` lies.** Chasing a single `F` seen in a truncated run,
`--lf` offered
`tests/unit/test_carbonite_wide_row_morsels.py::test_a_measured_width_still_wins_over_the_planned_one`
-- squarely in the carbonite
morsel-sizing area this work touches, so exactly the plausible-looking lead one wants to
believe. **That test does not exist**: it had been renamed to
`test_the_more_binding_of_the_two_widths_wins`. The cache entry was stale from an older run
against an older file. Attribute a failure by re-running the suite, never from `lastfailed`.

**Every distributed test hangs while the cluster is fully reserved, and it reads as a test
failure.** The case was
`tests/differential/test_diff_agg_arg_extreme.py::test_arg_extreme_null_value_single_node_equals_distributed`,
reported FAILED after a 28-minute run. It asserted nothing: the process took SIGTERM from the harness `timeout` while
the test sat on a distributed barrier, and the engine had been logging the cause once a minute
throughout:
`distributed barrier has waited 1680s with 0/3 tasks finished cluster CPU 96/96 in use`.
The file was clean at HEAD, the two non-distributed tests in it pass, and `ray status`
confirmed `96.0/96.0 CPU (96.0 used of 96.0 reserved in placement groups)` with 48+ pending
tasks. Deselect with `-k "not distributed"` to get a meaningful differential number on a
reserved box -- **but that filter is not sufficient**: it matches test *names*, and
`tests/differential/test_dist_hunt2_matrix.py` names none of its cases "distributed" while
every one of them calls `ds.collect(distributed=True, num_workers=...)`. Those four cases sit
at the end of an alphabetical selection, so they present as a single mystery `F` at ~95 %
after twenty minutes of green. Add `--ignore=tests/differential/test_dist_hunt2_matrix.py`
(plus `test_diff_distributed_join.py` and `test_diff_distributed_operator_matrix.py`).

Run in isolation on the reserved box, that file fails **22 of 22** -- which is itself the
diagnosis, since a code defect fails a subset. Two independent confirmations that it is
scheduling and not correctness: the run took 1,933 s for 22 tests against a `--timeout=90`,
so every test consumed essentially its whole timeout (an assertion fails instantly), and a
single test run under `--timeout=60` failed at exactly 61.00 s with

```text
ray/_private/worker.py:3196: in wait
    ready_ids, remaining_ids = worker.core_worker.wait(
E   SystemError: <cyfunction CoreWorker.wait ...> returned a result with an exception set
```

-- blocked inside `ray.wait()`, with no assertion anywhere in the traceback. The test never
reached a result comparison.

And the deselection has to be by *content*, not by name. A distributed case hides in files
whose names suggest nothing:
`test_diff_agg_expressions.py::test_aggregate_expression_is_partition_independent` calls
`collect(distributed=True, num_workers=4)` and dies with the same `CoreWorker.wait` error at
exactly its timeout, and it is the 70th test of the suite -- so it presents as an early `F`
in a run that looks otherwise clean. The reliable filter is the call itself:

```bash
grep -rln "distributed=True\|num_workers=" tests/differential/   # 13 of 367 files
```

Excluding those 13 leaves 354 files that can actually run on a reserved box, and read that diagnostic before attributing such a failure to a code change --
the engine is telling you exactly what happened.

**Pipe a long run to `tail` and you throw away the summary.** `pytest ... | tail -25` keeps
the last 25 *lines*, which on a `-q` run are all progress dots -- and the shell reports
`tail`'s exit code, not pytest's, so a killed or failing run reads as success. Use
`-rf --tb=short` and read the file the run wrote.

Also worth knowing before trusting a small effect measured here: `op-groupby-sum` was
measured at 3.9, 5.2, 3.9 and 5.0 ms across four runs of identical code (**33 %** spread), and
`op-groupby-2key` at 5.8-8.0 ms (**38 %**) -- wider than the 14 % this document assumed above,
and on a box whose load average was under 3. A best-of-5 benchmark on this host cannot resolve
anything smaller than about 40 %.

## Per-operation evaluation (2026-07-29)

Where each operator actually stands, measured on this host. `b/duckdb` below 1.0 means Batcher
wins. Operator-mix suite, TPC-H sf1 `lineitem`/`orders`, best-of-5, `duckdb_arrow` (identical
Arrow input, so this isolates execution from storage format):

| operation | batcher | duckdb | b/duckdb | read |
|---|--:|--:|--:|---|
| `filter` + `count` | 0.4 ms | 1.4 | **0.30x** | wins outright |
| `window` running-sum | 38.6 | 227.6 | **0.17x** | wins by 6x |
| `window` rank | 46.1 | 119.3 | **0.39x** | wins |
| `window` lag | 62.3 | 156.4 | **0.40x** | wins |
| `window` sum-partition | 31.7 | 44.5 | **0.71x** | wins |
| `join` + aggregate | 39.4 | 41.3 | **0.95x** | parity |
| `filter` + project | 9.1 | 8.0 | 1.14x | ~parity; ~1.4 ms of it is fixed cost |
| `group_by` 2-key | 6.1 | 5.1 | 1.20x | fixed cost is ~23 % of the query |
| `group_by` 1-key | 3.9 | 3.1 | 1.28x | fixed cost is ~36 % of the query |
| `sort` + limit | 8.2 | 6.3 | 1.30x | engine ~6.8 ms against 6.3 |
| global `sum` | 3.1 | 1.4 | 2.19x | **fixed cost is ~45 % of the query** |

The pattern is not per-operator quality. Every case Batcher loses is a *short* case, and the
gap tracks the share of the query that is fixed control-plane cost rather than anything the
operator does -- global `sum` is the worst ratio in the suite and also the smallest query in
it. The window family, where the engine does real work for tens of milliseconds, wins by up to
6x. This is the same conclusion the section above reaches from the profiler, arrived at from
the benchmark side.

Two caveats on reading this table. The ratios move 33-38 % run to run on this host (see the
measurement note), so nothing here below about 1.4x is resolvable -- the *direction* of the
short-query cases is trustworthy because it agrees with the profile, not because 1.28x is
precise. And these are `duckdb_arrow` numbers; against DuckDB's native compressed store the
storage-format gap documented at the top of this file applies on top.

Dictionary-encoded input is measured separately above and is a different story: 11-15x on
`group_by`/`filter`/`distinct`, which *is* a per-operator defect (the boundary decode), not
fixed cost.

## What the gate actually said (2026-07-29)

The verification for the three changes above, run on a shared tree with ~40 concurrent
sessions and a fully-reserved Ray cluster.

| suite | result |
|---|--:|
| `tests/unit` | **9,961 passed, 3 skipped, 0 failed** |
| `tests/differential`, the 354 files with no `distributed=True` call | **6,755 passed, 1 xfailed, 4 failed** |
| `cargo test -p bc-interp -p bc-runtime --release --lib` | **518 passed, 0 failed** |
| `bc-interp::dist` mergeability invariants | 7 passed |
| ruff / lint-tests / lint-layers / lint-structure / lint-guardrails | clean |
| operator-mix benchmark, 4 independent runs | correctness passed 4/4 |

`lint-docstrings` reports one violation, `io/source/inmemory.py::column_cheap_stat` -- a public
method another session added in its working copy. It does not exist at HEAD, so it is theirs.

The **4 differential failures are all
`test_diff_decimal_literals.py::test_every_comparison_against_a_decimal_matches_duckdb`** at
the `999999999.99` boundary, and none is attributable to the changes here:

- the file is **untracked** (`git status` reports `??`), so it does not exist at HEAD and
  cannot be a regression;
- it uses no dictionary or `LargeUtf8` column, and `widen(decimal128(18, 2))` is the identity,
  so the `widen` change is a provable no-op for it;
- 38 of the file's 42 cases pass, and only the boundary value fails -- the signature of a
  float-precision limit, not of a type-inference change;
- its own docstring states the cause: "The IR has no decimal literal, so the value is
  converted to the float that represents it", which stops being exact past ~1e9.

It is another session's in-flight work on a real decimal-literal gap, left alone deliberately.

## The profile after the refit fix, and where the next work is

Re-sampled at steady state (3,000 warm-up queries so the `op_stats` buckets are full and the
model is warm), 3,633 samples over 28 s, py-spy at 300 Hz:

| self % | symbol |
|--:|---|
| 24.30 | `execute_local_metered` -- the FFI call itself |
| 3.63 | `json.encoder.iterencode` |
| 2.01 | `open_private` (the event-log write) |
| 1.93 | `read_cgroup_bytes` |
| 1.76 | `available_cpu_count` + `cpu_contention` |
| 1.37 | `projected_input_bytes` |
| 0.74 | `planned_row_cap` |
| 0.74 | `write_event_log` |
| 0.63 | `unlink` (the event-log prune) |

**`_fit` does not appear at all**, against ~6 % before -- the independent confirmation that
the refit change landed, from the sampler rather than from the A/B.

The shape is now the healthy one: the engine is the single largest term, and no Python symbol
clears 4 %. What remains is diffuse, and the largest *cluster* is the event log --
`iterencode` + `open_private` + `write_event_log` + `unlink` is about **7 %** -- which is the
same conclusion the stage decomposition reached and the reason the async-write item is the
next one worth taking. Everything after that is a 1-2 % item, so the remaining fixed cost will
not come off in one change; it needs the small-query fast path, which is a design decision
rather than a tuning.

Note the ~5.7 % across `_compile_bytecode` / `_create_fn` / `_call_with_frames_removed` /
`_path_stat` is interpreter startup and module import, not per-query cost -- do not chase it.

## Intra-node scaling, measured (2026-07-29)

Multi-node speedup could not be measured (the cluster is reserved -- see above), but scaling
with *cores* can be, and it is the same property node-scaling needs: the parallel part has to
stay dominant. 40 M rows, `execution.parallelism` swept 1 to 96, best of 3, every arm's result
compared against the 1-core arm so a fast wrong answer cannot read as a win (all `True`).

| cores | groupby (100 k groups) | eff. | filter+count | eff. | sort+limit | eff. |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 31,793 ms | 1.00 | 2,444 ms | 1.00 | 6,985 ms | 1.00 |
| 2 | 15,996 (1.99x) | 0.99 | 1,258 (1.94x) | 0.97 | 3,572 (1.96x) | 0.98 |
| 4 | 8,137 (3.91x) | 0.98 | 639 (3.83x) | 0.96 | 1,836 (3.80x) | 0.95 |
| 8 | 4,175 (7.61x) | 0.95 | 318 (7.68x) | 0.96 | 949 (7.36x) | 0.92 |
| 16 | 2,190 (14.52x) | **0.91** | 162 (15.07x) | **0.94** | 506 (13.81x) | **0.86** |
| 24 | 1,868 (17.02x) | 0.71 | 113 (21.64x) | 0.90 | 357 (19.60x) | 0.82 |
| 32 | 1,457 (21.82x) | 0.68 | 86 (28.49x) | 0.89 | 286 (24.45x) | 0.76 |
| 48 | 1,160 (27.40x) | 0.57 | 78 (31.20x) | 0.65 | 239 (29.24x) | 0.61 |
| 64 | 1,084 (29.33x) | 0.46 | 72 (33.91x) | 0.53 | 228 (30.71x) | 0.48 |
| 96 | 1,091 (29.14x) | 0.30 | 67 (36.47x) | 0.38 | 218 (32.04x) | 0.33 |

**Linear to 16 cores and clearly sub-linear past 32.** Three things this pins down:

1. **It is not the fixed overhead.** The ~1.44 ms control-plane cost is serial, so Amdahl
   would bound the speedup -- but against a 1,091 ms query it is 0.1 % and cannot explain an
   efficiency of 0.30. The loss is in the parallel section.
2. **Group-by *regresses* past 64 cores** (1,083.9 ms at 64, 1,090.9 ms at 96) while filter and
   sort still creep up. This is the same effect the worker-over-subscription section above
   found from the other direction (64 workers beating 96 on ClickBench), now with the whole
   curve instead of four points, and it strengthens the case for the adaptive worker-width
   policy: the right width is workload-dependent, so it has to be measured, not configured.
3. **This host is 48 physical / 96 logical**, so the 64 and 96 rows are hyperthreads and a
   sub-linear reading there is expected. The honest number for real cores is **48: 0.57-0.65
   efficiency**.

A caveat on the high-core rows: the box carried a load average of ~3.8 from other sessions,
which penalises the wide arms disproportionately. Read the 1-16 range as solid and the 48-96
range as indicative.

**What this does and does not say about nodes.** It does not measure node-scaling, and it is
not a proxy for it in the pessimistic direction: the dominant suspect for the decay above 32
cores is contention for one machine's shared memory bandwidth on a group-by over 40 M rows,
and *that term does not exist across nodes* -- each node brings its own bandwidth. So the
mergeable algebra plus per-node bandwidth is an argument for node-scaling being **better** than
this curve, not worse, with the network shuffle as the new cost to pay. That remains an
argument, not a measurement, until it is run on a real cluster.
