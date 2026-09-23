# Join algorithms

A join in Batcher is one primitive: a builder that produces a pair of row-index vectors
describing the output. Every join type, every strategy, and both the parallel and distributed
paths are built on that one primitive. This page describes it, the algorithms layered over it,
and where its remaining headroom is.

On TPC-H at scale factor 1, the four-engine board of 2026-09-13 in
[`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md) puts Batcher at a 0.72 geomean ratio against DuckDB on its own
native storage, a win overall, with 6 of the 22 cases recorded as losing on that board. Against
DuckDB reading the same Arrow input Batcher reads, the ratio is 0.25
({doc}`TPC-H benchmarks </benchmarks/results/tpch>`). Distributed, Batcher's join beats
Daft's Ray runner by 1.7x to 2.2x at every scale measured. The headroom that remains on a single
node is in the serial work around the parallel per-bucket join rather than in the join kernel,
which is what the later sections of this page follow.

## One primitive: index pairs

[`crates/bc-runtime/src/join/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/mod.rs) computes two row-index vectors, `(left, right)`, that
describe the output. Output column `c` is `take(side_of_c, indices_of_that_side)`.

That is the whole design. An unmatched row on the null-supplying side gets a **null index**,
and Arrow's `take` yields null for it, which is exactly what an outer join means. So
inner, left, right, full, semi, and anti all fall out of one index-pair builder rather than
six kernels:

```rust
// crates/bc-runtime/src/join/mod.rs
pub fn hash_join_indices(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
) -> Result<JoinIndices, RuntimeError>
```

Keys are encoded with Arrow's row format, so multi-column keys of any type work through one
code path. SQL null semantics hold: a row with any null key never matches anything, because
`NULL != NULL`.

:::{important}
Matching is purely by key equality, so a global join equals the union of per-partition joins
whenever both sides are hash-partitioned by the join key. This module is the partition-local
primitive and carries no single-node assumption. That one property is what the parallel
executor and the distributed executor both stand on.
:::

## Hash join, and the bloom in front of it

This is the default. Batcher builds a chained hash table over the right side, the build side,
and probes it with the left. The table is a `hashbrown::HashTable` storing row ids, looked up
by the hash of the encoded key and confirmed by an equality re-check.

```text
      PROBE side (left)                       BUILD side (right)
      ─────────────────                       ──────────────────
        encode key                              encode key
             │                                       │
             │                                       ▼
             │                            hashbrown::HashTable
             │                             row ids, chained
             ▼                                       │
      ┌──────────────┐   engaged only when the       │
      │    bloom     │   build side is ≥ 2^16 rows   │
      │              │   AND the probe is ≥ build    │
      └──┬────────┬──┘                               │
    miss │        │ hit                              │
         │        └────────────► probe ◄─────────────┘
    skip │                         │
   (a provably                     ▼
    empty chain)      JoinIndices { left: [...], right: [...] }
         │                         │
         └─────────────────────────┤
                                   ▼
            output column c = take(side_of_c, indices_of_that_side)
            a NULL index on the null-supplying side makes take() yield null,
            which is exactly what an outer join means
```

The bloom is engaged when, and only when, it pays. Below ~64K build rows
(`bloom_min_build_rows`, 2^16) the hash table is cache-resident and a probe lookup is already
cheap, so a bloom is pure overhead. Above it the table spills L2/L3 and each probe becomes a
random cache miss that a compact bloom can skip for a non-matching key. At a 1% false positive
rate a bloom is a small fraction of the chained hash table's per-entry cost, so it stays
cache-resident after the table doesn't. A bloom has no false negatives, so it can only ever
skip a provably empty chain, which means it can never change a result. The gate also requires
the probe side to be at least as large as the build side, so the one-pass build cost
amortizes.

## Three strategies, one relation

`RelOp::HashJoin` carries a `strategy` the planner (Kyber) chooses.

| Strategy | Chosen when | Data movement |
|---|---|---|
| `hash` | the default: neither side is broadcastable | both sides hash-partitioned by key into one bucket per worker |
| `broadcast` | the build side fits `optimizer.broadcast_max_bytes` | the build side is replicated; the probe side never moves |
| `sort_merge` | the build side is too big to hash: at least 50 M rows per worker | no hash table; sort (or skip the sort) and merge |

:::{note}
All three produce the same relation. Only the data movement differs, so a wrong pick is slow
rather than wrong. That is what makes the strategy safe for Kyber to learn (see
{doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`) rather than something a user must get right.
:::

The decision that picks among the three, and the two later points at which it is made again on
better information, are below.

![How a join strategy and a build side are picked. The decision is made from each side's estimated rows times row width, the join type, since only an inner join may swap sides, and the broadcast ceiling, which is a quarter of L3 and on a cluster that times 16 with a 64 MiB floor. If the smaller side is under the ceiling the join broadcasts and the probe side never moves; otherwise a build side above the row floor of 50 million rows per worker takes sort_merge, which builds no hash table, and everything else takes hash, the shuffle join that partitions both sides by key. The runtime always builds on the right, so Kyber swaps the inputs when the smaller side is the left, and only an inner join may be swapped. The same decision is then made twice more: the bandit substitutes a learned arm before the run, and at run time the driver shuffles a broadcast that measured bytes show no longer fits. All three arms produce the same relation, so a wrong pick is slow rather than wrong, which is what makes the choice safe to learn across runs.](/_static/diagrams/join_strategy_choice.svg)

::::{tab-set}
:::{tab-item} hash (shuffle)
Hash-partition both sides by the join key into one bucket per worker and join the buckets in
parallel. Equal keys land in the same bucket, so the per-bucket joins are independent and their
union is the full join. This is the same strategy the distributed layer runs across actors.

The partitioning doesn't concatenate first. `ops/repartition.rs` builds each bucket directly
from the morsels with Arrow's `interleave`, so each row is gathered **once** instead of being
copied into one giant batch and then gathered again: two full copies of the query's largest
relation, back to back, is what the old path cost. The buckets stay contiguous, one
`RecordBatch` each. Partitioning each morsel independently was tried and reverted, because it
leaves each bucket holding one small piece per morsel. Partitioning 366 morsels into 96 buckets
gives each bucket 366 pieces of ~170 rows, and the per-piece overhead of the downstream join
swamps the copy it saved.
:::

:::{tab-item} broadcast
When the build side is small enough to replicate, the large probe side joins with no key
shuffle at all, parallelized over row ranges. `join/stream.rs` builds the hash table once and
probes **one morsel at a time**, so the probe relation is never concatenated. On a 60M-row
`lineitem` that copy was gigabytes of pure overhead and the single largest allocation in the
query, and it put every `Utf8` column at risk of Arrow's 2 GiB 32-bit offset ceiling.

The streaming probe is restricted to what is provably safe per morsel. It requires all three of
the following:

1. A left-driven join type: `Inner`, `Left`, `Semi`, or `Anti`. `Right` and `Full` must
   reconcile unmatched build rows across every morsel, so they can't be decided one morsel at a
   time.
1. Integer keys, any number of `Int64` columns, or a single `Utf8`, `LargeUtf8`, `Binary` or
   `LargeBinary` column. A row-encoded key would need its `RowConverter` shared across morsels,
   while a single byte column is hashed and compared on its raw bytes and carries no state.
1. A build side under a row ceiling (`RADIX_MIN_BUILD_ROWS_BROADCAST`), which is a cost
   comparison against the partitioned radix join rather than a correctness limit.

`BroadcastProbe::new` returns `None` for anything else and the caller keeps the materialized
path. Nothing silently changes shape.
:::

:::{tab-item} sort_merge
There is no hash table. Batcher sorts both sides by key and merges them. `join/sort_merge.rs`
skips the sort when the indices already arrive in ascending key order, which it establishes in
one linear pass. That saving is real, but it is not what selects this strategy: `SORT_MERGE_MIN_ROWS`
(50 M rows per worker) is. A note in `kyber/rules/selection.py` records that preferring sort-merge
for already-ordered inputs "was tried and reverted", because its encoding overhead loses to hash
even when the sort is skipped. Only a build side genuinely too big to hash keeps it. Output order differs from the hash
join, because these are unordered relations.
:::
::::

## Making the build side cheap

The build is the join's sequential prefix, and three pieces of `bc-runtime/src/join/` exist to
shrink it.

A surrogate key such as `o_orderkey`, `p_partkey` or `c_custkey` is a near-contiguous run of
integers, and for those `dense.rs` replaces the hash table with a direct map: `map[key - lo]`
holds the chain head, so neither side hashes and no collision chain is walked. It is gated on
the key's observed value range, the same way the aggregate's dense group map is. On
`lineitem` joined to `orders` at 6M to 1.5M rows the shared build took about 29.5 ms against 6.7 ms for the whole
probe, which is the cost this removes.

When the table must be hashed, `build.rs` shards it by a slice of the hash so every core builds
at once. A key's shard is a pure function of its hash, so build and probe agree without
communicating, and because each shard receives its rows in ascending order and chains prepend,
every chain comes out in the order the serial loop produced.

Once the build side exists, its key set is a superset filter on the probe side: a probe row
whose key is absent can produce nothing. `key_filter.rs` digests that set into a `KeyFilter`
the streaming executor sinks down the probe pipeline to the scan, so a dropped row also skips
every predicate and projection above it. On TPC-H q21, 411 of 10,000 suppliers survive
`n_name = 'SAUDI ARABIA'`, and the 6M-row `lineitem` probe is cut about 24x before its date
predicate runs. The filter is the literal key set bounded by the build side's `[lo, hi]`, never
a sketch, so it has no false negatives.

## Radix partitioning

Both radix join paths begin by scattering each side's non-null rows into cache-sized partitions
carrying the key inline as `(key, abs_row)`. That scatter used to be one serial loop over every
build *and* probe row, and on a 60M-row probe it was the join's whole Amdahl bottleneck. The
nominally parallel radix join scaled only 12.4x across 48 workers, against 19.8x for the
group-by aggregate, which has no such pass.

`join/radix.rs` runs the textbook three-phase parallel partition:

```text
   phase 1: histogram          phase 2: prefix sum        phase 3: scatter
   ──────────────────          ──────────────────         ────────────────
   chunk 0 ─► counts ─┐
   chunk 1 ─► counts ─┤   exclusive prefix sum over       each chunk writes into
   chunk 2 ─► counts ─┼─► (chunk, partition) reserves ──► its own reserved slices,
   chunk 3 ─► counts ─┘   every chunk a disjoint slice    in increasing row order,
                          of every partition's output     in parallel

                ┌──────────────┬──────────────┬──────────────┐
   result:      │ partition 0  │ partition 1  │ partition 2  │  ...
                │ c0 c1 c2 c3  │ c0 c1 c2 c3  │ c0 c1 c2 c3  │
                └──────────────┴──────────────┴──────────────┘
                  each partition holds its rows in ascending abs_row order
```

:::{warning}
The result is **bit-identical to the serial scatter**, and that is deliberate rather than
incidental. A chunk's slice is offset by the counts of all earlier chunks, and each chunk walks
its rows in increasing index order, so every partition ends up holding its rows in ascending
`abs_row` order: exactly what the old serial `push` loop produced. The join's output row order,
and with it the `seq == par` oracle, depends on that.
:::

### Which key types reach it

The radix join's key has to be a single `Copy` value, because the partition pass carries the key
*inline* next to the row index. Integer keys are such a value; a string is not, so for a long
time a string-keyed join took the flat build-and-probe path instead. That path is serial, and on
a 96-core machine the difference was not subtle: a 4-million-row join measured **2.10x**
parallelism on a string key against **25.1x** on an integer key over the identical shape, which
is 2,235 ms against 39 ms.

The fix is not a second join algorithm. A byte key of fifteen bytes or fewer packs losslessly
into one `u128`, the length in the top byte and the value bytes below it. That *is* a `Copy`
value, so it reaches the same radix join the integer paths use. The packing is **injective**, so
"packs equal" and "bytes equal" are the same predicate, and the partitions, chains and matches
are the ones the byte comparison would have produced. Longer keys keep the flat path.

Fifteen bytes is chosen to cover what join keys are in practice: ids, codes, SKUs, ISO dates,
categoricals. Measured on equal-sized string-keyed joins, against DuckDB:

| rows per side | before | after | DuckDB |
|---|---|---|---|
| 250,000 | 117.3 ms | **22.3 ms** | 23.4 ms |
| 1,000,000 | 633.1 ms | **46.1 ms** | 53.9 ms |
| 4,000,000 | 2,617.3 ms | **124.2 ms** | 59.7 ms |
| 10,000,000 | 6,835.6 ms | **304.8 ms** | 129.7 ms |

Up to **22.4x**, and the two smaller scales move from a loss to a win. The tell that it is the
parallelism rather than the packing is the shape of the cost: before, the join spent a flat
~650 ns per row at every scale, which is what a serial loop does; after, it falls from 89 to
30 ns per row as there is more work to spread.

:::{note}
The build side still has to clear `RADIX_MIN_BUILD_ROWS` (65,536) for any radix arm to fire, so
a string join against a small dimension table keeps the flat path. That is correct: the whole
build table fits in cache, and partitioning it would be overhead.
:::

## Skew and spill

**Skew.** A hash-partitioned bucket that is far hotter than the average would leave one worker
grinding while the rest idle. `par.rs` compares each bucket against the average on both rows and
bytes, using the `skew_bucket_factor` threshold with absolute row and byte floors so a small
bucket is never called skewed. `join_par/mod.rs` then spreads the hot bucket across workers by
broadcasting its build side and chunking its probe side. A `Full` join is ineligible, because it
must reconcile unmatched rows on both sides.

**Spill.** When the build side exceeds the memory envelope, the grace hash join partitions both
sides by key to disk and joins one bucket at a time, so only one build table is ever resident.
The streaming variant does this **one input batch at a time**, so a build side far larger than
memory spills instead of OOMing at the materialize step. Bucket count is sized from the build
batches' total bytes without materializing them, and the fixed-seed partitioner co-locates equal
keys, so the union of per-bucket joins is the full join for every join type.

The admission test, the fan-out, and what a single bucket pair costs are below.

![The grace hash join, from admission to one bucket pair. admit sizes the build side as its Arrow bytes plus 12 bytes per build row; if it fits, one hash table is built on the right and probed by the left. If it does not, both sides are partitioned by the same hash of the join key, one batch at a time so neither side is ever fully materialized, and written to disk as join-left/part-i.arrow and join-right/part-i.arrow, with the bucket count sized from the larger side divided by the budget, from 2 to 256. Each bucket pair is then joined on its own, and only the build bucket is resident: the probe bucket streams past it in chunks, so the cost is one bucket rather than two. A build bucket still over budget is re-partitioned with a fresh salt, at most three deep, because a re-split is a re-hash and so cannot separate rows that share a key, which leaves one hot key in one bucket at every level.](/_static/diagrams/hash_join_spill.svg)

## Range joins

An inequality, interval-containment or band join is `RelOp::RangeJoin`, and
[`crates/bc-runtime/src/join/range/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-runtime/src/join/range) answers it without materializing the cartesian product. It
emits the same `JoinIndices` the hash join does, so every join type and the caller's gather are
unchanged. The algorithm follows the shape of the condition:

| Condition | Algorithm |
|---|---|
| one inequality | sort the right side once; each left row's matches are a contiguous suffix found by binary search |
| a band, two inequalities bounding one right key (`L.a <= R.y AND R.y <= L.b`) | the matches are a slice of one sorted array whose bounds move monotonically with the left key (`band.rs`) |
| two general inequalities | IEJoin, the algorithm behind DuckDB's `PhysicalIEJoin`: sort both axes, sweep one, and read matches off a mark array |
| a right side of at most 32 rows | no sort at all: `|R|` vectorized comparisons over the left key column (`small.rs`) |

The sorts and the sweep fan out across cores, and the slices fold back in order, so the parallel
output is identical to the sequential sweep rather than merely equivalent. The competitive
scorecard records the result: a win against DuckDB below about 1M rows, parity at 1M, a loss
above it, and a win at any left size against a right side under 32 rows.

Distributed, a range join broadcasts its right side and range-joins each left partition against
it, which is exact for the left-driven join types. An inequality has no key to co-partition on,
so there is no shuffle fallback: a right side over `optimizer.broadcast_max_bytes` raises a
`PlanError` naming the fixes rather than running the whole join on one node.

## ASOF

`join/asof.rs` matches each left row to the right row whose `on` key is nearest in a direction
within the same `by` group: the time-series join. Every left row is emitted (left-style);
unmatched rows get a null right index, exactly as in a left outer join. Keys are row-encoded, so
`on` (order-preserving) and `by` (equality) work for any type, and rows with a null `on` never
match.

Like the equi-join, it carries no single-node assumption: partitioning both sides by `by` makes
a global ASOF equal the union of per-partition ASOFs.

### Distributing an ASOF with no `by` keys

A `by`-keyed ASOF co-partitions by hash, because a match only ever pairs rows inside one `by`
group. A **keyless** ASOF has no group to hash. Any left row may match any right row, and which
one it matches is decided by a global order on `on`, so hashing is not merely unbalanced, it
sends a row and its match to different workers.

Range partitioning is the shape that works, and it is the one the distributed sort already
uses. Batcher samples the left key's distribution, cuts it into ordered intervals, and sends
both sides through the *same* boundary list. Bucket `r` then holds every left row and every
right row whose key falls in interval `r`, so a match inside the interval is already local.

What remains is the match that is not. A left row in bucket `r` can match a right row in an
earlier bucket when the direction is `backward`, or a later one when it is `forward`, and the
gap between them is unbounded, so no fixed overlap covers it. Exactly one row per direction
does. The intervals are ordered, so among every right row below the bucket the only one that
can ever win a backward match is the largest, and among every row above it the only forward
candidate is the smallest. Batcher lends each bucket those rows before the reducer runs. The
carry costs one row per bucket per direction rather than a share of the data, and it is
measured inside the range task that already holds the bucket, so it adds no pass over the
input.

Two details decide whether the carry is exactly right rather than approximately right. It is
the *boundary member* of a tie group, not an arbitrary one: when several right rows share the
extreme key, a backward match takes the last of them and a forward match the first, so keeping
the wrong member returns the right key with a neighbouring row's payload. And a `tolerance`
does not shrink the carry, because whether the carried row is near enough is the engine's
decision, made after it arrives.

The buckets are concatenated in key order, which is a permutation of the single-node result
rather than a match for it. A single-node ASOF emits rows in left-input order. That is already
true of the `by`-keyed path's hash buckets, and of every distributed join.

## Using it

```python
import batcher as bt

left = bt.from_pydict({"k": [1, 2, 3], "v": ["a", "b", "c"]})
right = bt.from_pydict({"k": [2, 3, 4], "w": [20, 30, 40]})

inner = left.join(right, on="k", how="inner").sort("k")
outer = left.join(right, on="k", how="left").sort("k")
print(inner.to_pydict())
print(outer.to_pydict())
print(inner.explain())
```

```text
{'k': [2, 3], 'v': ['b', 'c'], 'w': [20, 30]}
{'k': [1, 2, 3], 'v': ['a', 'b', 'c'], 'w': [None, 20, 30]}
```

:::{dropdown} The `explain()` output, and the strategy it chose
```text
query plan (planned)                                            6 operators
───────────────────────────────────────────────────────────────────────────
OPERATOR                         ESTIMATE  NOTES
sort  [k]                           est≈1  (default)
└─ hash_join  [inner on k]          est≈1  (default)
   ├─ filter  [k ≥ 1 AND k ≤ 3]     est≈2  (default)
   │  └─ scan  [source 1]           est≈3  (exact)  pushed[k ≥ 1 AND k ≤ 3]
   └─ filter  [k ≥ 2 AND k ≤ 4]     est≈2  (default)
      └─ scan  [source 0]           est≈3  (exact)  pushed[k ≥ 2 AND k ≤ 4]

decisions:
  - [kyber/selection] join build side: left≈3 right≈3 [exact] → swap build→left + broadcast
```

Both sides are three rows, so it broadcasts, and it swaps which side builds. The strategy is
named in the decisions block, not in the tree. The two `filter` nodes are not in the query
either: Kyber derived each side's key range from the other and pushed it into the scan, which
is why the join's estimate falls to one row.
:::

The `None` in the left join is the null index in the index-pair builder, made visible.

## Parallelism

Join throughput is set by how much of the operator runs in parallel, and the profile says
exactly where that is decided: the serial prefixes around the parallel per-bucket join. The
radix scatter and the hash build are parallel, and the probe side is gathered once instead of
concatenated and re-gathered. The measurements are in [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md), and
{doc}`vs Daft </benchmarks/comparisons/vs-daft>` carries the distributed join against Daft.

## Code map

- [`crates/bc-runtime/src/join/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/mod.rs): `hash_join_indices`, the bloom gate, `JoinIndices`
- [`crates/bc-runtime/src/join/dense.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/dense.rs), `build.rs`: the direct-map build and the sharded parallel build
- [`crates/bc-runtime/src/join/key_filter.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/key_filter.rs): the build-side key set pushed to the probe scan
- [`crates/bc-runtime/src/join/range/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-runtime/src/join/range): range, band and IEJoin inequality joins
- [`crates/bc-runtime/src/join/radix.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/radix.rs): the parallel three-phase partition
- [`crates/bc-runtime/src/join/stream.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/stream.rs): `BroadcastProbe`, the streaming probe
- [`crates/bc-runtime/src/join/sort_merge.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/sort_merge.rs), `asof.rs`: the other two algorithms
- [`crates/bc-interp/src/join_par/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-interp/src/join_par): grace join, broadcast join, skew detection
- [`crates/bc-interp/src/ops/repartition.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/repartition.rs): gather-once bucket construction

## See also

- {doc}`Architecture </architecture/index>`: why there is one join and not a distributed second one.
- {doc}`Execution engine </architecture/internals/execution>`: the operator around these primitives.
- {doc}`Kyber </architecture/internals/kyber>`: the pass that picks the strategy and the build side.
- {doc}`Joins </user-guide/analyze/joins>`: the API, and how to help the planner.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: the decisions block above, explained.
- {doc}`vs DuckDB </benchmarks/comparisons/vs-duckdb>`: the join-heavy queries against DuckDB's native store.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: q5, q7, q8, q17 in context.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: the shuffle-into-buckets schedule.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why per-partition joins union to the whole join.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: the grace hash join, when the build side doesn't fit.
