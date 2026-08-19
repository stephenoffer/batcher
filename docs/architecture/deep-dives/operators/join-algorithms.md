# Join algorithms

A join in Batcher is one primitive: a builder that produces a pair of row-index vectors
describing the output. Every join type, every strategy, and both the parallel and distributed
paths are built on that one primitive. This page describes it, the algorithms layered over it,
and where its remaining headroom is.

The join is where single-node scaling has the most left to give. On TPC-H at scale factor 1 on
16 cores, Batcher matches DuckDB's result on all 22 queries and, against DuckDB reading the same
Arrow input, wins all 22. Against DuckDB's own native store the join- and subquery-heavy shapes
are where it trails: q17 at 7.9x, q20 at 2.8x, q3 at 2.6x, q21 at 2.4x, and the operator
microbenchmark `join → aggregate` at 98.3 ms against 85.6 ms. The cause isn't the join kernel.
Single-node parallelism on these shapes reaches only about 1.7x to 3.8x on 16 cores, because
serial prefixes such as materialize, gather, and shuffle run before the parallel per-bucket
join. Distributed, where those prefixes are spread across workers, the same operator leads.

## One primitive: index pairs

`crates/bc-runtime/src/join/mod.rs` computes two row-index vectors, `(left, right)`, that
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
| `sort_merge` | the inputs already arrive in key order | no hash table; sort (or skip the sort) and merge |

:::{note}
All three produce the same relation. Only the data movement differs, so a wrong pick is slow
rather than wrong. That is what makes the strategy safe for Kyber to learn (see
{doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`) rather than something a user must get right.
:::

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
1. Integer keys, one or two `Int64` columns. A row-encoded key would need its `RowConverter`
   shared across morsels.
1. A build side under a fixed row ceiling.

`BroadcastProbe::new` returns `None` for anything else and the caller keeps the materialized
path. Nothing silently changes shape.
:::

:::{tab-item} sort_merge
There is no hash table. Batcher sorts both sides by key and merges them. `join/sort_merge.rs`
skips the sort when the indices already arrive in ascending key order, which it establishes in
one linear pass, and that is what makes this the right pick for already-ordered inputs such as
time series, an upstream `Sort`, or sorted lakehouse files. Output order differs from the hash
join, because these are unordered relations.
:::
::::

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
into one `u128` — the length in the top byte, the value bytes below it — and that *is* a `Copy`
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
a string join against a small dimension table keeps the flat path — correctly, since the whole
build table fits in cache and partitioning it would be overhead.
:::

## Skew and spill

**Skew.** A hash-partitioned bucket that is far hotter than the average would leave one worker
grinding while the rest idle. `par.rs` compares each bucket against the average on both rows and
bytes, using the `skew_bucket_factor` threshold with absolute row and byte floors so a small
bucket is never called skewed. `join_par.rs` then spreads the hot bucket across workers by
broadcasting its build side and chunking its probe side. A `Full` join is ineligible, because it
must reconcile unmatched rows on both sides.

**Spill.** When the build side exceeds the memory envelope, the grace hash join partitions both
sides by key to disk and joins one bucket at a time, so only one build table is ever resident.
The streaming variant does this **one input batch at a time**, so a build side far larger than
memory spills instead of OOMing at the materialize step. Bucket count is sized from the build
batches' total bytes without materializing them, and the fixed-seed partitioner co-locates equal
keys, so the union of per-bucket joins is the full join for every join type.

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
sort                            est≈3 (default)
  hash_join                     est≈3 (default)
    scan                        est≈3 (exact)
    scan                        est≈3 (exact)

decisions:
  - [kyber/selection] join build side: left≈3 right≈3 [exact] → broadcast
  - ...
```

Both sides are three rows, so it broadcasts. The strategy is named in the decisions block, not
in the tree.
:::

The `None` in the left join is the null index in the index-pair builder, made visible.

## Parallelism

Join throughput is set by how much of the operator runs in parallel, and the profile says
exactly where that is decided: the serial prefixes around the parallel per-bucket join. The
radix scatter is now parallel, and the probe side is gathered once instead of concatenated and
re-gathered. Both changes are measured in `benchmarks/BENCHMARK_RESULTS.md`.

Scale-out is the strong axis: on the operator-mix benchmarks Batcher's distributed join beats
Daft's by 1.7x to 2.2x at every scale measured.

## Code map

- `crates/bc-runtime/src/join/mod.rs`: `hash_join_indices`, the bloom gate, `JoinIndices`
- `crates/bc-runtime/src/join/radix.rs`: the parallel three-phase partition
- `crates/bc-runtime/src/join/stream.rs`: `BroadcastProbe`, the streaming probe
- `crates/bc-runtime/src/join/sort_merge.rs`, `asof.rs`: the other two algorithms
- `crates/bc-interp/src/join_par.rs`: grace join, broadcast join, skew detection
- `crates/bc-interp/src/ops/repartition.rs`: gather-once bucket construction

## See also

- {doc}`Architecture </architecture/index>`: why there is one join and not a distributed second one.
- {doc}`Execution engine </architecture/internals/execution>`: the operator around these primitives.
- {doc}`Kyber </architecture/internals/kyber>`: the pass that picks the strategy and the build side.
- {doc}`Joins </user-guide/analyze/joins>`: the API, and how to help the planner.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: the decisions block above, explained.
- {doc}`vs DuckDB </benchmarks/comparisons/vs-duckdb>`: the multi-join gap this page opens with.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: q5, q7, q8, q17 in context.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: the shuffle-into-buckets schedule.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why per-partition joins union to the whole join.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: the grace hash join, when the build side doesn't fit.
