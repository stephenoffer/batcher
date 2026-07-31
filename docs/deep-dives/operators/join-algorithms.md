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
{doc}`Learned metadata </deep-dives/adaptive/learned-metadata>`) rather than something a user must get right.
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

## Closing the gap

The join gap against DuckDB is a parallelism gap, and the profile says where it lives: the
serial prefixes around the parallel per-bucket join. Two have been removed (the radix scatter is
now parallel; the probe side is gathered once instead of concatenated and re-gathered). What
remains (build-side materialization, the shuffle's own serial phases) is the open work, and
it's tracked in `benchmarks/BENCHMARK_RESULTS.md` rather than in an aspiration.

The distributed picture is different and better: on the operator-mix benchmarks Batcher's
distributed join beats Daft's by 1.7x to 2.2x at every scale measured. Scale-out isn't the
weak axis. Single-node join parallelism is.

## Code map

- `crates/bc-runtime/src/join/mod.rs`: `hash_join_indices`, the bloom gate, `JoinIndices`
- `crates/bc-runtime/src/join/radix.rs`: the parallel three-phase partition
- `crates/bc-runtime/src/join/stream.rs`: `BroadcastProbe`, the streaming probe
- `crates/bc-runtime/src/join/sort_merge.rs`, `asof.rs`: the other two algorithms
- `crates/bc-interp/src/join_par.rs`: grace join, broadcast join, skew detection
- `crates/bc-interp/src/ops/repartition.rs`: gather-once bucket construction

## See also

- {doc}`Architecture </architecture/index>`: why there is one join and not a distributed second one.
- {doc}`Execution engine </internals/execution>`: the operator around these primitives.
- {doc}`Kyber </internals/kyber>`: the pass that picks the strategy and the build side.
- {doc}`Joins </user-guide/analyze/joins>`: the API, and how to help the planner.
- {doc}`Reading a plan </user-guide/operate/explain-plans>`: the decisions block above, explained.
- {doc}`vs DuckDB </benchmarks/vs-duckdb>`: the multi-join gap this page opens with.
- {doc}`TPC-H benchmarks </benchmarks/tpch>`: q5, q7, q8, q17 in context.
- {doc}`Morsel parallelism </deep-dives/operators/morsel-parallelism>`: the shuffle-into-buckets schedule.
- {doc}`Mergeable algebra </deep-dives/operators/mergeable-algebra>`: why per-partition joins union to the whole join.
- {doc}`Spilling </deep-dives/memory/spilling>`: the grace hash join, when the build side doesn't fit.
