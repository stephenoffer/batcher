# Join algorithms

A join in Batcher is one primitive: a builder that produces a pair of row-index vectors describing the output. Every join type, every strategy, and both the parallel and distributed paths are built on it. This page covers that primitive, the algorithms layered over it, and how the planner picks among them.

On TPC-H at scale factor 1, Batcher's geomean ratio is 0.72 against DuckDB on its own native storage and 0.25 against DuckDB reading the same Arrow input ({doc}`TPC-H benchmarks </benchmarks/results/tpch>`, four-engine board of 2026-09-13 in [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md)). Distributed, its join beats Daft's Ray runner by 1.7x to 2.2x at every scale measured.

```python
import batcher as bt

left = bt.from_pydict({"k": [1, 2, 3], "v": ["a", "b", "c"]})
right = bt.from_pydict({"k": [2, 3, 4], "w": [20, 30, 40]})

print(left.join(right, on="k", how="inner").sort("k").to_pydict())
# {'k': [2, 3], 'v': ['b', 'c'], 'w': [20, 30]}
print(left.join(right, on="k", how="left").sort("k").to_pydict())
# {'k': [1, 2, 3], 'v': ['a', 'b', 'c'], 'w': [None, 20, 30]}
```

The `None` in the left join is a null index in the index-pair builder, made visible.

## One primitive: index pairs

[`crates/bc-runtime/src/join/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/mod.rs) computes two row-index vectors, `(left, right)`. Output column `c` is `take(side_of_c, indices_of_that_side)`. An unmatched row on the null-supplying side gets a null index, and Arrow's `take` yields null for it, which is exactly what an outer join means. So inner, left, right, full, semi, and anti all come from one builder rather than six kernels:

```rust
// crates/bc-runtime/src/join/mod.rs
pub fn hash_join_indices(
    left_keys: &[ArrayRef],
    right_keys: &[ArrayRef],
    join_type: JoinType,
) -> Result<JoinIndices, RuntimeError>
```

Keys are encoded with Arrow's row format, so multi-column keys of any type share one code path, and a row with any null key never matches, because `NULL != NULL`.

:::{important}
Matching is purely by key equality, so a global join equals the union of per-partition joins whenever both sides are hash-partitioned by the join key. That one property is what the parallel and distributed executors both stand on.
:::

## Hash join, and the bloom in front of it

The default builds an open-addressing slot table over the right (build) side, storing a 32-bit hash tag and a row id per slot, and probes it with the left. A key's starting slot is a pure function of its hash, so the probe hashes a block of rows ahead and prefetches their slots: a 10M-row probe of a 1M-row build on spread integer keys went from 133 ms to 99 ms on 16 cores.

```text
      PROBE side (left)                       BUILD side (right)
        encode key                              encode key
             │                                       ▼
             │                              slot table (tag, row id)
             ▼                                       │
      ┌──────────────┐   engaged only when the       │
      │    bloom     │   build side is ≥ 2^16 rows   │
      │              │   AND the probe is ≥ build    │
      └──┬────────┬──┘                               │
    miss │        │ hit                              │
    skip │        └────────────► probe ◄─────────────┘
                                   ▼
            JoinIndices { left: [...], right: [...] }
            output column c = take(side_of_c, indices_of_that_side)
```

The bloom engages only when it pays. Below ~64K build rows (`bloom_min_build_rows`, 2^16) the table is cache-resident and a bloom is pure overhead. Above it, each probe becomes a random cache miss that a compact bloom can skip for a non-matching key. A bloom has no false negatives, so it can only skip a provably empty chain and never changes a result.

## Three strategies, one relation

`RelOp::HashJoin` carries a `strategy` the planner (Kyber) chooses, and `explain()` names it in the decisions block:

| Strategy | Chosen when | Data movement |
|---|---|---|
| `hash` | the default: neither side is broadcastable | both sides hash-partitioned by key into one bucket per worker |
| `broadcast` | the build side fits `optimizer.broadcast_max_bytes` | the build side is replicated; the probe side never moves |
| `sort_merge` | the build side is too big to hash: at least 50 M rows per worker | no hash table; sort (or skip the sort) and merge |

```python
dim = bt.from_pydict({"k": [1, 2], "name": ["x", "y"]})
facts = bt.from_pydict({"k": [1, 2, 1, 3], "v": [10, 20, 30, 40]})
print(facts.join(dim, on="k").explain())
# ...
# decisions:
#   - [kyber/selection] join build side: left≈4 right≈2 [exact] → broadcast
```

All three produce the same relation, so a wrong pick is slow rather than wrong, which is what makes the strategy safe for Kyber to learn (see {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`).

![How a join strategy and a build side are picked. The decision is made from each side's estimated rows times row width, the join type, since only an inner join may swap sides, and the broadcast ceiling, which is a quarter of L3 and on a cluster that times 16 with a 64 MiB floor. If the smaller side is under the ceiling the join broadcasts and the probe side never moves; otherwise a build side above the row floor of 50 million rows per worker takes sort_merge, which builds no hash table, and everything else takes hash, the shuffle join that partitions both sides by key. The runtime always builds on the right, so Kyber swaps the inputs when the smaller side is the left, and only an inner join may be swapped. The same decision is then made twice more: the bandit substitutes a learned arm before the run, and at run time the driver shuffles a broadcast that measured bytes show no longer fits. All three arms produce the same relation, so a wrong pick is slow rather than wrong, which is what makes the choice safe to learn across runs.](/_static/diagrams/join_strategy_choice.svg)

::::{tab-set}
:::{tab-item} hash (shuffle)
Hash-partition both sides by the join key into one bucket per worker and join the buckets in parallel. Equal keys land in the same bucket, so the per-bucket joins are independent and their union is the full join. `ops/repartition.rs` builds each bucket directly from the morsels with Arrow's `interleave`, so each row is gathered once and each bucket stays one contiguous `RecordBatch`.
:::

:::{tab-item} broadcast
When the build side is small enough to replicate, the probe side joins with no shuffle, parallelized over row ranges. `join/stream.rs` builds the hash table once and probes one morsel at a time, so the probe relation is never concatenated. The streaming probe requires a left-driven join type (`Inner`, `Left`, `Semi`, `Anti`), integer keys or a single byte-string key, and a build side under `RADIX_MIN_BUILD_ROWS_BROADCAST`. Anything else keeps the materialized path.
:::

:::{tab-item} sort_merge
No hash table: sort both sides by key and merge. `join/sort_merge.rs` skips the sort when the indices already arrive in key order, found in one linear pass. The strategy is selected by `SORT_MERGE_MIN_ROWS` (50 M rows per worker), not by input order, because hash wins on ordered input whenever the build side fits.
:::
::::

## Making the build side cheap

The build is the join's sequential prefix, and three pieces of `bc-runtime/src/join/` shrink it:

- **Direct map** (`dense.rs`): a surrogate key such as `o_orderkey` is a near-contiguous run of integers, so `map[key - lo]` holds the chain head and neither side hashes.
- **Sharded build** (`build.rs`): when the table must be hashed, it's sharded by a slice of the hash so every core builds at once. Build and probe agree on a key's shard without communicating, and chains come out in serial order.
- **Key filter** (`key_filter.rs`): the build side's key set is pushed down the probe pipeline to the scan, so a row whose key is absent is dropped before any predicate above it runs. On TPC-H q21 the 6M-row `lineitem` probe is cut about 24x before its date predicate. The filter is the literal key set, never a sketch, so it has no false negatives. On TPC-DS sf1 q37, ordering several such filters most-selective-first took the query from 27.0 ms to 16.0 ms.

## Radix partitioning

Both radix join paths scatter each side's non-null rows into cache-sized partitions carrying the key inline as `(key, abs_row)`. `join/radix.rs` runs the textbook three-phase parallel partition, and its output is bit-identical to a serial scatter, which the `seq == par` oracle depends on:

```text
   phase 1: histogram          phase 2: prefix sum        phase 3: scatter
   chunk 0 ─► counts ─┐
   chunk 1 ─► counts ─┤   exclusive prefix sum over       each chunk writes into
   chunk 2 ─► counts ─┼─► (chunk, partition) reserves ──► its own reserved slices,
   chunk 3 ─► counts ─┘   every chunk a disjoint slice    in increasing row order
```

The radix key must be a single `Copy` value. A byte key of fifteen bytes or fewer packs losslessly into one `u128` (length in the top byte, value bytes below), so ids, codes, SKUs, ISO dates and categoricals reach the same parallel radix join as integers, up to **22.4x** faster than the flat path they used to take. The packing is injective, so matches are exactly those the byte comparison would produce. Longer keys, and builds under `RADIX_MIN_BUILD_ROWS` (65,536), keep the flat path, where the build table fits in cache anyway.

## Skew and spill

**Skew.** `par.rs` flags a bucket far hotter than the average on rows and bytes (`skew_bucket_factor`, with absolute floors), and `join_par/mod.rs` spreads it across workers by broadcasting its build side and chunking its probe side. A `Full` join is ineligible.

**Spill.** When the build side exceeds the memory envelope, the grace hash join partitions both sides by key to disk, one input batch at a time, and joins one bucket pair at a time, so only one build table is ever resident. The fixed-seed partitioner co-locates equal keys, so the union of per-bucket joins is the full join for every join type.

```python
from batcher.config import Config, MemoryConfig, config_context

l = bt.from_pydict({"k": list(range(1000)), "a": list(range(1000))})
r = bt.from_pydict({"k": list(range(0, 1000, 2)), "b": list(range(500))})
with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=1))):
    print(l.join(r, on="k").count())  # same answer under a one-byte budget
# 500
```

![The grace hash join, from admission to one bucket pair. admit sizes the build side as its Arrow bytes plus 12 bytes per build row; if it fits, one hash table is built on the right and probed by the left. If it does not, both sides are partitioned by the same hash of the join key, one batch at a time so neither side is ever fully materialized, and written to disk as join-left/part-i.arrow and join-right/part-i.arrow, with the bucket count sized from the larger side divided by the budget, from 2 to 256. Each bucket pair is then joined on its own, and only the build bucket is resident: the probe bucket streams past it in chunks, so the cost is one bucket rather than two. A build bucket still over budget is re-partitioned with a fresh salt, at most three deep, because a re-split is a re-hash and so cannot separate rows that share a key, which leaves one hot key in one bucket at every level.](/_static/diagrams/hash_join_spill.svg)

## Range joins

An inequality, interval-containment or band join is `RelOp::RangeJoin`, answered without materializing the cartesian product by [`crates/bc-runtime/src/join/range/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-runtime/src/join/range). It emits the same `JoinIndices` the hash join does:

```python
events = bt.from_pydict({"t": [3, 12, 25], "reading": [0.4, 0.9, 0.7]})
shifts = bt.from_pydict({"start": [0, 10, 20], "end": [10, 20, 30], "crew": ["x", "y", "z"]})
inside = events.join_where(shifts, bt.col("t") >= bt.col("start"), bt.col("t") < bt.col("end"))
print(inside.select("t", "crew").sort("t").to_pydict())  # explain() shows a range_join node
# {'t': [3, 12, 25], 'crew': ['x', 'y', 'z']}
```

| Condition | Algorithm |
|---|---|
| one inequality | sort the right side once; each left row's matches are a contiguous suffix found by binary search |
| a band (`L.a <= R.y AND R.y <= L.b`) | a slice of one sorted array whose bounds move monotonically with the left key (`band.rs`) |
| two general inequalities | IEJoin, the algorithm behind DuckDB's `PhysicalIEJoin`: sort both axes, sweep one, read matches off a mark array |
| a right side of at most 32 rows | no sort: `|R|` vectorized comparisons over the left key column (`small.rs`) |

The sorts and the sweep fan out across cores and fold back in order, so the parallel output is identical to the sequential sweep. Distributed, a range join broadcasts its right side; a right side over `optimizer.broadcast_max_bytes` raises a `PlanError` naming the fixes rather than running the whole join on one node.

## ASOF

`join/asof.rs` matches each left row to the right row whose `on` key is nearest in a direction, within the same `by` group. Every left row is emitted, with a null right index when nothing matched:

```python
trades = bt.from_pydict({"sym": ["A", "A", "B"], "t": [10, 40, 10], "size": [100, 200, 50]})
quotes = bt.from_pydict({"sym": ["A", "A", "B"], "t": [8, 38, 1], "price": [1.0, 1.1, 9.0]})
print(trades.join_asof(quotes, on="t", by="sym").sort("sym", "t").to_pydict())
# {'sym': ['A', 'A', 'B'], 't': [10, 40, 10], 'size': [100, 200, 50], 'price': [1.0, 1.1, 9.0]}
```

A `by`-keyed ASOF co-partitions by hash, since a match only pairs rows inside one group.

:::{dropdown} Distributing an ASOF with no `by` keys
A keyless ASOF has no group to hash: any left row may match any right row, decided by a global order on `on`. Batcher range-partitions instead, as the distributed sort does. It samples the left key, cuts it into ordered intervals, and sends both sides through the same boundary list, so a match inside an interval is already local.

A left row can still match a right row in an earlier bucket (`backward`) or a later one (`forward`). Because the intervals are ordered, only one right row per direction can ever win such a match: the largest below the bucket, or the smallest above it. Batcher lends each bucket those rows before the reducer runs, at a cost of one row per bucket per direction. The carried row is the boundary member of a tie group (the last for backward, the first for forward), and a `tolerance` doesn't shrink the carry, because the engine decides nearness after the row arrives.

The buckets concatenate in key order, a permutation of the single-node result, which emits rows in left-input order.
:::

## Code map

- [`crates/bc-runtime/src/join/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/mod.rs): `hash_join_indices`, the bloom gate, `JoinIndices`
- [`crates/bc-runtime/src/join/dense.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/dense.rs), `build.rs`: the direct-map build and the sharded parallel build
- [`crates/bc-runtime/src/join/slots.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/join/slots.rs): the slot table the hash probe prefetches
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
- {doc}`vs Daft </benchmarks/comparisons/vs-daft>`: the distributed join against Daft.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: the shuffle-into-buckets schedule.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why per-partition joins union to the whole join.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: the grace hash join, when the build side doesn't fit.
