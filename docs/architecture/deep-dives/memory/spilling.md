# Spilling

*Spilling* is how a stateful operator keeps running when its state no longer fits in the
memory envelope: it writes part of that state to disk and reads it back in bounded pieces.
This page describes when Batcher decides to spill, the grace-partitioning algorithm behind
it, which operators spill and how, and what spilling costs you.

A hash aggregate over a billion distinct keys doesn't fit in memory, and neither does the
build side of a join against a table larger than RAM. The two honest options are to fail
the query or to put part of the state on disk. Batcher does the second, so the failure mode
of a too-large query is *slower*, not *dead*.

:::{important}
Spilling is a property of the runtime primitive, not a separate operator. There's no "spilling
aggregate" node in the IR. The same `Aggregate` runs in memory or out of core depending on
whether its reservation was granted, and the result is **bit-identical either way**. A query
that spills is slower. It isn't different.
:::

## The admission decision

Everything routes through one function, `admit` in `crates/bc-interp/src/par.rs`:

```rust
enum Admit { InMemory(Option<MemoryReservation>), Spill }

fn admit(opts: &ExecOptions, op_id: u32, estimate_bytes: usize) -> Admit {
    match opts.pool.as_ref() {
        Some(pool) => match pool.try_reserve_cooperative(estimate_bytes) {
            Ok(reservation) => Admit::InMemory(Some(reservation)),
            Err(_) if opts.agg_spill.is_some() => Admit::Spill,
            Err(_) => Admit::InMemory(None),
        },
        None if opts.op_budget(op_id).is_some_and(|b| estimate_bytes > b) => Admit::Spill,
        None => Admit::InMemory(None),
    }
}
```

Read the first arm carefully. When a pool exists, *actual outstanding bytes* are the spill
authority, not a static plan estimate. The estimate is only what the operator asks for.
The per-operator budget path, `op_budget` keyed by Kyber's pre-order `op_id`, is the
fallback for pool-less contexts.

If `EngineConfig.memory_budget_bytes` is 0, `agg_spill` is `None` and the engine runs fully
in memory with no spill machinery engaged at all. That's the zero-cost default when you opt
out with `memory.unbounded_memory`.

## Grace partitioning

The core algorithm for aggregate and distinct is grace hashing, in
`crates/bc-runtime/src/agg/spill.rs`:

```rust
pub fn combine_finalize_spilling(
    chunk_partials: impl IntoIterator<Item = Partial>,
    funcs: &[AggFunc],
    store: &mut dyn SpillStore,
    budget_bytes: usize,
) -> Result<GroupAggResult, RuntimeError>
```

Phase one: `pack_partial` flattens each `Partial` into one batch (group columns, then state
columns), and `route` hashes the group key and appends the shard to its partition file.
Phase two: read one partition at a time, `combine` it, `finalize` it.

```text
   morsel ──partial──┐
   morsel ──partial──┤   pack_partial:  one batch of [ group cols | state cols ]
   morsel ──partial──┘             │
                                   │  route:  hash(group key) → partition id
                                   ▼
        ┌──────────┬──────────┬──────────┬──────────┐
        │  part-0  │  part-1  │  part-2  │  part-3  │   Arrow IPC stream files, in
        │  .arrow  │  .arrow  │  .arrow  │  .arrow  │   bc-spill-{pid}-{seq}/
        └────┬─────┴────┬─────┴────┬─────┴────┬─────┘
             │          │          │          │
             ▼          ▼          ▼          ▼          ONE AT A TIME, so peak
          combine    combine    combine    combine       resident memory is one
          finalize   finalize   finalize   finalize      partition, not the relation
             │          │          │          │
             └──────────┴─────┬────┴──────────┘
                              ▼
                     the global aggregate
```

The routing hash uses fixed `ahash` seeds so a key lands in the same bucket across every
chunk. That's the entire correctness argument. Equal keys co-locate, so partitions are
key-disjoint and each one can be reduced independently.

Partition count is sized from the state, not guessed: `grace_partitions` returns
`state_bytes.div_ceil(budget).max(2)`.

### When a bucket is still too big

A skewed key set can overflow a single bucket even after partitioning. `merge_partition`
handles that by recursing: if a partition's bytes still exceed the budget, it re-partitions
with a *different salt* (`salt = 0x9E37_79B9_7F4A_7C15 * (depth+1) | 1`) into sub-buckets
and reduces those one at a time, up to `MAX_DEPTH = 4`.

:::{warning}
The different salt isn't an optimization. Re-hashing an overflowing bucket with the same
function puts every key back in the same bucket, and the recursion never terminates.
:::

## What spills, and how

Every stateful operator has a spill path, and each one uses a mechanism suited to its
state. The table names the mechanism and the file that implements it:

| Operator | Mechanism | Code |
|---|---|---|
| Aggregate | grace partition + per-bucket combine | `bc-runtime/src/agg/spill.rs` |
| Distinct / UNION dedup | the same grace path with an empty agg list | `bc-interp/src/par.rs::distinct` |
| Sort (full) | sorted runs + bounded k-way merge | `bc-interp/src/ops/external_sort.rs` |
| Hash join | grace: co-partition both sides, join bucket-by-bucket | `bc-interp/src/join_par.rs` |
| ASOF join | partition by the `by` keys | `bc-interp/src/join_par.rs` |
| Window (PARTITION BY) | partition on the partition keys | `bc-interp/src/window_spill.rs` |

Top-N, a `Sort` carrying a `limit`, never spills. It runs a bounded heap, which is already
memory-bounded.

The **external sort** writes each morsel as a sorted run, then merges with a bounded
fan-in of `execution.sort_merge_fanin`, default 16, streaming the output back to disk
between passes. Peak memory is O(fan-in morsels), not O(input).

The **grace hash join** sizes its partition count from `build_bytes.div_ceil(budget)`
*without materializing the build side*. It sums `get_array_memory_size()` over the
incoming batches. Both sides are co-partitioned by join key into two stores, then bucket
`i` of the left is joined against bucket `i` of the right.

### The aggregates that would defeat grace

`median`, `quantile`, `count_distinct`, and `mode` have no bounded intermediate state.
Their "partial" is the whole value list, so grace partitioning by group key only moves an
unbounded list to disk and back.

For these, `bc-interp/src/ops/quantile_spill/` takes a different route. It sorts
`(group_keys…, value)` out of core with the external sort, then streams the sorted run and
computes each group's answer as its rows go past, so no group's value list is ever
resident. `bc-interp/src/ops/mixed_spill.rs::try_bounded_mixed_spill` composes the two
routes. A `median(x), sum(y)` aggregate runs the value list through an external sort and
the constant-state aggregate through grace, then merge-aligns them on the group key.

`listagg` and `array_agg` deliberately return `None` from this path and stay on grace.
Their output *is* the list, so there's nothing to bound.

## The file format

:::{dropdown} Arrow IPC on disk: naming, cleanup, and compression
Arrow IPC **stream** format, one file per partition, named `part-{i}.arrow` inside a
private directory `bc-spill-{pid}-{seq}` under the spill root. The per-process, per-store
directory is what lets many Ray workers and sibling breakers share one `spill_dir` without
clobbering each other. `DiskSpillStore` has a `Drop` that removes the directory.

Compression is `memory.spill_compression`, default `"auto"`, and auto is datatype-aware:

```rust
fn classify(schema: &Schema) -> Self {
    // LargeBinary | Binary | LargeUtf8 present -> Zstd, else None
}
```

On fast local disk, compressing numeric *or string* state costs more CPU than the I/O it
saves. Only blob payloads win. The read path never needs to know the codec, because IPC
self-describes its compression. That's why spill is result-invariant regardless of the
setting.
:::

## Two tiers

Local NVMe is fast but finite. `carbonite/spill.py::TieredSpillStore` writes to local disk
first and overflows to object storage when the local budget is exhausted.

The local tier writes Arrow IPC files to the spill directory. Its budget is
`memory.spill_local_budget_bytes`, which defaults to `None` and is then derived from
measured free disk. Whatever the budget, the store clamps it to 90% of the *measured* free
space on the filesystem holding the spill directory (`_SPILL_DISK_FRACTION`). That clamp is
what makes overflow track the disk that actually exists rather than a number someone typed
into a config file two quarters ago, and the remaining sliver leaves room for other tenants
and for log and temp writes.

The remote tier is any `fsspec` URL set as `memory.spill_remote_uri`, such as `s3://` or
`gs://`. Reaching it needs the `cloud` extra installed. It's always compressed: object
storage is slow and priced by the byte, so an unset or `"auto"` codec is upgraded to LZ4
there even though the local tier stays uncompressed under `"auto"`. Setting
`memory.spill_local_budget_bytes` to `0` sends every bucket straight to the remote tier,
which covers the case where a node has no usable local scratch disk.

:::{note}
The `"auto"` codec means different things on the two spill paths. The Rust grace store
picks Zstd when the schema carries a blob column and no compression otherwise. This Python
tiered store leaves the local tier uncompressed under `"auto"` and upgrades the remote tier
to LZ4. Both degrade silently to uncompressed if the codec isn't built into the installed
pyarrow, so spilling never fails on a missing optional codec.
:::

The tier is chosen lazily on the first batch, so an empty bucket opens no file.

A missing spill file, such as on a spot node whose scratch disk was reclaimed, maps to a
retryable {py:exc}`ResourceError <batcher.ResourceError>`, so the distributed recovery loop recomputes the partition
instead of crashing.

## Observing it

`explain(analyze=True, format="json")` reports spill per operator and in total.

```python
import json
import batcher as bt

ds = bt.from_pydict({"g": [i % 500 for i in range(20_000)], "x": [float(i) for i in range(20_000)]})
report = json.loads(ds.group_by("g").agg(s=bt.sum("x")).explain(analyze=True, format="json"))
print("spilled:", report["spilled"], "bytes:", report["total_spill_bytes"])
for op in report["ops"]:
    print(op["kind"], "spilled=", op["spilled"], "spill_bytes=", op["spill_bytes"])
```

This dataset fits comfortably, so it reports `spilled: False`. That's the point. Spill
engages when a reservation fails, not when you set a flag.

## Costs and limits

Spilling turns a memory-bound query into an I/O-bound one. The grace aggregate writes the
partial state once and reads it once. The external sort writes every run and reads it back
once per merge pass, so a very large sort with a small fan-in pays multiple passes. Raising
`sort_merge_fanin` reduces passes at the cost of more concurrent open files and more
resident merge buffers.

Recursion is bounded at depth 4 in Rust (`MAX_DEPTH`) and 3 on the out-of-core spill path
(`dist/spill/buckets.py::GRACE_DEPTH`). That bound is one value for every breaker that
grace-splits, so the aggregate, the join and the partitioned window all stop re-partitioning
at the same depth.

:::{warning}
A key set so skewed that one group's state exceeds the budget on its own can't be partitioned
out, because no hash split separates a single key from itself. That case degrades to running the
bucket over budget rather than failing. The engine logs a warning when spill skew, the
largest partition's bytes over the mean non-empty partition's, exceeds `SPILL_SKEW_WARN`
of 3.0. If you see that warning, the fix is upstream of the aggregate, not in the spill
configuration.
:::

On the out-of-core spill path, `dist/spill/scratch.py::_fd_safe` caps the bucket count at
1024 (`_FD_SAFE_PARTITIONS`) so a wide fan-out doesn't exhaust the process file-descriptor
limit.

## See also

- {doc}`Architecture </architecture/index>`: why bounded memory is an operator property, not a mode.
- {doc}`Carbonite </architecture/internals/carbonite>`: the resource manager whose reservation failure starts this.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page): the distributive equivalence grace rests on.
- {doc}`Performance </user-guide/operate/tuning/performance>`: the memory knobs, and when to raise them.
- {doc}`Troubleshooting </user-guide/operate/running/troubleshooting>`: what to do when a query is spilling and you did not expect it.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: larger-than-memory queries, measured.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: the reservation whose failure triggers all of this.
- {doc}`Aggregation internals </architecture/deep-dives/operators/aggregation-internals>`: the in-memory path grace falls back from.
- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: runs and the k-way merge.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: the in-memory hash join.
