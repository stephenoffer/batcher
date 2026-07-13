# Spilling

A hash aggregate over a billion distinct keys does not fit in memory, and neither does the
build side of a join against a table larger than RAM. The two honest options are to fail
the query or to put part of the state on disk. Batcher does the second, so the failure mode
of a too-large query is *slower*, not *dead*.

:::{important}
Spilling is a property of the runtime primitive, not a separate operator. There is no "spilling
aggregate" node in the IR. The same `Aggregate` runs in memory or out of core depending on
whether its reservation was granted, and the result is **bit-identical either way**. A query
that spills is slower. It is not different.
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

Read the first arm carefully: when a pool exists, *actual outstanding bytes* are the spill
authority, not a static plan estimate. The estimate is only what the operator asks for.
The per-operator budget path (`op_budget`, keyed by Kyber's pre-order `op_id`) is the
fallback for pool-less contexts.

If `EngineConfig.memory_budget_bytes` is 0, `agg_spill` is `None` and the engine runs fully
in memory with no spill machinery engaged at all. That is the zero-cost default when you
opt out with `memory.unbounded_memory`.

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
chunk. That is the entire correctness argument: equal keys co-locate, partitions are
key-disjoint; a partition can be reduced independently.

Partition count is sized from the state, not guessed: `grace_partitions` returns
`state_bytes.div_ceil(budget).max(2)`.

### When a bucket is still too big

A skewed key set can overflow a single bucket even after partitioning. `merge_partition`
handles that by recursing: if a partition's bytes still exceed the budget, it re-partitions
with a *different salt* (`salt = 0x9E37_79B9_7F4A_7C15 * (depth+1) | 1`) into sub-buckets
and reduces those one at a time, up to `MAX_DEPTH = 4`.

:::{warning}
The different salt is not an optimization. Re-hashing an overflowing bucket with the same
function puts every key back in the same bucket, and the recursion never terminates.
:::

## What spills, and how

| Operator | Mechanism | Code |
|---|---|---|
| Aggregate | grace partition + per-bucket combine | `bc-runtime/src/agg/spill.rs` |
| Distinct / UNION dedup | the same grace path with an empty agg list | `bc-interp/src/par.rs::distinct` |
| Sort (full) | sorted runs + bounded k-way merge | `bc-interp/src/ops/external_sort.rs` |
| Hash join | grace: co-partition both sides, join bucket-by-bucket | `bc-interp/src/join_par.rs` |
| ASOF join | partition by the `by` keys | `bc-interp/src/join_par.rs` |
| Window (PARTITION BY) | partition on the partition keys | `bc-interp/src/window_spill.rs` |

Top-N (a `Sort` with a `limit`) never spills. It runs a bounded heap, which is already
memory-bounded.

The **external sort** writes each morsel as a sorted run, then merges with a bounded fan-in
(`execution.sort_merge_fanin`, default 16), streaming the output back to disk between
passes. Peak memory is O(fan-in morsels), not O(input).

The **grace hash join** sizes its partition count from `build_bytes.div_ceil(budget)`
*without materializing the build side*. It sums `get_array_memory_size()` over the
incoming batches. Both sides are co-partitioned by join key into two stores, then bucket
`i` of the left is joined against bucket `i` of the right.

### The aggregates that would defeat grace

`median`, `quantile`, `count_distinct`, and `mode` do not have a bounded intermediate
state. Their "partial" is the whole value list, so grace partitioning by group key just
moves an unbounded list to disk and back.

For these, `bc-interp/src/ops/quantile_spill/` takes a different route: sort
`(group_keys…, value)` out of core with the external sort, then stream the sorted run and
compute each group's answer as its rows go past. No group's value list is ever resident.
`try_bounded_mixed_spill` composes the two. A `median(x), sum(y)` aggregate runs the value
list through an external sort and the constant-state one through grace, then merge-aligns
them on the group key.

`listagg` and `array_agg` deliberately return `None` from this path and stay on grace.
Their output *is* the list; there is nothing to bound.

## The file format

:::{dropdown} Arrow IPC on disk: naming, cleanup, and compression
Arrow IPC **stream** format, one file per partition, named `part-{i}.arrow` inside a
private directory `bc-spill-{pid}-{seq}` under the spill root. The per-process, per-store
directory is what lets many Ray workers and sibling breakers share one `spill_dir` without
clobbering each other. `DiskSpillStore` has a `Drop` that removes it.

Compression is `memory.spill_compression`, default `"auto"`, and auto is datatype-aware:

```rust
fn classify(schema: &Schema) -> Self {
    // LargeBinary | Binary | LargeUtf8 present -> Zstd, else None
}
```

On fast local disk, compressing numeric *or string* state costs more CPU than the I/O it
saves. Only blob payloads win. The read path never needs to know the codec, because IPC
self-describes its compression. That is why spill is result-invariant regardless of the
setting.
:::

## Two tiers

Local NVMe is fast but finite. `carbonite/spill.py::TieredSpillStore` overflows to object
storage when the local budget is exhausted.

::::{tab-set}
:::{tab-item} LOCAL
```text
Arrow IPC on the local disk
uncompressed by default — NVMe is fast enough that the codec costs more than it saves
budget: memory.spill_local_budget_bytes (auto), clamped to 90% of MEASURED free disk
```
The clamp is what makes overflow track the disk that actually exists rather than a number
someone typed into a config file two quarters ago.
:::

:::{tab-item} REMOTE
```text
object storage via fsspec, at memory.spill_remote_uri
always compressed — it is slow and it is priced per byte
reached when the local budget is exhausted
```
Setting `memory.spill_local_budget_bytes` to `0` sends every bucket straight here: the "this
node has no usable NVMe" case.
:::
::::

The tier is chosen lazily on the first batch, so an empty bucket opens no file.

A missing spill file (a spot node whose scratch disk was reclaimed) is mapped to a
retryable `ResourceError`, so the distributed recovery loop recomputes the partition
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

This dataset fits comfortably, so it reports `spilled: False`. That is the point. Spill
engages on the reservation failing, not on a flag.

## Costs and limits

Spilling turns a memory-bound query into an I/O-bound one. The grace aggregate writes the
partial state once and reads it once; the external sort writes every run and reads it back
once per merge pass, so a very large sort with a small fan-in pays multiple passes. Raising
`sort_merge_fanin` reduces passes at the cost of more concurrent open files and more
resident merge buffers.

Recursion is bounded at depth 4 (Rust) and 3 (`dist/spill.py::_MAX_SPILL_RECURSION`).

:::{warning}
A key set so skewed that one group's state exceeds the budget on its own cannot be partitioned
out, because no hash split separates a single key from itself. That case degrades to running the
bucket over budget rather than failing, and the engine logs a warning when spill skew
(max-over-mean bytes per partition) exceeds 3.0. If you see that warning, the fix is upstream of
the aggregate, not in the spill configuration.
:::

Simultaneously-open spill files are capped at 1024 (`_fd_safe`) so a wide fan-out does not
exhaust the process file-descriptor limit.

## See also

:::{seealso}
- [Architecture](../architecture/index.md): why bounded memory is an operator property, not a mode
- [Carbonite](../internals/carbonite.md): the resource manager whose reservation failure starts this
- `docs/internals/mathematical_foundations.md` (in the repo, not a site page): the distributive equivalence grace rests on
- [Performance](../user-guide/performance.md): the memory knobs, and when to raise them
- [Troubleshooting](../user-guide/troubleshooting.md): what to do when a query is spilling and you did not expect it
- [Scaling benchmarks](../benchmarks/scaling.md): larger-than-memory queries, measured
- [The buffer pool](buffer-pool.md): the reservation whose failure triggers all of this
- [Aggregation internals](aggregation-internals.md): the in-memory path grace falls back from
- [Sort internals](sort-internals.md): runs and the k-way merge
- [Join algorithms](join-algorithms.md): the in-memory hash join
:::
