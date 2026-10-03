# Spilling

*Spilling* is how a stateful operator keeps running when its state no longer fits in the memory envelope: it writes part of that state to disk and reads it back in bounded pieces. A query that is too large for RAM gets slower, not dead.

Spilling is a property of the runtime primitive, not a separate operator. There's no "spilling aggregate" node in the IR. The same `Aggregate` runs in memory or out of core depending on whether its reservation was granted, and the result is the same relation either way: the same rows, column names and column types.

:::{important}
"The same" is exact for integer, string and key-based results. Two things can legitimately move, the same two the {doc}`mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>` allows between single-node and distributed: a floating-point `sum` or `mean` can differ in the last bits, and an unordered `array_agg` can list a group's elements in a different order.
:::

## See it spill

Shrink the memory envelope and a sort goes out of core. The answer doesn't change:

```python
import dataclasses
import json
import batcher as bt
from batcher import Config

base = Config()
small = base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=4 << 20))
ds = bt.from_pydict({"x": [float(i % 1000) for i in range(300_000)]})

with bt.config_context(small):
    report = json.loads(ds.sort("x").explain(analyze=True, format="json"))
    out = ds.sort("x").collect()

print(report["spilled"], report["total_spill_bytes"] > 0)  # True True
print(out.column("x")[0].as_py(), out.column("x")[-1].as_py())  # 0.0 999.0
```

Run the same query under the default envelope and `report["spilled"]` is `False`. Spill engages when a reservation fails, not when you set a flag.

## The admission decision

Every stateful operator asks the {doc}`buffer pool </architecture/deep-dives/memory/buffer-pool>` for its estimated bytes before it builds state. A granted reservation runs in memory. A refused one spills. When a pool exists, the *actual outstanding bytes* are the spill authority, and the plan estimate is only what the operator asks for.

:::{dropdown} The `admit` function
Everything routes through `admit` in [`crates/bc-interp/src/par.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/par.rs):

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

The per-operator budget path, `op_budget` keyed by Kyber's pre-order `op_id`, is the fallback for pool-less contexts. The pool also reports a *level* (nominal, elevated, critical) through `engine_pool_stats()`, which nothing inside the data plane acts on.

If `EngineConfig.memory_budget_bytes` is 0, `agg_spill` is `None` and the engine runs fully in memory with no spill machinery engaged. That's what `memory.unbounded_memory` asks for.
:::

## Grace partitioning

Aggregate and distinct spill by grace hashing, in [`crates/bc-runtime/src/agg/spill/mod.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-runtime/src/agg/spill/mod.rs). Phase one packs each morsel's partial state into one batch and routes it by a hash of the group key into a partition file. Phase two reads one partition at a time, combines it and finalizes it:

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
             ▼          ▼          ▼          ▼          ONE AT A TIME, so peak
          combine    combine    combine    combine       resident memory is one
          finalize   finalize   finalize   finalize      partition, not the relation
             └──────────┴─────┬────┴──────────┘
                              ▼
                     the global aggregate
```

The routing hash uses fixed `ahash` seeds, so equal keys co-locate, partitions are key-disjoint, and each one reduces independently. That's the whole correctness argument.

:::{dropdown} When a bucket is still too big
A skewed key set can overflow a single bucket. `merge_partition` re-partitions it with a *different salt* (`0x9E37_79B9_7F4A_7C15` wrapping-multiplied by `depth + 1`, with the low bit set) into `bytes.div_ceil(budget)` sub-buckets, clamped to between 2 and 256, up to `MAX_MERGE_DEPTH = 4`. The different salt isn't an optimization: re-hashing with the same function puts every key back in the same bucket.
:::

## What spills, and how

Two mechanisms carry every stateful operator between them, chosen by what the state is keyed on.

![The two out-of-core mechanisms, and which operator takes which. An operator with keyed state takes grace partitioning: route by hash of the key into P buckets, one Arrow IPC file per bucket, then read one bucket at a time and run the ordinary in-memory kernel over it, the union of the buckets being the whole answer. A bucket still over budget is re-split under a salted hash, to a fan-out of at most 256 and a depth of at most 3, and each sub-bucket is an independent instance of the same operator. An operator with ordered state takes the external merge sort: sort into sized runs cut by size rather than by key, spill each, then merge up to 16 runs at a time with one batch per run resident, taking another pass over everything while more than one run is left, for log base 16 of the run count passes in all. Because the runs are cut by size, 64 MiB by default, key skew cannot defeat that family. What each operator writes differs: an aggregate writes partial state, one row per group per morsel rather than the input rows; a join co-partitions both sides by the join key and keeps only the build bucket resident while the probe streams through it; a window writes whole rows keyed by PARTITION BY, so every bucket holds complete partitions; DISTINCT ON is reduced to one row per key per morsel before it is written; and the sort writes sorted runs, over which median and n_unique stream a single pass instead of holding a per-group list. Both mechanisms return exactly what the in-memory kernel returns, and only peak memory differs.](/_static/diagrams/spill_ladder.svg)

The table names each operator's mechanism and the file that implements it:

| Operator | Mechanism | Code |
|---|---|---|
| Aggregate | grace partition + per-bucket combine | `bc-runtime/src/agg/spill/mod.rs` |
| Distinct / UNION dedup | the same grace path with an empty agg list | `bc-interp/src/par.rs::distinct` |
| Sort (full) | size-bounded sorted runs + bounded k-way merge | `bc-interp/src/ops/external_sort.rs` |
| Hash join | grace: co-partition both sides, join bucket-by-bucket | `bc-interp/src/join_par/mod.rs` |
| ASOF join | partition by the `by` keys | `bc-interp/src/join_par/mod.rs` |
| Window (PARTITION BY) | partition on the partition keys | `bc-interp/src/window_spill.rs` |
| Window, one partition over budget | external sort, then the kernel per chunk with a carried correction | `bc-interp/src/ops/window_stream.rs` |
| ASOF join without `by` | external sort of both sides, then a merge that keeps only candidate right rows | `bc-interp/src/join_par/asof_stream.rs` |
| Range join, right side over budget | block-nested over left and right chunks, unmatched rows decided once | `bc-interp/src/join_par/range_blocked.rs` |
| DISTINCT ON | grace, reduced to one row per key per morsel first | `bc-interp/src/distinct_on_spill.rs` |

Top-N, a `Sort` carrying a `limit`, never spills. It runs a bounded heap, which is already memory-bounded.

The **external sort** grows sorted runs to a quarter of the operator's budget, at least 1 MiB and at most 64 MiB (`DEFAULT_RUN_TARGET_BYTES`), then merges with a bounded fan-in of `execution.sort_merge_fanin`, so peak memory is O(fan-in batches) rather than O(input):

```python
import batcher as bt

print(bt.Config().execution.sort_merge_fanin)  # 16
```

The **grace hash join** sizes its fan-out from the larger of its two sides, `bytes.div_ceil(budget)` clamped to 2..256, without materializing either side, then joins bucket `i` of the left against bucket `i` of the right.

:::{dropdown} Windows and ASOF joins with no key to hash
A window with no `PARTITION BY`, or a partition still over budget after every re-split, is sorted out of core by its partition keys, its order keys and each row's input position. The sorted stream is cut into chunks, and each chunk's `row_number`, `rank`, `dense_rank`, running `count` and frameless `first_value` are corrected by counts carried from the rows before it. `lag` and `lead` run over the chunk plus the rows their offset reaches. Those corrections are integer arithmetic or copied values, so the result is identical to the kernel's. Other window functions in that shape, such as a running float `sum`, are declined and raise `MemoryBudgetExceededError`.

A keyless ASOF join sorts both sides out of core by the join key and keeps only the right rows that can match a left chunk. A range join is decomposed on both sides, with one mark per row deciding the unmatched rows an outer, semi or anti join emits.
:::

:::{dropdown} Holistic aggregates: median, quantile, count_distinct, mode
These have no partial state bounded by a constant, so grace partitioning would only move an unbounded state to disk and back. `bc-interp/src/ops/quantile_spill/` sorts `(group_keys…, value)` out of core and streams the sorted run, computing each group's answer as its rows go past. `bc-interp/src/ops/mixed_spill.rs::try_bounded_mixed_spill` composes the two routes, so `median(x), sum(y)` runs the value list through an external sort and the constant-state aggregate through grace, then merge-aligns them on the group key.

`listagg` and `array_agg` stay on grace. Their output *is* the list, so there's nothing to bound.
:::

## Where the bytes go

Spill files are Arrow IPC stream files, one per partition, named `part-{i}.arrow` inside a private `bc-spill-{pid}-{seq}` directory under the spill root, so many workers can share one `spill_dir`. The directory is removed when its store drops. Every spill setting lives under `memory.*`:

```python
import batcher as bt

mem = bt.Config().memory
print(mem.spill_compression, mem.spill_dir, mem.spill_remote_uri)  # auto None None
```

`carbonite/spill/store.py::TieredSpillStore` writes to local disk first and overflows to object storage:

- **Local tier.** Budget `memory.spill_local_budget_bytes`, clamped to 90% of the measured free space on the spill filesystem (`SPILL_DISK_FRACTION`).
- **Remote tier.** Any `fsspec` URL in `memory.spill_remote_uri`, such as `s3://` or `gs://`, with the `cloud` extra installed. Always compressed: an unset or `"auto"` codec becomes LZ4 there. A local budget of `0` sends every bucket straight to the remote tier.

A missing spill file, such as on a spot node whose scratch disk was reclaimed, maps to a retryable {py:exc}`ResourceError <batcher.ResourceError>`, so the distributed recovery loop recomputes the partition.

:::{dropdown} Compression under `"auto"`
The Rust grace store picks Zstd when the schema carries a blob column (`Binary`, `BinaryView`, `LargeBinary`, `LargeUtf8`, a wide `FixedSizeBinary`, including inside lists, structs and maps) and no compression otherwise. On fast local disk, compressing numeric or string state costs more CPU than the I/O it saves. The Python tiered store leaves the local tier uncompressed and upgrades the remote tier to LZ4. Both fall back to uncompressed if the installed pyarrow lacks the codec. IPC self-describes its compression, so the setting never changes a result.
:::

## Practical limits

- **I/O cost.** The grace aggregate writes and reads its partial state once. The external sort reads every run once per merge pass; raising `sort_merge_fanin` cuts passes at the cost of more open files and merge buffers.
- **Recursion depth.** The aggregate's merge stops at `MAX_MERGE_DEPTH = 4`. The shared grace re-split used by joins, partitioned windows and `DISTINCT ON` stops at `MAX_GRACE_SPLIT_DEPTH = 3` with a fan-out of at most 256 (`bc-interp/src/spill_split.rs`), matched by `dist/spill/buckets.py::GRACE_DEPTH = 3`.
- **One hot key.** No hash split separates a key from itself. Constant-state aggregates stay bounded regardless. `array_agg` runs that bucket over budget. The engine logs a warning when the largest partition exceeds the mean by `SPILL_SKEW_WARN` (3.0); the fix is upstream of the aggregate.
- **Open files.** A grace store holds up to 256 writers, and a skewed re-split can hold that many per level. The distributed path caps buckets at 1024 (`dist/spill/scratch.py::_FD_SAFE_PARTITIONS`). Raise a low open-file limit before deeply skewed spills.
- **Disk full.** `ENOSPC` or `EDQUOT` fails the query with an error naming the directory and bytes written (`RuntimeError::SpillOutOfSpace`). A truncated file can't produce a wrong answer, because every read is checked against the row count written. The engine's own stores don't reserve disk, so give the result cache's disk tier its own `memory.result_cache_disk_max_bytes` or point `memory.spill_dir` at a volume with room for both.

## See also

- {doc}`Architecture </architecture/index>`: why bounded memory is an operator property, not a mode.
- {doc}`Carbonite </architecture/internals/carbonite>`: the resource manager whose reservation failure starts this.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page). It is the v1-era design paper with an errata list at its top, and where it and the code differ the code decides. It covers the distributive equivalence grace rests on.
- {doc}`Performance </user-guide/operate/tuning/performance>`: the memory knobs, and when to raise them.
- {doc}`Troubleshooting </user-guide/operate/running/troubleshooting>`: what to do when a query is spilling and you did not expect it.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: larger-than-memory queries, measured.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: the reservation whose failure triggers all of this.
- {doc}`Aggregation internals </architecture/deep-dives/operators/aggregation-internals>`: the in-memory path grace falls back from.
- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: runs and the k-way merge.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: the in-memory hash join.
