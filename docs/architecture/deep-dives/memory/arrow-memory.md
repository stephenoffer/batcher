# Arrow and memory

Arrow is the only columnar format in this engine. Not the preferred one. The only one.

:::{important}
Every operator, the interpreter, the JIT, the spill files, the network shuffle, and the Python boundary speak Arrow `RecordBatch`. No internal row struct, no bespoke buffer, no "fast path" format that has to be converted back.
:::

Three consequences follow. The Python boundary is **zero-copy**: a pyarrow `RecordBatch` and a Rust `arrow::RecordBatch` are the same bytes, described by the Arrow C Data Interface. An operator's state is Arrow rather than generated code, so a compiled pipeline can be rebuilt at a pipeline breaker without losing progress. And spilling and the network shuffle are the same operation with a different sink, because Arrow IPC serializes what is already in memory.

![A RecordBatch is a schema plus a set of buffers per column. An Int64 column carries a validity bitmap of one bit per row, absent when nothing is null, and a values buffer of eight bytes per row back to back; it needs no offsets buffer, because every value is the same width and row i begins at byte i times 8. A Utf8 column carries a validity bitmap, an offsets buffer of n plus 1 int32s, and a values buffer holding every row's bytes end to end and unpadded, so row i is the stretch of the values buffer between offset i and offset i plus 1, and no per-row length has to be stored. batch.slice(off, len) copies nothing: the second RecordBatch is an offset and a length over the parent's buffers. But get_array_memory_size still reports the parent's whole allocation, which is why every size decision in the engine measures a morsel with bc_arrow::slice_bytes instead. Those same buffer pointers cross to pyarrow through the Arrow C Data Interface, with no copy and no serialization: the morsel Rust hands back is the batch Python already holds.](/_static/diagrams/arrow_memory_layout.svg)

{py:meth}`collect() <batcher.Dataset.collect>` returns a pyarrow `Table`: the same buffers the engine produced, with no Python list or per-row object in the path. Converting to Python containers with {py:meth}`to_pydict() <batcher.Dataset.to_pydict>` is a deliberate, explicit step.

## Who owns what

The crate DAG points one way, and it decides where each piece of memory machinery lives:

| Crate | Owns | Depends on |
|---|---|---|
| `bc-arrow` | `Morsel`, `MorselTarget`, `RuntimeTuning`, and the workspace's single Arrow version pin | arrow only |
| `bc-resource` | `MemoryPool`, `MemoryReservation`, `Pressure`, on `std` + `thiserror` with no Arrow | nothing in the workspace |
| `bc-expr` → `{bc-ir → bc-runtime, bc-codegen}` → `bc-interp` | the operators and the state they hold | strictly downward. `bc-codegen` compiles scalar `Expr`, so it sits beside `bc-ir` rather than under it |
| `bc-py` | the C Data Interface boundary, type normalization, and the global allocator | everything |

Crates depend on `bc_arrow`'s re-exports rather than on `arrow` directly, so an Arrow bump is a one-line change in one file. {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>` covers `bc-resource` in full.

## The morsel

```rust
// crates/bc-arrow/src/lib.rs
pub type Morsel = RecordBatch;
pub const DEFAULT_MORSEL_ROWS: usize = 16_384;
pub const DEFAULT_MORSEL_BYTES: usize = 1 << 20;   // 1 MiB
```

`Morsel` is a type alias, not a wrapper. A morsel is full at **either** bound. The row bound is cache-tuned for narrow data: 16,384 rows of `Int64` is ~128 KiB. The byte bound exists because 16,384 rows of multi-MB blobs is gigabytes, and an engine that serves multimodal workloads can't pretend otherwise. `MorselTarget::rows(n)` disables the byte bound to reproduce the historical row-only behavior.

## The boundary normalizes types once

[`crates/bc-py/src/normalize.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/normalize.rs) widens narrow types and decodes dictionary and view layouts on the way in, so no operator special-cases them and the JIT can assume `Int64` and `Float64`. A list of numbers keeps its width, because a `FixedSizeList<Float32>` is a tensor rather than a column:

```python
import batcher as bt
import pyarrow as pa

t = pa.table({
    "a": pa.array([1, 2, 3], pa.int32()),
    "f": pa.array([1.5, 2.5, 3.5], pa.float32()),
    "s": pa.array(["x", "y", "x"]).dictionary_encode(),
    "e": pa.array([[1.0, 2.0]] * 3, pa.list_(pa.float32(), 2)),
})
out = bt.from_arrow(t).collect()
print([str(f.type) for f in out.schema])
# ['int64', 'double', 'string', 'fixed_size_list<item: float>[2]']
```

:::{dropdown} The full normalization table
```text
Int8 / Int16 / Int32             ──►  Int64
UInt8 / UInt16 / UInt32 / UInt64 ──►  Int64
Float16 / Float32                ──►  Float64
Dictionary<K, V>                 ──►  V   (decoded, then normalized)
LargeUtf8 / Utf8View             ──►  Utf8
BinaryView                       ──►  Binary
ListView / LargeListView         ──►  List
RunEndEncoded                    ──►  its value type (runs expanded)
```

The rules recurse into `struct`, `list` and `map` children. A numeric leaf reached through a list keeps its width, and a column carrying an Arrow extension type passes through untouched. Widening is value-preserving everywhere except `UInt64` above `i64::MAX`, which has no `Int64` spelling, and there the boundary refuses the batch with an error naming the column rather than turning the value into a null.
:::

On the way out, a widened result stays widened by default. Set `execution.shrink_output_dtypes` to cast a pass-through of a narrow source column back to its source width where that is lossless:

```python
import dataclasses

cfg = bt.Config()
narrow = cfg.replace(execution=dataclasses.replace(cfg.execution, shrink_output_dtypes=True))
with bt.config_context(narrow):
    print(bt.from_arrow(t).select("a").collect().schema.field("a").type)  # int32
```

## Accounting bytes correctly

Arrow arrays share buffers when sliced, and `Array::get_array_memory_size()` reports the **whole parent buffer** for a slice. Morselize one 32 MB table into 122 morsels and that figure sums to **3.9 GB**. So every size decision measures slices instead:

```rust
// crates/bc-arrow/src/lib.rs
pub fn slice_bytes(array: &ArrayRef) -> u64 {
    let data = array.to_data();
    data.get_slice_memory_size()
        .map_or_else(|_| array.get_array_memory_size() as u64, |b| b as u64)
}
```

`bc_arrow::slice_bytes` is the one definition of a column's own footprint, shared by the sketches, the spill stores and `bc-interp`. The relation measure `bc_interp::batch_bytes` builds on it and counts each distinct dictionary once, by buffer address, because morsels of a dictionary-encoded column share one values array. Measured on 600 morsels over a 50,000-entry string dictionary, the naive sum reported 609 MB for 40 MB resident.

## The allocator

`bc-py` installs mimalloc as the `#[global_allocator]`, which covers the whole data plane because every `bc-*` crate links into that one cdylib. Per-thread heaps recycle the pages that morsel-parallel operators allocate and free, instead of returning them to the kernel with a TLB shootdown per free. Measured on a 6M-row filter, the same query scales to 1.46 ms, 15x over the 21.4 ms sequential time on 96 cores. The allocator changes no result, only where the bytes come from.

:::{dropdown} Retention and the release valve
`bc-py` lengthens mimalloc's purge delay from 10 ms to 10 s, unless `MIMALLOC_PURGE_DELAY` is set, so consecutive queries reuse the same regions instead of receiving fresh zero pages. Kernel page clearing had been 9.3% of a 9M-row, 13-column hash join.

The retained arena is handed back when it matters. Once a query commits to the out-of-core path, and before the first bucket is written, Carbonite force-trims the arena (`carbonite/memory/reclaim.py`, through `ResourceManager.going_out_of_core`). After three 8M-row Parquet group-bys, one forced trim handed **408 MiB** back. The release is measured against the kernel's resident figure, and `ResourceManager.stats()["reclaim"]` reports attempts and bytes.
:::

## The 2 GiB offset ceiling

Arrow's `Utf8` and `Binary` use 32-bit offsets, so one column holds at most 2 GiB of characters, and concatenating morsels at a pipeline breaker is where a large string column crosses that line. `ops/materialize.rs` widens an overflowing column to `LargeUtf8` and rebuilds the output schema from what was produced. The same `materialize` fans independent columns across cores and copies null-free fixed-width columns with a parallel memcpy, byte-identical to `concat_batches`.

## Where the code lives

- [`crates/bc-arrow/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-arrow/src/lib.rs): `Morsel`, `MorselTarget`, the Arrow pin, `RuntimeTuning`, `slice_bytes`
- [`crates/bc-py/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/lib.rs): the C Data Interface boundary and the global allocator
- [`crates/bc-py/src/normalize.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/normalize.rs): narrow/dictionary normalization in and out
- [`crates/bc-resource/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-resource/src/lib.rs): `MemoryPool`, `MemoryReservation`, `Pressure`
- [`crates/bc-interp/src/ops/materialize.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/materialize.rs): parallel concat and offset widening
- [`crates/bc-interp/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/lib.rs) (`batch_bytes`): dictionary-aware byte accounting
- [`python/batcher/carbonite/memory/reclaim.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/carbonite/memory/reclaim.py): the arena release valve

## See also

- {doc}`Architecture </architecture/index>`: the Arrow-only invariant, and the crate DAG it implies.
- {doc}`Carbonite </architecture/internals/carbonite>`: the resource manager that drives this pool.
- {doc}`Execution engine </architecture/internals/execution>`: what the operators do with these buffers.
- {doc}`Type system </user-guide/transform/columns/type-system>`: the types that survive the boundary normalization.
- {doc}`Performance </user-guide/operate/tuning/performance>`: staying out of Python containers on the hot path.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: where the 6M-row filter figures come from.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: what the byte budget is for.
- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: where the zero-copy handoff happens.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: the reservation contract and both pressure ladders, in full.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: where Arrow IPC turns memory into disk.
