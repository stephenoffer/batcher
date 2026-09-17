# Arrow and memory

Arrow is the only columnar format in this engine. Not the preferred one. The only one.

:::{important}
Every operator, the interpreter, the JIT, the spill files, the network shuffle, and the Python
boundary speak Arrow `RecordBatch`. No internal row struct, no bespoke buffer, no "fast path"
format that has to be converted back. This is a hard invariant, not a preference.
:::

That constraint has teeth, and three consequences follow from it. The Python boundary is
**zero-copy**: a pyarrow `RecordBatch` and a Rust `arrow::RecordBatch` are the same bytes,
described by the Arrow C Data Interface, and nothing is serialized. An operator's state is Arrow
rather than generated code, so a compiled pipeline can be thrown away at a pipeline breaker and
rebuilt without losing progress, because the relational state lives in `bc-runtime` structures.
And spilling and the network shuffle become the same operation with a different sink, because
Arrow IPC serializes what is already in memory.

One `RecordBatch` is a schema and a set of buffers per column, and what the slice and the
FFI boundary cost is decided by what neither of them copies.

![A RecordBatch is a schema plus a set of buffers per column. An Int64 column carries a validity bitmap of one bit per row, absent when nothing is null, and a values buffer of eight bytes per row back to back; it needs no offsets buffer, because every value is the same width and row i begins at byte i times 8. A Utf8 column carries a validity bitmap, an offsets buffer of n plus 1 int32s, and a values buffer holding every row's bytes end to end and unpadded, so row i is the stretch of the values buffer between offset i and offset i plus 1, and no per-row length has to be stored. batch.slice(off, len) copies nothing: the second RecordBatch is an offset and a length over the parent's buffers. But get_array_memory_size still reports the parent's whole allocation, which is why every size decision in the engine measures a morsel with bc_arrow::slice_bytes instead. Those same buffer pointers cross to pyarrow through the Arrow C Data Interface, with no copy and no serialization: the morsel Rust hands back is the batch Python already holds.](/_static/diagrams/arrow_memory_layout.svg)

The cost is that every kernel must be an Arrow kernel or must operate on Arrow buffers, and a
type Arrow does not have is a type the engine does not have.

## Who owns what

The crate DAG points one way, and where a piece of memory machinery lives is decided by it.

| Crate | Owns | Depends on |
|---|---|---|
| `bc-arrow` | `Morsel`, `MorselTarget`, `RuntimeTuning`, and the workspace's single Arrow version pin | arrow only |
| `bc-resource` | `MemoryPool`, `MemoryReservation`, `Pressure`, on `std` + `thiserror` with no Arrow | nothing in the workspace |
| `bc-expr` → `{bc-ir → bc-runtime, bc-codegen}` → `bc-interp` | the operators and the state they hold | strictly downward. `bc-codegen` compiles scalar `Expr`, so it sits beside `bc-ir` rather than under it |
| `bc-py` | the C Data Interface boundary, type normalization, and the global allocator | everything |

`bc-resource` sits at the bottom with no Arrow dependency precisely so that `bc-runtime` and
`bc-transport` can both draw on the same envelope without either depending on the other.

## The morsel

```rust
// crates/bc-arrow/src/lib.rs
pub type Morsel = RecordBatch;
pub const DEFAULT_MORSEL_ROWS: usize = 16_384;
pub const DEFAULT_MORSEL_BYTES: usize = 1 << 20;   // 1 MiB
```

`Morsel` is a type alias, not a wrapper. Naming it separately is scheduler vocabulary, not a
second data structure.

A morsel is full at **either** bound. The row bound (16,384) is cache-tuned for narrow data;
16,384 rows of `Int64` is ~128 KiB. The byte bound exists because a row count is byte-blind:
16,384 rows of multi-MB blobs is gigabytes, and an engine that also serves multimodal workloads
cannot pretend otherwise. `MorselTarget::rows(n)` disables the byte bound (`usize::MAX`), which is
how the historical row-only behavior stays byte-for-byte reproducible.

`bc-arrow` is also where the workspace pins its Arrow version. Crates depend on `bc_arrow`'s
re-exports rather than on `arrow` directly, so an Arrow bump is a one-line change in one file.

## The boundary normalizes types once

`crates/bc-py/src/normalize.rs`.

::::{tab-set}
:::{tab-item} On the way in
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
No operator special-cases a narrow, dictionary, or view-layout input, so the kernel surface stays small enough to test exhaustively, and the JIT can assume `Int64` and `Float64` columns. The rules recurse into `struct`, `list` and `map` children, so an `int32` buried in a struct widens exactly as a top-level one does.

Two exceptions keep the boundary honest. A numeric leaf reached through a list keeps its width, because a `FixedSizeList<Float32>` or `FixedSizeList<UInt8>` is a tensor rather than a column, and doubling it would force a full cast on an otherwise zero-copy path. A column carrying an Arrow extension type passes through untouched. Widening is value-preserving everywhere except `UInt64` above `i64::MAX`, which has no `Int64` spelling, and there the boundary refuses the batch with an error naming the column rather than turning the value into a null.
:::

:::{tab-item} On the way out
```text
execution.shrink_output_dtypes = False   (the default)
    a widened result stays widened

execution.shrink_output_dtypes = True
    a pass-through of a narrow SOURCE column is cast back to its source
    width, where that is lossless
```
Off by default, because a widened result is correct and re-narrowing costs a pass.
:::
::::

You can see it:

```python
import batcher as bt
import pyarrow as pa

t = pa.table({"a": pa.array([1, 2, 3], pa.int32())})
out = bt.from_arrow(t).select("a").collect()

print(type(out).__name__, out.schema.field("a").type)  # widened at the boundary
print(out.column("a").to_pylist())
```

```text
Table int64
[1, 2, 3]
```

{py:meth}`collect() <batcher.Dataset.collect>` returns a pyarrow `Table`: the same buffers the engine produced, handed back through
the C Data Interface. There is no Python list, no dict, no per-row object anywhere in that path.
Converting to Python containers ({py:meth}`to_pydict() <batcher.Dataset.to_pydict>`) is a deliberate, explicit step, and it is a
hot-path tuple touch if you do it inside a loop.

## Accounting bytes correctly

:::{warning}
Arrow arrays share buffers when sliced, and `Array::get_array_memory_size()` reports the **whole
parent buffer** for a slice. Morselize one 32 MB table into 122 morsels, sum that figure, and
you get **3.9 GB**: every morsel re-counting the entire buffer. Carbonite fits its memory model
on this number, so over-counting by ~100x has it budget a hundred times the real footprint. It
then spills, or outright rejects, plans that fit comfortably.
:::

```text
   one 32 MB table, morselized into 122 morsels

   get_array_memory_size()               get_slice_memory_size()
   ───────────────────────               ───────────────────────
   morsel   0  ──►  32 MB                morsel   0  ──►  its own slice
   morsel   1  ──►  32 MB   ← the whole  morsel   1  ──►  its own slice
      ...          ...        parent        ...           ...
   morsel 121  ──►  32 MB     buffer,    morsel 121  ──►  its own slice
   ───────────────────────    counted    ───────────────────────
   total        3.9 GB        122 times  total          32 MB
```

So every size decision measures slices instead:

```rust
// crates/bc-arrow/src/lib.rs
pub fn slice_bytes(array: &ArrayRef) -> u64 {
    let data = array.to_data();
    data.get_slice_memory_size()
        .map_or_else(|_| array.get_array_memory_size() as u64, |b| b as u64)
}
```

`bc_arrow::slice_bytes` is the one definition of a column's own footprint, shared by the sketches, the spill stores, and `bc-interp`. The relation measure `bc_interp::batch_bytes` builds on it and fixes the same bug one level down: morsels of a dictionary-encoded column share one values array, and each slice's size includes all of it. Measured on 600 morsels over a 50,000-entry string dictionary, the naive sum reported 609 MB for 40 MB resident. So `batch_bytes` counts each distinct dictionary once, by buffer address.

The morselizer's average-width guard deliberately keeps the over-counting version, because there
it only makes the guard conservative, and it never skips a per-row byte walk that was needed.

## The memory pool

`crates/bc-resource/src/lib.rs` is Carbonite's enforcement primitive inside the data plane: one process-wide `MemoryPool` with RAII `MemoryReservation`s. The contract is **reserve before you allocate**. A stateful breaker reserves its footprint before it builds or merges state, and a reservation the pool cannot grant forces that operator to spill instead of pushing the process toward OOM.

The pool accounts and admits. It does not decide. It exposes a coarse `Pressure` level, `Nominal`, `Elevated` at 80% of the limit, or `Critical` at the limit, and Carbonite's finer Python ladder reads the configured `memory.soft_limit` and `memory.hard_limit` (0.85 and 0.90) on top of it. The design follows DataFusion's `MemoryPool` and `MemoryReservation`, adopted rather than re-derived, and kept to `std` plus `thiserror` so it can sit at the bottom of the crate DAG. {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>` covers both ladders, cooperative spilling, and where the limit comes from.

## The allocator is a correctness-adjacent choice

Every morsel-parallel operator allocates its output buffers per morsel, and glibc's malloc serves
buffers that size through `mmap`/`munmap`. Each `munmap` must invalidate the mapping on every
core, so it broadcasts a TLB-shootdown IPI. With 96 workers freeing a buffer per morsel, that
interrupt storm is a serialization point inside an embarrassingly parallel scan.

Measured on a 6M-row filter: 21.4 ms sequential; parallel wall time bottomed out at 4.0 ms (5.3x
on 96 cores) and then *regressed* past 32 workers. mimalloc's per-thread heaps recycle the pages
instead of returning them, and the same filter scales to 1.46 ms: 15x, no regression.

`bc-py` installs mimalloc as the `#[global_allocator]`, because it is the cdylib every `bc-*`
crate is linked into, so one declaration covers the whole data plane. It changes no result, only
where the bytes come from, and it is invisible to `cargo test` on the pure crates, which link no
allocator and keep the system one.

### Retention, and the valve that makes it safe

Recycling pages is the point, so `bc-py` lengthens mimalloc's purge delay from its 10 ms default to 10 s, unless `MIMALLOC_PURGE_DELAY` is set. Consecutive queries then reuse the same regions instead of receiving fresh zero pages the kernel must clear on first touch, which was 9.3% of a 9M-row, 13-column hash join.

The cost is a resident set that keeps counting memory the engine has finished with. Three 8M-row Parquet group-bys whose results were dropped left a 1,397 MiB resident set, 1,289 MiB of it the engine's arena, and one forced trim handed **408 MiB** back.

So the retention has a release valve. Once a query commits to the out-of-core path, and before the first bucket is written, Carbonite trims the arena (`carbonite/memory/reclaim.py`, called through `ResourceManager.going_out_of_core`). A spilling query runs on a box where memory is scarce, and a third of a gigabyte nothing will use is a third of a gigabyte closer to an OOM kill. The unmapping costs tens of milliseconds against a spill measured in seconds.

The valve sits at the executor rather than at the spill decision. Three independent signals route a query to disk: admission's counter-offer, the plan's estimated peak, and the resident size of the input. Only the second reads live pressure, so a trim hung off that reading would miss the ordinary way a large query spills. It doesn't try to avoid the spill either. The pressure level is the maximum of two pool utilizations and the process footprint, a trim moves only the footprint, and the de-escalation average is built not to fall on one good reading.

Two details the measurements forced. The trim is *forced*, because an unforced collect walks only the calling thread's heap while the engine allocates on rayon workers: it returned 0 MiB where the forced one returned 408. And the release is measured against the kernel's resident figure, because mimalloc's committed figure doesn't move on a collect at all. `ResourceManager.stats()["reclaim"]` reports attempts and bytes. A rising attempt count with no bytes means the box is genuinely full rather than the engine sitting on memory.

## The 2 GiB offset ceiling

Arrow's `Utf8` and `Binary` use 32-bit offsets, so a single column cannot hold more than 2 GiB of
characters. Concatenating morsels at a pipeline breaker is exactly where a large string column
crosses that line.

`ops/materialize.rs` handles it by *widening*: a column that overflows comes back as `LargeUtf8`
(64-bit offsets), and the output schema is rebuilt from what was actually produced rather than
from the input's declared schema. Every other field passes through unchanged, so this is a no-op
on the overwhelming majority of batches.

That same `materialize` is where the parallel concat lives: independent columns fan across cores,
and a null-free fixed-width primitive column copies via a parallel memcpy (each morsel's values
slice to its own disjoint output offset), saturating memory bandwidth where Arrow's serial
per-chunk `concat` (~3 GB/s) does not. The result is byte-identical to `concat_batches`.

## Where the code lives

- `crates/bc-arrow/src/lib.rs`: `Morsel`, `MorselTarget`, the Arrow pin, `RuntimeTuning`
- `crates/bc-py/src/lib.rs`: the C Data Interface boundary and the global allocator
- `crates/bc-py/src/normalize.rs`: narrow/dictionary normalization in and out
- `crates/bc-resource/src/lib.rs`: `MemoryPool`, `MemoryReservation`, `Pressure`
- `crates/bc-interp/src/ops/materialize.rs`: parallel concat and offset widening
- `crates/bc-arrow/src/lib.rs` (`slice_bytes`) and `crates/bc-interp/src/lib.rs` (`batch_bytes`): slice- and dictionary-aware byte accounting
- `python/batcher/carbonite/memory/reclaim.py`: the arena release valve

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
