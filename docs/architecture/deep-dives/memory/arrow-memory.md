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

```text
    Python                      │                       Rust
    ──────────────────────      │      ──────────────────────────
    pa.RecordBatch              │      arrow::RecordBatch
          │                     │            │
          └────► ArrowArray / ArrowSchema pointers ◄────┘
                 (the Arrow C Data Interface)
                              │
                              ▼
                 ┌──────────────────────────────┐
                 │  ONE set of buffers          │   values
                 │  in ONE allocation           │   validity bitmap
                 │  neither side owns a copy    │   offsets
                 └──────────────────────────────┘

    Nothing is serialized. There is no Python object per row anywhere in this path.
    collect() hands back the same buffers the engine produced.
```

The cost is that every kernel must be an Arrow kernel or must operate on Arrow buffers, and a
type Arrow does not have is a type the engine does not have.

## Who owns what

The crate DAG points one way, and where a piece of memory machinery lives is decided by it.

| Crate | Owns | Depends on |
|---|---|---|
| `bc-arrow` | `Morsel`, `MorselTarget`, `RuntimeTuning`, and the workspace's single Arrow version pin | arrow only |
| `bc-resource` | `MemoryPool`, `MemoryReservation`, `Pressure`, on `std` + `thiserror` with no Arrow | nothing in the workspace |
| `bc-expr` → `bc-ir` → `bc-runtime` / `bc-codegen` → `bc-interp` | the operators and the state they hold | strictly downward |
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
Int8 / Int16 / Int32            ──►  Int64
UInt8 / UInt16 / UInt32 / UInt64 ──►  Int64
Float16 / Float32               ──►  Float64
Dictionary<K, V>                ──►  V   (decoded to its normalized value type)
```
So no operator special-cases a narrow or dictionary input, and the kernel surface stays small
enough to test exhaustively. This is value-preserving, and it is why the JIT can assume `Int64`
and `Float64` columns.
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

print(type(out).__name__, out.schema.field("a").type)   # widened at the boundary
print(out.column("a").to_pylist())
```

```text
Table int64
[1, 2, 3]
```

`collect()` returns a pyarrow `Table`: the same buffers the engine produced, handed back through
the C Data Interface. There is no Python list, no dict, no per-row object anywhere in that path.
Converting to Python containers (`to_pydict()`) is a deliberate, explicit step, and it is a
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

So `bc-interp`'s `batch_bytes` uses `get_slice_memory_size()`:

```rust
// crates/bc-interp/src/lib.rs
pub(crate) fn batch_bytes(batches: &[RecordBatch]) -> u64 {
    batches.iter()
        .flat_map(|b| b.columns().iter())
        .map(|c| c.to_data().get_slice_memory_size().unwrap_or(0) as u64)
        .sum()
}
```

The morselizer's average-width guard deliberately keeps the over-counting version, because there
it only makes the guard conservative, and it never skips a per-row byte walk that was needed.

## The memory pool

`crates/bc-resource/src/lib.rs` is Carbonite's enforcement primitive inside the data plane: one
process-wide `MemoryPool` with RAII `MemoryReservation`s. The contract is **reserve before you
allocate**. A stateful breaker reserves its footprint before it builds or merges state, and a
reservation the pool cannot grant (because other live reservations have filled the envelope)
forces that operator to spill instead of pushing the process toward OOM.

The pool is policy-free. It accounts and admits; it does not decide. What it exposes is a coarse
pressure level derived from `used / limit`:

| Pressure | Meaning |
|---|---|
| `Nominal` | below the soft line: no throttling |
| `Elevated` | at/above the soft line: spill proactively, narrow the in-flight window |
| `Critical` | at/above the hard cap: a new reservation succeeds only after something spills |

One signal, read by every backpressure mechanism the engine has: proactive spill, the morsel
admission gate, and the distributed credit window. Single-node and distributed throttle off the
same envelope rather than each inventing a threshold. Carbonite sets the limit and the soft
fraction (soft 85% / hard 90% of the budget by default) and drives the pool through `bc-py`.

The design follows DataFusion's `MemoryPool`/`MemoryReservation`, adopted rather than
re-derived, and kept dependency-light (std + `thiserror`) so it can sit at the bottom of the crate
DAG.

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
- `crates/bc-interp/src/lib.rs`: slice-aware byte accounting

## See also

- {doc}`Architecture </architecture/index>`: the Arrow-only invariant, and the crate DAG it implies.
- {doc}`Carbonite </architecture/internals/carbonite>`: the resource manager that drives this pool.
- {doc}`Execution engine </architecture/internals/execution>`: what the operators do with these buffers.
- {doc}`Type system </user-guide/transform/columns/type-system>`: the types that survive the boundary normalization.
- {doc}`Performance </user-guide/operate/tuning/performance>`: staying out of Python containers on the hot path.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: where the 6M-row filter figures come from.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: what the byte budget is for.
- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: where the zero-copy handoff happens.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: the pressure ladder above, in full.
