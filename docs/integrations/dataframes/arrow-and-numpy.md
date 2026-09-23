# Arrow and NumPy

This page covers the layer every other integration in this section is built on: PyArrow tables in and out of Batcher, NumPy arrays in and out, and what happens to a tensor column on the way.

Arrow is not one of Batcher's supported formats. It is the only one. Every operator, every execution tier, and the Python-to-Rust boundary all speak Arrow `RecordBatch`, so handing a PyArrow table to Batcher is not a conversion at all.

| | |
| --- | --- |
| **Read** | {py:obj}`bt.from_arrow(table) <batcher.from_arrow>`, {py:obj}`bt.from_numpy(array) <batcher.from_numpy>` |
| **Write** | {py:obj}`ds.to_arrow() <batcher.Dataset.to_arrow>`, {py:obj}`ds.to_numpy() <batcher.Dataset.to_numpy>` |
| **Extra** | None. PyArrow is a required dependency; `numpy` for the array side |
| **Cost** | Zero-copy into Arrow. `to_numpy` copies only where a dtype demands it |

## Arrow in, Arrow out

```python
import batcher as bt
import pyarrow as pa

table = pa.table({"id": pa.array([1, 2, 3]), "score": pa.array([0.5, 1.5, 2.5])})
high = bt.from_arrow(table).filter(bt.col("score") > 1)
print(high.to_arrow().to_pydict())
# {'id': [2, 3], 'score': [1.5, 2.5]}
```

The claim that this costs nothing is checkable rather than a figure of speech. A round trip hands back the same memory:

```python
original = table.column("id").chunk(0).buffers()[1]
returned = bt.from_arrow(table).to_arrow().column("id").chunk(0).buffers()[1]
print(original.address == returned.address)
# True
```

`from_arrow` accepts a `Table`, a single `RecordBatch`, or a sequence of batches. {py:obj}`bt.from_batches <batcher.from_batches>` takes a factory instead, for a source that produces batches lazily rather than a list that is already in memory.

## NumPy in, NumPy out

A dict of 1-D arrays becomes a dataset with one column per key:

```python
import numpy as np

ds = bt.from_numpy({"a": np.arange(3), "b": np.arange(3) * 1.5})
print(ds.to_pydict())
# {'a': [0, 1, 2], 'b': [0.0, 1.5, 3.0]}
```

A single 2-D array becomes one column whose rows are the array's rows. That is the shape a feature matrix already has, so it does not need to be unpacked into named columns first:

```python
matrix = np.arange(6, dtype="float64").reshape(3, 2)
print(bt.from_numpy(matrix, column="vec").to_pydict())
# {'vec': [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]}
```

{py:obj}`ds.to_numpy() <batcher.Dataset.to_numpy>` returns a `{column: ndarray}` dict, and a tensor column comes back with its shape restored as `(n, *shape)` rather than as a list of lists:

```python
arrays = bt.from_pydict({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]}).to_numpy()
print({name: (array.dtype.str, array.tolist()) for name, array in arrays.items()})
# {'x': ('<i8', [1, 2, 3]), 'y': ('<f8', [1.0, 2.0, 3.0])}
```

Pass `columns=` to take a subset, which also prunes the scan: a column you don't ask for is never read.

## Narrow types widen at the boundary

Batcher normalizes narrow numeric types once, at the FFI boundary: `Int8`, `Int16` and `Int32` become `Int64`, and `Float16` and `Float32` become `Float64`. An `int32` NumPy array therefore arrives as an `Int64` column, and that is the type every later step and every result sees.

This is a deliberate trade. One width per kind means the interpreter, the JIT and the GPU translator share one set of kernels rather than one per width, which is what keeps them provably in agreement. {doc}`/user-guide/transform/columns/type-system` covers the full table and how to cast back on the way out.

## Tensor columns

A column of fixed-shape arrays is an Arrow `FixedSizeList`, which is what a decoded image, a resampled waveform, or an embedding is. It stays in Arrow through the whole plan, so a model reads it without a copy and without leaving the engine.

{doc}`/architecture/deep-dives/memory/tensor-columns` covers the layout and what it costs. {doc}`/api/accessors/media` covers the accessors that produce one.

## See also

- {doc}`index`: the other in-process libraries built on this contract.
- {doc}`/architecture/deep-dives/memory/arrow-memory`: buffers, validity bitmaps, and what the engine allocates.
- {doc}`/user-guide/transform/columns/type-system`: every type, and how casts behave.
- {doc}`/api/symbols/construction`: every constructor, with signatures.
