# Tensor columns

A relational engine's type system stops at scalars. An ML pipeline's does not: a row is an
image `(224, 224, 3)`, an embedding `(768,)`, a LiDAR frame `(4096, 3)`, a video clip
`(8, 224, 224, 3)`. Somewhere between the Parquet file and the model's forward pass, that
shape has to survive.

The tempting answer is a bespoke `TensorArray` type. Batcher does not have one, and the
absence is the design.

What that absence buys is visible in the bytes: one contiguous Arrow buffer, against one
Python object per row.

![A batch of images held as one Python object per row, against the same batch held as one Arrow column. Per-row objects are n separate allocations and every row is a pointer, so the batch's bytes are scattered and nothing can be handed to a kernel or across the FFI boundary whole; that is what RecordBatch.to_pandas() gives back for a tensor column, which is why the batch is re-wrapped before a user function ever sees it. The Arrow column is a validity bitmap of one bit per row plus a values buffer of n by 150,528 uint8, contiguous. It needs no offsets buffer, because every row is the same size and row i begins at byte i times 150,528, and it is typed as arrow.fixed_shape_tensor over a FixedSizeList with the shape (224, 224, 3) in the field metadata. A reader gets it back as to_numpy_ndarray() shaped (n, 224, 224, 3), the shape taken from the type, or as a DLPack view over that same buffer through arrays_to_torch(zero_copy=True). Because the shape travels with the data, the column crosses the FFI boundary and comes back shaped with no IR tag and no two-sided contract, which is what choosing the canonical type buys. The default torch path owns a writable copy instead: a training loop mutates its batch, and the Arrow buffer is read-only.](/_static/diagrams/tensor_column_layout.svg)

## The canonical Arrow extension type

A tensor column is a `FixedSizeList` of the value type, carrying its shape in Arrow *field
metadata* under the canonical `arrow.fixed_shape_tensor` extension name. That is a pyarrow
type, not a Batcher type. There is no IR tag for it, no serde enum, no two-sided wire
contract to keep in lockstep.

```python
# docs: skip
# python/batcher/io/formats/ml/tensor.py
def tensor_type(value_type: pa.DataType, shape: tuple[int, ...]) -> pa.DataType:
    return pa.fixed_shape_tensor(value_type, list(shape))

def to_tensor_column(ndarray: np.ndarray) -> pa.Array:
    return pa.FixedShapeTensorArray.from_numpy_ndarray(ndarray)   # leading axis = rows
```

That whole module is 98 lines. The shape rides with the data, which means it crosses the FFI
boundary for free. The C Data Interface carries field metadata, so a shaped column
reconstructs on the pyarrow side with no conversion and no copy.

Two conventions coexist, and it is worth knowing which you have:

| Per-row rank | Arrow type | Produced by |
|---|---|---|
| 1 (an embedding) | plain `FixedSizeList<T, dim>` | {py:func}`bt.from_numpy <batcher.from_numpy>` on an `(n, dim)` array |
| ≥ 2 (image, clip, point cloud) | `arrow.fixed_shape_tensor` extension | native decode; a UDF returning `ndim >= 2` |

Both read back as `(n, *shape)` through the numpy and torch converters, so it is invisible
in practice, though the Arrow schema differs, and if you are inspecting {py:obj}`ds.schema() <batcher.Dataset.schema>` you will
see it.

```python
import numpy as np
import pyarrow as pa
import batcher as bt

# rank-1 per row: a plain fixed-size list
emb = bt.from_numpy(np.arange(12, dtype=np.float32).reshape(4, 3), column="emb")
print(emb.collect().schema.field("emb").type)

# rank-3 per row: the canonical extension type
def make_images(batch):
    return {"img": np.zeros((batch.num_rows, 2, 2, 3), dtype=np.uint8)}

imgs = bt.from_pydict({"i": [0, 1, 2, 3]}).map_batches(make_images)
field = imgs.collect().schema.field("img")
print(field.type)
```

```text
fixed_size_list<item: float>[3]
extension<arrow.fixed_shape_tensor[value_type=uint8, shape=[2,2,3], permutation=[0,1,2]]>
```

## Rust never sees a tensor type

There is no `FixedShapeTensor` in the `bc-*` crates. There is a `FixedSizeListArray` and
there is field metadata, and the kernels are careful not to drop the latter.

:::{important}
The kernels must not drop the field metadata. `crates/bc-interp/src/ops/project_field.rs` is the
whole story: a bare `Expr::Col` passthrough **clones the source field**, metadata and all.
Rebuilding the field from `array.data_type()` instead would silently downgrade a tensor column
to its plain storage type, and the shape would be gone by the time anything noticed.
:::

```rust
// A bare Expr::Col passthrough CLONES the source field, preserving its metadata,
// notably the Arrow extension type. Rebuilding from array.data_type() would drop it,
// downgrading a tensor column to its plain storage type.
```

And a native image decode *emits* the metadata directly:

```rust
fn tensor_field(alias: &str, dtype: DataType, h: u32, w: u32) -> Field {
    // ARROW:extension:name     = "arrow.fixed_shape_tensor"
    // ARROW:extension:metadata = {"shape":[h,w,3]}
}
```

:::{warning}
*Any* `map_batches`, even an identity one, roughly halves throughput and core utilization by
pulling the pipeline off the fully-parallel native path. Before the metadata was emitted from
Rust, `read.images(decode=True)` appended a Python `map_batches` whose only job was to re-type
the flat list as a shaped tensor. Removing it took image ingest from 2,000 to 4,600 img/s, and
the point-cloud path inherited the win for free.
:::

## Bytes at read time, tensors downstream

`read.images()`, `read.audio()`, and `read.video()` produce a `bytes` column, not pixels. Each
lists media files and yields `uri`, `bytes`, `size`, and `mime`, plus whatever cheap header-derived
columns that medium offers, such as `sample_rate`, `channels`, and `duration` for audio. Decoding
is a downstream Rust *expression*, never a read-time side effect:

```text
col("bytes").image.to_tensor(width, height)   -> FixedSizeList<UInt8> + tensor metadata
col("bytes").image.decode()                   -> struct {width, height, channels, mode}  (header only)
col("bytes").audio.to_waveform()              -> list<float32>  (variable length: NOT a tensor)
```

:::{note}
Audio waveforms are deliberately *not* fixed-shape tensors. Clip lengths vary, so the type is a
variable-length list and there is no shape to carry.
:::

The decode kernels live in `crates/bc-expr/src/eval/media/`. They are interpreter-only (the
JIT cannot compile a library-backed decode), and they fan out per *row* over rayon above a
threshold of 8 rows. That per-row fan-out exists because a 2,000-JPEG corpus is a single
morsel, and the parallel executor capped its thread pool at the morsel count, so the entire decode
ran on one core. `Expr::contains_media_decode()` lifts the pool to every core for a media plan,
which made decode alone 17x to 22x faster.

## Out to numpy and torch

`python/batcher/interop/arrays.py::_column_to_numpy` is the one place that knows about both
conventions. `ml/converters.py` re-exports it, deliberately rather than by accident, because
`ml.serving` and the tests reach for the old path:

```python
# docs: skip
if is_tensor_column(arr):
    return arr.to_numpy_ndarray()              # (n, *shape)
if fixed_size_list_of_primitives(arr):
    return child.reshape(-1, width)            # (n, W)
return arr.to_numpy(zero_copy_only=False)
```

`arrays_to_torch` handles numeric columns only; string columns are dropped rather than
silently mangled. By default it makes a **writable copy**, because Arrow buffers are
read-only and handing one to torch is undefined behavior the moment anything writes in place.
`zero_copy=True` opts into `torch.from_dlpack`, with a copy as fallback.

That default is a real cost, and it is the honest kind: correctness first. The
zero-copy DLPack path is what {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` uses for training ingest, where it streams
1.76 M rows/s on 10M rows by 32 features, well above what most training loops consume.

```python
import numpy as np
import batcher as bt

def make_images(batch):
    return {"img": np.zeros((batch.num_rows, 2, 2, 3), dtype=np.uint8)}

ds = bt.from_pydict({"i": [0, 1, 2, 3]}).map_batches(make_images)
for batch in ds.ml.iter_torch_batches(batch_size=2):
    print({k: (tuple(v.shape), str(v.dtype)) for k, v in batch.items()})
    break
```

```text
{'img': ((2, 2, 2, 3), 'torch.uint8')}
```

## Coming back from a UDF

`pa.RecordBatch.from_pydict` cannot build a column from a multi-dimensional numpy array. So
`core/udf/call.py::_tensorize_columns` intercepts any returned `ndarray` with `ndim >= 2`
and runs it through `to_tensor_column`.

That interception is what makes a two-stage decode-then-model pipeline expressible at all. A UDF
returning `{"emb": (B, 2048) float32}` round-trips zero-copy through the FFI, out to numpy and
torch, where without it the batch construction fails outright.

## When the rows have different shapes

The canonical type carries one shape for a whole column, so a corpus of mixed-resolution
images has no fixed-shape form at all. That is the ordinary result of decoding a folder of
photos, and it used to be where a multimodal pipeline stopped: the only advice was to resize
before the engine saw the data.

Those columns are carried by a second representation, in
`python/batcher/io/formats/ml/ragged.py`. Each row is stored as its own row-major buffer
beside its own shape and dtype:

```text
struct<data: binary, shape: list<int32>, dtype: string>
```

Both halves of that layout are doing work. It is a **plain struct**, so it crosses the FFI,
writes to Parquet, shuffles, and passes through every operator with no engine change, no IR
tag, and no wire-contract change. An extension type would have had to be taught to the Rust
side. And `data` is a **binary buffer rather than a list of elements**, because the
boundary widens narrow numerics: a `list<uint8>` image column arrives as `list<int64>`, eight
bytes per pixel, for the one workload the representation exists to carry.

Nothing asks for it. A `map_batches` returning arrays of differing shape, or a `from_pydict`
given them, produces one automatically, and `to_numpy` and `batch_format="numpy"` decode it
back to per-row arrays:

```python
# docs: skip
import numpy as np

ds = bt.from_pydict({"img": [np.zeros((2, 2), "uint8"), np.ones((3, 4), "uint8")]})
print([a.shape for a in ds.to_numpy()["img"]])
# [(2, 2), (3, 4)]
```

What it does not do is present itself to torch as a tensor. Rows of differing shape have no
stacked form, so the bridges hand back per-row arrays rather than silently padding. A model
that needs one tensor still needs a resize, and the resize is now a choice made in the
pipeline rather than a precondition for entering it.

## Costs and limits

A fixed-shape tensor column is dense. Every row pays the full `prod(shape) × sizeof(dtype)`
bytes whether it needs them or not, which is why
{py:meth}`.image.to_tensor() <batcher.plan.expr_ir.image._ImageNamespace.to_tensor>` takes a width and height rather than inferring one.
The variable-shape form above lifts the shape restriction but not the density: it stores the
same bytes, plus a shape and a dtype string per row.

One `batch_format` cannot carry the shape. `pyarrow`, `numpy`, and `pandas` all hand a
function per-row arrays with their real shape, and rebuild the tensor column from what comes
back. Polars has no dtype for a per-row tensor, so it reads the canonical extension type as
its flat storage and a `(3, 3)` image arrives as a 9-element list. The values are unharmed;
the shape is not. Use `numpy` or `pandas` when the function needs it.

The bytes are real, and they are what `execution.morsel_bytes` (1 MiB) exists for: a morsel is
split at whichever bound trips first, rows or bytes, so a column of 224×224×3 images produces
morsels of ~7 rows rather than 16,384. Without the byte bound, one morsel of images is 2.4 GB.

Video is the weak spot, and the two decode paths are worth seeing side by side.

::::{tab-set}
:::{tab-item} Native decode (image, audio, .npy)
```text
crates/bc-expr/src/eval/media/{image/, audio.rs}

  a Rust expression in the plan
  interpreter-only (the JIT cannot compile a library-backed decode)
  per-row rayon fan-out above 8 rows
  emits the tensor field metadata directly
  stays on the fully-parallel native path
```
:::

:::{tab-item} Python decode (video)
```text
python/batcher/ml/decode/video.py::video_dataset

  a Python map_batches over PyAV
  builds the FixedSizeListArray by hand
  reinterprets it with as_tensor_column
  pays the map_batches throughput penalty above
```
:::
::::

## Code map

Each concern below maps to one file, covering how a tensor column is declared, stored, and
handed to a model:

| Concern | File |
|---|---|
| The type helpers | `python/batcher/io/formats/ml/tensor.py` |
| The variable-shape representation | `python/batcher/io/formats/ml/ragged.py` |
| Metadata preservation in projection | `crates/bc-interp/src/ops/project_field.rs` |
| Decode kernels | `crates/bc-expr/src/eval/media/` |
| Decode orchestration | `python/batcher/ml/decode/` |
| Arrow → numpy / torch | `python/batcher/interop/arrays.py` (re-exported by `ml/converters.py`), `loader/` |
| UDF output tensorization | `python/batcher/core/udf/call.py` |

## See also

- {doc}`Architecture </architecture/index>`: why there is no Batcher tensor type.
- {doc}`Execution engine </architecture/internals/execution>`: where the decode expression is scheduled.
- {doc}`Multimodal guide </ml/preparing/multimodal/index>`: how to write these pipelines.
- {doc}`ML guide </ml/index>`: the loaders and converters on the other end.
- {doc}`Multimodal ingest benchmarks </benchmarks/results/multimodal-ingest>`: the img/s figures above.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: the training-ingest comparison.
- {doc}`Arrow and memory </architecture/deep-dives/memory/arrow-memory>`: what a `FixedSizeList` buffer actually is.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: what consumes these tensors.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: where the decode kernels run.
