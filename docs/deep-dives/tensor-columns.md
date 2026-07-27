# Tensor columns

A relational engine's type system stops at scalars. An ML pipeline's does not: a row is an
image `(224, 224, 3)`, an embedding `(768,)`, a LiDAR frame `(4096, 3)`, a video clip
`(8, 224, 224, 3)`. Somewhere between the Parquet file and the model's forward pass, that
shape has to survive.

The tempting answer is a bespoke `TensorArray` type. Batcher does not have one, and the
absence is the design.

```text
   read.images()  ──►  schema {uri, bytes, size, mime}   (no pixels yet)
                            │
                            ▼
                    ┌────────────────┐
                    │  bytes column  │   a ~5 KB encoded JPEG per row
                    └───────┬────────┘
                            │
                            │  col("bytes").image.to_tensor(w, h)
                            │  a Rust expression, with a per-row rayon fan-out
                            ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  FixedSizeList<UInt8, w*h*3>                                 │
   │  + Arrow field metadata:                                     │
   │      ARROW:extension:name     = "arrow.fixed_shape_tensor"   │
   │      ARROW:extension:metadata = {"shape":[h,w,3]}            │
   └───────────────────────────────┬──────────────────────────────┘
                                   │
                                   │  the shape rides WITH the data, so it
                                   │  crosses the FFI boundary for free
                                   ▼
                    numpy (n, h, w, 3)   /   torch tensor
```

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
| 1 (an embedding) | plain `FixedSizeList<T, dim>` | `bt.from_numpy` on an `(n, dim)` array |
| ≥ 2 (image, clip, point cloud) | `arrow.fixed_shape_tensor` extension | native decode; a UDF returning `ndim >= 2` |

Both read back as `(n, *shape)` through the numpy and torch converters, so it is invisible
in practice, though the Arrow schema differs, and if you are inspecting `ds.schema()` you will
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

`python/batcher/ml/converters.py::_column_to_numpy` is the one place that knows about both
conventions:

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
zero-copy DLPack path is what `iter_torch_batches` uses for training ingest, where it streams
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

## Costs and limits

A tensor column is dense and fixed-shape. Every row pays the full `prod(shape) ×
sizeof(dtype)` bytes whether it needs them or not, and rows of *differing* shape cannot share
a column, so a corpus of mixed-resolution images has to be resized on decode. Which is why
`.image.to_tensor()` takes a width and height rather than inferring one.

The bytes are real, and they are what `execution.morsel_bytes` (1 MiB) exists for: a morsel is
split at whichever bound trips first, rows or bytes, so a column of 224×224×3 images produces
morsels of ~7 rows rather than 16,384. Without the byte bound, one morsel of images is 2.4 GB.

Video is the weak spot, and the two decode paths are worth seeing side by side.

::::{tab-set}
:::{tab-item} Native decode (image, audio, .npy)
```text
crates/bc-expr/src/eval/media/{image,audio}.rs

  a Rust expression in the plan
  interpreter-only (the JIT cannot compile a library-backed decode)
  per-row rayon fan-out above 8 rows
  emits the tensor field metadata directly
  stays on the fully-parallel native path
```
:::

:::{tab-item} Python decode (video)
```text
python/batcher/ml/decode.py::video_dataset

  a Python map_batches over PyAV
  builds the FixedSizeListArray by hand
  reinterprets it with as_tensor_column
  pays the map_batches throughput penalty above
```
:::
::::

## Code map

| Concern | File |
|---|---|
| The type helpers | `python/batcher/io/formats/ml/tensor.py` |
| Metadata preservation in projection | `crates/bc-interp/src/ops/project_field.rs` |
| Decode kernels | `crates/bc-expr/src/eval/media/{image,audio,video}.rs` |
| Decode orchestration | `python/batcher/ml/decode.py` |
| Arrow → numpy / torch | `python/batcher/ml/{converters,batch_format,loader}.py` |
| UDF output tensorization | `python/batcher/core/udf/call.py` |

## See also

:::{seealso}
- [Architecture](../architecture/index.md): why there is no Batcher tensor type
- [Execution engine](../internals/execution.md): where the decode expression is scheduled
- [Multimodal guide](../ml/multimodal.md): how to write these pipelines
- [ML guide](../ml/index.md): the loaders and converters on the other end
- [Multimodal ingest benchmarks](../benchmarks/multimodal-ingest.md): the img/s figures above
- [AI and GPU benchmarks](../benchmarks/ai-and-gpu.md): the training-ingest comparison
- [Arrow and memory](arrow-memory.md): what a `FixedSizeList` buffer actually is
- [GPU execution](gpu-execution.md): what consumes these tensors
- [Expression evaluation](expression-evaluation.md): where the decode kernels run
:::
