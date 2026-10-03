# Tensor columns

A relational engine's type system stops at scalars. An ML pipeline's does not: a row is an image `(224, 224, 3)`, an embedding `(768,)`, a LiDAR frame `(4096, 3)`, a video clip `(8, 224, 224, 3)`. Between the Parquet file and the model's forward pass, that shape has to survive.

Batcher has no bespoke `TensorArray` type, and the absence is the design. A tensor column is one contiguous Arrow buffer with its shape in the field metadata, against one Python object per row.

![A batch of images held as one Python object per row, against the same batch held as one Arrow column. Per-row objects are n separate allocations and every row is a pointer, so the batch's bytes are scattered and nothing can be handed to a kernel or across the FFI boundary whole; that is what RecordBatch.to_pandas() gives back for a tensor column, which is why the batch is re-wrapped before a user function ever sees it. The Arrow column is a validity bitmap of one bit per row plus a values buffer of n by 150,528 uint8, contiguous. It needs no offsets buffer, because every row is the same size and row i begins at byte i times 150,528, and it is typed as arrow.fixed_shape_tensor over a FixedSizeList with the shape (224, 224, 3) in the field metadata. A reader gets it back as to_numpy_ndarray() shaped (n, 224, 224, 3), the shape taken from the type, or as a DLPack view over that same buffer through arrays_to_torch(zero_copy=True). Because the shape travels with the data, the column crosses the FFI boundary and comes back shaped with no IR tag and no two-sided contract, which is what choosing the canonical type buys. The default torch path owns a writable copy instead: a training loop mutates its batch, and the Arrow buffer is read-only.](/_static/diagrams/tensor_column_layout.svg)

## The canonical Arrow extension type

A tensor column is a `FixedSizeList` of the value type, carrying its shape in Arrow field metadata under the canonical `arrow.fixed_shape_tensor` extension name. That is a pyarrow type, not a Batcher type: there is no IR tag, no serde enum and no two-sided wire contract. The C Data Interface carries field metadata, so a shaped column crosses the FFI boundary with no conversion and no copy.

Two conventions coexist:

| Per-row rank | Arrow type | Produced by |
|---|---|---|
| 1 (an embedding) | plain `FixedSizeList<T, dim>` | {py:func}`bt.from_numpy <batcher.from_numpy>` on an `(n, dim)` array |
| 2 or more (image, clip, point cloud) | `arrow.fixed_shape_tensor` extension | native decode; a UDF returning `ndim >= 2` |

```python
import numpy as np
import batcher as bt

# rank-1 per row: a plain fixed-size list
emb = bt.from_numpy(np.arange(12, dtype=np.float32).reshape(4, 3), column="emb")
print(emb.collect().schema.field("emb").type)


# rank-3 per row: the canonical extension type
def make_images(batch):
    return {"img": np.zeros((batch.num_rows, 2, 2, 3), dtype=np.uint8)}


imgs = bt.from_pydict({"i": [0, 1, 2, 3]}).map_batches(make_images)
print(imgs.collect().schema.field("img").type)
```

```text
fixed_size_list<item: float>[3]
extension<arrow.fixed_shape_tensor[value_type=uint8, shape=[2,2,3], permutation=[0,1,2]]>
```

Both read back as `(n, *shape)` through the numpy and torch converters. A UDF can return an `ndarray` with `ndim >= 2` directly, because `core/udf/call.py::_tensorize_columns` turns it into a tensor column, which is what makes a decode-then-model pipeline expressible.

:::{dropdown} The Python type helpers
```python
# docs: skip
# python/batcher/io/formats/ml/tensor.py
def tensor_type(value_type: pa.DataType, shape: tuple[int, ...]) -> pa.DataType:
    return pa.fixed_shape_tensor(value_type, list(shape))


def to_tensor_column(ndarray: np.ndarray) -> pa.Array:
    return pa.FixedShapeTensorArray.from_numpy_ndarray(ndarray)  # leading axis = rows
```
:::

## Rust never sees a tensor type

There is no `FixedShapeTensor` in the `bc-*` crates, only a `FixedSizeListArray` and field metadata that the kernels must not drop. In [`crates/bc-interp/src/ops/project_field.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/project_field.rs), a bare `Expr::Col` passthrough **clones the source field**, metadata and all. Rebuilding the field from `array.data_type()` would downgrade a tensor column to its plain storage type.

A native decode emits the metadata directly, through one helper shared by the image (3-D) and video (4-D) paths:

```rust
// crates/bc-interp/src/ops/project_field.rs
fn tensor_field(alias: &str, dtype: DataType, shape: &[i64]) -> Field {
    // ARROW:extension:name     = "arrow.fixed_shape_tensor"
    // ARROW:extension:metadata = {"shape":[h,w,3]}  or  {"shape":[n,h,w,3]}
}
```

Emitting the shape from Rust keeps decode on the fully-parallel native path with no Python re-typing step.

## Bytes at read time, tensors downstream

`read.images()`, `read.audio()` and `read.video()` produce a `bytes` column, not pixels, with `uri`, `bytes`, `size` and `mime` plus cheap header-derived columns such as an audio file's `sample_rate`. Decoding is a downstream Rust *expression*:

```text
col("bytes").image.to_tensor(width, height)   -> FixedSizeList<UInt8> + tensor metadata
col("bytes").image.decode()                   -> struct {width, height, channels, mode}  (header only)
col("bytes").audio.to_waveform()              -> list<float32>  (variable length: NOT a tensor)
col("bytes").video.frames(n, width, height)   -> FixedSizeList<UInt8> + tensor metadata [n,h,w,3]
```

Audio waveforms are variable-length lists, because clip lengths vary. The decode kernels in [`crates/bc-expr/src/eval/media/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-expr/src/eval/media) are interpreter-only and fan out per *row* over rayon above 8 rows (`PAR_ROW_THRESHOLD`), and `Expr::contains_media_decode()` lifts the thread pool to every core for a media plan. That made decode alone 17x to 22x faster on a 2,000-JPEG corpus that fits in one morsel.

:::::{dropdown} Native decode and the Python video fallback
Clip decode links the system FFmpeg, so it sits behind the optional `video` cargo feature, and `ml/decode/video.py::video_dataset` checks `engine_features()` to pick a path.

::::{tab-set}
:::{tab-item} Native decode
```text
crates/bc-expr/src/eval/media/{image/, audio.rs, video/}

  a Rust expression in the plan
  interpreter-only (the JIT cannot compile a library-backed decode)
  per-row rayon fan-out above 8 rows
  emits the tensor field metadata directly
  stays on the fully-parallel native path
  video: only in an engine built with the `video` feature
```
:::

:::{tab-item} Python fallback (video without the feature)
```text
python/batcher/ml/decode/video.py::video_dataset

  a Python map_batches over PyAV
  builds the FixedSizeListArray by hand
  reinterprets it with as_tensor_column
```
:::
::::
:::::

## Out to numpy and torch

`python/batcher/interop/arrays.py::_column_to_numpy` is the one place that knows both conventions, returning `(n, *shape)` for a tensor column and `(n, W)` for a fixed-size list. `arrays_to_torch` converts numeric columns only. By default it makes a **writable copy**, because Arrow buffers are read-only and a training loop mutates its batch. `zero_copy=True` opts into `torch.from_dlpack`, and {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` takes the same flag. The training-ingest benchmark streams 1.76 M rows/s through `iter_torch_batches` on 10M rows by 32 features.

```python
for batch in imgs.ml.iter_torch_batches(batch_size=2):
    print({k: (tuple(v.shape), str(v.dtype)) for k, v in batch.items()})
    break
```

```text
{'img': ((2, 2, 2, 3), 'torch.uint8')}
```

## Rows with different shapes

The canonical type carries one shape per column, so a folder of mixed-resolution photos has no fixed-shape form. Those columns use a second representation, in [`python/batcher/io/formats/ml/ragged.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/io/formats/ml/ragged.py): each row is its own row-major buffer beside its shape and dtype. It's a plain struct, so it crosses the FFI, writes to Parquet and shuffles with no engine change, and the `binary` buffer carries every dtype at its native width. Nothing asks for it; arrays of differing shape produce one automatically:

```python
ragged = bt.from_pydict({"img": [np.zeros((2, 2), "uint8"), np.ones((3, 4), "uint8")]})
print(ragged.collect().schema.field("img").type)
# struct<data: binary, shape: list<item: int32>, dtype: string>
print([a.shape for a in ragged.to_numpy()["img"]])  # [(2, 2), (3, 4)]
```

Rows of differing shape have no stacked form, so the torch bridge hands back per-row arrays rather than silently padding. A model that needs one tensor still needs a resize in the pipeline.

## Practical limits

- **Density.** A fixed-shape column pays `prod(shape) * sizeof(dtype)` per row, which is why {py:meth}`.image.to_tensor() <batcher.plan.expr_ir.image._ImageNamespace.to_tensor>` takes a width and height. The ragged form stores the same bytes plus a shape and dtype per row.
- **Morsel size.** `execution.morsel_bytes` (1 MiB) splits a column of 224x224x3 images into morsels of ~7 rows rather than 16,384.
- **Polars batch format.** Polars has no per-row tensor dtype, so a `(3, 3)` image arrives as a 9-element list. Use `batch_format="numpy"` or `"pandas"` when the function needs the shape.

## Code map

| Concern | File |
|---|---|
| The type helpers | [`python/batcher/io/formats/ml/tensor.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/io/formats/ml/tensor.py) |
| The variable-shape representation | [`python/batcher/io/formats/ml/ragged.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/io/formats/ml/ragged.py) |
| Metadata preservation in projection | [`crates/bc-interp/src/ops/project_field.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/ops/project_field.rs) |
| Decode kernels, including FFmpeg video | [`crates/bc-expr/src/eval/media/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-expr/src/eval/media) |
| Decode orchestration | [`python/batcher/ml/decode/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/ml/decode) |
| Arrow to numpy and torch | [`python/batcher/interop/arrays.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/interop/arrays.py) (re-exported by `ml/converters.py`), `loader/` |
| UDF output tensorization | [`python/batcher/core/udf/call.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/udf/call.py) |

## See also

- {doc}`Architecture </architecture/index>`: why there is no Batcher tensor type.
- {doc}`Execution engine </architecture/internals/execution>`: where the decode expression is scheduled.
- {doc}`Multimodal guide </ml/preparing/multimodal/index>`: how to write these pipelines.
- {doc}`ML guide </ml/index>`: the loaders and converters on the other end.
- {doc}`Multimodal ingest benchmarks </benchmarks/results/multimodal-ingest>`: image ingest throughput.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: the training-ingest comparison.
- {doc}`Arrow and memory </architecture/deep-dives/memory/arrow-memory>`: what a `FixedSizeList` buffer actually is.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: what consumes these tensors.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: where the decode kernels run.
