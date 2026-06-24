# Multimodal data

A multimodal pipeline turns references (URLs, file paths) into bytes, decodes them
into tensors, and feeds a model. Each step is a lazy operator that runs on whole
batches and parallelizes across the cluster.

## Fetch remote bytes

`ds.ml.download(url_column)` fetches the bytes at each URL or path into a binary
column, reading `s3://` / `gs://` / `az://` / `http(s)://` / local paths through the
shared filesystem resolver. Each batch's rows are fetched concurrently, and the stage
parallelizes across workers like any operator. `on_error="null"` turns a failed fetch
into a null so one bad URL does not fail the job.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/catalog.parquet")  # has a "url" column
with_bytes = ds.ml.download("url", output_column="bytes", max_concurrency=32)
```

`ds.ml.upload(data_column, directory)` is the counterpart — write a bytes column back
to object storage (decoded thumbnails, re-encoded media), appending the written paths.
Names come from a `name_column` or a content hash, and writes are concurrent and
atomic.

```python
# docs: skip
written = with_bytes.ml.upload("thumbnail", "s3://bucket/thumbs/", extension=".jpg")
```

## Decode images, audio, video

Multimodal readers list files and expose header metadata without decoding pixels.
Pass `decode=True` to append decoded tensors:

```python
# docs: skip
import batcher as bt

# Image bytes -> a (224, 224, 3) uint8 tensor column, decoded/resized in the engine.
images = bt.read.images("s3://bucket/images/", decode=True, size=(224, 224))

# Audio -> a list<float32> waveform column; video -> sampled (N, H, W, 3) frames.
audio = bt.read.audio("data/clips/", decode=True, sample_rate=16000)
video = bt.read.video("data/videos/", decode=True, size=(112, 112), num_frames=8)
```

Image decode runs natively in the engine. Always pass a `size` so a batch of
full-resolution frames cannot exhaust memory. Audio (`soundfile`) and video (`PyAV`)
decode in Python UDFs behind the `batcher-engine[audio]` / `[video]` extras.

You can also decode inside a pipeline with the `.image` expression after a download:

```python
# docs: skip
import batcher as bt
from batcher import col

ds = bt.read.parquet("s3://bucket/catalog.parquet")
tensors = (
    ds.ml.download("url", output_column="bytes")
    .with_columns(image=col("bytes").image.to_tensor(224, 224))
)
```

## Tensor columns

A column where every row is a same-shape `N`-dimensional tensor is stored as Arrow's
canonical fixed-shape-tensor type, so the shape travels with the data across the
engine boundary and converts to a correctly-shaped training tensor. `from_numpy` and
the NumPy reader build them for rank-≥2 rows:

```python
import batcher as bt
import numpy as np

imgs = np.zeros((4, 8, 8, 3), dtype=np.uint8)
ds = bt.from_numpy(imgs, column="image")
print(ds.collect().schema.field("image").type.shape)  # [8, 8, 3]
```

The helpers in `batcher.io.formats.ml.tensor` (`tensor_type`, `to_tensor_column`,
`as_tensor_column`, `is_tensor_column`) build and classify these columns when you
construct data yourself.

## Vector search (RAG retrieval)

After embedding text or images and writing them to a Lance dataset, retrieve the
nearest rows to a query vector with `vector_search`, optionally building an ANN index
first so it scales:

```python
# docs: skip
from batcher.ml import vector_search, build_vector_index

build_vector_index("s3://bucket/vectors.lance", "embedding")
hits = vector_search("s3://bucket/vectors.lance", query_vector, column="embedding", k=10)
top = hits.collect()  # k rows nearest to the query, with a _distance column
```

Vector search needs `batcher-engine[lance]`. See [embeddings](inference.md) for the
compute side and [LLM inference](llm.md) for generation over retrieved context.
