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

Image and audio decode run natively in the engine (audio via the pure-Rust
`symphonia` decoder, for the common mono / source-rate case — an explicit
`sample_rate` resample falls back to `librosa`). Always pass a `size` so a batch of
full-resolution frames cannot exhaust memory. Video (`PyAV`, behind the
`batcher-engine[video]` extra) decodes one clip at a time so a batch of large clips
never all co-resides; keep `batch_size` small for multi-GB clips.

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

The `.image.to_tensor(width, height)` expression decodes and resizes natively in the
engine — no per-row Python and no model needed — so it runs here on a handful of
in-memory PNG bytes:

```python
import io

import numpy as np
from PIL import Image

import batcher as bt
from batcher import col

# Synthesize two tiny PNGs so the example needs no files.
raw = (np.random.default_rng(0).random((10, 12, 3)) * 255).astype("uint8")
buf = io.BytesIO()
Image.fromarray(raw).save(buf, format="PNG")
png = buf.getvalue()

ds = bt.from_pydict({"bytes": [png, png]})
decoded = ds.with_columns(image=col("bytes").image.to_tensor(8, 8)).collect()
print(decoded.num_rows, decoded.schema.field("image").type)
# 2 fixed_size_list<item: uint8 not null>[192]
```

Each row is now a flat `8 * 8 * 3 = 192`-element RGB block. `bt.read.images(...,
decode=True, size=(h, w))` re-types that flat result into a fixed-shape `(h, w, 3)`
tensor column so the shape travels with the data (see below); the bare expression
leaves it flat for when you reshape it yourself in a downstream `map_batches`.

## Blob-by-reference: keep large payloads out of shuffles and spills

A multi-GB payload (a video, audio file, or PDF) carried inline in a column is
copied through every sort, join, and spill buffer it crosses — even when those
operators only touch other columns. `offload_blobs` writes each payload to a
content-addressed store and leaves a tiny URI handle in its place; `materialize_blobs`
reads it back right before you need the bytes. In between, only the handle (a short
string) rides through the pipeline.

```python
import tempfile

import pyarrow as pa

import batcher as bt

ds = bt.from_arrow(pa.table({"id": [3, 1, 2], "payload": [b"c", b"a", b"b"]}))

# Offload -> sort by id (the payload rides as a handle) -> read the payload back.
out = (
    ds.offload_blobs("payload", root=tempfile.mkdtemp())
    .sort("id")
    .materialize_blobs(into="payload")
    .collect()
)
print(out.column("id").to_pylist(), out.column("payload").to_pylist())
# [1, 2, 3] [b'a', b'b', b'c']
```

Offload is content-addressed (SHA-256), so identical payloads are written once and
deduped, and a re-read after a spill fetches the same bytes. The store defaults to
the configured spill location (`spill_remote_uri` if set, so handles are reachable
cluster-wide, else the local spill dir).

To place this automatically around a sort, set `auto_offload_blobs` — the engine
then offloads any `large_binary` column the sort does not key on, and reads it back
after, with no plan changes on your side:

```python
# docs: skip
from batcher.config import Config, ExecutionConfig, config_context

with config_context(Config().replace(execution=ExecutionConfig(auto_offload_blobs=True))):
    ds.sort("id").collect()  # large_binary columns ride the sort as handles
```

It is off by default — the round-trip to the store is a win only for genuinely large
payloads, which the `large_binary` type signals.

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

When a tensor column reaches a `map_batches` model stage with `batch_format="numpy"`
or `"torch"`, the per-row tensors arrive **stacked** into one leading-batch array — a
`(batch, H, W, 3)` block — which is exactly the shape a vision model's forward pass
wants. No manual stacking or reshaping in the UDF.

## End to end: references to predictions

The steps compose into one lazy pipeline — fetch, decode, then a GPU model stage —
where preprocessing stays on CPU workers and only the model holds a GPU:

```python
# docs: skip
import batcher as bt
import pyarrow as pa


class Captioner:
    def __init__(self):
        import torch
        from transformers import pipeline

        self.pipe = pipeline("image-to-text", model="...", device="cuda")
        self._torch = torch

    def __call__(self, batch):
        # The "image" tensor column arrives as one (batch, 224, 224, 3) array.
        images = batch.column("image").to_numpy()
        with self._torch.no_grad():
            captions = [self.pipe(img)[0]["generated_text"] for img in images]
        return batch.append_column("caption", pa.array(captions))


catalog = bt.read.parquet("s3://bucket/catalog.parquet")  # has a "url" column
captioned = (
    catalog.ml.download("url", output_column="bytes")  # CPU: fetch
    .with_columns(image=bt.col("bytes").image.to_tensor(224, 224))  # engine: decode
    .ml.map_batches(Captioner, batch_size=64, num_gpus=1, concurrency=2)  # GPU: model
)
captioned.write.parquet("s3://bucket/captioned.parquet")
```

Passing the `Captioner` **class** (not an instance or a function) loads the model
once per GPU actor; a plain function would rebuild it on every batch. See
[GPU scheduling](gpu.md) for sizing the actor pool.

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
