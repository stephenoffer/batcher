# Multimodal data

A multimodal pipeline turns references such as URLs and file paths into bytes, decodes
them into tensors, and feeds a model. Each step is a lazy operator that runs on whole
batches and parallelizes across the cluster.

## Fetch remote bytes

`ds.ml.download(url_column)` fetches the bytes at each URL or path into a binary
column, reading `s3://` / `gs://` / `az://` / `http(s)://` / local paths through the
shared filesystem resolver. Each batch's rows are fetched concurrently, and the stage
parallelizes across workers the way any operator does. `on_error="null"` turns a failed
fetch into a null so one bad URL does not fail the job.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/catalog.parquet")  # has a "url" column
with_bytes = ds.ml.download("url", output_column="bytes", max_concurrency=32)
```

`ds.ml.upload(data_column, directory)` is the counterpart. It writes a bytes column such
as decoded thumbnails or re-encoded media back to object storage, and appends the written
paths. Names come from a `name_column` or a content hash, and writes are concurrent and
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

Image and audio decode run natively in the engine. Audio goes through the pure-Rust
`symphonia` decoder, and an explicit `sample_rate` resamples natively too, with a sinc
resampler in the data plane. Only multi-channel output falls back to Python. Always
pass a `size`, or a batch of full-resolution frames can exhaust memory. Video decode uses
`PyAV` behind the `batcher-engine[video]` extra, one clip at a time, so a batch of large
clips never all co-resides. Keep `batch_size` small for multi-GB clips.

Decode is fast because it runs in the data plane, not a Python loop. Image decode uses
SIMD JPEG, including a DCT-scaled path for large frames feeding small model inputs, and
SIMD resize, fanned out per row across every core. The result crosses into a shaped
tensor column with no per-batch re-type step. On a 96-core node this makes image decode
and resize **2.4x faster than Daft and 6.1x faster than Ray Data**, and LiDAR
point-cloud loading **2.4x faster than Ray Data**. See
[Multimodal ingest benchmarks](../benchmarks/multimodal-ingest.md) and the reproducible
head-to-heads under `benchmarks/scenarios/`.

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
engine, with no per-row Python and no model needed, so it runs here on a handful of
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
# 2 extension<arrow.fixed_shape_tensor[value_type=uint8, shape=[8,8,3]]>
```

Each row is now a flat `8 * 8 * 3 = 192`-element RGB block. `bt.read.images(...,
decode=True, size=(h, w))` re-types that flat result into a fixed-shape `(h, w, 3)`
tensor column so the shape travels with the data, as the tensor-column section below
describes. The bare expression leaves it flat for when you reshape it yourself in a
downstream `map_batches`.

`.image.resize(width, height)` is the other half of the pair. It decodes, resizes, then
**re-encodes to PNG bytes**, so the column stays a compact `binary` blob instead of
becoming a tensor. Reach for `resize` when you want to shrink payloads before a
shuffle, a spill, or a write. Use `to_tensor` when the next stage is a model.

```python
resized = ds.with_columns(small=col("bytes").image.resize(4, 4)).collect()
thumbnail = resized.column("small")[0].as_py()
print(resized.schema.field("small").type, thumbnail[:4] == b"\x89PNG")
# binary True
```

Two more expressions stay in the bytes-to-bytes lane. `.image.crop(x, y, width, height)`
cuts a named window out of each image, which is what a detection pipeline does with a
bounding box. `.image.encode(format)` rewrites the container without touching the pixels,
for normalizing a mixed-format corpus onto one codec or trading a PNG for a smaller JPEG.

```python
cut = ds.with_columns(
    region=col("bytes").image.crop(0, 0, 4, 4),
    as_jpeg=col("bytes").image.encode("jpeg"),
)
print(cut.select(d=col("region").image.decode()).to_pydict())
# {'d': [{'width': 4, 'height': 4, 'channels': 4, 'mode': 'RGBA'}]}
```

`crop` clips at the edge rather than padding. `center_crop` pads, because it feeds a model
that needs a fixed input size; a cropped image is something a person or another tool will
look at, and inventing black pixels there would be inventing data. A window starting past
the image entirely yields null.

The audio counterpart is `.audio.to_waveform()`, which decodes an encoded clip and
averages its channels down to a single mono PCM signal. That is a `list<float>` per row,
the shape most audio models take as input. To feed a model that expects a fixed rate,
such as the 16 kHz Whisper and wav2vec want, `.audio.resample(16000)` decodes and
band-limit-resamples in the same native pass. `.video.decode()` is the video equivalent.

```python
import math
import struct
import wave

buf = io.BytesIO()
with wave.open(buf, "wb") as clip_writer:
    clip_writer.setnchannels(2)
    clip_writer.setsampwidth(2)
    clip_writer.setframerate(8000)
    samples = [int(3000 * math.sin(i / 10)) for i in range(16)]
    clip_writer.writeframes(b"".join(struct.pack("<hh", s, s) for s in samples))

signal = bt.from_pydict({"clip": [buf.getvalue()]}).with_columns(
    mono=col("clip").audio.to_waveform()
)
decoded_audio = signal.collect()
print(decoded_audio.schema.field("mono").type, len(decoded_audio.column("mono")[0].as_py()))
# list<item: float> 16
```

## Deduplicating images

Curating a scraped image corpus means dropping the same picture re-encoded, rescaled or
re-cropped. `.image.dhash()` is the primitive for it. It is a 64-bit *perceptual* hash
built from the gradients of a 9x8 grayscale thumbnail, so it survives re-encoding and
rescaling while still separating different pictures.

It returns a plain integer, so no new operator is needed. Exact-duplicate collapse is a
`distinct`, and near-duplicate matching is the Hamming distance you already have,
`bitwise_xor(...).bit_count()`, with a threshold of about 5 for "the same picture".

```python
# docs: skip
import batcher as bt
from batcher import col

photos = bt.read.images("s3://bucket/scrape/").with_columns(
    h=col("bytes").image.dhash()
)

# Exact duplicates: one row per distinct image.
unique = photos.distinct(subset=["h"])

# Near-duplicates against a reference set: a join plus a bit count.
pairs = (
    photos.select("uri", left=col("h"))
    .cross_join(reference.select(right=col("h")))
    .filter(col("left").bitwise_xor(col("right")).bit_count() <= 5)
)
```

A hash is null for an image that will not decode, so a corrupt file drops out of the
dedup rather than failing the pass.

## Keep large payloads out of shuffles and spills

A multi-GB payload such as a video, an audio file, or a PDF carried inline in a column is
copied through every sort and join and spill buffer it crosses, even when those
operators only touch other columns. `offload_blobs` writes each payload to a
content-addressed store and leaves a tiny URI handle in its place. `materialize_blobs`
reads it back right before you need the bytes. In between, only the short handle string
rides through the pipeline.

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

Offload is content-addressed with SHA-256, so identical payloads are written once and
deduped, and a re-read after a spill fetches the same bytes. The store defaults to
the configured spill location. That is `spill_remote_uri` when it is set, so handles are
reachable cluster-wide, and the local spill directory otherwise.

To place this automatically around a sort, set `auto_offload_blobs`. The engine then
offloads any `large_binary` column the sort does not key on and reads it back after,
with no plan changes on your side:

```python
# docs: skip
from batcher.config import Config, ExecutionConfig, config_context

with config_context(Config().replace(execution=ExecutionConfig(auto_offload_blobs=True))):
    ds.sort("id").collect()  # large_binary columns ride the sort as handles
```

It is off by default, because the round-trip to the store is a win only for genuinely
large payloads, which the `large_binary` type signals.

## Tensor columns

A column where every row is a same-shape `N`-dimensional tensor is stored as Arrow's
canonical fixed-shape-tensor type, so the shape travels with the data across the
engine boundary and converts to a correctly-shaped training tensor. `from_numpy` and
the NumPy reader build them for rows of rank 2 or higher:

```python
import batcher as bt
import numpy as np

imgs = np.zeros((4, 8, 8, 3), dtype=np.uint8)
ds = bt.from_numpy(imgs, column="image")
print(ds.collect().schema.field("image").type.shape)  # [8, 8, 3]
```

The `tensor_type`, `to_tensor_column`, `as_tensor_column`, and `is_tensor_column` helpers
in `batcher.io.formats.ml.tensor` build and classify these columns when you construct data
yourself.

When a tensor column reaches a `map_batches` model stage with `batch_format="numpy"`
or `"torch"`, the per-row tensors arrive **stacked** into one leading-batch array, a
`(batch, H, W, 3)` block, which is exactly the shape a vision model's forward pass
wants. No manual stacking or reshaping in the UDF.

A nested list column holds several small vectors per row, such as per-frame features or a
ragged batch of patches. `.list.flatten()` collapses it into one flat list per row,
removing a single level of nesting and keeping element order. Use it before a
downstream stage that wants one contiguous vector rather than a list of lists:

```python
import batcher as bt
from batcher import col

# Each row is a list of per-frame feature vectors; flatten to one vector per row.
frames = bt.from_pydict({"clip": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]})
flat = frames.select(vec=col("clip").list.flatten())
print(flat.to_pydict())
# {'vec': [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]]}
```

## From references to predictions

The steps compose into one lazy pipeline of fetch, decode, then a GPU model stage, where
preprocessing stays on CPU workers and only the model holds a GPU:

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

Passing the `Captioner` **class**, rather than an instance or a function, loads the model
once per GPU actor. A plain function would rebuild it on every batch. See
[GPU scheduling](gpu.md) for sizing the actor pool.

## Cleaning scraped text

Scraped pages arrive as markup, and so do product descriptions and email bodies.
`.str.strip_html()` recovers the prose. It drops tags along with the contents of
`<script>` and `<style>`, strips comments and decodes entities, then collapses
whitespace, separating block elements with a space.

```python
import batcher as bt

pages = bt.from_pydict({"page": ["<p>Tom &amp; Jerry</p><p>x</p><script>f()</script>"]})
print(pages.select(text=bt.col("page").str.strip_html()).to_pydict())
# {'text': ['Tom & Jerry x']}
```

Reach for this over the `regexp_replace('<[^>]*>', '')` idiom, which is wrong in three
ways that quietly poison a corpus. It leaves the JavaScript in `<script>` as prose, it
leaves `&amp;` and `&nbsp;` undecoded, and it welds `<p>a</p><p>b</p>` into `ab`.
`.str.strip_html()` is a text extractor, not an HTML parser, so malformed markup never
raises and one bad row in a web scrape cannot abort the scan.

## Chunking documents for RAG ingest

A document is usually longer than an embedding model's context, so the ingest chain is
**load, split, embed, index**. `.str.chunk(size, overlap)` is the split stage. It
slices text into fixed-size overlapping windows as a `List<Utf8>`, which `explode` turns
into one row per chunk. Sizes are in characters, and a chunk boundary never splits a
Unicode codepoint.

```python
import batcher as bt

docs = bt.from_pydict({"id": [1, 2], "body": ["abcdef", "xyz"]})
chunks = docs.with_columns(chunk=bt.col("body").str.chunk(4, overlap=1)).explode("chunk")
print(chunks.select("id", "chunk").to_pydict())
# {'id': [1, 1, 2], 'chunk': ['abcd', 'def', 'xyz']}
```

`overlap` carries context across a boundary, so a sentence cut in half still appears
whole in one chunk. Chunks stop once one reaches the end of the text, so the last chunk
is never a redundant suffix of its predecessor. From here, `ds.ml.embed(...)` produces
the vectors and the section below indexes them.

The whole chain of scan, chunk, explode, and embed is a linear row-wise pipeline, so it
distributes across workers and streams over an unbounded source with no breaker. The
one thing no static rule can know is how many chunks a document yields. Kyber estimates
a fan-out of 1 on the first run, Core measures the real fan-out, and the next plan sizes
the downstream GPU stage for it. See [adaptive re-optimization](../internals/kyber.md).

## Vector search over the embeddings

`ds.ml.embed` produces the vectors on a `Dataset`. Over a bare batch stream, such as
chunks coming out of a reader or a stage you are composing by hand, `batcher.ml.embed`
does the same work and takes an `EncoderFactory`. That is a zero-argument callable
returning an encoder, which is any callable mapping `list[str]` to one equal-length
vector per string. The factory is called once per worker, so the embedding model loads a
single time and every batch that worker handles reuses it. It has the same shape as the
`WorkerFactory` in [inference](inference.md), which is the reason a sentence-transformers
model, a local ONNX encoder, and a hosted embedding API are interchangeable here.

```python
# docs: skip
from sentence_transformers import SentenceTransformer

from batcher.ml import embed


def encoder_factory():  # an EncoderFactory — one model per worker
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    return lambda texts: model.encode(texts)


vectors = embed(chunks.iter_batches(), encoder_factory, text_column="chunk", num_workers=4)
```

The embedding is appended as `output_column`, which defaults to `"embedding"`, and the
batches come back in input order. Write them to Lance, then index and search.

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

Vector search needs `batcher-engine[lance]`. See [embeddings](embeddings.md) for the
compute side and [LLM inference](llm.md) for generation over retrieved context.

Sometimes the embeddings already ride in a column, as in a reranking pass or a small
candidate set that does not warrant an index. Score them against a query vector in-engine
with the `.list` distance expressions, and no Lance is required. `.list.cosine_distance(q)`
is `1 - cosine_similarity`, so it reads 0 for identical direction, 1 for orthogonal, and 2
for opposite. That is the standard embedding metric. `.list.l2_distance(q)` is the
Euclidean distance. Each takes the query as another column or an `array(...)` literal and
returns a Float64 that sorts ascending, so the nearest rows come first:

```python
import batcher as bt
from batcher import array, col

# Embeddings already in a column, and a query vector.
docs = bt.from_pydict({"id": [1, 2, 3], "vec": [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]})
query = array(1.0, 0.0)

ranked = docs.with_columns(dist=col("vec").list.cosine_distance(query)).sort("dist")
out = ranked.to_pydict()
print(out["id"], [round(d, 4) for d in out["dist"]])
# [1, 2, 3] [0.0, 0.2, 1.0]
```

A dot product is a cheaper kernel than a full cosine, and on **unit-length** vectors
the two rank identically. Cosine similarity is the dot product divided by both
magnitudes, and those are 1 once the vectors are normalized. So normalize once, up front
at embedding time, with `.list.normalize()`, which L2-normalizes each vector to unit
length, and retrieve with the plain `.list.dot(q)` against a likewise-normalized query.
`.list.l2_norm()` reports a vector's Euclidean magnitude, which confirms a vector is
already unit-length before you skip the normalization:

```python
import batcher as bt
from batcher import array, col

vecs = bt.from_pydict({"id": [1, 2, 3], "vec": [[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]]})

# Magnitudes before normalization ...
print(vecs.select(n=col("vec").list.l2_norm()).to_pydict())
# {'n': [5.0, 2.0, 1.0]}

# ... normalize to unit length, then a plain dot ranks like cosine similarity.
unit = vecs.with_columns(u=col("vec").list.normalize())
print(unit.select("id", score=col("u").list.dot(array(1.0, 0.0))).to_pydict())
# {'id': [1, 2, 3], 'score': [0.6, 0.0, 1.0]}
```

## Next steps

- [Inference](inference.md): run a model over the decoded tensors.
- [Preprocessors](preprocessors/index.md): assemble the decoded features into a training matrix.
- [Expressions API](../api/expressions.md): the `.image`/`.audio`/`.video` and vector
  `.list` method reference.
