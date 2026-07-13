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

`ds.ml.upload(data_column, directory)` is the counterpart. It writes a bytes column back
to object storage (decoded thumbnails, re-encoded media) and appends the written paths.
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

Image and audio decode run natively in the engine: audio through the pure-Rust
`symphonia` decoder, and an explicit `sample_rate` resamples natively too (a sinc
resampler in the data plane; only multi-channel output falls back to Python). Always
pass a `size`, or a batch of full-resolution frames can exhaust memory. Video (`PyAV`,
behind the `batcher-engine[video]` extra) decodes one clip at a time so a batch of large
clips never all co-resides; keep `batch_size` small for multi-GB clips.

Decode is fast because it runs in the data plane, not a Python loop. Image decode uses
SIMD JPEG (with a DCT-scaled path for large frames → small model inputs) and SIMD resize,
fanned out per row across every core, and the result crosses into a shaped tensor column
with no per-batch re-type step. On a 96-core node this makes image decode+resize **~2.4×
faster than Daft and ~6× than Ray Data**, and LiDAR/point-cloud loading **~2.4× faster
than Ray Data**. See [Multimodal & physical-AI ingest](../user-guide/performance.md) and
the reproducible `benchmarks/scenarios/` head-to-heads.

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
# 2 fixed_size_list<item: uint8 not null>[192]
```

Each row is now a flat `8 * 8 * 3 = 192`-element RGB block. `bt.read.images(...,
decode=True, size=(h, w))` re-types that flat result into a fixed-shape `(h, w, 3)`
tensor column so the shape travels with the data (see below); the bare expression
leaves it flat for when you reshape it yourself in a downstream `map_batches`.

`.image.resize(width, height)` is the other half of the pair. It decodes, resizes, then
**re-encodes to PNG bytes**, so the column stays a compact `binary` blob instead of
becoming a tensor. Reach for `resize` when you want to shrink payloads before a
shuffle, a spill, or a write; use `to_tensor` when the next stage is a model.

```python
resized = ds.with_columns(small=col("bytes").image.resize(4, 4)).collect()
thumbnail = resized.column("small")[0].as_py()
print(resized.schema.field("small").type, thumbnail[:4] == b"\x89PNG")
# binary True
```

The audio counterpart is `.audio.to_waveform()`, which decodes an encoded clip and
averages its channels down to a single mono PCM signal: a `list<float>` per row, the
shape most audio models take as input. To feed a model that expects a fixed rate (16 kHz
for Whisper/wav2vec), `.audio.resample(16000)` decodes and band-limited-resamples in the
same native pass. (`.video.decode()` is the video equivalent.)

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

## Blob-by-reference: keep large payloads out of shuffles and spills

A multi-GB payload (a video, an audio file, a PDF) carried inline in a column is
copied through every sort and join and spill buffer it crosses, even when those
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

To place this automatically around a sort, set `auto_offload_blobs`. The engine then
offloads any `large_binary` column the sort does not key on and reads it back after,
with no plan changes on your side:

```python
# docs: skip
from batcher.config import Config, ExecutionConfig, config_context

with config_context(Config().replace(execution=ExecutionConfig(auto_offload_blobs=True))):
    ds.sort("id").collect()  # large_binary columns ride the sort as handles
```

It is off by default: the round-trip to the store is a win only for genuinely large
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
or `"torch"`, the per-row tensors arrive **stacked** into one leading-batch array, a
`(batch, H, W, 3)` block, which is exactly the shape a vision model's forward pass
wants. No manual stacking or reshaping in the UDF.

A nested list column (a row holding several small vectors, say per-frame features or a
ragged batch of patches) collapses into one flat list per row with `.list.flatten()`,
which removes a single level of nesting and keeps element order. Use it before a
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

## End to end: references to predictions

The steps compose into one lazy pipeline (fetch, decode, then a GPU model stage) where
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

Passing the `Captioner` **class** (not an instance or a function) loads the model
once per GPU actor; a plain function would rebuild it on every batch. See
[GPU scheduling](gpu.md) for sizing the actor pool.

## Cleaning scraped text

Scraped pages arrive as markup, and so do product descriptions and email bodies.
`.str.strip_html()` recovers the prose: it drops tags along with the contents of
`<script>`/`<style>`, strips comments and decodes entities, then collapses whitespace,
separating block elements with a space.

```python
import batcher as bt

pages = bt.from_pydict({"page": ["<p>Tom &amp; Jerry</p><p>x</p><script>f()</script>"]})
print(pages.select(text=bt.col("page").str.strip_html()).to_pydict())
# {'text': ['Tom & Jerry x']}
```

Reach for this over the `regexp_replace('<[^>]*>', '')` idiom, which is wrong in three
ways that quietly poison a corpus: it leaves the JavaScript in `<script>` as prose, it
leaves `&amp;` and `&nbsp;` undecoded, and it welds `<p>a</p><p>b</p>` into `ab`. It is a
text extractor, not an HTML parser, so malformed markup never raises and one bad row in
a web scrape cannot abort the scan.

## Chunking documents (RAG ingest)

A document is usually longer than an embedding model's context, so the ingest chain is
**load → split → embed → index**. `.str.chunk(size, overlap)` is the split stage: it
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

The whole chain (scan, chunk, explode, embed) is a linear row-wise pipeline, so it
distributes across workers and streams over an unbounded source with no breaker. The
one thing no static rule can know is how many chunks a document yields; Kyber estimates
1× on the first run, Core measures the real fan-out, and the next plan sizes the
downstream GPU stage for it (see [adaptive re-optimization](../internals/kyber.md)).

## Vector search (RAG retrieval)

`ds.ml.embed` produces the vectors on a `Dataset`. Over a bare batch stream (chunks
coming out of a reader, or a stage you are composing by hand) `batcher.ml.embed` does
the same work and takes an `EncoderFactory`: a zero-argument callable returning an
encoder, which is any callable mapping `list[str]` to one equal-length vector per string.
The factory is called once per worker, so the embedding model loads a single time and
every batch that worker handles reuses it. Same shape as the `WorkerFactory` in
[inference](inference.md), and the reason a sentence-transformers model, a local ONNX
encoder, and a hosted embedding API are interchangeable here.

```python
# docs: skip
from sentence_transformers import SentenceTransformer

from batcher.ml import embed


def encoder_factory():  # an EncoderFactory — one model per worker
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    return lambda texts: model.encode(texts)


vectors = embed(chunks.iter_batches(), encoder_factory, text_column="chunk", num_workers=4)
```

The embedding is appended as `output_column` (default `"embedding"`), and the batches
come back in input order. Write them to Lance, then index and search.

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

When the embeddings already ride in a column (a reranking pass, or a small candidate
set that does not warrant an index) score them against a query vector in-engine with
the `.list` distance expressions, no Lance required. `.list.cosine_distance(q)` is
`1 - cosine_similarity` (0 for identical direction, 1 for orthogonal, 2 for opposite),
the standard embedding metric; `.list.l2_distance(q)` is the Euclidean distance. Each
takes the query as another column or an `array(...)` literal and returns a Float64 that
sorts ascending, so the nearest rows come first:

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
the two rank identically: cosine similarity is the dot product divided by both
magnitudes, and those are 1 once the vectors are normalized. So normalize once, up front
at embedding time, with `.list.normalize()` (L2-normalize each vector to unit length),
and retrieve with the plain `.list.dot(q)` against a likewise-normalized query.
`.list.l2_norm()` reports a vector's Euclidean magnitude, handy for confirming a vector
is already unit-length (norm 1) before you skip the normalization:

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
- [Preprocessors](preprocessors.md): assemble the decoded features into a training matrix.
- [Expressions API](../api/expressions.md): the `.image`/`.audio`/`.video` and vector
  `.list` method reference.
