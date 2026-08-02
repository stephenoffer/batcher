# Media in a pipeline

What changes once media is a column: what it costs to move, how it reaches a model, and how it is retrieved.

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
{doc}`GPU scheduling </ml/inference/gpu>` for sizing the actor pool.

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
the downstream GPU stage for it. See {doc}`adaptive re-optimization </architecture/internals/kyber>`.

## Vector search over the embeddings

`ds.ml.embed` produces the vectors on a `Dataset`. Over a bare batch stream, such as
chunks coming out of a reader or a stage you are composing by hand, `batcher.ml.embed`
does the same work and takes an `EncoderFactory`. That is a zero-argument callable
returning an encoder, which is any callable mapping `list[str]` to one equal-length
vector per string. The factory is called once per worker, so the embedding model loads a
single time and every batch that worker handles reuses it. It has the same shape as the
`WorkerFactory` in {doc}`inference </ml/inference/inference>`, which is the reason a sentence-transformers
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

Vector search needs `batcher-engine[lance]`. See {doc}`embeddings </ml/retrieval/embeddings>` for the
compute side and {doc}`LLM inference </ml/retrieval/llm/index>` for generation over retrieved context.

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
