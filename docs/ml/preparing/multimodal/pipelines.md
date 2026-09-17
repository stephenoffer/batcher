# Media in a pipeline

This page covers what changes once media is a column: the cost of moving large payloads through a plan, the path from a table of references to predictions, and searching the embeddings that come out.

## Keep large payloads out of shuffles and spills

A multi-GB payload such as a video, an audio file, or a PDF, carried inline in a column, is copied through every sort, join and spill buffer it crosses, even when those operators only touch other columns. {py:meth}`offload_blobs <batcher.Dataset.offload_blobs>` writes each payload to a content-addressed store and leaves a tiny URI handle in its place. {py:meth}`materialize_blobs <batcher.Dataset.materialize_blobs>` reads it back right before you need the bytes. In between, only the short handle string rides through the pipeline.

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

Offload is content-addressed with SHA-256, so identical payloads are written once, and a re-read after a spill fetches the same bytes. The store defaults to the configured spill location: `spill_remote_uri` when it's set, which makes handles reachable cluster-wide, and the local spill directory otherwise.

The following diagram compares the payload carried inline with the payload offloaded:

![Two paths through the same sort and join. Inline, an id and payload row carries gigabytes of bytes, the sort by id copies the payload, the join copies it again, and only the next step reads the bytes, so every copy and every spill buffer carries the full payload although sort and join only touch id. Offloaded, offload_blobs writes each payload once per SHA-256 hash to a content-addressed store at root/sha256, which is spill_remote_uri when set and the local spill directory otherwise, and replaces it with a uri handle. The sort moves only short handle strings. materialize_blobs then reads the bytes back from the store just in time for the next step. auto_offload_blobs=True places this pair around a sort for large_binary columns, and it is off by default.](/_static/diagrams/blob_offload.svg)

To place the pair automatically around a sort, set `auto_offload_blobs`. The engine then offloads any `large_binary` column the sort doesn't key on and reads it back afterward, with no change to your plan:

```python
# docs: skip
from batcher.config import Config, ExecutionConfig, config_context

with config_context(Config().replace(execution=ExecutionConfig(auto_offload_blobs=True))):
    ds.sort("id").collect()  # large_binary columns ride the sort as handles
```

It's off by default. The round trip to the store only pays off for genuinely large payloads, which is what the `large_binary` type signals.

## From references to predictions

Fetch, decode and a GPU model stage compose into one lazy pipeline. Preprocessing stays on CPU workers, and only the model holds a GPU:

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
    .map_batches(Captioner, batch_size=64, num_gpus=1, concurrency=2)  # GPU: model
)
captioned.write.parquet("s3://bucket/captioned.parquet")
```

Pass the `Captioner` **class**, not an instance or a function. Batcher then loads the model once per GPU actor, where a plain function would rebuild it on every batch. {doc}`GPU scheduling </ml/inference/gpu>` covers sizing the actor pool.

## Chunking documents for RAG ingest

A document is usually longer than an embedding model's context, so the ingest chain is load, split, embed, index. {py:meth}`.str.chunk(size, overlap) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.chunk>` is the split stage. It slices text into overlapping windows as a `List<Utf8>`, and `explode` turns that into one row per chunk. Sizes are in characters, so a boundary never splits a Unicode codepoint.

```python
import batcher as bt

docs = bt.from_pydict({"id": [1, 2], "body": ["abcdef", "xyz"]})
chunks = docs.with_columns(chunk=bt.col("body").str.chunk(4, overlap=1)).explode("chunk")
print(chunks.select("id", "chunk").to_pydict())
# {'id': [1, 1, 2], 'chunk': ['abcd', 'def', 'xyz']}
```

`overlap` carries context across a boundary, so a sentence cut in half still appears whole in one chunk. Chunking stops once a chunk reaches the end of the text, so the last chunk is never a redundant suffix of its predecessor. The default `boundary="char"` cuts at exactly `size` characters and can split a word, and a half word embeds as something a query won't match. Pass `boundary="word"`, `"sentence"` or `"line"` to back each cut off to the last such separator. From here, {py:meth}`ds.ml.embed(...) <batcher.api.dataset.ml.DatasetML.embed>` produces the vectors, and the next section searches them.

Scan, chunk, explode and embed form a linear row-wise pipeline with no breaker, so the chain distributes across workers. The one thing no static rule can know is how many chunks a document yields. On the first run Kyber's estimate passes the input row count through, Core measures the real fan-out, and later runs correct the estimate for every stage below the explode. {doc}`/architecture/deep-dives/adaptive/cardinality-estimation` explains the learned correction.

## Vector search over the embeddings

`ds.ml.embed` produces the vectors on a {py:class}`Dataset <batcher.Dataset>`. For a bare batch stream, such as chunks coming out of a reader or a stage you compose by hand, {py:func}`batcher.ml.embed <batcher.ml.embed>` does the same work. It takes an `EncoderFactory`, a zero-argument callable returning an encoder, where an encoder is any callable mapping `list[str]` to one vector per string. The factory runs once per worker, so the model loads once and every batch on that worker reuses it. It has the same shape as the `WorkerFactory` in {doc}`inference </ml/inference/inference>`, which is why a sentence-transformers model, a local ONNX encoder and a hosted embedding API are interchangeable here.

```python
# docs: skip
from sentence_transformers import SentenceTransformer

from batcher.ml import embed


def encoder_factory():  # an EncoderFactory: one model per worker
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    return lambda texts: model.encode(texts)


vectors = embed(chunks.iter_batches(), encoder_factory, text_column="chunk", num_workers=4)
```

The embedding is appended as `output_column`, which defaults to `"embedding"`, and batches come back in input order. Once the vectors are in a Lance dataset, `vector_search` retrieves the rows nearest a query vector, and `build_vector_index` builds an ANN index first so the search scales:

```python
# docs: skip
from batcher.ml import vector_search, build_vector_index

build_vector_index("s3://bucket/vectors.lance", "embedding")
hits = vector_search("s3://bucket/vectors.lance", query_vector, column="embedding", k=10)
top = hits.collect()  # k rows nearest to the query, with a _distance column
```

Vector search needs `batcher-engine[lance]`. {doc}`/ml/retrieval/vector-search` covers it in depth.

Sometimes the embeddings already ride in a column, as in a reranking pass or a candidate set too small to warrant an index. Score them in the engine with the {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>` distance expressions, with no Lance required. {py:meth}`.list.cosine_distance(q) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.cosine_distance>` is `1 - cosine_similarity`, the standard embedding metric: 0 for identical direction, 1 for orthogonal, and 2 for opposite. {py:meth}`.list.l2_distance(q) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.l2_distance>` is the Euclidean distance. Each takes the query as another column or an {py:func}`array(...) <batcher.array>` literal and returns a Float64 that sorts ascending, so the nearest rows come first:

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

A dot product is cheaper than a full cosine, and on **unit-length** vectors the two rank identically, because cosine similarity is the dot product divided by both magnitudes. So normalize once at embedding time with {py:meth}`.list.normalize() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.normalize>`, which L2-normalizes each vector to unit length, and retrieve with the plain {py:meth}`.list.dot(q) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.dot>` against a likewise-normalized query. {py:meth}`.list.l2_norm() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.l2_norm>` reports a vector's Euclidean magnitude, which confirms a vector is already unit-length before you skip the normalization:

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

## See also

- {doc}`/ml/preparing/multimodal/decoding`: fetching bytes and decoding them into tensor columns.
- {doc}`/ml/preparing/multimodal/video`: sampling frames from clips, the largest payloads a pipeline carries.
- {doc}`/ml/retrieval/embeddings`: producing embeddings at scale with `ds.ml.embed`.
- {doc}`/ml/retrieval/rag`: the full retrieval-augmented generation pipeline.
- {doc}`/ml/inference/gpu`: sizing GPU actor pools for the model stage.
