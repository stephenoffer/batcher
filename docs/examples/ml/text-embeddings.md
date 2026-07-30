# Text embeddings

Embedding a corpus is the cheapest way to make a GPU look bad. The forward pass over a
MiniLM-sized model is milliseconds. Everything around it is where the wall clock goes:
reloading the weights, reading columns nobody asked for, shipping raw HTML to the
tokenizer. This page takes a table of documents to a normalized vector column ready for
retrieval.

## Clean the text first

Scraped text carries markup, and a tokenizer will happily spend its context on
`<script>` bodies and `&nbsp;`. `.str.strip_html()` extracts the prose (it drops script
and style contents, decodes entities, and separates block elements), and
`.str.normalize_whitespace()` collapses what is left. Both run in the engine, so this
costs a scan, not a Python loop.

```python
import batcher as bt
from batcher import col

docs = bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "html": [
            "<h1>Refund policy</h1><p>Returns accepted within 30 days.</p>",
            "<p>Shipping   is free over $50.</p><script>track()</script>",
            "<p>&amp;nbsp;</p>",
            "<p>Our support team answers within one business day.</p>",
        ],
    }
)

clean = (
    docs.with_columns(text=col("html").str.strip_html().str.normalize_whitespace())
    .filter(col("text").str.len() > 10)
    .select("id", "text")
)
print(clean.to_pydict()["text"])
# ['Refund policy Returns accepted within 30 days.', 'Shipping is free over $50.', 'Our support team answers within one business day.']
```

The `filter` matters more than it looks. An empty or near-empty document still costs a
full forward pass, and its vector is noise that will show up in every retrieval. Drop
those rows before they reach the GPU, not after.

## The model loads once per worker

::::{tab-set}
:::{tab-item} A model id
The model-id path is the short version: pass a sentence-transformers id and the column,
and the model loads once per worker and appends a vector column.

```python
# docs: skip
import batcher as bt

vectors = clean.ml.embed(
    "sentence-transformers/all-MiniLM-L6-v2",
    column="text",
    output_column="embedding",
    batch_size=256,
    num_gpus=1,
    concurrency=(1, 4),  # autoscale the actor pool to the backlog
)
vectors.write.parquet("s3://bucket/vectors.parquet")
```
:::

:::{tab-item} Your own encoder class
For any other encoder, pass a **class**. `map_batches`, `infer`, and `embed` instantiate
it once per worker, and the constructor is where the weights load.

```python
# docs: skip
import pyarrow as pa


class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    def __call__(self, batch):
        texts = batch.column("text").to_pylist()
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return batch.append_column("embedding", pa.array(vectors.tolist()))


vectors = clean.ml.embed(
    Embedder,  # the class, not an instance
    output_columns=["id", "text", "embedding"],
    batch_size=256,
    num_gpus=1,
    concurrency=2,
)
```
:::
::::

:::{warning}
Pass a plain function instead and the model is rebuilt on every batch: a 2-second load per
256 rows, which is the single most common way an embedding job ends up slower than the CPU
baseline. The engine warns (`PerformanceWarning`) when a GPU stage gets a bare function, but
it cannot fix it for you.
:::

Batcher's pools are session-warm: the model loads once per session and is reused across
calls, worth about 2× on repeated inference. That is only true if the load lives in the
constructor.

:::{tip}
`select` down to the columns the encoder reads before the stage. The optimizer cannot see
inside a Python function, so an embedding stage over one column of a 41-column Parquet
file reads all 41 unless you project first.
:::

## Normalize once, at write time

Cosine similarity is a dot product divided by two magnitudes. On unit-length vectors
those magnitudes are 1, so a normalized corpus retrieves with the cheaper `.list.dot`
kernel and ranks identically. Normalizing at query time instead pays for it on every
search, forever.

The encoder below is a deterministic bag-of-words hash, with no weights and no GPU, so the
whole shape runs here. Swap it for the `Embedder` above and nothing else changes.

```python
import zlib

import numpy as np
import pyarrow as pa

import batcher as bt
from batcher import array, col


class HashEmbedder:
    """Stands in for a real encoder: same call shape, no model."""

    def __init__(self, dim=16):
        self.dim = dim

    def __call__(self, batch):
        out = []
        for text in batch.column("text").to_pylist():
            vec = np.zeros(self.dim)
            for token in text.lower().split():
                vec[zlib.crc32(token.encode()) % self.dim] += 1.0
            out.append(vec.tolist())
        return batch.append_column("embedding", pa.array(out, pa.list_(pa.float64())))


embedded = clean.ml.embed(
    HashEmbedder, output_columns=["id", "text", "embedding"], batch_size=64
).with_columns(embedding=col("embedding").list.normalize())

print([round(v, 3) for v in embedded.to_pydict()["embedding"][1][:4]])
# [0.0, 0.0, 0.447, 0.0]
```

## Retrieve

With the corpus normalized, a query is one more embedding and a sort. `.list.dot` scores
every row in the engine; `top_k` keeps the nearest without materializing the corpus.
(A token-hash encoder has no semantics, so it only matches on literal overlap. The ranking
here proves the plumbing, not the recall.)

```python
query = bt.from_pydict({"id": [0], "text": ["returns accepted within 30 days"]})
qvec = (
    query.ml.embed(HashEmbedder, output_columns=["id", "text", "embedding"])
    .with_columns(embedding=col("embedding").list.normalize())
    .to_pydict()["embedding"][0]
)

hits = embedded.with_columns(score=col("embedding").list.dot(array(*qvec))).top_k(
    2, "score"
)
print(hits.to_pydict()["id"])
# [1, 4]
```

`.list.cosine_distance(q)` is the equivalent for a corpus you have *not* normalized; it
sorts ascending (0 is identical). Reach for it in a reranking pass over a small candidate
set, where the extra kernel cost is irrelevant.

At corpus scale, write the vectors to Lance and let an ANN index do the search instead of
scanning every row:

```python
# docs: skip
from batcher.ml import build_vector_index, vector_search

embedded.write.lance("s3://bucket/vectors.lance")
build_vector_index("s3://bucket/vectors.lance", "embedding")
hits = vector_search("s3://bucket/vectors.lance", qvec, column="embedding", k=10)
```

Three ways to score, and the corpus size decides which:

| Scoring | Kernel | Reach for it when |
| --- | --- | --- |
| `.list.dot(q)` + `top_k` | one dot product per row, in the engine | the corpus is normalized and small enough to scan |
| `.list.cosine_distance(q)` | dot plus both magnitudes, ascending | the corpus is not normalized, or you are reranking a candidate set |
| `build_vector_index` + `vector_search` | an ANN index over Lance | a scan per query is no longer a retrieval system |

## What to check when it is slow

:::{dropdown} The four things to check before you blame the GPU
- The stage got a class, not a function or an instance. A `PerformanceWarning` on a GPU
  stage means the model is reloading per batch.
- The scan reads only the columns the encoder needs (`select` before the stage).
- Empty and junk rows are filtered before the model, not after.
- `batch_size` is as large as device memory allows; the pool halves and retries a batch
  that OOMs rather than failing the job.
:::

## See also

- {doc}`RAG index <rag-index>`: chunk long documents before embedding them.
- {doc}`Training-data dedup <training-data-dedup>`: the near-duplicate pass to run before you
  spend a GPU-hour embedding the same document twice.
- {doc}`Embeddings <../../ml/embeddings>` and {doc}`vector search <../../ml/vector-search>`: the
  encoder surface and the index it feeds.
- {doc}`Inference <../../ml/inference>`: the pool and stage-overlap mechanics.
- {doc}`Multimodal <../../ml/multimodal>`: the `.list` vector expressions in full.
- {doc}`ML API reference <../../api/ml>`: `ds.ml.embed`, `build_vector_index`, `vector_search`.
- {doc}`AI and GPU benchmarks <../../benchmarks/ai-and-gpu>`: where the 33,611 text/s on text
  embeddings comes from.
- {doc}`Tensor columns <../../deep-dives/tensor-columns>`: how a vector column is laid out.
