# Embeddings

The expensive part of an embedding job is not the forward pass, it is everything around
it: loading the model once per worker instead of once per batch, keeping the GPU fed,
and not embedding the same document three times because the corpus has duplicates. Get
those right and a 100M-document embed job is a scan with a GPU stage bolted to it.

## Embed a column

`ds.ml.embed(model, column=...)` takes a model identifier and resolves a
`sentence-transformers` model, loading it once per worker. Pass a **class** instead when
the encoder is yours: a local ONNX model, a fine-tuned checkpoint, a non-text modality.

::::{tab-set}
:::{tab-item} A model identifier

```python
# docs: skip
import batcher as bt

docs = bt.read.parquet("s3://bucket/docs.parquet")
vectors = docs.ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="text")
vectors.write.parquet("s3://bucket/vectors.parquet")
```

That appends an `embedding` column and is the whole job for the common case.

:::

:::{tab-item} A custom encoder class

```python
# docs: skip
import pyarrow as pa


class Encoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    def __call__(self, batch):
        vecs = self.model.encode(batch.column("text").to_pylist(), batch_size=64)
        return batch.append_column("embedding", pa.array(vecs.tolist()))


vectors = docs.ml.embed(
    Encoder,
    output_columns=["id", "text", "embedding"],
    batch_size=256,
    num_gpus=1,
    concurrency=4,
)
```

The class is constructed once per worker, and you declare the output schema.

:::
::::

:::{tip}
Pass the class, not a function. A bare function would rebuild the model on every batch,
which is the single most expensive mistake in this API: on a 5,000-batch job it is 5,000
model loads instead of one per worker.
:::

Sizing the actor pool is the other half of keeping the device busy.

| Encoder | Pool | What you get |
| --- | --- | --- |
| Large enough to fill a device | `num_gpus=1, concurrency=4` | four actors, each holding a whole GPU |
| Small (MiniLM is 90 MB) | `num_gpus=0.25, concurrency=8` | several packed onto each device, usually double the throughput |

See {doc}`GPU scheduling <gpu>`.

## Embed via a served endpoint

The embedding model can also run behind a service, such as a HuggingFace
Text-Embeddings-Inference (TEI) server on a GPU box or a hosted API such as OpenAI.
The worker then calls it instead of
loading weights. Two load-once encoders speak the two common wire shapes and drop into
`ds.ml.embed` exactly like a local model:

```python
# docs: skip
from batcher.ml import openai_embedding_encoder, tei_encoder

# An OpenAI-compatible /embeddings endpoint (OpenAI, Azure, Together, vLLM's server).
openai = openai_embedding_encoder(
    "text-embedding-3-small", "text", api_key="sk-...", dimensions=256
)
vectors = docs.ml.embed(openai, output_columns=["id", "text", "embedding"])

# A HuggingFace TEI server (BGE / GTE / E5 on a GPU), normalizing server-side.
tei = tei_encoder("text", base_url="http://tei-host:8080")
vectors = docs.ml.embed(tei, output_columns=["id", "text", "embedding"])
```

Each batch's texts are sent in concurrent, size-bounded requests, so a served endpoint is
saturated rather than called one row at a time. The `dimensions` argument asks a Matryoshka
model (the `text-embedding-3-*` family) for a shorter vector, trading a little recall for a
smaller index. The output column is a `fixed_size_list<float32>`, the shape Lance ANN
indexing expects, so the served and local paths are interchangeable downstream.

The mechanics run without a GPU, which is worth seeing once. The encoder here is a toy,
but the pipeline shape is the real one.

```python
import pyarrow as pa

import batcher as bt


class ToyEncoder:  # stands in for a real model: same contract, no weights
    def __call__(self, batch):
        vecs = [
            [float(t.count("a")), float(t.count("b")), float(len(t))]
            for t in batch.column("text").to_pylist()
        ]
        return batch.append_column("embedding", pa.array(vecs, pa.list_(pa.float64())))


docs = bt.from_pydict({"id": [1, 2, 3], "text": ["aab", "bbb", "aaa"]})
vectors = docs.ml.embed(ToyEncoder, output_columns=["id", "text", "embedding"])
print(vectors.to_pydict())
# {'id': [1, 2, 3], 'text': ['aab', 'bbb', 'aaa'],
#  'embedding': [[2.0, 1.0, 3.0], [0.0, 3.0, 3.0], [3.0, 0.0, 3.0]]}
```

## Normalize once, at write time

Cosine similarity is a dot product divided by both magnitudes. On unit-length vectors
those magnitudes are 1, so cosine and dot rank identically, and dot is the cheaper
kernel. Normalize at ingest with `.list.normalize()` and every query afterwards gets to
use the cheap one.

```python
from batcher import col

unit = vectors.with_columns(embedding=col("embedding").list.normalize())
print(unit.select(norm=col("embedding").list.l2_norm()).to_pydict())
# {'norm': [1.0, 1.0, 1.0]}
```

`.list.l2_norm()` is how you check whether vectors from an unfamiliar source are already
normalized before you spend a pass normalizing them again. Many hosted embedding APIs
return unit vectors; many local models do not.

## Binarize for cheap Hamming search

When a small recall loss is acceptable, a binary embedding is far cheaper to search: each
dimension becomes one bit by its sign, and distance is a bit count rather than a float dot
product. `ds.ml.binarize_embeddings` produces the sign code, and `nearest_neighbors` ranks
it with `metric="hamming"`.

```python
coded = vectors.ml.binarize_embeddings("embedding", output_column="code")
print(coded.select(col("code")).to_pydict()["code"][0])
# a list of 0/1, one per dimension
```

## Shrink Matryoshka vectors, and drop the degenerate ones

A Matryoshka-trained model (the `text-embedding-3-*` family, Nomic, mxbai) packs the most
signal into the leading dimensions, so a prefix is a smaller, faster index for a small
recall cost. Take the prefix with `ds.ml.truncate_embeddings`, which re-normalizes it. A
raw slice is no longer unit length, and a cosine index silently assumes it is:

```python
short = unit.ml.truncate_embeddings("embedding", 2)
print(short.select(norm=col("embedding").list.l2_norm()).to_pydict())
# {'norm': [1.0, 1.0, 1.0]}
```

Before indexing, drop rows whose vector is the zero vector or null. A zero vector has no
direction, so an index returns it as a garbage neighbor; it usually means an empty input
or a failed encode. `ds.ml.drop_degenerate_embeddings` removes both:

```python
with_holes = bt.from_pydict(
    {"id": [1, 2, 3], "embedding": [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]}
)
clean = with_holes.ml.drop_degenerate_embeddings("embedding")
print(clean.to_pydict()["id"])
# [1, 3]
```

## Deduplicate before you embed, not after

Embedding is the expensive stage. Every duplicate document is a full forward pass you did
not need, and web corpora are full of them: the same article under three headers, the
same product blurb from four suppliers. `distinct` removes byte-identical rows, and
`drop_near_duplicates` removes the ones that are the same document with a different
header.

```python
corpus = bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "text": [
            "the quick brown fox jumps",
            "the quick brown fox jumps",
            "the quick brown fox jumps!",
            "a completely different sentence",
        ],
    }
)
deduped = corpus.distinct(["text"]).ml.drop_near_duplicates("text", threshold=0.7, key="id")
print(sorted(deduped.to_pydict()["id"]))
# [1, 4]
```

Two of the four documents were going to cost a forward pass each for nothing. On a real
crawl the near-duplicate rate is routinely 20% to 40%, and that is the same fraction off
your GPU bill. See {doc}`preprocessors <preprocessors/index>` for the MinHash and LSH tuning.

## Fuse dense and lexical rankings

Dense embedding search and lexical (BM25) search miss different things: the embedding finds
paraphrases, the keyword match finds exact terms and rare tokens. Hybrid retrieval runs both
and fuses the rankings. `ds.ml.reciprocal_rank_fusion` does the fusing without asking you to
put the two score scales into agreement. Each list contributes `1 / (k + rank)` per key, and
a document ranked highly by either retriever floats up.

```python
dense = bt.from_pydict({"id": [1, 2, 3], "score": [0.9, 0.5, 0.1]})
lexical = bt.from_pydict({"id": [2, 3, 4], "score": [0.8, 0.7, 0.6]})
fused = dense.ml.reciprocal_rank_fusion(lexical, key="id", score="score")
print(fused.to_pydict()["id"])
# [2, 3, 1, 4]
```

Documents 2 and 3, ranked by both retrievers, win; 1 and 4, each ranked by only one, follow.
Pass more result sets as extra arguments to fuse three or more retrievers.

## Score a whole query set at once

Evaluating retrieval means running many queries against the corpus, not one.
`ds.ml.batched_nearest_neighbors` scores a query set against this corpus and keeps each
query's top `k` in one pass. It is an exact, index-free brute force that is right for an eval set
and honest about being `O(queries x corpus)`.

```python
corpus = bt.from_pydict({"cid": [1, 2, 3], "emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]})
queries = bt.from_pydict({"qid": [10, 11], "qv": [[1.0, 0.05], [0.0, 1.0]]})
hits = corpus.ml.batched_nearest_neighbors(
    queries, query_key="qid", query_column="qv", corpus_key="cid", column="emb", k=1
)
print(sorted(zip(hits.to_pydict()["qid"], hits.to_pydict()["cid"])))
# [(10, 1), (11, 2)]
```

For a large corpus queried in production, build an ANN index instead, as described below.

Once you have the retrieved neighbors and a set of ground-truth relevant pairs,
`ds.ml.recall_at_k` scores the retrieval: of the documents that should have come back for
each query, what fraction did, averaged over queries.

```python
retrieved = bt.from_pydict({"qid": [1, 1, 2, 2], "cid": [10, 11, 20, 21]})
relevant = bt.from_pydict({"qid": [1, 2, 2], "cid": [10, 20, 22]})
print(round(retrieved.ml.recall_at_k(relevant, query_key="qid", corpus_key="cid"), 3))
# 0.75
```

Recall asks whether the right documents came back; `ds.ml.mrr` asks how *high* the first
right one ranked. Ask for the rank alongside the neighbors (`rank_column="rank"` on
`batched_nearest_neighbors`), then:

```python
ranked = bt.from_pydict({"qid": [1, 1, 2, 2], "cid": [10, 11, 20, 21], "rank": [1, 2, 1, 2]})
print(ranked.ml.mrr(relevant, query_key="qid", corpus_key="cid"))
# a first-relevant near the top scores near 1; missing it scores 0
```

## Chunk long documents first

:::{warning}
An embedding model has a context limit, and text past it is silently truncated. You get a
vector for the first 512 tokens and a false belief that it represents the document.
Nothing raises, nothing warns, and the retrieval quality quietly is not what you
think it is.
:::

Split first with `.str.chunk(size, overlap)`, then `explode` into one row per chunk, and
each chunk gets its own vector.

```python
long_docs = bt.from_pydict({"id": [1], "body": ["abcdefghij"]})
chunks = long_docs.with_columns(chunk=col("body").str.chunk(4, overlap=1)).explode("chunk")
print(chunks.to_pydict()["chunk"])
# ['abcd', 'defg', 'ghij']
```

`overlap` keeps a sentence cut across a boundary whole in one of the two chunks. Sizes
are in characters, so pick one comfortably under the model's token limit. A rough rule is
4 characters per token. Keep the document id on the row so you can attribute a retrieved
chunk back to its source. {doc}`RAG <rag>` walks the full ingest.

## Storing vectors

An embedding is a `List<Float64>` column. It is Arrow like any other column, so it writes
to Parquet, joins, and filters without ceremony. Write to Lance instead when you intend
to build an ANN index over it, which is what {doc}`vector search <vector-search>` needs at
scale.

```python
# docs: skip
unit.write.lance("s3://bucket/vectors.lance")
```

:::{note}
A 1024-dimension float64 vector is 8 KB per row, so a million rows is 8 GB. Cast to
`float32` before writing if the recall loss is acceptable, and keep the vector column out
of any sort or join that does not need it. `offload_blobs` exists for exactly that. See
{doc}`multimodal <multimodal>`.
:::

## Driving the pool yourself

Sometimes you are composing a stage rather than executing a `Dataset`, inside a custom
loop or a serving process. `batcher.ml.embed` does the same work over a bare batch iterator. It
takes an `EncoderFactory`: a zero-argument callable returning an encoder, which is any
callable from `list[str]` to one vector per string. The factory runs once per worker.

```python
# docs: skip
from sentence_transformers import SentenceTransformer

from batcher.ml import embed


def encoder_factory():
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    return lambda texts: model.encode(texts)


batches = embed(chunks.iter_batches(), encoder_factory, text_column="chunk", num_workers=4)
```

Same contract as the `WorkerFactory` in {doc}`inference <inference>`, which is why a local
model, an ONNX runtime, and a hosted embedding API are interchangeable at this seam.

## Scoring embedding quality

Once vectors exist, the questions you ask of them are numeric, and the metrics for that aggregate a per-row vector operation to a corpus score in one scan. `bt.mean_cosine_similarity(query, doc)` is the headline retrieval-alignment number; `bt.mean_euclidean_distance` and `bt.mean_dot_product` are the magnitude-sensitive and inner-product variants a distance-thresholded or MIPS index ranks by.

```python
import batcher as bt

pairs = bt.from_pydict({"q": [[1.0, 0.0], [1.0, 1.0]], "doc": [[1.0, 0.0], [0.0, 1.0]]})
print(pairs.agg(sim=bt.mean_cosine_similarity("q", "doc")).to_pydict())
```

The single-column checks catch a degenerate index before it returns garbage: `bt.unit_norm_rate` verifies the vectors are normalized the way a cosine index assumes, `bt.zero_vector_rate` finds empty or failed embeddings, and `bt.mean_embedding_norm` tracks the average magnitude for drift. All compose with `group_by` to monitor per model, per source, or per day.

Match the drift metric to the distance space your index uses. `bt.mean_cosine_distance` is the `1 - cosine` form a cosine index ranks by, `bt.mean_manhattan_distance` is the L1 metric that resists a single dominant dimension, `bt.mean_angular_distance` is the true-metric angle some indexes build on, and `bt.mean_hamming_distance` is the bit-disagreement count for binary or product-quantized vectors.

```python
import batcher as bt

vecs = bt.from_pydict({"a": [[1.0, 0.0], [1.0, 0.0]], "b": [[1.0, 0.0], [0.0, 1.0]]})
print(vecs.agg(drift=bt.mean_cosine_distance("a", "b")).to_pydict())
# {'drift': [0.5]}
```

## See also

- {doc}`Vector search <vector-search>`: retrieving against the vectors you built.
- {doc}`RAG <rag>`: the full chunk → embed → retrieve → generate pipeline.
- {doc}`GPU scheduling <gpu>`: sizing the actor pool that runs the encoder.
- {doc}`GPU execution <../deep-dives/gpu-execution>`: how a GPU stage is scheduled, and what
  "once per worker" means underneath.
- {doc}`Text embeddings recipe <../examples/ml/text-embeddings>`: the job, end to end, on a
  real corpus.
- {doc}`RAG index recipe <../examples/ml/rag-index>`: the same vectors, written and indexed.
- {doc}`AI and GPU benchmarks <../benchmarks/ai-and-gpu>`: what an embed job costs against
  the alternatives.
- {doc}`ML API <../api/ml>`: the `ml.embed` and `EncoderFactory` reference.
