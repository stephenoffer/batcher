# RAG index

The retrieval half of a RAG system is a data pipeline: load → chunk → embed → index. It
is also where most of the quality lives. A retriever that returns the wrong 800 characters
cannot be rescued by a better generator, and the usual cause is a chunking decision made
in thirty seconds.

## Chunk with overlap

A document is longer than the embedding model's context, so it has to be cut.
`.str.chunk(size, overlap)` slices text into fixed-size windows as a `List<Utf8>`, and
`explode` turns that into one row per chunk. Sizes are in characters, and a boundary never
splits a codepoint.

:::{warning}
Overlap is not optional. Cut at a hard boundary and the sentence that answers the question
is half in chunk 3 and half in chunk 4, so neither embeds close to the query and neither is
retrieved. An overlap of 10 to 20% of the chunk size costs a little storage and buys back the
straddling sentences.
:::

```python
import batcher as bt
from batcher import col

docs = bt.from_pydict(
    {
        "doc_id": [1, 2],
        "title": ["Refunds", "Shipping"],
        "body": [
            "Refunds are issued to the original payment method within 30 days. "
            "Items must be unused and in the original packaging.",
            "Orders over $50 ship free. Express shipping arrives in two business days.",
        ],
    }
)

chunks = (
    docs.with_columns(chunk=col("body").str.chunk(64, overlap=16))
    .explode("chunk")
    .with_row_index("chunk_id")
    .select("chunk_id", "doc_id", "title", "chunk")
)
print(chunks.count())
# 5
print(chunks.to_pydict()["chunk"][0])
# Refunds are issued to the original payment method within 30 days
```

:::{tip}
Keep `doc_id` and `title` on every chunk. Without them, retrieval returns a paragraph you
cannot cite and cannot filter. Metadata filtering (by tenant, by product, by date) is what
makes a RAG index usable in production rather than a demo.
:::

Chunk fan-out is the one thing no static optimizer can know. Kyber estimates 1× on the
first run; Core measures the real ratio and the next plan sizes the downstream embedding
stage for it.

## Embed the chunks

::::{tab-set}
:::{tab-item} A real encoder, on GPUs

The real encoder loads once per worker and runs on GPU actors:

```python
# docs: skip
vectors = chunks.ml.embed(
    "sentence-transformers/all-MiniLM-L6-v2",
    column="chunk",
    output_column="embedding",
    batch_size=256,
    num_gpus=1,
    concurrency=(1, 4),
)
vectors.write.lance("s3://bucket/chunks.lance")
```
:::

:::{tab-item} A stub, so the page runs

The stub below is a token-hash encoder: deterministic, no weights, so the rest of the page
runs. Normalizing at index time means retrieval is a dot product rather than a cosine,
and the two rank identically on unit vectors.

```python
import zlib

import numpy as np
import pyarrow as pa


class HashEmbedder:
    """Stands in for a real encoder: same call shape, no model."""

    def __init__(self, dim=32):
        self.dim = dim

    def __call__(self, batch):
        out = []
        for text in batch.column("chunk").to_pylist():
            vec = np.zeros(self.dim)
            for token in text.lower().split():
                vec[zlib.crc32(token.encode()) % self.dim] += 1.0
            out.append(vec.tolist())
        return batch.append_column("embedding", pa.array(out, pa.list_(pa.float64())))


index = chunks.ml.embed(
    HashEmbedder,
    output_columns=["chunk_id", "doc_id", "title", "chunk", "embedding"],
    batch_size=128,
).with_columns(embedding=col("embedding").list.normalize())

print(index.collect().schema.field("embedding").type)
# list<item: double>
```
:::
::::

## Retrieve

At corpus scale, write the vectors to Lance and search an ANN index. A scan over ten million
chunks per query is not a retrieval system.

```python
# docs: skip
from batcher.ml import build_vector_index, vector_search

build_vector_index("s3://bucket/chunks.lance", "embedding")
hits = vector_search("s3://bucket/chunks.lance", query_vector, column="embedding", k=5)
```

| Retrieval path | What it costs per query | Reach for it when |
| --- | --- | --- |
| `.list.dot` + `top_k` in the engine | a scan of the candidate rows | the candidates are already narrowed: one tenant, one product, a reranking pass |
| `build_vector_index` + `vector_search` over Lance | an ANN lookup | the corpus is large enough that a scan per query is not a retrieval system |

Below the index-is-worth-it threshold (a per-tenant index, a reranking pass, a candidate set
already narrowed by a filter) score in the engine instead: `.list.dot` against the
normalized query, `top_k` for the nearest, and a metadata filter in front of the scan so it
only touches the rows the tenant may see.

```python
from batcher import array

query = bt.from_pydict(
    {"chunk_id": [0], "doc_id": [0], "title": [""], "chunk": ["refunds issued to the original payment method"]}
)
qvec = (
    query.ml.embed(HashEmbedder, output_columns=["chunk_id", "doc_id", "title", "chunk", "embedding"])
    .with_columns(embedding=col("embedding").list.normalize())
    .to_pydict()["embedding"][0]
)

hits = (
    index.with_columns(score=col("embedding").list.dot(array(*qvec)))
    .top_k(2, "score")
    .select("doc_id", "title", "chunk", "score")
)
print(hits.to_pydict()["doc_id"])
# [1, 2]
```

The top hit is the refunds document, and `doc_id` rides along with it. That is the point of
keeping it: you can deduplicate hits by document, cite the source, or pull the neighbouring
chunks. (A token-hash encoder only matches literal overlap, so treat the
ranking here as proof of the plumbing, not of recall. A real encoder is the block above.)

## Generate over the retrieved context

The generation half is the same engine and the same operators. Retrieved chunks are rows;
a prompt is a `template` over their columns.

:::{dropdown} The generation stage, over the rows retrieval just produced

```python
# docs: skip
from batcher.ml import vllm_engine

engine = vllm_engine(
    "meta-llama/Llama-3-8B-Instruct",
    chat=True,
    system="Answer only from the provided context. Say 'unknown' if it is not there.",
    sampling={"temperature": 0.0, "max_tokens": 256},
)
answers = hits.ml.generate(
    engine,
    template="Context:\n{chunk}\n\nQuestion: how long do refunds take?",
    prompt_column="chunk",
    output_column="answer",
    num_gpus=1,
)
```
:::

RAG is retrieval plus an LLM, and Batcher runs both halves on one engine: 33,611 text/s
embedding with MiniLM and 814.8 prompt/s generating with gpt2 on 8xT4, with one model load per
worker and stage overlap between them.

## Keeping the index fresh

Re-embedding a corpus every night because 0.1% of it changed is the most common waste in a
RAG pipeline. The chunks are rows in a table; the fix is a filter.

```python
# docs: skip
import batcher as bt
from batcher import col

changed = bt.read.parquet("s3://bucket/docs.parquet").filter(col("updated_at") > last_run)
new_chunks = changed.with_columns(chunk=col("body").str.chunk(512, overlap=64)).explode("chunk")
new_vectors = new_chunks.ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="chunk", num_gpus=1)
new_vectors.write.lance("s3://bucket/chunks.lance", mode="append")
```

:::{important}
Delete the stale chunks for those `doc_id`s first, or you will retrieve two versions of the
same paragraph and the model will pick one at random.
:::

## See also

- [Text embeddings](text-embeddings.md): the encoder stage in detail.
- [LLM batch scoring](llm-batch-scoring.md): engines, prompt templates, typed outputs.
- [RAG](../../ml/rag.md) and [vector search](../../ml/vector-search.md): the retrieval surface
  end to end.
- [Embeddings](../../ml/embeddings.md): encoders, normalization, and the distance kernels.
- [Multimodal](../../ml/multimodal.md): `build_vector_index`, `vector_search`, and the
  `.list` distance expressions.
- [ML API reference](../../api/ml.md): `str.chunk`, `ds.ml.embed`, `ds.ml.generate`.
- [AI and GPU benchmarks](../../benchmarks/ai-and-gpu.md): the throughput figures quoted above.
- [Tensor columns](../../deep-dives/tensor-columns.md): how the vectors are stored and shipped.
