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

See [GPU scheduling](gpu.md).

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
crawl the near-duplicate rate is routinely 20–40%; that is the same fraction off your
GPU bill. See [preprocessors](preprocessors.md) for the MinHash/LSH tuning.

## Chunk long documents first

:::{warning}
An embedding model has a context limit, and text past it is silently truncated. You get a
vector for the first 512 tokens and a false belief that it represents the document.
Nothing raises, nothing warns, and the retrieval quality just quietly is not what you
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
are in characters, so pick one comfortably under the model's token limit (a rough rule is
4 characters per token). Keep the document id on the row so you can attribute a retrieved
chunk back to its source. [RAG](rag.md) walks the full ingest.

## Storing vectors

An embedding is a `List<Float64>` column. It is Arrow like any other column, so it writes
to Parquet, joins, and filters without ceremony. Write to Lance instead when you intend
to build an ANN index over it, which is what [vector search](vector-search.md) needs at
scale.

```python
# docs: skip
unit.write.lance("s3://bucket/vectors.lance")
```

:::{note}
A 1024-dimension float64 vector is 8 KB per row: a million rows is 8 GB. Cast to
`float32` before writing if the recall loss is acceptable (it usually is), and keep the
vector column out of any sort or join that does not need it. `offload_blobs` exists for
exactly that (see [multimodal](multimodal.md)).
:::

## Driving the pool yourself

When you are composing a stage rather than executing a `Dataset` (a custom loop, a
serving process), `batcher.ml.embed` does the same work over a bare batch iterator. It
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

Same contract as the `WorkerFactory` in [inference](inference.md), which is why a local
model, an ONNX runtime, and a hosted embedding API are interchangeable at this seam.

## See also

- [Vector search](vector-search.md): retrieving against the vectors you just built.
- [RAG](rag.md): the full chunk → embed → retrieve → generate pipeline.
- [GPU scheduling](gpu.md): sizing the actor pool that runs the encoder.
- [GPU execution](../deep-dives/gpu-execution.md): how a GPU stage is scheduled, and what
  "once per worker" means underneath.
- [Text embeddings recipe](../examples/ml/text-embeddings.md): the job, end to end, on a
  real corpus.
- [RAG index recipe](../examples/ml/rag-index.md): the same vectors, written and indexed.
- [AI and GPU benchmarks](../benchmarks/ai-and-gpu.md): what an embed job costs against
  the alternatives.
- [ML API](../api/ml.md): the `ml.embed` and `EncoderFactory` reference.
