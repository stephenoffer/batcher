# RAG from scratch

Retrieval-augmented generation is two dataset operations wearing a trench coat: embed a
corpus, then for each question retrieve the nearest chunks and hand them to a model. There
is no RAG operator in Batcher, and there does not need to be one. It is chunk, embed, score,
generate, and all four are ordinary Dataset work.

This tutorial builds the whole loop with a stub embedder and a stub model, so the retrieval
half runs here with no GPU and no downloads. Every block that needs a real model is marked
and shown, not run; swapping the stub for the real thing is a one-line change.

:::{note}
**What you'll build.** A three-document corpus, chunked with overlap, embedded, cached, and
searched by cosine similarity, then a generation step. The retrieval half runs here with
`pip install batcher-engine` alone. No GPU, no model download, no vector database.
:::

| Step | Runs here | Needs |
|---|---|---|
| Chunk | Yes | `pip install batcher-engine` |
| Embed | Yes, with a bag-of-words stub | A GPU and `sentence-transformers` for the real one |
| Retrieve | Yes | Nothing more |
| Generate | Yes, with a stub engine | A GPU and vLLM for the real one |

## 1. The corpus

```python
import batcher as bt
import pyarrow as pa

docs = bt.from_pydict(
    {
        "doc_id": ["refunds", "engine", "support"],
        "text": [
            "A refund is issued within five days of an approved return request.",
            "The engine runs SQL and DataFrames over one Rust data plane on Arrow.",
            "Support answers every ticket within one working day of receiving it.",
        ],
    }
)
print(docs.count())
# 3
```

In production this is `bt.read.parquet("s3://corpus/")`, or a directory of PDFs and HTML you
have already extracted. `bt.col("body").str.strip_html()` turns markup into prose if that is
what you have.

## 2. Chunk

A document is too big to embed usefully and too big to fit in a prompt. `.str.chunk(size,
overlap)` splits a string column into a list of chunks; `explode` turns that list into one
row per chunk. Both run in the engine, on every core, with no Python in the loop.

:::{tip}
Overlap matters: a sentence that straddles a chunk boundary is otherwise lost to retrieval.
Along with the prompt template, it is one of the two things on this page you will actually
spend time tuning.
:::

```python
chunks = (
    docs.with_columns(chunk=bt.col("text").str.chunk(40, overlap=8))
    .explode("chunk")
    .select("doc_id", "chunk")
)
print(chunks.to_pydict()["chunk"][:2])
# ['A refund is issued within five days of a', 'ays of an approved return request.']
```

Real chunks are 200 to 1,000 characters, not 40. The small size here keeps the output readable.

## 3. Embed

With a real model, embedding is one call. `ds.ml.embed` loads a sentence-transformers model
**once per worker**, keeps it warm across `collect()`s in the session, and appends the vector
as a tensor column:

```python
# docs: skip
vectors = chunks.ml.embed(
    "sentence-transformers/all-MiniLM-L6-v2",
    column="chunk",
    output_column="embedding",
    num_gpus=1,
)
vectors.write.parquet("s3://index/chunks/")
```

That warm pool is worth stating plainly, because it is the difference between a benchmark and
a bill: MiniLM loads in about 2 seconds and embeds nearly instantly, so an engine that reloads
the model per execution spends its entire runtime loading. Measured on 8xT4 over 8,192 texts,
Batcher embeds at **33,611 text/s** with the model loaded once for the session.

For the tutorial, a deterministic bag-of-words stub stands in. It is a `map_batches` that
appends a vector column, which is exactly the shape the real encoder has.

```python
VOCAB = ["refund", "day", "engine", "support", "sql"]


def encode(texts):
    return [[float(t.lower().count(word)) for word in VOCAB] for t in texts]


def embed_batch(batch):
    vectors = encode(batch.column("chunk").to_pylist())
    return batch.append_column(
        "embedding", pa.array(vectors, type=pa.list_(pa.float64()))
    )


index = chunks.map_batches(
    embed_batch, output_columns=["doc_id", "chunk", "embedding"]
).cache()
print(index.count())
# 6
```

`cache()` materializes the index once so the queries below reuse it instead of re-embedding.

## 4. Retrieve

Retrieval is a score and a top-N. `.list.cosine_similarity` scores each row's vector against
a query vector broadcast as a literal, and `top_k` keeps the best rows without sorting the
relation, using the fused top-N heap, which runs 8.1x faster than Daft on the top-N benchmark.

The `l2_norm` filter drops empty vectors: a zero vector has no direction, cannot clear any
threshold, and would otherwise pollute the ranking with undefined scores.

```python
question = "how many days until my refund arrives"
query = bt.array(*[bt.lit(x) for x in encode([question])[0]])

hits = (
    index.filter(bt.col("embedding").list.l2_norm() > 0)
    .with_columns(score=bt.col("embedding").list.cosine_similarity(query))
    .top_k(2, by="score")
    .select("doc_id", "chunk", "score")
)
found = hits.to_pydict()
print(found["doc_id"], [round(s, 3) for s in found["score"]])
# ['refunds', 'support'] [1.0, 0.707]
```

The refund chunk comes first; the support chunk, which also mentions a day, comes second. The
engine chunk does not appear at all.

That was a brute-force scan, which is the right answer up to a surprisingly large corpus.
Past that, put the vectors in Lance and use the ANN index.

::::{tab-set}
:::{tab-item} Brute-force scan
No index to build, keep warm, or invalidate. It is a vectorized pass over one column, and it
is what the block above already did.

```python
# docs: skip
scores = index.with_columns(score=bt.col("embedding").list.cosine_similarity(query))
best = scores.top_k(5, by="score")
```
:::

:::{tab-item} ANN index (Lance)
Worth the index once the corpus outgrows a scan. The retrieval is a lookup rather than a
pass.

```python
# docs: skip
from batcher.ml import build_vector_index, vector_search

build_vector_index("s3://index/chunks.lance", column="embedding")
hits = vector_search("s3://index/chunks.lance", query_vector, k=5)
```
:::
::::

And when you are matching *two* corpora rather than one question against one corpus,
`ds.ml.similarity_join` does it without the quadratic blowup: SimHash signatures band the
vectors into candidate pairs, and only the candidates get the exact cosine score.

## 5. Generate

Build the prompt with a string expression (it is a column operation like any other) and hand
it to a model. An *engine* is a zero-arg callable returning a
`list[str] -> list[str]` function, so a deterministic stub can stand in for a 7B model and
the pipeline is testable with no GPU.

```python
def stub_llm():
    def generate(prompts):
        return [p.split("CONTEXT: ")[1][:30] for p in prompts]

    return generate


answers = (
    hits.with_columns(
        prompt=bt.format_string("Q: {} CONTEXT: {}", bt.lit(question), bt.col("chunk"))
    )
    .ml.generate(stub_llm, prompt_column="prompt")
    .select("doc_id", "response")
)
print(answers.to_pydict()["doc_id"])
# ['refunds', 'support']
```

The real thing is the same call with a real engine. vLLM does its own continuous batching, so
Batcher hands it whole request lists and keeps the columnar work around it:

```python
# docs: skip
from batcher.ml import vllm_engine

engine = vllm_engine(
    "meta-llama/Llama-3-8B-Instruct",
    chat=True,
    system="Answer only from the context. If it is not there, say so.",
    sampling={"temperature": 0.0, "max_tokens": 256},
)

(
    bt.read.parquet("s3://questions/")
    .ml.generate(engine, prompt_column="prompt", num_gpus=1)
    .write.parquet("s3://answers/")
)
```

:::{warning}
Set `chat=True` for any instruction-tuned model. The default is the completion path, which is
right for a base model and quietly wrong for a chat model. It skips the chat template, so the
model answers a prompt in a format it was never trained on. The output degrades, and nothing
warns you. This is the most common way a RAG pipeline ends up producing plausible garbage.
:::

## 6. Why the loop is fast

Both halves of RAG are on the benchmark, and both are warm-pool workloads: the model loads
once per session rather than once per job. Distributed over 8×T4, correctness-gated on
output agreement:

| Half of RAG | Batcher |
|---|---:|
| Text embeddings (MiniLM, 8,192 texts) | **33,611 text/s** |
| LLM generation (gpt2, 2,048 prompts) | **814.8 prompt/s** |

Neither number comes from a RAG-specific code path. They come from the same mechanisms every
`map_batches` inference pipeline gets: session-warm pools, stage-overlapped streaming, and an
adaptive batch size that does not have to be tuned.

## What you learned

- RAG is chunk → embed → score → generate. Four Dataset operations, no framework.
- The chunk overlap and the prompt template are the two things you will actually tune.
- A stub embedder and a stub engine make the whole pipeline testable without a GPU. Use them.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` LLM inference
:link: /ml/retrieval/llm/index
:link-type: doc
vLLM engines, chat templates, structured output.
:::

:::{grid-item-card} {octicon}`search;1.1em` Vector search
:link: /ml/retrieval/vector-search
:link-type: doc
The ANN index, for when the brute-force scan stops being enough.
:::

:::{grid-item-card} {octicon}`graph;1.1em` AI and GPU benchmarks
:link: ../benchmarks/ai-and-gpu
:link-type: doc
Where the embedding and generation throughput comes from.
:::
::::

## See also

- {doc}`Batch inference <batch-inference>`: the `.ml` accessor, in full.
- {doc}`RAG guide </ml/retrieval/rag>` and {doc}`embeddings </ml/retrieval/embeddings>`: the production shape of
  each half.
- {doc}`RAG index recipe </cookbook/ml/pipelines/rag-index>` and
  {doc}`text embeddings recipe </cookbook/ml/pipelines/text-embeddings>`: the short versions.
- {doc}`Expressions </user-guide/transform/expressions>`: `.str.chunk`, `.list.cosine_similarity`, and
  the rest of the column language this page leans on.
- {doc}`AI and GPU benchmarks <../benchmarks/ai-and-gpu>`: the warm pool and the stage overlap
  behind both halves.
