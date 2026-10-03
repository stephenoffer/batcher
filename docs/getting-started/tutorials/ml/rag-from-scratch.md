# RAG from scratch

Retrieval-augmented generation is chunk, embed, score, and generate, and all four are ordinary Dataset work. This tutorial builds the whole loop with a stub embedder and a stub model, so it runs on `pip install batcher-engine` alone: no GPU, no model download, no vector database. Swapping a stub for the real model is a one-line change.

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

In production this is {py:meth}`bt.read.parquet("s3://corpus/") <batcher.api.io_namespace.reader.Reader.parquet>`, or a directory of PDFs and HTML you
have already extracted. {py:func}`bt.col("body").str.strip_html() <batcher.col>` turns markup into prose if that is
what you have.

## 2. Chunk

`.str.chunk(size, overlap)` splits a string column into a list of overlapping chunks, in the engine:

```python
tiny = bt.from_pydict({"text": ["abcdefghij"]})
print(tiny.select(chunks=bt.col("text").str.chunk(4, overlap=1)).to_pydict())
# {'chunks': [['abcd', 'defg', 'ghij']]}
```

`explode` then turns the list into one row per chunk. Overlap keeps a sentence that straddles a boundary retrievable.

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

With a real model, embedding is one call. {py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>` loads a sentence-transformers model
**once per worker**, keeps it warm across {py:meth}`collect() <batcher.Dataset.collect>`s in the session, and appends the vector
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

With the model loaded once for the session, Batcher embeds **33,611 text/s** on 8xT4.

For the tutorial, a deterministic bag-of-words stub stands in: a `map_batches` that appends
a vector column, exactly the shape the real encoder has.

```python
VOCAB = ["refund", "day", "engine", "support", "sql"]


def encode(texts):
    return [[float(t.lower().count(word)) for word in VOCAB] for t in texts]


def embed_batch(batch):
    vectors = encode(batch.column("chunk").to_pylist())
    return batch.append_column("embedding", pa.array(vectors, type=pa.list_(pa.float64())))


index = chunks.map_batches(embed_batch, output_columns=["doc_id", "chunk", "embedding"]).cache()
print(index.count())
# 6
```

`cache()` materializes the index once so the queries below reuse it instead of re-embedding.

## 4. Retrieve

Retrieval is a score and a top-N. {py:meth}`.list.cosine_similarity <batcher.plan.expr_ir.namespaces.collections._ListNamespace.cosine_similarity>` scores each row's vector against a query vector, and `top_k` keeps the best rows without sorting the relation:

```python
vecs = bt.from_pydict({"doc": ["a", "b", "c"], "embedding": [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]})
probe = bt.array(bt.lit(1.0), bt.lit(0.0))
print(vecs.with_columns(score=bt.col("embedding").list.cosine_similarity(probe)).to_pydict()["score"])
# [1.0, 0.6, 0.0]
```

{py:meth}`ds.ml.similarity_to <batcher.api.dataset.ml.DatasetML.similarity_to>` is the same score as one call:

```python
print(vecs.ml.similarity_to([1.0, 0.0]).top_k(1, by="score").to_pydict()["doc"])
# ['a']
```

On the corpus, an {py:meth}`l2_norm <batcher.plan.expr_ir.namespaces.collections._ListNamespace.l2_norm>` filter first drops zero vectors, which have no direction to score:

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

The refund chunk comes first. The support chunk, which also mentions a day, comes second,
and the engine chunk does not appear at all.

That was a brute-force scan, which serves a large corpus well. Past that, put the vectors in Lance and use the ANN index.

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
{py:meth}`ds.ml.similarity_join <batcher.api.dataset.ml.DatasetML.similarity_join>` does it without the quadratic blowup: SimHash signatures band the
vectors into candidate pairs, and only the candidates get the exact cosine score.

## 5. Generate

Build the prompt with a string expression:

```python
pair = bt.from_pydict({"q": ["when?"], "ctx": ["five days"]})
print(pair.select(prompt=bt.format_string("Q: {} CONTEXT: {}", bt.col("q"), bt.col("ctx"))).to_pydict())
# {'prompt': ['Q: when? CONTEXT: five days']}
```

Then hand it to a model. An *engine* is a zero-arg callable returning a `list[str] -> list[str]` function, so a deterministic stub can stand in for a 7B model:

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

:::{important}
Set `chat=True` for an instruction-tuned model so the engine applies its chat template. Use `chat=False` for a base model.
:::

## 6. Why the loop is fast

The model loads once per session rather than once per job. On 8xT4, the {doc}`AI and GPU benchmark </benchmarks/results/ai-and-gpu>` measured the following:

| Half of RAG | Batcher |
|---|---:|
| Text embeddings (MiniLM, 8,192 texts) | **33,611 text/s** |
| LLM generation (gpt2, 2,048 prompts) | **814.8 prompt/s** |

Both come from what every `map_batches` pipeline gets: session-warm pools, stage-overlapped streaming, and adaptive batch sizing.

## Where to go next

Keep the stubs after the real models arrive. They keep the pipeline testable on a machine with no GPU.

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
The ANN index, for corpora past a scan.
:::

:::{grid-item-card} {octicon}`graph;1.1em` AI and GPU benchmarks
:link: /benchmarks/results/ai-and-gpu
:link-type: doc
Where the embedding and generation throughput comes from.
:::
::::

## See also

- {doc}`Batch inference </getting-started/tutorials/ml/batch-inference>`: the `.ml` accessor, in full.
- {doc}`RAG guide </ml/retrieval/rag>` and {doc}`embeddings </ml/retrieval/embeddings>`: the production shape of
  each half.
- {doc}`RAG index recipe </cookbook/ml/pipelines/text/rag-index>` and
  {doc}`text embeddings recipe </cookbook/ml/pipelines/text/text-embeddings>`: the short versions.
- {doc}`Expressions </user-guide/transform/columns/expressions>`: {py:meth}`.str.chunk <batcher.plan.expr_ir.namespaces.strings._StrNamespace.chunk>`, {py:meth}`.list.cosine_similarity <batcher.plan.expr_ir.namespaces.collections._ListNamespace.cosine_similarity>`, and
  the rest of the column language this page leans on.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: the warm pool and the stage overlap
  behind both halves.
