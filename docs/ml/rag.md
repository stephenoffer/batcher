# RAG pipelines

Most of a RAG system is a data pipeline, and most RAG failures are data failures: HTML
tags embedded in the corpus, chunks that cut a sentence in half, the same document
indexed four times so every retrieval returns four copies of it, and a chunk that came
back with no way to trace it to a source. None of that is a model problem. All of it is
fixed on the ingest side, in operators.

The chain is **load → clean → chunk → dedupe → embed → index**, then at query time
**embed → retrieve → prompt → generate**. Ingest is a batch job over a `Dataset`.
Retrieval is a query. Keep them separate.

## Ingest

### Clean the markup

A scraped page is markup. The `regexp_replace('<[^>]*>', '')` idiom that everyone
reaches for is wrong in three ways: it leaves the body of `<script>` in the corpus as
prose, it leaves `&amp;` undecoded, and it welds `<p>a</p><p>b</p>` into `ab`.
`.str.strip_html()` is a text extractor: it drops script and style bodies, decodes
entities, and separates block elements.

```python
import batcher as bt
from batcher import col

pages = bt.from_pydict(
    {
        "url": ["http://a", "http://b"],
        "html": [
            "<p>Cats &amp; dogs</p><p>are pets</p><script>track()</script>",
            "<h1>Trains</h1><p>run on rails</p>",
        ],
    }
)
docs = pages.select("url", text=col("html").str.strip_html())
print(docs.to_pydict()["text"])
# ['Cats & dogs are pets', 'Trains run on rails']
```

It never raises on malformed markup, so one bad row in a crawl of ten million cannot
abort the scan.

### Chunk, and keep the provenance

`.str.chunk(size, overlap)` slices text into overlapping windows; `explode` turns the
list into one row per chunk. Carry the source id through, and add a chunk index. A
retrieved chunk that cannot name its source document is a citation you cannot render.

```python
chunks = (
    docs.with_columns(chunk=col("text").str.chunk(12, overlap=4))
    .explode("chunk")
    .with_row_index("chunk_id")
)
print(chunks.select("url", "chunk_id", "chunk").to_pydict())
# {'url': ['http://a', 'http://a', 'http://b', 'http://b'], 'chunk_id': [0, 1, 2, 3],
#  'chunk': ['Cats & dogs ', 'ogs are pets', 'Trains run o', 'un on rails']}
```

:::{warning}
Sizes are in characters. Pick one comfortably under the embedding model's token limit
(about 4 characters per token as a rough conversion), because text past the limit is
silently truncated, and a vector for the first half of a chunk is worse than no vector.
`overlap` is what keeps a sentence split across a boundary whole in one of the two
chunks.
:::

### Deduplicate before embedding, not after

Chunk-level duplicates are the reason a RAG system returns the same paragraph three
times in a top-5. They come from the same document being crawled twice, from boilerplate
headers and footers repeating across every page, and from near-identical product blurbs.
Drop them before you pay for the forward pass.

```python
corpus = bt.from_pydict(
    {
        "chunk_id": [1, 2, 3, 4],
        "chunk": [
            "the quick brown fox jumps over",
            "the quick brown fox jumps over",
            "the quick brown fox jumps over!",
            "an entirely unrelated paragraph",
        ],
    }
)
clean = corpus.distinct(["chunk"]).ml.drop_near_duplicates(
    "chunk", threshold=0.7, key="chunk_id"
)
print(sorted(clean.to_pydict()["chunk_id"]))
# [1, 4]
```

`distinct` gets the byte-identical ones cheaply; `drop_near_duplicates` (MinHash + LSH)
gets the ones that differ by a header or a trailing exclamation mark. On a web corpus
the near-duplicate rate is routinely 20–40%, and every one of them is a wasted GPU
forward pass and a polluted retrieval.

### Embed and index

```python
# docs: skip
import batcher as bt
from batcher.ml import build_vector_index

vectors = clean.ml.embed(
    "sentence-transformers/all-MiniLM-L6-v2",
    column="chunk",
    batch_size=256,
    num_gpus=1,
    concurrency=4,
)
vectors.write.lance("s3://bucket/chunks.lance")
build_vector_index("s3://bucket/chunks.lance", "embedding")
```

The whole ingest chain (scan, strip, chunk, explode, dedupe, embed) is row-wise apart
from the dedup, so it streams and distributes with no breaker before the GPU stage. The
one thing no static estimate can know is how many chunks a document yields; the engine
measures the real fan-out on the first run and sizes the downstream GPU stage for it on
the next. See [adaptive re-optimization](../internals/kyber.md).

## Retrieval

Embed the question with the *same* model, then rank. Both forms return a `Dataset`, so
everything downstream of them is identical.

::::{tab-set}
:::{tab-item} Brute force

On a corpus small enough to scan, this is a projection and a top-n: no index, no service.

```python
from batcher import array

# Vectors already in a column (a toy 2-d space; a real one is 384–1536 dims).
indexed = bt.from_pydict(
    {
        "chunk_id": [1, 2, 3],
        "url": ["http://a", "http://b", "http://c"],
        "chunk": ["cats are pets", "trains run on rails", "kittens are small cats"],
        "embedding": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
    }
)
question_vec = array(1.0, 0.0)  # the question, embedded by the same model

retrieved = (
    indexed.with_columns(dist=col("embedding").list.cosine_distance(question_vec))
    .top_k(2, by="dist", descending=False)
    .sort("dist")
)
print(retrieved.select("chunk_id", "chunk").to_pydict())
# {'chunk_id': [1, 3], 'chunk': ['cats are pets', 'kittens are small cats']}
```

:::

:::{tab-item} Against an index

```python
# docs: skip
from batcher.ml import vector_search

hits = vector_search(
    "s3://bucket/chunks.lance", question_vec, k=5, filter="tenant = 'acme'"
)
```

Metadata scoping is a predicate here, not a post-filter.

:::
::::

:::{tip}
That `filter` runs against the index rather than against the k rows it returned, which is
the difference between "5 results, all from this tenant" and "5 results, 2 of which you
have to throw away". See [vector search](vector-search.md).
:::

## Building the prompt

Concatenate the retrieved chunks into one context string per question. This is a
`group_by` with `array_agg` and a list join: an aggregate, not a Python loop.

```python
prompt = retrieved.group_by().agg(context=col("chunk").array_agg())
context = prompt.to_pydict()["context"][0]
print(" | ".join(context))
# cats are pets | kittens are small cats
```

Keep the `url` / `chunk_id` alongside, so the answer can cite what it used. A RAG system
that cannot show its sources cannot be debugged, and it cannot be trusted by anyone who
has to sign off on its output.

## Generation

`ds.ml.generate(engine, ...)` runs the LLM stage over batches. An engine is any callable
from a list of prompts to a list of completions, which is why a local vLLM engine and a
hosted OpenAI-compatible endpoint are interchangeable at this seam.

```python
# docs: skip
from batcher.ml import vllm_engine

questions = bt.from_pydict({"question": ["what are kittens?"], "context": [" ".join(context)]})
answers = questions.ml.generate(
    vllm_engine("meta-llama/Llama-3.1-8B-Instruct", chat=True),
    prompt_column="question",
    template="Answer using only this context:\n{context}\n\nQuestion: {question}",
    output_column="answer",
)
answers.write.parquet("s3://bucket/answers.parquet")
```

`template` builds the prompt from columns, so the context you assembled above lands in
the prompt without a per-row Python string format. For a JSON answer, `parse_json=True`
with a `guided_json` schema gets typed columns back instead of a string you have to
regex. See [LLM inference](llm.md).

## The failure modes, in order

Every one of these is a data bug that presents as a model bug, which is why they survive
so long.

| What you see | Where it comes from | The fix |
| --- | --- | --- |
| Answers that ignore the end of a document | chunks larger than the model's context, silently cut | check the length distribution first: `ds.select(n=col("chunk").str.len()).describe()` |
| The same paragraph three times in a top-5 | deduping at the document level while boilerplate repeats across pages | dedupe at the chunk level |
| An answer nobody can attribute | the source id dropped somewhere in ingest | carry `url` and `chunk_id` from ingest all the way through retrieval |
| Retrieval that is subtly, consistently poor | model skew: the query embedded by a different model than the corpus | store the model name alongside the vectors |
| Results from another tenant | a permission filter applied to the k rows the search returned | push the filter into the search |

:::{warning}
Model skew is the subtle one. The query must be embedded by the same model as the corpus,
and nothing enforces that. Store the model name alongside the vectors, so a re-embed with
a new model cannot silently mix two vector spaces — a mixed index does not fail, it just
retrieves nonsense with confident-looking distances.
:::

## See also

- [Embeddings](embeddings.md): the encode stage in detail.
- [Vector search](vector-search.md): brute force vs an ANN index.
- [LLM inference](llm.md): engines, chat templates, structured output.
- [Governance](../user-guide/governance.md): row filters and column masks, if the corpus
  is not all one tenant's.
- [RAG from scratch](../tutorials/rag-from-scratch.md): the tutorial, built up step by
  step.
- [RAG index recipe](../examples/ml/rag-index.md): the ingest half as a runnable job.
- [Adaptive re-optimization](../deep-dives/adaptive-reoptimization.md): how the engine
  learns the chunk fan-out no static estimate could know.
- [AI and GPU benchmarks](../benchmarks/ai-and-gpu.md): what the embed and generate
  stages cost.
