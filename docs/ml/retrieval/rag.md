# RAG pipelines

This page builds a retrieval-augmented generation (RAG) pipeline on Batcher, from a raw crawl to a cited answer. Most of a RAG system is a data pipeline, and most RAG failures are data failures: markup left in the corpus, chunks that cut a sentence in half, one document indexed four times, a chunk with no way back to its source. None of that is a model problem, and all of it is fixed at ingest.

Ingest runs load, clean, chunk, dedupe, embed and index as a batch job over a {py:class}`Dataset <batcher.Dataset>`. Query time runs embed, retrieve, prompt and generate. Keep the two separate.

The following diagram shows both halves and the one thing they share:

![Ingest is a batch job over the corpus. Load and clean with str.strip_html() passes text to chunk, which runs str.chunk and explode. Chunks go to dedupe, exact duplicates first and then near duplicates, and the unique chunks go to embed and index, ml.embed written to Lance. Every ingest stage is an engine operator, so the job streams and distributes. The chunk vectors are the only thing query time reads from ingest. Per question, embed the question with the same model, and pass the vector to retrieve, a top_k scan on a small corpus or vector_search against the Lance index at scale. Retrieve passes the top 100 to rerank, where a cross-encoder keeps 20 by relevance and MMR then keeps 5 by diversity. The top 5 go to generate, which assembles the context with array_agg and calls ml.generate. Carry url and chunk_id from ingest through retrieval so the answer can cite its sources.](/_static/diagrams/rag_ingest_query.svg)

## Ingest

Ingest turns a raw corpus into an indexed set of chunks. Every stage below is an engine operator, so the chain streams and distributes like any other query.

### Clean the markup

The {py:meth}`regexp_replace('<[^>]*>', '') <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_replace>` idiom is wrong in three ways. It leaves the body of `<script>` in the corpus as prose, leaves `&amp;` undecoded, and welds `<p>a</p><p>b</p>` into `ab`. {py:meth}`.str.strip_html() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.strip_html>` is a text extractor. It drops script and style bodies, decodes entities, and turns element boundaries into a single space.

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

Malformed markup never raises, so one bad row in a large crawl can't abort the scan.

### Chunk, and keep the provenance

{py:meth}`.str.chunk(size, overlap) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.chunk>` slices text into overlapping windows, and `explode` turns the list into one row per chunk. Carry the source id through and add a chunk index. A retrieved chunk that can't name its source is a citation you can't render.

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
Sizes are in characters. Pick one comfortably under the embedding model's token limit, using about 4 characters per token as a rough conversion. Text past the limit is silently truncated, and a vector for the first half of a chunk is worse than no vector. `overlap` keeps a sentence split across a boundary whole in one of the two chunks.
:::

The default cut can land mid-word. Pass `boundary="word"`, `"sentence"` or `"line"` to back each cut off to the last such separator inside the window.

### Deduplicate before embedding, not after

Chunk-level duplicates are why a RAG system returns the same paragraph three times in a top-5. They come from a document crawled twice, from headers and footers repeated on every page, and from near-identical product blurbs. Drop them before you pay for the forward pass.

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
clean = corpus.distinct(["chunk"]).ml.drop_near_duplicates("chunk", threshold=0.7, key="chunk_id")
print(sorted(clean.to_pydict()["chunk_id"]))
# [1, 4]
```

`distinct` removes the byte-identical ones cheaply. {py:meth}`drop_near_duplicates <batcher.api.dataset.ml.DatasetML.drop_near_duplicates>`, built on MinHash and LSH, catches the ones that differ by a header or a trailing exclamation mark. Every duplicate it drops is a GPU forward pass you don't pay for and a retrieval slot it can't pollute.

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

Scan, strip, chunk, explode and embed are all row-wise, so they stream. The dedup steps are the only operators in the chain that have to see many rows at once. How many chunks a document yields is the one number no static estimate knows. The optimizer treats an explode's fan-out as correctable: it measures the real one on the first run and uses it for the estimate on the next. See {doc}`Kyber </architecture/internals/kyber>`.

## Retrieval

Embed the question with the *same* model, then rank. Both forms below return a `Dataset`, so everything downstream is identical.

::::{tab-set}
:::{tab-item} Brute force

On a corpus small enough to scan, retrieval is a projection and a top-n, with no index and no service.

```python
from batcher import array

# Vectors already in a column (a toy 2-d space; a real one is 384 to 1536 dims).
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

hits = vector_search("s3://bucket/chunks.lance", question_vec, k=5, filter="tenant = 'acme'")
```

`filter` is a SQL predicate applied with the search. It isn't a post-filter.

:::
::::

:::{tip}
Because `filter` runs with the search rather than over the k rows it returned, you get "5 results, all from this tenant" instead of "5 results, 2 of which you have to throw away". See {doc}`vector search </ml/retrieval/vector-search>`.
:::

## Build the prompt

Concatenate the retrieved chunks into one context per question with a {py:meth}`group_by <batcher.Dataset.group_by>` and `array_agg`. It's an aggregate, not a Python loop.

```python
prompt = retrieved.group_by().agg(context=col("chunk").array_agg())
context = prompt.to_pydict()["context"][0]
print(" | ".join(context))
# cats are pets | kittens are small cats
```

Keep `url` and `chunk_id` alongside so the answer can cite what it used. A RAG system that can't show its sources can't be debugged, and nobody who signs off on its output can trust it.

## Generate the answer

{py:meth}`ds.ml.generate(engine, ...) <batcher.api.dataset.ml.DatasetML.generate>` runs the LLM stage over batches. `engine` is a zero-argument factory that returns a callable from a list of prompts to a list of completions, so a local vLLM engine and a hosted OpenAI-compatible endpoint are interchangeable here.

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

`template` is a `str.format` template over the row's columns, so the context you assembled lands in the prompt with no string formatting of your own. For a JSON answer, `parse_json=True` with a `vllm_engine(guided_json=...)` schema returns a struct column instead of a string you have to regex. See {doc}`LLM inference </ml/retrieval/llm/index>`.

## Rerank for relevance

A vector search is only the first stage of retrieval. A bi-encoder embeds each passage once, offline, without knowing the question, so the vector can't encode how the passage relates to any particular query. That blindness is what makes it fast enough to run over a whole corpus. It's also the ceiling.

A *cross-encoder* reads the query and one passage together and scores the pair. It sees the interaction the bi-encoder discarded and is more accurate for it. It can't be precomputed, so it only runs over a candidate set the first stage already narrowed: retrieve 100 with vectors, rerank to 5.

`bt.ml.cross_encoder_rerank_udf` is that stage. It takes the shape a vector search leaves behind, one row per query with the candidates in list columns.

```python
import batcher as bt
from batcher.ml import cross_encoder_rerank_udf

hits = bt.from_pydict(
    {
        "question": ["how tall is everest"],
        "passages": [["everest is 8849 m", "k2 is 8611 m", "a recipe for soup"]],
        "ids": [["p1", "p2", "p3"]],
    }
)


def scorer():  # stands in for a real cross-encoder in this example
    return lambda pairs: [float("everest" in passage) for _, passage in pairs]


reranked = hits.map_batches(
    cross_encoder_rerank_udf(
        scorer,
        query_column="question",
        document_column="passages",
        rerank_columns=("ids",),
        k=2,
    )
)
print(reranked.to_pydict()["passages"])
# [['everest is 8849 m', 'k2 is 8611 m']]
```

In production the first argument is a model id and the model loads once per worker:

```python
# docs: skip
reranked = hits.map_batches(
    cross_encoder_rerank_udf(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        query_column="question",
        document_column="passages",
        rerank_columns=("ids", "scores"),
        k=5,
        activation="sigmoid",
    ),
    num_gpus=1,
    concurrency=4,
)
```

Every `(query, passage)` pair in the batch is scored in one model call. A batch of 64 queries with 100 candidates each is 6,400 pairs, enough to fill a GPU, where row-by-row scoring would run 64 small forwards and leave the device mostly idle. Columns in `rerank_columns` are reordered alongside the passages so ids and first-stage scores stay aligned. The reranker's own scores land in `score_column`, which defaults to `rerank_score`.

`activation="sigmoid"` maps the raw logits into `[0, 1]`, which is what a threshold wants. The ordering is identical either way.

A zero-argument factory works in place of a model id. `scorer` above returns a `CrossEncoderScorer`, and the whole contract is a list of `(query, passage)` pairs in and one score per pair out, in order. Use it for a hosted reranking API, a model the package doesn't know, or a test with no GPU.

## Rerank for diversity

Nearest isn't the same as useful. On a real corpus several of the `k` nearest passages say the same thing: documents get republished, chunks overlap by design, and a boilerplate paragraph matches everything. The context window then holds one fact four times, and the model reads repetition as emphasis.

`bt.ml.mmr_rerank_udf` applies maximal marginal relevance. It builds the result greedily, penalizing each candidate for resembling what's already chosen, so the same token budget covers more of the answer.

```python
import batcher as bt
from batcher.ml import mmr_rerank_udf

hits = bt.from_pydict(
    {
        "vecs": [[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]],
        "passages": [["a fact", "the same fact again", "a different fact"]],
        "scores": [[0.9, 0.89, 0.5]],
    }
)
reranked = hits.map_batches(
    mmr_rerank_udf(
        embedding_column="vecs",
        score_column="scores",
        rerank_columns=("passages", "scores"),
        k=2,
        lambda_mult=0.7,
    )
)
print(reranked.to_pydict()["passages"])
# [['a fact', 'a different fact']]
```

`lambda_mult` is the dial. At `1.0` you get back the relevance ranking you had, duplicates included. At `0.0` it optimizes for spread alone. The useful range is 0.5 to 0.8. Every column in `rerank_columns` is cut to the selected candidates in selection order, and similarity is cosine on normalized copies, so an unnormalized index needs no separate pass.

Run the two rerankers in that order. The cross-encoder costs a model call and decides relevance. MMR costs nothing beyond the vectors you already hold and drops the near-duplicates among the survivors. Narrow 100 to 20 by relevance, then 20 to 5 by diversity.

## Measure the pipeline

The failures in the next section are easier to fix than to notice. Each metric here is an aggregate over a column, so a whole eval set is one scan, and any of them breaks down by index version, tenant or day with {py:meth}`group_by <batcher.Dataset.group_by>`.

Start with retrieval, because a grounding score over a context that was never retrieved measures nothing. {py:func}`bt.empty_retrieval_rate <batcher.empty_retrieval_rate>` counts the queries that got no passages. That failure looks like unexplained hallucination: with no context, the model answers from its parameters, fluently and without a citation. {py:func}`bt.duplicate_context_rate <batcher.duplicate_context_rate>` counts queries whose passages contain an exact repeat. {py:func}`bt.mean_retrieved_passages <batcher.mean_retrieved_passages>` shows a retriever quietly returning fewer than the `k` you asked for.

```python
import batcher as bt

runs = bt.from_pydict(
    {
        "index": ["v1", "v1", "v2", "v2"],
        "hits": [[], ["a chunk"], ["a chunk", "a chunk"], ["a chunk", "another"]],
    }
)
print(
    runs.group_by("index")
    .agg(
        empty=bt.empty_retrieval_rate("hits"),
        duplicated=bt.duplicate_context_rate("hits"),
        mean_k=bt.mean_retrieved_passages("hits"),
    )
    .sort("index")
    .to_pydict()
)
```

{py:func}`bt.context_token_estimate <batcher.context_token_estimate>` sizes what retrieval is about to cost, dividing characters by `chars_per_token` (4.0 by default) rather than running a tokenizer. Retrieved context is usually the largest part of a RAG prompt and grows silently: raising `k` from 5 to 10 doubles every request's input bill.

On the answer side, {py:func}`bt.answer_groundedness <batcher.answer_groundedness>` measures how much of the answer its context backs word by word, and {py:func}`bt.phrase_groundedness <batcher.phrase_groundedness>` does the same for phrases. Read them together. An answer that rearranges the context's own words into a claim the context never made scores perfectly on the first and badly on the second. That gap is what a confident hallucination looks like.

```python
answers = bt.from_pydict(
    {
        "answer": ["mat sat cat quietly the on"],
        "context": ["the cat sat quietly on the mat"],
    }
)
print(
    answers.agg(
        tokens=bt.answer_groundedness("answer", "context"),
        phrases=bt.phrase_groundedness("answer", "context"),
    ).to_pydict()
)
```

{py:func}`bt.unsupported_phrase_rate <batcher.unsupported_phrase_rate>` is the same signal inverted. It rises as the system gets worse, which is the direction a dashboard alert wants.

## Common failure modes

Each failure in the following table is a data bug that looks like a model bug, which is why it survives so long:

| What you see | Where it comes from | The fix |
| --- | --- | --- |
| Answers that ignore the end of a document | chunks larger than the model's context, silently cut | check the length distribution first: {py:meth}`ds.select(n=col("chunk").str.len_chars()).describe() <batcher.Dataset.select>` |
| The same paragraph three times in a top-5 | deduping at the document level while boilerplate repeats across pages | dedupe at the chunk level |
| An answer nobody can attribute | the source id dropped somewhere in ingest | carry `url` and `chunk_id` from ingest all the way through retrieval |
| Retrieval that is subtly, consistently poor | model skew: the query embedded by a different model than the corpus | store the model name alongside the vectors |
| Results from another tenant | a permission filter applied to the k rows the search returned | push the filter into the search |

:::{warning}
Model skew is the subtle one. Nothing enforces that the query and the corpus share a model. Store the model name alongside the vectors so a re-embed can't silently mix two vector spaces. A mixed index doesn't fail. It retrieves nonsense at confident-looking distances.
:::

## See also

- {doc}`Embeddings </ml/retrieval/embeddings>`: the encode stage in detail.
- {doc}`Vector search </ml/retrieval/vector-search>`: brute force against an ANN index.
- {doc}`LLM inference </ml/retrieval/llm/index>`: engines, chat templates and structured output.
- {doc}`LLM evaluation </ml/retrieval/llm-evaluation>`: grounding, retrieval and judge metrics over a whole eval set.
- {doc}`Governance </user-guide/trust/governance>`: row filters and column masks for a multi-tenant corpus.
- {doc}`RAG from scratch </getting-started/tutorials/ml/rag-from-scratch>`: the same pipeline as a step-by-step tutorial.
- {doc}`RAG index recipe </cookbook/ml/pipelines/text/rag-index>`: the ingest half as a runnable job.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: what the embed and generate stages cost.
