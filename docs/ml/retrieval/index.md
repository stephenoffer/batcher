# Embeddings, retrieval, and generation

This section covers the vector and language-model half of an ML pipeline: encoding columns into embeddings, searching them, building RAG, running LLMs over millions of rows, and parsing and scoring what the model says.

Vectors are ordinary columns in Batcher, not a separate store you sync to. An embedding is a list column, the distance functions are expressions, and a top-k search is a projection and a sort. One pipeline can chunk a corpus, embed it, retrieve against it, and call a model on the result, and every step streams, distributes, and joins like any other relation.

## A retrieval pipeline in miniature

The example below uses a toy two-dimensional embedding and a stand-in engine that upper-cases its prompt, so it runs without a GPU. It retrieves the two chunks closest to a query vector and generates one answer per chunk from a prompt template:

```python
import batcher as bt

docs = bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "chunk": ["cats are pets", "trains run on rails", "kittens are small cats", "dogs bark"],
        "vec": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.5, 0.5]],
    }
)
hits = docs.ml.nearest_neighbors([1.0, 0.0], column="vec", k=2)
print(hits.select("id", "chunk").to_pydict())
# {'id': [1, 3], 'chunk': ['cats are pets', 'kittens are small cats']}

engine = lambda: lambda prompts: [p.upper() for p in prompts]
answers = hits.ml.generate(engine, prompt_column="chunk", template="Answer from: {chunk}")
print(answers.sort("id").to_pydict()["response"])
# ['ANSWER FROM: CATS ARE PETS', 'ANSWER FROM: KITTENS ARE SMALL CATS']
```

In production the vectors come from {py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>` and the engine from {py:func}`bt.ml.vllm_engine <batcher.ml.vllm_engine>` or {py:func}`bt.ml.http_engine <batcher.ml.http_engine>`. The pipeline shape doesn't change.

## Score, threshold, and monitor

{py:meth}`ds.ml.similarity_to <batcher.api.dataset.ml.DatasetML.similarity_to>` scores every row without the top-k cut, which is what thresholding and reranking need:

```python
scored = docs.ml.similarity_to([1.0, 0.0], column="vec")
print(scored.filter(bt.col("score") > 0.9).select("id").to_pydict())
# {'id': [1, 3]}
```

Retrieval-quality metrics are aggregates, so they group by model, source, or day like any other measure:

```python
pairs = bt.from_pydict({"q": [[1.0, 0.0]], "doc": [[0.6, 0.8]]})
print(pairs.agg(sim=bt.mean_cosine_similarity("q", "doc")).to_pydict())
# {'sim': [0.6]}
```

## What you get

- **Warm, load-once actor pools** for embedding and generation. On an 8xT4 Ray cluster, sentence-transformers MiniLM embeds 33,611 texts per second.
- **One engine contract.** An LLM engine is a callable from a list of prompts to a list of completions. vLLM, SGLang, any OpenAI-compatible endpoint, Anthropic, Bedrock and Gemini all satisfy it, and a lambda is enough to test a pipeline in CI.
- **Typed outputs.** {py:meth}`ds.ml.extract <batcher.api.dataset.ml.DatasetML.extract>` and {py:meth}`ds.ml.classify <batcher.api.dataset.ml.DatasetML.classify>` parse generations back into typed columns.
- **Brute force to an index.** Exact search is a sort in the engine, and SQL spells it `ORDER BY list_cosine_similarity(...) LIMIT k`. Past a few million vectors, {py:func}`build_vector_index <batcher.ml.build_vector_index>` and {py:func}`vector_search <batcher.ml.vector_search>` put an approximate index over a Lance dataset.

## In this section

The following table lists the pages in the order a RAG system is usually built:

| Page | Covers |
|---|---|
| {doc}`/ml/retrieval/embeddings` | Embedding a text or image column locally or through a served endpoint, normalizing, and binarizing. |
| {doc}`/ml/retrieval/vector-search` | Brute-force search in the engine, filtering before scoring, an ANN index over Lance, and joining on meaning. |
| {doc}`/ml/retrieval/rag` | Ingest, retrieval, prompt building, and generation as one pipeline. |
| {doc}`/ml/retrieval/llm/index` | Running a language model over a column: the call, prompts and conversations, and engines. |
| {doc}`/ml/retrieval/llm-outputs` | Extracting typed columns, and parsing generated text without a second model call. |
| {doc}`/ml/retrieval/llm-evaluation` | Scoring generations against a reference, and reference-free output monitors. |

## See also

- {doc}`/getting-started/tutorials/ml/rag-from-scratch`: build a RAG pipeline step by step.
- {doc}`/ml/preparing/preprocessors/deduplication`: near-duplicate removal with MinHash, and similarity joins on embeddings.
- {doc}`/ml/training/training-corpus`: preparing text for pretraining or fine-tuning.
- {doc}`/ml/inference/gpu`: how `num_gpus` and `concurrency` size the actor pools.

```{toctree}
:hidden:

embeddings
vector-search
rag
llm/index
llm-outputs
llm-evaluation
```
