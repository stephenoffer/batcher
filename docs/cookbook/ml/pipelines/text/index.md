# Text and language models

These pipelines run a language model over a corpus: labelling it, embedding it, or making it
searchable. Each keeps the model inside the pipeline, so the corpus streams past it rather
than being materialized first.

| Pipeline | What it builds |
|---|---|
| {doc}`LLM batch scoring <llm-batch-scoring>` | Structured labels over millions of documents, with a schema the output has to match |
| {doc}`Text embeddings <text-embeddings>` | An embedding column, batched so the GPU is not waiting on tokenization |
| {doc}`RAG index <rag-index>` | The retrieval half of a RAG system: load, chunk, embed, index |

## See also

- {doc}`/ml/retrieval/index`: the guide to embeddings, vector search, and generation.
- {doc}`/cookbook/metrics/text/index`: scoring what a model generated.

```{toctree}
:hidden:

llm-batch-scoring
text-embeddings
rag-index
```
