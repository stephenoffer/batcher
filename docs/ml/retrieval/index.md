# Embeddings, retrieval, and generation

Vectors are first-class columns rather than a bolted-on index, which is what lets one
pipeline chunk a corpus, embed it, retrieve against it, and call a model on the result
without leaving the engine.

- {doc}`/ml/retrieval/embeddings`: encoding a text or image column into vectors.
- {doc}`/ml/retrieval/vector-search`: brute-force search in-engine, or an approximate index.
- {doc}`/ml/retrieval/rag`: chunk, embed, retrieve, generate, as one pipeline.
- {doc}`/ml/retrieval/llm/index`: batched text generation, engines, prompts, and throughput.
- {doc}`/ml/retrieval/llm-outputs`: parsing generated strings into typed columns, and guided decoding.
- {doc}`/ml/retrieval/llm-evaluation`: scoring generations, and the reference-free output monitors.

```{toctree}
:hidden:

embeddings
llm/index
llm-evaluation
llm-outputs
rag
vector-search
```
