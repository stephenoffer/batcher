# Models and features

These tutorials put a model in the pipeline. Each one runs the model inside the engine
rather than beside it, so a predicate can push beneath a decode and a tensor never has to
leave.

| Tutorial | What you build |
|---|---|
| {doc}`Batch inference <batch-inference>` | A function run over whole Arrow batches through the `.ml` accessor, with the model loaded once |
| {doc}`RAG from scratch <rag-from-scratch>` | Retrieval and generation as two dataset operations: embed a corpus, then search it |
| {doc}`Feature engineering <feature-engineering>` | A model-ready feature matrix from a raw table, built with fitted preprocessors |
| {doc}`A distributed training pipeline <distributed-training-pipeline>` | A loader that keeps data-parallel PyTorch ranks fed rather than starved |

## See also

- {doc}`/ml/index`: the guide behind these tutorials, one capability per page.
- {doc}`/cookbook/ml/index`: shorter, focused ML recipes.

```{toctree}
:hidden:

batch-inference
rag-from-scratch
feature-engineering
distributed-training-pipeline
```
