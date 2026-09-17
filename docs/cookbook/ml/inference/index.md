# Running a model over data

A model earns its keep when it runs over the whole table. These two recipes cover the two shapes that takes: a Python model called on Arrow batches with its weights loaded once, and vector retrieval written as ordinary expressions the engine runs itself.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/ml/inference/batch_inference` | A model over every row, without a Python loop |
| {doc}`/cookbook/ml/inference/vector_search` | Vector search over an embedding column, in the engine |

## See also

- {doc}`/cookbook/ml/pipelines/index`: the same idioms in complete GPU and multimodal pipelines.
- {doc}`/ml/inference/batch-scoring`: filtering before the model, and sizing the batch to the device.
- {doc}`/ml/retrieval/vector-search`: the engine scan, the ANN index, and which corpus size needs which.

```{toctree}
:hidden:

batch_inference
vector_search
```
