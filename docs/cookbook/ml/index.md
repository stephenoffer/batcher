# Machine learning cookbook

This section holds 27 recipes for the `batcher.ml` surface, ordered the way a model gets built: prepare the features, fit something, check it, then run it. The last group is complete pipelines rather than single calls.

The first four groups each embed one self-contained script that builds its own in-memory data and asserts on its own output, run end to end by `tests/docs/test_examples.py`. A pipeline page is a sequence of shorter blocks instead, executed in document order by `tests/docs/test_doc_examples.py`, with the stages that need a GPU or a cloud bucket shown but not run. Either way a page that stops matching the engine fails the suite rather than drifting.

| Group | Pages | Covers |
|---|---|---|
| {doc}`/cookbook/ml/preprocessing/index` | 7 | Scalers, encoders, imputers, binning, chaining, and feature construction |
| {doc}`/cookbook/ml/estimators/index` | 4 | Linear models, GLMs, classifiers, clustering, and decomposition |
| {doc}`/cookbook/ml/validation/index` | 3 | Cross-validation, class imbalance, and outliers |
| {doc}`/cookbook/ml/inference/index` | 2 | Batch inference and vector search |
| {doc}`/cookbook/ml/pipelines/index` | 11 | Whole workloads: embeddings, captioning, transcription, RAG, and training data |

## See also

- {doc}`/ml/index`: the ML guide these pages are the short form of.
- {doc}`/api/models/ml`: the {py:obj}`ds.ml <batcher.Dataset.ml>` accessor and the `batcher.ml` package reference.
- {doc}`/cookbook/metrics/index`: scoring the models fitted here.
- {doc}`/architecture/deep-dives/distribution/gpu-execution`: how the device work underneath is scheduled.

```{toctree}
:hidden:

preprocessing/index
estimators/index
validation/index
inference/index
pipelines/index
```
