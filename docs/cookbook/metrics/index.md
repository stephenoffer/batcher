# Measurement cookbook

This section holds 20 runnable recipes that compute metrics and statistics as aggregate expressions, so both describing a column and scoring a model are a `select` over the table rather than a pull into pandas.

That framing is the point of the section. A metric written as an aggregate runs in one pass over a billion scored rows, and the same report *per segment* costs the same as the report overall.

Every page embeds a complete, self-contained script from the [`examples/metrics/`](https://github.com/batcher/batcher/tree/main/examples/metrics) directory, and `tests/docs/test_examples.py` runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Group | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/metrics/model/index` | 5 | Predictions against labels, for classification and regression |
| {doc}`/cookbook/metrics/text/index` | 8 | Generated text, with and without a reference |
| {doc}`/cookbook/metrics/embeddings` | 1 | Corpus-level embedding metrics, in aggregate |
| {doc}`/cookbook/metrics/statistics/index` | 6 | Summary statistics, dispersion, distribution shape, association, and A/B inference |

Read the statistics group first when the question is about a column, and the metric groups when the question is about a prediction. They share one machinery: every entry lowers to an aggregate the engine evaluates in a single pass, which is why a per-segment report costs what the overall report costs.

## See also

- {doc}`/ml/evaluation/evaluation`: the guide to scoring a model and reading the result per segment.
- {doc}`/ml/retrieval/llm-evaluation`: the same monitors applied to a generation pipeline.
- {doc}`/api/models/metrics`: the complete metric-function reference.
- {doc}`/cookbook/dataset/inspecting/profiling`: the first pass over an unfamiliar table.

```{toctree}
:hidden:

model/index
text/index
embeddings
statistics/index
```
