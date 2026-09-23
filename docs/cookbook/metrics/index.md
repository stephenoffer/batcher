# Measurement cookbook

In Batcher, a metric is an aggregate expression. Describing a column, scoring a classifier, and checking a batch of LLM output are all a `select` over the table rather than a pull into pandas, and the same report per segment is that expression under a `group_by`, still in one pass.

These 20 recipes cover model scores, generated-text monitors, embedding health checks, and the statistics you run before trusting a summary.

Every page embeds a complete, self-contained script, from [`examples/metrics/`](https://github.com/stephenoffer/batcher/tree/main/examples/metrics) for the model and text groups and [`examples/statistics/`](https://github.com/stephenoffer/batcher/tree/main/examples/statistics) for the statistics group. [`tests/docs/test_examples.py`](https://github.com/stephenoffer/batcher/blob/main/tests/docs/test_examples.py) runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Group | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/metrics/model/index` | 6 | Predictions against labels, and the health of an embedding column |
| {doc}`/cookbook/metrics/text/index` | 8 | Generated text, with and without a reference |
| {doc}`/cookbook/metrics/statistics/index` | 6 | Summary statistics, dispersion, distribution shape, association, and A/B inference |

Read the statistics group first when the question is about a column, and the metric groups when the question is about a prediction. They share one machinery, so nothing here changes when you move between them.

## See also

- {doc}`/ml/evaluation/evaluation`: the guide to scoring a model and reading the result per segment.
- {doc}`/ml/retrieval/llm-evaluation`: the same monitors applied to a generation pipeline.
- {doc}`/api/models/metrics`: the complete metric-function reference.
- {doc}`/cookbook/dataset/inspecting/profiling`: the first pass over an unfamiliar table.

```{toctree}
:hidden:

model/index
text/index
statistics/index
```
