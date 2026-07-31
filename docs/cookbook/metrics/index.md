# Metrics cookbook

This section holds 14 runnable recipes that compute model and text metrics as aggregate expressions, so evaluation is a `select` over the table rather than a pull into pandas.

That framing is the point of the section. A metric written as an aggregate runs in one pass over a billion scored rows, and the same report *per segment* costs the same as the report overall.

Every page embeds a complete, self-contained script from the [`examples/metrics/`](https://github.com/batcher/batcher/tree/main/examples/metrics) directory, and `tests/docs/test_examples.py` runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Group | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/metrics/model/index` | 5 | Predictions against labels, for classification and regression |
| {doc}`/cookbook/metrics/text/index` | 8 | Generated text, with and without a reference |
| {doc}`/cookbook/metrics/embeddings` | 1 | Corpus-level embedding metrics, in aggregate |

## See also

- {doc}`/ml/evaluation/evaluation`: the guide to scoring a model and reading the result per segment.
- {doc}`/ml/retrieval/llm-evaluation`: the same monitors applied to a generation pipeline.
- {doc}`/api/models/metrics`: the complete metric-function reference.
- {doc}`/cookbook/statistics/index`: the general statistical aggregates these are built on.

```{toctree}
:hidden:

model/index
text/index
embeddings
```
