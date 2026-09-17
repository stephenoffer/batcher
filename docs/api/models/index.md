# Models and measurement

This section is the reference for everything that fits a model or scores one, from the `.ml` accessor down to the individual metric functions. Preprocessors fit over a `Dataset` and transform any other, and most metrics are ordinary aggregates, so scoring per segment is one `group_by(...).agg(...)` away. Five pages split it by task:

| Page | Covers |
|---|---|
| {doc}`/api/models/ml` | The `.ml` accessor, plus the LLM, serving, loader, and vector surfaces |
| {doc}`/api/models/preprocessors` | The fit/transform estimators and {py:class}`Chain <batcher.ml.preprocessors.Chain>` |
| {doc}`/api/models/ml-models` | Tabular scoring, plus the in-engine estimators |
| {doc}`/api/models/metrics` | Scoring and statistical aggregates |
| {doc}`/api/models/ml-statistics` | Drift, fairness, resampling, and cross-validation |

```{toctree}
:hidden:

ml
preprocessors
ml-models
metrics
ml-statistics
```
