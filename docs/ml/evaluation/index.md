# Measure what the model does

A model is only as trustworthy as the numbers around it, and those numbers are queries.
Every metric here is an expression the engine evaluates, so a report over a billion scored
rows is one pass and the same report *per segment* costs the same.

- {doc}`/ml/evaluation/evaluation`: metrics, per-segment scoring, and the diagnostic tables.
- {doc}`/ml/evaluation/statistics-and-drift`: statistical expressions, input drift, and honest splits.

```{toctree}
:hidden:

evaluation
statistics-and-drift
```
