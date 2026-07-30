# Cookbook

100 runnable recipes covering the public API, grouped by the surface they exercise.

Batcher's API is wide: over a thousand public names, and more than five hundred methods
reachable from an `Expr` alone. These pages exist so that surface is *shown* rather than
merely listed. Each recipe is a complete script you can run unchanged.

Every script builds its own in-memory data and asserts on its own output, so there are no
fixtures to set up and nothing to configure. `tests/docs/test_examples.py` executes all of
them on every test run.

| Section | Recipes | Covers |
|---|---|---|
| {doc}`expressions/index` | 34 | Recipes for the expression API: strings, temporal, lists, JSON, selectors, and the scalar algebra |
| {doc}`metrics/index` | 14 | Model and text metrics computed as aggregate expressions, so evaluation is a `select` over the table rather than a pull into pandas |
| {doc}`statistics/index` | 6 | Summary statistics, robust dispersion, distribution shape, association, and A/B test inference, all as aggregates in the engine |
| {doc}`ml/index` | 16 | Preprocessors, estimators, model selection, batch inference, and vector search on the `batcher.ml` surface |
| {doc}`io/index` | 6 | Reading and writing: Parquet, text formats, Arrow interop, save modes, streaming reads, and the source/sink registries |
| {doc}`governance/index` | 3 | Column masking, row filters, PII transforms, and column lineage, all applied as plan rewrites |
| {doc}`dataset/index` | 14 | The Dataset verbs: joins, grouping, reshaping, deduplication, sampling, profiling, null handling, and the `meta` accessor |
| {doc}`operations/index` | 7 | Running the engine: configuration, plan inspection, memory, observability, error handling, and streaming basics |

```{toctree}
:hidden:

expressions/index
metrics/index
statistics/index
ml/index
io/index
governance/index
dataset/index
operations/index
```
