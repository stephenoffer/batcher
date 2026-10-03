# Tutorials

Ten tutorials, each building one complete pipeline from the first line of data to a checked result. New to Batcher? Start with {doc}`your first pipeline </getting-started/tutorials/foundations/first-pipeline>`, then skip straight to your workload.

Every tutorial is built from the same few moves:

```python
import batcher as bt

ds = bt.from_pydict({"city": ["sf", "nyc", "sf"], "sales": [10.0, 20.0, 30.0]})
out = ds.group_by("city").agg(total=bt.col("sales").sum()).sort("city")
print(out.to_pydict())
# {'city': ['nyc', 'sf'], 'total': [20.0, 40.0]}
```

![A decision tree. If you're new to Batcher, start with Your first pipeline, which teaches the lazy, expression-first model; from there, a reader who wants an order goes to the learning paths, four roles in order. Everyone else, and new readers afterwards, picks by what they are building. Foundations holds From SQL to DataFrames, to bring SQL habits, and Optimizing a slow query, to find why a query is slow. Data pipelines holds Building a lakehouse for a transactional table, A streaming pipeline for a source that never ends, and Synthetic data generation for test data first. Machine learning holds Batch inference for a model over a corpus, RAG from scratch for retrieval and generation, Distributed training pipeline to feed DDP ranks, and Feature engineering for a feature matrix.](/_static/diagrams/tutorial_chooser.svg)

:::{tip}
Every code block on these pages runs in the docs test suite, so the examples always match the API.
:::

## Pick a tutorial

The following table maps what you want to do to the tutorial that does it:

| If you want to | Read |
|---|---|
| Learn the API from nothing | {doc}`Your first pipeline </getting-started/tutorials/foundations/first-pipeline>` |
| Bring SQL habits with you | {doc}`From SQL to DataFrames </getting-started/tutorials/foundations/sql-to-dataframe>` |
| Find out why a query is slow | {doc}`Optimizing a slow query </getting-started/tutorials/foundations/optimizing-a-slow-query>` |
| Build a transactional table | {doc}`Building a lakehouse </getting-started/tutorials/pipelines/building-a-lakehouse>` |
| Handle a source that never ends | {doc}`A streaming pipeline </getting-started/tutorials/pipelines/streaming-pipeline>` |
| Build test data first | {doc}`Synthetic data generation </getting-started/tutorials/pipelines/synthetic-data-generation>` |
| Run a model over a corpus | {doc}`Batch inference </getting-started/tutorials/ml/batch-inference>` |
| Build retrieval and generation | {doc}`RAG from scratch </getting-started/tutorials/ml/rag-from-scratch>` |
| Feed DDP ranks | {doc}`Distributed training pipeline </getting-started/tutorials/ml/distributed-training-pipeline>` |
| Make a feature matrix | {doc}`Feature engineering </getting-started/tutorials/ml/feature-engineering>` |

{doc}`Foundations <foundations/index>` teaches the engine, {doc}`data pipelines <pipelines/index>` covers lakehouse tables, streams, and generated data, and {doc}`machine learning <ml/index>` covers inference, retrieval, training input, and features. Prefer a fixed order? The {doc}`learning paths <paths/index>` sequence everything for four roles.

Already know the API? The {doc}`cookbook </cookbook/index>` solves one problem per page.

## See also

- {doc}`Quickstart </getting-started/quickstart>`: shorter than a tutorial, if you want to see the shape of the API first.
- {doc}`Core concepts </getting-started/concepts/index>`: the ideas the tutorials put to work.
- {doc}`User guide </user-guide/index>`: every capability, one topic per page.
- {doc}`API reference </api/index>`: every public name.

```{toctree}
:hidden:

foundations/index
pipelines/index
ml/index
paths/index
```
