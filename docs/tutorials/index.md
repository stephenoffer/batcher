# Tutorials

Worked, end-to-end walkthroughs. Each builds a small pipeline against the real API and
runs as written. Start with the first pipeline; from there, pick the workload you are
actually building.

:::{tip}
Every runnable block on every page below is executed by the test suite on each commit, in
page order, sharing one namespace. An example that stopped working is a failing build rather
than a paragraph you find out about the hard way. Copy them with that in mind.
:::

## Pick one

| If you want to | Read |
|---|---|
| Learn the API from nothing | [Your first pipeline](first-pipeline.md) |
| Bring SQL habits with you | [From SQL to DataFrames](sql-to-dataframe.md) |
| Find out why a query is slow | [Optimizing a slow query](optimizing-a-slow-query.md) |
| Build a transactional table | [Building a lakehouse](building-a-lakehouse.md) |
| Handle a source that never ends | [A streaming pipeline](streaming-pipeline.md) |
| Run a model over a corpus | [Batch inference](batch-inference.md) |
| Build retrieval and generation | [RAG from scratch](rag-from-scratch.md) |
| Feed DDP ranks | [Distributed training pipeline](distributed-training-pipeline.md) |
| Make a feature matrix | [Feature engineering](feature-engineering.md) |
| Build test data first | [Synthetic data generation](synthetic-data-generation.md) |

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Your first pipeline
:link: first-pipeline
:link-type: doc
Build a dataset, derive a column, roll it up with an aggregate, then point the same code
at files.
:::

:::{grid-item-card} {octicon}`code-square;1.1em` From SQL to DataFrames
:link: sql-to-dataframe
:link-type: doc
Rewrite one query both ways, then prove they compile to the same plan.
:::

:::{grid-item-card} {octicon}`stopwatch;1.1em` Optimizing a slow query
:link: optimizing-a-slow-query
:link-type: doc
Read the plan, measure the operators, find the real problem, fix it.
:::

:::{grid-item-card} {octicon}`database;1.1em` Building a lakehouse
:link: building-a-lakehouse
:link-type: doc
Bronze to gold on Delta: atomic commits, upserts, time travel, file skipping.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` A streaming pipeline
:link: streaming-pipeline
:link-type: doc
Windows, watermarks, triggers, and a checkpoint that survives a restart.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` Batch inference
:link: batch-inference
:link-type: doc
Run a model over Arrow batches with the `.ml` accessor.
:::

:::{grid-item-card} {octicon}`search;1.1em` RAG from scratch
:link: rag-from-scratch
:link-type: doc
Chunk, embed, retrieve, generate: four Dataset operations, no framework.
:::

:::{grid-item-card} {octicon}`server;1.1em` Distributed training pipeline
:link: distributed-training-pipeline
:link-type: doc
Feed DDP ranks a balanced, deterministic, resumable stream of tensors.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Synthetic data
:link: synthetic-data-generation
:link-type: doc
Build test datasets in memory with Python and expressions.
:::

:::{grid-item-card} {octicon}`gear;1.1em` Feature engineering
:link: feature-engineering
:link-type: doc
Build a model-ready feature matrix with fit/transform preprocessors and `Chain`.
:::
::::

## See also

- [Quickstart](../getting-started/quickstart.md): shorter than a tutorial, if you want to see
  the shape of the thing.
- [User guide](../user-guide/index.md): the reference-by-topic these tutorials teach from.
- [Examples](../examples/index.md): short recipes, one problem each, when you already know
  the API.
- [Learning paths](../learning-paths/index.md): these tutorials sequenced by the job you do.
- [API reference](../api/index.md): every public name.

```{toctree}
:hidden:

first-pipeline
sql-to-dataframe
optimizing-a-slow-query
building-a-lakehouse
streaming-pipeline
batch-inference
rag-from-scratch
distributed-training-pipeline
synthetic-data-generation
feature-engineering
```
