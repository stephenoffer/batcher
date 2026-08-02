# Tutorials

This section holds 10 end-to-end walkthroughs. Each builds a small pipeline against the real API and runs as written.

Start with the first pipeline if you are new. From there the tutorials are independent, so pick the workload you are actually building rather than reading in order.

:::{tip}
Every runnable block on every page below is executed by the test suite on each commit, in page order, sharing one namespace. An example that stopped working is a failing build rather than a paragraph you find out about the hard way. Copy them with that in mind.
:::

## Pick one

| If you want to | Read |
|---|---|
| Learn the API from nothing | {doc}`Your first pipeline </tutorials/foundations/first-pipeline>` |
| Bring SQL habits with you | {doc}`From SQL to DataFrames </tutorials/foundations/sql-to-dataframe>` |
| Find out why a query is slow | {doc}`Optimizing a slow query </tutorials/foundations/optimizing-a-slow-query>` |
| Build a transactional table | {doc}`Building a lakehouse </tutorials/pipelines/building-a-lakehouse>` |
| Handle a source that never ends | {doc}`A streaming pipeline </tutorials/pipelines/streaming-pipeline>` |
| Run a model over a corpus | {doc}`Batch inference </tutorials/ml/batch-inference>` |
| Build retrieval and generation | {doc}`RAG from scratch </tutorials/ml/rag-from-scratch>` |
| Feed DDP ranks | {doc}`Distributed training pipeline </tutorials/ml/distributed-training-pipeline>` |
| Make a feature matrix | {doc}`Feature engineering </tutorials/ml/feature-engineering>` |
| Build test data first | {doc}`Synthetic data generation </tutorials/pipelines/synthetic-data-generation>` |

If you would rather be handed an order than pick one, {doc}`the learning paths <paths/index>` sequence these tutorials together with the guides by role: data engineer, data scientist, ML engineer, or platform engineer.

## Where the cookbook fits

Two sections teach by code, and they differ in what they hold constant:

| Section | One page is | Pick it when |
|---|---|---|
| Tutorials (this section) | One pipeline, built step by step | You are learning the API |
| {doc}`Cookbook <../cookbook/index>` | One surface or one problem, demonstrated | You know roughly what you need |

## See also

- {doc}`Quickstart <../getting-started/quickstart>`: shorter than a tutorial, if you want to see the shape of the thing.
- {doc}`User guide <../user-guide/index>`: the reference-by-topic these tutorials teach from.
- {doc}`Learning paths </tutorials/paths/index>`: these tutorials sequenced by the job you do.
- {doc}`API reference <../api/index>`: every public name.

```{toctree}
:hidden:

foundations/index
pipelines/index
ml/index
paths/index
```
