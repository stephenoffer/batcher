# Tutorials

This section holds 10 end-to-end walkthroughs. Each builds a small pipeline against the real API and runs as written.

Start with the first pipeline if you are new. From there the tutorials are independent, so pick the workload you are actually building rather than reading in order.

:::{tip}
Every runnable block on every page below is executed by the test suite on each commit, in page order, sharing one namespace. An example that stopped working is a failing build rather than a paragraph you find out about the hard way. Copy them with that in mind.
:::

## Pick one

| If you want to | Read |
|---|---|
| Learn the API from nothing | {doc}`Your first pipeline <first-pipeline>` |
| Bring SQL habits with you | {doc}`From SQL to DataFrames <sql-to-dataframe>` |
| Find out why a query is slow | {doc}`Optimizing a slow query <optimizing-a-slow-query>` |
| Build a transactional table | {doc}`Building a lakehouse <building-a-lakehouse>` |
| Handle a source that never ends | {doc}`A streaming pipeline <streaming-pipeline>` |
| Run a model over a corpus | {doc}`Batch inference <batch-inference>` |
| Build retrieval and generation | {doc}`RAG from scratch <rag-from-scratch>` |
| Feed DDP ranks | {doc}`Distributed training pipeline <distributed-training-pipeline>` |
| Make a feature matrix | {doc}`Feature engineering <feature-engineering>` |
| Build test data first | {doc}`Synthetic data generation <synthetic-data-generation>` |

## Where the cookbook fits

Two sections teach by code, and they differ in what they hold constant:

| Section | One page is | Pick it when |
|---|---|---|
| Tutorials (this section) | One pipeline, built step by step | You are learning the API |
| {doc}`Cookbook <../cookbook/index>` | One surface or one problem, demonstrated | You know roughly what you need |

## See also

- {doc}`Quickstart <../getting-started/quickstart>`: shorter than a tutorial, if you want to see the shape of the thing.
- {doc}`User guide <../user-guide/index>`: the reference-by-topic these tutorials teach from.
- {doc}`Learning paths <../learning-paths/index>`: these tutorials sequenced by the job you do.
- {doc}`API reference <../api/index>`: every public name.

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
feature-engineering
synthetic-data-generation
```
