# Machine learning

This section covers running models where the data already is: inference, preprocessing, evaluation, embeddings and retrieval, serving, and feeding a training loop.

Most ML pipelines pay a tax at the seam: a query engine produces rows, something
converts them, and a separate system runs the model. Batcher removes the seam. The `ml`
accessor hands your Python functions and models whole Arrow batches rather than one row
at a time, and it places that work on GPUs and across worker actors for you. The same
`Dataset` you filtered and joined is the one the model reads.

That has a practical consequence worth stating up front. Because inference is an
operator rather than a separate job, it streams. A batch-scoring run over more data than
memory is the ordinary case, not something you build around.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`cpu;1.1em` Run a model
:link: /ml/inference/inference
:link-type: doc
Batched inference over Arrow, on CPU or GPU.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Prepare the data
:link: /ml/preparing/preprocessors/index
:link-type: doc
Feature transforms, media decode, tokenization.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Measure the model
:link: /ml/evaluation/evaluation
:link-type: doc
Metrics, per-segment scoring, drift.
:::

:::{grid-item-card} {octicon}`search;1.1em` Embeddings and retrieval
:link: /ml/retrieval/embeddings
:link-type: doc
Encode, index, search, and generate.
:::

:::{grid-item-card} {octicon}`workflow;1.1em` Feed a training loop
:link: /ml/training/distributed-training
:link-type: doc
Sharding that stays balanced, and resumes where it stopped.
:::
::::

## In this section

| Group | Pages | Covers |
|---|---|---|
| {doc}`/ml/inference/index` | 6 | The core loop, model reuse, GPU placement, and the offline scoring job |
| {doc}`/ml/preparing/index` | 2 | Media decode into tensor columns, and tokenization as a stage |
| {doc}`/ml/evaluation/index` | 2 | Metrics, per-segment scoring, drift, and honest splits |
| {doc}`/ml/retrieval/index` | 6 | Encoding, vector search, RAG, and the LLM surface |
| {doc}`/ml/training/index` | 5 | Serving patterns, sharded training feeds, and loaders |

## See also

- {doc}`../user-guide/index`: the relational half of the pipeline that feeds all of this.
- {doc}`/user-guide/transform/columns/udfs`: batch UDFs generally, of which model inference is one case.
- {doc}`/api/models/ml`: the reference for the `.ml` accessor and the `batcher.ml` package.
- {doc}`/cookbook/ml/pipelines/index`: runnable recipes for the workloads above.
- {doc}`/architecture/deep-dives/distribution/gpu-execution`: how device work is actually scheduled.
- {doc}`../benchmarks/index`: the measured throughput behind the claims here.

```{toctree}
:hidden:

inference/index
preparing/index
evaluation/index
retrieval/index
training/index
```
