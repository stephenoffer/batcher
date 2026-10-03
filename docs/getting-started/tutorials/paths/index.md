# Learning paths

Four ordered reading lists, one per role. Pick yours and follow it top to bottom to cover exactly the parts of Batcher your work uses. The paths share a common core, shown here:

![A matrix of twelve topics against the four paths. Data engineer covers Your first pipeline, reading and writing data, expressions and filtering, aggregations and window functions, joins, lakehouse tables and data quality, cloud storage, and best practices and troubleshooting. Data scientist covers core concepts, expressions and filtering, aggregations and window functions, and SQL. ML engineer covers Your first pipeline and inference, features, and GPUs. Platform engineer covers installation and configuration, cloud storage, and best practices and troubleshooting. Every path starts at Getting started and ends at an API reference.](/_static/diagrams/learning_paths_matrix.svg)

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Data engineer
:link: /getting-started/tutorials/paths/data-engineer
:link-type: doc
Build pipelines. Read a source, reshape it, join it, aggregate it, write the result.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Data scientist
:link: /getting-started/tutorials/paths/data-scientist
:link-type: doc
Interactive analysis: expressions, SQL, and group-by aggregations.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` ML engineer
:link: /getting-started/tutorials/paths/ml-engineer
:link-type: doc
Batch inference, embeddings, and GPU execution.
:::

:::{grid-item-card} {octicon}`server;1.1em` Platform engineer
:link: /getting-started/tutorials/paths/platform-engineer
:link-type: doc
Configure the engine, bound its memory, and keep it running under load.
:::
::::

## See also

- {doc}`/getting-started/index`: install and run a first query before starting a path.
- {doc}`/user-guide/index`: the capability-by-capability reference each path points into.
- {doc}`/cookbook/index`: runnable code for the workloads the paths describe.
- {doc}`/getting-started/migration/index`: start here instead if you are porting existing code.

```{toctree}
:hidden:

data-engineer
data-scientist
ml-engineer
platform-engineer
```
