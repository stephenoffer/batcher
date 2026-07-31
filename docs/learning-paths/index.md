# Learning paths

This section holds four ordered reading lists, one per role. Pick the path that matches your job, follow it top to bottom, and you land on the parts of Batcher your work actually uses.

Each path is a sequence through the guides and examples, with a few runnable scripts dropped in, rather than new material of its own.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Data engineer
:link: data-engineer
:link-type: doc
Build pipelines. Read a source, reshape it, join it, aggregate it, write the result.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Data scientist
:link: data-scientist
:link-type: doc
Interactive analysis: expressions, SQL, and group-by aggregations.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` ML engineer
:link: ml-engineer
:link-type: doc
Batch inference, embeddings, and GPU execution.
:::

:::{grid-item-card} {octicon}`server;1.1em` Platform engineer
:link: platform-engineer
:link-type: doc
Configure the engine, bound its memory, and keep it running under load.
:::
::::

## See also

- {doc}`../getting-started/index`: install and run a first query before starting a path.
- {doc}`../user-guide/index`: the capability-by-capability reference each path points into.
- {doc}`../cookbook/index`: runnable code for the workloads the paths describe.
- {doc}`../migration/index`: start here instead if you are porting existing code.

```{toctree}
:hidden:

data-engineer
data-scientist
ml-engineer
platform-engineer
```
