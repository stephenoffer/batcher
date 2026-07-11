# Tutorials

Worked, end-to-end walkthroughs. Each builds a small pipeline against the real API
and runs as written. Start with the first pipeline, then move to the workload that
matches what you are building.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Your first pipeline
:link: first-pipeline
:link-type: doc
Build, transform, aggregate, sort, and collect a dataset, then point the same code at files.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` Batch inference
:link: batch-inference
:link-type: doc
Run a model over Arrow batches with the `.ml` accessor.
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

```{toctree}
:hidden:

first-pipeline
batch-inference
synthetic-data-generation
feature-engineering
```
