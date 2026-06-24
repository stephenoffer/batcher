# Getting started

New to Batcher? Install it, run a first query, and learn the one idea that shapes the
whole API: a `Dataset` is a lazy handle to a plan, and nothing runs until you ask for
results.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Installation
:link: installation
:link-type: doc
`pip install batcher-engine`, plus the optional extras for cloud, ML, and formats.
:::

:::{grid-item-card} {octicon}`rocket;1.1em` Quickstart
:link: quickstart
:link-type: doc
Build, filter, transform, aggregate, and join a dataset in a few lines.
:::

:::{grid-item-card} {octicon}`light-bulb;1.1em` Core concepts
:link: concepts
:link-type: doc
Lazy, immutable datasets, expressions, and the control-plane / data-plane split.
:::
::::

```{toctree}
:hidden:

installation
quickstart
concepts
```
