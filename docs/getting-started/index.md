# Getting started

Install Batcher, run a query, then pick up the one idea the whole API rests on. A
`Dataset` is a lazy handle to a plan. Nothing runs until you ask for results.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Installation
:link: installation
:link-type: doc
`pip install batcher-engine`. Optional extras cover cloud storage, ML backends, and
file formats.
:::

:::{grid-item-card} {octicon}`rocket;1.1em` Quickstart
:link: quickstart
:link-type: doc
Build a dataset. Filter it, join it, aggregate it: a whole pipeline in a few lines.
:::

:::{grid-item-card} {octicon}`light-bulb;1.1em` Core concepts
:link: concepts/index
:link-type: doc
Why datasets are lazy and immutable, why expressions run in Rust, and where the Python
control plane hands off to the Rust data plane.
:::
::::

```{toctree}
:hidden:

installation
quickstart
concepts/index
```
