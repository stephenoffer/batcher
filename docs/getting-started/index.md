# Getting started

Install Batcher, run a query, then pick up the one idea the whole API rests on. A
{py:class}`Dataset <batcher.Dataset>` is a lazy handle to a plan. Nothing runs until you ask for results.

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

:::{grid-item-card} {octicon}`arrow-switch;1.1em` Coming from another tool
:link: migration/index
:link-type: doc
Spark, pandas, Polars, DuckDB, and Daft translated verb by verb, ending in a check that
the port returns the same rows.
:::
::::

## Where to go next

Once a query runs, the docs split by what you are trying to do. Reach for
{doc}`../tutorials/index` if you want to be walked through a complete pipeline,
{doc}`../user-guide/index` if you want one capability at a time, and
{doc}`../cookbook/index` if you would rather start from working code and change it.

## See also

- {doc}`/tutorials/paths/index`: an ordered reading list for your role.
- {doc}`/getting-started/migration/index`: the verb-by-verb mapping if you are coming from Spark,
  pandas, Polars, DuckDB, or Daft.
- {doc}`../api/reference`: the one-page cheat sheet to keep open while you work.
- {doc}`/user-guide/operate/running/troubleshooting`: what to read when the first query misbehaves.

```{toctree}
:hidden:

installation
quickstart
concepts/index
migration/index
```
