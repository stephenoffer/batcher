# Getting started

Batcher is one engine for your data work. You write DataFrame code or SQL in Python, and a compiled Rust engine runs it over Apache Arrow, on a laptop or across a Ray cluster, for tables, text, images, audio, and video alike. This section takes you from `pip install` to a working pipeline in a few minutes, then explains the handful of ideas that make it fast.

## Your first query

Install the package, then run this. It builds a small dataset, aggregates it, and prints the answer:

```bash
pip install batcher-engine
```

```python
import batcher as bt

orders = bt.from_pydict({"region": ["west", "east", "west"], "amount": [120.0, 80.0, 45.0]})
totals = orders.group_by("region").agg(revenue=bt.col("amount").sum()).sort("region")
print(totals.to_pydict())
# {'region': ['east', 'west'], 'revenue': [80.0, 165.0]}
```

That's the whole shape of a Batcher program: build a dataset, chain lazy steps, ask for the result. Swap `from_pydict` for {py:obj}`bt.read <batcher.read>` on a directory of Parquet files and nothing else changes. The engine works through Arrow batches in parallel on every core and spills to disk under memory pressure, so the input can be far larger than RAM. On a Ray cluster, `collect(distributed=True)` runs the same plan across machines.

## Start here

Most readers take these in order. Skip ahead if you already know the part a card covers.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Install
:link: install/index
:link-type: doc
One wheel with the compiled engine inside. Add extras for Ray, object stores, lakehouse tables, ML backends, and file formats.
:::

:::{grid-item-card} {octicon}`rocket;1.1em` Quickstart
:link: quickstart
:link-type: doc
Filter, join, aggregate, switch to SQL, and write Parquet. Every example runs as written.
:::

:::{grid-item-card} {octicon}`telescope;1.1em` A tour of the engine
:link: tour
:link-type: doc
One runnable example per capability, on one page: SQL, streaming, media, models, vectors, lakehouse, geospatial, and graphs.
:::

:::{grid-item-card} {octicon}`light-bulb;1.1em` Core concepts
:link: concepts/index
:link-type: doc
Lazy plans, expressions that run in Rust, mergeable operators that scale out, and an optimizer that learns from what it measures.
:::

:::{grid-item-card} {octicon}`arrow-switch;1.1em` Coming from another tool
:link: migration/index
:link-type: doc
Spark, pandas, Polars, DuckDB, Daft, and Ray Data translated verb by verb, ending in a check that the port returns the same rows.
:::
::::

## Where to go next

Once a query runs, pick the path that matches how you like to learn. The {doc}`tutorials <tutorials/index>` walk you through complete pipelines, from a first ETL job to a lakehouse, a streaming job, and batch inference. The {doc}`user guide </user-guide/index>` takes one capability at a time, and the {doc}`cookbook </cookbook/index>` starts you from working code you can change. If you'd rather follow a reading list, the {doc}`learning paths <tutorials/paths/index>` order the pages for data engineers, data scientists, ML engineers, and platform engineers.

Curious how fast it is? The {doc}`benchmarks </benchmarks/index>` page has the correctness-gated results against DuckDB, Polars, Daft, and Spark, with the hardware and the commands to reproduce each one.

## See also

- {doc}`/api/reference`: the one-page cheat sheet to keep open while you work.
- {doc}`/ml/index`: embeddings, batch inference, and training data on the same engine.
- {doc}`/user-guide/operate/running/troubleshooting`: what to read when the first query misbehaves.

```{toctree}
:hidden:

install/index
quickstart
tour
concepts/index
tutorials/index
migration/index
```
