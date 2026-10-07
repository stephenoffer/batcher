# Getting started

You write DataFrame code or SQL in Python. Batcher's Rust engine runs it over Apache Arrow, on a laptop or on a Ray cluster. This section gets you from `pip install` to a working pipeline in a few minutes.

## Your first query

Install, then run:

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

That's the whole shape of a Batcher program. Build a dataset, chain lazy steps, then ask for the result. Here is the same query in SQL:

```python
query = "SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region ORDER BY region"
print(bt.sql(query, orders=orders).to_pydict())
# {'region': ['east', 'west'], 'revenue': [80.0, 165.0]}
```

Real files change nothing else. A write replaces existing output unless you say otherwise, so this one goes to a fresh temporary directory and passes `mode="error"`, which refuses to touch a path that already exists:

```python
import os
import tempfile

out = os.path.join(tempfile.mkdtemp(), "orders.parquet")
orders.write.parquet(out, mode="error")
print(bt.read.parquet(out).count())
# 3
```

The engine streams Arrow batches across every core and spills to disk under memory pressure, so your input can be larger than RAM. Scaling out to a Ray cluster takes one argument:

```python
# docs: skip
totals.collect(distributed=True, num_workers=8)
```

## Start here

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

The {doc}`tutorials <tutorials/index>` build complete pipelines: a first ETL job, a lakehouse, a stream, batch inference. The {doc}`user guide </user-guide/index>` takes one capability per page. If you'd rather start from working code, open the {doc}`cookbook </cookbook/index>`, and the {doc}`learning paths <tutorials/paths/index>` order the pages by role. Keep {doc}`/api/reference` open while you work. {doc}`/ml/index` covers embeddings, batch inference, and training data on the same engine, and {doc}`/user-guide/operate/running/troubleshooting` helps when a first query misbehaves. For speed, see {doc}`/benchmarks/index`.

```{toctree}
:hidden:

install/index
quickstart
tour
concepts/index
tutorials/index
migration/index
```
