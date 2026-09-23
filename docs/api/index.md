# API reference

This section is the reference for everything reachable from `import batcher as bt`. The surface is small on purpose. You build a {py:class}`Dataset <batcher.Dataset>`, chain expression-based operations onto it, and nothing runs until a terminal call hands back Arrow or writes to a sink. Every expression lowers to the Rust data plane, so the Python you write describes the work and never touches a row.

```python
import batcher as bt

ds = bt.from_pydict({"city": ["Oslo", "Lima", "Oslo"], "temp": [3.5, 19.0, 5.5]})
out = ds.filter(bt.col("temp") > 0).group_by("city").agg(avg_temp=bt.col("temp").mean()).sort("city")

print(out.to_pydict())
# {'city': ['Lima', 'Oslo'], 'avg_temp': [19.0, 4.5]}
```

That chain is the whole model. The same plan runs on one core, across every core, or on a Ray cluster with {py:meth}`collect(distributed=True) <batcher.Dataset.collect>`, and the pages below document each piece of it.

## Three ways in

The same surface is written up three ways, because looking a name up, learning what it does, and finding out it exists are three different jobs.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`search;1.1em` Quick reference
:link: /api/reference
:link-type: doc
One page, every call you reach for daily, as a lookup table. Start here when you know the name.
:::

:::{grid-item-card} {octicon}`book;1.1em` By area
:link: /api/relational/index
:link-type: doc
A runnable example plus the semantics, one page per part of the engine. Start here when you know the job.
:::

:::{grid-item-card} {octicon}`list-unordered;1.1em` Every symbol
:link: /api/symbols/index
:link-type: doc
The full generated listing, one page per object family, one page per method. Start here when you want the signature.
:::
::::

## Browse by area

The area pages are grouped by what the call is for. Each group has its own index.

| Group | Pages | Covers |
| --- | --- | --- |
| {doc}`/api/relational/index` | 10 | `Dataset`, expressions, functions, SQL, sessions and catalogs, IO, and the geospatial, rigid-body, and graph libraries |
| {doc}`/api/accessors/index` | 5 | The typed namespaces an expression carries: `.str`, `.dt`, the nested ones, `.image`/`.audio`/`.video`, and `.seq` |
| {doc}`/api/models/index` | 5 | The `.ml` accessor, preprocessors, estimators, metrics, and statistics |
| {doc}`/api/operations/index` | 5 | Configuration, streaming, governance, and the typed exceptions |

Tuning lives next door: {doc}`/configuration/index` covers every `Config` field with its default.

## See also

- {doc}`/user-guide/index`: the task-oriented guides these pages are the reference for.
- {doc}`/getting-started/quickstart`: the shortest path to a running query.
- {doc}`/getting-started/migration/index`: the equivalent spelling if you know another engine's API.
- {doc}`/cookbook/index`: a runnable recipe for the call, when a signature is not enough.
- {doc}`/agents`: the same surface packaged as instructions for a coding agent.
- {doc}`/getting-started/concepts/glossary`: the vocabulary these signatures are written in.

```{toctree}
:hidden:
:caption: Reference

reference
symbols/index
```

```{toctree}
:hidden:
:caption: By area

relational/index
accessors/index
models/index
operations/index
```
