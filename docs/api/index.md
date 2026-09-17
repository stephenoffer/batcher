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

## Find what you need

The surface is written up three ways, for three kinds of question:

| Start with | When you want |
|---|---|
| {doc}`/api/reference` | The one-page cheat sheet of the calls you reach for most |
| An area page below | A runnable example plus the full surface for one area |
| {doc}`/api/complete/index` | The generated listing of every symbol without an area page |

## Browse by area

The area pages fall into three groups, each with its own index:

| Group | Pages | Covers |
|---|---|---|
| {doc}`/api/relational/index` | 11 | `Dataset`, expressions, accessors, functions, SQL, geospatial, graph, and IO |
| {doc}`/api/models/index` | 5 | The `.ml` accessor, preprocessors, estimators, metrics, and statistics |
| {doc}`/api/operations/index` | 4 | Configuration, streaming, governance, and the typed exceptions |

Tuning lives next door: {doc}`/configuration/index` covers every `Config` field with its default.

## See also

- {doc}`/user-guide/index`: the task-oriented guides these pages are the reference for.
- {doc}`/getting-started/quickstart`: the shortest path to a running query.
- {doc}`/getting-started/migration/index`: the equivalent spelling if you know another engine's API.
- {doc}`/cookbook/index`: a runnable recipe for the call, when a signature is not enough.
- {doc}`/agents`: the same surface packaged as instructions for a coding agent.

```{toctree}
:hidden:
:caption: Reference

reference
complete/index
```

```{toctree}
:hidden:
:caption: By area

relational/index
models/index
operations/index
```
