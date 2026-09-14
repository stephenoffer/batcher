# API reference

The API is small and lazy. A {py:class}`Dataset <batcher.Dataset>` is a handle to a plan: you chain expression-based operations onto it, and nothing runs until a terminal call returns Arrow or writes to a sink. Everything reachable from `import batcher as bt` is documented here.

## Three ways in

The same surface is written up three times, for three different questions:

| Start with | When |
|---|---|
| {doc}`/api/reference` | You want the one-page cheat sheet of the calls you reach for most |
| The area pages below | You want a runnable example plus the full surface for one area |
| {doc}`/api/complete/index` | You want the backstop index of every symbol without an area page |

## By area

The area pages are grouped into three sections, each with its own index:

| Group | Pages | Covers |
|---|---|---|
| {doc}`/api/relational/index` | 11 | `Dataset`, expressions, accessors, functions, SQL, geospatial, graph, and IO |
| {doc}`/api/models/index` | 5 | The `.ml` accessor, preprocessors, estimators, metrics, and statistics |
| {doc}`/api/operations/index` | 4 | Configuration, streaming, governance, and the typed exceptions |

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
