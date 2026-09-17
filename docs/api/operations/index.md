# Running it

This section is the reference for running Batcher in production: the tunables and how they're set, continuous queries, access policy, and the typed exceptions a failure raises. Every one subclasses `BatcherError`, and several also subclass the matching Python builtin, so existing `except` clauses keep working.

| Page | Covers |
|---|---|
| {doc}`/api/operations/configuration` | The tunables, and how they are set |
| {doc}`/api/operations/streaming` | Triggers, output modes, query progress, and listeners |
| {doc}`/api/operations/governance` | Row filters, column masks, grants, and lineage |
| {doc}`/api/operations/exceptions` | The typed exceptions, and what raises them |

```{toctree}
:hidden:

configuration
streaming
governance
exceptions
```
