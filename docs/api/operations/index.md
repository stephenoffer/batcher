# Running it

This section is the reference for running Batcher in production: the tunables and how they're set, continuous queries, access policy, and the typed exceptions a failure raises. Every one subclasses `BatcherError`, and several also subclass the matching Python builtin, so existing `except` clauses keep working.

| Page | Covers |
|---|---|
| {doc}`/api/operations/configuration` | How a setting resolves: the `Config` dataclass, the two entry points, and the precedence order |
| {doc}`/api/operations/configuration-reference` | The full listing: every option function, config dataclass, and cache control |
| {doc}`/api/operations/streaming` | Triggers, output modes, query progress, and listeners |
| {doc}`/api/operations/governance` | Row filters, column masks, grants, and lineage |
| {doc}`/api/operations/exceptions` | The typed exceptions, and what raises them |

```{toctree}
:hidden:

configuration
configuration-reference
streaming
governance
exceptions
```
