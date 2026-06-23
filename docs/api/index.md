# API

The Batcher API is small and lazy. You build a `Dataset` from a source, transform
it with expression-based operations, and execute it with a terminal operation that
returns Arrow or writes to a sink. Everything reachable from `import batcher as bt`
is documented in this section.

Start with the quick reference for a one-page map of the surface, then drill into
the page for the area you are working in.

```{toctree}
:maxdepth: 1

reference
dataset
expressions
sql
configuration
ml
exceptions
```
