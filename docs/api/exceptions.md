# Errors

Batcher raises typed exceptions so failures are specific and actionable. These
types live in `batcher._internal.errors`. That module is internal: there is no
public `batcher.exceptions` module to import from, and the names are not part of
the stable public API. You catch these errors by type when they surface, but you
do not construct them yourself.

In practice you handle them with `try` / `except`, usually catching the base type.

```python
import batcher as bt

ds = bt.from_pydict({"a": [1, 2, 3]})

try:
    bad = ds.select(bt.col("does_not_exist"))
    bad.to_pydict()
except Exception as exc:
    print(type(exc).__name__)
# PlanError
```

## The error types you may see

Every error shares a common base, so a single `except` can catch them all, or you
can catch a specific type when you want to react differently.

| Error | Raised when |
| --- | --- |
| `PlanError` | The plan or schema is invalid (an unknown column, a type mismatch). Raised at build time, before execution. |
| `ExecutionError` | An operator fails at runtime inside the engine. |
| `OptimizationError` | The optimizer cannot produce a valid physical plan. |
| `CompileError` | JIT compilation of a pipeline fails. The interpreter remains as a fallback, so this is rare. |
| `ResourceError` | The resource manager cannot satisfy a memory or credit request. |
| `BackpressureAbort` | Execution is aborted because backpressure could not be relieved. |
| `IOError` | A source or sink fails to read, write, list, or open a path. |
| `FormatError`, `BackendError`, `CommitError`, `TransportError` | Lower-level IO, backend, write-commit, and shuffle failures. |

`PlanError` is the one most user code encounters, because it is raised eagerly
when you build an invalid plan rather than when you execute it.

## Catching errors

Because the types are internal, the robust pattern is to catch broadly and
inspect the message, or to import the base type from its internal location if you
need to branch on it.

```python
from batcher._internal.errors import BatcherError

try:
    ds.select(bt.col("missing")).to_pydict()
except BatcherError as exc:
    print(f"query failed: {exc}")
# query failed: ...
```

Catching `BatcherError` covers every Batcher-specific failure while letting
unrelated exceptions (a bug in your own batch function, for example) propagate.

## Next steps

- [Dataset](dataset.md): the operations that raise these errors.
- [Configuration](configuration.md): resource limits that govern `ResourceError`.
