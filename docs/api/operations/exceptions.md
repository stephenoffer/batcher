# Errors

Batcher raises typed exceptions, so a failure names what went wrong and what to catch. They're reachable straight from the top level, so `except bt.BatcherError` works without importing anything else:

```{eval-rst}
.. currentmodule:: batcher

.. autoexception:: BatcherError
.. autoexception:: PlanError
.. autoexception:: ColumnNotFoundError
.. autoexception:: ConfigError
.. autoexception:: MissingDependencyError
.. autoexception:: AccessDeniedError
.. autoexception:: ExecutionError
.. autoexception:: OptimizationError
.. autoexception:: CompileError
.. autoexception:: ResourceError
.. autoexception:: IOError
.. autoexception:: FormatError
.. autoexception:: CommitError
.. autoexception:: SchemaError
.. autoexception:: DataQualityError
.. autoexception:: BackendError
.. autoexception:: TransportError
```

`BatcherError` is the root every other Batcher error subclasses, so catching it covers them all. Several also subclass a builtin so existing handlers keep working: `PlanError`, `ConfigError`, and `DataQualityError` are each a `ValueError`; `ColumnNotFoundError` is a `KeyError` and carries the missing `.column`; `MissingDependencyError` is an `ImportError` and carries the `.install` hint for the extra to install; `AccessDeniedError` is a `PermissionError`.

In practice you handle them with `try` and `except`, usually catching the base type.

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
| `DataQualityError` | A `ds.dq...fail()` expectation has violating rows. Carries the per-constraint counts. |
| `AccessDeniedError` | A principal may select no column of a governed table. A *column* it cannot select is instead absent, surfacing as `PlanError`. |
| `FormatError`, `BackendError`, `CommitError`, `TransportError` | Lower-level IO, backend, write-commit, and shuffle failures. |

`PlanError` is the one most user code encounters, because it's raised eagerly when you build an invalid plan rather than when you execute it.

## Catching errors

Because the types are internal, the safe pattern is to catch broadly and
inspect the message, or to import the base type from its internal location if you
need to branch on it.

```python
import batcher as bt

ds = bt.from_pydict({"a": [1, 2, 3]})

try:
    ds.select(bt.col("missing")).to_pydict()
except bt.BatcherError as exc:
    print(f"query failed: {exc}")
# query failed: projection 'missing' references unknown column(s) ['missing']; available: ['a']
```

Catching `bt.BatcherError` covers every Batcher-specific failure while letting unrelated exceptions propagate, such as a bug in your own batch function.

## See also

- {doc}`Dataset </api/relational/dataset>`: the operations that raise these errors.
- {doc}`Configuration </api/operations/configuration>`: resource limits that govern `ResourceError`.
- {doc}`/cookbook/operations/error_handling`: catching the failure you meant to catch, as a script.
