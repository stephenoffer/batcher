"""The Batcher exception hierarchy.

A single rooted hierarchy, with each subclass mapped to the layer that raises it.
Native (Rust) errors are translated into the matching subclass at the PyO3
boundary. Feedback/observability writes never raise into the hot path — they log
and drop — so none of these should ever surface from a metrics or learning call.
"""

from __future__ import annotations

__all__ = [
    "BackendError",
    "BackpressureAbort",
    "BatcherError",
    "CommitError",
    "CompileError",
    "ConfigError",
    "ExecutionError",
    "FormatError",
    "IOError",
    "OptimizationError",
    "PerformanceWarning",
    "PlanError",
    "ResourceError",
    "TransportError",
]


class PerformanceWarning(UserWarning):
    """A correctness-neutral usage pattern that will run far slower than intended.

    Raised (via `warnings.warn`) for the documented foot-guns Ray Data users hit —
    e.g. a plain-function UDF on a GPU stage that reloads the model every batch. The
    query still runs and returns the right answer; the warning points at the faster
    spelling."""


class BatcherError(Exception):
    """Base class for every error Batcher raises."""


class PlanError(BatcherError):
    """Invalid plan or schema mismatch (raised by `plan`/`api` at build time)."""


class ConfigError(BatcherError):
    """An invalid configuration value (out of range, inconsistent limits).

    Raised by `Config.validate()` at the config entry points so a bad tunable fails
    early and clearly instead of being silently accepted and surfacing as a confusing
    runtime failure."""


class OptimizationError(BatcherError):
    """The optimizer (Kyber) failed to produce a valid physical plan."""


class ResourceError(BatcherError):
    """The resource manager (Carbonite) could not satisfy a request."""


class BackpressureAbort(ResourceError):
    """Execution was aborted because backpressure could not be relieved."""


class ExecutionError(BatcherError):
    """An operator failed at runtime (raised by the engine / Core)."""


class BackendError(ExecutionError):
    """A specific execution backend failed; wraps the underlying error."""


class CompileError(ExecutionError):
    """JIT compilation of a pipeline failed (the interpreter remains a fallback)."""


class TransportError(BatcherError):
    """The distributed data plane (shared memory / Arrow Flight) failed."""


class IOError(BatcherError):
    """A source or sink failed to read, write, list, or open a path.

    Raised by the `io` layer, including for a missing optional filesystem/format
    dependency (e.g. reading ``s3://`` without the ``cloud`` extra installed).
    """


class FormatError(IOError):
    """An unknown/unsupported IO format, or a file malformed for its format."""


class CommitError(IOError):
    """An atomic write commit failed — e.g. a concurrent-writer conflict on a
    transactional (Delta/Iceberg) table, or a partial multi-file publish."""


class SchemaError(IOError):
    """Schemas could not be reconciled — e.g. reading multiple files whose column
    types are incompatible under the requested ``schema_mode``, or an input whose
    schema diverges from an expected one."""


class DataQualityError(BatcherError):
    """A data-quality expectation failed — raised by ``ds.dq...fail()`` when one or
    more constraints have violating rows. Carries the per-constraint violation counts."""

    def __init__(self, message: str, violations: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.violations = violations or {}
