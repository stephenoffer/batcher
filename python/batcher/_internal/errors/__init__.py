"""`errors` — the typed exception hierarchy, and the machinery that makes it readable.

Three modules, one import path. `hierarchy` holds the exception classes themselves;
`suggest` holds the "did you mean ...?" engine and the canonical unknown-name message
shape that every layer of the engine formats its lookup failures through; `validate`
turns a wrong-typed user argument into one of those exceptions at the API edge.

They are split because they answer different questions — *what* failed, *how to say so*,
and *when to say it* — and joined here because ``from batcher._internal.errors import
PlanError`` is the import the whole tree already writes, and a package split must never
move a name.
"""

from __future__ import annotations

from batcher._internal.errors.hierarchy import (
    AccessDeniedError,
    BackendError,
    BackpressureAbort,
    BatcherError,
    ColumnNotFoundError,
    CommitError,
    CompileError,
    ConfigError,
    DataQualityError,
    DataWarning,
    ExecutionError,
    FatalShuffleError,
    FormatError,
    IOError,
    MissingDependencyError,
    OptimizationError,
    PerformanceWarning,
    PlanError,
    PlanTooDeepError,
    QueryCancelledError,
    ResourceError,
    RetryableShuffleError,
    SchemaError,
    SecurityWarning,
    TransportError,
    unknown_value,
)
from batcher._internal.errors.suggest import (
    absent_error,
    candidate_list,
    did_you_mean,
    suggestion,
    unknown_message,
)
from batcher._internal.errors.validate import require_float, require_int

__all__ = [
    "AccessDeniedError",
    "BackendError",
    "BackpressureAbort",
    "BatcherError",
    "ColumnNotFoundError",
    "CommitError",
    "CompileError",
    "ConfigError",
    "DataQualityError",
    "DataWarning",
    "ExecutionError",
    "FatalShuffleError",
    "FormatError",
    "IOError",
    "MissingDependencyError",
    "OptimizationError",
    "PerformanceWarning",
    "PlanError",
    "PlanTooDeepError",
    "QueryCancelledError",
    "ResourceError",
    "RetryableShuffleError",
    "SchemaError",
    "SecurityWarning",
    "TransportError",
    "absent_error",
    "candidate_list",
    "did_you_mean",
    "require_float",
    "require_int",
    "suggestion",
    "unknown_message",
    "unknown_value",
]
