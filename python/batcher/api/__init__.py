"""The public, fluent, lazy, expression-first API surface.

`api` is the conductor: it builds `LogicalPlan`s and orchestrates the three layers
(Kyber → Carbonite → Core) to execute them. It is the only package allowed to
import all three layers. This module is a re-export façade — the expression
functions come from `api.functions` and the constructors, readers, SQL entry
points, and maintenance operations from `api.session`, each governed by its own
``__all__``.

`session` is imported *after* `functions` on purpose: `concat` means frame
concatenation at the top level (as it does in pandas and Polars), and the string
builder keeps the explicit name `concat_str`.
"""

from __future__ import annotations

# The typed exception hierarchy, so a user can write ``except bt.BatcherError`` (or a
# specific subclass) without importing an internal module. Every Batcher error
# subclasses `BatcherError`, so catching that one covers them all. The hierarchy itself
# lives in `batcher._internal.errors`; it is re-exported here, at the top level, because
# catching an error is public-API business.
from batcher._internal.errors import AccessDeniedError as AccessDeniedError
from batcher._internal.errors import BackendError as BackendError
from batcher._internal.errors import BatcherError as BatcherError
from batcher._internal.errors import ColumnNotFoundError as ColumnNotFoundError
from batcher._internal.errors import CommitError as CommitError
from batcher._internal.errors import CompileError as CompileError
from batcher._internal.errors import ConfigError as ConfigError
from batcher._internal.errors import DataQualityError as DataQualityError
from batcher._internal.errors import ExecutionError as ExecutionError
from batcher._internal.errors import FormatError as FormatError
from batcher._internal.errors import IOError as IOError
from batcher._internal.errors import MissingDependencyError as MissingDependencyError
from batcher._internal.errors import OptimizationError as OptimizationError
from batcher._internal.errors import PlanError as PlanError
from batcher._internal.errors import ResourceError as ResourceError
from batcher._internal.errors import SchemaError as SchemaError
from batcher._internal.errors import TransportError as TransportError
from batcher.api import functions as _functions
from batcher.api import session as _session
from batcher.api.dataset import Dataset, GroupBy
from batcher.api.functions import *  # noqa: F403  (governed by functions.__all__)
from batcher.api.io_namespace import read as _read_namespace
from batcher.api.security import authenticate, current_verifier, security, set_verifier
from batcher.api.session import *  # noqa: F403  (governed by session.__all__)
from batcher.api.sql_session import Session
from batcher.core.runtime import cancel_query, running_queries
from batcher.governance import GovernanceEvent, Principal, SecurityCatalog
from batcher.io.formats.streaming import ForeachWriter
from batcher.observe import start_ui, stop_ui, ui_url
from batcher.plan.streaming import (
    OutputMode,
    QueryProgressEvent,
    QueryStartedEvent,
    QueryTerminatedEvent,
    SinkProgress,
    SourceProgress,
    StateOperatorProgress,
    StreamingQueryListener,
    StreamingQueryProgress,
    StreamingQueryStatus,
    Trigger,
)

# `bt.read` is the accessor namespace (`bt.read.csv(...)`), which is also callable as
# `bt.read(path)`. It shadows the plain `read` function `session` exports; the two have
# the same call signature, so the namespace is strictly the richer of the pair.
read = _read_namespace

# The exceptions a user catches. `BatcherError` is the root every other one subclasses,
# so `except bt.BatcherError` is the catch-all; the subclasses narrow it.
_ERRORS = [
    "AccessDeniedError",
    "BackendError",
    "BatcherError",
    "ColumnNotFoundError",
    "CommitError",
    "CompileError",
    "ConfigError",
    "DataQualityError",
    "ExecutionError",
    "FormatError",
    "IOError",
    "MissingDependencyError",
    "OptimizationError",
    "PlanError",
    "ResourceError",
    "SchemaError",
    "TransportError",
]

__all__ = [
    "Dataset",
    "GovernanceEvent",
    "GroupBy",
    "ForeachWriter",
    "OutputMode",
    "QueryProgressEvent",
    "QueryStartedEvent",
    "QueryTerminatedEvent",
    "SinkProgress",
    "SourceProgress",
    "StateOperatorProgress",
    "StreamingQueryListener",
    "StreamingQueryProgress",
    "StreamingQueryStatus",
    "Principal",
    "SecurityCatalog",
    "Session",
    "Trigger",
    "authenticate",
    "cancel_query",
    "current_verifier",
    "security",
    "running_queries",
    "set_verifier",
    "start_ui",
    "stop_ui",
    "ui_url",
    *_ERRORS,
    *_functions.__all__,
    *_session.__all__,
]
