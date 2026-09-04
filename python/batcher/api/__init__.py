"""The public, fluent, lazy, expression-first API surface.

`api` is the conductor: it builds `LogicalPlan`s and orchestrates the three layers
(Kyber → Carbonite → Core) to execute them. It is the only package allowed to
import all three layers. This module is a re-export façade — the expression
functions come from `api.functions` and the constructors, readers, SQL entry
points, and maintenance operations from `api.session`, each governed by its own
``__all__``.

Every one of those names resolves **lazily** (PEP 562, via `batcher._lazy`). Python
imports a package's ancestors before the package itself, so this façade ran in full
whenever anything under `batcher.api` was imported — which made the root package's
laziness worth nothing on its own, since routing `from_pydict` to
`api.session.frames` still executed `from batcher.api.functions import *` on the way.

`bt.read` is the accessor namespace (`bt.read.csv(...)`), which is also callable as
`bt.read(path)`. It shadows the plain `read` function `session` exports; the two have
the same call signature, so the namespace is strictly the richer of the pair, and the
routing table records that resolution rather than leaving it to import order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._exports import API_EXPORTS as _API_EXPORTS
from batcher._exports import API_SHADOWED as _API_SHADOWED
from batcher._lazy import install as _install

if TYPE_CHECKING:
    # The declaration of this façade's surface, and the generator's input: every name
    # `__getattr__` serves is derived from these imports by `just gen-exports`. Eager
    # only for type checkers and editors — at runtime nothing here is executed.
    # The exceptions a user catches, named one by one rather than star-imported:
    # `errors.__all__` is wider than the public surface (it carries engine-internal
    # ones such as `FatalShuffleError`), and a star here would publish them.
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
    from batcher.api.dataset import Dataset as Dataset
    from batcher.api.dataset import GroupBy as GroupBy
    from batcher.api.functions import *  # noqa: F403
    from batcher.api.security import authenticate as authenticate
    from batcher.api.security import current_verifier as current_verifier
    from batcher.api.security import security as security
    from batcher.api.security import set_verifier as set_verifier
    from batcher.api.session import *  # noqa: F403
    from batcher.api.sql_session import Session as Session
    from batcher.core.runtime import cancel_query as cancel_query
    from batcher.core.runtime import running_queries as running_queries
    from batcher.governance import GovernanceEvent as GovernanceEvent
    from batcher.governance import Principal as Principal
    from batcher.governance import SecurityCatalog as SecurityCatalog
    from batcher.io.formats.streaming import ForeachWriter as ForeachWriter
    from batcher.observe import start_ui as start_ui
    from batcher.observe import stop_ui as stop_ui
    from batcher.observe import ui_url as ui_url
    from batcher.plan.resource import StorageLevel as StorageLevel
    from batcher.plan.streaming import OutputMode as OutputMode
    from batcher.plan.streaming import QueryProgressEvent as QueryProgressEvent
    from batcher.plan.streaming import QueryStartedEvent as QueryStartedEvent
    from batcher.plan.streaming import QueryTerminatedEvent as QueryTerminatedEvent
    from batcher.plan.streaming import SinkProgress as SinkProgress
    from batcher.plan.streaming import SourceProgress as SourceProgress
    from batcher.plan.streaming import StateOperatorProgress as StateOperatorProgress
    from batcher.plan.streaming import StreamingQueryListener as StreamingQueryListener
    from batcher.plan.streaming import StreamingQueryProgress as StreamingQueryProgress
    from batcher.plan.streaming import StreamingQueryStatus as StreamingQueryStatus
    from batcher.plan.streaming import Trigger as Trigger

if TYPE_CHECKING:
    # A second block, and the reason is load-bearing rather than stylistic. `bt.read` is the
    # accessor namespace (`bt.read.csv(...)`), which is also callable as `bt.read(path)`; it
    # shadows the plain `read` function `session` exports, and the two have the same call
    # signature, so the namespace is strictly the richer of the pair. The eager façade won
    # that by assigning `read = _read_namespace` after its star imports. isort sorts within
    # a block and never across blocks, so this is how the same "binds last" is expressed —
    # in the first block, `batcher.api.io_namespace` sorts *above* `batcher.api.session` and
    # the function would win instead. It did, until this block existed: `bt.read` came back
    # as a bare function and `bt.read.parquet` raised `AttributeError`.
    from batcher.api.io_namespace import read as read

#: The `api` surface, in the order the eager façade declared it. `read` is in it —
#: contributed by `session.__all__` — and the routing table is what decides the name
#: resolves to the richer namespace object rather than the plain function.
__all__ = list(_API_EXPORTS)

__getattr__, __dir__ = _install(__name__, _API_EXPORTS, shadowed=_API_SHADOWED)
