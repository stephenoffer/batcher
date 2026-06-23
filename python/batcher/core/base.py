"""The execution-strategy seam: one `Executor` Protocol, one `ExecutionContext`.

Core runs queries through several execution tiers — single-node native, the
linear `map_batches`/UDF orchestrator, and (assembled by the conductor) the
distributed path — and more are planned (morsel scheduler, JIT/LLVM, GPU). Rather
than grow an if/elif/else at the dispatch site for each, every tier is an
`Executor`: a strategy with one `execute` method, selected by a registry.

This module is the neutral seam only. It defines the Protocol and the read-only
`ExecutionContext` the strategies receive. The concrete strategies and the
registry that selects between them live in `api` (the conductor), because the
local-native strategy orchestrates Kyber + Carbonite and the distributed strategy
lives in `dist` — neither of which Core may import. Core owns the contract; the
conductor wires the implementations behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = ["ExecutionContext", "Executor"]


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Read-only inputs an `Executor` needs, beyond the plan and its sources.

    Carries exactly what the existing execution functions take as side inputs:
    the output column names (to build an empty-result schema), the process
    MetadataHub (the feedback sink the local path records into and Kyber learns
    from), and the distributed knobs (`num_workers`, `transport`). A strategy uses
    only the fields it needs; the rest are inert.
    """

    columns: list[str]
    hub: MetadataHub
    num_workers: int | None = None
    transport: str = "auto"


class Executor(Protocol):
    """A single execution strategy: run a plan over its sources to a table.

    Implementations are thin wrappers over the existing execution functions
    (`execute_local`, `execute_with_udfs`, `execute_distributed`). The selection
    of *which* strategy runs a given plan is the registry's job, not the
    strategy's — an `Executor.execute` assumes it was chosen correctly.
    """

    def execute(
        self,
        plan: LogicalPlan,
        sources: list[Source],
        ctx: ExecutionContext,
    ) -> pa.Table: ...
